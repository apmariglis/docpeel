"""
Mistral provider implementation.

Two-step pipeline: mistral-ocr-latest extracts raw markdown from the page image,
then a cheap Mistral chat model structures that markdown into the shared schema.

This class is intentionally not a VisionProvider subclass — the two-step pipeline
is architecturally different from the single vision call used by Anthropic and
Gemini. MistralExtractor in extraction.py owns the page-level orchestration.

There is no content filter in Mistral OCR, so the quadrant-split fallback chain
used by VisionExtractor is never needed here.
"""

import base64
import io
import json
import re

from docpeel.pricing import mistral_cost
from docpeel.providers.base import Usage
from docpeel.providers.base import _with_retry
from PIL import Image

# ── Structuring prompt ────────────────────────────────────────────────────────

# Sent to the cheap Mistral chat model to convert raw OCR markdown into the
# shared schema dict. Kept here (not in prompts.py) because it targets a chat
# model, not a vision model, and is entirely Mistral-specific.
_MISTRAL_STRUCTURE_PROMPT = """\
You are given the raw OCR output for one page of a PDF document.
Your task is to parse this into a structured JSON object matching the schema below.
Respond with ONLY the JSON object — no preamble, no explanation, no markdown fences.

The OCR output already contains markdown formatting (**bold**, *italic*, ***bold-italic***,
__underline__). Preserve all such formatting exactly as it appears in the OCR output when
populating the text field. Do not strip, alter, or re-interpret any markdown formatting.

Schema fields:
- skip (boolean): true if this page should be excluded from the output because
  it contains no extractable text content. Set skip to true for these page types:
    - table_of_contents: a page listing chapter/section titles with page numbers.
    - index: an alphabetical index of terms with page numbers at the back of the book.
    - blank: a completely blank page or one that only says 'This page intentionally left blank'.
    - illustration_only: a full-page illustration, artwork, or map with no meaningful text.
      A page qualifies as illustration_only even if it contains an artist signature, a
      small caption, or an ownership/order watermark — these are not meaningful text content.
      Do NOT classify it as title_page just because a watermark or signature is present.
    - title_page: a page containing only the book title or series title and nothing else —
      no body text, no chapter content, just the title (and possibly author/edition info).
    - part_divider: a decorative page that only announces a part, book, or section number
      (e.g. 'Part II', 'Book Two') with no other content.
  Set skip to false for all other pages.
- skip_reason (string|null): one of the labels above if skip is true, else null.
- page_number (integer|null): printed page number as it appears on the page — check
  all locations: page header, page footer, and either margin. Use the arabic numeral
  value (e.g. 20, not "xx"). null only if no page number is present anywhere on the page.
- title (string|null): single prominent display heading for this page/section,
  plain text only, null if none exists.
- text (string): verbatim transcription of all page text excluding the title and
  excluding all tables. A table is any block of pipe-delimited markdown rows (| ... |),
  together with its heading line if one immediately precedes it (e.g. a line like
  "Table 3: Annual Rainfall by Region" or "### Table 4: Pricing Tiers").
  Both the heading line AND the table body must be removed from this field entirely —
  the heading goes into the table's title field, the body into its content field.
  Do not leave any table heading lines in the text, even if they look like section headings.
  Do NOT insert any placeholder, marker, or reference (such as {{table_0}}) where a
  table was removed — simply omit the table and its heading, leaving the surrounding
  prose paragraphs adjacent to each other.
  Preserve all other markdown formatting from the OCR output exactly.
  Ignore and do not include any watermarks or ownership stamps — these are short
  isolated lines of text that appear to be overlaid on the page and are clearly
  not part of the original typeset document content (they interrupt the normal
  text flow, appear in unexpected positions, and do not belong to any paragraph,
  heading, caption, or table).
  Empty string if skip is true.
- watermarks (array of strings): the exact text of each watermark or ownership
  stamp that was detected and excluded from the text field, one entry per watermark.
  Empty array if none were detected.
- tables (array): one object per table, in the order they appear on the page, with:
    - title (string|null): the heading label for this table exactly as it appears in
      the document, plain text without markdown formatting (e.g. "Table 3: Annual Rainfall
      by Region" or "Table 4: Pricing Tiers"). null if the table has no heading.
      Every table heading removed from the text field must appear here.
    - caption (string): 1-2 sentence description of what the table contains
      and what questions it answers.
    - content (string): full markdown table body preserving all headers and values.
  Empty array if no tables or skip is true.

Raw OCR output to parse:
"""


# ── Provider ──────────────────────────────────────────────────────────────────


class MistralProvider:
    OCR_MODEL = "mistral-ocr-latest"

    def __init__(self, model: str | None = None, structure_fn=None):
        """
        model: the Mistral chat model for the structuring step. Required unless
               structure_fn is provided.
        structure_fn: optional callable that replaces the Mistral chat model entirely.
        """
        import os

        from mistralai import Mistral as _Mistral

        if model is None and structure_fn is None:
            raise ValueError(
                "Either --structure-model or a structure_fn must be provided for the Mistral OCR path."
            )

        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "MISTRAL_API_KEY is not set. Add it to your .env file."
            )
        self._client = _Mistral(api_key=api_key)
        self._chat_model = model
        self._structure_fn = structure_fn
        self._sanitisation_warnings: list[str] = []

    def close(self) -> None:
        """Close the underlying HTTP client. Call before process exit to avoid SDK cleanup errors."""
        self._client.close()

    @property
    def model_id(self) -> str:
        # If an external structure_fn was injected, we can't know its model ID here.
        # provider_factory sets a _structure_model_id attribute on the fn if it can.
        if self._structure_fn is not None:
            structure_label = getattr(self._structure_fn, "_model_id", "external")
            return f"{self.OCR_MODEL}+{structure_label}"
        return f"{self.OCR_MODEL}+{self._chat_model}"

    def _image_to_b64_url(self, image: Image.Image) -> str:
        """Encode a PIL image as a base64 data URL for the Mistral OCR API."""
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def ocr(self, image: Image.Image) -> tuple[str, int]:
        """
        Run Mistral OCR on a single page image.
        Returns (markdown_text, page_count). Page count is always 1 for image input.
        Call ocr_with_retry() for the retry-wrapped version.
        """
        resp = self._client.ocr.process(
            model=self.OCR_MODEL,
            document={
                "type": "image_url",
                "image_url": self._image_to_b64_url(image),
            },
        )
        text = "\n\n".join(page.markdown for page in resp.pages if page.markdown)
        return text, len(resp.pages)

    def structure(self, ocr_text: str, extra_context: str = "") -> tuple[dict, Usage]:
        """
        Send OCR markdown to the structuring model to produce the structured
        schema dict. Returns (result_dict, usage).

        Delegates to self._structure_fn if one was injected (e.g. Gemini),
        otherwise calls the Mistral chat model directly.

        extra_context: optional text appended to the prompt — used by
        MistralExtractor when stitching quadrant texts alongside the OCR output.
        Call structure_with_retry() for the retry-wrapped version.
        """
        if self._structure_fn is not None:
            return self._structure_fn(ocr_text, extra_context)
        prompt = _MISTRAL_STRUCTURE_PROMPT + ocr_text
        if extra_context:
            prompt += f"\n\nAdditional context (quadrant texts for stitching):\n{extra_context}"

        resp = self._client.chat.complete(
            model=self._chat_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""

        # Strip markdown fences if the model added them despite instructions
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0]

        # Detect and strip raw control characters (U+0000–U+001F).
        # Tab \x09, newline \x0a, and CR \x0d are valid JSON structural whitespace,
        # but json.loads (strict mode) rejects them as unescaped bytes *inside string
        # values* — which is exactly where they appear in multi-line OCR output.
        # Strip all of them; the content loss is negligible vs. a hard parse failure.
        # We only warn for the uncommon ones (\x00–\x08, \x0b, \x0c, \x0e–\x1f) since
        # newlines/tabs in OCR text are expected and not worth surfacing.
        control_chars = re.findall(r"[\x00-\x1f]", raw)
        if control_chars:
            unexpected = [c for c in control_chars if c not in "\x09\x0a\x0d"]
            if unexpected:
                counts = ", ".join(
                    f"U+{ord(c):04X} ×{control_chars.count(c)}"
                    for c in sorted(set(unexpected))
                )
                self._sanitisation_warnings.append(
                    f"chat model response contained unexpected control character(s) "
                    f"({counts}) that were stripped before JSON parsing — "
                    f"these likely originate from an unusual glyph or symbol on "
                    f"the page that Mistral OCR could not map cleanly."
                )
            raw = re.sub(r"[\x00-\x1f]", "", raw)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Last-resort: strict=False tolerates control characters that
            # remain inside string values after the strip above.
            self._sanitisation_warnings.append(
                "json.loads failed on sanitised output; retrying with "
                "strict=False -- some control characters may remain in text fields."
            )
            parsed = json.loads(raw, strict=False)

        # Normalise types
        result = {
            "skip": bool(parsed.get("skip", False)),
            "skip_reason": parsed.get("skip_reason"),
            "page_number": parsed.get("page_number"),
            "title": parsed.get("title"),
            "text": parsed.get("text", ""),
            "tables": parsed.get("tables") or [],
            "watermarks": parsed.get("watermarks") or [],
        }
        if not isinstance(result["skip_reason"], str):
            result["skip_reason"] = None
        if result["skip"]:
            result["skip_reason"] = result["skip_reason"] or "unknown"
        if not isinstance(result["page_number"], int):
            result["page_number"] = None
        if not isinstance(result["title"], str):
            result["title"] = None
        normalised_tables = []
        for t in result["tables"]:
            if isinstance(t, dict):
                raw_title = t.get("title")
                normalised_tables.append(
                    {
                        "title": raw_title if isinstance(raw_title, str) else None,
                        "caption": t.get("caption") or "",
                        "content": t.get("content") or "",
                    }
                )
        result["tables"] = normalised_tables

        u = resp.usage
        usage = Usage(
            input_tokens=u.prompt_tokens or 0,
            output_tokens=u.completion_tokens or 0,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            cost_usd=mistral_cost(
                ocr_model=self.OCR_MODEL,
                chat_model=self._chat_model,
                ocr_pages=0,  # OCR cost tracked separately by MistralExtractor
                chat_input_tokens=u.prompt_tokens or 0,
                chat_output_tokens=u.completion_tokens or 0,
            ),
        )
        return result, usage

    # ── Retry-wrapped entry points (called by MistralExtractor) ───────────────

    def ocr_with_retry(self, image: Image.Image) -> tuple[str, int]:
        """Run ocr() with exponential backoff. All Mistral errors are transient."""
        return _with_retry(
            lambda exc: False,
            lambda: self.ocr(image),
            step=f"OCR ({self.OCR_MODEL})",
        )

    # def structure_with_retry(
    #     self, ocr_text: str, extra_context: str = ""
    # ) -> tuple[dict, Usage]:
    #     """Run structure() with exponential backoff. All Mistral errors are transient."""
    #     return _with_retry(
    #         lambda exc: False, lambda: self.structure(ocr_text, extra_context)
    #     )
    def structure_with_retry(
        self, ocr_text: str, extra_context: str = ""
    ) -> tuple[dict, Usage]:
        """Run structure() with exponential backoff.
        JSON parse failures are not retried — the same input will produce the same
        broken output on every attempt. Only true API/network errors are retried.
        """
        step_model = self._chat_model or getattr(
            self._structure_fn, "_model_id", "unknown"
        )
        return _with_retry(
            lambda exc: isinstance(exc, json.JSONDecodeError),
            lambda: self.structure(ocr_text, extra_context),
            step=f"structuring ({step_model})",
        )

    def ocr_page_cost(self, n_pages: int) -> float:
        """Return the OCR cost for n_pages, for use by MistralExtractor."""
        return mistral_cost(
            ocr_model=self.OCR_MODEL,
            chat_model=self._chat_model,
            ocr_pages=n_pages,
            chat_input_tokens=0,
            chat_output_tokens=0,
        )

    def drain_sanitisation_warnings(self) -> list[str]:
        """Return and clear warnings accumulated during the last structure() call."""
        warnings, self._sanitisation_warnings = self._sanitisation_warnings, []
        return warnings

    def resolve_model_id(self) -> None:
        if self._chat_model is not None:
            result = self._client.models.retrieve(model_id=self._chat_model)
            self._chat_model = result.id
        elif self._structure_fn is not None:
            # Delegate to the embedded provider (e.g. GeminiProvider)
            embedded = getattr(self._structure_fn, "_provider", None)
            if embedded is not None:
                embedded.resolve_model_id()
                self._structure_fn._model_id = embedded.model_id
