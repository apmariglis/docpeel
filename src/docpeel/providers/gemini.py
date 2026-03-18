"""
Gemini provider implementation.

Uses response_schema + response_mime_type=application/json to enforce structured
output. Content-filter blocks surface as a non-STOP finish_reason on the response
object; _GeminiContentFilterError is raised internally so the retry/fallback
machinery treats them identically to Anthropic's BadRequestError.
"""

import json

from docpeel.pricing import gemini_cost
from docpeel.providers.base import PAGE_EXTRACTION_SCHEMA
from docpeel.providers.base import Usage
from docpeel.providers.base import VisionProvider
from docpeel.providers.base import _with_retry
from PIL import Image

# ── Internal filter-error type ────────────────────────────────────────────────


class _GeminiContentFilterError(Exception):
    """Raised internally when Gemini returns a blocked finish_reason."""


# finish_reason values that mean "blocked by policy", not a transient error.
# In the new SDK these are string enums.
_GEMINI_BLOCKED_REASONS = {
    "SAFETY",
    "RECITATION",
    "OTHER",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
    "SPII",
}
_GEMINI_STOP_REASON = "STOP"


def _gemini_finish_reason(response) -> str | None:
    """Extract the finish_reason string from a google.genai response, or None."""
    try:
        candidates = response.candidates
        if candidates:
            reason = candidates[0].finish_reason
            # The SDK exposes this as an enum; .name gives the string form
            return reason.name if hasattr(reason, "name") else str(reason)
    except Exception:
        pass
    return None


# ── Provider ──────────────────────────────────────────────────────────────────


class GeminiProvider(VisionProvider):
    DEFAULT_MODEL = "gemini-2.5-flash-lite"

    def __init__(self, model: str | None = None):
        import os

        from google import genai as _genai
        from google.genai import errors as _gerrors

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. Add it to your .env file."
            )
        self._client = _genai.Client(api_key=api_key)
        self._gerrors = _gerrors
        self._model = model or self.DEFAULT_MODEL

    @property
    def model_id(self) -> str:
        return self._model

    def _parse_usage(self, response) -> Usage:
        meta = response.usage_metadata
        inp = getattr(meta, "prompt_token_count", 0) or 0
        out = getattr(meta, "candidates_token_count", 0) or 0
        cost = gemini_cost(model=self._model, input_tokens=inp, output_tokens=out)
        return Usage(
            input_tokens=inp,
            output_tokens=out,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            cost_usd=cost,
        )

    def _extract_text(self, resp) -> str:
        """
        Return response text, classifying the finish_reason:

        - Blocked reason   → _GeminiContentFilterError  (no retry, fallback chain handles it)
        - STOP but no text → empty string               (image-only quadrant)
        - Anything else    → let SDK exception propagate (transient, _with_retry will retry)
        """
        reason = _gemini_finish_reason(resp)

        if reason in _GEMINI_BLOCKED_REASONS:
            raise _GeminiContentFilterError(
                f"Gemini blocked response (finish_reason={reason})"
            )

        if reason == _GEMINI_STOP_REASON:
            try:
                return resp.text
            except Exception:
                # STOP with no text part is a legitimate empty response —
                # the quadrant contained only artwork or whitespace.
                return ""

        return resp.text

    def _generate(self, contents) -> tuple[str, Usage]:
        resp = self._client.models.generate_content(
            model=self._model,
            contents=contents,
        )
        return self._extract_text(resp), self._parse_usage(resp)

    def _generate_structured(self, contents) -> tuple[dict, Usage]:
        """Use Gemini's response_schema to enforce structured output."""
        from google.genai import types as _gtypes

        schema = PAGE_EXTRACTION_SCHEMA
        props = schema["properties"]

        table_props = props["tables"]["items"]["properties"]

        gemini_schema = _gtypes.Schema(
            type=_gtypes.Type.OBJECT,
            properties={
                "skip": _gtypes.Schema(
                    type=_gtypes.Type.BOOLEAN,
                    description=props["skip"]["description"],
                ),
                "skip_reason": _gtypes.Schema(
                    type=_gtypes.Type.STRING,
                    nullable=True,
                    description=props["skip_reason"]["description"],
                ),
                "page_number": _gtypes.Schema(
                    type=_gtypes.Type.INTEGER,
                    nullable=True,
                    description=props["page_number"]["description"],
                ),
                "title": _gtypes.Schema(
                    type=_gtypes.Type.STRING,
                    nullable=True,
                    description=props["title"]["description"],
                ),
                "text": _gtypes.Schema(
                    type=_gtypes.Type.STRING,
                    description=props["text"]["description"],
                ),
                "tables": _gtypes.Schema(
                    type=_gtypes.Type.ARRAY,
                    description=props["tables"]["description"],
                    items=_gtypes.Schema(
                        type=_gtypes.Type.OBJECT,
                        properties={
                            "title": _gtypes.Schema(
                                type=_gtypes.Type.STRING,
                                nullable=True,
                                description=table_props["title"]["description"],
                            ),
                            "caption": _gtypes.Schema(
                                type=_gtypes.Type.STRING,
                                description=table_props["caption"]["description"],
                            ),
                            "content": _gtypes.Schema(
                                type=_gtypes.Type.STRING,
                                description=table_props["content"]["description"],
                            ),
                        },
                        required=["caption", "content"],
                    ),
                ),
                "watermarks": _gtypes.Schema(
                    type=_gtypes.Type.ARRAY,
                    description=props["watermarks"]["description"],
                    items=_gtypes.Schema(type=_gtypes.Type.STRING),
                ),
            },
            required=schema["required"],
        )

        resp = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=_gtypes.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=gemini_schema,
            ),
        )
        reason = _gemini_finish_reason(resp)
        if reason in _GEMINI_BLOCKED_REASONS:
            raise _GeminiContentFilterError(
                f"Gemini blocked response (finish_reason={reason})"
            )

        parsed = json.loads(resp.text)
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
        return result, self._parse_usage(resp)

    def call(self, image: Image.Image, prompt: str) -> tuple[str, Usage]:
        return _with_retry(
            self.is_content_filter_error, lambda: self._generate([image, prompt])
        )

    def call_structured(self, image: Image.Image, prompt: str) -> tuple[dict, Usage]:
        return _with_retry(
            self.is_content_filter_error,
            lambda: self._generate_structured([image, prompt]),
        )

    def call_with_image_and_text_structured(
        self, image: Image.Image, text_prompt: str
    ) -> tuple[dict, Usage]:
        return _with_retry(
            self.is_content_filter_error,
            lambda: self._generate_structured([image, text_prompt]),
        )

    def is_content_filter_error(self, exc: Exception) -> bool:
        if isinstance(exc, _GeminiContentFilterError):
            return True
        # ClientError 400 can also be a safety block in some cases
        if (
            isinstance(exc, self._gerrors.ClientError)
            and getattr(exc, "code", None) == 400
        ):
            return True
        return False

    def resolve_model_id(self) -> None:
        model = self._client.models.get(model=self._model)
        # model.name is a resource path: "models/gemini-2.5-flash-lite"
        name = model.name or self._model
        self._model = name.removeprefix("models/")

    def call_structured_text(self, prompt: str) -> tuple[dict, Usage]:
        """Structured call with text-only input (no image). Used by the OCR structuring path."""
        return _with_retry(
            self.is_content_filter_error,
            lambda: self._generate_structured([prompt]),
        )
