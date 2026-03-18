"""
Page extraction orchestration: two extractor classes and the iter_pages entry point.

VisionExtractor
    Used for AnthropicProvider and GeminiProvider. Performs OCR and structuring
    in a single vision call, with a multi-stage fallback chain per page:
      1. Full-page verbatim (call_structured)
      2. If content-filter blocked → quadrant split, per quadrant:
           2a. Verbatim quadrant call
           2b. If blocked → obfuscated-image retry
           2c. If still blocked → paraphrase (flagged for manual review)
      3. Stitch: reassemble quadrant texts into structured output using the
         original page image for cross-check (call_with_image_and_text_structured)
      4. If stitch blocked → full-page paraphrase fallback

MistralExtractor
    Used for MistralProvider. Runs the two-step pipeline:
      1. OCR: mistral-ocr-latest produces raw markdown from the page image
      2. Structure: a cheap Mistral chat model converts the markdown into the
         shared schema dict
    No content filter exists in Mistral OCR, so no fallback chain is needed.

Memory model
------------
convert_from_path() is called one page at a time (first_page/last_page) so
only a single PIL image lives in RAM at once. The page image is explicitly
deleted after extraction. Results are yielded one at a time so the caller
can stream them to disk rather than accumulating a full-book list.
"""

import time
from collections.abc import Iterator
from pathlib import Path

from pdf2image import convert_from_path
from pdf2image import pdfinfo_from_path
from PIL import Image

from docpeel.image_utils import obfuscate
from docpeel.image_utils import split_quadrants
from docpeel.prompts import PAGE_EXTRACT_PROMPT
from docpeel.prompts import PARAPHRASE_PROMPT
from docpeel.prompts import STITCH_PROMPT
from docpeel.prompts import quadrant_extract_prompt
from docpeel.providers.base import Usage
from docpeel.providers.base import VisionProvider
from docpeel.providers.mistral import MistralProvider

# ── Shared result helpers ─────────────────────────────────────────────────────

# Type alias for the tuple both extractors return from their extract() method.
# (skip, skip_reason, text, title, book_page, tables, watermarks,
#  usage, method, warnings, paraphrased[, ocr_cost_usd, structure_cost_usd])
# The last two fields are only present for MistralExtractor results.
_PageResult = tuple[
    bool,
    str | None,
    str,
    str | None,
    int | None,
    list[dict],
    list[str],
    Usage,
    str,
    list[str],
    str | None,
]


def _unpack(
    result: dict,
) -> tuple[bool, str | None, str, str | None, int | None, list[dict], list[str]]:
    """
    Unpack a structured result dict returned by provider.call_structured() into
    (skip, skip_reason, text, title, page_number, tables, watermarks).
    """
    skip = bool(result.get("skip", False))
    skip_reason = result.get("skip_reason") if skip else None
    if skip and not isinstance(skip_reason, str):
        skip_reason = "unknown"
    text = result.get("text") or ""
    title = result.get("title")
    if not isinstance(title, str):
        title = None
    page_number = result.get("page_number")
    if not isinstance(page_number, int):
        page_number = None
    raw_tables = result.get("tables") or []
    if not isinstance(raw_tables, list):
        raw_tables = []
    tables = []
    for t in raw_tables:
        if not isinstance(t, dict):
            continue
        raw_title = t.get("title")
        tables.append(
            {
                "title": raw_title if isinstance(raw_title, str) else None,
                "caption": t.get("caption") or "",
                "content": t.get("content") or "",
            }
        )
    watermarks = result.get("watermarks") or []
    if not isinstance(watermarks, list):
        watermarks = []
    return skip, skip_reason, text, title, page_number, tables, watermarks


# ── VisionExtractor ───────────────────────────────────────────────────────────


class VisionExtractor:
    """
    Extracts a single page using a VisionProvider (Anthropic or Gemini).

    Implements the full fallback chain: full-page → quadrant split →
    stitch, with obfuscation and paraphrase fallbacks at each stage.
    """

    def __init__(self, provider: VisionProvider) -> None:
        self._provider = provider

    def _paraphrase(
        self,
        img: Image.Image,
        is_full_page: bool = False,
    ) -> tuple[
        bool,
        str | None,
        str,
        str | None,
        int | None,
        list[dict],
        list[str],
        Usage,
        list[str],
    ]:
        """
        Last-resort extraction: ask the model to paraphrase rather than
        transcribe verbatim. Used when even an obfuscated image is blocked
        by the content/recitation filter.

        is_full_page: if True, use call_structured (whole page image visible,
                      table extraction possible).
                      if False, use plain call (quadrant slice — raw text only).
        Returns (..., sanitisation_warnings).
        """
        provider = self._provider
        if is_full_page:
            result, usage = provider.call_structured(img, PARAPHRASE_PROMPT)
            skip, skip_reason, text, title, page_number, tables, watermarks = _unpack(
                result
            )
            return (
                skip,
                skip_reason,
                text,
                title,
                page_number,
                tables,
                watermarks,
                usage,
                [],
            )
        raw, usage = provider.call(img, PARAPHRASE_PROMPT)
        return False, None, raw, None, None, [], [], usage, []

    def extract(self, page_image: Image.Image, page_num: int) -> _PageResult:
        """
        Extract text from a single page image with the full fallback chain.
        """
        provider = self._provider

        # ── Stage 1: full-page verbatim ───────────────────────────────────────
        try:
            result, usage = provider.call_structured(page_image, PAGE_EXTRACT_PROMPT)
            skip, skip_reason, text, title, book_page, tables, watermarks = _unpack(
                result
            )
            return (
                skip,
                skip_reason,
                text,
                title,
                book_page,
                tables,
                watermarks,
                usage,
                "full-page",
                [],
                None,
            )
        except Exception as exc:
            if not provider.is_content_filter_error(exc):
                raise
            print("    Content filter triggered — splitting into quadrants …")

        # ── Stage 2: quadrant split ───────────────────────────────────────────
        quadrants = split_quadrants(page_image)
        chunk_texts: dict[str, str] = {}
        total_usage = Usage.zero()
        warnings: list[str] = []
        paraphrased_quads: list[str] = []

        for label, quad_img in quadrants.items():
            # Stage 2a: verbatim quadrant
            try:
                text, usage = provider.call(quad_img, quadrant_extract_prompt(label))
                chunk_texts[label] = text
                total_usage = total_usage + usage
                print(f"      Chunk {label}: OK")
                continue
            except Exception as exc:
                if not provider.is_content_filter_error(exc):
                    chunk_texts[label] = ""
                    warnings.append(f"chunk '{label}' failed: {exc}")
                    print(f"      Chunk {label}: error — skipped")
                    continue

            # Stage 2b: obfuscated retry
            print(f"      Chunk {label}: blocked — retrying with obfuscated image ...")
            try:
                text, usage = provider.call(
                    obfuscate(quad_img), quadrant_extract_prompt(label)
                )
                chunk_texts[label] = text
                total_usage = total_usage + usage
                print(f"      Chunk {label}: OK (obfuscated retry)")
                continue
            except Exception as exc2:
                if not provider.is_content_filter_error(exc2):
                    chunk_texts[label] = ""
                    warnings.append(
                        f"chunk '{label}' failed on obfuscated retry: {exc2}"
                    )
                    print(f"      Chunk {label}: error on obfuscated retry — skipped")
                    continue

            # Stage 2c: paraphrase
            print(f"      Chunk {label}: blocked after obfuscation — paraphrasing ...")
            try:
                _, _, text, _, _, _, _, usage, san_warnings = self._paraphrase(
                    quad_img, is_full_page=False
                )
                chunk_texts[label] = text
                total_usage = total_usage + usage
                warnings.extend(san_warnings)
                paraphrased_quads.append(label)
                warnings.append(
                    f"chunk '{label}' could not be transcribed verbatim — "
                    "paraphrased instead (manual review recommended)"
                )
                print(
                    f"      Chunk {label}: OK (paraphrased — manual review recommended)"
                )
            except Exception as exc3:
                chunk_texts[label] = ""
                if provider.is_content_filter_error(exc3):
                    warnings.append(
                        f"chunk '{label}' blocked even on paraphrase attempt — skipped"
                    )
                    print(f"      Chunk {label}: blocked on paraphrase — skipped")
                else:
                    warnings.append(
                        f"chunk '{label}' failed on paraphrase attempt: {exc3}"
                    )
                    print(f"      Chunk {label}: error on paraphrase — skipped")

        # ── Image-only page detection ─────────────────────────────────────────
        # If every quadrant is empty the page is pure artwork with no text.
        # Skip the stitch entirely — return empty text cleanly.
        if not any(t.strip() for t in chunk_texts.values()):
            print("      All quadrants empty — image-only page, skipping stitch.")
            return (
                False,
                None,
                "",
                None,
                None,
                [],
                [],
                total_usage,
                "quadrant-split",
                warnings,
                None,
            )

        # ── Stage 3: stitch ───────────────────────────────────────────────────
        chunk_block = "\n\n".join(
            (
                f"=== {lbl} ===\n{chunk_texts[lbl]}"
                if chunk_texts.get(lbl)
                else f"=== {lbl} ===\n(empty)"
            )
            for lbl in ["top-left", "top-right", "bottom-left", "bottom-right"]
        )

        book_page = None
        try:
            print("      Stitching with original page image for cross-check …")
            result, stitch_usage = provider.call_with_image_and_text_structured(
                page_image, STITCH_PROMPT + chunk_block
            )
            skip, skip_reason, stitched, title, book_page, tables, watermarks = _unpack(
                result
            )
        except Exception as exc:
            if not provider.is_content_filter_error(exc):
                raise
            print("      Stitch blocked by content filter — paraphrasing full page …")
            (
                skip,
                skip_reason,
                stitched,
                title,
                book_page,
                tables,
                watermarks,
                stitch_usage,
                san_warnings,
            ) = self._paraphrase(page_image, is_full_page=True)
            warnings.extend(san_warnings)
            paraphrased_quads.append("stitch")
            warnings.append(
                "stitch call blocked by content filter — full page paraphrased instead "
                "(manual review recommended)"
            )

        total_usage = total_usage + stitch_usage
        if "stitch" in paraphrased_quads:
            paraphrased = "full"
        elif paraphrased_quads:
            paraphrased = "partial"
        else:
            paraphrased = None
        return (
            skip,
            skip_reason,
            stitched,
            title,
            book_page,
            tables,
            watermarks,
            total_usage,
            "quadrant-split",
            warnings,
            paraphrased,
        )


# ── MistralExtractor ──────────────────────────────────────────────────────────


class MistralExtractor:
    """
    Extracts a single page using the Mistral two-step pipeline:
      1. mistral-ocr-latest produces raw markdown from the page image
      2. A cheap Mistral chat model structures the markdown into the shared schema

    No fallback chain — Mistral OCR has no content filter.
    """

    def __init__(self, provider: MistralProvider) -> None:
        self._provider = provider

    def extract(self, page_image: Image.Image, page_num: int) -> _PageResult:
        """
        Run the OCR → structure pipeline for a single page image.
        """
        provider = self._provider

        ocr_text, n_pages = provider.ocr_with_retry(page_image)
        ocr_cost = provider.ocr_page_cost(n_pages)

        result, chat_usage = provider.structure_with_retry(ocr_text)
        san_warnings = provider.drain_sanitisation_warnings()

        total_usage = Usage(
            input_tokens=chat_usage.input_tokens,
            output_tokens=chat_usage.output_tokens,
            cache_creation_tokens=0,
            cache_read_tokens=0,
            cost_usd=ocr_cost + chat_usage.cost_usd,
        )

        skip, skip_reason, text, title, book_page, tables, watermarks = _unpack(result)
        return (
            skip,
            skip_reason,
            text,
            title,
            book_page,
            tables,
            watermarks,
            total_usage,
            "ocr+structure",
            san_warnings,
            None,  # paraphrased is never applicable for Mistral
            ocr_cost,  # extra: OCR step cost only
            chat_usage.cost_usd,  # extra: structure step cost only
        )


# ── Page count helper ─────────────────────────────────────────────────────────


def page_count(pdf_path: Path) -> int:
    """Return the total number of pages in the PDF without loading any images."""
    info = pdfinfo_from_path(str(pdf_path))
    return info["Pages"]


# ── iter_pages ────────────────────────────────────────────────────────────────


def iter_pages(
    pdf_path: Path,
    provider: VisionProvider | MistralProvider,
    dpi: int = 150,
) -> Iterator[dict]:
    """
    Yield one result dict per page, converting and releasing each page image
    immediately so only a single image lives in RAM at any time.

    dpi controls the render resolution. 150 DPI is sufficient for OCR and
    keeps image sizes small. Use higher values (200-300) for pages with very
    small or dense text. Default: 150.

    Dispatches to VisionExtractor for AnthropicProvider/GeminiProvider, or
    MistralExtractor for MistralProvider. Both yield identical result dicts.
    """
    if isinstance(provider, MistralProvider):
        extractor = MistralExtractor(provider)
    else:
        extractor = VisionExtractor(provider)

    n_pages = page_count(pdf_path)

    for page_num in range(1, n_pages + 1):
        print(f"  Processing page {page_num}/{n_pages} …")
        t0 = time.perf_counter()

        # Load exactly one page — released at end of loop body
        page_images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=page_num,
            last_page=page_num,
        )
        page_image = page_images[0]

        try:
            extract_result = extractor.extract(page_image, page_num)
            # MistralExtractor returns 13-tuple (with ocr_cost_usd, structure_cost_usd);
            # VisionExtractor returns 11-tuple. Unpack the common fields first.
            (
                skip,
                skip_reason,
                text,
                title,
                book_page,
                tables,
                watermarks,
                usage,
                method,
                warnings,
                paraphrased,
                *extra_costs,
            ) = extract_result
            ocr_cost_usd, structure_cost_usd = (
                extra_costs if extra_costs else (None, None)
            )
            elapsed = time.perf_counter() - t0

            if skip:
                print(f"    Page {page_num} skipped ({skip_reason}).")
            elif warnings:
                print(f"    Page {page_num} stitched with warnings: {warnings}")
            elif method == "quadrant-split":
                print(f"    Page {page_num} stitched successfully from quadrants.")

            page_dict: dict = {
                "page": page_num,
                "model": provider.model_id,
                "dpi": dpi,
                "skip": skip,
                "skip_reason": skip_reason,
                "text": text,
                "tables": tables,
                "watermarks": watermarks,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_creation_tokens": usage.cache_creation_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "cost_usd": usage.cost_usd,
                "elapsed_seconds": round(elapsed, 2),
                "book_page": book_page,
                "title": title,
                "extraction_method": method,
                "chunk_warnings": warnings,
                "paraphrased": paraphrased,
                "error": None,
            }
            if ocr_cost_usd is not None:
                page_dict["ocr_cost_usd"] = ocr_cost_usd
                page_dict["structure_cost_usd"] = structure_cost_usd
            yield page_dict

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            error_msg = str(exc)
            # Determine the correct failed method label based on extractor type
            failed_method = (
                "ocr+structure"
                if isinstance(extractor, MistralExtractor)
                else "full-page"
            )
            print(f"  ⛔ ERROR Page {page_num}: {error_msg}")
            print(
                f"     ↳ Page {page_num} content is MISSING from the output — manual intervention required."
            )
            yield {
                "page": page_num,
                "model": provider.model_id,
                "dpi": dpi,
                "skip": False,
                "skip_reason": None,
                "text": "",
                "tables": [],
                "watermarks": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "cost_usd": 0.0,
                "elapsed_seconds": round(elapsed, 2),
                "extraction_method": failed_method,
                "chunk_warnings": [],
                "paraphrased": None,
                "book_page": None,
                "error": error_msg,
            }

        finally:
            # Explicitly release the PIL image and its pixel buffer
            page_image.close()
            del page_image
            del page_images
