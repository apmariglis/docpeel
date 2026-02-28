"""
Core page extraction logic: per-chunk fallback chain and page-level orchestration.

Fallback chain per quadrant:
  1. Verbatim extraction (quadrant_extract_prompt)
  2. Obfuscated retry (rotation + noise)
  3. Paraphrase (describe-don't-transcribe) — flagged for manual review

Fallback chain for the stitch call:
  1. Image-assisted stitch
  2. Full-page paraphrase if stitch is blocked

Memory model
------------
convert_from_path() is called one page at a time (first_page/last_page) so
only a single PIL image lives in RAM at once. The page image is explicitly
deleted after extraction. Results are yielded one at a time so the caller
can stream them to disk rather than accumulating a full-book list.
"""

import json
import time
from collections.abc import Iterator
from pathlib import Path

from pdf2image import convert_from_path
from pdf2image import pdfinfo_from_path
from PIL import Image

from image_utils import obfuscate
from image_utils import split_quadrants
from prompts import PAGE_EXTRACT_PROMPT
from prompts import PARAPHRASE_PROMPT
from prompts import STITCH_PROMPT
from prompts import quadrant_extract_prompt
from providers import LLMProvider
from providers import Usage


def _repair_json(s: str) -> str:
    """
    Fix JSON where string values contain:
      - Unescaped literal newlines/carriage returns (common Gemini artefact)
      - Invalid backslash escapes such as \* from markdown (e.g. \\* in tables)
    Walks the string character by character tracking whether we are inside a
    JSON string value and corrects both classes of error in one pass.
    """
    VALID_ESCAPES = set('"\\/bfnrtu')
    result = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and in_string:
            next_c = s[i + 1] if i + 1 < len(s) else ""
            if next_c in VALID_ESCAPES:
                result.append(c)  # valid escape — keep both chars
                result.append(next_c)
                i += 2
            else:
                result.append("\\\\")  # invalid escape — double the backslash
                i += 1  # next_c processed on next iteration
            continue
        if c == '"':
            in_string = not in_string
            result.append(c)
        elif c == "\n" and in_string:
            result.append("\\n")
        elif c == "\r" and in_string:
            result.append("\\r")
        else:
            result.append(c)
        i += 1
    return "".join(result)


def _parse_page_response(raw: str) -> tuple[str, str | None, int | None]:
    """
    Parse a whole-page model response that should be JSON:
        {"page_number": <int|null>, "title": <str|null>, "text": "<content>"}

    Returns (text, title, page_number). Falls back gracefully if the model
    returned plain text instead of JSON: (raw, None, None).
    """
    raw = raw.strip()
    # Strip markdown code fences if the model wrapped the JSON anyway
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()

    def _try_parse(s):
        obj = json.loads(s)
        text = obj.get("text", raw)
        title = obj.get("title")
        if not isinstance(title, str):
            title = None
        page_number = obj.get("page_number")
        if isinstance(page_number, float):
            page_number = int(page_number)
        if not isinstance(page_number, int):
            page_number = None
        return text, title, page_number

    try:
        return _try_parse(raw)
    except (json.JSONDecodeError, AttributeError):
        pass
    # Second attempt: repair unescaped newlines inside string values
    try:
        return _try_parse(_repair_json(raw))
    except (json.JSONDecodeError, AttributeError):
        # Model returned plain text — use as-is, no title or page number
        return raw, None, None


def _paraphrase(
    provider: LLMProvider,
    img: Image.Image,
    is_full_page: bool = False,
) -> tuple[str, str | None, int | None, Usage]:
    """
    Last-resort extraction: ask the model to paraphrase rather than transcribe
    verbatim. Used when even an obfuscated image is blocked by the
    content/recitation filter. The caller records the warning.
    is_full_page: if True, parse JSON response (whole page image visible).
                  if False, treat as plain text (quadrant slice).
    """
    raw, usage = provider.call(img, PARAPHRASE_PROMPT)
    if is_full_page:
        text, title, page_number = _parse_page_response(raw)
        return text, title, page_number, usage
    return raw, None, None, usage


def _extract_with_fallback(
    provider: LLMProvider,
    page_image: Image.Image,
    page_num: int,
) -> tuple[str, str | None, int | None, Usage, str, list[str], str | None]:
    """
    Extract text from a single page image with a multi-stage fallback chain.

    Returns:
        (text, title, book_page, usage, extraction_method, warnings, paraphrased)
        paraphrased: None | "partial" | "full"
    """
    # ── Stage 1: full-page verbatim ───────────────────────────────────────────
    try:
        raw, usage = provider.call(page_image, PAGE_EXTRACT_PROMPT)
        text, title, book_page = _parse_page_response(raw)
        return text, title, book_page, usage, "full-page", [], None
    except Exception as exc:
        if not provider.is_content_filter_error(exc):
            raise
        print("    Content filter triggered — splitting into quadrants …")

    # ── Stage 2: quadrant split ───────────────────────────────────────────────
    quadrants = split_quadrants(page_image)
    chunk_texts: dict[str, str] = {}
    total_usage = Usage.zero()
    warnings: list[str] = []
    paraphrased_quads: list[str] = []  # quadrant labels that were paraphrased

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
                warnings.append(f"chunk '{label}' failed on obfuscated retry: {exc2}")
                print(f"      Chunk {label}: error on obfuscated retry — skipped")
                continue

        # Stage 2c: paraphrase
        print(f"      Chunk {label}: blocked after obfuscation — paraphrasing ...")
        try:
            text, _, _, usage = _paraphrase(provider, quad_img, is_full_page=False)
            chunk_texts[label] = text
            total_usage = total_usage + usage
            paraphrased_quads.append(label)
            warnings.append(
                f"chunk '{label}' could not be transcribed verbatim — "
                "paraphrased instead (manual review recommended)"
            )
            print(f"      Chunk {label}: OK (paraphrased — manual review recommended)")
        except Exception as exc3:
            chunk_texts[label] = ""
            if provider.is_content_filter_error(exc3):
                warnings.append(
                    f"chunk '{label}' blocked even on paraphrase attempt — skipped"
                )
                print(f"      Chunk {label}: blocked on paraphrase — skipped")
            else:
                warnings.append(f"chunk '{label}' failed on paraphrase attempt: {exc3}")
                print(f"      Chunk {label}: error on paraphrase — skipped")

    # ── Image-only page detection ─────────────────────────────────────────────
    # If every quadrant is empty the page is pure artwork with no text.
    # Skip the stitch and paraphrase entirely — return empty text cleanly.
    if not any(t.strip() for t in chunk_texts.values()):
        print("      All quadrants empty — image-only page, skipping stitch.")
        return "", None, None, total_usage, "quadrant-split", warnings, None

    # ── Stage 3: stitch ───────────────────────────────────────────────────────
    stitch_prompt = STITCH_PROMPT
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
        raw, stitch_usage = provider.call_with_image_and_text(
            page_image, stitch_prompt + chunk_block
        )
        stitched, title, book_page = _parse_page_response(raw)
    except Exception as exc:
        if not provider.is_content_filter_error(exc):
            raise
        print("      Stitch blocked by content filter — paraphrasing full page …")
        stitched, title, book_page, stitch_usage = _paraphrase(
            provider, page_image, is_full_page=True
        )
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
        stitched,
        title,
        book_page,
        total_usage,
        "quadrant-split",
        warnings,
        paraphrased,
    )


def page_count(pdf_path: Path) -> int:
    """Return the total number of pages in the PDF without loading any images."""
    info = pdfinfo_from_path(str(pdf_path))
    return info["Pages"]


def iter_pages(
    pdf_path: Path,
    provider: LLMProvider,
    dpi: int = 150,
) -> Iterator[dict]:
    """
    Yield one result dict per page, converting and releasing each page image
    immediately so only a single image lives in RAM at any time.

    dpi controls the render resolution. 150 DPI is sufficient for OCR and
    keeps image sizes small. Use higher values (200-300) for pages with very
    small or dense text. Default: 150.

    This is the memory-efficient core — callers should stream results to disk
    as they arrive rather than collecting them into a list.
    """
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
            text, title, book_page, usage, method, warnings, paraphrased = (
                _extract_with_fallback(provider, page_image, page_num)
            )
            elapsed = time.perf_counter() - t0

            error = None
            if warnings:
                print(f"    Page {page_num} stitched with warnings: {warnings}")
            elif method == "quadrant-split":
                print(f"    Page {page_num} stitched successfully from quadrants.")

            yield {
                "page": page_num,
                "model": provider.model_id,
                "dpi": dpi,
                "text": text,
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
                "error": error,
            }

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            error_msg = str(exc)
            print(f"  ERROR Page {page_num}: {error_msg}")
            yield {
                "page": page_num,
                "model": provider.model_id,
                "dpi": dpi,
                "text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_tokens": 0,
                "cache_read_tokens": 0,
                "cost_usd": 0.0,
                "elapsed_seconds": round(elapsed, 2),
                "extraction_method": "full-page",
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
