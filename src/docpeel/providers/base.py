"""
Shared foundations for all providers: Usage accumulator, PAGE_EXTRACTION_SCHEMA,
the VisionProvider ABC, and the _with_retry helper.

Nothing in this module imports from the concrete provider modules — it is a
pure dependency base with no circular-import risk.
"""

import logging
import random
import time
from abc import ABC
from abc import abstractmethod
from typing import NamedTuple

from PIL import Image

logger = logging.getLogger(__name__)

# ── Retry configuration ───────────────────────────────────────────────────────

_MAX_RETRIES = 4
_BASE_DELAY = 5.0  # seconds before first retry
_MAX_DELAY = 60.0  # cap on backoff delay
_JITTER = 0.25  # ±25% random jitter


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter: base * 2^attempt ± jitter."""
    delay = min(_BASE_DELAY * (2**attempt), _MAX_DELAY)
    delay *= 1 + _JITTER * (2 * random.random() - 1)
    return delay


def _is_rate_limit_error(exc: Exception) -> bool:
    raw = getattr(exc, "raw_response", None)
    return getattr(raw, "status_code", None) == 429


def _build_rate_limit_message(exc: Exception, step: str = "") -> str:
    """
    Interpret rate-limit headers into a plain-English diagnosis.
    Falls back to a generic message if no headers are available.
    """
    step_label = f" on {step}" if step else ""
    raw = getattr(exc, "raw_response", None)
    headers = dict(getattr(raw, "headers", {})) if raw else {}

    def _int(key: str) -> int | None:
        v = headers.get(key)
        return int(v) if v is not None else None

    monthly_remaining = _int("x-ratelimit-remaining-tokens-month")
    monthly_limit = _int("x-ratelimit-limit-tokens-month")
    minute_remaining = _int("x-ratelimit-remaining-tokens-minute")
    minute_limit = _int("x-ratelimit-limit-tokens-minute")

    if monthly_remaining is not None and monthly_remaining == 0:
        used = f"{monthly_limit:,}" if monthly_limit is not None else "unknown"
        return (
            f"Rate limit (429){step_label}: monthly token quota exhausted "
            f"(0 / {used} tokens remaining this month).\n"
            f"        Action: wait for the monthly quota to reset, or upgrade your plan."
        )

    if minute_remaining is not None and minute_remaining == 0:
        limit = f"{minute_limit:,}" if minute_limit is not None else "unknown"
        return (
            f"Rate limit (429){step_label}: per-minute token limit reached "
            f"(0 / {limit} tokens remaining this minute).\n"
            f"        Action: wait a minute and retry."
        )

    # Headers present but no known zero-remaining field — show them raw
    if headers:
        header_str = ", ".join(f"{k}: {v}" for k, v in headers.items())
        return f"Rate limit (429){step_label}. Headers: {header_str}"

    return f"Rate limit (429){step_label}: no further detail available from the API."


def _with_retry(is_filter_error, fn, step: str = ""):
    """
    Call fn(), retrying on transient errors up to _MAX_RETRIES times.
    Content-filter errors and rate-limit (429) errors are re-raised immediately.

    is_filter_error: callable(exc) -> bool
    step: optional label for the operation (shown in rate-limit messages)
    """
    last_exc = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            result = fn()
            if attempt > 0:
                step_label = f" [{step}]" if step else ""
                logger.info("        ✓ Retry succeeded%s (attempt %d)", step_label, attempt + 1)
            return result

        except Exception as exc:
            if is_filter_error(exc):
                raise
            if _is_rate_limit_error(exc):
                logger.warning("        ⛔ %s", _build_rate_limit_message(exc, step))
                raise
            last_exc = exc
            if attempt < _MAX_RETRIES:
                delay = _backoff_delay(attempt)
                step_label = f" [{step}]" if step else ""
                logger.warning(
                    "        ⚠ Transient error%s (attempt %d/%d): %s — retrying in %.1fs …",
                    step_label, attempt + 1, _MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
    raise last_exc


# ── Usage accumulator ─────────────────────────────────────────────────────────


class Usage(NamedTuple):
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int  # Anthropic-only; always 0 for Gemini/Mistral
    cache_read_tokens: int  # Anthropic-only; always 0 for Gemini/Mistral
    cost_usd: float

    @staticmethod
    def zero() -> "Usage":
        return Usage(0, 0, 0, 0, 0.0)

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_creation_tokens + other.cache_creation_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cost_usd + other.cost_usd,
        )


# ── Structured-output schema (shared by all providers) ────────────────────────

# Used by VisionProviders (tool_choice / response_schema) and by
# MistralProvider's chat structuring prompt.
# Quadrant calls are plain-text and do NOT use this schema.
PAGE_EXTRACTION_SCHEMA = {
    "name": "page_extraction",
    "description": (
        "Return the structured extraction result for one PDF page or stitch call."
    ),
    "properties": {
        "skip": {
            "type": "boolean",
            "description": (
                "Set to true if this page should be excluded from the RAG system "
                "because it contains no extractable structured content. "
                "Pages that must be skipped: table of contents pages, index pages, "
                "blank pages, illustration-only pages with no text, "
                "'This page intentionally left blank' pages, half-title pages "
                "(only the book title, nothing else), part divider pages "
                "(decorative pages that only announce a part or book number, "
                "e.g. 'Part II' or 'Book Two'). "
                "Set to false for all other pages."
            ),
        },
        "skip_reason": {
            "type": ["string", "null"],
            "description": (
                "If skip is true, a short label identifying why this page is skipped. "
                "Use one of: 'table_of_contents', 'index', 'blank', "
                "'illustration_only', 'title_page', 'part_divider'. "
                "Set to null if skip is false."
            ),
        },
        "page_number": {
            "type": ["integer", "null"],
            "description": (
                "Printed page number visible in the margin "
                "(small isolated number at the top/bottom edge). "
                "null if no printed page number is visible."
            ),
        },
        "title": {
            "type": ["string", "null"],
            "description": (
                "Single prominent display title for this page/section — "
                "large decorative font, centred at the top, clearly set apart "
                "from body text (e.g. 'Chapter 3: Market Analysis', 'Appendix A: Methodology'). "
                "Combine label + subtitle into one string. Plain text only, "
                "no markdown. null if no such heading exists."
            ),
        },
        "text": {
            "type": "string",
            "description": "Verbatim transcription of all page text excluding table bodies.",
        },
        "tables": {
            "type": "array",
            "description": (
                "One entry per table found on the page, in the order they appear. "
                "Empty array if the page contains no tables or is skipped."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The heading label for this table as it appears in the document (e.g. 'Table 7: RACIAL ABILITY REQUIREMENTS'), plain text. Omit or null if no heading exists.",
                    },
                    "caption": {
                        "type": "string",
                        "description": "1-2 sentence semantic description of what the table contains and what questions it answers, for RAG retrieval.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full markdown table preserving all headers and values.",
                    },
                    "placement": {
                        "type": "object",
                        "description": "Position of the table within the page text. Null only when skip is true.",
                        "properties": {
                            "char_offset": {
                                "type": "integer",
                                "description": "Character offset into the text field immediately before which this table appeared.",
                            },
                            "context_before": {
                                "type": "string",
                                "description": "Last 10-15 words of text immediately before the table.",
                            },
                            "context_after": {
                                "type": "string",
                                "description": "First 10-15 words of text immediately after the table.",
                            },
                        },
                        "required": ["char_offset", "context_before", "context_after"],
                    },
                },
                "required": ["caption", "content"],
            },
        },
        "watermarks": {
            "type": "array",
            "description": (
                "List of watermark or ownership stamp strings that were found on "
                "the page and excluded from the text field. Each entry is the "
                "exact text of one watermark as it appeared on the page. "
                "Empty array if no watermarks were detected."
            ),
            "items": {"type": "string"},
        },
    },
    "required": [
        "skip",
        "skip_reason",
        "page_number",
        "title",
        "text",
        "tables",
        "watermarks",
    ],
}


# ── VisionProvider ABC ────────────────────────────────────────────────────────


class VisionProvider(ABC):
    """
    Interface for providers that perform OCR and structuring in a single vision
    call (Anthropic, Gemini). Extraction logic in VisionExtractor never imports
    SDK libraries directly — all calls go through this abstraction.

    Not implemented by MistralProvider, which uses a separate two-step pipeline
    (OCR then structuring) owned entirely by MistralExtractor.
    """

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def call(self, image: Image.Image, prompt: str) -> tuple[str, Usage]:
        """Send one image + prompt, return (text, usage). Used for quadrant calls."""

    @abstractmethod
    def call_structured(self, image: Image.Image, prompt: str) -> tuple[dict, Usage]:
        """
        Send one image + prompt and return a guaranteed-structured dict enforced
        via the provider's native tool/function calling mechanism.
        """

    @abstractmethod
    def call_with_image_and_text_structured(
        self, image: Image.Image, text_prompt: str
    ) -> tuple[dict, Usage]:
        """
        Structured stitch call: send the original page image alongside assembled
        quadrant texts. Returns the same dict shape as call_structured.
        """

    @abstractmethod
    def is_content_filter_error(self, exc: Exception) -> bool:
        """Return True if exc is a content-filter block, not a real API error."""

    @abstractmethod
    def resolve_model_id(self) -> None:
        """Pre-flight API call to resolve any alias in self._model to its canonical ID."""
