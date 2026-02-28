"""
LLM provider abstraction: Usage type, LLMProvider ABC, and concrete
implementations for Anthropic and Gemini.

Retry behaviour
---------------
All three call methods are wrapped with exponential backoff for transient
errors (network timeouts, 429 rate-limits, 503/504 gateway errors).
Content-filter blocks are NOT retried — they are returned immediately so
the extraction fallback chain can handle them.
"""

import base64
import io
import logging
import random
import time
from abc import ABC
from abc import abstractmethod
from typing import NamedTuple

from PIL import Image

from pricing import anthropic_cost
from pricing import gemini_cost

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


def _with_retry(provider: "LLMProvider", fn):
    """
    Call fn(), retrying on transient errors up to _MAX_RETRIES times.
    Content-filter errors are re-raised immediately without retry.
    """
    last_exc = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:
            if provider.is_content_filter_error(exc):
                raise
            last_exc = exc
            if attempt < _MAX_RETRIES:
                delay = _backoff_delay(attempt)
                logger.warning(
                    "Transient error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    _MAX_RETRIES,
                    exc,
                    delay,
                )
                print(
                    f"        ⚠ Transient error (attempt {attempt + 1}/{_MAX_RETRIES}): "
                    f"{exc} — retrying in {delay:.1f}s …"
                )
                time.sleep(delay)
    raise last_exc


# ── Usage accumulator ─────────────────────────────────────────────────────────


class Usage(NamedTuple):
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int  # Anthropic-only; always 0 for Gemini
    cache_read_tokens: int  # Anthropic-only; always 0 for Gemini
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


# ── Provider abstraction ──────────────────────────────────────────────────────


class LLMProvider(ABC):
    """
    Provider-agnostic interface. Extraction logic never imports anthropic
    or google directly — all SDK calls go through this abstraction.
    """

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def call(self, image: Image.Image, prompt: str) -> tuple[str, Usage]:
        """Send one image + prompt, return (text, usage)."""

    @abstractmethod
    def call_with_image_and_text(
        self, image: Image.Image, text_prompt: str
    ) -> tuple[str, Usage]:
        """Send image + long text block together (stitch-with-image call)."""

    @abstractmethod
    def is_content_filter_error(self, exc: Exception) -> bool:
        """Return True if exc is a content-filter block, not a real API error."""


# ── Anthropic ─────────────────────────────────────────────────────────────────


class AnthropicProvider(LLMProvider):
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, model: str | None = None):
        import anthropic as _anthropic

        self._anthropic = _anthropic
        self._client = _anthropic.Anthropic()
        self._model = model or self.DEFAULT_MODEL

    @property
    def model_id(self) -> str:
        return self._model

    def _parse_usage(self, response) -> Usage:
        u = response.usage
        cost = anthropic_cost(
            model=response.model,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_creation=u.cache_creation_input_tokens or 0,
            cache_read=u.cache_read_input_tokens or 0,
        )
        return Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_creation_tokens=u.cache_creation_input_tokens or 0,
            cache_read_tokens=u.cache_read_input_tokens or 0,
            cost_usd=cost,
        )

    def _create(self, content):
        return self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": content}],
        )

    def _to_b64(self, image: Image.Image) -> str:
        from image_utils import to_b64_safe

        return to_b64_safe(image)

    def call(self, image: Image.Image, prompt: str) -> tuple[str, Usage]:
        b64 = self._to_b64(image)
        resp = _with_retry(
            self,
            lambda: self._create(
                [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ]
            ),
        )
        return resp.content[0].text, self._parse_usage(resp)

    def call_with_image_and_text(
        self, image: Image.Image, text_prompt: str
    ) -> tuple[str, Usage]:
        b64 = self._to_b64(image)
        resp = _with_retry(
            self,
            lambda: self._create(
                [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": text_prompt},
                ]
            ),
        )
        return resp.content[0].text, self._parse_usage(resp)

    def is_content_filter_error(self, exc: Exception) -> bool:
        return isinstance(exc, self._anthropic.BadRequestError)


# ── Gemini ────────────────────────────────────────────────────────────────────


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


class GeminiProvider(LLMProvider):
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

        - Blocked reason  → _GeminiContentFilterError  (no retry, fallback chain handles it)
        - STOP but no text → RuntimeError              (transient, _with_retry will retry)
        - Anything else   → let SDK exception propagate (transient, _with_retry will retry)
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

    def call(self, image: Image.Image, prompt: str) -> tuple[str, Usage]:
        return _with_retry(self, lambda: self._generate([image, prompt]))

    def call_with_image_and_text(
        self, image: Image.Image, text_prompt: str
    ) -> tuple[str, Usage]:
        return _with_retry(self, lambda: self._generate([image, text_prompt]))

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


# ── Factory ───────────────────────────────────────────────────────────────────


def build_provider(name: str, model: str | None = None) -> LLMProvider:
    name = name.lower()
    if name == "anthropic":
        return AnthropicProvider(model)
    if name == "gemini":
        return GeminiProvider(model)
    raise ValueError(f"Unknown provider '{name}'. Choose: anthropic, gemini")
