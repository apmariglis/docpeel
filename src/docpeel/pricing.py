"""
Token cost estimation for Anthropic, Gemini, and Mistral models.

Fetches live pricing from litellm's public JSON index on first use and
writes the resolved rates to ~/.cache/docpeel/pricing.json.  On subsequent
runs the cache is refreshed whenever litellm is reachable; if it is not,
the cached rates are used silently.  When neither litellm nor a local cache
is available all cost functions return None and a single WARNING is logged.
"""

import json
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main"
    "/model_prices_and_context_window.json"
)

# Mistral OCR bootstrap rates — written to pricing.json on first use.
# Not permanently hardcoded: once in the cache they can be edited by the user.
_MISTRAL_OCR_BOOTSTRAP: dict[str, float] = {
    "mistral-ocr-latest": 0.001,  # $0.001 per page (Mistral's published rate)
}


# ── Cache helpers ─────────────────────────────────────────────────────────────


def _cache_path() -> Path:
    return Path.home() / ".cache" / "docpeel" / "pricing.json"


def _load_cache() -> dict | None:
    p = _cache_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


_CACHE_NOTE = (
    "Rates are USD per million tokens (input/output) or USD per page (ocr_per_page). "
    "Anthropic models may also have cache_creation and cache_read rates. "
    "Values are sourced from litellm or bootstrapped on first use — edit freely."
)


def _save_cache(models: dict) -> None:
    """Merge *models* into the cache file, updating last_updated."""
    p = _cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_cache() or {"models": {}}
    existing["_note"] = _CACHE_NOTE
    existing.setdefault("models", {}).update(models)
    existing["last_updated"] = datetime.now().isoformat(timespec="seconds")
    p.write_text(json.dumps(existing, indent=2))


# ── litellm fetch (cached per process) ───────────────────────────────────────


@lru_cache(maxsize=1)
def _fetch_litellm_pricing() -> dict:
    r = httpx.get(_LITELLM_URL, timeout=5)
    r.raise_for_status()
    return r.json()


# ── Internal calculators ──────────────────────────────────────────────────────


def _calc_anthropic(
    rates: dict,
    input_tokens: int,
    output_tokens: int,
    cache_creation: int,
    cache_read: int,
) -> float:
    return (
        (input_tokens / 1e6) * rates.get("input", 0)
        + (output_tokens / 1e6) * rates.get("output", 0)
        + (cache_creation / 1e6) * rates.get("cache_creation", 0)
        + (cache_read / 1e6) * rates.get("cache_read", 0)
    )


def _calc_simple(rates: dict, input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1e6) * rates.get("input", 0)
        + (output_tokens / 1e6) * rates.get("output", 0)
    )


def _warn_no_pricing() -> None:
    logger.warning(
        "Pricing unavailable: litellm is unreachable and no local cache exists. "
        "Costs will not be reported. Run with network access to populate the cache."
    )


# ── Public cost functions ─────────────────────────────────────────────────────


def anthropic_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> float | None:
    # Try litellm
    try:
        all_models = _fetch_litellm_pricing()
        pricing = {
            k.replace("anthropic/", ""): v
            for k, v in all_models.items()
            if v.get("litellm_provider") == "anthropic"
        }
        p = pricing.get(model) or pricing.get(model.rsplit("-", 1)[0])
        if p:
            rates = {
                "input": p.get("input_cost_per_token", 0) * 1e6,
                "output": p.get("output_cost_per_token", 0) * 1e6,
                "cache_creation": p.get("cache_creation_input_token_cost", 0) * 1e6,
                "cache_read": p.get("cache_read_input_token_cost", 0) * 1e6,
            }
            _save_cache({model: rates})
            return _calc_anthropic(rates, input_tokens, output_tokens, cache_creation, cache_read)
    except Exception:
        pass

    # Try local cache
    cache = _load_cache()
    if cache is not None:
        m = cache.get("models", {}).get(model)
        if m:
            return _calc_anthropic(m, input_tokens, output_tokens, cache_creation, cache_read)
    else:
        _warn_no_pricing()

    return None


def gemini_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    # Try litellm
    try:
        all_models = _fetch_litellm_pricing()
        pricing = {
            k.replace("gemini/", ""): v
            for k, v in all_models.items()
            if v.get("litellm_provider") in ("google", "gemini")
        }
        p = pricing.get(model) or pricing.get(model.rsplit("-", 1)[0])
        if p:
            rates = {
                "input": p.get("input_cost_per_token", 0) * 1e6,
                "output": p.get("output_cost_per_token", 0) * 1e6,
            }
            _save_cache({model: rates})
            return _calc_simple(rates, input_tokens, output_tokens)
    except Exception:
        pass

    # Try local cache
    cache = _load_cache()
    if cache is not None:
        m = cache.get("models", {}).get(model)
        if m:
            return _calc_simple(m, input_tokens, output_tokens)
    else:
        _warn_no_pricing()

    return None


def mistral_cost(
    ocr_model: str,
    chat_model: str,
    ocr_pages: int,
    chat_input_tokens: int,
    chat_output_tokens: int,
) -> float | None:
    # Resolve OCR rate: cache → bootstrap default (written to cache on first use)
    cache = _load_cache()
    ocr_rate = (cache or {}).get("models", {}).get(ocr_model, {}).get("ocr_per_page")
    if ocr_rate is None:
        ocr_rate = _MISTRAL_OCR_BOOTSTRAP.get(ocr_model, 0.001)
        _save_cache({ocr_model: {"ocr_per_page": ocr_rate}})
    ocr_cost = ocr_pages * ocr_rate

    # If no chat tokens, OCR cost alone is sufficient
    if chat_input_tokens == 0 and chat_output_tokens == 0:
        return float(ocr_cost)

    # Try litellm for chat rates
    chat_rates = None
    try:
        all_models = _fetch_litellm_pricing()
        pricing = {
            k.replace("mistral/", ""): v
            for k, v in all_models.items()
            if v.get("litellm_provider") == "mistral"
        }
        p = pricing.get(chat_model) or pricing.get(chat_model.rsplit("-", 1)[0])
        if p:
            chat_rates = {
                "input": p.get("input_cost_per_token", 0) * 1e6,
                "output": p.get("output_cost_per_token", 0) * 1e6,
            }
            _save_cache({chat_model: chat_rates})
    except Exception:
        pass

    if chat_rates is None:
        if cache is not None:
            chat_rates = cache.get("models", {}).get(chat_model)
        else:
            _warn_no_pricing()

    if chat_rates is None:
        logger.warning(
            "Structure model pricing unavailable for '%s' — "
            "showing OCR cost only (structure cost excluded).",
            chat_model,
        )
        return float(ocr_cost)

    return ocr_cost + _calc_simple(chat_rates, chat_input_tokens, chat_output_tokens)
