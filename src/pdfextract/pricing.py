"""
Token cost estimation for Anthropic, Gemini, and Mistral models.
Fetches live pricing from litellm's public JSON, with hardcoded fallbacks
for models not yet indexed.
"""

from functools import lru_cache

import httpx


@lru_cache(maxsize=1)
def _fetch_litellm_pricing() -> dict:
    url = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
    r = httpx.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


def anthropic_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> float:
    all_models = _fetch_litellm_pricing()
    pricing = {
        k.replace("anthropic/", ""): v
        for k, v in all_models.items()
        if v.get("litellm_provider") == "anthropic"
    }
    p = pricing.get(model) or pricing.get(model.rsplit("-", 1)[0])
    if not p:
        raise ValueError(f"No Anthropic pricing for '{model}'.")
    return (
        (input_tokens / 1e6) * p.get("input_cost_per_token", 0) * 1e6
        + (output_tokens / 1e6) * p.get("output_cost_per_token", 0) * 1e6
        + (cache_creation / 1e6) * p.get("cache_creation_input_token_cost", 0) * 1e6
        + (cache_read / 1e6) * p.get("cache_read_input_token_cost", 0) * 1e6
    )


def gemini_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    all_models = _fetch_litellm_pricing()
    pricing = {
        k.replace("gemini/", ""): v
        for k, v in all_models.items()
        if v.get("litellm_provider") == "google"
    }
    p = pricing.get(model) or pricing.get(model.rsplit("-", 1)[0])
    if not p:
        FALLBACK: dict[str, tuple[float, float]] = {
            "gemini-2.5-flash-lite": (0.10, 0.40),
            "gemini-2.5-flash": (0.30, 1.00),
            "gemini-2.5-pro": (1.25, 10.00),
            "gemini-2.0-flash-lite": (0.075, 0.30),
            "gemini-2.0-flash": (0.10, 0.40),
        }
        rates = FALLBACK.get(model)
        if not rates:
            raise ValueError(f"No Gemini pricing for '{model}'.")
        inp_rate, out_rate = rates
        return (input_tokens / 1e6) * inp_rate + (output_tokens / 1e6) * out_rate
    return (input_tokens / 1e6) * p.get("input_cost_per_token", 0) * 1e6 + (
        output_tokens / 1e6
    ) * p.get("output_cost_per_token", 0) * 1e6


def mistral_cost(
    ocr_model: str,
    chat_model: str,
    ocr_pages: int,
    chat_input_tokens: int,
    chat_output_tokens: int,
) -> float:
    """
    Mistral OCR is billed per page processed, not per token.
    The structuring chat call is billed per token.
    Rates are hardcoded as Mistral OCR is not yet in litellm.
    """
    OCR_COST_PER_PAGE: dict[str, float] = {
        "mistral-ocr-latest": 0.001,  # $0.001 per page
    }
    CHAT_RATES: dict[str, tuple[float, float]] = {
        "mistral-small-latest": (0.10, 0.30),
        "mistral-medium-latest": (0.40, 1.20),
        "mistral-large-latest": (2.00, 6.00),
    }
    ocr_rate = OCR_COST_PER_PAGE.get(ocr_model, 0.001)
    chat_inp_rate, chat_out_rate = CHAT_RATES.get(chat_model, (0.10, 0.30))
    return (
        ocr_pages * ocr_rate
        + (chat_input_tokens / 1e6) * chat_inp_rate
        + (chat_output_tokens / 1e6) * chat_out_rate
    )
