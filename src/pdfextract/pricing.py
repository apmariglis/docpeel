"""
Token cost estimation for Anthropic and Gemini models.
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
        (input_tokens     / 1e6) * p.get("input_cost_per_token", 0)              * 1e6
        + (output_tokens  / 1e6) * p.get("output_cost_per_token", 0)             * 1e6
        + (cache_creation / 1e6) * p.get("cache_creation_input_token_cost", 0)   * 1e6
        + (cache_read     / 1e6) * p.get("cache_read_input_token_cost", 0)        * 1e6
    )


def gemini_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    all_models = _fetch_litellm_pricing()
    # litellm keys Gemini as "gemini/<model>"
    pricing = {
        k.replace("gemini/", ""): v
        for k, v in all_models.items()
        if v.get("litellm_provider") == "google"
    }
    p = pricing.get(model) or pricing.get(model.rsplit("-", 1)[0])
    if not p:
        # Hardcoded fallback rates (per-million tokens) for models not yet in litellm
        FALLBACK: dict[str, tuple[float, float]] = {
            "gemini-2.5-flash-lite": (0.10,  0.40),
            "gemini-2.5-flash":      (0.30,  1.00),
            "gemini-2.5-pro":        (1.25, 10.00),
            "gemini-2.0-flash-lite": (0.075, 0.30),
            "gemini-2.0-flash":      (0.10,  0.40),
        }
        rates = FALLBACK.get(model)
        if not rates:
            raise ValueError(f"No Gemini pricing for '{model}'.")
        inp_rate, out_rate = rates
        return (input_tokens / 1e6) * inp_rate + (output_tokens / 1e6) * out_rate
    return (
        (input_tokens   / 1e6) * p.get("input_cost_per_token",  0) * 1e6
        + (output_tokens / 1e6) * p.get("output_cost_per_token", 0) * 1e6
    )