"""
Tests for pricing.py — cost calculation for Anthropic, Gemini, and Mistral.
The litellm network fetch and filesystem cache are mocked throughout.
"""

from unittest.mock import MagicMock, patch

import pytest

import src.docpeel.pricing as pricing_mod
from src.docpeel.pricing import anthropic_cost, gemini_cost, mistral_cost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_PRICING = {
    "anthropic/claude-3-haiku-20240307": {
        "litellm_provider": "anthropic",
        "input_cost_per_token": 0.00000025,
        "output_cost_per_token": 0.00000125,
        "cache_creation_input_token_cost": 0.0000003,
        "cache_read_input_token_cost": 0.00000003,
    },
    "anthropic/claude-3-5-sonnet-20241022": {
        "litellm_provider": "anthropic",
        "input_cost_per_token": 0.000003,
        "output_cost_per_token": 0.000015,
        "cache_creation_input_token_cost": 0.00000375,
        "cache_read_input_token_cost": 0.0000003,
    },
    "gemini/gemini-2.0-flash": {
        "litellm_provider": "google",
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000004,
    },
    "mistral/mistral-small-latest": {
        "litellm_provider": "mistral",
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000003,
    },
    "mistral/mistral-large-latest": {
        "litellm_provider": "mistral",
        "input_cost_per_token": 0.000002,
        "output_cost_per_token": 0.000006,
    },
}


def _patch_pricing(data=None):
    """Patch litellm fetch and disable cache writes."""
    target = data if data is not None else _FAKE_PRICING
    pricing_mod._fetch_litellm_pricing.cache_clear()
    return (
        patch.object(pricing_mod, "_fetch_litellm_pricing", return_value=target),
        patch.object(pricing_mod, "_save_cache"),
        patch.object(pricing_mod, "_load_cache", return_value=None),
    )


# Helper to use all three patches together
from contextlib import ExitStack


def with_pricing(data=None):
    stack = ExitStack()
    for ctx in _patch_pricing(data):
        stack.enter_context(ctx)
    return stack


# ---------------------------------------------------------------------------
# anthropic_cost
# ---------------------------------------------------------------------------


def test_anthropic_cost_zero_tokens():
    with with_pricing():
        cost = anthropic_cost("claude-3-haiku-20240307", 0, 0)
    assert cost == 0.0


def test_anthropic_cost_input_only():
    with with_pricing():
        cost = anthropic_cost("claude-3-haiku-20240307", 1_000_000, 0)
    # 1M * 0.00000025 * 1e6 = 0.25
    assert abs(cost - 0.25) < 1e-9


def test_anthropic_cost_output_only():
    with with_pricing():
        cost = anthropic_cost("claude-3-haiku-20240307", 0, 1_000_000)
    # 1M * 0.00000125 * 1e6 = 1.25
    assert abs(cost - 1.25) < 1e-9


def test_anthropic_cost_with_cache():
    with with_pricing():
        cost = anthropic_cost(
            "claude-3-haiku-20240307",
            input_tokens=0,
            output_tokens=0,
            cache_creation=1_000_000,
            cache_read=1_000_000,
        )
    # 0.0000003 * 1e6 + 0.00000003 * 1e6 = 0.3 + 0.03 = 0.33
    assert abs(cost - 0.33) < 1e-9


def test_anthropic_cost_unknown_model_returns_none():
    with with_pricing():
        cost = anthropic_cost("claude-nonexistent-model", 100, 100)
    assert cost is None


def test_anthropic_cost_result_non_negative():
    with with_pricing():
        cost = anthropic_cost("claude-3-5-sonnet-20241022", 500, 500)
    assert cost >= 0


# ---------------------------------------------------------------------------
# gemini_cost
# ---------------------------------------------------------------------------


def test_gemini_cost_zero_tokens():
    with with_pricing():
        cost = gemini_cost("gemini-2.0-flash", 0, 0)
    assert cost == 0.0


def test_gemini_cost_from_litellm():
    with with_pricing():
        cost = gemini_cost("gemini-2.0-flash", 1_000_000, 1_000_000)
    # 0.0000001 * 1e6 + 0.0000004 * 1e6 = 0.1 + 0.4 = 0.5
    assert abs(cost - 0.5) < 1e-9


def test_gemini_cost_unknown_model_returns_none():
    with with_pricing({}):
        cost = gemini_cost("gemini-nonexistent-99", 100, 100)
    assert cost is None


def test_gemini_cost_result_non_negative():
    with with_pricing():
        cost = gemini_cost("gemini-2.0-flash", 200, 300)
    assert cost >= 0


# ---------------------------------------------------------------------------
# mistral_cost
# ---------------------------------------------------------------------------


def test_mistral_cost_ocr_only():
    # No chat tokens — OCR rate is always known, no litellm call needed
    with with_pricing():
        cost = mistral_cost("mistral-ocr-latest", "mistral-small-latest", 5, 0, 0)
    # 5 pages × $0.001 = $0.005
    assert abs(cost - 0.005) < 1e-9


def test_mistral_cost_chat_only():
    with with_pricing():
        cost = mistral_cost("mistral-ocr-latest", "mistral-small-latest", 0, 1_000_000, 1_000_000)
    # 0.0000001 * 1e6 + 0.0000003 * 1e6 = 0.10 + 0.30 = 0.40
    assert abs(cost - 0.40) < 1e-9


def test_mistral_cost_combined():
    with with_pricing():
        cost = mistral_cost("mistral-ocr-latest", "mistral-small-latest", 10, 500_000, 200_000)
    ocr = 10 * 0.001
    chat = (500_000 / 1e6) * 0.10 + (200_000 / 1e6) * 0.30
    assert abs(cost - (ocr + chat)) < 1e-9


def test_mistral_cost_large_model():
    with with_pricing():
        cost = mistral_cost("mistral-ocr-latest", "mistral-large-latest", 0, 1_000_000, 1_000_000)
    # rates: 0.000002 * 1e6 = 2.00, 0.000006 * 1e6 = 6.00
    assert abs(cost - 8.0) < 1e-9


def test_mistral_cost_zero():
    with with_pricing():
        assert mistral_cost("mistral-ocr-latest", "mistral-small-latest", 0, 0, 0) == 0.0


def test_mistral_cost_non_negative():
    with with_pricing():
        cost = mistral_cost("mistral-ocr-latest", "mistral-small-latest", 3, 100, 200)
    assert cost >= 0
