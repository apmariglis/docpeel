"""
Tests for the redesigned pricing module:
- Live fetch from litellm → write to ~/.cache/docpeel/pricing.json
- Fallback to cache when litellm is unreachable
- Return None (not raise) when no pricing available anywhere
- Warning logged when litellm unreachable and no cache exists
- last_updated timestamp written to cache
- Usage.cost_usd accepts None; addition with None propagates None

All filesystem and HTTP calls are mocked — no real network or disk access.
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

import src.docpeel.pricing as pricing_mod
from src.docpeel.providers.base import Usage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Minimal litellm payload covering one model per provider
_LITELLM_PAYLOAD = {
    "anthropic/claude-3-haiku-20240307": {
        "litellm_provider": "anthropic",
        "input_cost_per_token": 0.00000025,
        "output_cost_per_token": 0.00000125,
        "cache_creation_input_token_cost": 0.0000003,
        "cache_read_input_token_cost": 0.00000003,
    },
    "gemini/gemini-2.0-flash": {
        "litellm_provider": "google",  # older models use "google"
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000004,
    },
    "gemini/gemini-2.5-flash-lite": {
        "litellm_provider": "gemini",  # newer models use "gemini"
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000004,
    },
    "mistral/mistral-small-latest": {
        "litellm_provider": "mistral",
        "input_cost_per_token": 0.0000001,
        "output_cost_per_token": 0.0000003,
    },
}

_CACHE_CONTENT = {
    "last_updated": "2026-01-01T00:00:00",
    "models": {
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25,
                                    "cache_creation": 0.30, "cache_read": 0.03},
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
        "mistral-small-latest": {"input": 0.10, "output": 0.30},
        "mistral-ocr-latest": {"ocr_per_page": 0.001},
    },
}


@pytest.fixture(autouse=True)
def reset_pricing_state():
    """Clear any module-level cache state between tests."""
    pricing_mod._fetch_litellm_pricing.cache_clear()
    yield
    pricing_mod._fetch_litellm_pricing.cache_clear()


def _mock_litellm_fetch(payload=None):
    """Patch _fetch_litellm_pricing to return payload (default: _LITELLM_PAYLOAD)."""
    data = payload if payload is not None else _LITELLM_PAYLOAD
    return patch.object(pricing_mod, "_fetch_litellm_pricing", return_value=data)


def _mock_litellm_unavailable():
    """Patch _fetch_litellm_pricing to raise a network error."""
    return patch.object(
        pricing_mod,
        "_fetch_litellm_pricing",
        side_effect=httpx.ConnectError("unreachable"),
    )


# ---------------------------------------------------------------------------
# Cache file management
# ---------------------------------------------------------------------------


def test_successful_fetch_creates_cache_file(tmp_path):
    cache_file = tmp_path / "pricing.json"
    with _mock_litellm_fetch(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        pricing_mod.anthropic_cost("claude-3-haiku-20240307", 100, 100)
    assert cache_file.exists()


def test_cache_file_has_last_updated(tmp_path):
    cache_file = tmp_path / "pricing.json"
    with _mock_litellm_fetch(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        pricing_mod.anthropic_cost("claude-3-haiku-20240307", 100, 100)
    data = json.loads(cache_file.read_text())
    assert "last_updated" in data
    # Should be a non-empty ISO-format string
    assert len(data["last_updated"]) >= 10


def test_cache_file_has_models_section(tmp_path):
    cache_file = tmp_path / "pricing.json"
    with _mock_litellm_fetch(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        pricing_mod.anthropic_cost("claude-3-haiku-20240307", 100, 100)
    data = json.loads(cache_file.read_text())
    assert "models" in data
    assert isinstance(data["models"], dict)
    assert len(data["models"]) > 0


def test_successful_fetch_overwrites_stale_cache(tmp_path):
    cache_file = tmp_path / "pricing.json"
    # Write a stale cache with wrong rates
    stale = {
        "last_updated": "2020-01-01T00:00:00",
        "models": {"claude-3-haiku-20240307": {"input": 999.0, "output": 999.0}},
    }
    cache_file.write_text(json.dumps(stale))

    with _mock_litellm_fetch(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        cost = pricing_mod.anthropic_cost("claude-3-haiku-20240307", 1_000_000, 0)

    # Should use live rate (0.25 per 1M input), not the stale 999.0
    assert cost is not None
    assert cost < 1.0
    # And the cache timestamp should be updated
    data = json.loads(cache_file.read_text())
    assert data["last_updated"] != "2020-01-01T00:00:00"


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------


def test_uses_cache_when_litellm_unreachable(tmp_path):
    cache_file = tmp_path / "pricing.json"
    cache_file.write_text(json.dumps(_CACHE_CONTENT))

    with _mock_litellm_unavailable(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        cost = pricing_mod.anthropic_cost("claude-3-haiku-20240307", 1_000_000, 0)

    # Cache rate: $0.25 per 1M input tokens
    assert cost is not None
    assert abs(cost - 0.25) < 1e-6


def test_returns_none_when_no_cache_and_litellm_unreachable(tmp_path):
    cache_file = tmp_path / "pricing.json"
    # No cache file exists

    with _mock_litellm_unavailable(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        cost = pricing_mod.anthropic_cost("claude-3-haiku-20240307", 100, 100)

    assert cost is None


def test_warns_when_litellm_unreachable_and_no_cache(tmp_path, caplog):
    cache_file = tmp_path / "pricing.json"

    with _mock_litellm_unavailable(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file), \
         caplog.at_level(logging.WARNING):
        pricing_mod.anthropic_cost("claude-3-haiku-20240307", 100, 100)

    assert any("pricing" in r.message.lower() for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_no_warning_when_cache_available(tmp_path, caplog):
    cache_file = tmp_path / "pricing.json"
    cache_file.write_text(json.dumps(_CACHE_CONTENT))

    with _mock_litellm_unavailable(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file), \
         caplog.at_level(logging.WARNING):
        pricing_mod.anthropic_cost("claude-3-haiku-20240307", 100, 100)

    # No warning — cache silently covers the gap
    assert not any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# Cost functions — happy path (return float)
# ---------------------------------------------------------------------------


def test_anthropic_cost_returns_float(tmp_path):
    cache_file = tmp_path / "pricing.json"
    with _mock_litellm_fetch(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        cost = pricing_mod.anthropic_cost("claude-3-haiku-20240307", 1_000_000, 0)
    assert isinstance(cost, float)
    assert cost > 0


def test_gemini_cost_returns_float(tmp_path):
    cache_file = tmp_path / "pricing.json"
    with _mock_litellm_fetch(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        cost = pricing_mod.gemini_cost("gemini-2.0-flash", 1_000_000, 0)
    assert isinstance(cost, float)
    assert cost > 0


def test_gemini_cost_returns_float_for_gemini_provider_label(tmp_path):
    """Newer Gemini models use litellm_provider='gemini' not 'google'."""
    cache_file = tmp_path / "pricing.json"
    with _mock_litellm_fetch(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        cost = pricing_mod.gemini_cost("gemini-2.5-flash-lite", 1_000_000, 0)
    assert isinstance(cost, float)
    assert cost > 0


def test_mistral_chat_cost_returns_float(tmp_path):
    cache_file = tmp_path / "pricing.json"
    with _mock_litellm_fetch(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        cost = pricing_mod.mistral_cost(
            "mistral-ocr-latest", "mistral-small-latest",
            ocr_pages=0, chat_input_tokens=1_000_000, chat_output_tokens=0,
        )
    assert isinstance(cost, float)
    assert cost > 0


def test_mistral_ocr_cost_returns_float(tmp_path):
    cache_file = tmp_path / "pricing.json"
    with _mock_litellm_fetch(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        cost = pricing_mod.mistral_cost(
            "mistral-ocr-latest", "mistral-small-latest",
            ocr_pages=10, chat_input_tokens=0, chat_output_tokens=0,
        )
    assert isinstance(cost, float)
    assert cost > 0


# ---------------------------------------------------------------------------
# Cost functions — missing pricing returns None
# ---------------------------------------------------------------------------


def test_anthropic_cost_returns_none_when_missing(tmp_path):
    cache_file = tmp_path / "pricing.json"
    with _mock_litellm_unavailable(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        cost = pricing_mod.anthropic_cost("claude-nonexistent-model", 100, 100)
    assert cost is None


def test_gemini_cost_returns_none_when_missing(tmp_path):
    cache_file = tmp_path / "pricing.json"
    with _mock_litellm_unavailable(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        cost = pricing_mod.gemini_cost("gemini-nonexistent-99", 100, 100)
    assert cost is None


def test_mistral_cost_returns_ocr_only_when_chat_rates_missing(tmp_path):
    """When chat rates are unavailable, return OCR cost (not None) with a warning."""
    cache_file = tmp_path / "pricing.json"
    with _mock_litellm_unavailable(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        cost = pricing_mod.mistral_cost(
            "mistral-ocr-latest", "mistral-nonexistent",
            ocr_pages=1, chat_input_tokens=100, chat_output_tokens=100,
        )
    assert cost is not None
    assert abs(cost - 0.001) < 1e-9  # only OCR cost (1 page × $0.001)


def test_mistral_cost_warns_when_chat_rates_missing(tmp_path, caplog):
    """A WARNING is logged when chat rates are unavailable."""
    cache_file = tmp_path / "pricing.json"
    with _mock_litellm_unavailable(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file), \
         caplog.at_level(logging.WARNING):
        pricing_mod.mistral_cost(
            "mistral-ocr-latest", "mistral-nonexistent",
            ocr_pages=1, chat_input_tokens=100, chat_output_tokens=100,
        )
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# ---------------------------------------------------------------------------
# Mistral OCR rate — bootstrapped to cache, not permanently hardcoded
# ---------------------------------------------------------------------------


def test_mistral_ocr_rate_bootstrapped_to_cache_on_first_use(tmp_path):
    """With no cache file, OCR rate is written to pricing.json on first call."""
    cache_file = tmp_path / "pricing.json"
    with _mock_litellm_unavailable(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        pricing_mod.mistral_cost(
            "mistral-ocr-latest", "mistral-small-latest",
            ocr_pages=1, chat_input_tokens=0, chat_output_tokens=0,
        )
    assert cache_file.exists()
    data = json.loads(cache_file.read_text())
    assert "mistral-ocr-latest" in data["models"]
    assert "ocr_per_page" in data["models"]["mistral-ocr-latest"]


def test_mistral_ocr_rate_read_from_cache(tmp_path):
    """Custom OCR rate in cache is used instead of the bootstrap default."""
    cache_file = tmp_path / "pricing.json"
    custom_rate = 0.002  # double the default
    cache_file.write_text(json.dumps({
        "last_updated": "2026-01-01T00:00:00",
        "models": {
            "mistral-ocr-latest": {"ocr_per_page": custom_rate},
            "mistral-small-latest": {"input": 0.10, "output": 0.30},
        },
    }))
    with _mock_litellm_unavailable(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        cost = pricing_mod.mistral_cost(
            "mistral-ocr-latest", "mistral-small-latest",
            ocr_pages=1, chat_input_tokens=0, chat_output_tokens=0,
        )
    assert abs(cost - custom_rate) < 1e-9


def test_mistral_ocr_bootstrap_value_is_correct(tmp_path):
    """Bootstrap default is $0.001 per page."""
    cache_file = tmp_path / "pricing.json"
    with _mock_litellm_unavailable(), \
         patch.object(pricing_mod, "_cache_path", return_value=cache_file):
        cost = pricing_mod.mistral_cost(
            "mistral-ocr-latest", "mistral-small-latest",
            ocr_pages=5, chat_input_tokens=0, chat_output_tokens=0,
        )
    assert abs(cost - 0.005) < 1e-9  # 5 × $0.001


# ---------------------------------------------------------------------------
# Usage namedtuple — cost_usd=None
# ---------------------------------------------------------------------------


def test_usage_cost_usd_can_be_none():
    u = Usage(100, 200, 0, 0, None)
    assert u.cost_usd is None


def test_usage_zero_has_none_cost():
    assert Usage.zero().cost_usd is None


def test_usage_add_both_none():
    a = Usage(100, 200, 0, 0, None)
    b = Usage(50, 100, 0, 0, None)
    assert (a + b).cost_usd is None


def test_usage_add_one_none_propagates():
    a = Usage(100, 200, 0, 0, 0.5)
    b = Usage(50, 100, 0, 0, None)
    assert (a + b).cost_usd is None


def test_usage_add_both_float():
    a = Usage(100, 200, 0, 0, 0.5)
    b = Usage(50, 100, 0, 0, 0.3)
    assert abs((a + b).cost_usd - 0.8) < 1e-9
