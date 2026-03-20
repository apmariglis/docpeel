"""
Tests for MistralExtractor.extract() — the OCR→structure pipeline.

Covers cost combination behaviour (the bug where ocr_cost + None raised
TypeError) and the 13-tuple return shape. No real API calls are made.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.docpeel.extraction import MistralExtractor
from src.docpeel.providers.base import Usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_RESULT = {
    "skip": False,
    "skip_reason": None,
    "text": "some text",
    "title": None,
    "page_number": None,
    "tables": [],
    "watermarks": [],
}


def _make_provider(ocr_cost: float, chat_cost: float | None):
    """Build a mock MistralProvider with controllable costs."""
    provider = MagicMock()
    provider.ocr_with_retry.return_value = ("ocr text", 1)
    provider.ocr_page_cost.return_value = ocr_cost
    provider.structure_with_retry.return_value = (
        _MINIMAL_RESULT,
        Usage(100, 50, 0, 0, chat_cost),
    )
    provider.drain_sanitisation_warnings.return_value = []
    return provider


def _fake_image():
    return Image.new("RGB", (10, 10))


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------


def test_extract_returns_13_tuple():
    provider = _make_provider(0.001, 0.002)
    extractor = MistralExtractor(provider)
    result = extractor.extract(_fake_image(), page_num=1)
    assert len(result) == 13


def test_extract_method_is_ocr_structure():
    provider = _make_provider(0.001, 0.002)
    extractor = MistralExtractor(provider)
    result = extractor.extract(_fake_image(), page_num=1)
    assert result[8] == "ocr+structure"


def test_extract_paraphrased_is_none():
    provider = _make_provider(0.001, 0.002)
    extractor = MistralExtractor(provider)
    result = extractor.extract(_fake_image(), page_num=1)
    assert result[10] is None  # paraphrased never applies to Mistral


# ---------------------------------------------------------------------------
# Cost combination — both known
# ---------------------------------------------------------------------------


def test_extract_total_cost_sums_ocr_and_chat():
    provider = _make_provider(ocr_cost=0.001, chat_cost=0.002)
    extractor = MistralExtractor(provider)
    result = extractor.extract(_fake_image(), page_num=1)
    total_usage = result[7]
    assert abs(total_usage.cost_usd - 0.003) < 1e-9


def test_extract_ocr_cost_in_extra_field():
    provider = _make_provider(ocr_cost=0.001, chat_cost=0.002)
    extractor = MistralExtractor(provider)
    result = extractor.extract(_fake_image(), page_num=1)
    assert abs(result[11] - 0.001) < 1e-9  # ocr_cost_usd


def test_extract_chat_cost_in_extra_field():
    provider = _make_provider(ocr_cost=0.001, chat_cost=0.002)
    extractor = MistralExtractor(provider)
    result = extractor.extract(_fake_image(), page_num=1)
    assert abs(result[12] - 0.002) < 1e-9  # structure_cost_usd


# ---------------------------------------------------------------------------
# Cost combination — chat cost None (pricing unavailable)
# ---------------------------------------------------------------------------


def test_extract_total_cost_none_when_chat_cost_unknown():
    """ocr_cost (float) + chat_cost (None) must not raise — result is None."""
    provider = _make_provider(ocr_cost=0.001, chat_cost=None)
    extractor = MistralExtractor(provider)
    result = extractor.extract(_fake_image(), page_num=1)
    total_usage = result[7]
    assert total_usage.cost_usd is None


def test_extract_ocr_extra_field_still_set_when_chat_cost_none():
    """OCR cost is still recorded even when chat pricing is unavailable."""
    provider = _make_provider(ocr_cost=0.001, chat_cost=None)
    extractor = MistralExtractor(provider)
    result = extractor.extract(_fake_image(), page_num=1)
    assert abs(result[11] - 0.001) < 1e-9  # ocr_cost_usd


def test_extract_chat_extra_field_none_when_chat_cost_none():
    provider = _make_provider(ocr_cost=0.001, chat_cost=None)
    extractor = MistralExtractor(provider)
    result = extractor.extract(_fake_image(), page_num=1)
    assert result[12] is None  # structure_cost_usd


# ---------------------------------------------------------------------------
# Token counts are always present regardless of cost availability
# ---------------------------------------------------------------------------


def test_extract_token_counts_present_when_cost_none():
    provider = _make_provider(ocr_cost=0.001, chat_cost=None)
    extractor = MistralExtractor(provider)
    result = extractor.extract(_fake_image(), page_num=1)
    total_usage = result[7]
    assert total_usage.input_tokens == 100
    assert total_usage.output_tokens == 50
