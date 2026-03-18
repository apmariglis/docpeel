"""
Tests for VisionExtractor's fallback chain using a mocked VisionProvider.

Stages covered:
  1. Stage 1 success (full-page verbatim)
  2. Stage 1 filter → Stage 2 quadrant split → Stage 3 stitch
  3. All quadrants empty → image-only page (stitch skipped)
  4. Some quadrants blocked → obfuscated retry → paraphrase fallback
  5. Stitch blocked → full-page paraphrase fallback
  6. Non-filter error in Stage 1 → re-raised immediately
"""

from unittest.mock import MagicMock, call, patch

import pytest
from PIL import Image

from src.docpeel.extraction import VisionExtractor
from src.docpeel.providers.base import Usage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image(w=100, h=100):
    return Image.new("RGB", (w, h), (128, 128, 128))


class FilterError(Exception):
    """Sentinel exception used as content-filter error in tests."""


def _make_provider(*, filter_error_cls=FilterError):
    """
    Build a MagicMock VisionProvider.
    is_content_filter_error returns True only for FilterError instances.
    """
    provider = MagicMock()
    provider.is_content_filter_error.side_effect = lambda exc: isinstance(
        exc, filter_error_cls
    )
    return provider


def _zero_usage():
    return Usage(0, 0, 0, 0, 0.0)


def _structured(text="body text", skip=False, tables=None):
    """Minimal structured dict as a provider would return."""
    return {
        "skip": skip,
        "skip_reason": None,
        "text": text,
        "title": None,
        "page_number": None,
        "tables": tables or [],
        "watermarks": [],
    }


# ---------------------------------------------------------------------------
# Stage 1: success
# ---------------------------------------------------------------------------


def test_stage1_success_returns_full_page_method():
    provider = _make_provider()
    provider.call_structured.return_value = (_structured("hello"), _zero_usage())

    extractor = VisionExtractor(provider)
    result = extractor.extract(_image(), page_num=1)

    skip, skip_reason, text, title, book_page, tables, watermarks, usage, method, warnings, paraphrased = result
    assert method == "full-page"
    assert text == "hello"
    assert skip is False
    assert paraphrased is None
    assert warnings == []
    assert provider.call_structured.call_count == 1


def test_stage1_non_filter_error_raises_immediately():
    provider = _make_provider()
    provider.call_structured.side_effect = RuntimeError("network failure")

    extractor = VisionExtractor(provider)
    with pytest.raises(RuntimeError, match="network failure"):
        extractor.extract(_image(), page_num=1)

    # No quadrant calls should have been made
    provider.call.assert_not_called()


# ---------------------------------------------------------------------------
# Stage 1 → Stage 2 → Stage 3: quadrant split + stitch
# ---------------------------------------------------------------------------


def test_quadrant_split_triggered_on_filter_error():
    provider = _make_provider()
    provider.call_structured.side_effect = FilterError("blocked")  # Stage 1 blocked
    provider.call_with_image_and_text_structured.return_value = (
        _structured("stitched"), _zero_usage()
    )
    provider.call.return_value = ("chunk text", _zero_usage())

    extractor = VisionExtractor(provider)
    result = extractor.extract(_image(), page_num=1)

    _, _, text, _, _, _, _, _, method, _, _ = result
    assert method == "quadrant-split"
    assert provider.call.call_count == 4  # one per quadrant


def test_stitch_receives_all_four_quadrant_labels():
    """The stitch prompt should contain all four quadrant labels."""
    provider = _make_provider()
    provider.call_structured.side_effect = FilterError("blocked")
    provider.call_with_image_and_text_structured.return_value = (
        _structured("stitched"), _zero_usage()
    )
    provider.call.return_value = ("chunk", _zero_usage())

    extractor = VisionExtractor(provider)
    extractor.extract(_image(), page_num=1)

    stitch_call_args = provider.call_with_image_and_text_structured.call_args
    text_prompt = stitch_call_args[0][1]  # second positional arg
    for label in ["top-left", "top-right", "bottom-left", "bottom-right"]:
        assert label in text_prompt


def test_stitch_success_paraphrased_is_none():
    provider = _make_provider()
    provider.call_structured.side_effect = FilterError()
    provider.call_with_image_and_text_structured.return_value = (
        _structured("stitched"), _zero_usage()
    )
    provider.call.return_value = ("text", _zero_usage())

    extractor = VisionExtractor(provider)
    result = extractor.extract(_image(), page_num=1)
    paraphrased = result[10]
    assert paraphrased is None


# ---------------------------------------------------------------------------
# Image-only page: all quadrants empty → no stitch
# ---------------------------------------------------------------------------


def test_image_only_page_skips_stitch():
    provider = _make_provider()
    provider.call_structured.side_effect = FilterError()
    provider.call.return_value = ("", _zero_usage())  # all quadrants empty

    extractor = VisionExtractor(provider)
    result = extractor.extract(_image(), page_num=1)

    _, _, text, _, _, _, _, _, method, _, _ = result
    assert text == ""
    assert method == "quadrant-split"
    provider.call_with_image_and_text_structured.assert_not_called()


def test_image_only_page_whitespace_only_quadrants():
    """Whitespace-only quadrant text counts as empty."""
    provider = _make_provider()
    provider.call_structured.side_effect = FilterError()
    provider.call.return_value = ("   \n\t  ", _zero_usage())

    extractor = VisionExtractor(provider)
    result = extractor.extract(_image(), page_num=1)

    provider.call_with_image_and_text_structured.assert_not_called()


# ---------------------------------------------------------------------------
# Quadrant obfuscated retry
# ---------------------------------------------------------------------------


def test_obfuscated_retry_on_quadrant_filter():
    provider = _make_provider()
    provider.call_structured.side_effect = FilterError()
    provider.call_with_image_and_text_structured.return_value = (
        _structured("stitched"), _zero_usage()
    )
    # First call per quadrant: blocked; second (obfuscated): succeeds
    provider.call.side_effect = [
        FilterError(),  # top-left verbatim blocked
        ("obf text", _zero_usage()),  # top-left obfuscated ok
        ("text", _zero_usage()),  # top-right verbatim
        ("text", _zero_usage()),  # bottom-left verbatim
        ("text", _zero_usage()),  # bottom-right verbatim
    ]

    extractor = VisionExtractor(provider)
    result = extractor.extract(_image(), page_num=1)

    assert provider.call.call_count == 5
    # No paraphrased quads — obfuscated retry succeeded
    paraphrased = result[10]
    assert paraphrased is None


# ---------------------------------------------------------------------------
# Quadrant paraphrase fallback
# ---------------------------------------------------------------------------


def test_quad_paraphrase_sets_partial():
    """If a quadrant needs paraphrase, paraphrased should be 'partial'."""
    provider = _make_provider()
    provider.call_structured.side_effect = FilterError()
    provider.call_with_image_and_text_structured.return_value = (
        _structured("stitched"), _zero_usage()
    )
    # top-left: verbatim blocked, obfuscated blocked → paraphrase
    # remaining 3 quads: verbatim succeeds
    provider.call.side_effect = [
        FilterError(),           # top-left verbatim blocked
        FilterError(),           # top-left obfuscated blocked
        ("paraphrased tl", _zero_usage()),  # top-left paraphrase (plain call)
        ("text", _zero_usage()),  # top-right
        ("text", _zero_usage()),  # bottom-left
        ("text", _zero_usage()),  # bottom-right
    ]

    extractor = VisionExtractor(provider)
    result = extractor.extract(_image(), page_num=1)

    paraphrased = result[10]
    assert paraphrased == "partial"


def test_quad_all_blocked_chunk_skipped():
    """Quadrant blocked even on paraphrase → warning added, empty chunk."""
    provider = _make_provider()
    provider.call_structured.side_effect = FilterError()
    provider.call_with_image_and_text_structured.return_value = (
        _structured("stitched"), _zero_usage()
    )
    # top-left: all three attempts blocked
    provider.call.side_effect = [
        FilterError(),  # top-left verbatim
        FilterError(),  # top-left obfuscated
        FilterError(),  # top-left paraphrase
        ("text", _zero_usage()),  # top-right
        ("text", _zero_usage()),  # bottom-left
        ("text", _zero_usage()),  # bottom-right
    ]

    extractor = VisionExtractor(provider)
    result = extractor.extract(_image(), page_num=1)

    warnings = result[9]
    assert any("skipped" in w for w in warnings)


# ---------------------------------------------------------------------------
# Stitch blocked → full-page paraphrase
# ---------------------------------------------------------------------------


def test_stitch_blocked_triggers_full_page_paraphrase():
    provider = _make_provider()
    # Stage 1 blocked, then stitch blocked, then full-page paraphrase via call_structured
    provider.call_structured.side_effect = [
        FilterError(),           # Stage 1 blocked
        (_structured("paraphrased full page"), _zero_usage()),  # Stage 4 paraphrase
    ]
    provider.call_with_image_and_text_structured.side_effect = FilterError()  # stitch blocked
    provider.call.return_value = ("quad text", _zero_usage())

    extractor = VisionExtractor(provider)
    result = extractor.extract(_image(), page_num=1)

    paraphrased = result[10]
    assert paraphrased == "full"


def test_stitch_blocked_warning_added():
    provider = _make_provider()
    provider.call_structured.side_effect = [
        FilterError(),
        (_structured("paraphrased"), _zero_usage()),
    ]
    provider.call_with_image_and_text_structured.side_effect = FilterError()
    provider.call.return_value = ("quad text", _zero_usage())

    extractor = VisionExtractor(provider)
    result = extractor.extract(_image(), page_num=1)

    warnings = result[9]
    assert any("stitch" in w.lower() for w in warnings)


def test_stitch_non_filter_error_raises():
    provider = _make_provider()
    provider.call_structured.side_effect = FilterError()
    provider.call.return_value = ("quad text", _zero_usage())
    provider.call_with_image_and_text_structured.side_effect = RuntimeError("stitch failed")

    extractor = VisionExtractor(provider)
    with pytest.raises(RuntimeError, match="stitch failed"):
        extractor.extract(_image(), page_num=1)


# ---------------------------------------------------------------------------
# Usage accumulation
# ---------------------------------------------------------------------------


def test_usage_accumulated_across_quadrants():
    provider = _make_provider()
    provider.call_structured.side_effect = FilterError()
    provider.call_with_image_and_text_structured.return_value = (
        _structured("stitched"), Usage(5, 10, 0, 0, 0.001)
    )
    provider.call.return_value = ("text", Usage(1, 2, 0, 0, 0.0001))

    extractor = VisionExtractor(provider)
    result = extractor.extract(_image(), page_num=1)

    usage = result[7]
    # 4 quad calls × (1, 2) + stitch (5, 10)
    assert usage.input_tokens == 4 * 1 + 5
    assert usage.output_tokens == 4 * 2 + 10
