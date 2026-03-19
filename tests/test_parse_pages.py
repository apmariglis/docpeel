"""
Tests for cli._parse_pages() — the page specification parser.
"""

import pytest

from src.docpeel.cli import _parse_pages


# ---------------------------------------------------------------------------
# Single pages
# ---------------------------------------------------------------------------


def test_single_page():
    assert _parse_pages("5") == {5}


def test_single_page_1():
    assert _parse_pages("1") == {1}


def test_comma_separated_pages():
    assert _parse_pages("1,3,5") == {1, 3, 5}


def test_leading_trailing_whitespace_ignored():
    assert _parse_pages(" 3 , 7 ") == {3, 7}


def test_duplicate_pages_deduplicated():
    assert _parse_pages("3,3,3") == {3}


# ---------------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------------


def test_simple_range():
    assert _parse_pages("2-5") == {2, 3, 4, 5}


def test_single_element_range():
    assert _parse_pages("4-4") == {4}


def test_range_from_one():
    assert _parse_pages("1-3") == {1, 2, 3}


# ---------------------------------------------------------------------------
# Mixed
# ---------------------------------------------------------------------------


def test_mixed_pages_and_ranges():
    assert _parse_pages("1,3,7-10") == {1, 3, 7, 8, 9, 10}


def test_overlapping_range_and_page():
    assert _parse_pages("3-6,5") == {3, 4, 5, 6}


def test_multiple_ranges():
    assert _parse_pages("1-3,8-10") == {1, 2, 3, 8, 9, 10}


# ---------------------------------------------------------------------------
# Invalid input — ValueError expected
# ---------------------------------------------------------------------------


def test_invalid_non_numeric():
    with pytest.raises(ValueError, match="invalid page number"):
        _parse_pages("abc")


def test_invalid_range_non_numeric():
    with pytest.raises(ValueError, match="invalid page range"):
        _parse_pages("a-b")


def test_invalid_range_missing_end():
    with pytest.raises(ValueError, match="invalid page range"):
        _parse_pages("3-")


def test_invalid_range_missing_start():
    with pytest.raises(ValueError, match="invalid page range"):
        _parse_pages("-5")


def test_page_zero_rejected():
    with pytest.raises(ValueError, match="≥ 1"):
        _parse_pages("0")


def test_negative_page_rejected():
    with pytest.raises(ValueError):
        _parse_pages("-1")


def test_range_with_zero_rejected():
    with pytest.raises(ValueError, match="≥ 1"):
        _parse_pages("0-5")


def test_inverted_range_rejected():
    with pytest.raises(ValueError, match="start must be ≤ end"):
        _parse_pages("10-5")


def test_float_rejected():
    with pytest.raises(ValueError, match="invalid page number"):
        _parse_pages("3.5")
