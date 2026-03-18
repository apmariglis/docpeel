"""
Tests for extraction._unpack() — the function that sanitises the raw
structured dict returned by a provider call into typed fields.
No API calls or real PDFs required.
"""

import pytest

from src.docpeel.extraction import _unpack


# ---------------------------------------------------------------------------
# skip / skip_reason
# ---------------------------------------------------------------------------


def test_unpack_skip_false_by_default():
    skip, skip_reason, *_ = _unpack({})
    assert skip is False
    assert skip_reason is None


def test_unpack_skip_true():
    skip, skip_reason, *_ = _unpack({"skip": True, "skip_reason": "blank"})
    assert skip is True
    assert skip_reason == "blank"


def test_unpack_skip_true_no_reason_becomes_unknown():
    skip, skip_reason, *_ = _unpack({"skip": True})
    assert skip is True
    assert skip_reason == "unknown"


def test_unpack_skip_true_non_string_reason_becomes_unknown():
    skip, skip_reason, *_ = _unpack({"skip": True, "skip_reason": 42})
    assert skip_reason == "unknown"


def test_unpack_skip_false_reason_cleared():
    """skip_reason must be None when skip is False."""
    skip, skip_reason, *_ = _unpack({"skip": False, "skip_reason": "leftover"})
    assert skip is False
    assert skip_reason is None


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------


def test_unpack_text_present():
    _, _, text, *_ = _unpack({"text": "hello world"})
    assert text == "hello world"


def test_unpack_text_missing_defaults_to_empty():
    _, _, text, *_ = _unpack({})
    assert text == ""


def test_unpack_text_none_defaults_to_empty():
    _, _, text, *_ = _unpack({"text": None})
    assert text == ""


# ---------------------------------------------------------------------------
# title
# ---------------------------------------------------------------------------


def test_unpack_title_string():
    _, _, _, title, *_ = _unpack({"title": "Chapter 1"})
    assert title == "Chapter 1"


def test_unpack_title_none():
    _, _, _, title, *_ = _unpack({"title": None})
    assert title is None


def test_unpack_title_missing_is_none():
    _, _, _, title, *_ = _unpack({})
    assert title is None


def test_unpack_title_non_string_becomes_none():
    _, _, _, title, *_ = _unpack({"title": 123})
    assert title is None


# ---------------------------------------------------------------------------
# page_number
# ---------------------------------------------------------------------------


def test_unpack_page_number_int():
    _, _, _, _, page_number, *_ = _unpack({"page_number": 42})
    assert page_number == 42


def test_unpack_page_number_none():
    _, _, _, _, page_number, *_ = _unpack({"page_number": None})
    assert page_number is None


def test_unpack_page_number_missing_is_none():
    _, _, _, _, page_number, *_ = _unpack({})
    assert page_number is None


def test_unpack_page_number_string_becomes_none():
    """String page numbers must be rejected — book_page must always be int|None."""
    _, _, _, _, page_number, *_ = _unpack({"page_number": "42"})
    assert page_number is None


def test_unpack_page_number_float_becomes_none():
    _, _, _, _, page_number, *_ = _unpack({"page_number": 42.0})
    assert page_number is None


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def test_unpack_tables_empty():
    _, _, _, _, _, tables, _ = _unpack({"tables": []})
    assert tables == []


def test_unpack_tables_missing_is_empty():
    _, _, _, _, _, tables, _ = _unpack({})
    assert tables == []


def test_unpack_tables_non_list_is_empty():
    _, _, _, _, _, tables, _ = _unpack({"tables": "not a list"})
    assert tables == []


def test_unpack_tables_basic():
    raw = [{"title": "T1", "caption": "Cap", "content": "| A | B |\n|---|---|\n| 1 | 2 |"}]
    _, _, _, _, _, tables, _ = _unpack({"tables": raw})
    assert len(tables) == 1
    assert tables[0]["title"] == "T1"
    assert tables[0]["caption"] == "Cap"
    assert "| A |" in tables[0]["content"]


def test_unpack_tables_non_dict_entries_skipped():
    raw = [{"title": "ok", "caption": "c", "content": "x"}, "not a dict", None]
    _, _, _, _, _, tables, _ = _unpack({"tables": raw})
    assert len(tables) == 1


def test_unpack_tables_title_none_if_not_string():
    raw = [{"title": 99, "caption": "c", "content": "x"}]
    _, _, _, _, _, tables, _ = _unpack({"tables": raw})
    assert tables[0]["title"] is None


def test_unpack_tables_missing_caption_defaults_empty():
    raw = [{"content": "| a |"}]
    _, _, _, _, _, tables, _ = _unpack({"tables": raw})
    assert tables[0]["caption"] == ""


def test_unpack_tables_missing_content_defaults_empty():
    raw = [{"caption": "c"}]
    _, _, _, _, _, tables, _ = _unpack({"tables": raw})
    assert tables[0]["content"] == ""


# ---------------------------------------------------------------------------
# watermarks
# ---------------------------------------------------------------------------


def test_unpack_watermarks_list():
    _, _, _, _, _, _, watermarks = _unpack({"watermarks": ["CONFIDENTIAL"]})
    assert watermarks == ["CONFIDENTIAL"]


def test_unpack_watermarks_empty():
    _, _, _, _, _, _, watermarks = _unpack({"watermarks": []})
    assert watermarks == []


def test_unpack_watermarks_missing_is_empty():
    _, _, _, _, _, _, watermarks = _unpack({})
    assert watermarks == []


def test_unpack_watermarks_non_list_is_empty():
    _, _, _, _, _, _, watermarks = _unpack({"watermarks": "STAMP"})
    assert watermarks == []


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------


def test_unpack_returns_7_tuple():
    result = _unpack({"skip": False, "text": "x", "page_number": 1})
    assert len(result) == 7
