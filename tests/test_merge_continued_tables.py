"""
Tests for output._merge_continued_tables() — the post-processing pass that
joins table continuations across pages.

Continuation is detected programmatically: a table is merged into the previous
page's last table when it has no title AND its column count (from the | --- |
separator row) matches the previous page's last table's column count.

All tests use synthetic result dicts; no API calls or real PDFs required.
"""

from src.docpeel.output import _merge_continued_tables


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table(content: str, title: str | None = None) -> dict:
    return {"title": title, "caption": "cap", "content": content}


def _page(page_num: int, tables: list[dict]) -> dict:
    return {
        "page": page_num,
        "text": f"page {page_num} text",
        "tables": tables,
        "skip": False,
        "error": None,
    }


# Reusable content snippets with separator rows (required for column detection)
_1COL = "| H |\n| --- |\n| 1 |"
_2COL = "| H1 | H2 |\n| --- | --- |\n| 1 | 2 |"
_3COL = "| H1 | H2 | H3 |\n| --- | --- | --- |\n| 1 | 2 | 3 |"


# ---------------------------------------------------------------------------
# 1. No continuations — output unchanged
# ---------------------------------------------------------------------------


def test_no_continuations_unchanged():
    """Independent tables with different column headers are not merged."""
    results = [
        _page(1, [_table(_1COL, title="T1")]),
        _page(2, [_table(_2COL, title="T2")]),
    ]
    merged = _merge_continued_tables(results)
    assert merged[0]["tables"][0]["content"] == _1COL
    assert merged[1]["tables"][0]["content"] == _2COL


def test_no_continuations_page_count_unchanged():
    results = [_page(1, [_table(_1COL, title="T1")]), _page(2, [_table(_2COL, title="T2")])]
    merged = _merge_continued_tables(results)
    assert len(merged) == 2
    assert len(merged[0]["tables"]) == 1
    assert len(merged[1]["tables"]) == 1


def test_different_column_count_not_merged():
    """No title but different column count → not a continuation."""
    results = [
        _page(1, [_table(_2COL, title="T1")]),
        _page(2, [_table(_3COL)]),  # no title, but 3 cols ≠ 2 cols
    ]
    merged = _merge_continued_tables(results)
    assert len(merged[0]["tables"]) == 1
    assert len(merged[1]["tables"]) == 1


# ---------------------------------------------------------------------------
# 2. Simple two-page merge
# ---------------------------------------------------------------------------


def test_two_page_merge_content_appended():
    results = [
        _page(4, [_table("| H1 | H2 |\n| --- | --- |\n| 1 | a |\n| 2 | b |", title="T")]),
        _page(5, [_table("| 3 | c |\n| --- | --- |\n| 4 | d |")]),  # no title, 2 cols
    ]
    merged = _merge_continued_tables(results)
    combined = merged[0]["tables"][0]["content"]
    assert "| 1 | a |" in combined
    assert "| 3 | c |" in combined
    assert combined.index("| 1 | a |") < combined.index("| 3 | c |")


def test_two_page_merge_continuation_removed_from_page():
    results = [
        _page(4, [_table(_1COL, title="T")]),
        _page(5, [_table("| 2 |\n| --- |\n| 3 |")]),
    ]
    merged = _merge_continued_tables(results)
    assert merged[1]["tables"] == []


def test_two_page_merge_page_count_unchanged():
    results = [
        _page(4, [_table(_1COL, title="T")]),
        _page(5, [_table("| 2 |\n| --- |\n| 3 |")]),
    ]
    merged = _merge_continued_tables(results)
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# 3. Three-page chain
# ---------------------------------------------------------------------------


def test_three_page_chain_all_merged_into_first():
    results = [
        _page(4, [_table("| H |\n| --- |\n| r1 |", title="T")]),
        _page(5, [_table("| r2 |\n| --- |\n| r3 |")]),
        _page(6, [_table("| r4 |\n| --- |\n| r5 |")]),
    ]
    merged = _merge_continued_tables(results)
    combined = merged[0]["tables"][0]["content"]
    for row in ["| r1 |", "| r2 |", "| r3 |", "| r4 |", "| r5 |"]:
        assert row in combined


def test_three_page_chain_continuations_removed():
    results = [
        _page(4, [_table(_1COL, title="T")]),
        _page(5, [_table("| r2 |\n| --- |")]),
        _page(6, [_table("| r3 |\n| --- |")]),
    ]
    merged = _merge_continued_tables(results)
    assert merged[1]["tables"] == []
    assert merged[2]["tables"] == []


def test_three_page_chain_order_preserved():
    # Realistic content: page 1 has a header row + separator (middle, not stripped),
    # continuation pages have a false | --- | on line 1 (stripped on merge).
    results = [
        _page(1, [_table("| H |\n| --- |\nrow1", title="T")]),
        _page(2, [_table("row2\n| --- |\nrow2b")]),
        _page(3, [_table("row3\n| --- |\nrow3b")]),
    ]
    merged = _merge_continued_tables(results)
    combined = merged[0]["tables"][0]["content"]
    assert combined.index("row1") < combined.index("row2") < combined.index("row3")


# ---------------------------------------------------------------------------
# 4. Continuation on first page — ignored (no previous page)
# ---------------------------------------------------------------------------


def test_continuation_on_first_page_ignored():
    """First page: no previous to merge into, even if structural signals match."""
    results = [
        _page(1, [_table("| x |\n| --- |\n| 1 |")]),  # no title, has sep → would merge if prior existed
        _page(2, [_table(_1COL, title="T2")]),
    ]
    merged = _merge_continued_tables(results)
    assert len(merged[0]["tables"]) == 1
    assert merged[0]["tables"][0]["content"] == "| x |\n| --- |\n| 1 |"


# ---------------------------------------------------------------------------
# 5. Previous page has no tables — continuation ignored
# ---------------------------------------------------------------------------


def test_continuation_when_previous_page_has_no_tables():
    results = [
        _page(3, []),  # no tables
        _page(4, [_table("| x |\n| --- |\n| 1 |")]),  # no title, 1 col
    ]
    merged = _merge_continued_tables(results)
    # Nothing to merge into — stays on page 4
    assert len(merged[1]["tables"]) == 1
    assert merged[1]["tables"][0]["content"] == "| x |\n| --- |\n| 1 |"


# ---------------------------------------------------------------------------
# 6. Multiple tables on page, only first continues
# ---------------------------------------------------------------------------


def test_only_first_table_merged_second_stays():
    results = [
        _page(3, [_table(_1COL, title="T1")]),
        _page(4, [
            _table("| 2 |\n| --- |\n| 3 |"),           # no title, 1 col → continuation
            _table(_2COL, title="T3"),                  # titled fresh table
        ]),
    ]
    merged = _merge_continued_tables(results)
    assert "| 2 |" in merged[0]["tables"][0]["content"]
    assert len(merged[1]["tables"]) == 1
    assert merged[1]["tables"][0]["title"] == "T3"


# ---------------------------------------------------------------------------
# 7. Non-first table on a page — never checked for continuation
# ---------------------------------------------------------------------------


def test_non_first_table_not_checked():
    """Only the first table on a page is a continuation candidate."""
    results = [
        _page(3, [_table(_1COL, title="T1")]),
        _page(4, [
            _table(_2COL, title="T2"),          # different headers — no merge
            _table("| x |\n| --- |"),           # no title, 1 col — but not first
        ]),
    ]
    merged = _merge_continued_tables(results)
    assert merged[0]["tables"][0]["content"] == _1COL
    assert len(merged[1]["tables"]) == 2


# ---------------------------------------------------------------------------
# 8. Content order — rows appended, not prepended
# ---------------------------------------------------------------------------


def test_merged_rows_order():
    results = [
        _page(1, [_table("header\n| --- |\nrow_a\nrow_b", title="T")]),
        _page(2, [_table("row_c\n| --- |\nrow_d")]),
    ]
    merged = _merge_continued_tables(results)
    content = merged[0]["tables"][0]["content"]
    assert content.index("row_a") < content.index("row_c")
    assert content.index("row_b") < content.index("row_d")


def test_empty_pages_list():
    assert _merge_continued_tables([]) == []


def test_single_page_no_tables():
    results = [_page(1, [])]
    merged = _merge_continued_tables(results)
    assert merged == results


# ---------------------------------------------------------------------------
# 9. Table without separator row — not detected as continuation
# ---------------------------------------------------------------------------


def test_captions_concatenated_on_merge():
    """After merge the caption covers both fragments."""
    results = [
        _page(1, [{"title": "T", "caption": "First half.", "content": _1COL}]),
        _page(2, [{"title": None, "caption": "Second half.", "content": "| x |\n| --- |\n| y |"}]),
    ]
    merged = _merge_continued_tables(results)
    assert merged[0]["tables"][0]["caption"] == "First half. Second half."


def test_trailing_separator_stripped_at_join():
    """
    When the previous fragment ends with a | --- | row (Mistral OCR quirk),
    it is stripped before appending so no spurious separator appears at the join.
    """
    results = [
        _page(1, [_table("| H |\n| --- |\n| r1 |\n| --- |", title="T")]),
        _page(2, [_table("| r2 |\n| --- |\n| r3 |")]),
    ]
    merged = _merge_continued_tables(results)
    content = merged[0]["tables"][0]["content"]
    lines = content.splitlines()
    r1_idx = next(i for i, l in enumerate(lines) if "r1" in l)
    assert "r2" in lines[r1_idx + 1], (
        f"Expected | r2 | after | r1 |, got: {lines[r1_idx + 1]!r}"
    )


def test_leading_false_header_separator_stripped():
    """
    When the OCR processes a continuation page in isolation it treats the first
    data row as a column header and inserts | --- | after it. That separator
    must be removed in the merged table — it is not a real header separator.
    """
    results = [
        _page(1, [_table("| H |\n| --- |\n| r1 |", title="T")]),
        # Continuation: first row followed by a false | --- | separator
        _page(2, [_table("| r2 |\n| --- |\n| r3 |\n| --- |\n| r4 |")]),
    ]
    merged = _merge_continued_tables(results)
    content = merged[0]["tables"][0]["content"]
    lines = content.splitlines()
    r2_idx = next(i for i, l in enumerate(lines) if "r2" in l)
    # The line after | r2 | must be | r3 |, not | --- |
    assert "r3" in lines[r2_idx + 1], (
        f"Expected | r3 | after | r2 |, got: {lines[r2_idx + 1]!r}"
    )


def test_no_separator_row_not_merged():
    """A table with no | --- | row has col count 0 and is never merged."""
    results = [
        _page(1, [_table("| H |\n| 1 |", title="T")]),
        _page(2, [_table("| 2 |\n| 3 |")]),  # no title, but no separator row → col count 0
    ]
    merged = _merge_continued_tables(results)
    assert len(merged[0]["tables"]) == 1
    assert len(merged[1]["tables"]) == 1


# ---------------------------------------------------------------------------
# 10. Repeated-header continuation (identical header row on consecutive pages)
# ---------------------------------------------------------------------------

_3COL_HDR = "| A | B | C |"
_3COL_SEP = "| --- | --- | --- |"


def _rh_table(rows: str, title: str | None = None) -> dict:
    """Table with a repeated header format: header + sep + data rows."""
    content = f"{_3COL_HDR}\n{_3COL_SEP}\n{rows}"
    return {"title": title, "caption": "cap", "content": content}


def test_repeated_header_tables_merged():
    """Tables with identical header rows on consecutive pages are merged."""
    results = [
        _page(1, [_rh_table("| 1 | 2 | 3 |", title="T")]),
        _page(2, [_rh_table("| 4 | 5 | 6 |", title="T CONTINUED")]),
    ]
    merged = _merge_continued_tables(results)
    content = merged[0]["tables"][0]["content"]
    assert "| 1 | 2 | 3 |" in content
    assert "| 4 | 5 | 6 |" in content


def test_repeated_header_continuation_removed_from_page():
    results = [
        _page(1, [_rh_table("| r1 |", title="T")]),
        _page(2, [_rh_table("| r2 |", title="T CONTINUED")]),
    ]
    merged = _merge_continued_tables(results)
    assert merged[1]["tables"] == []


def test_repeated_header_strips_duplicate_header_from_continuation():
    """The repeated header row and separator must not appear twice in the merged content."""
    results = [
        _page(1, [_rh_table("| r1 |", title="T")]),
        _page(2, [_rh_table("| r2 |", title="T CONTINUED")]),
    ]
    merged = _merge_continued_tables(results)
    content = merged[0]["tables"][0]["content"]
    # Header row should appear exactly once
    assert content.count(_3COL_HDR) == 1
    # Separator row should appear exactly once
    assert content.count(_3COL_SEP) == 1


def test_repeated_header_data_order_preserved():
    results = [
        _page(1, [_rh_table("| r1 |\n| r2 |", title="T")]),
        _page(2, [_rh_table("| r3 |\n| r4 |", title="T CONTINUED")]),
    ]
    merged = _merge_continued_tables(results)
    content = merged[0]["tables"][0]["content"]
    assert content.index("r1") < content.index("r2") < content.index("r3") < content.index("r4")


def test_repeated_header_different_headers_not_merged():
    """Tables with different header rows are NOT merged even if both have titles."""
    results = [
        _page(1, [_table("| X | Y |\n| --- | --- |\n| 1 | 2 |", title="T1")]),
        _page(2, [_table("| A | B |\n| --- | --- |\n| 3 | 4 |", title="T2 CONTINUED")]),
    ]
    merged = _merge_continued_tables(results)
    assert len(merged[0]["tables"]) == 1
    assert len(merged[1]["tables"]) == 1


def test_repeated_header_captions_concatenated():
    results = [
        _page(1, [{"title": "T", "caption": "First part.", "content": f"{_3COL_HDR}\n{_3COL_SEP}\n| r1 |"}]),
        _page(2, [{"title": "T CONTINUED", "caption": "Second part.", "content": f"{_3COL_HDR}\n{_3COL_SEP}\n| r2 |"}]),
    ]
    merged = _merge_continued_tables(results)
    assert merged[0]["tables"][0]["caption"] == "First part. Second part."


def test_repeated_header_three_page_chain():
    """Three-page chain with repeated headers all merges into the first page."""
    results = [
        _page(1, [_rh_table("| r1 |", title="T")]),
        _page(2, [_rh_table("| r2 |", title="T CONTINUED")]),
        _page(3, [_rh_table("| r3 |", title="T CONTINUED")]),
    ]
    merged = _merge_continued_tables(results)
    content = merged[0]["tables"][0]["content"]
    assert "| r1 |" in content
    assert "| r2 |" in content
    assert "| r3 |" in content
    assert merged[1]["tables"] == []
    assert merged[2]["tables"] == []


def test_repeated_header_no_title_still_merges_by_column_count():
    """No-title detection still works even when header rows happen to match."""
    results = [
        _page(1, [_rh_table("| r1 |", title="T")]),
        _page(2, [_rh_table("| r2 |")]),  # no title → no-title path
    ]
    merged = _merge_continued_tables(results)
    assert merged[1]["tables"] == []
    assert "| r2 |" in merged[0]["tables"][0]["content"]
