"""
Tests verifying that markdown output files are strictly derived from the JSONL,
which is the single source of truth.

Invariants tested:
  1. Every page in the JSONL has a corresponding per-page markdown file.
  2. The text in each per-page markdown (after stripping frontmatter) matches
     the text field in the corresponding JSONL record exactly.
  3. The YAML frontmatter fields (pdf_page, book_page, source) match the
     corresponding fields in the JSONL record.
  4. The combined markdown contains the text of every JSONL record, in order.
  5. No per-page markdown file exists without a corresponding JSONL record.
  6. The JSONL contains all required fields on every record.
  7. book_page in the JSONL is always an integer or null — never a string.
"""

import json
import re
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers to invoke stream_outputs with synthetic page data (no LLM calls)
# ---------------------------------------------------------------------------


def _make_page(
    page: int,
    text: str,
    book_page: int | None = None,
    method: str = "full-page",
    error: str | None = None,
) -> dict:
    """Build a minimal result dict matching what iter_pages yields."""
    return {
        "page": page,
        "model": "test-model",
        "text": text,
        "book_page": book_page,
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "cost_usd": 0.0001,
        "elapsed_seconds": 1.0,
        "extraction_method": method,
        "chunk_warnings": [],
        "paraphrased": None,
        "error": error,
    }


def _run_stream(pages: list[dict], tmp_path: Path) -> tuple[dict, list[dict]]:
    """
    Call stream_outputs with a fake PDF path pointing at tmp_path.
    Monkey-patches OUTPUT_FOLDER so all output lands in tmp_path.
    """
    import src.docpeel.output as out_mod

    original_output_folder = out_mod.OUTPUT_FOLDER
    out_mod.OUTPUT_FOLDER = tmp_path

    try:
        fake_pdf = tmp_path / "test_book.pdf"
        fake_pdf.touch()
        saved, results = out_mod.stream_outputs(fake_pdf, iter(pages))
    finally:
        out_mod.OUTPUT_FOLDER = original_output_folder

    return saved, results


def _parse_frontmatter(md_text: str) -> tuple[dict, str]:
    """
    Split YAML frontmatter from markdown body.
    Returns (frontmatter_dict, body_text).
    """
    if not md_text.startswith("---\n"):
        return {}, md_text
    end = md_text.index("\n---\n", 4)
    fm_block = md_text[4:end]
    body = md_text[end + 5 :]  # skip \n---\n
    fm = {}
    for line in fm_block.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            val = val.strip()
            # Parse null / integers
            if val == "null":
                fm[key.strip()] = None
            elif val.isdigit():
                fm[key.strip()] = int(val)
            else:
                fm[key.strip()] = val
    return fm, body.lstrip("\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_pages():
    return [
        _make_page(1, "First page content.", book_page=1),
        _make_page(2, "Second page content.\n\nWith multiple paragraphs.", book_page=2),
        _make_page(3, "# Heading\n\nThird page.", book_page=None),
        _make_page(
            4,
            "| Col A | Col B |\n|---|---|\n| 1 | 2 |",
            book_page=42,
            method="quadrant-split",
        ),
        _make_page(
            5,
            "[Page 5 could not be extracted: timeout]",
            book_page=None,
            error="timeout",
        ),
    ]


@pytest.fixture()
def run_output(sample_pages, tmp_path):
    saved, results = _run_stream(sample_pages, tmp_path)
    records = _read_jsonl(saved["json"])
    return saved, results, records, sample_pages


# ---------------------------------------------------------------------------
# 1. Every JSONL record has a per-page markdown file
# ---------------------------------------------------------------------------


def test_every_jsonl_record_has_page_md(run_output):
    saved, results, records, pages = run_output
    pages_dir = saved["pages_dir"]
    for r in records:
        expected = pages_dir / f"page_{r['page']:03d}.md"
        assert (
            expected.exists()
        ), f"Missing per-page markdown for JSONL record page={r['page']}"


# ---------------------------------------------------------------------------
# 2. Per-page markdown text matches JSONL text exactly
# ---------------------------------------------------------------------------


def test_page_md_text_matches_jsonl(run_output):
    saved, results, records, pages = run_output
    pages_dir = saved["pages_dir"]
    for r in records:
        md_path = pages_dir / f"page_{r['page']:03d}.md"
        _, body = _parse_frontmatter(md_path.read_text(encoding="utf-8"))
        assert body == r["text"], (
            f"Page {r['page']}: markdown body does not match JSONL text.\n"
            f"  JSONL : {r['text']!r}\n"
            f"  MD    : {body!r}"
        )


# ---------------------------------------------------------------------------
# 3. Frontmatter fields match JSONL fields
# ---------------------------------------------------------------------------


def test_frontmatter_pdf_page_matches_jsonl(run_output):
    saved, results, records, pages = run_output
    pages_dir = saved["pages_dir"]
    for r in records:
        md_path = pages_dir / f"page_{r['page']:03d}.md"
        fm, _ = _parse_frontmatter(md_path.read_text(encoding="utf-8"))
        assert fm.get("pdf_page") == r["page"], (
            f"Page {r['page']}: frontmatter pdf_page={fm.get('pdf_page')} "
            f"!= JSONL page={r['page']}"
        )


def test_frontmatter_book_page_matches_jsonl(run_output):
    saved, results, records, pages = run_output
    pages_dir = saved["pages_dir"]
    for r in records:
        md_path = pages_dir / f"page_{r['page']:03d}.md"
        fm, _ = _parse_frontmatter(md_path.read_text(encoding="utf-8"))
        assert fm.get("book_page") == r["book_page"], (
            f"Page {r['page']}: frontmatter book_page={fm.get('book_page')} "
            f"!= JSONL book_page={r['book_page']}"
        )


def test_frontmatter_source_matches_pdf_name(run_output, tmp_path):
    saved, results, records, pages = run_output
    pages_dir = saved["pages_dir"]
    for r in records:
        md_path = pages_dir / f"page_{r['page']:03d}.md"
        fm, _ = _parse_frontmatter(md_path.read_text(encoding="utf-8"))
        assert fm.get("source") == "test_book.pdf", (
            f"Page {r['page']}: frontmatter source={fm.get('source')!r} "
            f"!= expected 'test_book.pdf'"
        )


# ---------------------------------------------------------------------------
# 4. Combined markdown contains every page's text, in order
# ---------------------------------------------------------------------------


def test_combined_md_contains_all_pages_in_order(run_output):
    saved, results, records, pages = run_output
    combined = saved["combined_md"].read_text(encoding="utf-8")
    last_pos = 0
    for r in records:
        pos = combined.find(r["text"], last_pos)
        assert (
            pos != -1
        ), f"Page {r['page']} text not found in combined markdown after position {last_pos}"
        last_pos = pos + len(r["text"])


# ---------------------------------------------------------------------------
# 5. No orphan markdown files (every md has a JSONL record)
# ---------------------------------------------------------------------------


def test_no_orphan_page_md_files(run_output):
    saved, results, records, pages = run_output
    pages_dir = saved["pages_dir"]
    jsonl_pages = {r["page"] for r in records}
    md_pattern = re.compile(r"^page_(\d+)\.md$")
    for md_file in pages_dir.iterdir():
        m = md_pattern.match(md_file.name)
        assert m, f"Unexpected file in pages dir: {md_file.name}"
        page_num = int(m.group(1))
        assert (
            page_num in jsonl_pages
        ), f"Orphan markdown file {md_file.name} has no corresponding JSONL record"


# ---------------------------------------------------------------------------
# 6. JSONL records contain all required fields
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "page",
    "model",
    "text",
    "book_page",
    "input_tokens",
    "output_tokens",
    "cache_creation_tokens",
    "cache_read_tokens",
    "cost_usd",
    "elapsed_seconds",
    "extraction_method",
    "chunk_warnings",
    "paraphrased",
    "error",
}


def test_jsonl_records_have_required_fields(run_output):
    saved, results, records, pages = run_output
    for r in records:
        missing = REQUIRED_FIELDS - r.keys()
        assert (
            not missing
        ), f"Page {r['page']} JSONL record is missing fields: {missing}"


# ---------------------------------------------------------------------------
# 7. book_page is always int or None, never a string
# ---------------------------------------------------------------------------


def test_book_page_type_in_jsonl(run_output):
    saved, results, records, pages = run_output
    for r in records:
        bp = r["book_page"]
        assert bp is None or isinstance(bp, int), (
            f"Page {r['page']}: book_page={bp!r} is type {type(bp).__name__}, "
            f"expected int or None"
        )


# ---------------------------------------------------------------------------
# 8. JSONL record count matches number of input pages
# ---------------------------------------------------------------------------


def test_jsonl_record_count_matches_input(run_output, sample_pages):
    saved, results, records, pages = run_output
    assert len(records) == len(
        sample_pages
    ), f"JSONL has {len(records)} records but {len(sample_pages)} pages were processed"


# ---------------------------------------------------------------------------
# 9. JSONL page numbers are unique and sequential
# ---------------------------------------------------------------------------


def test_jsonl_page_numbers_unique(run_output):
    saved, results, records, pages = run_output
    page_nums = [r["page"] for r in records]
    assert len(page_nums) == len(
        set(page_nums)
    ), f"Duplicate page numbers in JSONL: {page_nums}"


def test_jsonl_pages_in_order(run_output):
    saved, results, records, pages = run_output
    page_nums = [r["page"] for r in records]
    assert page_nums == sorted(
        page_nums
    ), f"JSONL records are not in page order: {page_nums}"


# ---------------------------------------------------------------------------
# 10. Combined markdown uses HTML comment separators, not ## Page headers
# ---------------------------------------------------------------------------


def test_combined_md_uses_comment_separators_not_headers(run_output):
    saved, results, records, pages = run_output
    combined = saved["combined_md"].read_text(encoding="utf-8")

    # No ## Page N headings should appear
    assert not re.search(r"^## Page \d+", combined, re.MULTILINE), (
        "Combined markdown still contains '## Page N' headings — "
        "these should be HTML comment separators instead"
    )

    # Every page should have a ↓ arrow comment separator with its pdf page number
    for r in records:
        pattern = rf"<!-- ↓ PDF page: {r['page']}\s"
        assert re.search(
            pattern, combined
        ), f"No ↓ comment separator found for PDF page {r['page']} in combined markdown"
