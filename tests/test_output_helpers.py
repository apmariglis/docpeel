"""
Tests for output.py helper functions — folder naming, outcome/method labels,
and page-note formatting. No API calls or real PDFs required.
"""

import re
from pathlib import Path

import pytest

import src.docpeel.output as out_mod
from src.docpeel.output import (
    _method_col,
    _outcome_col,
    _page_note_md,
    _resolve_output_folder,
)


# ---------------------------------------------------------------------------
# _resolve_output_folder — folder naming and auto-increment
# ---------------------------------------------------------------------------


def test_resolve_folder_first_run(tmp_path):
    folder = _resolve_output_folder_patched(tmp_path, "claude-3-5-sonnet", "mybook")
    assert folder.name == "claude-3-5-sonnet__mybook[1]"


def test_resolve_folder_increments(tmp_path):
    # Create a fake run-1 folder
    (tmp_path / "claude__book[1]").mkdir()
    folder = _resolve_output_folder_patched(tmp_path, "claude", "book")
    assert folder.name == "claude__book[2]"


def test_resolve_folder_finds_max(tmp_path):
    (tmp_path / "m__pdf[1]").mkdir()
    (tmp_path / "m__pdf[3]").mkdir()  # gap — should still pick 4
    (tmp_path / "m__pdf[2]").mkdir()
    folder = _resolve_output_folder_patched(tmp_path, "m", "pdf")
    assert folder.name == "m__pdf[4]"


def test_resolve_folder_ignores_different_provider(tmp_path):
    (tmp_path / "other__book[1]").mkdir()
    folder = _resolve_output_folder_patched(tmp_path, "mine", "book")
    assert folder.name == "mine__book[1]"


def test_resolve_folder_ignores_different_pdf(tmp_path):
    (tmp_path / "model__book_a[1]").mkdir()
    folder = _resolve_output_folder_patched(tmp_path, "model", "book_b")
    assert folder.name == "model__book_b[1]"


def test_resolve_folder_ignores_non_matching_dirs(tmp_path):
    (tmp_path / "model__book[bad]").mkdir()
    (tmp_path / "model__book").mkdir()  # missing brackets
    folder = _resolve_output_folder_patched(tmp_path, "model", "book")
    assert folder.name == "model__book[1]"


def test_resolve_folder_base_not_exist(tmp_path):
    """Works even if the base output directory doesn't yet exist."""
    base = tmp_path / "output"
    # base does not exist
    folder = _resolve_output_folder_patched(base, "model", "pdf")
    assert folder.name == "model__pdf[1]"


def _resolve_output_folder_patched(base: Path, provider: str, pdf_stem: str) -> Path:
    """Call _resolve_output_folder with OUTPUT_FOLDER monkey-patched to base."""
    original = out_mod.OUTPUT_FOLDER
    out_mod.OUTPUT_FOLDER = base
    try:
        pdf_path = Path(f"/fake/{pdf_stem}.pdf")
        return _resolve_output_folder(pdf_path, provider)
    finally:
        out_mod.OUTPUT_FOLDER = original


# ---------------------------------------------------------------------------
# _outcome_col
# ---------------------------------------------------------------------------


def _r(**kwargs):
    """Build a minimal result dict for _outcome_col / _method_col."""
    return {
        "skip": False,
        "error": None,
        "paraphrased": None,
        "chunk_warnings": [],
        "skip_reason": None,
        "extraction_method": "full-page",
        **kwargs,
    }


def test_outcome_ok():
    assert _outcome_col(_r()) == "✅ ok"


def test_outcome_skipped():
    result = _outcome_col(_r(skip=True, skip_reason="blank"))
    assert "skipped" in result
    assert "blank" in result


def test_outcome_skipped_no_reason():
    result = _outcome_col(_r(skip=True))
    assert "skipped" in result


def test_outcome_failed():
    assert _outcome_col(_r(error="timeout")) == "⛔ failed"


def test_outcome_paraphrased_full():
    assert _outcome_col(_r(paraphrased="full")) == "⚠️ paraphrased"


def test_outcome_paraphrased_partial_no_missing():
    assert _outcome_col(_r(paraphrased="partial")) == "⚠️ partial paraphrase"


def test_outcome_paraphrased_partial_with_missing_chunks():
    warnings = ["chunk 'top-left' blocked and skipped"]
    result = _outcome_col(_r(paraphrased="partial", chunk_warnings=warnings))
    assert "chunks missing" in result
    assert "partial paraphrase" in result


def test_outcome_chunks_missing():
    warnings = ["chunk 'top-right' blocked and skipped"]
    result = _outcome_col(_r(chunk_warnings=warnings))
    assert "chunks missing" in result


def test_outcome_chunk_warning_with_paraphrase_not_counted_as_missing():
    """Warnings mentioning 'paraphrase' should not trigger chunks-missing path."""
    warnings = ["chunk 'x' could not be transcribed verbatim — paraphrased instead"]
    assert _outcome_col(_r(chunk_warnings=warnings)) == "✅ ok"


# ---------------------------------------------------------------------------
# _method_col
# ---------------------------------------------------------------------------


def test_method_col_full_page():
    assert _method_col(_r(extraction_method="full-page")) == "full-page"


def test_method_col_quadrant_split():
    assert _method_col(_r(extraction_method="quadrant-split")) == "quadrant-split"


def test_method_col_ocr_structure():
    assert _method_col(_r(extraction_method="ocr+structure")) == "ocr+structure"


def test_method_col_skipped():
    assert _method_col(_r(skip=True)) == "—"


def test_method_col_error_shows_method():
    """Even failed pages should show their attempted extraction method."""
    assert _method_col(_r(error="boom", extraction_method="full-page")) == "full-page"


# ---------------------------------------------------------------------------
# _page_note_md
# ---------------------------------------------------------------------------


def test_page_note_md_no_issues():
    note = _page_note_md({"paraphrased": None, "chunk_warnings": []})
    assert note.startswith("*[Extracted via quadrant-split")
    assert "paraphrased" not in note
    assert "skipped" not in note


def test_page_note_md_full_paraphrase():
    note = _page_note_md({"paraphrased": "full", "chunk_warnings": []})
    assert "full page paraphrased" in note


def test_page_note_md_partial_paraphrase():
    note = _page_note_md({"paraphrased": "partial", "chunk_warnings": []})
    assert "some chunks paraphrased" in note


def test_page_note_md_chunk_warnings():
    note = _page_note_md({"paraphrased": None, "chunk_warnings": ["x skipped"]})
    assert "some chunks skipped" in note


def test_page_note_md_is_italic_markdown():
    note = _page_note_md({"paraphrased": None, "chunk_warnings": []})
    assert note.startswith("*[")
    assert note.endswith("]*")
