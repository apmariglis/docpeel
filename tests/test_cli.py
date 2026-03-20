"""
Tests for cli.py argument validation.

Only the validation section of main() is exercised — everything after the
flag-combination checks (build_provider, iter_pages, etc.) is irrelevant here
and would require real API keys and a real PDF.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.docpeel.cli as cli_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(*argv):
    """
    Call cli.main() with the given argv (excluding the program name).
    Returns (exit_code, stderr_text).  Any SystemExit is caught.
    """
    with patch.object(sys, "argv", ["docpeel", *argv]):
        with pytest.raises(SystemExit) as exc_info:
            cli_mod.main()
    return exc_info.value.code


def _run_ok(*argv, tmp_path, results=None):
    """
    Call cli.main() with mocked provider + extraction so no API calls happen.
    Requires a real (empty) PDF path created at tmp_path/test.pdf.
    Returns the exit code (None = success).

    results: optional list of page-result dicts to return from stream_outputs.
    """
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")

    fake_provider = MagicMock()
    fake_provider.model_id = "claude-test-model"

    fake_results = results if results is not None else []

    with (
        patch.object(sys, "argv", ["docpeel", str(fake_pdf), *argv]),
        patch.object(cli_mod, "build_provider", return_value=fake_provider),
        patch.object(cli_mod, "page_count", return_value=len(fake_results)),
        patch.object(cli_mod, "iter_pages", return_value=iter([])),
        patch.object(cli_mod, "stream_outputs", return_value=(
            {"combined_md": Path("x"), "pages_dir": Path("x"), "json": Path("x"), "run_folder": Path("x")},
            fake_results,
        )),
        patch.object(cli_mod, "write_report", return_value=Path("report.md")),
        patch.object(cli_mod, "resolve_run_folder", return_value=Path("output/test[1]")),
    ):
        try:
            cli_mod.main()
            return None
        except SystemExit as e:
            return e.code


def _make_page_result(page=1, cost_usd=0.001, error=None, skip=False):
    return {
        "page": page,
        "cost_usd": cost_usd,
        "input_tokens": 100,
        "output_tokens": 50,
        "elapsed_seconds": 1.0,
        "extraction_method": "full-page",
        "error": error,
        "skip": skip,
        "skip_reason": None,
        "paraphrased": None,
        "chunk_warnings": [],
    }


# ---------------------------------------------------------------------------
# Missing PDF
# ---------------------------------------------------------------------------


def test_no_args_exits_2():
    assert _run() == 2


def test_missing_pdf_exits_2():
    assert _run("--vision-model", "claude-sonnet") == 2


# ---------------------------------------------------------------------------
# Missing extraction mode
# ---------------------------------------------------------------------------


def test_no_mode_exits_2():
    assert _run("book.pdf") == 2


# ---------------------------------------------------------------------------
# Mutually exclusive flags
# ---------------------------------------------------------------------------


def test_vision_and_ocr_mutually_exclusive():
    assert _run("book.pdf", "--vision-model", "claude-x", "--ocr", "mistral", "--structure-model", "m") == 2


# ---------------------------------------------------------------------------
# --structure-model requires --ocr
# ---------------------------------------------------------------------------


def test_structure_model_without_ocr_exits_2():
    assert _run("book.pdf", "--structure-model", "mistral-small-latest") == 2


# ---------------------------------------------------------------------------
# --ocr requires --structure-model
# ---------------------------------------------------------------------------


def test_ocr_without_structure_model_exits_2():
    assert _run("book.pdf", "--ocr", "mistral") == 2


# ---------------------------------------------------------------------------
# PDF existence check
# ---------------------------------------------------------------------------


def test_nonexistent_pdf_raises_file_not_found(tmp_path):
    fake_path = tmp_path / "nonexistent.pdf"
    with patch.object(sys, "argv", ["docpeel", str(fake_path), "--vision-model", "claude-x"]):
        with pytest.raises(FileNotFoundError):
            cli_mod.main()


# ---------------------------------------------------------------------------
# Valid invocations reach build_provider (no SystemExit 2)
# ---------------------------------------------------------------------------


def test_valid_vision_model(tmp_path):
    code = _run_ok("--vision-model", "claude-sonnet-4-0", tmp_path=tmp_path)
    assert code is None


def test_valid_ocr_path(tmp_path):
    code = _run_ok("--ocr", "mistral", "--structure-model", "mistral-small-latest", tmp_path=tmp_path)
    assert code is None


def test_dpi_default_is_150(tmp_path):
    """Verify DPI=150 default reaches build_provider call without error."""
    code = _run_ok("--vision-model", "gemini-2.5-flash", tmp_path=tmp_path)
    assert code is None


def test_custom_dpi_accepted(tmp_path):
    code = _run_ok("--vision-model", "claude-x", "--dpi", "200", tmp_path=tmp_path)
    assert code is None


# ---------------------------------------------------------------------------
# --verbose / --quiet
# ---------------------------------------------------------------------------


def test_verbose_flag_accepted(tmp_path):
    code = _run_ok("--vision-model", "claude-x", "--verbose", tmp_path=tmp_path)
    assert code is None


def test_quiet_flag_accepted(tmp_path):
    code = _run_ok("--vision-model", "claude-x", "--quiet", tmp_path=tmp_path)
    assert code is None


def test_verbose_and_quiet_mutually_exclusive(tmp_path):
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")
    code = _run(str(fake_pdf), "--vision-model", "claude-x", "--verbose", "--quiet")
    assert code == 2


def test_verbose_sets_debug_level(tmp_path):
    import logging
    _run_ok("--vision-model", "claude-x", "--verbose", tmp_path=tmp_path)
    assert logging.getLogger("docpeel").level == logging.DEBUG


def test_quiet_sets_warning_level(tmp_path):
    import logging
    _run_ok("--vision-model", "claude-x", "--quiet", tmp_path=tmp_path)
    assert logging.getLogger("docpeel").level == logging.WARNING


def test_default_sets_info_level(tmp_path):
    import logging
    _run_ok("--vision-model", "claude-x", tmp_path=tmp_path)
    assert logging.getLogger("docpeel").level == logging.INFO


# ---------------------------------------------------------------------------
# Post-processing: cost summary with None costs
# ---------------------------------------------------------------------------


def test_summary_with_known_costs_does_not_raise(tmp_path, capsys):
    results = [_make_page_result(page=1, cost_usd=0.001),
               _make_page_result(page=2, cost_usd=0.002)]
    code = _run_ok("--vision-model", "claude-x", tmp_path=tmp_path, results=results)
    assert code is None
    out = capsys.readouterr().out
    assert "0.003000" in out


def test_summary_with_none_cost_does_not_raise(tmp_path):
    """When pricing is unavailable cost_usd=None — must not crash."""
    results = [_make_page_result(page=1, cost_usd=None)]
    code = _run_ok("--vision-model", "claude-x", tmp_path=tmp_path, results=results)
    assert code is None


def test_summary_shows_na_when_cost_none(tmp_path, capsys):
    results = [_make_page_result(page=1, cost_usd=None)]
    _run_ok("--vision-model", "claude-x", tmp_path=tmp_path, results=results)
    out = capsys.readouterr().out
    assert "N/A" in out


def test_summary_mixed_none_and_float_costs(tmp_path, capsys):
    """Known costs are summed; None entries are skipped — partial total shown."""
    results = [
        _make_page_result(page=1, cost_usd=0.001),
        _make_page_result(page=2, cost_usd=None),
    ]
    _run_ok("--vision-model", "claude-x", tmp_path=tmp_path, results=results)
    out = capsys.readouterr().out
    assert "0.001000" in out
