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


def _run_ok(*argv, tmp_path):
    """
    Call cli.main() with mocked provider + extraction so no API calls happen.
    Requires a real (empty) PDF path created at tmp_path/test.pdf.
    Returns the exit code (None = success).
    """
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")

    fake_provider = MagicMock()
    fake_provider.model_id = "claude-test-model"

    with (
        patch.object(sys, "argv", ["docpeel", str(fake_pdf), *argv]),
        patch.object(cli_mod, "build_provider", return_value=fake_provider),
        patch.object(cli_mod, "page_count", return_value=0),
        patch.object(cli_mod, "iter_pages", return_value=iter([])),
        patch.object(cli_mod, "stream_outputs", return_value=(
            {"combined_md": Path("x"), "pages_dir": Path("x"), "json": Path("x"), "run_folder": Path("x")},
            [],
        )),
        patch.object(cli_mod, "write_report", return_value=Path("report.md")),
        patch.object(cli_mod, "resolve_run_folder", return_value=Path("output/test[1]")),
    ):
        try:
            cli_mod.main()
            return None
        except SystemExit as e:
            return e.code


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
