"""
Tests for iter_pages() logging behaviour.

Verifies that unhandled page-extraction errors are emitted at ERROR level
so that monitoring systems and log aggregators can surface missing content.
No real PDFs or API calls required.
"""

import logging
from unittest.mock import MagicMock, patch

from src.docpeel.extraction import iter_pages


def _make_provider():
    """Minimal mock of a VisionProvider (non-Mistral)."""
    provider = MagicMock()
    provider.model_id = "test-model"
    # is_content_filter_error must return False so the error propagates
    provider.is_content_filter_error.return_value = False
    return provider


def _run_iter_pages(provider, error):
    """
    Drive iter_pages() for a single-page PDF, with the extractor raising
    the given error. Returns the consumed results list.
    """
    # Build a fitz mock that returns a valid-looking pixmap
    fake_pix = MagicMock()
    fake_pix.width = 100
    fake_pix.height = 100
    fake_pix.samples = b"\xff" * (100 * 100 * 3)
    mock_page = MagicMock()
    mock_page.get_pixmap.return_value = fake_pix
    mock_doc = MagicMock()
    mock_doc.__getitem__ = MagicMock(return_value=mock_page)
    mock_doc.__enter__ = MagicMock(return_value=mock_doc)
    mock_doc.__exit__ = MagicMock(return_value=False)

    with (
        patch("src.docpeel.extraction.page_count", return_value=1),
        patch("src.docpeel.extraction.fitz") as mock_fitz,
        patch(
            "src.docpeel.extraction.VisionExtractor.extract",
            side_effect=error,
        ),
    ):
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix.return_value = MagicMock()
        return list(iter_pages(MagicMock(), provider))


# ---------------------------------------------------------------------------
# Error logging level
# ---------------------------------------------------------------------------


def test_page_error_logged_at_error_level(caplog):
    provider = _make_provider()
    with caplog.at_level(logging.ERROR):
        _run_iter_pages(provider, RuntimeError("API timeout"))
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_page_error_not_silently_swallowed(caplog):
    """iter_pages must emit at least one ERROR record — never silence a page failure."""
    provider = _make_provider()
    with caplog.at_level(logging.DEBUG):
        results = _run_iter_pages(provider, RuntimeError("boom"))
    # The page is still yielded (as an error record), and an ERROR was logged
    assert len(results) == 1
    assert results[0]["error"] == "boom"
    assert any(r.levelno == logging.ERROR for r in caplog.records)
