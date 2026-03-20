"""
Tests for scripts/benchmark.py — pure logic only.

Integration tests (API calls, PDF rendering, file I/O) are not covered here;
those require real credentials and files.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the benchmark module without executing main()
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).parent.parent / "scripts" / "benchmark" / "benchmark.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("benchmark", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bm = _load_benchmark()


# ---------------------------------------------------------------------------
# _anonymize_record
# ---------------------------------------------------------------------------

_FULL_RECORD = {
    "page": 3,
    "model": "claude-sonnet-4-6",
    "dpi": 150,
    "skip": False,
    "skip_reason": None,
    "book_page": 42,
    "title": "Chapter 4",
    "text": "Some body text.",
    "tables": [],
    "watermarks": [],
    "extraction_method": "full-page",
    "paraphrased": None,
    "chunk_warnings": [],
    "error": None,
    "input_tokens": 718,
    "output_tokens": 1993,
    "cache_creation_tokens": 0,
    "cache_read_tokens": 0,
    "cost_usd": 0.002208,
    "elapsed_seconds": 24.4,
}

_IDENTIFYING_FIELDS = {
    "model",
    "dpi",
    "extraction_method",
    "paraphrased",
    "chunk_warnings",
    "input_tokens",
    "output_tokens",
    "cache_creation_tokens",
    "cache_read_tokens",
    "cost_usd",
    "elapsed_seconds",
}

_KEPT_FIELDS = {
    "page", "skip", "skip_reason", "book_page",
    "title", "text", "tables", "watermarks", "error",
}


def test_anonymize_strips_identifying_fields():
    result = bm._anonymize_record(_FULL_RECORD)
    for field in _IDENTIFYING_FIELDS:
        assert field not in result, f"Field '{field}' should have been stripped"


def test_anonymize_keeps_content_fields():
    result = bm._anonymize_record(_FULL_RECORD)
    for field in _KEPT_FIELDS:
        assert field in result, f"Field '{field}' should be kept"


def test_anonymize_does_not_mutate_input():
    original = dict(_FULL_RECORD)
    bm._anonymize_record(_FULL_RECORD)
    assert _FULL_RECORD == original


def test_anonymize_preserves_values():
    result = bm._anonymize_record(_FULL_RECORD)
    assert result["page"] == 3
    assert result["title"] == "Chapter 4"
    assert result["text"] == "Some body text."


# ---------------------------------------------------------------------------
# _path_label
# ---------------------------------------------------------------------------


def test_path_label_vision():
    assert bm._path_label({"vision_model": "claude-sonnet-4-6"}) == "claude-sonnet-4-6"


def test_path_label_vision_gemini():
    assert bm._path_label({"vision_model": "gemini-2.5-flash"}) == "gemini-2.5-flash"


def test_path_label_ocr_mistral_structure():
    label = bm._path_label({"ocr": "mistral", "structure_model": "mistral-small-latest"})
    assert label == "mistral-ocr+mistral-small-latest"


def test_path_label_ocr_claude_structure():
    label = bm._path_label({"ocr": "mistral", "structure_model": "claude-haiku-4-5-20251001"})
    assert label == "mistral-ocr+claude-haiku-4-5-20251001"


def test_path_label_ocr_gemini_structure():
    label = bm._path_label({"ocr": "mistral", "structure_model": "gemini-2.5-flash-lite"})
    assert label == "mistral-ocr+gemini-2.5-flash-lite"


# ---------------------------------------------------------------------------
# _required_env_keys
# ---------------------------------------------------------------------------


def test_required_keys_claude_vision():
    assert bm._required_env_keys({"vision_model": "claude-sonnet-4-6"}) == {"ANTHROPIC_API_KEY"}


def test_required_keys_gemini_vision():
    assert bm._required_env_keys({"vision_model": "gemini-2.5-flash"}) == {"GOOGLE_API_KEY"}


def test_required_keys_mistral_ocr_mistral_structure():
    keys = bm._required_env_keys({"ocr": "mistral", "structure_model": "mistral-small-latest"})
    assert keys == {"MISTRAL_API_KEY"}


def test_required_keys_mistral_ocr_claude_structure():
    keys = bm._required_env_keys({"ocr": "mistral", "structure_model": "claude-haiku-4-5-20251001"})
    assert keys == {"MISTRAL_API_KEY", "ANTHROPIC_API_KEY"}


def test_required_keys_mistral_ocr_gemini_structure():
    keys = bm._required_env_keys({"ocr": "mistral", "structure_model": "gemini-2.5-flash-lite"})
    assert keys == {"MISTRAL_API_KEY", "GOOGLE_API_KEY"}


# ---------------------------------------------------------------------------
# _is_path_available
# ---------------------------------------------------------------------------


def test_path_available_when_key_present():
    env = {"ANTHROPIC_API_KEY": "sk-ant-xxx"}
    assert bm._is_path_available({"vision_model": "claude-sonnet-4-6"}, env) is True


def test_path_unavailable_when_key_missing():
    env = {}
    assert bm._is_path_available({"vision_model": "claude-sonnet-4-6"}, env) is False


def test_path_available_when_all_keys_present():
    env = {"MISTRAL_API_KEY": "x", "ANTHROPIC_API_KEY": "y"}
    path = {"ocr": "mistral", "structure_model": "claude-haiku-4-5-20251001"}
    assert bm._is_path_available(path, env) is True


def test_path_unavailable_when_one_key_missing():
    env = {"MISTRAL_API_KEY": "x"}  # missing ANTHROPIC_API_KEY
    path = {"ocr": "mistral", "structure_model": "claude-haiku-4-5-20251001"}
    assert bm._is_path_available(path, env) is False


# ---------------------------------------------------------------------------
# _bench_label  (folder-safe label for a benchmark entry)
# ---------------------------------------------------------------------------


def test_bench_label_basic():
    label = bm._bench_label(Path("/data/books/my_book.pdf"), "1,3,7-10")
    assert label == "my_book_pdf__pp_1_3_7-10"


def test_bench_label_no_special_chars_in_output():
    label = bm._bench_label(Path("some file (1).pdf"), "5-8")
    # Should not contain spaces or parentheses
    assert " " not in label
    assert "(" not in label
    assert ")" not in label


# ---------------------------------------------------------------------------
# _render_report  (smoke test — checks structural elements)
# ---------------------------------------------------------------------------

_MOCK_RESULTS = [
    {
        "pdf": "/data/book.pdf",
        "pages": "1,3",
        "notes": "Dense tables",
        "paths": [
            {
                "label": "claude-sonnet-4-6",
                "page_assessments": [
                    {
                        "page": 1,
                        "assessment": {
                            "text_score": 9,
                            "text_notes": "Accurate",
                            "tables_score": 8,
                            "tables_notes": "Minor column gap",
                            "skip_correct": True,
                            "skip_notes": None,
                            "title_correct": True,
                            "title_notes": None,
                            "page_number_correct": True,
                            "overall_score": 9,
                            "summary": "Good extraction.",
                        },
                    },
                    {
                        "page": 3,
                        "assessment": {
                            "text_score": 7,
                            "text_notes": "Missing last paragraph",
                            "tables_score": None,
                            "tables_notes": None,
                            "skip_correct": True,
                            "skip_notes": None,
                            "title_correct": None,
                            "title_notes": None,
                            "page_number_correct": True,
                            "overall_score": 7,
                            "summary": "Mostly good but paragraph cut off.",
                        },
                    },
                ],
            },
            {
                "label": "gemini-2.5-flash-lite",
                "error": "GOOGLE_API_KEY not set",
            },
        ],
    }
]

_MOCK_RUN_CONFIG = {
    "timestamp": "2026-03-20 14:30",
    "judge_model": "claude-sonnet-4-6",
    "git_commit": "abc1234",
}


def test_report_contains_timestamp():
    report = bm._render_report(_MOCK_RESULTS, _MOCK_RUN_CONFIG)
    assert "2026-03-20 14:30" in report


def test_report_contains_judge_model():
    report = bm._render_report(_MOCK_RESULTS, _MOCK_RUN_CONFIG)
    assert "claude-sonnet-4-6" in report


def test_report_contains_pdf_name():
    report = bm._render_report(_MOCK_RESULTS, _MOCK_RUN_CONFIG)
    assert "book.pdf" in report


def test_report_contains_path_label():
    report = bm._render_report(_MOCK_RESULTS, _MOCK_RUN_CONFIG)
    assert "claude-sonnet-4-6" in report


def test_report_contains_overall_score():
    report = bm._render_report(_MOCK_RESULTS, _MOCK_RUN_CONFIG)
    # Average of 9 and 7 = 8.0
    assert "8.0" in report


def test_report_contains_skipped_path_error():
    report = bm._render_report(_MOCK_RESULTS, _MOCK_RUN_CONFIG)
    assert "gemini-2.5-flash-lite" in report
    assert "GOOGLE_API_KEY not set" in report


def test_report_contains_page_details():
    report = bm._render_report(_MOCK_RESULTS, _MOCK_RUN_CONFIG)
    assert "Missing last paragraph" in report


def test_report_contains_git_commit():
    report = bm._render_report(_MOCK_RESULTS, _MOCK_RUN_CONFIG)
    assert "abc1234" in report


def test_report_contains_summary_table():
    report = bm._render_report(_MOCK_RESULTS, _MOCK_RUN_CONFIG)
    # Summary table should have a markdown table header
    assert "|" in report


# ---------------------------------------------------------------------------
# _render_report — cost section
# ---------------------------------------------------------------------------

_MOCK_COSTS = {
    "extraction": {
        "claude-sonnet-4-6":     0.012340,
        "gemini-2.5-flash-lite": 0.001820,
    },
    "extraction_seconds": {
        "claude-sonnet-4-6":     12.5,
        "gemini-2.5-flash-lite": 8.3,
    },
    "judging": {
        "claude-sonnet-4-6":     0.004100,
        "gemini-2.5-flash-lite": 0.003900,
    },
    "judging_seconds": {
        "claude-sonnet-4-6":     5.2,
        "gemini-2.5-flash-lite": 4.8,
    },
    "judge_model":   "claude-sonnet-4-6",
    "total_elapsed": 35.1,
}


def test_report_cost_section_present():
    report = bm._render_report(_MOCK_RESULTS, _MOCK_RUN_CONFIG, costs=_MOCK_COSTS)
    assert "## Costs & Time" in report


def test_report_cost_section_contains_labels():
    report = bm._render_report(_MOCK_RESULTS, _MOCK_RUN_CONFIG, costs=_MOCK_COSTS)
    assert "claude-sonnet-4-6" in report
    assert "gemini-2.5-flash-lite" in report


def test_report_cost_section_contains_totals():
    report = bm._render_report(_MOCK_RESULTS, _MOCK_RUN_CONFIG, costs=_MOCK_COSTS)
    # Grand extraction total: 0.012340 + 0.001820 = 0.014160
    assert "0.014160" in report
    # Grand judging total: 0.004100 + 0.003900 = 0.008000
    assert "0.008000" in report


def test_report_cost_section_contains_timing():
    report = bm._render_report(_MOCK_RESULTS, _MOCK_RUN_CONFIG, costs=_MOCK_COSTS)
    assert "12.5s" in report   # extraction time for claude
    assert "35.1s" in report   # total elapsed


def test_report_no_cost_section_when_costs_omitted():
    report = bm._render_report(_MOCK_RESULTS, _MOCK_RUN_CONFIG)
    assert "## Costs" not in report
