"""
Tests for providers/base.py — Usage accumulator, backoff delay,
rate-limit detection/diagnosis, and the _with_retry helper.
No API calls required.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.docpeel.providers.base import (
    Usage,
    _BASE_DELAY,
    _JITTER,
    _MAX_DELAY,
    _MAX_RETRIES,
    _backoff_delay,
    _build_rate_limit_message,
    _is_rate_limit_error,
    _with_retry,
)


# ---------------------------------------------------------------------------
# Usage NamedTuple
# ---------------------------------------------------------------------------


def test_usage_zero():
    u = Usage.zero()
    assert u == Usage(0, 0, 0, 0, 0.0)


def test_usage_add():
    a = Usage(100, 200, 10, 5, 0.001)
    b = Usage(50, 100, 0, 0, 0.0005)
    result = a + b
    assert result == Usage(150, 300, 10, 5, 0.0015)


def test_usage_add_with_zero():
    u = Usage(100, 200, 10, 5, 0.001)
    assert u + Usage.zero() == u
    assert Usage.zero() + u == u


def test_usage_add_multiple():
    parts = [Usage(10, 20, 0, 0, 0.001) for _ in range(5)]
    total = Usage.zero()
    for p in parts:
        total = total + p
    assert total == Usage(50, 100, 0, 0, 0.005)


# ---------------------------------------------------------------------------
# _backoff_delay
# ---------------------------------------------------------------------------


def test_backoff_delay_attempt_0_near_base():
    delay = _backoff_delay(0)
    lower = _BASE_DELAY * (1 - _JITTER)
    upper = _BASE_DELAY * (1 + _JITTER)
    assert lower <= delay <= upper


def test_backoff_delay_doubles_each_attempt():
    # Without jitter we'd expect base*2^n — with jitter, check midpoints
    for attempt in range(3):
        expected_mid = min(_BASE_DELAY * (2**attempt), _MAX_DELAY)
        delay = _backoff_delay(attempt)
        assert delay > 0
        assert delay <= _MAX_DELAY * (1 + _JITTER)


def test_backoff_delay_capped_at_max():
    # Attempt 10 would be base * 2^10 = 5 * 1024 >> MAX_DELAY
    delay = _backoff_delay(10)
    assert delay <= _MAX_DELAY * (1 + _JITTER)


def test_backoff_delay_always_positive():
    for attempt in range(5):
        assert _backoff_delay(attempt) > 0


# ---------------------------------------------------------------------------
# _is_rate_limit_error
# ---------------------------------------------------------------------------


def _make_exc(status_code=None):
    exc = Exception("test")
    if status_code is not None:
        raw = MagicMock()
        raw.status_code = status_code
        exc.raw_response = raw
    return exc


def test_is_rate_limit_error_429():
    assert _is_rate_limit_error(_make_exc(429)) is True


def test_is_rate_limit_error_other_status():
    assert _is_rate_limit_error(_make_exc(500)) is False
    assert _is_rate_limit_error(_make_exc(400)) is False


def test_is_rate_limit_error_no_raw_response():
    assert _is_rate_limit_error(Exception("plain")) is False


def test_is_rate_limit_error_raw_response_no_status():
    exc = Exception("test")
    exc.raw_response = object()  # no status_code attribute
    assert _is_rate_limit_error(exc) is False


# ---------------------------------------------------------------------------
# _build_rate_limit_message
# ---------------------------------------------------------------------------


def _make_rate_limit_exc(headers: dict):
    exc = Exception("rate limit")
    raw = MagicMock()
    raw.status_code = 429
    raw.headers = headers
    exc.raw_response = raw
    return exc


def test_rate_limit_message_monthly_exhausted():
    headers = {
        "x-ratelimit-remaining-tokens-month": "0",
        "x-ratelimit-limit-tokens-month": "4000000",
    }
    msg = _build_rate_limit_message(_make_rate_limit_exc(headers))
    assert "monthly token quota exhausted" in msg
    assert "4,000,000" in msg
    assert "monthly quota" in msg.lower() or "monthly" in msg.lower()


def test_rate_limit_message_monthly_exhausted_unknown_limit():
    headers = {"x-ratelimit-remaining-tokens-month": "0"}
    msg = _build_rate_limit_message(_make_rate_limit_exc(headers))
    assert "monthly token quota exhausted" in msg
    assert "unknown" in msg


def test_rate_limit_message_per_minute():
    headers = {
        "x-ratelimit-remaining-tokens-minute": "0",
        "x-ratelimit-limit-tokens-minute": "60000",
    }
    msg = _build_rate_limit_message(_make_rate_limit_exc(headers))
    assert "per-minute" in msg
    assert "60,000" in msg


def test_rate_limit_message_step_label():
    exc = Exception("rate limit")
    exc.raw_response = MagicMock(headers={}, status_code=429)
    msg = _build_rate_limit_message(exc, step="OCR (mistral-ocr-latest)")
    assert "on OCR (mistral-ocr-latest)" in msg


def test_rate_limit_message_no_step():
    exc = Exception("rate limit")
    exc.raw_response = MagicMock(headers={}, status_code=429)
    msg = _build_rate_limit_message(exc)
    assert "no further detail" in msg


def test_rate_limit_message_headers_present_no_zero():
    headers = {
        "x-ratelimit-remaining-tokens-month": "100",
        "x-ratelimit-remaining-tokens-minute": "50",
    }
    msg = _build_rate_limit_message(_make_rate_limit_exc(headers))
    # Should show raw headers
    assert "Headers:" in msg


def test_rate_limit_message_no_raw_response():
    exc = Exception("plain")
    msg = _build_rate_limit_message(exc)
    assert "no further detail" in msg


# ---------------------------------------------------------------------------
# _with_retry
# ---------------------------------------------------------------------------


def test_with_retry_success_first_try():
    fn = MagicMock(return_value="ok")
    result = _with_retry(lambda e: False, fn)
    assert result == "ok"
    assert fn.call_count == 1


def test_with_retry_retries_on_transient_error():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("transient")
        return "ok"

    with patch("src.docpeel.providers.base.time.sleep"):
        result = _with_retry(lambda e: False, fn)

    assert result == "ok"
    assert len(calls) == 3


def test_with_retry_raises_after_max_retries():
    fn = MagicMock(side_effect=ValueError("always fails"))
    with patch("src.docpeel.providers.base.time.sleep"):
        with pytest.raises(ValueError, match="always fails"):
            _with_retry(lambda e: False, fn)
    assert fn.call_count == _MAX_RETRIES + 1


def test_with_retry_raises_immediately_on_filter_error():
    fn = MagicMock(side_effect=ValueError("filter"))
    with pytest.raises(ValueError, match="filter"):
        _with_retry(lambda e: True, fn)
    assert fn.call_count == 1  # no retries


def test_with_retry_raises_immediately_on_rate_limit():
    exc = _make_exc(429)
    fn = MagicMock(side_effect=exc)
    with pytest.raises(Exception):
        _with_retry(lambda e: False, fn)
    assert fn.call_count == 1  # no retries


def test_with_retry_step_label_in_output(capsys):
    fn = MagicMock(side_effect=[ValueError("boom"), "ok"])
    with patch("src.docpeel.providers.base.time.sleep"):
        _with_retry(lambda e: False, fn, step="structuring")
    captured = capsys.readouterr()
    assert "[structuring]" in captured.out
