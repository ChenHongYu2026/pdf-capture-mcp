"""Tests for proxy environment sanitation and page-range parsing."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_proxy_env(monkeypatch):
    """Start every test with no proxy-related variables set."""
    for var in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def test_no_proxy_untouched_when_no_proxy_configured(monkeypatch):
    import os

    from pdf_capture_mcp.server import _sanitize_proxy_env

    _sanitize_proxy_env()
    assert "NO_PROXY" not in os.environ


def test_localhost_appended_when_proxy_set(monkeypatch):
    import os

    from pdf_capture_mcp.server import _sanitize_proxy_env

    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7890")
    _sanitize_proxy_env()
    for var in ("NO_PROXY", "no_proxy"):
        entries = os.environ[var].split(",")
        assert "localhost" in entries
        assert "127.0.0.1" in entries


def test_existing_no_proxy_entries_preserved(monkeypatch):
    import os

    from pdf_capture_mcp.server import _sanitize_proxy_env

    monkeypatch.setenv("http_proxy", "http://127.0.0.1:7890")
    monkeypatch.setenv("NO_PROXY", "huggingface.co,*.huggingface.co")
    _sanitize_proxy_env()
    entries = os.environ["NO_PROXY"].split(",")
    # Original entries kept (appended, never overwritten)
    assert "huggingface.co" in entries
    assert "*.huggingface.co" in entries
    assert "localhost" in entries
    assert "127.0.0.1" in entries


def test_idempotent_when_localhost_already_present(monkeypatch):
    import os

    from pdf_capture_mcp.server import _sanitize_proxy_env

    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7890")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1")
    _sanitize_proxy_env()
    assert os.environ["NO_PROXY"] == "localhost,127.0.0.1"
    assert os.environ["no_proxy"] == "localhost,127.0.0.1"


class TestParsePageRange:
    def test_empty_returns_none(self):
        from pdf_capture_mcp.server import _parse_page_range

        assert _parse_page_range("") is None
        assert _parse_page_range("   ") is None

    def test_simple_range(self):
        from pdf_capture_mcp.server import _parse_page_range

        assert _parse_page_range("0-3") == [0, 1, 2, 3]

    def test_mixed_list_and_range_deduplicated_sorted(self):
        from pdf_capture_mcp.server import _parse_page_range

        assert _parse_page_range("5, 0, 2-4, 3") == [0, 2, 3, 4, 5]

    def test_invalid_raises_value_error(self):
        from pdf_capture_mcp.server import _parse_page_range

        with pytest.raises(ValueError):
            _parse_page_range("abc")
