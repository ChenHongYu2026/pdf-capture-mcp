"""Tests for the MCP server and tools."""

from __future__ import annotations

import asyncio

import pytest

from pdf_capture_mcp.server import mcp


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestServerInit:
    """Test server initialization and tool registration."""

    def test_server_name(self):
        assert mcp.name == "pdf-capture"

    def test_tools_registered(self):
        async def _check():
            tools = await mcp.list_tools()
            names = {t.name for t in tools}
            assert "setup_vlm" in names
            assert "check_environment" in names
            assert "pdf_to_markdown" in names
            assert "extract_tables" in names
            assert "classify_document" in names
            assert "pdf_info" in names
            assert "install_engine" in names
            return len(tools)

        count = asyncio.run(_check())
        assert count == 7


class TestPdfInfo:
    """Test pdf_info tool with error handling."""

    def test_nonexistent_file(self):
        async def _call():
            result = await mcp.call_tool("pdf_info", {"pdf_path": "/nonexistent/ghost.pdf"})
            return result

        result = asyncio.run(_call())
        # Should return error JSON, not crash
        assert result is not None


class TestClassifyDocument:
    """Test classify_document tool."""

    def test_nonexistent_file(self):
        async def _call():
            result = await mcp.call_tool(
                "classify_document", {"pdf_path": "/nonexistent/ghost.pdf"}
            )
            return result

        result = asyncio.run(_call())
        assert result is not None
