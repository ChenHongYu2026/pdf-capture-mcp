"""Tests for v0.10.0 scanned-document detection (single source of truth)."""

from __future__ import annotations

import fitz

from pdf_capture_mcp.classifier import classify_document, detect_text_layer


def _pdf(path, pages_with_text, pages_blank):
    doc = fitz.open()
    line = "plenty of digital text layer content on this page for the probe"
    for _ in range(pages_with_text):
        page = doc.new_page()
        for row in range(6):
            page.insert_text((36, 60 + 14 * row), line)
    for _ in range(pages_blank):
        doc.new_page()
    doc.save(str(path))
    doc.close()
    return path


def test_pure_scan_detected(tmp_path):
    pdf = _pdf(tmp_path / "scan.pdf", 0, 12)
    is_scanned, coverage = detect_text_layer(pdf)
    assert is_scanned is True
    assert coverage == 0.0


def test_digital_document_not_scanned(tmp_path):
    pdf = _pdf(tmp_path / "digital.pdf", 12, 0)
    is_scanned, coverage = detect_text_layer(pdf)
    assert is_scanned is False
    assert coverage == 1.0


def test_mixed_document_below_threshold(tmp_path):
    # 4 text pages leading, 6 blank: 60% textless < 80% threshold.
    pdf = _pdf(tmp_path / "mixed.pdf", 4, 6)
    is_scanned, coverage = detect_text_layer(pdf)
    assert is_scanned is False
    assert 0.0 < coverage < 1.0


def test_sampling_spreads_beyond_first_pages(tmp_path):
    # Digital front matter (3 text pages) then a 60-page scan: a
    # first-pages-only sampler would call this digital.
    pdf = _pdf(tmp_path / "frontmatter.pdf", 3, 60)
    is_scanned, _ = detect_text_layer(pdf)
    assert is_scanned is True


def test_unreadable_pdf_fails_open(tmp_path):
    bogus = tmp_path / "not_a_pdf.pdf"
    bogus.write_bytes(b"hello")
    assert detect_text_layer(bogus) == (False, 1.0)


def test_classify_document_carries_scan_fields(tmp_path):
    pdf = _pdf(tmp_path / "scan.pdf", 0, 8)
    result = classify_document(pdf)
    assert result.is_scanned is True
    assert result.text_layer_coverage == 0.0
    digital = classify_document(_pdf(tmp_path / "d.pdf", 8, 0))
    assert digital.is_scanned is False


def test_pdf_info_reuses_the_same_detector(tmp_path):
    """pdf_info and the pipeline must share one source of truth."""
    import asyncio
    import json

    from pdf_capture_mcp.server import mcp

    pdf = _pdf(tmp_path / "scan.pdf", 0, 12)
    expected = detect_text_layer(pdf)

    async def _call():
        return await mcp.call_tool("pdf_info", {"pdf_path": str(pdf)})

    res = asyncio.run(_call())
    payload = json.loads(res.content[0].text)  # FastMCP ToolResult
    assert payload["ok"] is True
    assert (payload["is_scanned"], payload["text_layer_coverage"]) == expected
    assert payload["is_scanned"] is True
