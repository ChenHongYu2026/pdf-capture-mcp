"""Tests for the content-aware markdown audit (quality/md_audit.py).

Fixtures reproduce the defect patterns found in a real 75-page paper
conversion audit: control chars at in-cell word wraps, torn scientific
notation, header/data fusion, empty header rows, span placeholders.
"""

from __future__ import annotations

import pytest

from pdf_capture_mcp.quality.md_audit import (
    audit_markdown,
    check_content_coverage,
    run_markdown_audit,
    sanitize_markdown,
)

# ── MD-102: control characters (auto-fix) ───────────────────────────────────


def test_control_chars_removed_and_words_rejoined():
    text = "| Buddhism | 'En\x02lightenment', |\n|---|---|\n| a | b |"
    fixed, fixes = sanitize_markdown(text)
    assert "\x02" not in fixed
    assert "Enlightenment" in fixed  # word fragments rejoined
    assert any(f.rule == "MD-102" for f in fixes)
    assert fixes[0].severity == "critical"


def test_tab_and_newlines_survive_sanitize():
    text = "line1\nline2\twith tab\n"
    fixed, fixes = sanitize_markdown(text)
    assert fixed == text
    assert fixes == []


# ── MD-106: span placeholders (auto-fix) ────────────────────────────────────


def test_span_placeholders_removed():
    text = "| <span></span> | <span></span> |\n|---|---|\n| A | B |"
    fixed, fixes = sanitize_markdown(text)
    assert "<span></span>" not in fixed
    assert any(f.rule == "MD-106" for f in fixes)


# ── MD-101: garbled characters (detect only) ────────────────────────────────


def test_replacement_char_detected():
    issues = audit_markdown("normal text\nbroken \ufffd here\n")
    assert any(i.rule == "MD-101" and i.severity == "critical" for i in issues)


def test_private_use_area_detected():
    issues = audit_markdown("glyph \ue123 leaked\n")
    assert any(i.rule == "MD-101" for i in issues)


# ── MD-103: empty header row ────────────────────────────────────────────────


def test_empty_header_detected():
    text = "|  |  |  |\n|---|---|---|\n| Alice | Bob | Carol |"
    issues = audit_markdown(text)
    assert any(i.rule == "MD-103" and i.severity == "warn" for i in issues)


def test_normal_header_not_flagged():
    text = "| Name | Score |\n|---|---|\n| A | 1 |"
    issues = audit_markdown(text)
    assert not any(i.rule == "MD-103" for i in issues)


# ── MD-104: numeric column tearing ──────────────────────────────────────────

TORN_TABLE = """\
| Model | Batch Size | | | Learning | Rate |
|-------|-----------|---|---|----------|------|
| Small | 0.5M | 6 | 0 | × 10 | − 4 |
| Medium | 0.5M | 3 | 0 | × 10 | − 4 |
| Large | 0.5M | 2 | 5 | × 10 | − 4 |
"""


def test_torn_scientific_notation_detected():
    issues = audit_markdown(TORN_TABLE)
    hits = [i for i in issues if i.rule == "MD-104"]
    assert len(hits) == 1
    assert hits[0].severity == "critical"
    assert len(hits[0].lines) == 3  # all three torn rows located


def test_intact_scientific_notation_not_flagged():
    text = "| Model | Learning Rate |\n|---|---|\n| Small | 6.0 × 10−4 |\n| Medium | 3.0 × 10−4 |"
    issues = audit_markdown(text)
    assert not any(i.rule == "MD-104" for i in issues)


# ── MD-105: header fused with data row ──────────────────────────────────────


def test_header_fusion_detected():
    text = (
        "| Dataset Common Crawl | (filtered) 410 | Quantity billion | mix 60% | tokens 0.44 |\n"
        "|---|---|---|---|---|\n"
        "| WebText2 | 19 | billion | 22% | 2.9 |"
    )
    issues = audit_markdown(text)
    hits = [i for i in issues if i.rule == "MD-105"]
    assert len(hits) == 1
    assert hits[0].severity == "critical"


def test_headers_with_attached_digits_not_flagged():
    # 'F1', 'RACE-h', 'SQuADv2' — digits glued to letters are legitimate.
    text = "| CoQA F1 | RACE-h | SQuADv2 |\n|---|---|---|\n| 85.0 | 46.8 | 69.8 |"
    issues = audit_markdown(text)
    assert not any(i.rule == "MD-105" for i in issues)


# ── MD-201: content coverage ────────────────────────────────────────────────


@pytest.fixture()
def tiny_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "t.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Multiple short lines: insert_text does not wrap, and the coverage
    # check skips PDFs whose text layer is under 200 chars (scanned guard).
    lines = [
        "Quantum entanglement researchers Brown and Kaplan published results",
        "The experiment measured decoherence rates across arrays.",
        "Superconducting qubit devices were calibrated repeatedly.",
        "Measurement fidelity exceeded baseline expectations everywhere.",
        "Additional trials confirmed reproducibility of the findings.",
    ]
    for n, ln in enumerate(lines):
        page.insert_text((72, 72 + n * 18), ln)
    doc.save(str(pdf))
    doc.close()
    return pdf


def test_coverage_reports_missing_tokens(tiny_pdf):
    md_missing_authors = (
        "Quantum entanglement researchers published results. "
        "The experiment measured decoherence rates across arrays. "
        "Superconducting qubit devices were calibrated repeatedly. "
        "Measurement fidelity exceeded baseline expectations everywhere. "
        "Additional trials confirmed reproducibility of the findings."
    )
    issue = check_content_coverage(tiny_pdf, md_missing_authors)
    assert issue is not None
    assert issue.rule == "MD-201"
    assert "brown" in issue.suggestion.lower()
    assert "kaplan" in issue.suggestion.lower()


def test_coverage_silent_when_complete(tiny_pdf):
    md_full = (
        "Quantum entanglement researchers Brown and Kaplan published results. "
        "The experiment measured decoherence rates across arrays. "
        "Superconducting qubit devices were calibrated repeatedly. "
        "Measurement fidelity exceeded baseline expectations everywhere. "
        "Additional trials confirmed reproducibility of the findings."
    )
    assert check_content_coverage(tiny_pdf, md_full) is None


# ── Orchestrator ────────────────────────────────────────────────────────────


def test_run_markdown_audit_end_to_end():
    dirty = "text \x02 here\n" + TORN_TABLE
    result = run_markdown_audit(dirty, pdf_path=None, autofix=True)
    assert result["modified"] is True
    assert "\x02" not in result["text"]
    assert any(f.rule == "MD-102" for f in result["fixes"])
    assert any(i.rule == "MD-104" for i in result["issues"])
    assert result["counts"]["critical"] >= 1


def test_run_markdown_audit_no_autofix():
    dirty = "text \x02 here"
    result = run_markdown_audit(dirty, autofix=False)
    assert result["modified"] is False
    assert "\x02" in result["text"]


# ── v0.4.1 calibration: MD-107 flattened group header ───────────────────


def test_md107_flattened_group_header_detected():
    # Real pattern from a two-column paper: group names glued onto metric names.
    text = (
        "| Method | Exact Match | LoCoMo Temporal Answer F1 | Knowledge Substring EM "
        "| Update ROUGE-L F1 | LongMemEval Temporal Substring EM | Reasoning ROUGE-L F1 |\n"
        "|---|---|---|---|---|---|---|\n"
        "| Long Context | 8.1 | 26.9 | 20.0 | 18.0 | 12.0 | 24.0 |"
    )
    issues = audit_markdown(text)
    hits = [i for i in issues if i.rule == "MD-107"]
    assert len(hits) == 1
    assert hits[0].severity == "warn"
    assert "substring em" in hits[0].message or "rouge-l f1" in hits[0].message


def test_md107_normal_wide_header_not_flagged():
    # Distinct headers without repeated trailing metric bigrams must pass.
    text = (
        "| Method | Exact Match | Answer F1 | Substring EM | ROUGE-L F1 |\n"
        "|---|---|---|---|---|\n"
        "| Long Context | 8.1 | 26.9 | 20.0 | 18.0 |"
    )
    issues = audit_markdown(text)
    assert not any(i.rule == "MD-107" for i in issues)


# ── v0.4.1 calibration: MD-201 dehyphenation ───────────────────────────


@pytest.fixture()
def hyphenated_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "hyph.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Line-wrap hyphenation in the text layer: 'manage-' / 'ment'
    lines = [
        "The platform provides persistent storage plus reliable manage-",
        "ment for every workload under continuous operation today.",
        "Additional sentences ensure the text layer clears the guard.",
        "Operational dashboards summarize throughput and latency curves.",
    ]
    for n, ln in enumerate(lines):
        page.insert_text((72, 72 + n * 18), ln)
    doc.save(str(pdf))
    doc.close()
    return pdf


def test_md201_dehyphenation_no_false_deficit(hyphenated_pdf):
    # Markdown correctly merged the hyphenated word -> no deficit reported.
    md = (
        "The platform provides persistent storage plus reliable management "
        "for every workload under continuous operation today. "
        "Additional sentences ensure the text layer clears the guard. "
        "Operational dashboards summarize throughput and latency curves."
    )
    assert check_content_coverage(hyphenated_pdf, md) is None


# ── v0.4.1 calibration: MD-202 figure-text separation ───────────────────


@pytest.fixture()
def figure_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf = tmp_path / "fig.pdf"
    doc = fitz.open()
    page = doc.new_page()
    body_lines = [
        "The architecture diagram below illustrates the routing design.",
        "Body paragraphs continue with detailed explanations afterwards.",
        "Further discussion covers evaluation methodology and results.",
    ]
    for n, ln in enumerate(body_lines):
        page.insert_text((72, 72 + n * 18), ln)
    # Vector figure region (drawing) with embedded labels inside it.
    rect = fitz.Rect(72, 300, 400, 460)
    page.draw_rect(rect, color=(0, 0, 0), width=1)
    page.insert_text((90, 340), "RouterNode dispatches EncoderNode payloads")
    page.insert_text((90, 370), "StorageNode persists VectorIndex shards")
    doc.save(str(pdf))
    doc.close()
    return pdf


def test_md202_figure_text_not_billed_as_body_loss(figure_pdf):
    from pdf_capture_mcp.quality.md_audit import check_figure_text_omission

    # Markdown has the full body but none of the figure-embedded labels.
    md = (
        "The architecture diagram below illustrates the routing design. "
        "Body paragraphs continue with detailed explanations afterwards. "
        "Further discussion covers evaluation methodology and results."
    )
    body_issue = check_content_coverage(figure_pdf, md)
    # Figure labels must NOT surface as body loss (None or tiny info only).
    assert body_issue is None or body_issue.severity == "info"
    fig_issue = check_figure_text_omission(figure_pdf, md)
    assert fig_issue is not None
    assert fig_issue.rule == "MD-202"
    assert fig_issue.severity == "info"
    assert "routernode" in fig_issue.suggestion.lower()


# ── v0.4.1 calibration: single-letter formulas are legitimate ─────────────


def test_single_letter_formula_not_broken():
    from pdf_capture_mcp.quality.qc_gate import _assess_formula_integrity

    assert _assess_formula_integrity("inline $k$ and $M$ and $K$ math") == 1.0
    # Genuinely broken content still fails.
    assert _assess_formula_integrity("broken $?$ formula") == 0.0
    assert _assess_formula_integrity("broken $a???b$ formula") == 0.0


# ── v0.4.2: MD-108 math-delimiter collision ──────────────────────────


def test_md108_citation_links_deescaped():
    # Real marker citation shapes from two audited papers.
    text = (
        "word vectors [\\[MCCD13,](#page-71-0) [PSM14\\]](#page-72-0) and "
        "models [\\[VSP](#page-73-0)<sup>+</sup>17] here [\\[39\\]](#page-13-7)."
    )
    fixed, fixes = sanitize_markdown(text)
    md108 = [f for f in fixes if f.rule == "MD-108"]
    assert len(md108) == 1
    assert "\\[" not in fixed and "\\]](" not in fixed
    # Link structure survives: anchors still resolve.
    assert "[[MCCD13,](#page-71-0)" in fixed
    assert "[PSM14]](#page-72-0)" in fixed
    assert "[[39]](#page-13-7)" in fixed


def test_md108_genuine_display_math_untouched():
    # Standalone \[ ... \] display math is NOT a link shape - must survive.
    text = "Consider the equation\n\\[ x^2 + y^2 = z^2 \\]\nas shown above."
    fixed, fixes = sanitize_markdown(text)
    assert fixed == text
    assert not any(f.rule == "MD-108" for f in fixes)


def test_md108_residual_detection():
    # A shape the sanitizer does not rewrite, co-located with an anchor link.
    text = "weird \\[orphan escape with [link](#page-1-0) on the same line"
    issues = audit_markdown(text)
    hits = [i for i in issues if i.rule == "MD-108"]
    assert len(hits) == 1
    assert hits[0].severity == "warn"


def test_md108_plain_links_not_flagged():
    text = "normal [citation](#page-1-0) link without escapes"
    issues = audit_markdown(text)
    assert not any(i.rule == "MD-108" for i in issues)
