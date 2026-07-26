"""Regression corpus for cross-channel repair (quality/repair.py).

Each test synthesizes a golden PDF reproducing a defect class found in a
real 75-page paper audit, feeds the corresponding broken markdown through
the repairer, and asserts either a verified repair (with the ground truth
restored) or a clean report — never a silent guess.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from pdf_capture_mcp.quality.md_audit import audit_markdown  # noqa: E402
from pdf_capture_mcp.quality.repair import repair_markdown  # noqa: E402

# ── Golden PDF builders ─────────────────────────────────────────────────────


def _pdf_with_lines(tmp_path, lines, name="golden.pdf", positioned=False):
    """Create a one-page PDF; `lines` is list[str] or list[list[(x, str)]]."""
    pdf = tmp_path / name
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        if positioned:
            for x, word in line:
                page.insert_text((x, y), word)
        else:
            page.insert_text((72, y), line)
        y += 18
    doc.save(str(pdf))
    doc.close()
    return pdf


# ── MD-104: torn scientific notation ────────────────────────────────────────

TORN_MD = """\
| Model | Batch Size | | | Learning | Rate |
|-------|-----------|---|---|----------|------|
| Small | 0.5M | 6 | 0 | x 10 | - 4 |
| Medium | 0.5M | 3 | 0 | x 10 | - 4 |
| Large | 0.5M | 2 | 5 | x 10 | - 4 |
"""

TORN_PDF_LINES = [
    "Model Batch Size Learning Rate",
    "Small 0.5M 6.0x10-4",
    "Medium 0.5M 3.0x10-4",
    "Large 0.5M 2.5x10-4",
    # padding so the text layer clears the scanned-PDF guard elsewhere
    "The models were trained with the hyperparameters listed above.",
    "Additional configuration details are documented separately below.",
]


def test_md104_repair_restores_decimal_points(tmp_path):
    pdf = _pdf_with_lines(tmp_path, TORN_PDF_LINES)
    issues = [i for i in audit_markdown(TORN_MD) if i.rule == "MD-104"]
    assert issues, "fixture must trigger MD-104"

    result = repair_markdown(TORN_MD, pdf, issues)
    action = next(a for a in result["actions"] if a.rule == "MD-104")
    assert action.status == "repaired", action.description
    assert result["modified"] is True
    # Ground truth restored — decimal points are back.
    assert "6.0x10-4" in result["text"]
    assert "3.0x10-4" in result["text"]
    assert "2.5x10-4" in result["text"]
    # Defect is actually gone after repair.
    assert not any(i.rule == "MD-104" for i in audit_markdown(result["text"]))


def test_md104_gate_refuses_mismatched_source(tmp_path):
    # PDF values do NOT match the torn fragments -> gate must refuse.
    wrong = [
        ln.replace("6.0", "7.5").replace("3.0", "8.5").replace("2.5", "9.5")
        for ln in TORN_PDF_LINES
    ]
    pdf = _pdf_with_lines(tmp_path, wrong)
    issues = [i for i in audit_markdown(TORN_MD) if i.rule == "MD-104"]

    result = repair_markdown(TORN_MD, pdf, issues)
    action = next(a for a in result["actions"] if a.rule == "MD-104")
    assert action.status == "reported"
    assert result["text"] == TORN_MD  # untouched — never guess


# ── MD-105: header fused with data row ──────────────────────────────────────

FUSED_MD = """\
| Dataset CommonCrawl | Quantity 410 | Weight 60% |
|---|---|---|
| WebText2 | 19 | 22% |
| Books1 | 12 | 8% |
"""

# Column x-positions simulate the real table geometry.
FUSED_PDF_LINES = [
    [(72, "Dataset"), (220, "Quantity"), (340, "Weight")],
    [(72, "CommonCrawl"), (220, "410"), (340, "60%")],
    [(72, "WebText2"), (220, "19"), (340, "22%")],
    [(72, "Books1"), (220, "12"), (340, "8%")],
]


def test_md105_repair_rebuilds_table(tmp_path):
    pdf = _pdf_with_lines(
        tmp_path,
        FUSED_PDF_LINES + [[(72, "Filler text ensures the layer is long enough for checks.")]],
        positioned=True,
    )
    issues = [i for i in audit_markdown(FUSED_MD) if i.rule == "MD-105"]
    assert issues, "fixture must trigger MD-105"

    result = repair_markdown(FUSED_MD, pdf, issues)
    action = next(a for a in result["actions"] if a.rule == "MD-105")
    if action.status == "repaired":
        lines = result["text"].splitlines()
        header = lines[0]
        # Header no longer carries fused data values.
        assert "410" not in header and "60%" not in header
        # Data landed in a proper data row.
        assert any("CommonCrawl" in ln and "410" in ln for ln in lines[2:])
        # Defect cleared.
        assert not any(i.rule == "MD-105" for i in audit_markdown(result["text"]))
    else:
        # repair-or-report: a report must carry the recovered candidate.
        assert "Candidate" in action.evidence or "candidate" in action.evidence


# ── MD-201: content loss re-injection ───────────────────────────────────────


def _authors_pdf(tmp_path):
    return _pdf_with_lines(
        tmp_path,
        [
            "Tom Brown Benjamin Mann Nick Ryder Melanie Subbiah",
            "Language models learn tasks from very few demonstrations.",
            "Scaling laws determine compute-optimal training budgets today.",
            "Evaluation covers translation, question answering, and cloze tasks.",
            "Results indicate strong few-shot performance across benchmarks.",
        ],
    )


MD_MISSING_AUTHOR = """\
Benjamin Mann Nick Ryder Melanie Subbiah

Language models learn tasks from very few demonstrations.
Scaling laws determine compute-optimal training budgets today.
Evaluation covers translation, question answering, and cloze tasks.
Results indicate strong few-shot performance across benchmarks.
"""


def test_md201_reinjects_dropped_author(tmp_path):
    pdf = _authors_pdf(tmp_path)
    from pdf_capture_mcp.quality.md_audit import check_content_coverage

    issue = check_content_coverage(pdf, MD_MISSING_AUTHOR)
    assert issue is not None and issue.rule == "MD-201"

    result = repair_markdown(MD_MISSING_AUTHOR, pdf, [issue])
    action = next(a for a in result["actions"] if a.rule == "MD-201")
    assert action.status == "repaired", action.description
    # The dropped author is back, right before the anchor.
    assert "Brown" in result["text"]
    idx_brown = result["text"].find("Brown")
    idx_anchor = result["text"].find("Benjamin")
    assert 0 <= idx_brown < idx_anchor
    # No over-injection: coverage now silent.
    assert check_content_coverage(pdf, result["text"]) is None


def test_md201_rejects_bulk_loss(tmp_path):
    pdf = _authors_pdf(tmp_path)
    from pdf_capture_mcp.quality.md_audit import check_content_coverage

    # Markdown lost almost everything -> injection must refuse (>5% deficit).
    md_gutted = "Language models learn tasks."
    issue = check_content_coverage(pdf, md_gutted)
    assert issue is not None

    result = repair_markdown(md_gutted, pdf, [issue])
    action = next(a for a in result["actions"] if a.rule == "MD-201")
    assert action.status == "reported"
    assert result["text"] == md_gutted


# ── Robustness ──────────────────────────────────────────────────────────────


def test_repair_survives_unreadable_pdf(tmp_path):
    bogus = tmp_path / "nope.pdf"
    bogus.write_bytes(b"not a pdf")
    issues = [i for i in audit_markdown(TORN_MD) if i.rule == "MD-104"]
    result = repair_markdown(TORN_MD, bogus, issues)
    assert result["modified"] is False
    assert result["text"] == TORN_MD
