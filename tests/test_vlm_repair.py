"""Tests for VLM arbitration (quality/vlm_repair.py).

call_vlm is monkeypatched throughout — these tests verify region location,
the numeric-conservation gate, HTML replacement, and figure-description
injection without network access or API tokens.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from pdf_capture_mcp.quality import vlm_repair  # noqa: E402
from pdf_capture_mcp.quality.md_audit import audit_markdown  # noqa: E402
from pdf_capture_mcp.quality.vlm_repair import (  # noqa: E402
    _numeric_gate,
    run_vlm_arbitration,
)

# ── Fixtures ────────────────────────────────────────────────────────────────

TORN_MD = """\
| Model | Batch Size | | | Learning | Rate |
|-------|-----------|---|---|----------|------|
| Small | 0.5M | 6 | 0 | x 10 | - 4 |
| Medium | 0.5M | 3 | 0 | x 10 | - 4 |
"""

GOOD_HTML = (
    "<table><thead><tr><th>Model</th><th>Batch Size</th><th>Learning Rate</th>"
    "</tr></thead><tbody>"
    "<tr><td>Small</td><td>0.5M</td><td>6.0 x 10<sup>-4</sup></td></tr>"
    "<tr><td>Medium</td><td>0.5M</td><td>3.0 x 10<sup>-4</sup></td></tr>"
    "</tbody></table>"
)


@pytest.fixture()
def table_pdf(tmp_path):
    pdf = tmp_path / "t.pdf"
    doc = fitz.open()
    page = doc.new_page()
    lines = [
        "Model Batch Size Learning Rate",
        "Small 0.5M 6.0x10-4",
        "Medium 0.5M 3.0x10-4",
        "Padding sentence so the page has enough surrounding text tokens.",
        "Another padding sentence with additional vocabulary for anchors.",
    ]
    for n, ln in enumerate(lines):
        page.insert_text((72, 72 + n * 18), ln)
    doc.save(str(pdf))
    doc.close()
    return pdf


@pytest.fixture(autouse=True)
def _vlm_enabled(monkeypatch):
    """Pretend VLM is configured; individual tests patch call_vlm."""
    import pdf_capture_mcp.llm_client as llm

    monkeypatch.setattr(llm, "is_vlm_enabled", lambda: True)
    yield


def _issues_for(md: str):
    return [i for i in audit_markdown(md) if i.rule in vlm_repair.TABLE_RULES]


# ── Numeric gate unit tests ─────────────────────────────────────────────────


def test_gate_passes_on_conserved_numbers():
    status, detail, _ = _numeric_gate(
        GOOD_HTML, "Small 0.5M 6.0x10-4 Medium 3.0x10-4 10 4", TORN_MD
    )
    assert status == "pass", detail


def test_gate_rejects_invented_numbers():
    bad = GOOD_HTML.replace("6.0", "7.77")
    status, detail, disputed = _numeric_gate(
        bad, "Small 0.5M 6.0x10-4 Medium 3.0x10-4 10 4", TORN_MD
    )
    # Tampering trips the gate either way: the fabricated 7.77 is invented
    # AND the original fragment 6 is lost — both are hard fails.
    assert status == "fail"


def test_gate_rejects_lost_numbers():
    # HTML drops the Medium row entirely -> md numbers '3' lost.
    partial = "<table><tr><td>Small</td><td>0.5M</td><td>6.0 x 10<sup>-4</sup></td></tr></table>"
    status, detail, _ = _numeric_gate(partial, "Small 0.5M 6.0x10-4 Medium 3.0x10-4 10 4", TORN_MD)
    assert status == "fail"
    assert "lost" in detail


# ── End-to-end arbitration with mocked VLM ──────────────────────────────────


def test_table_replaced_when_gate_passes(table_pdf, monkeypatch):
    monkeypatch.setattr(vlm_repair, "call_vlm", None, raising=False)
    import pdf_capture_mcp.llm_client as llm

    monkeypatch.setattr(llm, "call_vlm", lambda *a, **k: GOOD_HTML)

    issues = _issues_for(TORN_MD)
    assert issues, "fixture must trigger a table rule"
    result = run_vlm_arbitration(TORN_MD, table_pdf, issues, repair_tables=True)
    action = result["actions"][0]
    assert action.status == "repaired", action.description
    assert "<table>" in result["text"]
    assert "6.0" in result["text"]  # decimal restored
    # Broken markdown table is gone.
    assert "| 6 | 0 |" not in result["text"]


def test_table_kept_when_vlm_hallucinates(table_pdf, monkeypatch):
    import pdf_capture_mcp.llm_client as llm

    monkeypatch.setattr(llm, "call_vlm", lambda *a, **k: GOOD_HTML.replace("6.0", "9.99"))

    issues = _issues_for(TORN_MD)
    result = run_vlm_arbitration(TORN_MD, table_pdf, issues, repair_tables=True)
    action = result["actions"][0]
    assert action.status == "reported"
    assert "numeric gate" in action.description
    assert result["text"] == TORN_MD  # untouched


def test_no_table_element_reported(table_pdf, monkeypatch):
    import pdf_capture_mcp.llm_client as llm

    monkeypatch.setattr(llm, "call_vlm", lambda *a, **k: "sorry, cannot help")

    issues = _issues_for(TORN_MD)
    result = run_vlm_arbitration(TORN_MD, table_pdf, issues, repair_tables=True)
    assert result["actions"][0].status == "reported"
    assert result["modified"] is False


def test_disabled_vlm_is_noop(table_pdf, monkeypatch):
    import pdf_capture_mcp.llm_client as llm

    monkeypatch.setattr(llm, "is_vlm_enabled", lambda: False)
    issues = _issues_for(TORN_MD)
    result = run_vlm_arbitration(TORN_MD, table_pdf, issues, repair_tables=True)
    assert result["actions"] == [] and result["modified"] is False


# ── Figure description injection ────────────────────────────────────────────


def test_figure_description_injected(tmp_path, table_pdf, monkeypatch):
    import pdf_capture_mcp.llm_client as llm

    monkeypatch.setattr(
        llm,
        "call_vlm",
        lambda *a, **k: "Architecture diagram showing RouterNode dispatching to EncoderNode.",
    )
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "fig1.jpeg").write_bytes(b"jpegdata")
    md = "Intro paragraph.\n\n![](images/fig1.jpeg)\n\nNext paragraph.\n"

    result = run_vlm_arbitration(
        md, table_pdf, [], base_dir=tmp_path, repair_tables=False, describe_figures=True
    )
    lines = result["text"].splitlines()
    img_idx = lines.index("![](images/fig1.jpeg)")
    assert lines[img_idx + 1].startswith("> **Figure (VLM description):**")
    assert "RouterNode" in lines[img_idx + 1]
    # Idempotent: second run does not duplicate.
    again = run_vlm_arbitration(
        result["text"],
        table_pdf,
        [],
        base_dir=tmp_path,
        repair_tables=False,
        describe_figures=True,
    )
    assert again["modified"] is False


# ── v0.6.0: tri-state feature resolution & VLM policy ───────────────────


def test_tristate_explicit_bool_wins():
    from pdf_capture_mcp.server import _resolve_vlm_feature

    assert _resolve_vlm_feature(True, vlm_on=False, policy_allows=False) is True
    assert _resolve_vlm_feature(False, vlm_on=True, policy_allows=True) is False


def test_tristate_string_overrides():
    from pdf_capture_mcp.server import _resolve_vlm_feature

    assert _resolve_vlm_feature("on", vlm_on=False, policy_allows=False) is True
    assert _resolve_vlm_feature("off", vlm_on=True, policy_allows=True) is False
    assert _resolve_vlm_feature("TRUE", vlm_on=False, policy_allows=False) is True


def test_tristate_auto_follows_vlm_and_policy():
    from pdf_capture_mcp.server import _resolve_vlm_feature

    assert _resolve_vlm_feature("auto", vlm_on=True, policy_allows=True) is True
    assert _resolve_vlm_feature("auto", vlm_on=True, policy_allows=False) is False
    assert _resolve_vlm_feature("auto", vlm_on=False, policy_allows=True) is False


def test_vlm_policy_defaults_to_full_for_legacy_config(monkeypatch):
    import pdf_capture_mcp.llm_client as llm

    # Legacy config saved before the policy field existed.
    monkeypatch.setattr(llm, "_load_config", lambda: {"enabled": True, "model": "m"})
    assert llm.get_vlm_policy() == "full"
    # Corrupted value falls back safely.
    monkeypatch.setattr(llm, "_load_config", lambda: {"policy": "banana"})
    assert llm.get_vlm_policy() == "full"
    monkeypatch.setattr(llm, "_load_config", lambda: {"policy": "tables_only"})
    assert llm.get_vlm_policy() == "tables_only"


# ── v0.9.1: three-tier autonomous gate ──────────────────────────────────────


def test_gate_l1_dual_vision_baseline():
    """A number marker's OCR saw (md block) is never 'invented', even when
    the text layer lacks it (ligature corruption)."""
    html = "<table><tr><td>1928</td><td>2000</td></tr></table>"
    md_block = "| 1928 | 2000 |"  # marker read these from pixels
    corrupted_region = "fifl ffiff fifl"  # text layer garbage, no digits
    status, detail, _ = _numeric_gate(html, corrupted_region, md_block)
    assert status == "pass", detail


def test_gate_l2_escalates_to_verify_on_corrupted_layer():
    """Text layer covering <50% of marker's numbers is unfit to veto."""
    html = "<table><tr><td>1928</td><td>2000</td><td>77</td></tr></table>"
    md_block = "| 1928 | 2000 |"  # 77 is disputed (neither source has it)
    corrupted_region = "fifl ffiff"  # covers 0% of marker's numbers
    status, detail, disputed = _numeric_gate(html, corrupted_region, md_block)
    assert status == "verify"
    assert disputed == {"77"}


def test_gate_l2_healthy_layer_still_vetoes():
    """A healthy text layer keeps its veto power over invented numbers."""
    html = "<table><tr><td>1928</td><td>1967</td><td>1980</td><td>2000</td><td>77</td></tr></table>"
    md_block = "| 1928 | 1967 | 1980 | 2000 |"  # sample >= 4: ratio is meaningful
    healthy_region = "years 1928 1967 1980 2000 in text"  # covers 100%
    status, detail, _ = _numeric_gate(html, healthy_region, md_block)
    assert status == "fail"


def test_l3_self_verification_accepts_confirmed(table_pdf, monkeypatch):
    """Full path: corrupted layer -> verify -> VLM confirms -> repaired."""
    import pdf_capture_mcp.llm_client as llm

    calls = []

    def fake_vlm(prompt, img, **kw):
        calls.append(prompt)
        if "CONFIRMED" in prompt or "visible in" in prompt:
            return "CONFIRMED"
        return GOOD_HTML.replace("0.5M", "0.5M</td><td>777")  # adds disputed 777

    monkeypatch.setattr(llm, "call_vlm", fake_vlm)
    monkeypatch.setattr(vlm_repair, "_locate_table_region", lambda *a: (0, None, "fifl ffiff"))
    monkeypatch.setattr(vlm_repair, "_render_region", lambda *a, **k: "imgb64")
    issues = _issues_for(TORN_MD)
    result = run_vlm_arbitration(TORN_MD, table_pdf, issues, repair_tables=True)
    action = result["actions"][0]
    assert action.status == "repaired", action.description
    assert "L3 self-verification" in action.description
    assert len(calls) == 2  # transcription + verification round


def test_gate_l1_neighborhood_context_baseline():
    """Numbers marker placed NEAR the table (figure alt-text) count too."""
    html = "<table><tr><th>1912</th><th>2025</th></tr></table>"
    md_block = "| U.S. Steel | Nvidia |"  # no digits in the block itself
    ctx = "![A timeline chart from 1912 to 2025 showing firms]"
    status, detail, _ = _numeric_gate(html, "fifl garbage", md_block, md_context=ctx)
    assert status == "pass", detail


def test_gate_l2_small_sample_escalates():
    """Too few marker numbers -> coverage ratio is noise -> verify, not fail."""
    html = "<table><tr><td>1928</td><td>99</td></tr></table>"
    md_block = "| a | 1928 |"  # single number: sample too small to veto
    status, _, disputed = _numeric_gate(html, "healthy words 1928 here", md_block)
    assert status == "verify" and disputed == {"99"}
