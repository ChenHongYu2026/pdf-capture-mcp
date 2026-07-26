"""Tests for the v0.7.0 knowledge package: chunker, packaging, MD-110.

Each test encodes an audit finding (S/N/M numbers refer to the two design
audit rounds recorded in CHANGELOG 0.7.0).
"""

from __future__ import annotations

import json

import pytest

fitz = pytest.importorskip("fitz")

from pdf_capture_mcp.chunking.chunker import (  # noqa: E402
    Chunk,
    chunk_markdown,
    estimate_tokens,
    strip_frontmatter,
)
from pdf_capture_mcp.packaging import (  # noqa: E402
    assemble_package,
    build_frontmatter,
    export_package_to_vault,
    extract_summary,
    make_slug,
)
from pdf_capture_mcp.quality.cross_page_tables import (  # noqa: E402
    detect_cross_page_tables,
    merge_cross_page_tables,
)

DOC = "doc0123456789abcd"


# ── Token estimation (S7/N8) ────────────────────────────────────────────────


def test_token_estimate_cjk_vs_english():
    assert estimate_tokens("中文四个字") == 5
    assert estimate_tokens("abcdefgh") == 2  # 8 chars / 4


def test_token_estimate_strips_html_tags():
    html = "<table><tr><td>alpha</td><td>beta</td></tr></table>"
    assert estimate_tokens(html, strip_tags=True) < estimate_tokens(html)


# ── Frontmatter handling (N3) ───────────────────────────────────────────────


def test_strip_frontmatter():
    text = '---\ntitle: "x"\n---\n\n# Body\n'
    assert strip_frontmatter(text).startswith("# Body")
    assert strip_frontmatter("# No FM\n") == "# No FM\n"


# ── Chunking basics + heading paths (N11) ───────────────────────────────────


def test_heading_path_and_sibling_disambiguation():
    md = (
        "# Report\n\n## Balance Sheet\n\nAlpha section one.\n\n"
        "## Balance Sheet\n\nBeta section two.\n"
    )
    chunks = chunk_markdown(md, DOC)["chunks"]
    paths = [c.heading_path for c in chunks]
    assert ["Report", "Balance Sheet"] in paths
    assert ["Report", "Balance Sheet#2"] in paths  # N11


def test_duplicate_content_ids_do_not_collide():
    md = "# S\n\nSee Appendix A.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nSee Appendix A.\n"
    chunks = chunk_markdown(md, DOC)["chunks"]
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))  # N1


def test_table_embed_text_context_does_not_affect_id():
    """N2: editing upstream text must not change the table's chunk_id."""
    md1 = "# S\n\nTable 1: Results overview.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    md2 = "# S\n\nTable 1: DIFFERENT caption text.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    t1 = next(c for c in chunk_markdown(md1, DOC)["chunks"] if c.chunk_type == "table")
    t2 = next(c for c in chunk_markdown(md2, DOC)["chunks"] if c.chunk_type == "table")
    assert t1.chunk_id == t2.chunk_id  # content identical -> same id
    assert t1.embed_text != t2.embed_text  # context differs -> different embedding
    assert "Results overview" in t1.embed_text  # S2: caption present


def test_oversized_table_split_repeats_header():
    rows = "\n".join(f"| r{i} | {'x' * 60} |" for i in range(80))
    md = f"# S\n\n| name | data |\n|---|---|\n{rows}\n"
    tables = [c for c in chunk_markdown(md, DOC)["chunks"] if c.chunk_type == "table"]
    assert len(tables) > 1  # S1: split
    for t in tables:
        assert t.content.splitlines()[0].startswith("| name |")  # header repeated
        assert t.extra["table_parts"] == len(tables)


def test_bare_figure_folds_into_text_and_described_figure_is_chunk():
    md = (
        "# S\n\nIntro text.\n\n![](images/a.jpeg)\n\nMore text.\n\n"
        "![](images/b.jpeg)\n> **Figure (VLM description):** RouterNode diagram.\n"
    )
    chunks = chunk_markdown(md, DOC)["chunks"]
    fig = [c for c in chunks if c.chunk_type == "figure"]
    assert len(fig) == 1  # N6: only the described figure
    assert "RouterNode" in fig[0].embed_text
    text_all = " ".join(c.content for c in chunks if c.chunk_type == "text")
    assert "images/a.jpeg" in text_all  # bare link folded into text


def test_running_headers_dropped():
    page_header = "Annual Report 2026"
    parts = [f"# S{i}\n\n{page_header}\n\nBody paragraph {i}." for i in range(5)]
    result = chunk_markdown("\n\n".join(parts), DOC)
    assert page_header in result["dropped_headers"]  # N12
    assert all(page_header not in c.content for c in result["chunks"])


# ── Page anchoring (S4/M4) ──────────────────────────────────────────────────


@pytest.fixture()
def two_page_pdf(tmp_path):
    pdf = tmp_path / "p.pdf"
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 100), "alpha bravo charlie delta echo foxtrot")
    p1.insert_text((72, 130), "golf hotel india juliet kilo lima")
    p2 = doc.new_page()
    p2.insert_text((72, 100), "november oscar papa quebec romeo sierra")
    p2.insert_text((72, 130), "tango uniform victor whiskey xray yankee")
    doc.save(str(pdf))
    doc.close()
    return pdf


def test_page_anchoring_monotonic(two_page_pdf):
    md = (
        "# A\n\nalpha bravo charlie delta echo foxtrot golf hotel india.\n\n"
        "# B\n\nnovember oscar papa quebec romeo sierra tango uniform victor.\n"
    )
    chunks = chunk_markdown(md, DOC, pdf_path=two_page_pdf)["chunks"]
    assert chunks[0].page == 1
    assert chunks[1].page == 2
    assert chunks[0].extra["anchor"] == "matched"


# ── Packaging: naming standard ──────────────────────────────────────────────


def test_make_slug_rules():
    assert make_slug("A: B | C #1 [x]", DOC) == "A-B-C-1-x"
    assert make_slug("论文：模型/评估", DOC) == "论文-模型-评估"
    assert make_slug("", DOC) == DOC  # empty title falls back to identity
    assert make_slug("Paper", DOC, taken={"Paper"}) == f"Paper-{DOC[:8]}"
    assert len(make_slug("x" * 200, DOC)) <= 60


def test_extract_summary_chain():
    md_abs = "# T\n\n## Abstract\n\nThis paper shows things. It is long enough to count.\n"
    s, src = extract_summary(md_abs)
    assert src == "abstract" and "shows things" in s
    md_para = (
        "# T\n\n"
        + "This is a normal opening paragraph with several sentences in it. "
        + "It talks about the actual document content at reasonable length. "
        + "It clearly has enough length and structure to qualify as a summary.\n"
    )
    _, src2 = extract_summary(md_para)
    assert src2 == "first_paragraph"
    _, src3 = extract_summary("# T\n\n| a |\n|---|\n")
    assert src3 == "unavailable"  # N7: honest degradation


def test_frontmatter_yaml_quoting():
    fm = build_frontmatter(
        title='Bad: "title" | with #chars',
        doc_id=DOC,
        source_pdf="x.pdf",
        pages=3,
        qc_verdict="PASS",
        tool_version="0.7.0",
    )
    assert fm.startswith("---\n") and fm.rstrip().endswith("---")
    assert '\\"title\\"' in fm  # M3: quotes escaped


# ── Packaging: assembly + vault export ──────────────────────────────────────


def _mk_chunks(n=2):
    return [
        Chunk(
            chunk_id=f"c{i:015d}",
            doc_id=DOC,
            seq=i,
            heading_path=["S"],
            page=1,
            chunk_type="text",
            content=f"body {i}",
            embed_text=f"body {i}",
            token_est=2,
        )
        for i in range(n)
    ]


def _assemble(tmp_path, title="My Paper"):
    common = {
        "title": title,
        "doc_id": DOC,
        "source_pdf": "x.pdf",
        "pages": 2,
        "tool_version": "0.7.0",
        "summary": "Sum.",
        "summary_source": "abstract",
    }
    return assemble_package(
        output_root=tmp_path / "out",
        title=title,
        doc_id=DOC,
        markdown_text="# Body\n\nText.\n",
        frontmatter='---\ntitle: "t"\n---\n\n',
        images_dir=None,
        tables=[{"page": 3, "markdown": "| a | b |\n|---|---|\n| 1 | 2 |"}],
        chunks=_mk_chunks(),
        qc_report={"verdict": "PASS"},
        readme_kwargs={**common, "qc_verdict": "PASS"},
        metadata_kwargs={
            **common,
            "conversion_params": {},
            "heading_tree": [{"level": 1, "title": "Body"}],
            "dropped_headers": [],
        },
    )


def test_assemble_package_layout(tmp_path):
    info = _assemble(tmp_path)
    pkg = tmp_path / "out" / "My-Paper"
    assert (pkg / "My-Paper.md").read_text().startswith("---")  # frontmatter injected
    assert (pkg / "README.md").exists()
    assert (pkg / "data" / "chunks.jsonl").exists()
    assert (pkg / "tables" / "p3_table_1.csv").exists()  # N9: page-stamped
    meta = json.loads((pkg / "data" / "metadata.json").read_text())
    assert meta["doc_id"] == DOC and meta["content_hash"]  # N4
    assert meta["tables_csv"][0]["page"] == 3
    assert info["slug"] == "My-Paper"
    # README map mentions the actual main file name
    assert "My-Paper.md" in (pkg / "README.md").read_text()


def test_assemble_idempotent_same_doc(tmp_path):
    _assemble(tmp_path)
    info2 = _assemble(tmp_path)  # same doc_id: reuse folder, no -hash suffix
    assert info2["slug"] == "My-Paper"


def test_export_to_vault_idempotent(tmp_path):
    info = _assemble(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    r1 = export_package_to_vault(info["package_dir"], vault, category="Papers")
    assert r1["ok"] and not r1["skipped"]
    assert (vault / "Papers" / "My-Paper" / "My-Paper.md").exists()
    r2 = export_package_to_vault(info["package_dir"], vault, category="Papers")
    assert r2["skipped"]  # content_hash match


def test_export_rejects_non_package(tmp_path):
    (tmp_path / "junk").mkdir()
    r = export_package_to_vault(tmp_path / "junk", tmp_path)
    assert not r["ok"]


# ── MD-110: cross-page table merge (S3 three-evidence gate) ─────────────────


def _cross_page_pdf(tmp_path, split=True):
    """Two-page PDF: a table torn across the page break (or two separate)."""
    pdf = tmp_path / "t.pdf"
    doc = fitz.open()
    p1 = doc.new_page()  # height 842
    p1.insert_text((72, 80), "Intro words before the table start here")
    y = 780 if split else 400  # near bottom vs mid-page
    p1.insert_text((72, y), "name value")
    p1.insert_text((72, y + 15), "alpha 111")
    p2 = doc.new_page()
    y2 = 60 if split else 400
    if not split:
        p2.insert_text((72, 100), "Table 2: another table caption")
    p2.insert_text((72, y2), "beta 222")
    p2.insert_text((72, y2 + 15), "gamma 333")
    doc.save(str(pdf))
    doc.close()
    return pdf


MD_SPLIT = (
    "Intro words before the table start here\n\n"
    "| name | value |\n|---|---|\n| alpha | 111 |\n\n"
    "| name | value |\n|---|---|\n| beta | 222 |\n| gamma | 333 |\n"
)


def test_md110_merges_true_cross_page_table(tmp_path):
    pdf = _cross_page_pdf(tmp_path, split=True)
    issues = detect_cross_page_tables(MD_SPLIT, pdf_path=pdf)
    assert issues and issues[0].evidence["geo_pass"] is True
    lines = MD_SPLIT.splitlines()
    action = merge_cross_page_tables(lines, issues[0])
    assert action.status == "repaired"
    merged = "\n".join(lines)
    assert merged.count("| name | value |") == 1  # duplicate header removed
    assert "beta" in merged and "gamma" in merged  # no data loss


def test_md110_reports_separate_tables(tmp_path):
    pdf = _cross_page_pdf(tmp_path, split=False)  # mid-page + caption between
    issues = detect_cross_page_tables(MD_SPLIT, pdf_path=pdf)
    assert issues and issues[0].evidence["geo_pass"] is False  # S3 gate holds
    lines = MD_SPLIT.splitlines()
    action = merge_cross_page_tables(lines, issues[0])
    assert action.status == "reported"
    assert "\n".join(lines) == MD_SPLIT.rstrip("\n")  # untouched


def test_md110_ignores_different_column_counts():
    md = "| a | b |\n|---|---|\n| 1 | 2 |\n\n| x | y | z |\n|---|---|---|\n| 3 | 4 | 5 |\n"
    assert detect_cross_page_tables(md) == []


# ── v0.7.1 polish fixes ─────────────────────────────────────────────────────


def test_horizontal_rules_not_dropped_as_headers():
    """'---' thematic breaks are markdown structure, not running headers."""
    parts = [f"# S{i}\n\nBody {i}.\n\n---\n" for i in range(6)]
    result = chunk_markdown("\n".join(parts), DOC)
    assert "---" not in result["dropped_headers"]


def test_summary_ligature_echo_collapsed():
    from pdf_capture_mcp.packaging import _dedup_ligature_echo

    s = _dedup_ligature_echo("requires task-speciﬁc ﬁne-tuning task-specific fine-tuning datasets")
    assert s == "requires task-specific fine-tuning datasets"
    # Legitimate distant repetition is untouched
    t = "models are models of the world"
    assert _dedup_ligature_echo(t) == t
