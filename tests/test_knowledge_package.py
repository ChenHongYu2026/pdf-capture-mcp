"""Tests for the v0.7.0 knowledge package: chunker, packaging, MD-110.

Each test encodes an audit finding (S/N/M numbers refer to the two design
audit rounds recorded in CHANGELOG 0.7.0).
"""

from __future__ import annotations

import json
from pathlib import Path

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


# ── Micro-chunk merge (N13) ─────────────────────────────────────────────────


def test_micro_text_chunk_merges_into_previous_sibling():
    big = "word " * 420  # ~525 tokens: forces a budget flush before the tail
    md = f"# S\n\n{big.strip()}\n\nTiny tail.\n"
    chunks = chunk_markdown(md, DOC)["chunks"]
    assert len(chunks) == 1  # N13: the tail folded back into its sibling
    assert "Tiny tail." in chunks[0].content
    assert chunks[0].embed_text == chunks[0].content


def test_micro_chunk_never_crosses_heading():
    md = "# Report\n\n## A\n\nShort a.\n\n## B\n\nShort b.\n"
    chunks = chunk_markdown(md, DOC)["chunks"]
    assert len(chunks) == 2  # different heading_path -> no merge
    assert [c.heading_path for c in chunks] == [["Report", "A"], ["Report", "B"]]
    assert all(c.extra.get("micro") for c in chunks)  # honest noise tag


def test_isolated_bare_image_survives_tagged():
    md = "# S\n\n![](images/a.jpeg)\n\n# T\n\nReal body text for the next section.\n"
    chunks = chunk_markdown(md, DOC)["chunks"]
    img = next(c for c in chunks if "images/a.jpeg" in c.content)
    assert img.chunk_type == "text"  # N6 fold had no neighbour...
    assert img.extra.get("micro") is True  # ...so N13 tags it as noise


# ── Footnote demotion (N14) ─────────────────────────────────────────────────


def test_numeric_h4_demoted_to_footnote_text():
    md = (
        "# Intro\n\nBody paragraph with actual content.\n\n"
        "#### 3\n\nSee the methodology for details.\n"
    )
    chunks = chunk_markdown(md, DOC)["chunks"]
    for c in chunks:
        assert "3" not in c.heading_path  # no fake ['Intro', '3'] section
    joined = " ".join(c.content for c in chunks)
    assert "3. See the methodology for details." in joined  # note survives


def test_low_level_numeric_heading_is_not_demoted():
    md = "# A\n\n## 2\n\nSection two body text here.\n"
    chunks = chunk_markdown(md, DOC)["chunks"]
    assert ["A", "2"] in [c.heading_path for c in chunks]  # level < 4: real section


# ── Caption context: <sup> footnotes excluded (S2 refinement) ───────────────


def test_sup_footnote_line_not_used_as_table_caption():
    md = (
        "# S\n\nIntro paragraph with some real content here.\n\n"
        "<sup>(--)</sup> Additional country data note.\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n"
    )
    chunks = chunk_markdown(md, DOC)["chunks"]
    table = next(c for c in chunks if c.chunk_type == "table")
    assert "Additional country data" not in table.embed_text  # not a caption
    text_all = " ".join(c.content for c in chunks if c.chunk_type == "text")
    assert "Additional country data" in text_all  # body text is untouched


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


# ── v0.9.0 polish ───────────────────────────────────────────────────────────


def test_heading_path_strips_span_anchors():
    """marker's <span id=...></span> anchors must not pollute heading_path."""
    md = '# <span id="page-3-0"></span>Approach\n\nBody text for the approach section goes here.\n'
    chunks = chunk_markdown(md, DOC)["chunks"]
    assert chunks[0].heading_path == ["Approach"]


def test_heading_tree_strips_span_anchors():
    from pdf_capture_mcp.packaging import build_heading_tree

    tree = build_heading_tree('## <span id="page-5-1"></span>3.1 Results\n')
    assert tree[0]["title"] == "3.1 Results"


def test_locate_table_region_ignores_stray_hits(tmp_path):
    """v0.9.0: densest y-band keeps the crop tight around the real table."""
    from pdf_capture_mcp.quality.vlm_repair import _locate_table_region

    pdf = tmp_path / "s.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # Stray hit far above the table (citation mentioning a table value)
    page.insert_text((72, 80), "as shown by value 111 in prior work")
    # The actual table block, vertically contiguous near the bottom
    for k, row in enumerate(["name value", "alpha 111", "beta 222", "gamma 333", "delta 444"]):
        page.insert_text((72, 600 + k * 14), row)
    doc.save(str(pdf))
    doc.close()

    lines = [
        "| name | value |",
        "|---|---|",
        "| alpha | 111 |",
        "| beta | 222 |",
        "| gamma | 333 |",
        "| delta | 444 |",
    ]
    located = _locate_table_region(pdf, lines)
    assert located is not None
    _, rect, _ = located
    assert rect.y0 > 400  # stray hit at y=80 excluded from the crop


def test_md201_exempts_running_furniture(tmp_path):
    """v0.9.1: nav bars/footers repeating across pages are exempt from the
    coverage deficit — dropping them is a cleanup victory, not content loss."""
    from pdf_capture_mcp.quality.md_audit import check_content_coverage

    pdf = tmp_path / "mag.pdf"
    doc = fitz.open()
    bodies = []
    for i in range(6):
        page = doc.new_page()
        body = f"Unique body paragraph number {i} with meaningful editorial content here."
        page.insert_text((72, 200), body)
        page.insert_text((72, 800), "Chapter navigation bar repeated on every page")
        bodies.append(body)
    doc.save(str(pdf))
    doc.close()

    md = "\n\n".join(bodies) + "\n"  # engine dropped the nav bar (correctly)
    issue = check_content_coverage(pdf, md)
    # All missing tokens are furniture -> exempt -> no issue (or info-level dust)
    assert issue is None or issue.severity == "info", issue and issue.message


# ── v0.9.5: metadata contract + vault conflict protection ───────────────────


def test_metadata_records_degraded_segments(tmp_path):
    common = {
        "title": "DS",
        "doc_id": DOC,
        "source_pdf": "x.pdf",
        "pages": 2,
        "tool_version": "0.9.5",
        "summary": "Sum.",
        "summary_source": "abstract",
    }
    info = assemble_package(
        output_root=tmp_path / "out",
        title="DS",
        doc_id=DOC,
        markdown_text="# B\n\nText.\n",
        frontmatter='---\ntitle: "t"\n---\n\n',
        images_dir=None,
        tables=[],
        chunks=_mk_chunks(),
        qc_report={"verdict": "WARN"},
        readme_kwargs={**common, "qc_verdict": "WARN"},
        metadata_kwargs={
            **common,
            "conversion_params": {},
            "heading_tree": [],
            "dropped_headers": [],
            "degraded_segments": [2, 3],
        },
    )
    meta = json.loads(Path(info["metadata_path"]).read_text())
    assert meta["degraded_segments"] == [2, 3]


def test_vault_conflict_blocks_overwrite_by_default(tmp_path):
    """v0.9.5: a diverged vault copy is refused, not clobbered."""
    info = _assemble(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    r1 = export_package_to_vault(info["package_dir"], vault)
    assert r1["ok"]
    # user hand-edits the vault copy -> content_hash diverges
    dest_meta = Path(r1["dest"]) / "data" / "metadata.json"
    meta = json.loads(dest_meta.read_text())
    meta["content_hash"] = "user-edited-divergence"
    dest_meta.write_text(json.dumps(meta))

    r2 = export_package_to_vault(info["package_dir"], vault)
    assert not r2["ok"] and r2["conflict"]
    r3 = export_package_to_vault(info["package_dir"], vault, overwrite=True)
    assert r3["ok"] and not r3.get("skipped")


def test_vault_unreadable_metadata_is_conflict(tmp_path):
    info = _assemble(tmp_path)
    vault = tmp_path / "vault"
    vault.mkdir()
    r1 = export_package_to_vault(info["package_dir"], vault)
    (Path(r1["dest"]) / "data" / "metadata.json").write_text("{corrupt")
    r2 = export_package_to_vault(info["package_dir"], vault)
    assert not r2["ok"] and r2["conflict"]


# ── v0.12.0: vector index as standard equipment ─────────────────────────────


def _mini_pdf(path):
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    for row in range(8):
        page.insert_text((36, 60 + 16 * row), "auto index doctrine body text row")
    doc.save(str(path))
    doc.close()
    return path


def test_auto_index_runs_when_embedding_configured(tmp_path, monkeypatch):
    """Configured embedding + packaged conversion -> index phase fires."""
    from pdf_capture_mcp import server

    calls: list[str] = []
    monkeypatch.setattr("pdf_capture_mcp.embedding_client.is_embedding_enabled", lambda: True)
    import pdf_capture_mcp.rag_store as rag_store

    monkeypatch.setattr(
        rag_store,
        "build_vector_index",
        lambda pkg: calls.append(str(pkg)) or {"ok": True, "embedded": 7},
    )
    result = server._run_pipeline(
        _mini_pdf(tmp_path / "t.pdf"),
        engine="pymupdf",
        enable_formula=False,
        out_dir=str(tmp_path / "out"),
        package=True,
    )
    assert result["ok"]
    assert len(calls) == 1 and calls[0] == result["package"]["package_dir"]
    assert result["index"] == {"ok": True, "embedded": 7}
    assert result["features"]["vector_index"]["enabled"] is True


def test_auto_index_skipped_without_embedding(tmp_path, monkeypatch):
    from pdf_capture_mcp import server

    monkeypatch.setattr("pdf_capture_mcp.embedding_client.is_embedding_enabled", lambda: False)
    import pdf_capture_mcp.rag_store as rag_store

    monkeypatch.setattr(
        rag_store, "build_vector_index", lambda pkg: (_ for _ in ()).throw(AssertionError)
    )
    result = server._run_pipeline(
        _mini_pdf(tmp_path / "t.pdf"),
        engine="pymupdf",
        enable_formula=False,
        out_dir=str(tmp_path / "out"),
        package=True,
    )
    assert result["ok"] and result["index"] == {}
    assert result["features"]["vector_index"]["enabled"] is False
    assert "setup_embedding" in result["features"]["vector_index"]["reason"]


def test_index_failure_never_kills_conversion(tmp_path, monkeypatch):
    from pdf_capture_mcp import server

    monkeypatch.setattr("pdf_capture_mcp.embedding_client.is_embedding_enabled", lambda: True)
    import pdf_capture_mcp.rag_store as rag_store

    def _boom(pkg):
        raise RuntimeError("embedding API down")

    monkeypatch.setattr(rag_store, "build_vector_index", _boom)
    result = server._run_pipeline(
        _mini_pdf(tmp_path / "t.pdf"),
        engine="pymupdf",
        enable_formula=False,
        out_dir=str(tmp_path / "out"),
        package=True,
    )
    assert result["ok"] is True  # conversion survives
    assert result["index"]["ok"] is False
    assert "embedding API down" in result["index"]["error"]


def test_explicit_off_wins_over_configured_embedding(tmp_path, monkeypatch):
    from pdf_capture_mcp import server

    monkeypatch.setattr("pdf_capture_mcp.embedding_client.is_embedding_enabled", lambda: True)
    import pdf_capture_mcp.rag_store as rag_store

    monkeypatch.setattr(
        rag_store, "build_vector_index", lambda pkg: (_ for _ in ()).throw(AssertionError)
    )
    result = server._run_pipeline(
        _mini_pdf(tmp_path / "t.pdf"),
        engine="pymupdf",
        enable_formula=False,
        out_dir=str(tmp_path / "out"),
        package=True,
        index="off",
    )
    assert result["ok"] and result["index"] == {}
    assert result["features"]["vector_index"]["enabled"] is False
