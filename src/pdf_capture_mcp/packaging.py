"""Knowledge package assembly: naming standard, layout, self-description.

Package layout (audited design, see CHANGELOG 0.7.0):

    <output_root>/<slug>/
        <slug>.md          main document, same name as dir -> Obsidian [[slug]]
        README.md          agent + human entry map (first thing any LLM reads)
        images/            relative refs, render natively in Obsidian
        tables/            per-page CSVs (independent extraction channel)
        data/
            metadata.json  doc-level metadata + manifest + content_hash (N4)
            chunks.jsonl   semantic chunks (the sole RAG data source)
            qc_report.json archived quality report

Naming standard:
- doc_id = sha256(pdf bytes)[:16] — content-addressed identity: renaming
  the file never changes it; re-imports deduplicate naturally.
- slug: NFC-normalized (N10 — macOS NFD vs git/cloud-sync duplicates),
  illegal chars stripped, spaces -> '-', <=60 chars, human-readable first;
  the doc_id suffix is added ONLY on collision.
- Timestamps go in metadata/frontmatter (ISO 8601 + offset), never in file
  names: re-runs are idempotent overwrites, not copy pileups.
- The whole package moves as a unit (N5): image refs stay relative to the
  package dir, so dropping the folder into an Obsidian vault just works.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import unicodedata
from pathlib import Path
from typing import Any

from pdf_capture_mcp.chunking.chunker import CHUNKS_SCHEMA_VERSION, Chunk
from pdf_capture_mcp.config import get_logger

logger = get_logger("packaging")

METADATA_SCHEMA_VERSION = "1"

# ASCII-illegal chars plus their fullwidth CJK variants — legal on some
# filesystems but break cross-platform sync and Obsidian linking (M3/N10).
_ILLEGAL = re.compile(r'[\\/:*?"<>|#^\[\]：？＊＂＜＞｜／＼【】]+')
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def compute_doc_id(pdf_path: Path | str) -> str:
    """Content-addressed document identity: sha256 of the PDF bytes."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def make_slug(title: str, doc_id: str, taken: set[str] | None = None) -> str:
    """Human-readable, filesystem-safe, cross-platform-stable folder name."""
    slug = unicodedata.normalize("NFC", title)
    slug = _ILLEGAL.sub(" ", slug)
    slug = re.sub(r"\s+", "-", slug.strip()).strip("-.")
    slug = slug[:60].rstrip("-.")
    if not slug:
        slug = doc_id  # title reduced to nothing -> fall back to identity
    if taken and slug in taken:
        slug = f"{slug}-{doc_id[:8]}"  # collision only: append identity
    return slug


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


# ── Summary extraction (N7: honest degradation chain) ───────────────────────


def _dedup_ligature_echo(text: str) -> str:
    """Collapse marker's ligature-echo artifact (v0.7.1 fix).

    The engine sometimes emits a phrase twice — once with typographic
    ligatures, once without: 'task-speciﬁc ﬁne-tuning task-specific
    fine-tuning'. After NFKC folding (ﬁ -> fi) the echo becomes an exact
    adjacent repeat, which we collapse. Only ADJACENT repeats of 2-6 word
    groups are removed — legitimate distant repetition is never touched.
    """
    text = unicodedata.normalize("NFKC", text)
    words = text.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        collapsed = False
        for n in range(6, 1, -1):
            if i + 2 * n <= len(words) and words[i : i + n] == words[i + n : i + 2 * n]:
                out.extend(words[i : i + n])
                i += 2 * n
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    return " ".join(out)


def extract_summary(markdown_text: str) -> tuple[str, str]:
    """Return (summary, source). Never fabricates: abstract -> first
    substantial paragraph -> honest 'unavailable'."""
    lines = markdown_text.splitlines()
    # 1) Abstract section
    for i, line in enumerate(lines):
        m = _HEADING.match(line.strip())
        if m and m.group(2).strip().lower() in ("abstract", "摘要"):
            para: list[str] = []
            for nxt in lines[i + 1 :]:
                s = nxt.strip()
                if _HEADING.match(s):
                    break
                if s:
                    para.append(s)
                elif para:
                    break
            if para:
                return _dedup_ligature_echo(" ".join(para))[:600], "abstract"

    # 2) First substantial paragraph (skip ToC-ish and boilerplate lines)
    def _qualify(para_lines: list[str]) -> str | None:
        text = " ".join(para_lines)
        if len(text) > 150 and text.count(".") + text.count("。") >= 2:
            return _dedup_ligature_echo(text)[:600]
        return None

    para = []
    for line in lines:
        s = line.strip()
        if _HEADING.match(s) or s.startswith(("|", "!", ">", "```", "<", "---")):
            para = []
            continue
        if s:
            para.append(s)
        elif para:
            if (text := _qualify(para)) is not None:
                return text, "first_paragraph"
            para = []
    if para and (text := _qualify(para)) is not None:  # document-final paragraph
        return text, "first_paragraph"
    return "(no reliable summary could be extracted)", "unavailable"


# ── Frontmatter (M3: everything quoted, injected LAST per N3) ───────────────


def _yq(value: str) -> str:
    """Quote a YAML scalar defensively."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_frontmatter(
    *,
    title: str,
    doc_id: str,
    source_pdf: str,
    pages: int,
    qc_verdict: str,
    tool_version: str,
    tags: list[str] | None = None,
) -> str:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    lines = [
        "---",
        f"title: {_yq(title)}",
        f"doc_id: {_yq(doc_id)}",
        f"source: {_yq(source_pdf)}",
        f"pages: {pages}",
        f"qc_verdict: {_yq(qc_verdict)}",
        f"converted: {_yq(ts)}",
        f"generator: {_yq('pdf-capture-mcp ' + tool_version)}",
    ]
    if tags:
        lines.append("tags:")
        lines += [f"  - {_yq(t)}" for t in tags]
    lines.append("---")
    return "\n".join(lines) + "\n\n"


# ── README (the agent map — one read explains the whole package) ────────────


def build_readme(
    *,
    title: str,
    doc_id: str,
    source_pdf: str,
    pages: int,
    qc_verdict: str,
    tool_version: str,
    summary: str,
    summary_source: str,
    main_md_name: str,
    n_chunks: int,
    n_tables_csv: int,
) -> str:
    return f"""---
title: {_yq(title + " — 知识包说明")}
doc_id: {_yq(doc_id)}
---

# {title} — PDF Capture 知识包

> pdf-capture-mcp v{tool_version} | 源: {source_pdf} ({pages} 页) | \
QC: {qc_verdict} | doc_id: `{doc_id}`

## 这是什么

{summary}

*(摘要来源: {summary_source})*

## 文件地图（读完本表再决定打开什么）

| 文件 | 用途 | 何时使用 |
|---|---|---|
| `{main_md_name}` | 完整正文（含表格/图/公式，QC 已审计修复） | 回答内容相关问题 |
| `data/chunks.jsonl` | {n_chunks} 个语义分块，含页码/章节路径元数据 | RAG 索引、引用定位 |
| `data/metadata.json` | 文档级元数据 + 标题树 + 产物清单(manifest) | 了解结构、校验完整性 |
| `data/qc_report.json` | 质量审计与修复记录（11+ 规则） | 评估内容可信度 |
| `images/` | 提取的图片（正文内相对引用） | 视觉内容 |
| `tables/` | 表格 CSV，独立提取通道（{n_tables_csv} 个，命名含页码） | 表格数据交叉验证 |

## chunks.jsonl 行 schema (v{CHUNKS_SCHEMA_VERSION})

```json
{{"chunk_id": "内容寻址ID", "doc_id": "...", "seq": 0,
 "heading_path": ["章", "节"], "page": 12,
 "chunk_type": "text|table|figure|code",
 "content": "原文(用于展示/重组)", "embed_text": "原文+语境(用于向量化)",
 "token_est": 480}}
```

注意: `content` 参与 chunk_id 哈希，`embed_text` 不参与 —— 上游文本编辑
不会改变未变块的身份。若手工编辑了 `{main_md_name}`，`data/` 下的分块
即过时（以 metadata.json 的 content_hash 校验为准）。
"""


# ── Metadata (M5: doc-level ONLY; chunk-level lives in chunks.jsonl) ────────


def build_metadata(
    *,
    doc_id: str,
    title: str,
    slug: str,
    source_pdf: str,
    pages: int,
    tool_version: str,
    conversion_params: dict[str, Any],
    summary: str,
    summary_source: str,
    heading_tree: list[dict[str, Any]],
    package_dir: Path,
    main_md_name: str,
    csv_mapping: list[dict[str, Any]],
    dropped_headers: list[str],
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for rel in [main_md_name, "README.md", "data/chunks.jsonl", "data/qc_report.json"]:
        p = package_dir / rel
        if p.exists():
            files.append({"path": rel, "sha256": file_sha256(p), "bytes": p.stat().st_size})
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "chunks_schema": CHUNKS_SCHEMA_VERSION,
        "doc_id": doc_id,
        "title": title,
        "slug": slug,
        "source_pdf": source_pdf,
        "pages": pages,
        "generator": f"pdf-capture-mcp {tool_version}",
        "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "conversion_params": conversion_params,
        "summary": summary,
        "summary_source": summary_source,
        "heading_tree": heading_tree,
        # N4: build_vector_index must verify this before indexing — a
        # hand-edited main md means chunks.jsonl is stale.
        "content_hash": file_sha256(package_dir / main_md_name),
        "manifest": files,
        "tables_csv": csv_mapping,  # N9: csv <-> page <-> nearest heading
        "dropped_running_headers": dropped_headers,
    }


def build_heading_tree(markdown_text: str) -> list[dict[str, Any]]:
    tree: list[dict[str, Any]] = []
    for line in markdown_text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            tree.append({"level": len(m.group(1)), "title": m.group(2).strip()})
    return tree


# ── Assembly ────────────────────────────────────────────────────────────────


def _md_table_to_rows(md: str) -> list[list[str]]:
    """Parse a markdown pipe table into CSV rows (separator row dropped)."""
    rows: list[list[str]] = []
    for line in md.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        if re.fullmatch(r"\|?[\s:|-]+\|?", s) and "-" in s:
            continue
        rows.append([c.strip() for c in s.strip("|").split("|")])
    return rows


def write_chunks_jsonl(chunks: list[Chunk], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def assemble_package(
    *,
    output_root: Path,
    title: str,
    doc_id: str,
    markdown_text: str,
    frontmatter: str,
    images_dir: Path | None,
    tables: list[dict[str, Any]],
    chunks: list[Chunk],
    qc_report: dict[str, Any],
    readme_kwargs: dict[str, Any],
    metadata_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Materialize the self-describing knowledge package. Idempotent:
    re-runs overwrite in place (identity = doc_id, layout = slug)."""
    output_root.mkdir(parents=True, exist_ok=True)
    taken = {p.name for p in output_root.iterdir() if p.is_dir()}
    # Same doc re-converted: reuse its existing folder instead of colliding
    for existing in list(taken):
        meta = output_root / existing / "data" / "metadata.json"
        if meta.exists():
            try:
                if json.loads(meta.read_text())["doc_id"] == doc_id:
                    taken.discard(existing)
            except Exception:  # noqa: BLE001 — unreadable metadata: treat as foreign
                pass
    slug = make_slug(title, doc_id, taken)
    pkg = output_root / slug
    (pkg / "data").mkdir(parents=True, exist_ok=True)

    main_md_name = f"{slug}.md"
    # N3: frontmatter is injected here, AFTER all QC/repair phases.
    (pkg / main_md_name).write_text(frontmatter + markdown_text, encoding="utf-8")

    if images_dir and images_dir.exists() and images_dir != pkg / "images":
        shutil.copytree(images_dir, pkg / "images", dirs_exist_ok=True)

    # N9: page-stamped CSV names + mapping (CSV regenerated from the
    # independent pdfplumber channel — cross-validation data source)
    csv_mapping: list[dict[str, Any]] = []
    if tables:
        import csv as _csv

        tdir = pkg / "tables"
        tdir.mkdir(exist_ok=True)
        for i, t in enumerate(tables):
            page = t.get("page", 0)
            rows = _md_table_to_rows(t.get("markdown", ""))
            if not rows:
                continue
            name = f"p{page}_table_{i + 1}.csv"
            with open(tdir / name, "w", newline="", encoding="utf-8") as f:
                _csv.writer(f).writerows(rows)
            csv_mapping.append({"file": f"tables/{name}", "page": page})

    write_chunks_jsonl(chunks, pkg / "data" / "chunks.jsonl")
    (pkg / "data" / "qc_report.json").write_text(
        json.dumps(qc_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = build_readme(
        main_md_name=main_md_name,
        n_chunks=len(chunks),
        n_tables_csv=len(csv_mapping),
        **readme_kwargs,
    )
    (pkg / "README.md").write_text(readme, encoding="utf-8")

    metadata = build_metadata(
        slug=slug,
        package_dir=pkg,
        main_md_name=main_md_name,
        csv_mapping=csv_mapping,
        **metadata_kwargs,
    )
    (pkg / "data" / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Knowledge package assembled: %s (%d chunks)", pkg, len(chunks))
    return {
        "package_dir": str(pkg),
        "slug": slug,
        "markdown_path": str(pkg / main_md_name),
        "readme_path": str(pkg / "README.md"),
        "chunks_path": str(pkg / "data" / "chunks.jsonl"),
        "metadata_path": str(pkg / "data" / "metadata.json"),
    }


def export_package_to_vault(
    package_dir: Path | str, vault_dir: Path | str, category: str = ""
) -> dict[str, Any]:
    """Copy a knowledge package into an Obsidian vault as a whole unit.

    N5 contract: the package is never flattened and image refs are never
    rewritten — the folder IS the namespace. Idempotent by content_hash.
    """
    src = Path(package_dir)
    if not (src / "data" / "metadata.json").exists():
        return {"ok": False, "error": f"Not a knowledge package (no data/metadata.json): {src}"}
    vault = Path(vault_dir)
    if not vault.is_dir():
        return {"ok": False, "error": f"Vault directory does not exist: {vault}"}
    dest_parent = vault / category if category else vault
    dest_parent.mkdir(parents=True, exist_ok=True)
    dest = dest_parent / src.name

    if dest.exists():
        try:
            old = json.loads((dest / "data" / "metadata.json").read_text())["content_hash"]
            new = json.loads((src / "data" / "metadata.json").read_text())["content_hash"]
            if old == new:
                return {
                    "ok": True,
                    "dest": str(dest),
                    "skipped": True,
                    "message": "Identical content already in vault (content_hash match).",
                }
        except Exception:  # noqa: BLE001 — unreadable: fall through to overwrite
            pass
    shutil.copytree(src, dest, dirs_exist_ok=True)
    return {"ok": True, "dest": str(dest), "skipped": False}
