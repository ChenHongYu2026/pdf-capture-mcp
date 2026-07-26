"""Hierarchical chunker: audited markdown -> semantic chunks with metadata.

Design contract (two audit rounds, see CHANGELOG 0.7.0):

- Chunks follow the heading hierarchy; tables/code split off as dedicated
  chunks; figures only when they carry a VLM description (N6 — a bare image
  link embeds as pure noise).
- ``content`` is the verbatim block (hashed for chunk_id); ``embed_text``
  adds retrieval context (caption, heading path) WITHOUT affecting the id
  (N2 — upstream edits must not re-embed unchanged tables).
- chunk_id is content-addressed: sha1(doc_id | heading_path | dup_index |
  content). The dup_index disambiguates identical blocks under the same
  heading (N1) while keeping ids position-independent (S6).
- Token estimation is dependency-free (S7 — tiktoken downloads its encoder
  from a CDN that is unreachable in some regions): CJK chars count as one
  token, everything else ~4 chars/token; HTML tags are stripped first for
  tables (N8).
- Page anchoring scans the PDF text layer monotonically (S4 — headers
  repeat on every page, independent matching mis-anchors short chunks);
  scanned PDFs degrade to page=None, never a guess (M4).
- Repeated short lines (running headers that survived layout cleaning) are
  dropped and reported (N12).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pdf_capture_mcp.config import get_logger

logger = get_logger("chunking")

CHUNKS_SCHEMA_VERSION = "1"

DEFAULT_BUDGET = 512  # target tokens per chunk (M1: conservative default)
DEFAULT_HARD_CAP = 768  # never exceed; oversized blocks are split

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_PIPE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_PIPE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_FIGURE = re.compile(r"^!\[[^\]]*\]\([^)]+\)\s*$")
_FIG_DESC = re.compile(r"^>\s*\*\*Figure")
_TAG = re.compile(r"<[^>]+>")
_TOKEN = re.compile(r"[A-Za-z0-9]{4,}")
_CAPTION = re.compile(r"^(Table|Figure|表|图)\s*[\d.::]+", re.IGNORECASE)


@dataclass
class Chunk:
    """One semantic unit destined for the vector store and chunks.jsonl."""

    chunk_id: str
    doc_id: str
    seq: int
    heading_path: list[str]
    page: int | None
    chunk_type: str  # text | table | figure | code
    content: str
    embed_text: str
    token_est: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "seq": self.seq,
            "heading_path": self.heading_path,
            "page": self.page,
            "chunk_type": self.chunk_type,
            "content": self.content,
            "embed_text": self.embed_text,
            "token_est": self.token_est,
        }
        if self.extra:
            d["extra"] = self.extra
        return d


# ── Token estimation (S7/N8: zero-dependency, region-safe) ──────────────────


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3000 <= code <= 0x30FF  # CJK punctuation + kana
        or 0xF900 <= code <= 0xFAFF
    )


def estimate_tokens(text: str, *, strip_tags: bool = False) -> int:
    """Heuristic token count: CJK chars ~1 token, other text ~4 chars/token.

    strip_tags removes HTML markup first (N8: tag overhead inflates table
    estimates 2-3x and causes needless splitting).
    """
    if strip_tags:
        text = _TAG.sub(" ", text)
    cjk = sum(1 for c in text if _is_cjk(c))
    other = len(text) - cjk
    return cjk + max(0, other) // 4


# ── Frontmatter / block parsing ─────────────────────────────────────────────


def strip_frontmatter(text: str) -> str:
    """Remove a leading YAML frontmatter block (N3: chunk the body only)."""
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---", 4)
    if end == -1:
        return text
    return text[end + 4 :].lstrip("\n")


@dataclass
class _Block:
    kind: str  # heading | text | table | code | figure
    lines: list[str]
    level: int = 0  # headings only

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def _parse_blocks(lines: list[str]) -> list[_Block]:
    """Split markdown lines into typed blocks."""
    blocks: list[_Block] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        m = _HEADING.match(stripped)
        if m:
            # Strip inline HTML (marker's `<span id="page-N"></span>` anchors)
            # from heading METADATA — the main md keeps them for links, but
            # heading_path must stay citation-clean (v0.9.0 fix).
            title = _TAG.sub("", m.group(2)).strip()
            blocks.append(_Block("heading", [title], level=len(m.group(1))))
            i += 1
            continue
        if stripped.startswith("```"):
            j = i + 1
            while j < n and not lines[j].strip().startswith("```"):
                j += 1
            blocks.append(_Block("code", lines[i : min(j + 1, n)]))
            i = j + 1
            continue
        if stripped.startswith("<table"):
            j = i
            while j < n and "</table>" not in lines[j]:
                j += 1
            blocks.append(_Block("table", lines[i : min(j + 1, n)]))
            i = j + 1
            continue
        if _PIPE_ROW.match(stripped):
            j = i
            while j < n and _PIPE_ROW.match(lines[j].strip()):
                j += 1
            blocks.append(_Block("table", lines[i:j]))
            i = j
            continue
        if _FIGURE.match(stripped):
            fig_lines = [line]
            j = i + 1
            if j < n and _FIG_DESC.match(lines[j].strip()):
                fig_lines.append(lines[j])
                j += 1
            blocks.append(_Block("figure", fig_lines))
            i = j
            continue
        # Paragraph: accumulate until blank line or structural marker
        j = i
        para: list[str] = []
        while j < n:
            s = lines[j].strip()
            if (
                not s
                or _HEADING.match(s)
                or s.startswith(("```", "<table"))
                or _PIPE_ROW.match(s)
                or _FIGURE.match(s)
            ):
                break
            para.append(lines[j])
            j += 1
        blocks.append(_Block("text", para))
        i = j
    return blocks


def _drop_running_headers(blocks: list[_Block], min_repeats: int = 4) -> list[str]:
    """Remove short lines repeating across the document (N12).

    Returns the list of dropped line texts (reported in qc/extra).
    """
    counts: dict[str, int] = {}
    for b in blocks:
        if b.kind == "text":
            for ln in b.lines:
                s = ln.strip()
                # Header candidates must carry actual content — pure
                # punctuation lines ('---' rules, '***') are markdown
                # structure, not running headers (v0.7.1 fix).
                if 0 < len(s) < 60 and re.search(r"[A-Za-z0-9\u4e00-\u9fff]", s):
                    counts[s] = counts.get(s, 0) + 1
    repeated = {s for s, c in counts.items() if c >= min_repeats}
    if not repeated:
        return []
    dropped: list[str] = []
    for b in blocks:
        if b.kind == "text":
            kept = []
            for ln in b.lines:
                if ln.strip() in repeated:
                    dropped.append(ln.strip())
                else:
                    kept.append(ln)
            b.lines = kept
    return dropped


# ── Heading paths (N11: disambiguate same-named siblings) ───────────────────


class _HeadingTracker:
    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []
        self._sibling_seen: dict[tuple[str, ...], dict[str, int]] = {}

    def push(self, level: int, title: str) -> None:
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        parent = tuple(t for _, t in self._stack)
        seen = self._sibling_seen.setdefault(parent, {})
        seen[title] = seen.get(title, 0) + 1
        if seen[title] > 1:
            title = f"{title}#{seen[title]}"
        self._stack.append((level, title))

    @property
    def path(self) -> list[str]:
        return [t for _, t in self._stack]


# ── Table splitting (S1: oversized tables, repeat the header) ───────────────


def _split_pipe_table(lines: list[str], hard_cap: int) -> list[list[str]]:
    header: list[str] = []
    body_start = 0
    if len(lines) >= 2 and _PIPE_SEP.match(lines[1].strip()):
        header = lines[:2]
        body_start = 2
    parts: list[list[str]] = []
    current = list(header)
    for row in lines[body_start:]:
        current.append(row)
        if estimate_tokens("\n".join(current)) >= hard_cap:
            parts.append(current)
            current = list(header)
    if len(current) > len(header):
        parts.append(current)
    return parts or [lines]


def _split_html_table(text: str, hard_cap: int) -> list[str]:
    m = re.search(r"(<table[^>]*>)(.*?)(</table>)", text, flags=re.S)
    if not m:
        return [text]
    open_tag, inner, close_tag = m.groups()
    thead_m = re.search(r"<thead>.*?</thead>", inner, flags=re.S)
    thead = thead_m.group(0) if thead_m else ""
    body = inner.replace(thead, "", 1)
    rows = re.findall(r"<tr>.*?</tr>", body, flags=re.S)
    if not rows:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    for row in rows:
        current.append(row)
        candidate = open_tag + thead + "<tbody>" + "".join(current) + "</tbody>" + close_tag
        if estimate_tokens(candidate, strip_tags=True) >= hard_cap:
            parts.append(candidate)
            current = []
    if current:
        parts.append(open_tag + thead + "<tbody>" + "".join(current) + "</tbody>" + close_tag)
    return parts or [text]


def _split_table_block(block_text: str, hard_cap: int) -> list[str]:
    if estimate_tokens(block_text, strip_tags=True) < hard_cap:
        return [block_text]
    if block_text.lstrip().startswith("<table"):
        return _split_html_table(block_text, hard_cap)
    return ["\n".join(p) for p in _split_pipe_table(block_text.splitlines(), hard_cap)]


def _split_long_text(text: str, hard_cap: int) -> list[str]:
    """Split an oversized paragraph run at sentence-ish boundaries."""
    pieces = re.split(r"(?<=[.。!?！？])\s+", text)
    parts: list[str] = []
    current = ""
    for piece in pieces:
        candidate = (current + " " + piece).strip()
        if current and estimate_tokens(candidate) > hard_cap:
            parts.append(current)
            current = piece
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts or [text]


# ── chunk_id (S6 + N1) ──────────────────────────────────────────────────────


def _make_chunk_id(doc_id: str, heading_path: list[str], content: str, dup_index: int) -> str:
    key = f"{doc_id}|{'/'.join(heading_path)}|{dup_index}|{content}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# ── Page anchoring (S4 monotonic scan, M4 honest degradation) ───────────────


def _page_token_sets(pdf_path: Path | str) -> list[set[str]]:
    import fitz

    pages: list[set[str]] = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pages.append({t.lower() for t in _TOKEN.findall(page.get_text())})
    return pages


def anchor_pages(chunks: list[Chunk], pdf_path: Path | str, window: int = 6) -> None:
    """Assign 1-based page numbers via a monotonic scan of the text layer.

    Chunk order and page order must both be monotonic, so each chunk only
    searches forward from the previous anchor (S4). Low-confidence text
    chunks inherit the previous page (marked in extra); table/figure chunks
    stay None rather than guess (M4).
    """
    try:
        pages = _page_token_sets(pdf_path)
    except Exception as exc:  # noqa: BLE001 — anchoring is best-effort
        logger.warning("Page anchoring unavailable: %s", exc)
        return
    if not pages or all(len(p) < 10 for p in pages):
        return  # scanned PDF without a text layer: keep page=None (M4)

    pointer = 0
    prev_page: int | None = None
    for chunk in chunks:
        tokens = {t.lower() for t in _TOKEN.findall(_TAG.sub(" ", chunk.content))}
        if not tokens:
            chunk.page = prev_page
            chunk.extra["anchor"] = "inherited"
            continue
        best_q, best_score = pointer, 0
        for q in range(pointer, min(pointer + window, len(pages))):
            score = len(tokens & pages[q])
            if score > best_score:
                best_q, best_score = q, score
        threshold = max(3, int(0.2 * len(tokens)))
        if best_score >= threshold:
            chunk.page = best_q + 1
            chunk.extra["anchor"] = "matched"
            pointer = best_q
            prev_page = chunk.page
        elif chunk.chunk_type == "text" and prev_page is not None:
            chunk.page = prev_page
            chunk.extra["anchor"] = "inherited"
        else:
            chunk.page = None
            chunk.extra["anchor"] = "none"


# ── Main assembly ───────────────────────────────────────────────────────────


def chunk_markdown(
    markdown_text: str,
    doc_id: str,
    pdf_path: Path | str | None = None,
    *,
    budget: int = DEFAULT_BUDGET,
    hard_cap: int = DEFAULT_HARD_CAP,
) -> dict[str, Any]:
    """Chunk audited markdown into semantic units.

    Returns {"chunks": [Chunk], "dropped_headers": [str]}.
    """
    body = strip_frontmatter(markdown_text)
    blocks = _parse_blocks(body.splitlines())
    dropped = _drop_running_headers(blocks)

    tracker = _HeadingTracker()
    chunks: list[Chunk] = []
    dup_counter: dict[tuple[tuple[str, ...], str], int] = {}
    pending_text: list[str] = []
    pending_path: list[str] = []
    last_text_block: str = ""  # caption source for S2 context

    def flush_text() -> None:
        nonlocal pending_text
        if not pending_text:
            return
        merged = "\n\n".join(pending_text)
        for piece in (
            _split_long_text(merged, hard_cap) if estimate_tokens(merged) > hard_cap else [merged]
        ):
            _emit(piece, "text", list(pending_path), piece)
        pending_text = []

    def _emit(content: str, kind: str, path: list[str], embed_text: str, **extra: Any) -> None:
        key = (tuple(path), content)
        dup = dup_counter.get(key, 0)
        dup_counter[key] = dup + 1
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(doc_id, path, content, dup),
                doc_id=doc_id,
                seq=len(chunks),
                heading_path=path,
                page=None,
                chunk_type=kind,
                content=content,
                embed_text=embed_text,
                token_est=estimate_tokens(content, strip_tags=(kind == "table")),
                extra=dict(extra),
            )
        )

    def _context_prefix(path: list[str]) -> str:
        """Heading path + nearest caption line (S2/N2: embed-only context)."""
        parts = [" > ".join(path)] if path else []
        if last_text_block:
            lines = [ln.strip() for ln in last_text_block.splitlines() if ln.strip()]
            caption = next((ln for ln in reversed(lines) if _CAPTION.match(ln)), "")
            if not caption and lines:
                caption = lines[-1][:200]
            if caption:
                parts.append(caption)
        return " | ".join(parts)

    for block in blocks:
        if block.kind == "heading":
            flush_text()
            tracker.push(block.level, block.text)
            pending_path = tracker.path
            continue
        path = tracker.path
        if block.kind == "text":
            if pending_text and estimate_tokens("\n\n".join(pending_text)) >= budget:
                flush_text()
            pending_path = path
            pending_text.append(block.text)
            last_text_block = block.text
            continue
        if block.kind == "table":
            flush_text()
            parts = _split_table_block(block.text, hard_cap)
            ctx = _context_prefix(path)
            for pi, part in enumerate(parts):
                embed = f"{ctx}\n{_TAG.sub(' ', part)}".strip()
                extra: dict[str, Any] = {}
                if len(parts) > 1:
                    extra = {"table_part": pi + 1, "table_parts": len(parts)}
                _emit(part, "table", path, embed, **extra)
            continue
        if block.kind == "figure":
            has_desc = len(block.lines) > 1
            if has_desc:
                flush_text()
                desc = _FIG_DESC.sub("", block.lines[1]).strip(" *:>")
                ctx = " > ".join(path)
                _emit(block.text, "figure", path, f"{ctx} | {desc}".strip(" |"))
            else:
                # N6: a bare image link has zero semantics — fold it into
                # the surrounding text instead of emitting a noise chunk.
                pending_path = path
                pending_text.append(block.text)
            continue
        if block.kind == "code":
            flush_text()
            ctx = " > ".join(path)
            _emit(block.text, "code", path, f"{ctx}\n{block.text}".strip())

    flush_text()

    if pdf_path is not None:
        anchor_pages(chunks, pdf_path)

    logger.info(
        "Chunked into %d chunks (%d table, %d figure, %d code); %d running headers dropped",
        len(chunks),
        sum(1 for c in chunks if c.chunk_type == "table"),
        sum(1 for c in chunks if c.chunk_type == "figure"),
        sum(1 for c in chunks if c.chunk_type == "code"),
        len(dropped),
    )
    return {"chunks": chunks, "dropped_headers": dropped}
