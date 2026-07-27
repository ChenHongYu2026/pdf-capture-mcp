"""Post-extraction markdown audit: detect and (safely) auto-fix conversion defects.

Motivated by a real-world audit of a 75-page academic paper conversion, where
the generic QC gate (chars-per-page, heading count, formula validity, table
row count) missed every actual defect. The failure modes below are structurally
valid markdown — they can only be caught by content-aware rules.

Rule catalog
============

=======  ========================================  ========  =========
Rule     Detects                                   Severity  Auto-fix
=======  ========================================  ========  =========
MD-101   Garbled chars (U+FFFD, private-use area)  critical  no
MD-102   C0/C1 control chars inside text/tables    critical  yes
MD-103   Table with an all-empty header row        warn      no
MD-104   Numeric column tearing (e.g. a learning-
         rate cell split into `6 | 0 | x 10 | - 4`
         fragments, decimal point lost)            critical  no
MD-105   Table header fused with first data row
         (header cells contain standalone numbers) critical  no
MD-106   Empty <span></span> placeholder cells     info      yes
=======  ========================================  ========  =========

Auto-fix policy: only deterministic, information-preserving fixes are applied
(control-char removal, placeholder cleanup). Structural defects (MD-103/104/105)
are DETECTED and located but never rewritten automatically — repairing them
requires re-reading the source layout. The recommended remediation is
cross-validation against the two independent extraction channels that ship
with this package: `extract_tables` (pdfplumber, bypasses layout analysis)
and VLM table enrichment (`setup_vlm` + `enable_table_enrich`).

Content-loss coverage check
===========================

`check_content_coverage` compares the token multiset of the PDF text layer
(pymupdf — an independent channel from the extraction engine) against the
produced markdown. Large deficits (dropped pages/sections) raise warn/critical
issues; small deficits are reported with token examples for human/agent review.
Note: tiny deficits are expected (hyphenation merges, repeated page headers,
arXiv watermarks), so absence of a perfect 100% coverage is not itself an error.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pdf_capture_mcp.config import get_logger

logger = get_logger("quality.md_audit")

SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_CRITICAL = "critical"


@dataclass
class AuditIssue:
    """A single defect found in converted markdown."""

    rule: str
    severity: str
    message: str
    lines: list[int] = field(default_factory=list)
    suggestion: str = ""
    # Structured detector evidence (e.g. MD-110 geometric gate verdict);
    # consumed by repairers, serialized into qc_report.
    evidence: dict[str, Any] | None = None


# ── Shared table helpers ────────────────────────────────────────────────────


def _table_blocks(lines: list[str]) -> list[list[tuple[int, str]]]:
    """Group consecutive `|`-prefixed lines into table blocks (1-based lines)."""
    blocks: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        if line.strip().startswith("|"):
            cur.append((i, line))
        else:
            if len(cur) >= 2:
                blocks.append(cur)
            cur = []
    if len(cur) >= 2:
        blocks.append(cur)
    return blocks


def _cells(row: str) -> list[str]:
    """Split a markdown table row into stripped cell strings."""
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _is_separator_row(row: str) -> bool:
    """True for markdown separator rows like `|---|:--:|`."""
    return bool(re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", row)) and "-" in row


# ── Auto-fix (sanitize) ─────────────────────────────────────────────────────

# C0 controls except TAB(0x09); LF/CR are line structure, never inside lines.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SPAN_PLACEHOLDER = "<span></span>"

# MD-108: escaped brackets inside citation-anchor links. Engines emit
# references as ``[\[MCCD13,](#page-71-0)`` / ``[PSM14\]](#page-72-0)``.
# Valid CommonMark — but math-enabled renderers (MathJax/KaTeX) treat
# ``\[`` as a display-math opener; the anchor's ``#`` then explodes with
# "You can't use macro parameter character # in math mode", wrecking math
# rendering document-wide. Only the two link-scoped shapes are rewritten,
# so genuine display math (``\[x^2\]``) is never touched.
_ESC_OPEN_IN_LINK = re.compile(r"\[\\\[")  # literal '[\[' -> '[['
_ESC_CLOSE_IN_LINK = re.compile(r"\\\]\]\(")  # literal '\]](' -> ']]('


def sanitize_markdown(text: str) -> tuple[str, list[AuditIssue]]:
    """Apply deterministic, information-preserving fixes.

    Fixes:
    - MD-102: strip C0/C1 control characters. These typically appear at
      in-cell word wraps (e.g. ``En\\x02lightenment``); removing the char
      rejoins the word.
    - MD-106: drop empty ``<span></span>`` placeholder cells emitted for
      blank header cells.
    - MD-108: de-escape citation-link brackets that collide with math
      delimiters in MathJax/KaTeX renderers (link-scoped shapes only).

    Returns:
        (sanitized_text, fixes) — fixes describe what was changed and where.
    """
    fixes: list[AuditIssue] = []
    lines = text.splitlines()

    ctrl_lines = [i for i, line in enumerate(lines, 1) if _CONTROL_CHARS.search(line)]
    if ctrl_lines:
        text = _CONTROL_CHARS.sub("", text)
        fixes.append(
            AuditIssue(
                rule="MD-102",
                severity=SEVERITY_CRITICAL,
                message=f"Removed {len(ctrl_lines)} line(s) containing C0 control "
                "characters (word-wrap artifacts inside table cells).",
                lines=ctrl_lines,
                suggestion="Fixed automatically; word fragments rejoined.",
            )
        )

    span_lines = [i for i, line in enumerate(lines, 1) if _SPAN_PLACEHOLDER in line]
    if span_lines:
        text = text.replace(_SPAN_PLACEHOLDER, "")
        fixes.append(
            AuditIssue(
                rule="MD-106",
                severity=SEVERITY_INFO,
                message=f"Removed {text.count(_SPAN_PLACEHOLDER) or len(span_lines)} "
                "empty <span></span> placeholder cell(s).",
                lines=span_lines,
                suggestion="Fixed automatically; cells left blank.",
            )
        )

    n_open = len(_ESC_OPEN_IN_LINK.findall(text))
    n_close = len(_ESC_CLOSE_IN_LINK.findall(text))
    if n_open or n_close:
        esc_lines = [
            i
            for i, line in enumerate(lines, 1)
            if _ESC_OPEN_IN_LINK.search(line) or _ESC_CLOSE_IN_LINK.search(line)
        ]
        text = _ESC_OPEN_IN_LINK.sub("[[", text)
        text = _ESC_CLOSE_IN_LINK.sub("]](", text)
        fixes.append(
            AuditIssue(
                rule="MD-108",
                severity=SEVERITY_WARN,
                message=f"De-escaped {n_open + n_close} citation-link bracket(s) "
                "('[\\[' / '\\]](') that math-enabled renderers misread as "
                "display-math delimiters, breaking formula rendering with "
                "'macro parameter character #' errors.",
                lines=esc_lines[:20],
                suggestion="Fixed automatically; rendering equivalence preserved "
                "(brackets still display literally in CommonMark).",
            )
        )

    return text, fixes


# ── Detectors ───────────────────────────────────────────────────────────────


def _detect_garbled_chars(lines: list[str]) -> list[AuditIssue]:
    """MD-101: replacement char / private-use-area chars = irrecoverable garble."""
    hits: list[int] = []
    for i, line in enumerate(lines, 1):
        for ch in line:
            cp = ord(ch)
            if ch == "\ufffd" or 0xE000 <= cp <= 0xF8FF or unicodedata.category(ch) == "Co":
                hits.append(i)
                break
    if not hits:
        return []
    return [
        AuditIssue(
            rule="MD-101",
            severity=SEVERITY_CRITICAL,
            message=f"Garbled characters (U+FFFD / private-use area) on {len(hits)} line(s).",
            lines=hits[:20],
            suggestion="Source glyphs were not recognized. Re-run with force_ocr=True "
            "or enable VLM enrichment for the affected pages.",
        )
    ]


def _detect_empty_header(blocks: list[list[tuple[int, str]]]) -> list[AuditIssue]:
    """MD-103: header row where every cell is blank."""
    hits: list[int] = []
    for block in blocks:
        header_cells = _cells(block[0][1])
        if header_cells and all(c == "" for c in header_cells):
            hits.append(block[0][0])
    if not hits:
        return []
    return [
        AuditIssue(
            rule="MD-103",
            severity=SEVERITY_WARN,
            message=f"{len(hits)} table(s) have an all-empty header row — usually a "
            "multi-column layout (e.g. an author block) misread as a table.",
            lines=hits,
            suggestion="Verify against the source; content may belong outside a table. "
            "Cross-check the region with extract_tables (pdfplumber).",
        )
    ]


# A cell that is only a fragment of a scientific-notation number:
# '× 10' / 'x 10' / '− 4' / '- 4' / a lone digit or two.
_FRAGMENT_CELL = re.compile(r"^(?:[×x]\s*10|10\s*[−-]\s*\d{1,2}|[−-]\s*\d{1,2}|\d{1,2})$")


def _detect_numeric_tearing(blocks: list[list[tuple[int, str]]]) -> list[AuditIssue]:
    """MD-104: numeric value torn across adjacent cells (decimal point lost).

    Signature: a data row containing a `× 10`-style fragment cell PLUS at
    least two other fragment cells (the mantissa digits and the exponent).
    One such row may be a coincidence; two or more rows in the same table is
    a systematic column-splitting failure.
    """
    issues: list[AuditIssue] = []
    for block in blocks:
        torn_rows: list[int] = []
        for lineno, row in block:
            if _is_separator_row(row):
                continue
            cells = _cells(row)
            has_sci = any(re.fullmatch(r"[×x]\s*10", c) for c in cells)
            frag_count = sum(1 for c in cells if c and _FRAGMENT_CELL.fullmatch(c))
            if has_sci and frag_count >= 3:
                torn_rows.append(lineno)
        if len(torn_rows) >= 2:
            issues.append(
                AuditIssue(
                    rule="MD-104",
                    severity=SEVERITY_CRITICAL,
                    message=f"Table at line {block[0][0]}: {len(torn_rows)} row(s) show "
                    "scientific-notation values torn across cells "
                    "(e.g. '6 | 0 | × 10 | − 4' — the decimal point is lost).",
                    lines=torn_rows,
                    suggestion="Numbers cannot be trusted as-is. Recover the column from "
                    "extract_tables (pdfplumber) output, or re-run with VLM "
                    "table enrichment (enable_table_enrich=True).",
                )
            )
    return issues


# Standalone number token: not glued to letters (so 'F1'/'RACE-h' don't match).
_STANDALONE_NUM = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?%?(?![A-Za-z0-9_])")


def _detect_header_fusion(blocks: list[list[tuple[int, str]]]) -> list[AuditIssue]:
    """MD-105: first data row fused INTO the header row.

    Signature: two or more header cells contain standalone numeric tokens
    (e.g. header '| Dataset Common Crawl | (filtered) 410 | ... 60% | ... 0.44 |').
    Legitimate headers rarely contain more than one bare number.
    """
    issues: list[AuditIssue] = []
    for block in blocks:
        header_cells = _cells(block[0][1])
        numeric_cells = [c for c in header_cells if c and _STANDALONE_NUM.search(c)]
        if len(numeric_cells) >= 2:
            issues.append(
                AuditIssue(
                    rule="MD-105",
                    severity=SEVERITY_CRITICAL,
                    message=f"Table at line {block[0][0]}: header row contains "
                    f"{len(numeric_cells)} cell(s) with standalone numbers — the first "
                    "data row was likely merged into the header "
                    f"(e.g. {numeric_cells[0]!r}).",
                    lines=[block[0][0]],
                    suggestion="One data row is corrupted. Recover it from "
                    "extract_tables (pdfplumber) output for this page.",
                )
            )
    return issues


def _detect_flattened_group_header(blocks: list[list[tuple[int, str]]]) -> list[AuditIssue]:
    """MD-107: multi-row group header flattened into single-row labels.

    Wide tables often use a two-row header (group names above metric names).
    When layout analysis flattens them, group labels get glued onto the wrong
    metric cells — observed in a real two-column paper as headers like
    'Knowledge Substring EM' and 'LongMemEval Temporal Substring EM'.

    Signature: two or more header cells that (a) end with an identical metric
    bigram and (b) carry extra leading words (the misattached group labels).
    """
    issues: list[AuditIssue] = []
    for block in blocks:
        header_cells = _cells(block[0][1])
        trailing: dict[str, int] = {}
        for cell in header_cells:
            words = cell.split()
            if len(words) >= 3:  # metric name + at least one glued group word
                bigram = " ".join(words[-2:]).lower()
                trailing[bigram] = trailing.get(bigram, 0) + 1
        repeated = [bg for bg, n in trailing.items() if n >= 2]
        if repeated:
            issues.append(
                AuditIssue(
                    rule="MD-107",
                    severity=SEVERITY_WARN,
                    message=f"Table at line {block[0][0]}: header cells repeat the "
                    f"trailing metric name(s) {repeated!r} with extra leading words "
                    "— a multi-row group header was likely flattened, so group "
                    "labels may be attached to the wrong columns.",
                    lines=[block[0][0]],
                    suggestion="Verify column grouping against the source table. "
                    "Cross-check with extract_tables (pdfplumber) or re-run with "
                    "VLM table enrichment.",
                )
            )
    return issues


def _detect_math_delimiter_collision(lines: list[str]) -> list[AuditIssue]:
    """MD-108 (residual): escaped brackets co-located with anchor links.

    Catches shapes the link-scoped sanitizer does not rewrite — any remaining
    ``\\[``/``\\]`` on a line that also carries a ``](#`` anchor link will
    still break math-enabled renderers.
    """
    hits = [
        i for i, line in enumerate(lines, 1) if ("\\[" in line or "\\]" in line) and "](#" in line
    ]
    if not hits:
        return []
    return [
        AuditIssue(
            rule="MD-108",
            severity=SEVERITY_WARN,
            message=f"{len(hits)} line(s) still mix escaped brackets with anchor "
            "links — math-enabled renderers may misread them as display-math "
            "delimiters.",
            lines=hits[:20],
            suggestion="Review these lines; consider replacing '\\[' / '\\]' with "
            "plain brackets if they are not genuine display math.",
        )
    ]


def audit_markdown(text: str) -> list[AuditIssue]:
    """Run all content-aware detectors against converted markdown."""
    lines = text.splitlines()
    blocks = _table_blocks(lines)
    issues: list[AuditIssue] = []
    issues += _detect_garbled_chars(lines)
    issues += _detect_empty_header(blocks)
    issues += _detect_numeric_tearing(blocks)
    issues += _detect_header_fusion(blocks)
    issues += _detect_flattened_group_header(blocks)
    issues += _detect_math_delimiter_collision(lines)
    return issues


# ── Content-loss coverage check ─────────────────────────────────────────────

_TOKEN = re.compile(r"[A-Za-z][A-Za-z\-']{3,}")

# Vector-figure region thresholds: ignore hairlines/underline strokes, and
# only treat reasonably large merged drawing clusters as figures.
_MIN_FIGURE_SIDE = 30.0
_MIN_FIGURE_AREA = 4000.0


def _merge_rects(rects: list[Any], pad: float = 5.0) -> list[Any]:
    """Iteratively union overlapping/nearby rects (page-level counts are small)."""
    import fitz

    rects = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        merged: list[Any] = []
        for r in rects:
            hit = None
            for m in merged:
                if fitz.Rect(m.x0 - pad, m.y0 - pad, m.x1 + pad, m.y1 + pad).intersects(r):
                    hit = m
                    break
            if hit is not None:
                hit.include_rect(r)
                changed = True
            else:
                merged.append(fitz.Rect(r))
        rects = merged
    return rects


def _figure_rects(page: Any) -> list[Any]:
    """Figure regions on a page: merged vector-drawing clusters plus images.

    All drawing fragments (arrows, boxes, short connectors) are merged FIRST
    and the size filter is applied to the merged clusters only — filtering
    fragments up front breaks clusters apart and leaves figure text
    unclassified (observed on a real two-column paper).
    """
    import fitz

    rects: list[Any] = []
    try:
        for d in page.get_drawings():
            r = fitz.Rect(d["rect"])
            if r.width >= 2.0 or r.height >= 2.0:  # skip degenerate specks only
                rects.append(r)
        for info in page.get_image_info():
            rects.append(fitz.Rect(info["bbox"]))
    except Exception:  # noqa: BLE001 — figure detection is best-effort
        return []
    if len(rects) > 3000:  # pathological pages: skip rather than stall
        return []
    return [
        r
        for r in _merge_rects(rects)
        if r.get_area() >= _MIN_FIGURE_AREA
        and r.width >= _MIN_FIGURE_SIDE
        and r.height >= _MIN_FIGURE_SIDE
    ]


def _pdf_token_layers(pdf_path: Path | str) -> tuple[Counter[str], Counter[str]] | None:
    """Token multisets from the PDF text layer, split into body vs figure text.

    Two calibrations learned from a real two-column paper audit:
    - Body words are dehyphenated (a word ending in '-' merges with the next),
      so line-wrap hyphenation no longer shows up as fake deficits
      ('manage-' + 'ment' vs the correctly merged 'management').
    - Words inside vector-figure regions are counted separately: engines
      export figures as images, so figure-embedded text never reaches the
      markdown body and must not be billed as body content loss.

    Returns (body_tokens, figure_tokens), or None without a usable text layer.
    """
    try:
        import fitz
    except ImportError:
        logger.warning("pymupdf not available — coverage check skipped")
        return None

    body: Counter[str] = Counter()
    figure: Counter[str] = Counter()
    total_chars = 0
    line_pages: dict[str, set[int]] = {}
    page_lines: list[list[str]] = []
    try:
        with fitz.open(str(pdf_path)) as doc:
            for pno, page in enumerate(doc):
                regions = _figure_rects(page)
                body_words: list[str] = []
                for w in page.get_text("words"):
                    total_chars += len(w[4])
                    center = fitz.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)
                    if any(r.contains(center) for r in regions):
                        figure.update(t.lower() for t in _TOKEN.findall(w[4]))
                    else:
                        body_words.append(w[4])
                merged_words: list[str] = []
                skip = False
                for k, wtext in enumerate(body_words):
                    if skip:
                        skip = False
                        continue
                    if wtext.endswith("-") and len(wtext) > 1 and k + 1 < len(body_words):
                        merged_words.append(wtext[:-1] + body_words[k + 1])
                        skip = True
                    else:
                        merged_words.append(wtext)
                body.update(t.lower() for t in _TOKEN.findall(" ".join(merged_words)))
                # Track lines for running-furniture detection (magazine
                # calibration): nav bars / footers repeat across pages.
                lines_here = [
                    s
                    for s in (ln.strip() for ln in page.get_text().splitlines())
                    if 4 < len(s) < 120
                ]
                page_lines.append(lines_here)
                for s in set(lines_here):
                    line_pages.setdefault(s, set()).add(pno)
    except Exception as exc:  # noqa: BLE001 — coverage is best-effort
        logger.warning("Coverage check failed to read PDF: %s", exc)
        return None

    if total_chars < 200:  # scanned PDF: no usable text layer
        return None

    # Running-furniture exemption (learned from an InDesign magazine audit):
    # lines repeating on many pages (chapter nav bars, branded footers) are
    # NOISE the engine correctly drops — billing their tokens as "missing
    # body content" turns a cleanup victory into a false MD-201 critical.
    furniture_threshold = max(3, int(len(page_lines) * 0.3))
    furniture_lines = {s for s, pages in line_pages.items() if len(pages) >= furniture_threshold}
    if furniture_lines:
        furniture: Counter[str] = Counter()
        for lines_here in page_lines:
            for s in lines_here:
                if s in furniture_lines:
                    furniture.update(t.lower() for t in _TOKEN.findall(s))
        body.subtract(furniture)
        body = Counter({t: n for t, n in body.items() if n > 0})
    return body, figure


def check_content_coverage(
    pdf_path: Path | str,
    markdown_text: str,
    max_examples: int = 20,
) -> AuditIssue | None:
    """MD-201: compare BODY token multisets — PDF text layer vs markdown.

    The comparison is hyphenation-normalized and excludes figure-embedded
    text (reported separately as MD-202), so the deficit ratio reflects
    genuine body content loss only.

    Severity: critical if >5% of body token occurrences are missing, warn if
    >1%, info otherwise. Returns None when the text layer is unavailable or
    the deficit is zero.
    """
    layers = _pdf_token_layers(pdf_path)
    if layers is None:
        return None
    body, _figure = layers
    md_tokens = Counter(t.lower() for t in _TOKEN.findall(markdown_text))

    total = sum(body.values())
    deficits = {t: n - md_tokens[t] for t, n in body.items() if n > md_tokens[t]}
    missing = sum(deficits.values())
    if total == 0 or missing == 0:
        return None

    ratio = missing / total
    if ratio > 0.05:
        severity = SEVERITY_CRITICAL
    elif ratio > 0.01:
        severity = SEVERITY_WARN
    else:
        severity = SEVERITY_INFO

    examples = [t for t, _ in sorted(deficits.items(), key=lambda kv: -kv[1])][:max_examples]
    return AuditIssue(
        rule="MD-201",
        severity=severity,
        message=f"{missing}/{total} body token occurrences ({ratio:.2%}) from the PDF "
        "text layer are missing in the markdown (hyphenation-normalized; "
        "figure-embedded text excluded — see MD-202). Small deficits are "
        "normal (page headers, watermarks); review the examples.",
        suggestion="Search the example tokens in the source PDF to locate dropped "
        f"regions. Examples: {', '.join(examples)}",
    )


def check_figure_text_omission(
    pdf_path: Path | str,
    markdown_text: str,
    max_examples: int = 15,
) -> AuditIssue | None:
    """MD-202: text embedded in vector figures that is absent from the body.

    Engines export figures as images, so their embedded text never reaches
    the markdown flow. This is expected behavior — reported at info level so
    downstream consumers know transcription requires VLM enrichment.
    """
    layers = _pdf_token_layers(pdf_path)
    if layers is None:
        return None
    _body, figure = layers
    if not figure:
        return None
    md_tokens = Counter(t.lower() for t in _TOKEN.findall(markdown_text))
    deficits = {t: n - md_tokens[t] for t, n in figure.items() if n > md_tokens[t]}
    missing = sum(deficits.values())
    if missing == 0:
        return None

    examples = [t for t, _ in sorted(deficits.items(), key=lambda kv: -kv[1])][:max_examples]
    return AuditIssue(
        rule="MD-202",
        severity=SEVERITY_INFO,
        message=f"{missing} token occurrence(s) embedded in vector-figure regions "
        "are not in the markdown body. Figures are exported as images, so "
        "no body content is lost — expected engine behavior.",
        suggestion="If figure text matters downstream, enable VLM enrichment "
        f"(setup_vlm) to transcribe figure regions. Examples: {', '.join(examples)}",
    )


# ── Orchestrator ────────────────────────────────────────────────────────────


_IMG_REF = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")


def fix_image_links(
    text: str,
    base_dir: Path | str,
    autofix: bool = True,
) -> tuple[str, list[AuditIssue], list[AuditIssue]]:
    """MD-109: image references that do not resolve from the markdown's dir.

    Engines may emit bare filenames (``![](_page_2_Figure_0.jpeg)``) while
    the files are saved into an ``images/`` subdirectory — every image then
    renders as broken. When the target exists under ``images/<basename>``
    the reference is rewritten (deterministic, information-preserving);
    references that resolve nowhere on disk are reported.

    Args:
        text: Markdown content.
        base_dir: Directory the markdown file lives in (link resolution root).
        autofix: Rewrite recoverable references; False = detect only.

    Returns:
        (text, fixes, issues) — applied rewrites and unresolvable references.
    """
    base = Path(base_dir)
    fixes: list[AuditIssue] = []
    issues: list[AuditIssue] = []
    rewritten: list[str] = []
    broken: list[str] = []

    for ref in dict.fromkeys(_IMG_REF.findall(text)):  # unique, ordered
        if ref.startswith(("http://", "https://", "data:")):
            continue
        if (base / ref).exists():
            continue
        candidate = base / "images" / Path(ref).name
        if candidate.exists():
            if autofix:
                text = text.replace(f"]({ref})", f"](images/{Path(ref).name})")
                rewritten.append(ref)
            else:
                broken.append(ref)
        else:
            broken.append(ref)

    if rewritten:
        fixes.append(
            AuditIssue(
                rule="MD-109",
                severity=SEVERITY_WARN,
                message=f"Rewrote {len(rewritten)} image reference(s) from bare "
                "filenames to 'images/<name>' — the engine saved files into an "
                "images/ subdirectory without updating the links, so every "
                "image rendered as broken.",
                suggestion="Fixed automatically; links now resolve relative to the markdown file.",
            )
        )
    if broken:
        issues.append(
            AuditIssue(
                rule="MD-109",
                severity=SEVERITY_WARN,
                message=f"{len(broken)} image reference(s) do not resolve to any "
                f"file on disk (checked as-is and under images/): "
                f"{', '.join(broken[:8])}",
                suggestion="Verify the extraction output directory is intact; "
                "re-run the conversion if image files are missing.",
            )
        )
    return text, fixes, issues


def run_markdown_audit(
    markdown_text: str,
    pdf_path: Path | str | None = None,
    autofix: bool = True,
    base_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Sanitize (optional) then audit converted markdown.

    Args:
        markdown_text: The converted markdown.
        pdf_path: Source PDF for the content-coverage check (skipped if None).
        autofix: Apply safe deterministic fixes (MD-102, MD-106, MD-108,
            MD-109 link rewrites).
        base_dir: Directory the markdown lives in — enables the MD-109
            image-link integrity check (skipped if None).

    Returns:
        {
          "text": possibly-sanitized markdown,
          "modified": bool — True when autofix changed the text,
          "fixes":  [AuditIssue, ...] — applied fixes,
          "issues": [AuditIssue, ...] — remaining detected defects,
          "counts": {"critical": n, "warn": n, "info": n},
        }
    """
    fixes: list[AuditIssue] = []
    text = markdown_text
    if autofix:
        text, fixes = sanitize_markdown(markdown_text)

    img_issues: list[AuditIssue] = []
    if base_dir is not None:
        text, img_fixes, img_issues = fix_image_links(text, base_dir, autofix=autofix)
        fixes += img_fixes

    issues = audit_markdown(text) + img_issues
    if pdf_path is not None:
        coverage = check_content_coverage(pdf_path, text)
        if coverage is not None:
            issues.append(coverage)
        figure_omission = check_figure_text_omission(pdf_path, text)
        if figure_omission is not None:
            issues.append(figure_omission)

    counts = {
        SEVERITY_CRITICAL: sum(1 for i in issues if i.severity == SEVERITY_CRITICAL),
        SEVERITY_WARN: sum(1 for i in issues if i.severity == SEVERITY_WARN),
        SEVERITY_INFO: sum(1 for i in issues if i.severity == SEVERITY_INFO),
    }
    logger.info(
        "Markdown audit: %d fix(es) applied, issues critical=%d warn=%d info=%d",
        len(fixes),
        counts[SEVERITY_CRITICAL],
        counts[SEVERITY_WARN],
        counts[SEVERITY_INFO],
    )
    return {
        "text": text,
        "modified": text != markdown_text,
        "fixes": fixes,
        "issues": issues,
        "counts": counts,
    }
