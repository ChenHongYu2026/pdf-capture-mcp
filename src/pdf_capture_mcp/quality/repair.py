"""Cross-channel repair: fix structural conversion defects using the PDF source.

Design principle — "repair-or-report":
    The information destroyed by the engine's layout analysis (torn numeric
    cells, fused headers, dropped text) still exists in the PDF. This module
    re-reads it through an INDEPENDENT channel — the pymupdf text layer with
    word geometry — and patches the markdown only when a machine-checkable
    verification gate passes. If verification fails, the defect is reported
    with recovered candidates instead. No defect is ever silently guessed at.

Repairers and their verification gates
======================================

MD-104 (torn scientific notation):
    Recover the true cell value from the text-layer line matching the row.
    GATE: squash(candidate) minus decimal points must equal squash(joined
    fragments) — the repair may only ADD the lost '.', never change digits.

MD-105 (header fused with data row):
    Rebuild the entire table from text-layer word geometry (columns derived
    from x-interval clustering of data lines).
    GATE: token multiset of the rebuilt table must equal the token multiset
    of the broken markdown table — content is rearranged, never invented.

MD-201 (content loss):
    Locate missing-token runs in the text layer, find an adjacent anchor run
    that DOES exist in the markdown, and inject the missing words next to it.
    GATE: after injection, no token may exceed its PDF text-layer count
    (over-injection triggers a full rollback), and the run must be small.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pdf_capture_mcp.config import get_logger
from pdf_capture_mcp.quality.md_audit import (
    _FRAGMENT_CELL,
    _STANDALONE_NUM,
    _TOKEN,
    AuditIssue,
    _cells,
    _is_separator_row,
)

logger = get_logger("quality.repair")

STATUS_REPAIRED = "repaired"
STATUS_REPORTED = "reported"

# Unify glyph variants so text-layer and markdown compare equal.
_NORM = str.maketrans({"−": "-", "–": "-", "—": "-", "×": "x", "·": "x"})


def _squash(s: str) -> str:
    """Whitespace-free, glyph-normalized, lowercase form for comparisons."""
    return re.sub(r"\s+", "", s).translate(_NORM).lower()


def _tokens(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(s)]


@dataclass
class RepairAction:
    """Outcome of one repair attempt (patched or candidate-reported)."""

    rule: str
    status: str  # 'repaired' | 'reported'
    description: str
    lines: list[int] = field(default_factory=list)
    evidence: str = ""


# ── Text-layer geometry helpers ─────────────────────────────────────────────


def _load_pages(pdf_path: Path | str) -> list[list[tuple[float, float, float, float, str]]]:
    """Per page: word boxes (x0, y0, x1, y1, text), reading order."""
    import fitz

    pages = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            words = [(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]
            pages.append(words)
    return pages


def _word_lines(
    words: list[tuple[float, float, float, float, str]],
    y_tol: float = 3.0,
) -> list[list[tuple[float, float, float, float, str]]]:
    """Cluster page words into visual lines by y proximity, x-sorted."""
    lines: list[list[tuple[float, float, float, float, str]]] = []
    for w in sorted(words, key=lambda w: (w[1], w[0])):
        if lines and abs(w[1] - lines[-1][-1][1]) <= y_tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w[0]) for line in lines]


def _best_page(pages: list[list[Any]], probe_tokens: set[str]) -> int:
    """Index of the page whose text shares the most probe tokens."""
    best, best_score = 0, -1
    for i, words in enumerate(pages):
        page_tokens = {t.lower() for w in words for t in _TOKEN.findall(w[4])}
        score = len(probe_tokens & page_tokens)
        if score > best_score:
            best, best_score = i, score
    return best


def _table_block_at(md_lines: list[str], lineno: int) -> tuple[int, int] | None:
    """(start, end) 0-based inclusive bounds of the table block covering lineno."""
    idx = lineno - 1
    if idx < 0 or idx >= len(md_lines) or not md_lines[idx].strip().startswith("|"):
        return None
    start = idx
    while start > 0 and md_lines[start - 1].strip().startswith("|"):
        start -= 1
    end = idx
    while end + 1 < len(md_lines) and md_lines[end + 1].strip().startswith("|"):
        end += 1
    return start, end


# ── MD-104: torn scientific notation ────────────────────────────────────────


def _find_value_run(
    line_words: list[tuple[float, float, float, float, str]],
    target_squashed: str,
    max_run: int = 6,
) -> str | None:
    """Find a word run whose squash, with '.' removed, equals the target.

    This is the verification gate: the recovered value may only differ from
    the torn fragments by decimal points — digits and symbols must match.
    """
    texts = [w[4] for w in line_words]
    for j in range(len(texts), 0, -1):  # prefer rightmost (values trail labels)
        for i in range(max(0, j - max_run), j):
            candidate = " ".join(texts[i:j])
            if _squash(candidate).replace(".", "") == target_squashed:
                return candidate
    return None


def _repair_torn_numeric(
    md_lines: list[str],
    issue: AuditIssue,
    pages: list[list[Any]],
) -> RepairAction:
    """Merge torn fragment cells using values recovered from the text layer."""
    bounds = _table_block_at(md_lines, issue.lines[0])
    if bounds is None:
        return RepairAction("MD-104", STATUS_REPORTED, "Table block not found.", issue.lines)
    start, end = bounds
    block = md_lines[start : end + 1]

    # Fragment column span must be identical across all torn rows.
    torn_idx = [ln - 1 - start for ln in issue.lines]
    span: tuple[int, int] | None = None
    for ti in torn_idx:
        cells = _cells(block[ti])
        frag_cols = [
            k
            for k, c in enumerate(cells)
            if c and (_FRAGMENT_CELL.fullmatch(c) or re.fullmatch(r"[×x]\s*10", c))
        ]
        if not frag_cols:
            return RepairAction("MD-104", STATUS_REPORTED, "No fragment cells.", issue.lines)
        cur = (min(frag_cols), max(frag_cols))
        if span is None:
            span = cur
        elif cur != span:
            return RepairAction(
                "MD-104",
                STATUS_REPORTED,
                "Fragment columns differ between rows — not a uniform tear.",
                issue.lines,
            )
    assert span is not None
    s_col, e_col = span

    probe = {t for row in block for t in _tokens(row)}
    page_words = pages[_best_page(pages, probe)]
    text_lines = _word_lines(page_words)

    # Recover each torn row's true value; every row must verify.
    repaired: dict[int, str] = {}
    for ti in torn_idx:
        cells = _cells(block[ti])
        target = _squash("".join(cells[s_col : e_col + 1]))
        anchors = [t for c in cells[:s_col] for t in _tokens(c)]
        candidate = None
        for line in text_lines:
            line_text = " ".join(w[4] for w in line).lower()
            if anchors and not any(a in line_text for a in anchors):
                continue
            candidate = _find_value_run(line, target)
            if candidate:
                break
        if candidate is None:
            return RepairAction(
                "MD-104",
                STATUS_REPORTED,
                f"Row at line {start + ti + 1}: no text-layer run verifies "
                f"against fragments {target!r}.",
                issue.lines,
                evidence="Verification gate: candidate minus '.' must equal fragments.",
            )
        repaired[ti] = candidate

    # All rows verified — rebuild the block, merging the span everywhere.
    new_block: list[str] = []
    for ri, row in enumerate(block):
        cells = _cells(row)
        if _is_separator_row(row):
            merged_cells = cells[:s_col] + ["---"] + cells[e_col + 1 :]
        elif ri in repaired:
            merged_cells = cells[:s_col] + [repaired[ri]] + cells[e_col + 1 :]
        else:
            joined = " ".join(c for c in cells[s_col : e_col + 1] if c).strip()
            merged_cells = cells[:s_col] + [joined] + cells[e_col + 1 :]
        new_block.append("| " + " | ".join(merged_cells) + " |")
    md_lines[start : end + 1] = new_block

    return RepairAction(
        "MD-104",
        STATUS_REPAIRED,
        f"Merged torn columns {s_col}-{e_col}; {len(repaired)} value(s) recovered "
        "from the PDF text layer.",
        issue.lines,
        evidence="; ".join(f"L{start + ti + 1}: {v!r}" for ti, v in sorted(repaired.items())),
    )


# ── MD-105: header fused with data row ──────────────────────────────────────


def _column_intervals(
    data_lines: list[list[tuple[float, float, float, float, str]]],
    gap: float = 6.0,
) -> list[tuple[float, float]]:
    """Merge word x-intervals across data lines into column intervals."""
    intervals = sorted((w[0], w[2]) for line in data_lines for w in line)
    merged: list[list[float]] = []
    for x0, x1 in intervals:
        if merged and x0 <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    return [(a, b) for a, b in merged]


def _assign_column(
    cols: list[tuple[float, float]], w: tuple[float, float, float, float, str]
) -> int:
    """Column index with maximum x-overlap (nearest center as fallback)."""
    best, best_ov = 0, -1.0
    for k, (a, b) in enumerate(cols):
        ov = min(b, w[2]) - max(a, w[0])
        if ov > best_ov:
            best, best_ov = k, ov
    return best


def _repair_fused_header(
    md_lines: list[str],
    issue: AuditIssue,
    pages: list[list[Any]],
) -> RepairAction:
    """Rebuild the whole table from text-layer geometry (token-conserving)."""
    bounds = _table_block_at(md_lines, issue.lines[0])
    if bounds is None:
        return RepairAction("MD-105", STATUS_REPORTED, "Table block not found.", issue.lines)
    start, end = bounds
    block = md_lines[start : end + 1]
    block_tokens = Counter(t for row in block for t in _tokens(row))

    page_words = pages[_best_page(pages, set(block_tokens))]
    text_lines = _word_lines(page_words)

    # Region = maximal contiguous run of lines dominated by block tokens.
    flags = []
    for line in text_lines:
        lt = _tokens(" ".join(w[4] for w in line))
        flags.append(bool(lt) and sum(1 for t in lt if block_tokens[t] > 0) / len(lt) >= 0.6)
    best_run: tuple[int, int] | None = None
    i = 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            if best_run is None or (j - i) > (best_run[1] - best_run[0]):
                best_run = (i, j)
            i = j + 1
        else:
            i += 1
    if best_run is None:
        return RepairAction("MD-105", STATUS_REPORTED, "Source region not found.", issue.lines)
    region = text_lines[best_run[0] : best_run[1] + 1]

    def _has_standalone_num(line: list[Any]) -> bool:
        return any(_STANDALONE_NUM.search(w[4]) for w in line)

    data_lines = [ln for ln in region if _has_standalone_num(ln)]
    header_lines = []
    for ln in region:
        if _has_standalone_num(ln):
            break
        header_lines.append(ln)
    if not data_lines or not header_lines:
        return RepairAction(
            "MD-105", STATUS_REPORTED, "Could not separate header/data lines.", issue.lines
        )

    cols = _column_intervals(data_lines)
    if len(cols) < 2:
        return RepairAction("MD-105", STATUS_REPORTED, "Column detection failed.", issue.lines)

    def _line_to_cells(line: list[Any]) -> list[str]:
        cells = [""] * len(cols)
        for w in line:
            k = _assign_column(cols, w)
            cells[k] = (cells[k] + " " + w[4]).strip()
        return cells

    header_cells = [""] * len(cols)
    for ln in header_lines:
        for k, c in enumerate(_line_to_cells(ln)):
            if c:
                header_cells[k] = (header_cells[k] + " " + c).strip()

    rebuilt = ["| " + " | ".join(header_cells) + " |"]
    rebuilt.append("|" + "|".join(["---"] * len(cols)) + "|")
    for ln in data_lines:
        rebuilt.append("| " + " | ".join(_line_to_cells(ln)) + " |")

    # GATE: token conservation — rearranged, never invented or dropped.
    rebuilt_tokens = Counter(t for row in rebuilt for t in _tokens(row))
    if rebuilt_tokens != block_tokens:
        return RepairAction(
            "MD-105",
            STATUS_REPORTED,
            "Rebuilt table failed the token-conservation gate; original kept.",
            issue.lines,
            evidence="Candidate table:\n" + "\n".join(rebuilt),
        )

    md_lines[start : end + 1] = rebuilt
    return RepairAction(
        "MD-105",
        STATUS_REPAIRED,
        f"Table rebuilt from text-layer geometry: {len(cols)} column(s), "
        f"{len(data_lines)} data row(s); token multiset conserved.",
        issue.lines,
        evidence=f"New header: {header_cells}",
    )


# ── MD-201: content loss re-injection ───────────────────────────────────────

_MAX_RUNS = 10
_MAX_RUN_WORDS = 12


def _repair_content_loss(
    md_text: str,
    issue: AuditIssue,
    pdf_path: Path | str,
    pages: list[list[Any]],
) -> tuple[str, RepairAction]:
    """Inject missing word runs next to anchors that exist in the markdown."""
    import fitz

    with fitz.open(str(pdf_path)) as doc:
        pdf_text = " ".join(page.get_text() for page in doc)
    pdf_counts = Counter(_tokens(pdf_text))
    md_counts = Counter(_tokens(md_text))
    deficit = {t: n - md_counts[t] for t, n in pdf_counts.items() if n > md_counts[t]}
    if not deficit:
        return md_text, RepairAction("MD-201", STATUS_REPAIRED, "No deficit remains.", [])

    total = sum(pdf_counts.values())
    if sum(deficit.values()) / total > 0.05:
        return md_text, RepairAction(
            "MD-201",
            STATUS_REPORTED,
            "Deficit exceeds 5% — bulk re-injection is unsafe; re-extract instead.",
            evidence=f"Missing examples: {', '.join(list(deficit)[:15])}",
        )

    def _in_md(word: str) -> bool:
        toks = _tokens(word)
        if toks:
            return all(deficit.get(t, 0) <= 0 for t in toks)
        return True  # punctuation/short glue words carry no signal

    injections: list[tuple[int, str]] = []  # (char_pos, text)
    budget: Counter[str] = Counter()  # tokens we are adding
    runs_found = 0

    for words in pages:
        if runs_found >= _MAX_RUNS:
            break
        for line in _word_lines(words):
            texts = [w[4] for w in line]
            missing_like = [
                bool(_tokens(t))
                and any(deficit.get(tok, 0) > 0 for tok in _tokens(t))
                or (not _in_md(t) and len(_squash(t)) >= 2)
                for t in texts
            ]
            k = 0
            while k < len(texts):
                if not missing_like[k]:
                    k += 1
                    continue
                j = k
                # extend across short glue words between missing words
                while j + 1 < len(texts) and (
                    missing_like[j + 1]
                    or (
                        j + 2 < len(texts)
                        and missing_like[j + 2]
                        and len(_squash(texts[j + 1])) <= 3
                    )
                ):
                    j += 1
                run_words = texts[k : j + 1]
                if 0 < len(run_words) <= _MAX_RUN_WORDS:
                    # anchor: following words on this line that exist in markdown
                    anchor_words = [t for t in texts[j + 1 : j + 5] if _in_md(t) and _tokens(t)]
                    pos = None
                    before = True
                    if anchor_words:
                        m2 = re.search(
                            r"\W*".join(re.escape(a.strip(",.*∗†")) for a in anchor_words[:2]),
                            md_text,
                            re.IGNORECASE,
                        )
                        if m2:
                            pos = m2.start()
                    if pos is None:
                        prev_words = [
                            t for t in texts[max(0, k - 4) : k] if _in_md(t) and _tokens(t)
                        ]
                        if prev_words:
                            m3 = re.search(
                                r"\W*".join(re.escape(a.strip(",.*∗†")) for a in prev_words[-2:]),
                                md_text,
                                re.IGNORECASE,
                            )
                            if m3:
                                pos, before = m3.end(), False
                    if pos is not None:
                        run_text = " ".join(run_words)
                        injections.append((pos, run_text + " " if before else " " + run_text))
                        budget.update(_tokens(run_text))
                        for tok in _tokens(run_text):
                            if tok in deficit:
                                deficit[tok] -= 1
                        runs_found += 1
                k = j + 1

    if not injections:
        return md_text, RepairAction(
            "MD-201",
            STATUS_REPORTED,
            "Missing runs located but no reliable markdown anchor found.",
            evidence=f"Deficit examples: {', '.join(list(deficit)[:15])}",
        )

    # GATE: injected tokens must not push any count above the PDF count.
    for tok, n in budget.items():
        if md_counts[tok] + n > pdf_counts.get(tok, 0):
            return md_text, RepairAction(
                "MD-201",
                STATUS_REPORTED,
                f"Injection rolled back: token {tok!r} would exceed its PDF count.",
                evidence="Over-injection guard triggered.",
            )

    new_text = md_text
    for pos, txt in sorted(injections, key=lambda x: -x[0]):
        new_text = new_text[:pos] + txt + new_text[pos:]

    return new_text, RepairAction(
        "MD-201",
        STATUS_REPAIRED,
        f"Injected {len(injections)} missing run(s) from the PDF text layer.",
        evidence="; ".join(repr(t.strip()) for _, t in injections),
    )


# ── Orchestrator ────────────────────────────────────────────────────────────


def repair_markdown(
    markdown_text: str,
    pdf_path: Path | str,
    issues: list[AuditIssue],
) -> dict[str, Any]:
    """Attempt cross-channel repair for repairable audit issues.

    Returns:
        {
          "text": repaired markdown,
          "modified": bool,
          "actions": [RepairAction, ...]  # repaired and reported alike
        }
    """
    actions: list[RepairAction] = []
    try:
        pages = _load_pages(pdf_path)
    except Exception as exc:  # noqa: BLE001 — repair is strictly best-effort
        logger.warning("Repair skipped, cannot read PDF text layer: %s", exc)
        return {"text": markdown_text, "modified": False, "actions": []}

    text = markdown_text
    md_lines = text.splitlines()

    for issue in issues:
        if issue.rule == "MD-104":
            actions.append(_repair_torn_numeric(md_lines, issue, pages))
        elif issue.rule == "MD-105":
            actions.append(_repair_fused_header(md_lines, issue, pages))
    text = "\n".join(md_lines) + ("\n" if markdown_text.endswith("\n") else "")

    for issue in issues:
        if issue.rule == "MD-201":
            text, action = _repair_content_loss(text, issue, pdf_path, pages)
            actions.append(action)

    repaired_n = sum(1 for a in actions if a.status == STATUS_REPAIRED)
    logger.info(
        "Cross-channel repair: %d repaired, %d reported",
        repaired_n,
        len(actions) - repaired_n,
    )
    return {"text": text, "modified": text != markdown_text, "actions": actions}
