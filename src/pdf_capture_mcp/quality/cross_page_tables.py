"""MD-110: Cross-page table detection and merge.

Problem: marker (and most PDF engines) emit a separate pipe table for each
page that continues a single logical table. The column set is identical but
the first page's table is truncated at the page break and the continuation
on the next page either repeats the header row or starts straight into data
rows. This produces duplicate headers in the output, and downstream
chunkers/RAG treat them as unrelated tables.

Contract (S3 audit: geometric THREE-EVIDENCE GATE):
1. Previous table's last row bbox touches page bottom (bottom 5% margin).
2. Next table's first data row bbox touches page top (top 5% margin).
3. No caption text ("Table N" pattern) separates them in the PDF text layer.

All three must be satisfied for merge — token conservation alone is
insufficient because two legitimately separate tables with the same columns
will pass a token-only gate (S3 — "only prevents loss, not false merge").

Detection emits an AuditIssue (MD-110) with suggestion and candidate merge
targets. The repair function performs the merge if the gate passes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pdf_capture_mcp.config import get_logger
from pdf_capture_mcp.quality.md_audit import (
    SEVERITY_WARN,
    AuditIssue,
    _cells,
    _is_separator_row,
    _table_blocks,
)
from pdf_capture_mcp.quality.repair import (
    STATUS_REPAIRED,
    STATUS_REPORTED,
    RepairAction,
    _load_pages,
)

logger = get_logger("quality.cross_page_tables")

_CAPTION_PAT = re.compile(r"\b(Table|表)\s*[\d.:]+", re.IGNORECASE)


def _page_height(pages: list[list[Any]], page_no: int) -> float:
    """Estimate page height from the bounding box of all words."""
    words = pages[page_no]
    if not words:
        return 792.0  # US letter default
    return float(max(w[3] for w in words)) + 36  # +margin estimate


def _table_page_bbox(
    pages: list[list[Any]], table_lines: list[tuple[int, str]]
) -> tuple[int, float, float] | None:
    """Find the page best matching this table and return (page_no, min_y, max_y)."""
    tokens: set[str] = set()
    for _, row in table_lines:
        tokens.update(c.lower() for c in _cells(row) if c and len(c) > 2)
    if not tokens:
        return None
    best_page, best_score = 0, 0
    for pno, words in enumerate(pages):
        score = sum(1 for w in words if w[4].lower() in tokens)
        if score > best_score:
            best_page, best_score = pno, score
    if best_score < 3:
        return None
    matched_words = [w for w in pages[best_page] if w[4].lower() in tokens]
    return best_page, min(w[1] for w in matched_words), max(w[3] for w in matched_words)


def detect_cross_page_tables(text: str, pdf_path: Path | str | None = None) -> list[AuditIssue]:
    """Detect adjacent pipe tables with identical column structures.

    If pdf_path is provided, the geometric three-evidence gate is checked.
    Returns MD-110 issues for each candidate merge pair.
    """
    lines = text.splitlines()
    blocks = _table_blocks(lines)
    if len(blocks) < 2:
        return []

    pages = None
    if pdf_path is not None:
        try:
            pages = _load_pages(pdf_path)
        except Exception:  # noqa: BLE001 — best-effort geometry
            pass

    issues: list[AuditIssue] = []
    for i in range(len(blocks) - 1):
        a, b = blocks[i], blocks[i + 1]
        # Structural check: same number of columns
        cols_a = len(_cells(a[0][1]))
        cols_b = len(_cells(b[0][1]))
        if cols_a != cols_b:
            continue
        # Check if table B starts with a header identical to A's header
        header_a = _cells(a[0][1])
        header_b = _cells(b[0][1])
        has_dup_header = header_a == header_b

        # Proximity: the two blocks must be "adjacent" (<=3 blank lines apart)
        last_a_line = a[-1][0]
        first_b_line = b[0][0]
        gap_lines = [lines[ln - 1].strip() for ln in range(last_a_line + 1, first_b_line)]
        if any(g and not g.startswith("#") for g in gap_lines):
            continue  # non-heading content between them — not a continuation

        # Geometric three-evidence gate (S3)
        geo_pass = False
        geo_reason = "no PDF geometry available"
        if pages is not None:
            bbox_a = _table_page_bbox(pages, a)
            bbox_b = _table_page_bbox(pages, b)
            if bbox_a is not None and bbox_b is not None:
                page_a, _, max_y_a = bbox_a
                page_b, min_y_b, _ = bbox_b
                if page_a == page_b:
                    geo_reason = "both tables on same page — not a page break"
                elif page_b == page_a + 1:
                    ph = _page_height(pages, page_a)
                    evidence_1 = max_y_a >= ph * 0.85  # table A near page bottom
                    evidence_2 = min_y_b <= ph * 0.15  # table B near page top
                    # Evidence 3: no caption text between
                    between_text = ""
                    for w in pages[page_a]:
                        if w[1] > max_y_a:
                            between_text += w[4] + " "
                    for w in pages[page_b]:
                        if w[3] < min_y_b:
                            between_text += w[4] + " "
                    evidence_3 = not _CAPTION_PAT.search(between_text)
                    geo_pass = evidence_1 and evidence_2 and evidence_3
                    geo_reason = f"bottom={evidence_1}, top={evidence_2}, no_caption={evidence_3}"
                else:
                    geo_reason = f"tables on non-adjacent pages ({page_a + 1}, {page_b + 1})"

        issues.append(
            AuditIssue(
                rule="MD-110",
                severity=SEVERITY_WARN,
                message=(
                    f"Adjacent tables (lines {a[0][0]}-{a[-1][0]} and "
                    f"{b[0][0]}-{b[-1][0]}) have identical column count ({cols_a}) "
                    f"and {'duplicate' if has_dup_header else 'similar'} headers. "
                    f"Geometry: {geo_reason}."
                ),
                lines=[a[0][0], b[0][0]],
                suggestion=(
                    "Merge candidate: remove duplicate header from second table."
                    if geo_pass
                    else "Manual review recommended — geometric evidence insufficient."
                ),
                evidence={"geo_pass": geo_pass, "has_dup_header": has_dup_header},
            )
        )
    return issues


def merge_cross_page_tables(md_lines: list[str], issue: AuditIssue) -> RepairAction:
    """Merge two adjacent tables when the geometric gate passed."""
    evidence = issue.evidence or {}
    if not evidence.get("geo_pass"):
        return RepairAction(
            "MD-110",
            STATUS_REPORTED,
            "Geometric gate not passed — tables kept separate.",
            issue.lines,
        )

    lines_list = list(md_lines)
    # Locate second table and remove its header + separator
    second_start = issue.lines[1] - 1  # 0-based
    # Find the separator row at the top of second table
    removed = 0
    while second_start < len(lines_list) and lines_list[second_start].strip().startswith("|"):
        if removed == 0:
            lines_list.pop(second_start)  # header row
            removed += 1
        elif _is_separator_row(lines_list[second_start]):
            lines_list.pop(second_start)
            removed += 1
            break
        else:
            break

    # Remove blank lines between the two tables
    first_end = issue.lines[0] - 1  # approximate: find end of first table
    while first_end < len(lines_list) and lines_list[first_end].strip().startswith("|"):
        first_end += 1
    # Remove blanks between first_end and second_start
    while first_end < second_start and first_end < len(lines_list):
        if not lines_list[first_end].strip():
            lines_list.pop(first_end)
            second_start -= 1
        else:
            break

    md_lines[:] = lines_list
    return RepairAction(
        "MD-110",
        STATUS_REPAIRED,
        f"Merged cross-page table (removed duplicate header, {removed} lines).",
        issue.lines,
    )
