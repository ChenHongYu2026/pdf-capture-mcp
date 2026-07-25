"""Layout cleaning: remove headers/footers, fix TOC, normalize sections."""

from __future__ import annotations

import re
from typing import Any

from pdf_capture_mcp.config import get_logger

logger = get_logger("processors.layout")

# Header/footer patterns (repeated lines at page boundaries)
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")
_FOOTER_RE = re.compile(
    r"^(?:Page\s+\d+|第\s*\d+\s*页|-\s*\d+\s*-|©.*\d{4}|\d{4}\s*©)",
    re.IGNORECASE,
)

# TOC patterns
_TOC_LINE_RE = re.compile(
    r"^(.{2,60}?)\s*[\.·…]{3,}\s*(\d{1,4})\s*$"
)

# Orphan section labels (e.g. "3.2" alone on a line, next line is the title)
_ORPHAN_LABEL_RE = re.compile(r"^(\d+(?:\.\d+){0,3})\s*$")


def _remove_headers_footers(lines: list[str]) -> tuple[list[str], int]:
    """Remove repeated header/footer lines and page numbers."""
    removed = 0
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if _PAGE_NUMBER_RE.match(stripped):
            removed += 1
            continue
        if _FOOTER_RE.match(stripped):
            removed += 1
            continue
        cleaned.append(line)
    return cleaned, removed


def _clean_toc(lines: list[str]) -> tuple[list[str], int]:
    """Clean table-of-contents dot leaders."""
    fixes = 0
    cleaned = []
    for line in lines:
        m = _TOC_LINE_RE.match(line.strip())
        if m:
            title = m.group(1).strip()
            page = m.group(2)
            cleaned.append(f"- {title} (p.{page})")
            fixes += 1
        else:
            cleaned.append(line)
    return cleaned, fixes


def _promote_orphan_labels(lines: list[str]) -> tuple[list[str], int]:
    """Merge orphan section numbers with their following title line."""
    fixes = 0
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _ORPHAN_LABEL_RE.match(line.strip())
        if m and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            # If next line looks like a title (not empty, not a number)
            if next_line and not _PAGE_NUMBER_RE.match(next_line):
                label = m.group(1)
                cleaned.append(f"## {label} {next_line}")
                fixes += 1
                i += 2
                continue
        cleaned.append(line)
        i += 1
    return cleaned, fixes


def _collapse_blank_lines(lines: list[str]) -> list[str]:
    """Collapse runs of 3+ blank lines into 2."""
    result: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    return result


def clean_layout(markdown_text: str) -> dict[str, Any]:
    """Apply layout cleaning to extracted markdown.

    Fixes:
    - Remove repeated headers/footers and page numbers
    - Clean TOC dot leaders
    - Promote orphan section labels
    - Collapse excessive blank lines

    Args:
        markdown_text: Raw extracted markdown.

    Returns:
        Dict with ok, cleaned_text, total_fixes, stats.
    """
    lines = markdown_text.split("\n")
    stats: dict[str, Any] = {}

    # Pass 1: Headers/footers
    lines, hf_removed = _remove_headers_footers(lines)
    stats["header_footer"] = {"lines_removed": hf_removed}

    # Pass 2: TOC cleanup
    lines, toc_fixes = _clean_toc(lines)
    stats["toc_cleanup"] = {"fixes": toc_fixes}

    # Pass 3: Orphan labels
    lines, orphan_fixes = _promote_orphan_labels(lines)
    stats["orphan_labels"] = {"fixes": orphan_fixes}

    # Pass 4: Collapse blanks
    lines = _collapse_blank_lines(lines)

    cleaned_text = "\n".join(lines)
    total_fixes = hf_removed + toc_fixes + orphan_fixes

    if total_fixes > 0:
        logger.info("Layout cleaning: %d fixes applied", total_fixes)

    return {
        "ok": True,
        "cleaned_text": cleaned_text,
        "total_fixes": total_fixes,
        "stats": stats,
    }
