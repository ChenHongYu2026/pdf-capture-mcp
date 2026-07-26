"""QC quality gate: multi-dimensional quality assessment for extracted markdown."""

from __future__ import annotations

import re
from typing import Any

from pdf_capture_mcp.config import get_logger
from pdf_capture_mcp.types import QCResult

logger = get_logger("quality.qc_gate")

# ── Quality dimensions ──────────────────────────────────────────────────────

# Minimum thresholds for each dimension
_HARD_FLOORS: dict[str, float] = {
    "text_completeness": 0.3,  # At least 30% of expected text extracted
    "heading_structure": 0.1,  # At least some headings present
    "formula_integrity": 0.5,  # At least 50% of formulas are valid
}

_WARN_THRESHOLDS: dict[str, float] = {
    "text_completeness": 0.6,
    "heading_structure": 0.4,
    "formula_integrity": 0.8,
    "table_coverage": 0.5,
}


def _assess_text_completeness(text: str, page_count: int) -> float:
    """Assess text extraction completeness based on chars-per-page ratio."""
    if page_count <= 0:
        return 1.0 if len(text) > 100 else 0.0
    chars_per_page = len(text.strip()) / page_count
    # Expect ~1500-3000 chars per page for typical documents
    if chars_per_page >= 1000:
        return 1.0
    if chars_per_page >= 500:
        return 0.8
    if chars_per_page >= 200:
        return 0.5
    if chars_per_page >= 50:
        return 0.3
    return 0.1


def _assess_heading_structure(text: str) -> float:
    """Assess whether the document has proper heading structure."""
    headings = re.findall(r"^#{1,6}\s+.+$", text, re.MULTILINE)
    if len(headings) >= 5:
        return 1.0
    if len(headings) >= 3:
        return 0.7
    if len(headings) >= 1:
        return 0.4
    return 0.1


def _assess_formula_integrity(text: str) -> float:
    """Assess formula quality (ratio of valid vs broken formulas)."""
    # Count all formula instances
    all_formulas = re.findall(r"\$[^$]+\$|\$\$[^$]+\$\$", text)
    if not all_formulas:
        return 1.0  # No formulas → not applicable, pass

    broken = 0
    for f in all_formulas:
        inner = f.strip("$").strip()
        # Only unmistakably damaged content counts as broken. Single-character
        # formulas like $k$, $M$, $K$ are perfectly legitimate math symbols
        # (calibrated after a real paper was HALTed for exactly this).
        if not inner or set(inner) <= {"?"} or "???" in inner:
            broken += 1

    return 1.0 - (broken / len(all_formulas)) if all_formulas else 1.0


def _assess_table_coverage(text: str, expected_tables: int = 0) -> float:
    """Assess table extraction coverage."""
    md_tables = len(re.findall(r"^\|.+\|$", text, re.MULTILINE)) // 3  # rough count
    if expected_tables <= 0:
        return 1.0 if md_tables > 0 else 0.5
    return min(1.0, md_tables / max(expected_tables, 1))


def run_qc_gate(
    text: str,
    *,
    page_count: int = 0,
    expected_tables: int = 0,
) -> QCResult:
    """Run multi-dimensional quality assessment on extracted markdown.

    Dimensions:
    - text_completeness: chars-per-page ratio
    - heading_structure: presence of markdown headings
    - formula_integrity: ratio of valid vs broken formulas
    - table_coverage: table extraction completeness

    Verdicts:
    - PASS: All dimensions above warn thresholds
    - WARN: Some dimensions below warn but above hard floors
    - HALT: Any dimension below hard floor

    Args:
        text: Extracted markdown text.
        page_count: Number of PDF pages (for completeness assessment).
        expected_tables: Expected number of tables (from extraction report).

    Returns:
        QCResult with verdict, dimensions, and failed/warn lists.
    """
    dimensions: dict[str, Any] = {}

    # Assess each dimension
    scores = {
        "text_completeness": _assess_text_completeness(text, page_count),
        "heading_structure": _assess_heading_structure(text),
        "formula_integrity": _assess_formula_integrity(text),
        "table_coverage": _assess_table_coverage(text, expected_tables),
    }

    failed: list[str] = []
    warned: list[str] = []

    for dim, score in scores.items():
        hard_floor = _HARD_FLOORS.get(dim, 0.0)
        warn_threshold = _WARN_THRESHOLDS.get(dim, 0.5)

        status = "pass"
        if score < hard_floor:
            status = "fail"
            failed.append(dim)
        elif score < warn_threshold:
            status = "warn"
            warned.append(dim)

        dimensions[dim] = {
            "value": round(score, 3),
            "hard_floor": hard_floor,
            "warn_threshold": warn_threshold,
            "status": status,
        }

    # Determine verdict
    if failed:
        verdict = "HALT"
    elif warned:
        verdict = "WARN"
    else:
        verdict = "PASS"

    result = QCResult(
        verdict=verdict,
        dimensions=dimensions,
        failed_dimensions=failed,
        warn_dimensions=warned,
    )

    logger.info(
        "QC gate: %s (failed=%s, warn=%s)",
        verdict,
        failed or "none",
        warned or "none",
    )
    return result
