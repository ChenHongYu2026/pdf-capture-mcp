"""Table extraction via pdfplumber (rule-based lattice/stream detection)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pdf_capture_mcp.config import get_logger

logger = get_logger("extractors.tables")

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore[assignment]


def _score_table(rows: list[list[str]]) -> float:
    """Score a table's structural quality (0.0-1.0)."""
    if not rows:
        return 0.0
    col_counts = [len(r) for r in rows]
    if not col_counts:
        return 0.0
    # Consistency: how uniform are column counts
    max_cols = max(col_counts)
    if max_cols == 0:
        return 0.0
    consistency = sum(1 for c in col_counts if c == max_cols) / len(col_counts)
    # Density: how many cells are non-empty
    total_cells = sum(col_counts)
    filled = sum(1 for r in rows for cell in r if cell.strip())
    density = filled / max(total_cells, 1)
    return round(0.6 * consistency + 0.4 * density, 3)


def _table_to_markdown(rows: list[list[str]]) -> str:
    """Convert a list of rows to a Markdown table string."""
    if not rows:
        return ""
    # Normalize column count
    max_cols = max(len(r) for r in rows)
    normalized = [r + [""] * (max_cols - len(r)) for r in rows]

    lines = []
    # Header
    header = normalized[0]
    lines.append("| " + " | ".join(cell.strip() for cell in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    # Body
    for row in normalized[1:]:
        lines.append("| " + " | ".join(cell.strip() for cell in row) + " |")
    return "\n".join(lines)


def extract_tables(pdf_path: str | Path, *, max_tables: int = 50) -> dict[str, Any]:
    """Extract tables from a PDF using pdfplumber.

    Args:
        pdf_path: Path to the PDF file.
        max_tables: Maximum number of tables to extract.

    Returns:
        Dict with ok, tables (list of {page, markdown, rows, score}), stats.
    """
    if pdfplumber is None:
        return {"ok": False, "error": "pdfplumber not installed"}

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return {"ok": False, "error": f"File not found: {pdf_path}"}

    tables: list[dict[str, Any]] = []
    strategy_counts: dict[str, int] = {}

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                if len(tables) >= max_tables:
                    break

                # Try lattice strategy first (ruled tables)
                for strategy in ("lattice", "stream"):
                    try:
                        found = page.extract_tables(
                            {"vertical_strategy": strategy, "horizontal_strategy": strategy}
                        )
                    except Exception:
                        found = []

                    for raw_rows in found or []:
                        if not raw_rows or len(raw_rows) < 2:
                            continue
                        # Clean None values
                        rows = [
                            [str(cell or "").strip() for cell in row]
                            for row in raw_rows
                        ]
                        score = _score_table(rows)
                        if score < 0.3:
                            continue  # Skip low-quality tables

                        tables.append({
                            "page": page_idx,
                            "strategy": strategy,
                            "markdown": _table_to_markdown(rows),
                            "rows": rows,
                            "score": score,
                        })
                        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

                        if len(tables) >= max_tables:
                            break

    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    scores = [t["score"] for t in tables]
    return {
        "ok": True,
        "tables": tables,
        "stats": {
            "total_tables": len(tables),
            "avg_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "by_strategy": strategy_counts,
        },
    }
