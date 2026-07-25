"""Lightweight document classifier based on PDF structural features."""

from __future__ import annotations

import re
from pathlib import Path

from pdf_capture_mcp.config import get_logger
from pdf_capture_mcp.types import ClassifyResult

logger = get_logger("classifier")

# Document type keywords for filename-based heuristics
_FILENAME_HINTS: dict[str, list[str]] = {
    "academic_paper": ["paper", "journal", "arxiv", "ieee", "acm", "thesis", "dissertation"],
    "consulting_report": ["mckinsey", "bcg", "bain", "deloitte", "accenture", "consulting"],
    "policy_doc": ["policy", "regulation", "guideline", "standard", "whitepaper", "白皮书"],
    "tech_research": ["technical", "engineering", "architecture", "design", "spec"],
    "industry_report": ["market", "industry", "sector", "outlook", "forecast", "行业"],
}

# Formula indicators in text
_FORMULA_PATTERN = re.compile(
    r"\$[^$]+\$|\\frac|\\sum|\\int|\\begin\{equation|\\begin\{align",
    re.IGNORECASE,
)

# Table indicators
_TABLE_PATTERN = re.compile(r"^\|.+\|$", re.MULTILINE)


def classify_document(pdf_path: str | Path) -> ClassifyResult:
    """Classify a PDF document using structural heuristics.

    Analyzes filename, page count, text density, formula/table presence
    to determine the most likely document type.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        ClassifyResult with doc_type, confidence, and detected features.
    """
    pdf_path = Path(pdf_path)
    result = ClassifyResult(source="heuristic")

    # ── Filename-based hints ────────────────────────────────────────────
    stem_lower = pdf_path.stem.lower()
    filename_scores: dict[str, int] = {}
    for doc_type, keywords in _FILENAME_HINTS.items():
        score = sum(1 for kw in keywords if kw in stem_lower)
        if score > 0:
            filename_scores[doc_type] = score

    # ── PDF content analysis ────────────────────────────────────────────
    try:
        import fitz
        import pymupdf4llm  # noqa: F401

        doc = fitz.open(str(pdf_path))
        result.page_count = doc.page_count

        # Sample text from first 5 pages
        sample_text = ""
        for i in range(min(5, doc.page_count)):
            sample_text += doc[i].get_text()

        # Detect formulas
        result.has_formulas = bool(_FORMULA_PATTERN.search(sample_text))

        # Detect tables
        result.has_tables = bool(_TABLE_PATTERN.search(sample_text))

        # Language detection
        cjk_chars = sum(1 for c in sample_text if "\u4e00" <= c <= "\u9fff")
        latin_chars = sum(1 for c in sample_text if c.isascii() and c.isalpha())
        if cjk_chars > latin_chars * 0.5:
            result.language = "ch"
        elif latin_chars > 0:
            result.language = "en"

        doc.close()
    except ImportError:
        logger.debug("pymupdf not available for content analysis")
    except Exception as exc:
        logger.debug("Content analysis failed: %s", exc)

    # ── Scoring ─────────────────────────────────────────────────────────
    scores: dict[str, float] = {}

    # Filename signal
    for doc_type, score in filename_scores.items():
        scores[doc_type] = scores.get(doc_type, 0) + score * 0.3

    # Structural signals
    if result.has_formulas:
        scores["academic_paper"] = scores.get("academic_paper", 0) + 0.4
    if result.has_tables and result.page_count > 20:
        scores["consulting_report"] = scores.get("consulting_report", 0) + 0.2
        scores["industry_report"] = scores.get("industry_report", 0) + 0.2
    if result.page_count > 50:
        scores["consulting_report"] = scores.get("consulting_report", 0) + 0.1
    if result.page_count <= 15 and result.has_formulas:
        scores["academic_paper"] = scores.get("academic_paper", 0) + 0.2

    # Pick best
    if scores:
        best_type = max(scores, key=lambda k: scores[k])
        best_score = min(scores[best_type], 1.0)
        result.doc_type = best_type
        result.confidence = round(best_score, 2)
        if filename_scores.get(best_type, 0) > 0:
            result.source = "filename"
    else:
        result.doc_type = "general"
        result.confidence = 0.1

    logger.info(
        "Classified %s as %s (confidence=%.2f, source=%s)",
        pdf_path.name,
        result.doc_type,
        result.confidence,
        result.source,
    )
    return result
