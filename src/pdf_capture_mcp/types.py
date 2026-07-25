"""Shared type definitions for the PDF capture pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractReport:
    """Result of a PDF extraction engine run."""

    ok: bool
    engine: str = ""
    full_text_md: str = ""
    content_json: str = ""
    content_dir: str = ""
    page_count: int = 0
    image_count: int = 0
    table_count: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QCResult:
    """Result of quality gate assessment."""

    verdict: str = "PASS"  # PASS | WARN | HALT
    dimensions: dict[str, Any] = field(default_factory=dict)
    failed_dimensions: list[str] = field(default_factory=list)
    warn_dimensions: list[str] = field(default_factory=list)
    repair_log: list[dict[str, Any]] = field(default_factory=list)
    repaired_text: str | None = None


@dataclass
class ClassifyResult:
    """Result of document classification."""

    doc_type: str = "general"
    confidence: float = 0.0
    source: str = "heuristic"  # heuristic | filename | llm
    has_formulas: bool = False
    has_tables: bool = False
    page_count: int = 0
    language: str = ""


@dataclass
class PipelineResult:
    """Final result of the full capture pipeline."""

    ok: bool
    markdown_text: str = ""
    markdown_path: str = ""
    pipeline_root: str = ""
    title: str = ""
    classification: ClassifyResult | None = None
    extract: ExtractReport | None = None
    qc: QCResult | None = None
    stage_reports: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    stage: str = ""
