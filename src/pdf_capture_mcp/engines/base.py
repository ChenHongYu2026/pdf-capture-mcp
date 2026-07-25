"""Abstract base for PDF extraction engines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pdf_capture_mcp.types import ExtractReport


@runtime_checkable
class ExtractEngine(Protocol):
    """Protocol that all extraction engines must implement."""

    @property
    def name(self) -> str:
        """Human-readable engine name (e.g. 'marker', 'mineru')."""
        ...

    def is_available(self) -> bool:
        """Check if this engine's dependencies are installed and usable."""
        ...

    def extract(
        self,
        pdf_path: Path,
        out_dir: Path,
        *,
        enable_formula: bool = True,
        enable_table: bool = True,
        language: str = "auto",
        **kwargs: Any,
    ) -> ExtractReport:
        """Extract PDF content to Markdown.

        Args:
            pdf_path: Path to the source PDF file.
            out_dir: Directory to write extraction output.
            enable_formula: Enable formula/equation recognition.
            enable_table: Enable table structure recognition.
            language: Language hint ('auto', 'en', 'ch', etc.).
            **kwargs: Engine-specific options.

        Returns:
            ExtractReport with extraction results.
        """
        ...
