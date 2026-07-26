"""PymupdfEngine: lightweight PDF-to-Markdown via pymupdf4llm.

Always-available fallback engine (pymupdf4llm is a base dependency).
Fastest startup, no model downloads, good quality for text-based PDFs.
For scanned documents or complex layouts, prefer marker or MinerU.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pdf_capture_mcp.config import get_logger
from pdf_capture_mcp.types import ExtractReport

logger = get_logger("engines.pymupdf")


class PymupdfEngine:
    """Lightweight PDF extraction engine using pymupdf4llm.

    This engine is always available as part of the base installation.
    It provides fast, rule-based PDF-to-Markdown conversion without
    requiring any ML models or heavy dependencies.

    Best for:
    - Text-based PDFs with clear structure
    - Quick extraction when speed matters
    - Environments where installing marker/MinerU is impractical

    Limitations:
    - No OCR for scanned documents
    - Basic formula handling (no deep learning recognition)
    - Simple table detection (no structure reconstruction)
    """

    @property
    def name(self) -> str:
        return "pymupdf"

    def is_available(self) -> bool:
        """Check if pymupdf4llm is installed (should always be True)."""
        try:
            import pymupdf4llm  # noqa: F401

            return True
        except ImportError:
            return False

    def extract(
        self,
        pdf_path: Path,
        out_dir: Path,
        *,
        enable_formula: bool = True,
        enable_table: bool = True,
        language: str = "auto",
        pages: list[int] | None = None,
        **kwargs: Any,
    ) -> ExtractReport:
        """Extract PDF to Markdown using pymupdf4llm.

        Args:
            pdf_path: Source PDF file.
            out_dir: Output directory for markdown + images.
            enable_formula: Kept for interface compat (pymupdf extracts inline math).
            enable_table: Kept for interface compat (pymupdf extracts basic tables).
            language: Language hint (unused, kept for interface compat).
            pages: Optional list of 0-based page indices to extract (None = all).
        """
        if not self.is_available():
            return ExtractReport(
                ok=False,
                engine=self.name,
                error="pymupdf4llm not installed. Run: pip install pymupdf4llm",
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        try:
            import pymupdf4llm

            # Build kwargs for pymupdf4llm.to_markdown
            md_kwargs: dict[str, Any] = {}
            if pages is not None:
                md_kwargs["pages"] = pages

            # Extract markdown text
            md_text = pymupdf4llm.to_markdown(str(pdf_path), **md_kwargs)

            # Write markdown output
            md_path = out_dir / "full_text.md"
            md_path.write_text(md_text, encoding="utf-8")

            elapsed = round(time.time() - t0, 2)

            # Get page count via fitz (bundled with pymupdf4llm)
            page_count = 0
            image_count = 0
            try:
                import fitz

                doc = fitz.open(str(pdf_path))
                page_count = doc.page_count
                # Count images across pages
                for page in doc:
                    image_count += len(page.get_images(full=True))
                doc.close()
            except Exception:
                pass

            logger.info(
                "Pymupdf extraction complete: %d chars, %d pages, %.1fs",
                len(md_text),
                page_count,
                elapsed,
            )

            return ExtractReport(
                ok=True,
                engine=self.name,
                full_text_md=str(md_path),
                content_dir=str(out_dir),
                page_count=page_count,
                image_count=image_count,
                table_count=0,  # pymupdf handles tables inline as markdown
                elapsed_seconds=elapsed,
                metadata={"note": "Lightweight engine. For higher quality, install marker."},
            )

        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            logger.error("Pymupdf extraction failed: %s", exc)
            return ExtractReport(
                ok=False,
                engine=self.name,
                elapsed_seconds=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            )
