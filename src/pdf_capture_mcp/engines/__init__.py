"""Extraction engines for PDF-to-Markdown conversion."""

from __future__ import annotations

from pdf_capture_mcp.config import (
    ENGINE_AUTO,
    ENGINE_MARKER,
    ENGINE_MINERU,
    get_default_engine,
    get_logger,
)
from pdf_capture_mcp.engines.base import ExtractEngine

logger = get_logger("engines")

# Engine name for the lightweight pymupdf fallback
ENGINE_PYMUPDF = "pymupdf"

# Cached engine instances
_engines: dict[str, ExtractEngine] = {}


def get_engine(name: str = "") -> ExtractEngine:
    """Get an extraction engine by name.

    Engines are lazily imported and instantiated on first access.
    Priority for 'auto': marker > mineru > pymupdf.

    Args:
        name: Engine name ('marker', 'mineru', 'pymupdf', 'auto'). Defaults to env config.

    Returns:
        An ExtractEngine instance.

    Raises:
        RuntimeError: If no engine is available.
    """
    if not name:
        name = get_default_engine()

    if name == ENGINE_AUTO:
        # Prefer marker, fallback to mineru, then pymupdf (always available)
        for candidate in (ENGINE_MARKER, ENGINE_MINERU, ENGINE_PYMUPDF):
            try:
                engine = get_engine(candidate)
                if engine.is_available():
                    return engine
            except RuntimeError:
                continue
        raise RuntimeError(
            "No extraction engine available. "
            "Install marker: pip install pdf-capture-mcp[marker] "
            "or setup MinerU: pdf-capture-mcp setup-mineru"
        )

    if name not in _engines:
        if name == ENGINE_MARKER:
            from pdf_capture_mcp.engines.marker_engine import MarkerEngine

            _engines[name] = MarkerEngine()
        elif name == ENGINE_MINERU:
            from pdf_capture_mcp.engines.mineru_engine import MineruEngine

            _engines[name] = MineruEngine()
        elif name == ENGINE_PYMUPDF:
            from pdf_capture_mcp.engines.pymupdf_engine import PymupdfEngine

            _engines[name] = PymupdfEngine()
        else:
            raise RuntimeError(f"Unknown engine: {name!r}. Valid: marker, mineru, pymupdf, auto")

    return _engines[name]


__all__ = ["ExtractEngine", "MarkerEngine", "MineruEngine", "PymupdfEngine", "get_engine"]
