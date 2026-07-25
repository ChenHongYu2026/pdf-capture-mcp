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
from pdf_capture_mcp.engines.marker_engine import MarkerEngine
from pdf_capture_mcp.engines.mineru_engine import MineruEngine

logger = get_logger("engines")

# Cached engine instances
_engines: dict[str, ExtractEngine] = {}


def get_engine(name: str = "") -> ExtractEngine:
    """Get an extraction engine by name.

    Args:
        name: Engine name ('marker', 'mineru', 'auto'). Defaults to env config.

    Returns:
        An ExtractEngine instance.

    Raises:
        RuntimeError: If no engine is available.
    """
    if not name:
        name = get_default_engine()

    if name == ENGINE_AUTO:
        # Prefer marker, fallback to mineru
        for candidate in (ENGINE_MARKER, ENGINE_MINERU):
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
            _engines[name] = MarkerEngine()
        elif name == ENGINE_MINERU:
            _engines[name] = MineruEngine()
        else:
            raise RuntimeError(f"Unknown engine: {name!r}. Valid: marker, mineru, auto")

    return _engines[name]


__all__ = ["ExtractEngine", "MarkerEngine", "MineruEngine", "get_engine"]
