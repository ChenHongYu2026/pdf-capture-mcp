"""Configuration management via environment variables."""

from __future__ import annotations

import logging
import os
from pathlib import Path

# ── Engine selection ────────────────────────────────────────────────────────

ENGINE_MARKER = "marker"
ENGINE_MINERU = "mineru"
ENGINE_AUTO = "auto"

VALID_ENGINES = (ENGINE_MARKER, ENGINE_MINERU, ENGINE_AUTO)


def get_default_engine() -> str:
    """Return the configured default engine name."""
    engine = os.getenv("PDF_CAPTURE_ENGINE", ENGINE_AUTO).strip().lower()
    if engine not in VALID_ENGINES:
        return ENGINE_AUTO
    return engine


# ── Paths ───────────────────────────────────────────────────────────────────


def get_cache_dir() -> Path:
    """Return the model/data cache directory."""
    env = os.getenv("PDF_CAPTURE_CACHE_DIR", "").strip()
    if env:
        p = Path(env).expanduser()
    else:
        p = Path.home() / ".cache" / "pdf-capture-mcp"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_mineru_venv_dir() -> Path:
    """Return the MinerU virtual environment directory."""
    env = os.getenv("PDF_CAPTURE_MINERU_VENV", "").strip()
    if env:
        return Path(env).expanduser()
    return get_cache_dir() / "venv-mineru"


# ── Logging ─────────────────────────────────────────────────────────────────

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_configured = False


def setup_logging() -> None:
    """Configure package-level logging (idempotent)."""
    global _configured
    if _configured:
        return
    _configured = True

    level_name = os.getenv("PDF_CAPTURE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger("pdf_capture_mcp")
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_LOG_DATE_FMT))
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger for this package."""
    setup_logging()
    return logging.getLogger(f"pdf_capture_mcp.{name}")
