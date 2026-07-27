"""Marker engine: PDF-to-Markdown via the marker-pdf library.

Default engine. Requires: pip install marker-pdf (includes PyTorch).
Models are auto-downloaded to ~/.cache/ on first use.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pdf_capture_mcp.config import get_logger
from pdf_capture_mcp.types import ExtractReport

logger = get_logger("engines.marker")


def restart_inference_services() -> int:
    """Kill marker's helper inference services so the next call starts clean.

    Pilot forensics: one oversized document flooded the resident
    llama-server queue and every SUBSEQUENT document timed out against the
    backlog — the failure was contagious. marker re-spawns these services
    on demand, so killing them is always safe. Returns processes signalled.
    """
    import subprocess

    killed = 0
    for pattern in ("llama-server", "surya.[a-z_]*.server"):
        try:
            r = subprocess.run(["pkill", "-f", pattern], capture_output=True, timeout=10)
            if r.returncode == 0:
                killed += 1
        except Exception:  # noqa: BLE001 — best-effort hygiene
            pass
    if killed:
        import time as _time

        _time.sleep(2)  # let sockets close before the next spawn
        logger.info("Inference services restarted (%d pattern(s) matched)", killed)
    return killed


# Lazy-loaded model dict (shared across calls to avoid re-initialization)
_model_dict: Any = None


def _get_model_dict() -> Any:
    """Load marker models once, cache for subsequent calls."""
    global _model_dict
    if _model_dict is not None:
        return _model_dict
    from marker.models import create_model_dict

    logger.info("Loading marker models (first run may download ~1GB)...")
    _model_dict = create_model_dict()
    logger.info("Marker models loaded.")
    return _model_dict


class MarkerEngine:
    """PDF extraction engine using the marker-pdf library."""

    @property
    def name(self) -> str:
        return "marker"

    def is_available(self) -> bool:
        """Check if marker-pdf is installed."""
        try:
            import marker  # noqa: F401

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
        mode: str = "balanced",
        disable_ocr: bool = False,
        force_ocr: bool = False,
        **kwargs: Any,
    ) -> ExtractReport:
        """Extract PDF to Markdown using marker.

        Args:
            pdf_path: Source PDF file.
            out_dir: Output directory for markdown + images.
            enable_formula: Enable inline math / equation OCR.
            enable_table: Enable table reconstruction.
            language: Language hint (unused by marker, kept for interface compat).
            mode: 'balanced' (GPU, highest quality) or 'fast' (CPU-optimized).
            disable_ocr: Disable all VLM calls (pure text-layer extraction).
            force_ocr: Force OCR on all pages.
        """
        if not self.is_available():
            return ExtractReport(
                ok=False,
                engine=self.name,
                error="marker-pdf not installed. Run: pip install marker-pdf",
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        try:
            from marker.config.parser import ConfigParser
            from marker.converters.pdf import PdfConverter
            from marker.output import text_from_rendered

            # Build marker config
            config: dict[str, Any] = {
                "output_format": "markdown",
                "mode": mode,
            }
            # Segment/preview support: marker accepts a page_range string
            # ("0-79"). Previously accepted via **kwargs but never wired
            # into the config — fixed in v0.9.3 (segmented extraction
            # depends on it).
            page_range = kwargs.get("page_range", "")
            if page_range:
                config["page_range"] = page_range
            if disable_ocr:
                config["disable_ocr"] = True
            if force_ocr:
                config["force_ocr"] = True
            if not enable_formula:
                config["ocr_inline_math"] = False

            config_parser = ConfigParser(config)
            converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=_get_model_dict(),
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
            )

            rendered = converter(str(pdf_path))
            text, _, images = text_from_rendered(rendered)

            # Write markdown output
            md_path = out_dir / "full_text.md"
            md_path.write_text(text, encoding="utf-8")

            # Save images
            images_dir = out_dir / "images"
            if images:
                images_dir.mkdir(parents=True, exist_ok=True)
                for img_name, img_data in images.items():
                    img_path = images_dir / img_name
                    if hasattr(img_data, "save"):
                        img_data.save(str(img_path))

            # Fix image references: marker emits bare filenames (e.g.
            # `![](_page_2_Figure_0.jpeg)`) but we save images into an
            # `images/` subdirectory, so links must include the prefix.
            if images:
                for img_name in images:
                    bare_ref = f"]({img_name})"
                    fixed_ref = f"](images/{img_name})"
                    if bare_ref in text:
                        text = text.replace(bare_ref, fixed_ref)
                # Re-write the md with fixed paths
                md_path.write_text(text, encoding="utf-8")

            elapsed = round(time.time() - t0, 2)

            # Count pages from metadata; marker's metadata key set varies
            # across versions — fall back to counting via pymupdf so the
            # knowledge package never reports "0 pages" (v0.7.1 fix).
            page_count = 0
            metadata = getattr(rendered, "metadata", {}) or {}
            if isinstance(metadata, dict):
                page_count = metadata.get("page_count", 0)
            if not page_count:
                try:
                    import fitz

                    with fitz.open(str(pdf_path)) as _doc:
                        page_count = len(_doc)
                except Exception:  # noqa: BLE001 — count stays 0 if even this fails
                    pass

            logger.info(
                "Marker extraction complete: %d chars, %d images, %.1fs",
                len(text),
                len(images) if images else 0,
                elapsed,
            )

            return ExtractReport(
                ok=True,
                engine=self.name,
                full_text_md=str(md_path),
                content_dir=str(out_dir),
                page_count=page_count,
                image_count=len(images) if images else 0,
                table_count=0,  # marker handles tables inline
                elapsed_seconds=elapsed,
                metadata={"mode": mode},
            )

        except Exception as exc:
            elapsed = round(time.time() - t0, 2)
            logger.error("Marker extraction failed: %s", exc)
            return ExtractReport(
                ok=False,
                engine=self.name,
                elapsed_seconds=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            )
