"""TATR table structure detection via Table Transformer (optional, requires torch)."""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("pdf_capture_mcp.extractors.tatr")

# Lazy-loaded model cache
_tatr_cache: tuple[Any, Any, Any, Any] | None = None


def _load_models() -> tuple[Any, Any, Any, Any]:
    """Load TATR detection + structure models (cached)."""
    global _tatr_cache
    if _tatr_cache is not None:
        return _tatr_cache

    import torch
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Loading TATR models on %s...", device)

    det_processor = AutoImageProcessor.from_pretrained(
        "microsoft/table-transformer-detection", local_files_only=False
    )
    det_model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-detection", local_files_only=False
    ).to(device)

    struct_processor = AutoImageProcessor.from_pretrained(
        "microsoft/table-transformer-structure-recognition", local_files_only=False
    )
    struct_model = TableTransformerForObjectDetection.from_pretrained(
        "microsoft/table-transformer-structure-recognition", local_files_only=False
    ).to(device)

    _tatr_cache = (det_processor, det_model, struct_processor, struct_model)
    logger.info("TATR models loaded.")
    return _tatr_cache


def extract_tables_tatr(pdf_path: str, *, max_tables: int = 50) -> dict[str, Any]:
    """Extract table structures from PDF using Table Transformer.

    Args:
        pdf_path: Path to the PDF file.
        max_tables: Maximum tables to detect.

    Returns:
        Dict with ok, count, tables, elapsed_s.
    """
    t0 = time.time()

    try:
        import fitz
        import torch
    except ImportError as exc:
        return {
            "ok": False,
            "error": f"Missing dependency: {exc}. Install: pip install pdf-capture-mcp[ml]",
        }

    try:
        det_processor, det_model, _, _ = _load_models()
    except Exception as exc:
        return {"ok": False, "error": f"Failed to load TATR models: {exc}"}

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tables: list[dict[str, Any]] = []

    try:
        doc = fitz.open(pdf_path)
        for page_idx in range(min(doc.page_count, 100)):  # Cap at 100 pages
            if len(tables) >= max_tables:
                break

            page = doc[page_idx]
            pix = page.get_pixmap(dpi=150)
            from PIL import Image

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Run detection
            inputs = det_processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = det_model(**inputs)

            # Post-process
            results = det_processor.post_process_object_detection(
                outputs, threshold=0.7, target_sizes=[(pix.height, pix.width)]
            )[0]

            for score, box in zip(results["scores"], results["boxes"]):
                if len(tables) >= max_tables:
                    break
                x1, y1, x2, y2 = box.tolist()
                tables.append(
                    {
                        "page": page_idx,
                        "bbox": [round(x1), round(y1), round(x2), round(y2)],
                        "confidence": round(float(score), 3),
                    }
                )

        doc.close()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {
        "ok": True,
        "count": len(tables),
        "tables": tables,
        "elapsed_s": round(time.time() - t0, 2),
    }
