"""Model cache inspection and pre-download helpers.

The marker engine (surya) lazily downloads its models the first time an
internal inference server spawns — inside a 300s health-check window. On slow
networks the download cannot finish in time, the server is force-killed, and
the caller only sees an opaque timeout. Pre-downloading models outside that
window (via the ``download_models`` tool) avoids the problem entirely.

Model inventory for the marker engine:

===========================  =========================  ==================
Model                        Source                     Purpose
===========================  =========================  ==================
datalab-to/surya_layout2     HuggingFace Hub            layout detection
datalab-to/surya-ocr-2-gguf  HuggingFace Hub            OCR (llama.cpp)
text_detection/*             models.datalab.to (S3)     text detection
ocr_error_detection/*        models.datalab.to (S3)     OCR error check
===========================  =========================  ==================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pdf_capture_mcp.config import get_logger

logger = get_logger("models")

# HF repos required by marker/surya. GGUF filenames must match surya settings.
HF_LAYOUT_REPO = "datalab-to/surya_layout2"
HF_OCR_GGUF_REPO = "datalab-to/surya-ocr-2-gguf"
HF_OCR_GGUF_FILES = ("surya-2.gguf", "surya-2-mmproj.gguf")


def _hf_hub_cache() -> Path:
    """Resolve the huggingface hub cache dir (respects HF_HOME etc.)."""
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except ImportError:
        return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_repo_cached(repo_id: str, filenames: tuple[str, ...] = ()) -> bool:
    """Check whether an HF repo snapshot (and given files) exist locally."""
    repo_dir = _hf_hub_cache() / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not repo_dir.is_dir():
        return False
    for snapshot in repo_dir.iterdir():
        if not snapshot.is_dir():
            continue
        if all((snapshot / f).exists() for f in filenames):
            return True
    return False


def _surya_settings() -> Any | None:
    """Import surya settings if the marker stack is installed."""
    try:
        from surya.settings import settings

        return settings
    except ImportError:
        return None


def _s3_model_cached(settings: Any, checkpoint: str) -> bool:
    """Check whether a surya s3:// model is fully present in the local cache."""
    from surya.common.s3 import check_manifest

    local = os.path.join(settings.MODEL_CACHE_DIR, checkpoint.replace("s3://", ""))
    return bool(check_manifest(local))


def check_model_cache() -> dict[str, Any]:
    """Report local cache status for every model the marker engine needs.

    Returns a dict with per-model status and an aggregate ``all_cached`` flag.
    When marker/surya is not installed, returns ``{"available": False}``.
    """
    settings = _surya_settings()
    if settings is None:
        return {
            "available": False,
            "note": "marker/surya not installed — model check skipped.",
        }

    models: dict[str, Any] = {
        "surya_layout2": {
            "source": f"hf://{HF_LAYOUT_REPO}",
            "cached": _hf_repo_cached(HF_LAYOUT_REPO),
        },
        "surya_ocr_gguf": {
            "source": f"hf://{HF_OCR_GGUF_REPO}",
            "cached": _hf_repo_cached(HF_OCR_GGUF_REPO, HF_OCR_GGUF_FILES),
        },
    }
    for key, checkpoint in (
        ("text_detection", settings.DETECTOR_MODEL_CHECKPOINT),
        ("ocr_error_detection", settings.OCR_ERROR_MODEL_CHECKPOINT),
    ):
        try:
            cached = _s3_model_cached(settings, checkpoint)
        except Exception as exc:  # noqa: BLE001 — cache probe must not break env check
            logger.warning("S3 model cache check failed for %s: %s", key, exc)
            cached = False
        models[key] = {"source": checkpoint, "cached": cached}

    return {
        "available": True,
        "models": models,
        "all_cached": all(m["cached"] for m in models.values()),
    }


def apply_mirror_compat_env() -> list[str]:
    """Apply env tweaks needed when downloading through an HF mirror.

    hf-mirror.com does not implement the Xet transfer protocol; large files
    fail mid-download unless Xet is disabled. Returns a list of human-readable
    notes describing what was changed.
    """
    notes: list[str] = []
    endpoint = os.environ.get("HF_ENDPOINT", "")
    if endpoint and "huggingface.co" not in endpoint:
        if os.environ.get("HF_HUB_DISABLE_XET") != "1":
            os.environ["HF_HUB_DISABLE_XET"] = "1"
            notes.append(
                f"HF_ENDPOINT points to a mirror ({endpoint}); "
                "set HF_HUB_DISABLE_XET=1 (mirrors do not support Xet transfer)."
            )
    return notes


def download_marker_models(progress: Any = None) -> dict[str, Any]:
    """Download all marker/surya models to the local caches.

    Runs OUTSIDE surya's 300s inference-server health-check window, so slow
    networks only cost time instead of causing an opaque spawn failure.

    Args:
        progress: Optional callable ``progress(stage: str)`` for job updates.

    Returns:
        Dict summarizing what was downloaded vs already cached.
    """
    settings = _surya_settings()
    if settings is None:
        raise RuntimeError(
            "marker/surya is not installed. Run install_engine or: "
            "pip install pdf-capture-mcp[marker]"
        )

    def _report(stage: str) -> None:
        logger.info("download_models: %s", stage)
        if progress is not None:
            progress(stage)

    notes = apply_mirror_compat_env()
    summary: dict[str, Any] = {"downloaded": [], "already_cached": [], "notes": notes}

    from huggingface_hub import hf_hub_download, snapshot_download

    # HF: layout model (small, ~150MB)
    if _hf_repo_cached(HF_LAYOUT_REPO):
        summary["already_cached"].append(HF_LAYOUT_REPO)
    else:
        _report(f"downloading {HF_LAYOUT_REPO}")
        snapshot_download(HF_LAYOUT_REPO)
        summary["downloaded"].append(HF_LAYOUT_REPO)

    # HF: OCR GGUF weights (~1.5GB total)
    if _hf_repo_cached(HF_OCR_GGUF_REPO, HF_OCR_GGUF_FILES):
        summary["already_cached"].append(HF_OCR_GGUF_REPO)
    else:
        for filename in HF_OCR_GGUF_FILES:
            _report(f"downloading {HF_OCR_GGUF_REPO}/{filename}")
            hf_hub_download(repo_id=HF_OCR_GGUF_REPO, filename=filename)
        summary["downloaded"].append(HF_OCR_GGUF_REPO)

    # datalab S3 CDN models
    from surya.common.s3 import check_manifest, download_directory

    for checkpoint in (
        settings.DETECTOR_MODEL_CHECKPOINT,
        settings.OCR_ERROR_MODEL_CHECKPOINT,
    ):
        remote = checkpoint.replace("s3://", "")
        local = os.path.join(settings.MODEL_CACHE_DIR, remote)
        os.makedirs(local, exist_ok=True)
        if check_manifest(local):
            summary["already_cached"].append(checkpoint)
            continue
        _report(f"downloading {checkpoint}")
        download_directory(remote, local)
        summary["downloaded"].append(checkpoint)

    _report("all models ready")
    return summary
