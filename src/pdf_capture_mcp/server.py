"""FastMCP server exposing PDF capture tools with VLM setup guidance."""

from __future__ import annotations

import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from pdf_capture_mcp import __version__
from pdf_capture_mcp.config import get_logger, setup_logging

setup_logging()
logger = get_logger("server")


def _sanitize_proxy_env() -> None:
    """Keep localhost traffic off any configured HTTP proxy.

    The marker engine spawns local inference servers (surya / llama.cpp) and
    health-checks them over 127.0.0.1. If the user has http_proxy/https_proxy
    set without excluding localhost, those health checks get routed to the
    proxy and fail until the 300s startup window expires — which surfaces as
    an opaque timeout. Appending localhost to NO_PROXY prevents that.
    """
    proxy_set = any(
        os.environ.get(var)
        for var in (
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "all_proxy",
            "ALL_PROXY",
        )
    )
    if not proxy_set:
        return
    required = ("localhost", "127.0.0.1")
    for var in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(var, "")
        entries = [e.strip() for e in current.split(",") if e.strip()]
        missing = [host for host in required if host not in entries]
        if missing:
            os.environ[var] = ",".join(entries + missing)
            logger.info(
                "Proxy detected: appended %s to %s so local inference "
                "health checks bypass the proxy.",
                ",".join(missing),
                var,
            )


_sanitize_proxy_env()

mcp = FastMCP(
    name="pdf-capture",
    version=__version__,
    instructions=(
        "PDF Capture Pipeline — multi-phase PDF document extraction service. "
        "Converts PDF to high-quality structured Markdown with formula recognition, "
        "table extraction, and quality control.\n\n"
        "Quick start (tools work immediately, no setup required):\n"
        "- pdf_info: Get PDF metadata and page count.\n"
        "- classify_document: Detect document type.\n"
        "- extract_tables: Rule-based table extraction (pdfplumber).\n"
        "- pdf_to_markdown: Full pipeline (uses best available engine).\n\n"
        "Large PDFs & timeouts:\n"
        "- pdf_to_markdown runs in mode='auto': small PDFs return inline; "
        "large PDFs return a job_id immediately (poll with get_job_status). "
        "This prevents MCP client timeouts on long conversions.\n"
        "- Slow/blocked networks: run download_models FIRST to pre-fetch "
        "marker models (supports HF_ENDPOINT mirrors), otherwise the first "
        "conversion may fail on model downloads.\n\n"
        "Engines (auto-selected by priority: marker > mineru > pymupdf):\n"
        "- pymupdf: Built-in, zero setup, fast. Good for text-based PDFs.\n"
        "- marker: Highest quality for complex layouts. "
        "Install: pip install pdf-capture-mcp[marker]\n"
        "- mineru: Best for multi-column/InDesign. Setup: pdf-capture-mcp setup-mineru\n\n"
        "Optional enhancements (configure when user requests):\n"
        "1. VLM enhancement: Call 'setup_vlm' to configure a vision-capable model "
        "for better table/formula extraction. Supports any OpenAI-compatible provider "
        "(Qwen-VL, GLM-4V, MiniMax, Moonshot, OpenAI, Ollama). "
        "API key via PDF_CAPTURE_VLM_API_KEY env var. Consumes tokens.\n"
        "2. Environment check: Call 'check_environment' to verify all dependencies.\n"
        "3. Engine install: Call 'install_engine' to install marker on behalf of the user.\n\n"
        "SECURITY: Never display or log the user's API key."
    ),
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _resolve_pdf(pdf_path: str) -> Path:
    """Resolve and validate a PDF file path."""
    p = Path(pdf_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")
    if p.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {p.suffix}")
    return p


def _json(data: dict[str, Any]) -> str:
    """Serialize result to JSON string."""

    def _default(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return str(obj)

    return json.dumps(data, ensure_ascii=False, indent=2, default=_default)


# ── Tool 0: setup_vlm ───────────────────────────────────────────────────────


@mcp.tool()
def setup_vlm(
    model: str = "",
    api_key: str = "",
    api_base: str = "",
    provider: str = "",
    action: str = "status",
    policy: str = "full",
) -> str:
    """Configure or check VLM (Vision Language Model) for enhanced PDF extraction.

    VLM enhances table and formula recognition by using a vision-capable AI model
    to re-extract complex content from PDF page images.

    Actions:
    - status: Check current VLM configuration (call this first).
    - enable: Configure VLM. Validates the model supports vision input before saving.
    - disable: Turn off VLM features.

    Supported providers (any model with vision/image capability):
    - Qwen-VL (api_base: https://dashscope.aliyuncs.com/compatible-mode/v1)
    - Zhipu GLM-4V (api_base: https://open.bigmodel.cn/api/paas/v4)
    - MiniMax (api_base: https://api.minimaxi.com/v1)
    - Moonshot (api_base: https://api.moonshot.cn/v1)
    - DeepSeek (api_base: https://api.deepseek.com/v1)
    - OpenAI (api_base: https://api.openai.com/v1)
    - Local Ollama (api_base: http://localhost:11434/v1)
    - Any other provider with chat/completions + image support

    API Key security:
    - Recommended: set PDF_CAPTURE_VLM_API_KEY environment variable.
    - If passed directly, it is stored locally and NEVER displayed in responses.

    Note: Using VLM will consume your API tokens.

    Args:
        model: VLM model name (required for action='enable').
        api_key: API key (or set PDF_CAPTURE_VLM_API_KEY env var).
        api_base: API endpoint URL for the provider.
        provider: Provider name for display (e.g. qwen, zhipu, minimax, moonshot).
        action: 'status', 'enable', or 'disable'.

    Returns:
        JSON with configuration status or validation result. API key is never included.
    """
    from pdf_capture_mcp.llm_client import disable_vlm, get_vlm_config, is_vlm_enabled
    from pdf_capture_mcp.llm_client import setup_vlm as do_setup

    if action == "status":
        config = get_vlm_config()
        enabled = is_vlm_enabled()
        if enabled:
            return _json(
                {
                    "ok": True,
                    "configured": True,
                    "enabled": True,
                    "model": config.get("model", ""),
                    "provider": config.get("provider", ""),
                    "api_base": config.get("api_base", ""),
                    "validated_at": config.get("validated_at", ""),
                    "message": f"VLM active: {config.get('provider')}/{config.get('model')}",
                }
            )
        return _json(
            {
                "ok": True,
                "configured": False,
                "enabled": False,
                "message": (
                    "VLM is not configured. To enable enhanced table/formula extraction, "
                    "call setup_vlm with action='enable', providing a vision-capable model name "
                    "and API endpoint. The model must support image input. "
                    "API key can be set via PDF_CAPTURE_VLM_API_KEY environment variable. "
                    "Using VLM will consume your API tokens. "
                    "Without VLM, rule-based extraction is still available "
                    "and produces good results."
                ),
            }
        )

    if action == "disable":
        result = disable_vlm()
        return _json(result)

    if action == "enable":
        import os

        if not model.strip():
            return _json(
                {
                    "ok": False,
                    "error": "Model name is required. Provide a vision-capable model "
                    "(e.g. qwen-vl-max, glm-4v, minimax-m3, gpt-4o).",
                }
            )

        # Resolve API key: parameter > environment variable
        resolved_key = api_key.strip() or os.getenv("PDF_CAPTURE_VLM_API_KEY", "").strip()
        if not resolved_key:
            return _json(
                {
                    "ok": False,
                    "error": (
                        "API key not found. Either:\n"
                        "  1. Set environment variable: export PDF_CAPTURE_VLM_API_KEY=your_key\n"
                        "  2. Or pass api_key parameter directly (stored locally, never displayed)."
                    ),
                }
            )

        if not api_base.strip():
            return _json(
                {
                    "ok": False,
                    "error": "API endpoint URL is required. Examples:\n"
                    "  Qwen-VL: https://dashscope.aliyuncs.com/compatible-mode/v1\n"
                    "  Zhipu: https://open.bigmodel.cn/api/paas/v4\n"
                    "  MiniMax: https://api.minimaxi.com/v1\n"
                    "  Moonshot: https://api.moonshot.cn/v1\n"
                    "  DeepSeek: https://api.deepseek.com/v1\n"
                    "  OpenAI: https://api.openai.com/v1\n"
                    "  Ollama: http://localhost:11434/v1",
                }
            )

        result = do_setup(
            model=model,
            api_key=resolved_key,
            api_base=api_base,
            provider=provider,
            policy=policy,
        )
        # SECURITY: strip api_key from response before returning to caller
        result.pop("api_key", None)
        return _json(result)

    return _json({"ok": False, "error": f"Unknown action: {action}. Use: status, enable, disable"})


# ── Tool 0.5: check_environment ─────────────────────────────────────────────


@mcp.tool()
def check_environment() -> str:
    """Check if all runtime dependencies are ready for PDF extraction.

    Call this to verify the environment is ready before running extraction tasks.
    Reports available engines, missing dependencies with install commands,
    and filesystem compatibility warnings.

    Checks:
    - Extraction engine availability (marker, MinerU, pymupdf)
    - Core libraries (pdfplumber, pymupdf, numpy, Pillow)
    - Optional features (TATR/torch, VLM client)
    - Filesystem compatibility (symlink support, cache dir writability)

    Returns:
        JSON with ready status, available engines, missing deps, and install hints.
    """
    import importlib
    import os

    results: dict[str, Any] = {
        "ok": True,
        "ready": True,
        "engines": {},
        "dependencies": {},
        "missing": [],
        "warnings": [],
    }

    # ── Check extraction engines ────────────────────────────────────────
    # Marker
    try:
        importlib.import_module("marker")
        results["engines"]["marker"] = {"available": True, "note": "Highest quality engine"}
    except ImportError:
        results["engines"]["marker"] = {
            "available": False,
            "install": "pip install pdf-capture-mcp[marker]",
        }

    # MinerU
    from pdf_capture_mcp.config import get_mineru_venv_dir

    mineru_python = get_mineru_venv_dir() / "bin" / "python3"
    if mineru_python.exists():
        results["engines"]["mineru"] = {"available": True, "note": "Complex layout engine"}
    else:
        results["engines"]["mineru"] = {
            "available": False,
            "install": "pdf-capture-mcp setup-mineru (requires Python 3.11)",
        }

    # Pymupdf (always available as base dependency)
    try:
        importlib.import_module("pymupdf4llm")
        results["engines"]["pymupdf"] = {
            "available": True,
            "note": "Built-in lightweight engine (always available)",
        }
    except ImportError:
        results["engines"]["pymupdf"] = {
            "available": False,
            "install": "pip install pymupdf4llm",
        }

    # At least one engine must be available
    any_engine = any(e["available"] for e in results["engines"].values())
    if not any_engine:
        results["ready"] = False
        results["missing"].append(
            "No extraction engine available. Install at least one:\n"
            "  - marker (recommended): pip install pdf-capture-mcp[marker]\n"
            "  - MinerU (highest quality): pdf-capture-mcp setup-mineru"
        )

    # ── Check core dependencies ─────────────────────────────────────────
    core_deps = {
        "pdfplumber": "pip install pdfplumber",
        "fitz": "pip install pymupdf4llm",
        "numpy": "pip install numpy",
        "PIL": "pip install Pillow",
        "pypdf": "pip install pypdf",
    }
    for module_name, install_cmd in core_deps.items():
        try:
            importlib.import_module(module_name)
            results["dependencies"][module_name] = {"available": True}
        except ImportError:
            results["dependencies"][module_name] = {"available": False, "install": install_cmd}
            results["ready"] = False
            results["missing"].append(f"{module_name}: {install_cmd}")

    # ── Check optional features ─────────────────────────────────────────
    optional_deps = {
        "torch": {"install": "pip install pdf-capture-mcp[ml]", "feature": "TATR table detection"},
        "transformers": {"install": "pip install pdf-capture-mcp[ml]", "feature": "TATR + DePlot"},
    }
    results["optional"] = {}
    for module_name, info in optional_deps.items():
        try:
            importlib.import_module(module_name)
            results["optional"][module_name] = {"available": True, "feature": info["feature"]}
        except ImportError:
            results["optional"][module_name] = {
                "available": False,
                "feature": info["feature"],
                "install": info["install"],
            }

    # VLM status
    from pdf_capture_mcp.llm_client import is_vlm_enabled

    results["vlm"] = {"enabled": is_vlm_enabled()}

    # ── Filesystem compatibility check ────────────────────────────────────
    from pdf_capture_mcp.config import get_cache_dir

    cache_dir = get_cache_dir()
    if not os.access(str(cache_dir), os.W_OK):
        results["warnings"].append(
            f"Cache dir {cache_dir} is not writable. "
            "Set PDF_CAPTURE_CACHE_DIR to a local disk path."
        )

    # Check symlink support (common issue on external/exFAT drives)
    import tempfile

    try:
        test_dir = tempfile.mkdtemp(prefix="pdfcap_symlink_test_", dir=str(cache_dir))
        test_link = Path(test_dir) / "test_link"
        test_target = Path(test_dir) / "test_target"
        test_target.write_text("test")
        test_link.symlink_to(test_target)
        # Cleanup
        test_link.unlink()
        test_target.unlink()
        Path(test_dir).rmdir()
    except OSError:
        results["warnings"].append(
            "Current filesystem does not support symlinks (common on external/exFAT drives). "
            "If installation fails, set: export UV_LINK_MODE=copy"
        )

    # ── Model cache check (marker/surya models) ──────────────────────
    from pdf_capture_mcp.models import check_model_cache

    model_status = check_model_cache()
    results["models"] = model_status
    models_ready = bool(model_status.get("all_cached"))
    results["models_ready"] = models_ready
    if model_status.get("available") and not models_ready:
        missing_models = [
            name for name, info in model_status.get("models", {}).items() if not info.get("cached")
        ]
        results["warnings"].append(
            "Marker models not cached yet: "
            + ", ".join(missing_models)
            + ". First conversion will download them and may be slow or fail "
            "on restricted networks — run download_models first."
        )

    # ── Network configuration report (informational only) ────────────────
    results["network"] = {
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "(default: huggingface.co)"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE", "0") == "1",
        "hf_hub_disable_xet": os.environ.get("HF_HUB_DISABLE_XET", "0") == "1",
        "proxy": os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or "",
        "no_proxy": os.environ.get("NO_PROXY", ""),
    }

    # ── Summary message ─────────────────────────────────────────────────
    if results["ready"]:
        available_engines = [name for name, e in results["engines"].items() if e["available"]]
        results["message"] = (
            f"Environment ready. Available engines: {', '.join(available_engines)}. "
            f"VLM: {'enabled' if is_vlm_enabled() else 'disabled (rule-based extraction active)'}. "
            "You can now use pdf_to_markdown, extract_tables, classify_document, pdf_info."
        )
        if model_status.get("available") and not models_ready:
            results["message"] += (
                " NOTE: marker models are not fully cached — "
                "call download_models before the first conversion."
            )
    else:
        results["message"] = (
            "Environment NOT ready. Please install missing dependencies:\n"
            + "\n".join(f"  - {m}" for m in results["missing"])
        )

    if results["warnings"]:
        results["message"] += "\n\nWarnings:\n" + "\n".join(f"  - {w}" for w in results["warnings"])

    return _json(results)


# ── Tool 0.7: install_engine ────────────────────────────────────────────────


@mcp.tool()
def install_engine(engine: str = "marker") -> str:
    """Install an extraction engine into the current Python environment.

    Attempts to install the specified engine using pip/uv. This is a convenience
    tool so the AI agent can help users set up engines without leaving the chat.

    Supported engines:
    - marker: High-quality PDF extraction (installs marker-pdf + PyTorch, ~2.5GB).
    - ml: TATR deep-learning table detection (installs torch + transformers).
    - all: Everything (marker + ml).

    Note: Installation may take several minutes for large packages.
    If installation fails due to filesystem issues, suggest: export UV_LINK_MODE=copy

    Args:
        engine: Engine to install ('marker', 'ml', 'all').

    Returns:
        JSON with installation result and next steps.
    """
    import subprocess
    import sys

    valid_extras = {"marker", "ml", "all"}
    if engine not in valid_extras:
        return _json(
            {
                "ok": False,
                "error": (
                    f"Unknown engine: {engine!r}. Valid options: {', '.join(sorted(valid_extras))}"
                ),
            }
        )

    package = f"pdf-capture-mcp[{engine}]"

    # Try uv pip first (faster), fallback to pip
    install_cmds = [
        [sys.executable, "-m", "pip", "install", package],
    ]

    # Check if uv is available
    import shutil

    uv_path = shutil.which("uv")
    if uv_path:
        install_cmds.insert(0, [uv_path, "pip", "install", "--python", sys.executable, package])

    for cmd in install_cmds:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min max
            )
            if result.returncode == 0:
                # Verify installation
                import importlib

                if engine in ("marker", "all"):
                    try:
                        importlib.import_module("marker")
                    except ImportError:
                        pass  # May need restart

                return _json(
                    {
                        "ok": True,
                        "message": (
                            f"Successfully installed {package}. "
                            "The engine will be available for subsequent extraction calls. "
                            "You may need to restart the MCP server for changes to take effect."
                        ),
                        "engine": engine,
                        "output": result.stdout[-500:] if result.stdout else "",
                    }
                )
            else:
                # Try next command
                continue
        except subprocess.TimeoutExpired:
            return _json(
                {
                    "ok": False,
                    "error": (
                        f"Installation timed out (>10min). The package {package} is very large. "
                        "Try installing manually: pip install "
                        f"{package}"
                    ),
                }
            )
        except Exception:
            continue

    return _json(
        {
            "ok": False,
            "error": (
                f"Failed to install {package}. Try manually:\n"
                f"  pip install {package}\n"
                "If on an external drive, first: export UV_LINK_MODE=copy"
            ),
        }
    )


# ── Tool 0.8: download_models ─────────────────────────────────────────


@mcp.tool()
def download_models(engine: str = "marker") -> str:
    """Pre-download all models required by an extraction engine.

    Strongly recommended before the first pdf_to_markdown call on slow or
    restricted networks. The marker engine normally downloads models lazily
    inside a 300s inference-server startup window — on slow networks that
    window expires and the conversion fails with an opaque timeout. This tool
    downloads the same models WITHOUT any time limit.

    Runs as a background job (downloads total ~2GB). Poll progress with
    get_job_status(job_id).

    Network tips (e.g. mainland China where huggingface.co is unreachable):
    - Set HF_ENDPOINT=https://hf-mirror.com to use a mirror. Xet transfer is
      auto-disabled for mirrors (HF_HUB_DISABLE_XET=1).
    - If you use an HTTP proxy, keep 'localhost,127.0.0.1' in NO_PROXY
      (the server enforces this automatically at startup).
    - Once all models are cached, set HF_HUB_OFFLINE=1 for fully offline runs.

    Args:
        engine: Engine whose models to download. Currently only 'marker'.

    Returns:
        JSON with job_id for polling, or already_cached=true if nothing to do.
    """
    if engine != "marker":
        return _json(
            {
                "ok": False,
                "error": f"Unsupported engine: {engine!r}. Only 'marker' has "
                "pre-downloadable models (mineru manages its own; pymupdf needs none).",
            }
        )

    from pdf_capture_mcp.models import check_model_cache, download_marker_models

    status = check_model_cache()
    if not status.get("available"):
        return _json(
            {
                "ok": False,
                "error": "marker is not installed — run install_engine first.",
            }
        )
    if status.get("all_cached"):
        return _json(
            {
                "ok": True,
                "already_cached": True,
                "models": status["models"],
                "message": "All marker models are already cached — nothing to download.",
            }
        )

    from pdf_capture_mcp.jobs import create_job, update_stage

    def _target(job: dict[str, Any]) -> dict[str, Any]:
        return download_marker_models(progress=lambda stage: update_stage(job, stage))

    job = create_job("download_models", _target, params={"engine": engine})
    return _json(
        {
            "ok": True,
            "async": True,
            "job_id": job["job_id"],
            "status": job["status"],
            "models": status["models"],
            "hint": (
                f"Model download started (~2GB total). "
                f"Poll with get_job_status(job_id='{job['job_id']}')."
            ),
        }
    )


# ── Tool 1: pdf_to_markdown ─────────────────────────────────────────────────


# In mode='auto', PDFs with more pages than this run as a background job.
ASYNC_PAGE_THRESHOLD = 15
# Rough marker throughput used only for user-facing ETA hints (min/page).
_EST_MINUTES_PER_PAGE = 0.4
# Scanned documents run full-page OCR — far slower (v0.10.0).
_EST_MINUTES_PER_PAGE_SCANNED = 1.5


def _derive_title(pdf: Path) -> str:
    """Human-readable document title: PDF metadata -> filename stem fallback.

    Download artifacts like '2026_Report_050526-4_69f9b41ca0945' make poor
    titles; the metadata Title field wins when it looks like real prose
    (has words, sane length, not an authoring-tool source filename).
    """
    try:
        import fitz

        with fitz.open(str(pdf)) as doc:
            meta = ((doc.metadata or {}).get("title") or "").strip()
    except Exception:  # noqa: BLE001 — title derivation is best-effort
        meta = ""
    looks_like_source_file = meta.lower().endswith(
        (".doc", ".docx", ".pdf", ".indd", ".qxd", ".pptx", ".tex")
    )
    if 4 <= len(meta) <= 200 and not looks_like_source_file:
        if sum(c.isalpha() for c in meta) >= 4:
            return meta
    return pdf.stem


def _parse_page_range(page_range: str) -> list[int] | None:
    """Parse a page range string like '0-9' or '0,2,5-7' into 0-based indices."""
    page_range = page_range.strip()
    if not page_range:
        return None
    pages: list[int] = []
    for part in page_range.split(","):
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            pages.extend(range(int(start_s), int(end_s) + 1))
        elif part:
            pages.append(int(part))
    return sorted(set(pages))


def _resolve_vlm_feature(value: str | bool, *, vlm_on: bool, policy_allows: bool) -> bool:
    """Resolve a tri-state VLM feature flag ('auto' | bool-like | explicit bool).

    'auto' activates the feature when a VLM is configured AND the stored
    policy allows it — configuring a VLM is the user's opt-in. Explicit
    booleans (or 'on'/'off' strings) always win.
    """
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("on", "true", "1", "yes"):
        return True
    if v in ("off", "false", "0", "no"):
        return False
    return vlm_on and policy_allows  # 'auto' and anything unrecognized


def _quick_page_count(pdf: Path) -> int:
    """Fast page count via pymupdf (returns 0 if unavailable)."""
    try:
        import fitz

        with fitz.open(str(pdf)) as doc:
            return int(doc.page_count)
    except Exception:  # noqa: BLE001 — page count is best-effort
        return 0


# ── Segmented extraction for oversized documents (v0.9.3) ─────────────────
# Pilot forensics: a 245-page PDF overloaded the resident inference service
# (13 consecutive timeouts, no self-recovery) and the backlog then poisoned
# every subsequent document. Splitting extraction into windows keeps each
# inference batch inside the service's safe zone.

SEGMENT_PAGE_THRESHOLD = 100
SEGMENT_WINDOW = 80
# 20 min per 80-page window — generous yet bounded. Env-tunable for slow
# hosts; scanned documents additionally get a 3x multiplier in the pipeline.
SEGMENT_TIMEOUT_S = int(os.environ.get("PDF_CAPTURE_SEGMENT_TIMEOUT_S", "1200"))
# Model-load budget awaited before the OCR clock starts (ready sentinel).
SEGMENT_READY_TIMEOUT_S = 300


def _watch_parent(parent_pid: int, interval_s: float = 5.0) -> None:
    """Daemon thread: hard-exit when the parent process dies.

    902-page field lesson: killing the coordinator does NOT cascade to
    spawned children (daemon=False because marker needs its own workers),
    and an orphaned segment child kept writing into seg_* dirs for hours,
    racing the successor run. Reparenting (getppid changes) is the death
    signal.
    """
    import threading
    import time as time_mod

    def _loop() -> None:
        while os.getppid() == parent_pid:
            time_mod.sleep(interval_s)
        os._exit(1)

    threading.Thread(target=_loop, daemon=True, name="parent-watchdog").start()


def _ensure_healthy_stdin() -> None:
    """Re-anchor fd 0 to /dev/null when the inherited stdin has died.

    902-page field lesson: nohup without `< /dev/null` leaves stdin on the
    launching pty; once that pty is reclaimed, every spawned child dies in
    the interpreter's init_sys_streams ("Bad file descriptor") before any
    of our code runs. The parent must guarantee a healthy fd 0 BEFORE
    spawning — the child cannot save itself.
    """
    try:
        os.fstat(0)
    except OSError:
        devnull_fd = os.open(os.devnull, os.O_RDONLY)
        try:
            os.dup2(devnull_fd, 0)
        finally:
            if devnull_fd != 0:
                os.close(devnull_fd)


def _sync_inference_timeout(timeout_s: float) -> None:
    """Align surya's OpenAI-client timeout with the segment budget.

    902-page field lesson: surya's default 600 s client timeout mass-
    expired every queued request of an 80-page OCR window long before the
    segment budget did (13 consecutive APITimeoutErrors, zero output).
    Must run before surya is imported — its settings snapshot the
    environment at import time — so segment children call it first thing.
    Explicit user settings always win (setdefault).
    """
    if timeout_s > 0:
        os.environ.setdefault("SURYA_INFERENCE_TIMEOUT_SECONDS", str(int(max(timeout_s, 600))))


def _segment_child(
    engine_name: str,
    pdf_str: str,
    seg_dir_str: str,
    kwargs: dict[str, Any],
    queue: Any,
    parent_pid: int = 0,
    timeout_s: float = 0,
) -> None:
    """Subprocess entry for one segment (module-level for spawn).

    Emits a 'ready' sentinel once the engine models are warm (v0.10.0):
    every spawned segment reloads marker models from scratch (1-3 min),
    and on scanned documents that load must not eat the OCR budget.
    v0.11.1: parent watchdog (orphan protection) + surya client timeout
    aligned with the segment budget before anything imports surya.
    """
    try:
        if parent_pid:
            _watch_parent(parent_pid)
        _sync_inference_timeout(timeout_s)
        from pdf_capture_mcp.engines import get_engine

        eng = get_engine(engine_name)
        warmup = getattr(eng, "warmup", None)
        if callable(warmup):
            warmup()
        queue.put("ready")
        queue.put(eng.extract(Path(pdf_str), Path(seg_dir_str), **kwargs))
    except Exception as exc:  # noqa: BLE001 — child must always report
        from pdf_capture_mcp.types import ExtractReport

        queue.put(ExtractReport(ok=False, engine=engine_name, error=str(exc)[:300]))


def _run_segment_isolated(
    eng: Any,
    pdf: Path,
    seg_dir: Path,
    kwargs: dict[str, Any],
    timeout_s: float,
    ready_timeout_s: float = SEGMENT_READY_TIMEOUT_S,
) -> Any:
    """Run one segment extraction in a subprocess with a hard timeout.

    Two-phase wait (v0.10.0): a fixed model-load budget for the child's
    'ready' sentinel first, then `timeout_s` for the extraction itself —
    spawned model loading never eats the segment's OCR budget.

    Returns the ExtractReport, or None on timeout (245-page regression:
    a single ultra-dense page can wedge the OCR service mid-segment —
    marker then retries internally forever and never returns, so failure
    can only be enforced from outside). daemon=False: marker spawns its
    own workers and daemonic processes may not have children.
    """
    import multiprocessing as mp
    import queue as queue_mod

    _ensure_healthy_stdin()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(
        target=_segment_child,
        args=(eng.name, str(pdf), str(seg_dir), kwargs, q, os.getpid(), timeout_s),
        daemon=False,
    )
    proc.start()
    try:
        first = q.get(timeout=ready_timeout_s)
        # A child that fails before models are warm sends its report
        # directly instead of the sentinel.
        report = q.get(timeout=timeout_s) if first == "ready" else first
        proc.join(30)
        return report
    except queue_mod.Empty:
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
        return None
    finally:
        q.close()


def _window_text_chars_per_page(pdf: Path, start: int, end: int) -> float:
    """Average extractable characters per page inside a page window.

    Best-effort: on any read failure assume a healthy text layer so the
    caller keeps the legacy pymupdf-degradation path.
    """
    try:
        import fitz

        with fitz.open(str(pdf)) as doc:
            n, chars = 0, 0
            for i in range(start, min(end + 1, doc.page_count)):
                chars += len(doc[i].get_text().strip())
                n += 1
            return chars / max(n, 1)
    except Exception:  # noqa: BLE001 — probe is best-effort
        return float("inf")


def _extract_segmented(
    eng: Any,
    pdf: Path,
    extraction_dir: Path,
    total_pages: int,
    *,
    enable_formula: bool,
    stage_cb: Any,
    segment_timeout_s: float = SEGMENT_TIMEOUT_S,
    force_ocr: bool = False,
    _runner: Any = None,
    _fallback: Any = None,
) -> Any:
    """Extract an oversized PDF in page windows and merge the segments.

    Per segment: subprocess with a hard timeout (v0.9.4) -> on timeout or
    failure, restart inference services and retry once -> still failing:

    * window HAS a text layer -> DEGRADE HONESTLY to the pymupdf engine
      (content preserved, layout fidelity reduced) and record it in
      metadata['degraded_segments']. A near-empty fallback product
      (<50 chars/page) is never accepted as a degradation — it becomes a
      missing window instead (v0.10.0: empty text must not impersonate
      preserved content).
    * window has NO text layer (scanned document, v0.10.0) -> pymupdf
      cannot rescue image-only pages, so the window is split in half and
      each half retried once; halves that still fail are recorded in
      metadata['missing_segments'] with an explicit placeholder block in
      the markdown. Loss is always visible, never silent.

    Checkpoints (v0.10.0): each successfully merged segment persists its
    rewritten markdown plus a .seg_meta.json (kwargs hash); re-running the
    same out_dir after an interruption reuses finished segments instead of
    re-OCRing them. Segment dirs are cleaned up only after the full merge
    succeeds.

    _runner/_fallback are injectable for tests.
    """
    import hashlib
    import json as json_mod
    import math
    import shutil

    from pdf_capture_mcp.engines.marker_engine import restart_inference_services
    from pdf_capture_mcp.types import ExtractReport

    runner = _runner or _run_segment_isolated
    images_dir = extraction_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    n_seg = math.ceil(total_pages / SEGMENT_WINDOW)
    md_parts: list[str] = []
    degraded: list[int] = []
    missing: list[dict[str, Any]] = []
    finished_dirs: list[Path] = []
    image_count, elapsed = 0, 0.0

    def _kwargs_hash(kwargs: dict[str, Any]) -> str:
        return hashlib.sha1(
            json_mod.dumps(kwargs, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    def _reuse_checkpoint(seg_dir: Path, khash: str) -> dict[str, Any] | None:
        meta_p = seg_dir / ".seg_meta.json"
        md_p = seg_dir / "full_text.md"
        if not (meta_p.exists() and md_p.exists()):
            return None
        try:
            meta = json_mod.loads(meta_p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — corrupt checkpoint: rebuild
            return None
        if meta.get("kwargs_hash") != khash:
            return None
        return dict(meta)

    def _merge_segment(prefix: str, seg_dir: Path, report: Any, khash: str) -> None:
        """Move images under a unique prefix, rewrite refs, checkpoint."""
        nonlocal image_count, elapsed
        text = Path(report.full_text_md).read_text(encoding="utf-8", errors="replace")
        seg_images = seg_dir / "images"
        if seg_images.is_dir():
            for img in sorted(seg_images.iterdir()):
                if img.name.startswith("._"):
                    # macOS AppleDouble metadata on external volumes —
                    # renaming would launder the '._' marker away and the
                    # junk becomes indistinguishable from real images
                    # (902-page audit: 714 laundered ghosts, one per figure).
                    continue
                new_name = f"{prefix}_{img.name}"
                shutil.move(str(img), str(images_dir / new_name))
                text = text.replace(f"images/{img.name}", f"images/{new_name}")
        # Persist the rewritten text so the checkpoint is self-consistent
        # (its images already live in the shared images/ dir).
        (seg_dir / "full_text.md").write_text(text, encoding="utf-8")
        (seg_dir / ".seg_meta.json").write_text(
            json_mod.dumps(
                {
                    "kwargs_hash": khash,
                    "engine": report.engine,
                    "image_count": report.image_count,
                    "elapsed_seconds": report.elapsed_seconds,
                }
            ),
            encoding="utf-8",
        )
        md_parts.append(text.strip())
        image_count += report.image_count
        elapsed += report.elapsed_seconds
        finished_dirs.append(seg_dir)

    def _attempt(seg_dir: Path, kwargs: dict[str, Any]) -> Any:
        """One runner attempt + one hygiene-restart retry."""
        report = runner(eng, pdf, seg_dir, kwargs, segment_timeout_s)
        if report is None or not report.ok:
            restart_inference_services()
            shutil.rmtree(seg_dir, ignore_errors=True)
            report = runner(eng, pdf, seg_dir, kwargs, segment_timeout_s)
        return report

    def _window_kwargs(start: int, end: int) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "enable_formula": enable_formula,
            "enable_table": True,
            "pages": list(range(start, end + 1)),
            "page_range": f"{start}-{end}",
        }
        if force_ocr:
            kw["force_ocr"] = True
        return kw

    for k, seg_start in enumerate(range(0, total_pages, SEGMENT_WINDOW)):
        seg_end = min(seg_start + SEGMENT_WINDOW, total_pages) - 1
        seg_dir = extraction_dir / f"seg_{k}"
        seg_kwargs = _window_kwargs(seg_start, seg_end)
        khash = _kwargs_hash(seg_kwargs)

        ckpt = _reuse_checkpoint(seg_dir, khash)
        if ckpt is not None:
            stage_cb(f"segment {k + 1}/{n_seg}: reusing checkpoint")
            md_parts.append(
                (seg_dir / "full_text.md").read_text(encoding="utf-8", errors="replace").strip()
            )
            image_count += int(ckpt.get("image_count", 0))
            elapsed += float(ckpt.get("elapsed_seconds", 0.0))
            finished_dirs.append(seg_dir)
            continue
        shutil.rmtree(seg_dir, ignore_errors=True)  # stale/mismatched checkpoint

        stage_cb(f"extracting segment {k + 1}/{n_seg} (pages {seg_start + 1}-{seg_end + 1})")
        report = _attempt(seg_dir, seg_kwargs)
        if report is not None and report.ok:
            _merge_segment(f"s{k}", seg_dir, report, khash)
            continue

        # Both attempts failed. Choose the honest path for this window.
        restart_inference_services()
        shutil.rmtree(seg_dir, ignore_errors=True)
        window_pages = seg_end - seg_start + 1
        has_text_layer = _window_text_chars_per_page(pdf, seg_start, seg_end) >= 200

        if has_text_layer:
            # Text-layer degradation (v0.9.4 path).
            stage_cb(f"segment {k + 1}/{n_seg}: degrading to text-layer engine")
            if _fallback is not None:
                fallback_eng = _fallback
            else:
                from pdf_capture_mcp.engines import get_engine

                fallback_eng = get_engine("pymupdf")
            report = fallback_eng.extract(
                pdf,
                seg_dir,
                enable_formula=False,
                enable_table=True,
                pages=list(range(seg_start, seg_end + 1)),
            )
            if not report.ok:
                report.error = (
                    f"Segment {k + 1}/{n_seg} (pages {seg_start + 1}-{seg_end + 1}) "
                    f"failed even in degraded mode: {report.error}"
                )
                return report
            # Cheap guard (v0.10.0): a near-empty product is loss, not a
            # degradation — never let it impersonate preserved content.
            fb_text = Path(report.full_text_md).read_text(encoding="utf-8", errors="replace")
            if len(fb_text.strip()) / max(window_pages, 1) < 50:
                shutil.rmtree(seg_dir, ignore_errors=True)
                missing.append({"segment": k + 1, "pages": [seg_start + 1, seg_end + 1]})
                md_parts.append(
                    f"> [WARNING] Pages {seg_start + 1}-{seg_end + 1}: extraction "
                    "failed and the fallback produced no content; "
                    "content NOT captured."
                )
                continue
            _merge_segment(f"s{k}", seg_dir, report, khash)
            degraded.append(k + 1)
            continue

        # Scanned window (v0.10.0): pymupdf cannot rescue image-only pages.
        # Halve the window and retry each half once; report what still fails.
        stage_cb(f"segment {k + 1}/{n_seg}: no text layer — half-window OCR retry")
        mid = (seg_start + seg_end) // 2
        halves = [(seg_start, mid)]
        if mid + 1 <= seg_end:
            halves.append((mid + 1, seg_end))
        for j, (h_start, h_end) in enumerate(halves):
            h_dir = extraction_dir / f"seg_{k}_h{j}"
            shutil.rmtree(h_dir, ignore_errors=True)
            h_kwargs = _window_kwargs(h_start, h_end)
            h_report = runner(eng, pdf, h_dir, h_kwargs, segment_timeout_s)
            if h_report is not None and h_report.ok:
                _merge_segment(f"s{k}h{j}", h_dir, h_report, _kwargs_hash(h_kwargs))
            else:
                restart_inference_services()
                shutil.rmtree(h_dir, ignore_errors=True)
                missing.append({"segment": k + 1, "pages": [h_start + 1, h_end + 1]})
                md_parts.append(
                    f"> [WARNING] Pages {h_start + 1}-{h_end + 1}: OCR failed and "
                    "no text layer exists; content NOT captured."
                )

    md_path = extraction_dir / "full_text.md"
    md_path.write_text("\n\n".join(md_parts) + "\n", encoding="utf-8")
    # Merge complete: checkpoints have served their purpose.
    for d in finished_dirs:
        shutil.rmtree(d, ignore_errors=True)
    logger.info(
        "Segmented extraction merged: %d segments, %d pages, %d degraded, %d missing windows",
        n_seg,
        total_pages,
        len(degraded),
        len(missing),
    )
    return ExtractReport(
        ok=True,
        engine=eng.name,
        full_text_md=str(md_path),
        content_dir=str(extraction_dir),
        page_count=total_pages,
        image_count=image_count,
        elapsed_seconds=round(elapsed, 2),
        metadata={
            "segments": n_seg,
            "segment_window": SEGMENT_WINDOW,
            "degraded_segments": degraded,
            "missing_segments": missing,
        },
    )


def _run_pipeline(
    pdf: Path,
    *,
    engine: str,
    enable_formula: bool,
    enable_table_enrich: str | bool = "auto",
    skip_qc: bool = False,
    out_dir: str = "",
    auto_repair: bool = True,
    enrich_figures: str | bool = "auto",
    package: bool = True,
    pages: list[int] | None = None,
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full conversion pipeline and return the response dict.

    Shared by the synchronous path and background jobs. When ``job`` is
    provided, stage transitions are persisted for get_job_status polling.
    """
    from pdf_capture_mcp.classifier import classify_document
    from pdf_capture_mcp.engines import get_engine
    from pdf_capture_mcp.extractors.tables import extract_tables
    from pdf_capture_mcp.jobs import update_stage
    from pdf_capture_mcp.llm_client import get_vlm_policy, is_vlm_enabled

    def _stage(name: str) -> None:
        if job is not None:
            update_stage(job, name)

    # ── Feature resolution: 'auto' follows the configured VLM policy ──────
    # Configuring a VLM (setup_vlm) is itself the user's opt-in to spend
    # tokens on quality; 'auto' honors that intent, explicit bools override.
    vlm_on = is_vlm_enabled()
    policy = get_vlm_policy() if vlm_on else ""
    table_enrich_on = _resolve_vlm_feature(
        enable_table_enrich, vlm_on=vlm_on, policy_allows=policy in ("full", "tables_only")
    )
    figures_on = _resolve_vlm_feature(enrich_figures, vlm_on=vlm_on, policy_allows=policy == "full")
    features: dict[str, Any] = {
        "deterministic_fixes": {"enabled": not skip_qc, "cost": "free, always on with QC"},
        "geometric_repair": {"enabled": auto_repair and not skip_qc, "cost": "free"},
        "vlm_table_repair": {
            "enabled": table_enrich_on,
            "reason": (
                "active (VLM configured, policy allows)"
                if table_enrich_on
                else "VLM not configured — run setup_vlm to unlock"
                if not vlm_on
                else "disabled by parameter"
            ),
        },
        "vlm_figure_descriptions": {
            "enabled": figures_on,
            "reason": (
                "active (VLM configured, policy=full)"
                if figures_on
                else "VLM not configured — run setup_vlm to unlock"
                if not vlm_on
                else f"off (policy={policy or 'n/a'}); pass enrich_figures=True to force"
            ),
        },
    }

    # Pre-flight notice only when the caller EXPLICITLY asked and VLM is off
    vlm_notice = ""
    if enable_table_enrich is True and not vlm_on:
        vlm_notice = (
            "VLM table enrichment requested but VLM is not configured. "
            "Call 'setup_vlm' with action='enable' to configure a vision-capable model. "
            "Falling back to rule-based extraction (still good quality)."
        )

    # Phase 0: Classify
    _stage("classify")
    classification = classify_document(pdf)

    # Phase 1: Extract
    extraction_dir = Path(out_dir) / "extraction"
    extraction_dir.mkdir(parents=True, exist_ok=True)

    try:
        eng = get_engine(engine)
    except RuntimeError as engine_err:
        return {
            "ok": False,
            "error": str(engine_err),
            "stage": "engine_select",
            "quick_fix": "pip install pdf-capture-mcp[marker]",
            "alternative": (
                "The built-in pymupdf engine should be available as a fallback. "
                "If this error persists, run: pip install pymupdf4llm"
            ),
        }

    _stage("extracting")
    total_pages = _quick_page_count(pdf)
    # Scanned documents (v0.10.0): force OCR on every page and triple the
    # per-segment budget — full-page OCR runs far slower than text-layer
    # assisted extraction.
    force_ocr = bool(classification.is_scanned)
    if pages is None and total_pages > SEGMENT_PAGE_THRESHOLD:
        # Oversized document: windowed extraction (v0.9.3)
        extract_report = _extract_segmented(
            eng,
            pdf,
            extraction_dir,
            total_pages,
            enable_formula=enable_formula,
            stage_cb=_stage,
            segment_timeout_s=SEGMENT_TIMEOUT_S * (3 if force_ocr else 1),
            force_ocr=force_ocr,
        )
    else:
        extract_kwargs: dict[str, Any] = {}
        if pages is not None:
            # marker expects a range string; pymupdf expects a list of indices.
            extract_kwargs["pages"] = pages
            extract_kwargs["page_range"] = ",".join(str(p) for p in pages)
        if force_ocr:
            extract_kwargs["force_ocr"] = True
        extract_report = eng.extract(
            pdf,
            extraction_dir,
            enable_formula=enable_formula,
            enable_table=True,
            **extract_kwargs,
        )

    if not extract_report.ok:
        return {
            "ok": False,
            "error": extract_report.error,
            "stage": "extract",
            "engine": eng.name,
        }

    # Read extracted markdown
    markdown_text = ""
    md_path = Path(extract_report.full_text_md)
    if md_path.exists():
        markdown_text = md_path.read_text(encoding="utf-8", errors="replace")

    # Phase 2: Table extraction (supplementary)
    _stage("table_extraction")
    table_result = extract_tables(pdf, max_tables=30)
    table_count = table_result.get("stats", {}).get("total_tables", 0)

    # Phase 3: Content-aware audit (auto-fix safe defects, detect the rest)
    # + Phase 3.5: cross-channel repair (repair-or-report)
    # + Phase 4: multi-dimensional QC gate
    _stage("qc")
    # Hoisted out of the QC block (v0.9.5): with skip_qc=True the degraded
    # info was silently lost from the package metadata.
    degraded_segments = extract_report.metadata.get("degraded_segments") or []
    missing_segments = extract_report.metadata.get("missing_segments") or []
    qc_verdict = "PASS"
    qc_report: dict[str, Any] = {}
    if not skip_qc and markdown_text:
        from pdf_capture_mcp.quality.md_audit import run_markdown_audit
        from pdf_capture_mcp.quality.qc_gate import run_qc_gate

        audit = run_markdown_audit(
            markdown_text, pdf_path=pdf, autofix=True, base_dir=extraction_dir
        )
        markdown_text = audit["text"]
        audit_issues = audit["issues"]
        audit_counts = audit["counts"]

        # Phase 3.5: structural defects get a cross-channel repair attempt
        # against the PDF text layer. Verified repairs are applied; failed
        # attempts stay as reported issues with recovered candidates.
        repair_actions: list[dict[str, Any]] = []
        repairable = [i for i in audit_issues if i.rule in ("MD-104", "MD-105", "MD-201")]
        if auto_repair and repairable:
            from pdf_capture_mcp.quality.repair import repair_markdown

            _stage("repair")
            rep = repair_markdown(markdown_text, pdf, repairable)
            repair_actions = [vars(a) for a in rep["actions"]]
            if rep["modified"]:
                markdown_text = rep["text"]
                # Re-audit the repaired text: verified repairs clear their
                # issues; anything left is genuinely outstanding.
                post = run_markdown_audit(
                    markdown_text, pdf_path=pdf, autofix=False, base_dir=extraction_dir
                )
                audit_issues = post["issues"]
                audit_counts = post["counts"]

        # Phase 3.6: VLM arbitration — table defects beyond geometric reach
        # (merged cells, group headers) plus optional figure descriptions for
        # text-only RAG. Opt-in: consumes API tokens.
        from pdf_capture_mcp.llm_client import is_vlm_enabled

        table_defects = [i for i in audit_issues if i.rule in ("MD-104", "MD-105", "MD-107")]
        want_vlm = (table_enrich_on and table_defects) or figures_on
        if want_vlm and is_vlm_enabled():
            from pdf_capture_mcp.quality.vlm_repair import run_vlm_arbitration

            _stage("vlm_arbitration")
            vlm = run_vlm_arbitration(
                markdown_text,
                pdf,
                table_defects,
                base_dir=extraction_dir,
                repair_tables=table_enrich_on,
                describe_figures=figures_on,
            )
            repair_actions += [vars(a) for a in vlm["actions"]]
            if vlm["modified"]:
                markdown_text = vlm["text"]
                post = run_markdown_audit(
                    markdown_text, pdf_path=pdf, autofix=False, base_dir=extraction_dir
                )
                audit_issues = post["issues"]
                audit_counts = post["counts"]
        elif table_enrich_on and table_defects and not is_vlm_enabled():
            vlm_notice = (
                "Table defects detected and enable_table_enrich requested, but VLM "
                "is not configured — call setup_vlm to enable vision-based repair."
            )

        if markdown_text != audit["text"] or audit["modified"]:
            # Persist sanitized/repaired markdown
            md_path.write_text(markdown_text, encoding="utf-8")

        gate = run_qc_gate(
            markdown_text,
            page_count=extract_report.page_count,
            expected_tables=table_count,
        )
        qc_verdict = gate.verdict
        # Content-aware critical findings escalate a PASS to WARN: the text
        # may look statistically fine while specific tables/values are broken.
        if qc_verdict == "PASS" and audit_counts["critical"] > 0:
            qc_verdict = "WARN"

        # Degraded segments (v0.9.4): part of the document was extracted by
        # the text-layer fallback — content preserved, layout fidelity
        # reduced. Missing windows (v0.10.0): OCR failed on image-only
        # pages — content LOST. Never let either pass silently.
        if (degraded_segments or missing_segments) and qc_verdict == "PASS":
            qc_verdict = "WARN"

        qc_report = {
            "verdict": qc_verdict,
            "dimensions": gate.dimensions,
            "audit_counts": audit_counts,
            "degraded_segments": degraded_segments,
            "missing_segments": missing_segments,
            # Plain dicts so both MCP JSON responses and persisted job state
            # serialize cleanly.
            "audit_fixes": [vars(f) for f in audit["fixes"]],
            "audit_issues": [vars(i) for i in audit_issues],
            "repairs": repair_actions,
        }
        if classification.is_scanned:
            # Blindness made explicit (v0.10.0): these checks need a text
            # layer and silently no-op on scans — say so instead.
            qc_report["notes"] = [
                "text layer unavailable (scanned document): MD-201/MD-202 "
                "coverage checks and VLM table repair are not applicable"
            ]
    elif not markdown_text:
        qc_verdict = "HALT"

    # Phase 5: knowledge package assembly — self-describing folder that any
    # agent can read (README map) and that drops into an Obsidian vault as a
    # unit. MD-110 cross-page table merge runs first so chunks see the
    # merged tables; frontmatter is injected LAST (N3: QC sees body only).
    package_info: dict[str, Any] = {}
    if package and markdown_text and qc_verdict != "HALT":
        _stage("package")
        try:
            from pdf_capture_mcp import __version__ as _ver
            from pdf_capture_mcp.chunking.chunker import chunk_markdown
            from pdf_capture_mcp.packaging import (
                assemble_package,
                build_frontmatter,
                build_heading_tree,
                compute_doc_id,
                extract_summary,
            )
            from pdf_capture_mcp.quality.cross_page_tables import (
                detect_cross_page_tables,
                merge_cross_page_tables,
            )

            # MD-110: merge split cross-page tables (geometric gate inside)
            md110 = detect_cross_page_tables(markdown_text, pdf_path=pdf)
            if md110:
                md_lines = markdown_text.splitlines()
                merge_actions = []
                # Merge bottom-up so earlier line numbers stay valid
                for issue in sorted(md110, key=lambda i: -i.lines[0]):
                    merge_actions.append(merge_cross_page_tables(md_lines, issue))
                merged_text = "\n".join(md_lines) + ("\n" if markdown_text.endswith("\n") else "")
                if merged_text != markdown_text:
                    markdown_text = merged_text
                    md_path.write_text(markdown_text, encoding="utf-8")
                if qc_report:
                    qc_report.setdefault("repairs", []).extend(vars(a) for a in merge_actions)

            doc_id = compute_doc_id(pdf)
            title = _derive_title(pdf)
            chunk_result = chunk_markdown(markdown_text, doc_id, pdf_path=pdf)
            summary, summary_source = extract_summary(markdown_text)
            frontmatter = build_frontmatter(
                title=title,
                doc_id=doc_id,
                source_pdf=pdf.name,
                pages=extract_report.page_count,
                qc_verdict=qc_verdict,
                tool_version=_ver,
            )
            conversion_params = {
                "engine": eng.name,
                "enable_formula": enable_formula,
                "table_enrich": table_enrich_on,
                "enrich_figures": figures_on,
                "auto_repair": auto_repair,
            }
            package_info = assemble_package(
                output_root=Path(out_dir),
                title=title,
                doc_id=doc_id,
                markdown_text=markdown_text,
                frontmatter=frontmatter,
                images_dir=extraction_dir / "images",
                tables=table_result.get("tables", []),
                chunks=chunk_result["chunks"],
                qc_report=qc_report,
                readme_kwargs={
                    "title": title,
                    "doc_id": doc_id,
                    "source_pdf": pdf.name,
                    "pages": extract_report.page_count,
                    "qc_verdict": qc_verdict,
                    "tool_version": _ver,
                    "summary": summary,
                    "summary_source": summary_source,
                },
                metadata_kwargs={
                    "doc_id": doc_id,
                    "title": title,
                    "source_pdf": pdf.name,
                    "pages": extract_report.page_count,
                    "tool_version": _ver,
                    "conversion_params": conversion_params,
                    "summary": summary,
                    "summary_source": summary_source,
                    "heading_tree": build_heading_tree(markdown_text),
                    "dropped_headers": chunk_result["dropped_headers"],
                    "degraded_segments": degraded_segments,
                    "missing_segments": missing_segments,
                },
            )
        except Exception as exc:  # noqa: BLE001 — packaging must not kill conversion
            logger.warning("Package assembly failed: %s", exc)
            package_info = {"error": f"Package assembly failed: {exc}"}

    return {
        "ok": qc_verdict != "HALT",
        "markdown_text": markdown_text,
        "markdown_path": package_info.get("markdown_path")
        or (str(md_path) if md_path.exists() else ""),
        "package": package_info,
        "pipeline_root": str(out_dir),
        "title": _derive_title(pdf),
        "engine": eng.name,
        "classification": {
            "doc_type": classification.doc_type,
            "confidence": classification.confidence,
            "source": classification.source,
            "has_formulas": classification.has_formulas,
            "has_tables": classification.has_tables,
            "is_scanned": classification.is_scanned,
            "text_layer_coverage": classification.text_layer_coverage,
        },
        "page_count": extract_report.page_count,
        "table_count": table_count,
        "image_count": extract_report.image_count,
        "qc_verdict": qc_verdict,
        "qc_report": qc_report,
        "elapsed_seconds": extract_report.elapsed_seconds,
        "vlm_notice": vlm_notice,
        "features": features,
        "missing_segments": missing_segments,
        **(
            {
                "warning": "Content NOT captured for pages: "
                + ", ".join(f"{m['pages'][0]}-{m['pages'][1]}" for m in missing_segments)
                + " (OCR failed, no text layer to fall back on)."
            }
            if missing_segments
            else {}
        ),
    }


@mcp.tool()
def pdf_to_markdown(
    pdf_path: str,
    engine: str = "auto",
    enable_formula: bool = True,
    enable_table_enrich: str = "auto",
    enable_tatr: bool = False,
    skip_qc: bool = False,
    out_dir: str = "",
    mode: str = "auto",
    page_range: str = "",
    auto_repair: bool = True,
    enrich_figures: str = "auto",
    package: bool = True,
) -> str:
    """Convert a PDF into a self-describing Markdown knowledge package.

    Pipeline: Engine extraction (marker/MinerU) -> table extraction (pdfplumber)
    -> QC quality gate -> repair -> knowledge package.

    Output (package=True, default) is a self-describing folder that drops
    into an Obsidian vault as a unit and that any LLM agent can understand
    from its README.md alone:

        <out_dir>/<slug>/
            <slug>.md        main document (frontmatter included)
            README.md        agent + human entry map
            images/, tables/ assets (relative refs)
            data/            chunks.jsonl + metadata.json + qc_report.json

    When out_dir is empty the package goes to $PDF_CAPTURE_OUTPUT_ROOT or
    ~/Documents/pdf-capture (stable location — not a temp dir).

    Large-PDF handling: conversion of big documents can exceed MCP client
    timeouts. In mode='auto' (default), PDFs above ~15 pages are converted in
    a background job — the call returns a job_id immediately; poll progress
    with get_job_status(job_id). The result markdown is always written to
    <out_dir>/extraction/full_text.md.

    VLM feature flags are tri-state: 'auto' (default) activates the feature
    when a VLM is configured (setup_vlm) and its stored policy allows it —
    configuring a VLM is the user's opt-in to spend tokens on quality.
    Pass 'on'/'off' to override per call. The response's `features` section
    reports exactly what ran and how to unlock what didn't.

    Args:
        pdf_path: Absolute path to the PDF file.
        engine: Extraction engine ('auto', 'marker', 'mineru').
        enable_formula: Enable formula/equation recognition.
        enable_table_enrich: 'auto' | 'on' | 'off'. Escalate broken tables
            (torn cells, fused/flattened group headers) to the configured
            VLM: the table region is re-read from a hi-res page render and
            replaced with an HTML <table> — only when the numeric-
            conservation gate passes (no invented numbers, no lost numbers).
            'auto' = on whenever a VLM is configured.
        enable_tatr: DEPRECATED no-op in this pipeline; use the
            extract_tables tool with strategy='tatr' instead.
        skip_qc: Skip quality gate (debug only).
        out_dir: Output directory (auto-created if empty).
        mode: 'auto' (async for large PDFs), 'sync' (always inline, may time
            out on large PDFs), or 'async' (always return a job_id).
        page_range: Optional pages to convert, e.g. '0-9' or '0,2,5-7'
            (0-based). Useful for a fast preview of a large document.
        auto_repair: Attempt cross-channel repair of structural defects
            (torn numeric cells, fused headers, dropped text) using the PDF
            text layer. Repairs are applied only when a verification gate
            passes; otherwise the defect is reported with candidates
            (repair-or-report). Set False to keep the raw engine output.
        enrich_figures: 'auto' | 'on' | 'off'. Inject a short VLM-generated
            description under each extracted figure image so figure-embedded
            content becomes retrievable by text-only RAG (closes the MD-202
            gap). 'auto' = on when a VLM is configured with policy='full'.
            Consumes API tokens per figure.
        package: Assemble the knowledge package (chunks.jsonl, README,
            metadata, frontmatter, MD-110 cross-page table merge). Set
            False for the bare markdown-only layout of earlier versions.

    Returns:
        JSON with markdown_text (sync) or job_id + polling hint (async).
        Always includes `features` describing which deep capabilities ran.
        Async note: `result_path` points at the RAW intermediate markdown;
        with package=True the final document is the job result's
        `markdown_path` inside the <slug>/ knowledge package.
    """
    try:
        if enable_tatr:
            logger.warning(
                "enable_tatr is a deprecated no-op in pdf_to_markdown; "
                "use extract_tables(strategy='tatr') instead."
            )
        pdf = _resolve_pdf(pdf_path)

        if mode not in ("auto", "sync", "async"):
            return _json({"ok": False, "error": f"Invalid mode: {mode!r}. Use: auto, sync, async"})

        try:
            pages = _parse_page_range(page_range)
        except ValueError:
            return _json(
                {"ok": False, "error": f"Invalid page_range: {page_range!r}. Example: '0-9'"}
            )

        if not out_dir.strip():
            if package:
                # Stable, discoverable default — knowledge packages are
                # assets, not scratch files (naming-standard audit).
                out_dir = os.environ.get(
                    "PDF_CAPTURE_OUTPUT_ROOT",
                    str(Path.home() / "Documents" / "pdf-capture"),
                )
            else:
                out_dir = tempfile.mkdtemp(prefix="pdf_capture_")

        effective_pages = len(pages) if pages is not None else _quick_page_count(pdf)
        run_async = mode == "async" or (mode == "auto" and effective_pages > ASYNC_PAGE_THRESHOLD)

        pipeline_kwargs: dict[str, Any] = {
            "engine": engine,
            "enable_formula": enable_formula,
            "enable_table_enrich": enable_table_enrich,
            "skip_qc": skip_qc,
            "out_dir": out_dir,
            "auto_repair": auto_repair,
            "enrich_figures": enrich_figures,
            "package": package,
            "pages": pages,
        }

        if not run_async:
            return _json(_run_pipeline(pdf, **pipeline_kwargs))

        from pdf_capture_mcp.jobs import create_job

        def _target(job: dict[str, Any]) -> dict[str, Any]:
            result = _run_pipeline(pdf, job=job, **pipeline_kwargs)
            if not result.get("ok"):
                raise RuntimeError(result.get("error") or "pipeline failed")
            # Drop the full markdown from persisted job state — clients read
            # it from markdown_path; keeps job JSON and MCP responses small.
            result.pop("markdown_text", None)
            return result

        job = create_job(
            "pdf_to_markdown",
            _target,
            params={"pdf_path": str(pdf), "engine": engine, "out_dir": out_dir},
        )
        # Scanned documents run full-page OCR: honest ETA (v0.10.0).
        from pdf_capture_mcp.classifier import detect_text_layer

        is_scanned, _cov = detect_text_layer(pdf)
        rate = _EST_MINUTES_PER_PAGE_SCANNED if is_scanned else _EST_MINUTES_PER_PAGE
        estimated = max(1, round(effective_pages * rate))
        scan_note = (
            f"Scanned document detected: full-OCR path, expect ~{estimated / 60:.1f} h. "
            if is_scanned
            else ""
        )
        return _json(
            {
                "ok": True,
                "async": True,
                "job_id": job["job_id"],
                "status": job["status"],
                "page_count": effective_pages,
                "estimated_minutes": estimated,
                "is_scanned": is_scanned,
                "result_path": str(Path(out_dir) / "extraction" / "full_text.md"),
                "hint": (
                    f"Conversion started in the background (~{estimated} min). "
                    f"{scan_note}"
                    f"Poll with get_job_status(job_id='{job['job_id']}'). "
                    "Note: result_path is the RAW intermediate markdown; with "
                    "package=True the final document is at the job result's "
                    "markdown_path (inside the <slug>/ knowledge package)."
                ),
            }
        )

    except Exception as exc:
        return _json(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-500:],
            }
        )


@mcp.tool()
def export_to_obsidian(
    package_dir: str, vault_path: str, category: str = "", overwrite: bool = False
) -> str:
    """Copy a knowledge package into an Obsidian vault as a whole unit.

    The package folder produced by pdf_to_markdown is vault-ready: main md
    carries frontmatter, image refs are relative, README.md doubles as the
    document card. This tool copies the ENTIRE folder (never flattens,
    never rewrites links) into the vault, optionally under a category
    subfolder. Idempotent: skips when the vault already holds an identical
    content_hash. If the vault copy DIVERGED (hand-edited or unreadable
    metadata), the export is refused with conflict=True unless
    overwrite=True — your vault edits are never silently clobbered (v0.9.5).

    Args:
        package_dir: Path to the knowledge package (contains data/metadata.json).
        vault_path: Absolute path to the Obsidian vault root.
        category: Optional subfolder inside the vault (e.g. 'Papers/LLM').

    Returns:
        JSON with ok, dest, skipped.
    """
    try:
        from pdf_capture_mcp.packaging import export_package_to_vault

        return _json(
            export_package_to_vault(package_dir, vault_path, category, overwrite=overwrite)
        )
    except Exception as exc:  # noqa: BLE001 — tool boundary
        logger.error("export_to_obsidian failed: %s", traceback.format_exc())
        return _json({"ok": False, "error": str(exc)})


@mcp.tool()
def setup_embedding(
    model: str = "",
    api_key: str = "",
    api_base: str = "https://api.openai.com/v1",
    provider: str = "openai",
    action: str = "status",
) -> str:
    """Configure the embedding endpoint that powers RAG indexing/search.

    Any OpenAI-compatible /embeddings endpoint works: OpenAI, MiniMax
    (embo-01), SiliconFlow/BGE, or a local Ollama /v1 shim for fully-
    offline setups. Validation performs one real call and records the
    vector dimensionality (required at collection creation).

    Actions:
    - status: current configuration (never exposes the key).
    - enable: validate and persist (model + api_key required; api_key may
      also come from the PDF_CAPTURE_EMBEDDING_API_KEY env var).
    - disable: turn embedding features off.

    Returns:
        JSON with ok, message, dimensions.
    """
    try:
        from pdf_capture_mcp import embedding_client as emb

        if action == "status":
            return _json({"ok": True, **emb.get_embedding_info()})
        if action == "disable":
            return _json(emb.disable_embedding())
        if action != "enable":
            return _json(
                {"ok": False, "error": f"Invalid action: {action!r}. Use: status, enable, disable"}
            )
        resolved = api_key.strip() or os.getenv("PDF_CAPTURE_EMBEDDING_API_KEY", "").strip()
        if not resolved:
            return _json(
                {
                    "ok": False,
                    "error": "api_key required (or set PDF_CAPTURE_EMBEDDING_API_KEY).",
                }
            )
        result = emb.setup_embedding(
            model=model, api_key=resolved, api_base=api_base, provider=provider
        )
        result.pop("api_key", None)
        return _json(result)
    except Exception as exc:  # noqa: BLE001 — tool boundary
        logger.error("setup_embedding failed: %s", traceback.format_exc())
        return _json({"ok": False, "error": str(exc)})


@mcp.tool()
def build_vector_index(package_dir: str) -> str:
    """Index a knowledge package into the Qdrant vector store (incremental).

    Embedded Qdrant runs locally with zero services (file-persisted under
    <output_root>/vector_store); set PDF_CAPTURE_QDRANT_URL to use a
    Docker/cluster deployment with the same API. Content-addressed chunk
    ids make re-indexing incremental: unchanged chunks cost nothing, only
    new/changed chunks are embedded, vanished chunks are deleted.

    Refuses to index a stale package (main markdown edited after chunking
    — content_hash mismatch) instead of silently indexing drifted data.

    Args:
        package_dir: Path to a knowledge package (contains data/metadata.json).

    Returns:
        JSON with ok, embedded/unchanged/deleted counts, store location.
    """
    try:
        from pdf_capture_mcp.rag_store import build_vector_index as do_build

        return _json(do_build(package_dir))
    except Exception as exc:  # noqa: BLE001 — tool boundary
        logger.error("build_vector_index failed: %s", traceback.format_exc())
        return _json({"ok": False, "error": str(exc)})


@mcp.tool()
def search_corpus(
    query: str,
    top_k: int = 5,
    doc_id: str = "",
    chunk_type: str = "",
    page_from: int = 0,
    page_to: int = 0,
) -> str:
    """Semantic search across all indexed knowledge packages.

    This tool IS the RAG API: dense vector similarity plus metadata
    filters, returning content with heading_path and page for citation.

    Args:
        query: Natural-language query (embedded with the configured model).
        top_k: Number of results (1-50).
        doc_id: Restrict to one document (from metadata.json / previous hits).
        chunk_type: Restrict to 'text' | 'table' | 'figure' | 'code'.
        page_from: Minimum page number (1-based, 0 = no bound).
        page_to: Maximum page number (0 = no bound).

    Returns:
        JSON with hits: [{score, title, heading_path, page, chunk_type,
        content, doc_id, chunk_id}].
    """
    try:
        from pdf_capture_mcp.rag_store import search_corpus as do_search

        return _json(
            do_search(
                query,
                top_k=top_k,
                doc_id=doc_id,
                chunk_type=chunk_type,
                page_from=page_from,
                page_to=page_to,
            )
        )
    except Exception as exc:  # noqa: BLE001 — tool boundary
        logger.error("search_corpus failed: %s", traceback.format_exc())
        return _json({"ok": False, "error": str(exc)})


# ── Per-file circuit breaker for batch conversion (v0.9.3) ──────────────
# Pilot forensics: one formula-dense 58-page paper ran 115 minutes. Without
# a breaker, a single slow document can eat a whole overnight budget. Each
# file runs in a fresh subprocess: timeouts are enforceable (terminate) and
# engine state/memory is returned to the OS after every document.


def _pipeline_child(pdf_str: str, kwargs: dict[str, Any], queue: Any, parent_pid: int = 0) -> None:
    """Subprocess entry (module-level for spawn picklability).

    v0.11.1: parent watchdog — a batch coordinator dying must not leave
    an orphaned conversion racing the next run's output directories.
    """
    try:
        if parent_pid:
            _watch_parent(parent_pid)
        result = _run_pipeline(Path(pdf_str), **kwargs)
        result.pop("markdown_text", None)  # keep the pipe payload small
        queue.put(result)
    except Exception as exc:  # noqa: BLE001 — child must always report
        queue.put({"ok": False, "error": str(exc)[:300]})


def _run_pipeline_isolated(
    pdf: Path,
    kwargs: dict[str, Any],
    timeout_s: float,
    target: Any = _pipeline_child,
) -> dict[str, Any]:
    """Run one conversion in a fresh subprocess with a hard timeout.

    Reads the result from the queue BEFORE joining (large payloads deadlock
    the pipe otherwise). On timeout the child is terminated and the
    inference services are restarted — an overloaded document likely left
    their queues poisoned (pilot contagion finding).
    """
    import multiprocessing as mp
    import queue as queue_mod

    from pdf_capture_mcp.engines.marker_engine import restart_inference_services

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    # daemon=False: this child runs marker, which spawns its own worker
    # processes — daemonic processes may not have children (v0.9.4 fix for
    # segmented extraction nested inside the batch per-file isolation).
    _ensure_healthy_stdin()
    proc = ctx.Process(target=target, args=(str(pdf), kwargs, q, os.getpid()), daemon=False)
    proc.start()
    try:
        result: dict[str, Any] = q.get(timeout=timeout_s)
        proc.join(30)
        return result
    except queue_mod.Empty:
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
        restart_inference_services()
        return {
            "ok": False,
            "timed_out": True,
            "error": f"Per-file timeout after {timeout_s / 60:.0f} min — conversion "
            "terminated, inference services restarted for the next file.",
        }
    finally:
        q.close()


@mcp.tool()
def batch_convert(
    dir_path: str,
    out_dir: str = "",
    vault_path: str = "",
    category: str = "",
    index: bool = False,
    engine: str = "auto",
    skip_existing: bool = True,
    per_file_timeout_minutes: int = 40,
) -> str:
    """Convert every PDF in a directory into knowledge packages (async job).

    Runs as a background job (directory-scale work always exceeds MCP
    timeouts); poll with get_job_status. Per file: convert -> optionally
    export to an Obsidian vault -> optionally index into the vector store.
    One file failing never aborts the batch — results carry per-file status.

    Deduplication: doc_id is content-addressed (sha256 of the PDF bytes),
    so with skip_existing=True a PDF whose package already exists in
    out_dir is skipped regardless of its filename.

    Args:
        dir_path: Directory scanned recursively for *.pdf (AppleDouble
            '._*' files ignored).
        out_dir: Package root (default $PDF_CAPTURE_OUTPUT_ROOT or
            ~/Documents/pdf-capture).
        vault_path: If set, each package is copied into this Obsidian vault.
        category: Optional vault subfolder (used with vault_path).
        index: If True and embedding is configured, each package is indexed
            into the vector store after conversion.
        engine: Extraction engine ('auto', 'marker', 'mineru').
        skip_existing: Skip PDFs whose doc_id already has a package.
        per_file_timeout_minutes: Circuit breaker — a document exceeding
            this budget is terminated (recorded as timed_out) and the batch
            moves on; inference services are restarted to clear any backlog
            the oversized document left behind. Each file runs in a fresh
            subprocess, so memory is fully returned between documents.

    Returns:
        JSON with job_id — poll get_job_status(job_id) for progress/results.
    """
    try:
        src = Path(dir_path).expanduser()
        if not src.is_dir():
            return _json({"ok": False, "error": f"Not a directory: {dir_path}"})
        pdfs = sorted(p for p in src.rglob("*.pdf") if not p.name.startswith("._") and p.is_file())
        if not pdfs:
            return _json({"ok": False, "error": f"No PDF files found under {src}"})
        if not out_dir.strip():
            out_dir = os.environ.get(
                "PDF_CAPTURE_OUTPUT_ROOT", str(Path.home() / "Documents" / "pdf-capture")
            )

        from pdf_capture_mcp.jobs import create_job, update_stage

        def _target(job: dict[str, Any]) -> dict[str, Any]:
            import json as _json_mod

            from pdf_capture_mcp.classifier import detect_text_layer
            from pdf_capture_mcp.packaging import compute_doc_id, export_package_to_vault

            # Existing doc_ids in out_dir (content-addressed dedup)
            known: dict[str, str] = {}
            root = Path(out_dir)
            if root.is_dir():
                for meta in root.glob("*/data/metadata.json"):
                    try:
                        known[_json_mod.loads(meta.read_text())["doc_id"]] = str(meta.parent.parent)
                    except Exception:  # noqa: BLE001 — foreign folder, ignore
                        pass

            results: list[dict[str, Any]] = []
            for n, pdf in enumerate(pdfs, 1):
                update_stage(job, f"{n}/{len(pdfs)} {pdf.name}")
                entry: dict[str, Any] = {"file": pdf.name}
                try:
                    doc_id = compute_doc_id(pdf)
                    if skip_existing and doc_id in known:
                        entry.update(skipped=True, package_dir=known[doc_id])
                        results.append(entry)
                        continue
                    # Scanned files run full-page OCR — a fixed 40-min
                    # breaker would kill every large scan (v0.10.0).
                    timeout_s = per_file_timeout_minutes * 60.0
                    is_scanned, _cov = detect_text_layer(pdf)
                    if is_scanned:
                        n_pages = _quick_page_count(pdf)
                        timeout_s = max(timeout_s, n_pages * 90.0)
                        entry["scanned"] = True
                        entry["timeout_minutes"] = round(timeout_s / 60)
                    result = _run_pipeline_isolated(
                        pdf,
                        {
                            "engine": engine,
                            "enable_formula": True,
                            "out_dir": out_dir,
                            "package": True,
                        },
                        timeout_s=timeout_s,
                    )
                    entry["ok"] = bool(result.get("ok"))
                    if result.get("timed_out"):
                        entry["timed_out"] = True
                    if not entry["ok"] and result.get("error"):
                        entry["error"] = str(result["error"])[:300]
                    pkg_dir = result.get("package", {}).get("package_dir", "")
                    entry["package_dir"] = pkg_dir
                    entry["qc_verdict"] = result.get("qc_verdict", "")
                    if pkg_dir:
                        known[doc_id] = pkg_dir
                        if vault_path.strip():
                            exp = export_package_to_vault(pkg_dir, vault_path, category)
                            if exp.get("conflict"):
                                entry["vault_conflict"] = True
                            entry["vault"] = exp.get("dest", exp.get("error", ""))
                        if index:
                            from pdf_capture_mcp.rag_store import (
                                build_vector_index as _build,
                            )

                            idx = _build(pkg_dir)
                            entry["indexed"] = (
                                idx.get("embedded", 0) if idx.get("ok") else idx.get("error")
                            )
                except Exception as exc:  # noqa: BLE001 — batch must not abort
                    entry["ok"] = False
                    entry["error"] = str(exc)[:300]
                results.append(entry)

            converted = sum(1 for r in results if r.get("ok"))
            skipped = sum(1 for r in results if r.get("skipped"))
            return {
                "total": len(pdfs),
                "converted": converted,
                "skipped": skipped,
                "failed": len(pdfs) - converted - skipped,
                "results": results,
            }

        job = create_job(
            "batch_convert",
            _target,
            params={"dir_path": str(src), "files": len(pdfs), "out_dir": out_dir},
        )
        return _json(
            {
                "ok": True,
                "async": True,
                "job_id": job["job_id"],
                "files": len(pdfs),
                "hint": (
                    f"Batch conversion of {len(pdfs)} PDFs started. "
                    f"Poll with get_job_status(job_id='{job['job_id']}')."
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001 — tool boundary
        logger.error("batch_convert failed: %s", traceback.format_exc())
        return _json({"ok": False, "error": str(exc)})


@mcp.tool()
def get_job_status(job_id: str = "") -> str:
    """Get the status of a background job, or list recent jobs.

    Args:
        job_id: The job id returned by pdf_to_markdown / download_models.
            Pass an empty string to list the 10 most recent jobs.

    Returns:
        JSON with status/stage/elapsed. For finished conversion jobs, includes
        markdown_path and a short content preview (first 2000 chars).
    """
    from pdf_capture_mcp.jobs import get_job, list_recent, public_view

    if not job_id.strip():
        return _json({"ok": True, "jobs": [public_view(j) for j in list_recent()]})

    job = get_job(job_id.strip())
    if job is None:
        return _json({"ok": False, "error": f"Unknown job_id: {job_id!r}"})

    view = public_view(job)
    result = job.get("result") or {}
    md_path = result.get("markdown_path", "")
    if job["status"] == "done" and md_path and Path(md_path).exists():
        text = Path(md_path).read_text(encoding="utf-8", errors="replace")
        view["markdown_preview"] = text[:2000]
        view["markdown_chars"] = len(text)
    return _json({"ok": True, **view})


# ── Tool 2: extract_tables ──────────────────────────────────────────────────


@mcp.tool()
def extract_tables(
    pdf_path: str,
    strategy: str = "pdfplumber",
    max_tables: int = 50,
) -> str:
    """Extract structured tables from a PDF document.

    Strategies:
    - pdfplumber: Rule-based table detection (lattice/stream). Always available.
    - tatr: Deep-learning table structure detection (requires torch + transformers).
    - all: Run both strategies.

    Args:
        pdf_path: Absolute path to the PDF file.
        strategy: Extraction strategy ('pdfplumber', 'tatr', 'all').
        max_tables: Maximum number of tables to extract.

    Returns:
        JSON with tables list (markdown + structured rows) and stats.
    """
    try:
        pdf = _resolve_pdf(pdf_path)
        results: dict[str, Any] = {"ok": True, "stats": {}}

        if strategy in ("pdfplumber", "all"):
            from pdf_capture_mcp.extractors.tables import extract_tables as pp_extract

            pp_result = pp_extract(pdf, max_tables=max_tables)
            results["pdfplumber"] = pp_result

        if strategy in ("tatr", "all"):
            try:
                from pdf_capture_mcp.extractors.tatr import extract_tables_tatr

                tatr_result = extract_tables_tatr(str(pdf))
                results["tatr"] = tatr_result
            except ImportError:
                results["tatr"] = {
                    "ok": False,
                    "error": "TATR requires: pip install pdf-capture-mcp[ml]",
                }
            except Exception as exc:
                results["tatr"] = {"ok": False, "error": str(exc)}

        pp_count = (results.get("pdfplumber") or {}).get("stats", {}).get("total_tables", 0)
        tatr_count = (results.get("tatr") or {}).get("count", 0)
        results["stats"] = {
            "pdfplumber_tables": pp_count,
            "tatr_tables": tatr_count,
            "strategy": strategy,
        }
        return _json(results)

    except Exception as exc:
        return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


# ── Tool 3: classify_document ───────────────────────────────────────────────


@mcp.tool()
def classify_document(pdf_path: str) -> str:
    """Analyze a PDF and classify its document type.

    Returns document type (academic_paper, consulting_report, policy_doc, etc.),
    confidence score, detected features (formulas, tables), and language.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        JSON with doc_type, confidence, has_formulas, has_tables, page_count, language.
    """
    try:
        pdf = _resolve_pdf(pdf_path)

        from pdf_capture_mcp.classifier import classify_document as do_classify

        result = do_classify(pdf)
        return _json(
            {
                "ok": True,
                "doc_type": result.doc_type,
                "confidence": result.confidence,
                "source": result.source,
                "has_formulas": result.has_formulas,
                "has_tables": result.has_tables,
                "page_count": result.page_count,
                "language": result.language,
                "file": {
                    "path": str(pdf),
                    "name": pdf.name,
                    "size_mb": round(pdf.stat().st_size / 1048576, 2),
                },
            }
        )

    except Exception as exc:
        return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


# ── Tool 4: pdf_info ────────────────────────────────────────────────────────


@mcp.tool()
def pdf_info(pdf_path: str) -> str:
    """Get PDF metadata and statistics without running the full pipeline.

    Returns page count, text layer status, scanned detection, file size,
    and document metadata (title, author).

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        JSON with page_count, has_text_layer, is_scanned, metadata, size.
    """
    try:
        pdf = _resolve_pdf(pdf_path)

        info: dict[str, Any] = {
            "ok": True,
            "path": str(pdf),
            "name": pdf.name,
            "size_mb": round(pdf.stat().st_size / 1048576, 2),
        }

        try:
            import fitz

            doc = fitz.open(str(pdf))
            info["page_count"] = doc.page_count
            info["metadata"] = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
            }

            # Text layer detection
            sample_pages = min(3, doc.page_count)
            text_chars = sum(len(doc[i].get_text().strip()) for i in range(sample_pages))
            info["has_text_layer"] = text_chars > 50
            info["avg_chars_per_page"] = text_chars // max(sample_pages, 1)

            doc.close()

            # Scanned detection — the same detector the pipeline consumes
            # (classifier.detect_text_layer, v0.10.0), evenly sampled.
            from pdf_capture_mcp.classifier import detect_text_layer

            is_scanned, coverage = detect_text_layer(pdf)
            info["is_scanned"] = is_scanned
            info["text_layer_coverage"] = coverage
        except ImportError:
            info["page_count"] = -1
            info["has_text_layer"] = None
            info["note"] = "pymupdf not installed — limited inspection"

        return _json(info)

    except Exception as exc:
        return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
