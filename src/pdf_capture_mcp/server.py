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
        "table extraction, layout cleaning, and quality control.\n\n"
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

        result = do_setup(model=model, api_key=resolved_key, api_base=api_base, provider=provider)
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


def _quick_page_count(pdf: Path) -> int:
    """Fast page count via pymupdf (returns 0 if unavailable)."""
    try:
        import fitz

        with fitz.open(str(pdf)) as doc:
            return int(doc.page_count)
    except Exception:  # noqa: BLE001 — page count is best-effort
        return 0


def _run_pipeline(
    pdf: Path,
    *,
    engine: str,
    enable_formula: bool,
    enable_table_enrich: bool,
    skip_qc: bool,
    out_dir: str,
    auto_repair: bool = True,
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
    from pdf_capture_mcp.llm_client import is_vlm_enabled

    def _stage(name: str) -> None:
        if job is not None:
            update_stage(job, name)

    # Pre-flight: check VLM availability if enrichment requested
    vlm_notice = ""
    if enable_table_enrich and not is_vlm_enabled():
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
    extract_kwargs: dict[str, Any] = {}
    if pages is not None:
        # marker expects a range string; pymupdf expects a list of indices.
        extract_kwargs["pages"] = pages
        extract_kwargs["page_range"] = ",".join(str(p) for p in pages)
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

        qc_report = {
            "verdict": qc_verdict,
            "dimensions": gate.dimensions,
            "audit_counts": audit_counts,
            # Plain dicts so both MCP JSON responses and persisted job state
            # serialize cleanly.
            "audit_fixes": [vars(f) for f in audit["fixes"]],
            "audit_issues": [vars(i) for i in audit_issues],
            "repairs": repair_actions,
        }
    elif not markdown_text:
        qc_verdict = "HALT"

    return {
        "ok": qc_verdict != "HALT",
        "markdown_text": markdown_text,
        "markdown_path": str(md_path) if md_path.exists() else "",
        "pipeline_root": str(out_dir),
        "title": pdf.stem,
        "engine": eng.name,
        "classification": {
            "doc_type": classification.doc_type,
            "confidence": classification.confidence,
            "source": classification.source,
            "has_formulas": classification.has_formulas,
            "has_tables": classification.has_tables,
        },
        "page_count": extract_report.page_count,
        "table_count": table_count,
        "image_count": extract_report.image_count,
        "qc_verdict": qc_verdict,
        "qc_report": qc_report,
        "elapsed_seconds": extract_report.elapsed_seconds,
        "vlm_notice": vlm_notice,
    }


@mcp.tool()
def pdf_to_markdown(
    pdf_path: str,
    engine: str = "auto",
    enable_formula: bool = True,
    enable_table_enrich: bool = False,
    enable_tatr: bool = False,
    skip_qc: bool = False,
    out_dir: str = "",
    mode: str = "auto",
    page_range: str = "",
    auto_repair: bool = True,
) -> str:
    """Convert a PDF document to high-quality structured Markdown.

    Pipeline: Engine extraction (marker/MinerU) -> table extraction (pdfplumber)
    -> layout cleaning -> QC quality gate -> output Markdown.

    Large-PDF handling: conversion of big documents can exceed MCP client
    timeouts. In mode='auto' (default), PDFs above ~15 pages are converted in
    a background job — the call returns a job_id immediately; poll progress
    with get_job_status(job_id). The result markdown is always written to
    <out_dir>/extraction/full_text.md.

    Args:
        pdf_path: Absolute path to the PDF file.
        engine: Extraction engine ('marker', 'mineru', 'auto').
        enable_formula: Enable formula/equation recognition.
        enable_table_enrich: Enable VLM-based table enrichment (configure via setup_vlm).
        enable_tatr: Enable TATR deep-learning table structure detection (requires torch).
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

    Returns:
        JSON with markdown_text (sync) or job_id + polling hint (async).
    """
    try:
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
        estimated = max(1, round(effective_pages * _EST_MINUTES_PER_PAGE))
        return _json(
            {
                "ok": True,
                "async": True,
                "job_id": job["job_id"],
                "status": job["status"],
                "page_count": effective_pages,
                "estimated_minutes": estimated,
                "result_path": str(Path(out_dir) / "extraction" / "full_text.md"),
                "hint": (
                    f"Conversion started in the background (~{estimated} min). "
                    f"Poll with get_job_status(job_id='{job['job_id']}'). "
                    "When done, read the markdown from result_path."
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

            # Scanned detection
            low_text_pages = sum(
                1 for i in range(min(5, doc.page_count)) if len(doc[i].get_text().strip()) < 200
            )
            info["is_scanned"] = low_text_pages == min(5, doc.page_count)

            doc.close()
        except ImportError:
            info["page_count"] = -1
            info["has_text_layer"] = None
            info["note"] = "pymupdf not installed — limited inspection"

        return _json(info)

    except Exception as exc:
        return _json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
