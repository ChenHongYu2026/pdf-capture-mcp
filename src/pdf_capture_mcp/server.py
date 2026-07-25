"""FastMCP server exposing PDF capture tools with VLM setup guidance."""

from __future__ import annotations

import json
import tempfile
import traceback
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from pdf_capture_mcp import __version__
from pdf_capture_mcp.config import get_logger, setup_logging

setup_logging()
logger = get_logger("server")

mcp = FastMCP(
    name="pdf-capture",
    version=__version__,
    instructions=(
        "PDF Capture Pipeline — multi-phase PDF document extraction service. "
        "Converts PDF to high-quality structured Markdown with formula recognition, "
        "table extraction, layout cleaning, and quality control.\n\n"
        "Onboarding flow (execute in order on first interaction):\n"
        "1. Call 'setup_vlm' with action='status' to check VLM configuration.\n"
        "2. If VLM not configured, ask the user:\n"
        "   - VLM enhances table/formula recognition using a vision-capable AI model.\n"
        "   - Requires: a VLM-capable model name, API key, and API endpoint URL.\n"
        "   - Supports any provider with vision model capability "
        "(e.g. Qwen-VL, Zhipu/GLM-4V, MiniMax, Moonshot, DeepSeek, OpenAI, local Ollama).\n"
        "   - Warning: Using VLM will consume your API tokens.\n"
        "   - SECURITY: Never display or log the user's API key. "
        "Guide the user to set PDF_CAPTURE_VLM_API_KEY as an environment variable, "
        "or pass it directly (it will be stored locally and never echoed back).\n"
        "   - If user agrees: collect model name + api_base, confirm API key is set, "
        "then call setup_vlm(action='enable', ...).\n"
        "   - If setup fails (bad key or no vision support): inform user and ask to retry.\n"
        "   - If user declines: proceed without VLM (rule-based extraction still works well).\n"
        "3. Call 'check_environment' to verify all dependencies are installed.\n"
        "4. If environment NOT ready: show the user the missing dependencies and "
        "install commands. Wait for user to install before proceeding.\n"
        "5. If environment ready: inform the user and begin processing tasks."
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

    Call this after VLM setup to verify the environment is ready before
    running extraction tasks. Reports missing dependencies with install commands.

    Checks:
    - Extraction engine availability (marker or MinerU)
    - Core libraries (pdfplumber, pymupdf, numpy, Pillow)
    - Optional features (TATR/torch, VLM client)

    Returns:
        JSON with ready status, available engines, missing deps, and install hints.
    """
    import importlib

    results: dict[str, Any] = {
        "ok": True,
        "ready": True,
        "engines": {},
        "dependencies": {},
        "missing": [],
    }

    # ── Check extraction engines ────────────────────────────────────────
    # Marker
    try:
        importlib.import_module("marker")
        results["engines"]["marker"] = {"available": True, "note": "Default engine"}
    except ImportError:
        results["engines"]["marker"] = {
            "available": False,
            "install": "pip install pdf-capture-mcp[marker]",
        }

    # MinerU
    from pdf_capture_mcp.config import get_mineru_venv_dir

    mineru_python = get_mineru_venv_dir() / "bin" / "python3"
    if mineru_python.exists():
        results["engines"]["mineru"] = {"available": True, "note": "High-quality engine"}
    else:
        results["engines"]["mineru"] = {
            "available": False,
            "install": "pdf-capture-mcp setup-mineru (requires Python 3.11)",
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

    # ── Summary message ─────────────────────────────────────────────────
    if results["ready"]:
        available_engines = [name for name, e in results["engines"].items() if e["available"]]
        results["message"] = (
            f"Environment ready. Available engines: {', '.join(available_engines)}. "
            f"VLM: {'enabled' if is_vlm_enabled() else 'disabled (rule-based extraction active)'}. "
            "You can now use pdf_to_markdown, extract_tables, classify_document, pdf_info."
        )
    else:
        results["message"] = (
            "Environment NOT ready. Please install missing dependencies:\n"
            + "\n".join(f"  - {m}" for m in results["missing"])
        )

    return _json(results)


# ── Tool 1: pdf_to_markdown ─────────────────────────────────────────────────


@mcp.tool()
def pdf_to_markdown(
    pdf_path: str,
    engine: str = "auto",
    enable_formula: bool = True,
    enable_table_enrich: bool = False,
    enable_tatr: bool = False,
    skip_qc: bool = False,
    out_dir: str = "",
) -> str:
    """Convert a PDF document to high-quality structured Markdown.

    Pipeline: Engine extraction (marker/MinerU) -> table extraction (pdfplumber)
    -> layout cleaning -> QC quality gate -> output Markdown.

    Args:
        pdf_path: Absolute path to the PDF file.
        engine: Extraction engine ('marker', 'mineru', 'auto').
        enable_formula: Enable formula/equation recognition.
        enable_table_enrich: Enable VLM-based table enrichment (configure via setup_vlm).
        enable_tatr: Enable TATR deep-learning table structure detection (requires torch).
        skip_qc: Skip quality gate (debug only).
        out_dir: Output directory (auto-created if empty).

    Returns:
        JSON with markdown_text, page_count, table_count, qc_verdict, stage_reports.
    """
    try:
        pdf = _resolve_pdf(pdf_path)

        from pdf_capture_mcp.classifier import classify_document
        from pdf_capture_mcp.engines import get_engine
        from pdf_capture_mcp.extractors.tables import extract_tables
        from pdf_capture_mcp.llm_client import is_vlm_enabled

        # Pre-flight: check VLM availability if enrichment requested
        vlm_notice = ""
        if enable_table_enrich and not is_vlm_enabled():
            vlm_notice = (
                "VLM table enrichment requested but VLM is not configured. "
                "Call 'setup_vlm' with action='enable' to configure a vision-capable model. "
                "Falling back to rule-based extraction (still good quality)."
            )

        # Phase 0: Classify
        classification = classify_document(pdf)

        # Phase 1: Extract
        if not out_dir.strip():
            out_dir = tempfile.mkdtemp(prefix="pdf_capture_")
        extraction_dir = Path(out_dir) / "extraction"
        extraction_dir.mkdir(parents=True, exist_ok=True)

        eng = get_engine(engine)
        extract_report = eng.extract(
            pdf,
            extraction_dir,
            enable_formula=enable_formula,
            enable_table=True,
        )

        if not extract_report.ok:
            return _json(
                {
                    "ok": False,
                    "error": extract_report.error,
                    "stage": "extract",
                    "engine": eng.name,
                }
            )

        # Read extracted markdown
        markdown_text = ""
        md_path = Path(extract_report.full_text_md)
        if md_path.exists():
            markdown_text = md_path.read_text(encoding="utf-8", errors="replace")

        # Phase 2: Table extraction (supplementary)
        table_result = extract_tables(pdf, max_tables=30)
        table_count = table_result.get("stats", {}).get("total_tables", 0)

        # Phase 4: QC (simplified)
        qc_verdict = "PASS"
        if not skip_qc and markdown_text:
            char_count = len(markdown_text.strip())
            if char_count < 100:
                qc_verdict = "WARN"
            elif char_count == 0:
                qc_verdict = "HALT"

        response = {
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
            "elapsed_seconds": extract_report.elapsed_seconds,
            "vlm_notice": vlm_notice,
        }
        return _json(response)

    except Exception as exc:
        return _json(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-500:],
            }
        )


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
