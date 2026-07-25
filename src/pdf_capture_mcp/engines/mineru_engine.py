"""MinerU engine: PDF-to-Markdown via MinerU (subprocess, highest quality).

Optional engine. Requires a dedicated Python 3.11 venv with magic-pdf installed.
Use `pdf-capture-mcp setup-mineru` or call ensure_mineru_env() to auto-create.
MinerU is AGPL-3.0 licensed and runs as a separate subprocess.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from pdf_capture_mcp.config import get_logger, get_mineru_venv_dir
from pdf_capture_mcp.types import ExtractReport

logger = get_logger("engines.mineru")


def ensure_mineru_env() -> Path:
    """Create the MinerU virtual environment if it doesn't exist.

    Returns the path to the MinerU Python executable.
    Raises RuntimeError if creation fails.
    """
    venv_dir = get_mineru_venv_dir()
    python_exe = venv_dir / "bin" / "python3"

    if python_exe.exists():
        return python_exe

    logger.info("Creating MinerU virtual environment at %s ...", venv_dir)

    # Find Python 3.11 (MinerU requires it for C extensions)
    py311 = shutil.which("python3.11")
    if not py311:
        raise RuntimeError(
            "Python 3.11 not found on PATH. MinerU requires Python 3.11. "
            "Install it via: brew install python@3.11 (macOS) or apt install python3.11 (Linux)"
        )

    # Create venv
    result = subprocess.run(
        [py311, "-m", "venv", str(venv_dir)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create venv: {result.stderr[:300]}")

    # Install magic-pdf (MinerU)
    pip_exe = venv_dir / "bin" / "pip"
    logger.info("Installing magic-pdf (this may take several minutes)...")
    result = subprocess.run(
        [str(pip_exe), "install", "magic-pdf[full]", "--quiet"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to install magic-pdf: {result.stderr[:500]}")

    logger.info("MinerU environment ready at %s", venv_dir)
    return python_exe


def _compute_timeout(pdf_path: Path) -> int:
    """Dynamic timeout based on PDF size: max(600, pdf_mb * 60) seconds."""
    try:
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
    except OSError:
        size_mb = 1
    return max(600, int(size_mb * 60))


def _infer_language(pdf_path: Path) -> str:
    """Infer PDF language from text sample."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        sample = ""
        for page in reader.pages[:3]:
            text = str(page.extract_text() or "").strip()
            sample += text
            if len(sample) >= 1200:
                break
        if not sample:
            return "ch"
        cjk = sum(1 for c in sample if "\u4e00" <= c <= "\u9fff")
        latin = sum(1 for c in sample if c.isascii() and c.isalpha())
        return "ch" if cjk >= max(20, int(latin * 1.2)) else "en"
    except Exception:
        return "ch"


class MineruEngine:
    """PDF extraction engine using MinerU via subprocess."""

    @property
    def name(self) -> str:
        return "mineru"

    def is_available(self) -> bool:
        """Check if MinerU venv exists and is usable."""
        venv_dir = get_mineru_venv_dir()
        python_exe = venv_dir / "bin" / "python3"
        return python_exe.exists()

    def extract(
        self,
        pdf_path: Path,
        out_dir: Path,
        *,
        enable_formula: bool = True,
        enable_table: bool = True,
        language: str = "auto",
        backend: str = "pipeline",
        method: str = "auto",
        auto_setup: bool = True,
        **kwargs: Any,
    ) -> ExtractReport:
        """Extract PDF to Markdown using MinerU subprocess.

        Args:
            pdf_path: Source PDF file.
            out_dir: Output directory.
            enable_formula: Enable formula recognition.
            enable_table: Enable table recognition.
            language: Language hint ('auto', 'en', 'ch').
            backend: MinerU backend ('pipeline', 'vlm-auto-engine').
            method: Parse method ('auto', 'txt', 'ocr').
            auto_setup: Auto-create MinerU venv if missing.
        """
        venv_dir = get_mineru_venv_dir()
        python_exe = venv_dir / "bin" / "python3"

        if not python_exe.exists():
            if not auto_setup:
                return ExtractReport(
                    ok=False,
                    engine=self.name,
                    error=("MinerU environment not found. Run: pdf-capture-mcp setup-mineru"),
                )
            try:
                python_exe = ensure_mineru_env()
            except RuntimeError as exc:
                return ExtractReport(ok=False, engine=self.name, error=str(exc))

        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        timeout = _compute_timeout(pdf_path)

        # Resolve language
        resolved_lang = language
        if resolved_lang == "auto":
            resolved_lang = _infer_language(pdf_path)

        # Build MinerU execution script.
        # MINERU_MODEL_SOURCE: honor caller's env; default to modelscope so
        # models auto-download on first run (local requires pre-downloaded models).
        model_source = os.getenv("MINERU_MODEL_SOURCE", "modelscope")
        script = f"""\
import os, sys
os.environ.setdefault("MINERU_MODEL_SOURCE", {model_source!r})
os.environ["MINERU_PDF_RENDER_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import mineru.utils.pdf_image_tools as _pit
from concurrent.futures import ThreadPoolExecutor
_pit._create_pdf_render_executor = lambda max_workers=1: ThreadPoolExecutor(max_workers=1)
_pit._get_pdf_render_pool_capacity = lambda cpu_count=None: 1

from mineru.cli.common import do_parse, read_fn
pdf_bytes = read_fn({str(pdf_path)!r})
do_parse(
    {str(out_dir)!r},
    [{pdf_path.stem!r}],
    [pdf_bytes],
    [{resolved_lang!r}],
    backend={backend!r},
    parse_method={method!r},
    formula_enable={enable_formula!r},
    table_enable={enable_table!r},
    start_page_id=0,
    end_page_id=None,
)
print("MINERU_SUBPROCESS_OK")
"""

        script_path = Path(tempfile.mktemp(suffix=".py", prefix="mineru_"))
        try:
            script_path.write_text(script, encoding="utf-8")

            # Strip proxy env vars that may interfere
            env = {
                k: v
                for k, v in os.environ.items()
                if k.upper()
                not in (
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                    "ALL_PROXY",
                    "NO_PROXY",
                    "http_proxy",
                    "https_proxy",
                    "all_proxy",
                )
            }
            env["PYTHONPATH"] = ":".join(
                str(p) for p in sorted((venv_dir / "lib").glob("python*/site-packages"))
            )

            result = subprocess.run(
                [str(python_exe), str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )

            if result.returncode != 0:
                stderr_tail = result.stderr[-400:]
                error_msg = f"MinerU failed (rc={result.returncode}): {stderr_tail}"
                # Friendly hint for missing/corrupt model files
                if "HFValidationError" in result.stderr or "Repo id" in result.stderr:
                    error_msg += (
                        "\nHint: MinerU model files are missing or the model path is stale. "
                        "Set MINERU_MODEL_SOURCE=modelscope (or huggingface) to auto-download "
                        "models on next run, or pre-download via: mineru-models-download"
                    )
                return ExtractReport(
                    ok=False,
                    engine=self.name,
                    elapsed_seconds=round(time.time() - t0, 2),
                    error=error_msg,
                )

            if "MINERU_SUBPROCESS_OK" not in (result.stdout + result.stderr):
                return ExtractReport(
                    ok=False,
                    engine=self.name,
                    elapsed_seconds=round(time.time() - t0, 2),
                    error=f"MinerU did not confirm completion: {result.stdout[:200]}",
                )

        except subprocess.TimeoutExpired:
            return ExtractReport(
                ok=False,
                engine=self.name,
                elapsed_seconds=timeout,
                error=f"MinerU timed out after {timeout}s",
            )
        except Exception as exc:
            return ExtractReport(
                ok=False,
                engine=self.name,
                elapsed_seconds=round(time.time() - t0, 2),
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            script_path.unlink(missing_ok=True)

        # Locate output
        elapsed = round(time.time() - t0, 2)
        report = self._collect_output(pdf_path, out_dir, elapsed, backend, method)
        return report

    def _collect_output(
        self, pdf_path: Path, out_dir: Path, elapsed: float, backend: str, method: str
    ) -> ExtractReport:
        """Locate and collect MinerU output artifacts."""
        stem = pdf_path.stem
        possible_dirs = [
            out_dir / stem / "auto",
            out_dir / stem / backend,
            out_dir / stem,
            out_dir,
        ]

        content_dir: Path | None = None
        for d in possible_dirs:
            if (d / f"{stem}.md").exists() or list(d.glob("*.md")):
                content_dir = d
                break

        if content_dir is None:
            for md_file in out_dir.rglob("*.md"):
                content_dir = md_file.parent
                break

        if content_dir is None:
            return ExtractReport(
                ok=False,
                engine=self.name,
                elapsed_seconds=elapsed,
                error=f"MinerU completed but no output found in {out_dir}",
            )

        # Collect artifacts
        md_files = list(content_dir.glob("*.md"))
        json_files = list(content_dir.glob("*content_list.json")) or list(
            content_dir.glob("*.json")
        )
        image_dir = content_dir / "images"
        table_files = list(content_dir.rglob("*.html"))

        full_text_md = md_files[0] if md_files else None
        content_json = json_files[0] if json_files else None

        # Promote to standard location
        target_md: Path | None = None
        if full_text_md:
            standard_md = out_dir / "full_text.md"
            if full_text_md != standard_md:
                shutil.copy2(full_text_md, standard_md)
                target_md = standard_md
            else:
                target_md = full_text_md

        # Count pages from content JSON
        page_count = 0
        if content_json and content_json.exists():
            try:
                data = json.loads(content_json.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    pages = {item.get("page_idx") for item in data if isinstance(item, dict)}
                    page_count = len(pages) if pages else len(data)
            except Exception:
                pass

        logger.info("MinerU extraction complete: %d pages, %.1fs", page_count, elapsed)

        return ExtractReport(
            ok=True,
            engine=self.name,
            full_text_md=str(target_md) if target_md else "",
            content_json=str(content_json) if content_json else "",
            content_dir=str(content_dir),
            page_count=page_count,
            image_count=len(list(image_dir.glob("*"))) if image_dir.exists() else 0,
            table_count=len(table_files),
            elapsed_seconds=elapsed,
            metadata={"backend": backend, "method": method},
        )
