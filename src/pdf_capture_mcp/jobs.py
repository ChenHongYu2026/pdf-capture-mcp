"""Background job manager for long-running tasks.

Large PDF conversions and model downloads can take far longer than a typical
MCP client timeout. Jobs run in a daemon thread and persist their state as
JSON files under ``<cache_dir>/jobs/``, so status survives server restarts
and can be polled via the ``get_job_status`` tool.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pdf_capture_mcp.config import get_cache_dir, get_logger

logger = get_logger("jobs")

# Terminal states — a job in one of these will never change again.
STATUS_DONE = "done"
STATUS_FAILED = "failed"
TERMINAL_STATUSES = (STATUS_DONE, STATUS_FAILED)

# In-memory registry (fast path); JSON files are the durable source of truth.
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _jobs_dir() -> Path:
    d = get_cache_dir() / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job_path(job_id: str) -> Path:
    return _jobs_dir() / f"{job_id}.json"


def _persist(job: dict[str, Any]) -> None:
    """Write job state to disk (best-effort; never raises)."""
    try:
        _job_path(job["job_id"]).write_text(
            json.dumps(job, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Failed to persist job %s: %s", job["job_id"], exc)


def create_job(
    kind: str,
    target: Callable[[dict[str, Any]], dict[str, Any]],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a job and start it in a background daemon thread.

    Args:
        kind: Job kind label (e.g. 'pdf_to_markdown', 'download_models').
        target: Callable receiving the job dict; may call update_stage() and
            must return a result dict merged into the job on success.
        params: Input parameters recorded on the job for later inspection.

    Returns:
        The initial job dict (status='queued') — safe to serialize immediately.
    """
    job_id = uuid.uuid4().hex[:12]
    job: dict[str, Any] = {
        "job_id": job_id,
        "kind": kind,
        "status": "queued",
        "stage": "queued",
        "params": params or {},
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "result": None,
    }
    with _lock:
        _jobs[job_id] = job
    _persist(job)

    def _run() -> None:
        job["status"] = "running"
        job["started_at"] = time.time()
        _persist(job)
        try:
            result = target(job)
            job["status"] = STATUS_DONE
            job["stage"] = STATUS_DONE
            job["result"] = result
        except Exception as exc:  # noqa: BLE001 — job boundary must capture all
            logger.error("Job %s (%s) failed: %s", job_id, kind, exc)
            job["status"] = STATUS_FAILED
            job["stage"] = STATUS_FAILED
            job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            job["finished_at"] = time.time()
            _persist(job)

    thread = threading.Thread(target=_run, name=f"job-{kind}-{job_id}", daemon=True)
    thread.start()
    return job


def update_stage(job: dict[str, Any], stage: str, **extra: Any) -> None:
    """Update the current stage of a running job (called from within target)."""
    job["stage"] = stage
    job.update(extra)
    _persist(job)


def get_job(job_id: str) -> dict[str, Any] | None:
    """Look up a job by id — memory first, then disk (survives restarts)."""
    with _lock:
        job = _jobs.get(job_id)
    if job is not None:
        return job
    path = _job_path(job_id)
    if path.exists():
        try:
            loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            # A 'running' job loaded from disk after a server restart is dead.
            if loaded.get("status") not in TERMINAL_STATUSES:
                loaded["status"] = STATUS_FAILED
                loaded["error"] = (
                    "Job state recovered from disk but the worker is no longer "
                    "running (server was likely restarted). Check whether the "
                    "output file exists — extraction may have completed."
                )
            return loaded
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load job file %s: %s", path, exc)
    return None


def list_recent(limit: int = 10) -> list[dict[str, Any]]:
    """List the most recent jobs (from disk, newest first)."""
    files = sorted(_jobs_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    jobs: list[dict[str, Any]] = []
    for path in files[:limit]:
        job = get_job(path.stem)
        if job is not None:
            jobs.append(job)
    return jobs


def public_view(job: dict[str, Any]) -> dict[str, Any]:
    """Build a client-facing summary of a job (compact, no huge payloads)."""
    now = time.time()
    started = job.get("started_at")
    finished = job.get("finished_at")
    elapsed = None
    if started:
        elapsed = round((finished or now) - started, 1)
    return {
        "job_id": job["job_id"],
        "kind": job["kind"],
        "status": job["status"],
        "stage": job["stage"],
        "elapsed_seconds": elapsed,
        "error": job.get("error"),
        "params": job.get("params"),
        "result": job.get("result"),
    }
