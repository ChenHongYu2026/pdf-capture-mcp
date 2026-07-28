"""Tests for the background job manager."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Point the job store at a temp dir so tests never touch the real cache."""
    monkeypatch.setenv("PDF_CAPTURE_CACHE_DIR", str(tmp_path))
    yield


def _wait_terminal(job_id: str, timeout: float = 5.0) -> dict[str, Any]:
    from pdf_capture_mcp.jobs import TERMINAL_STATUSES, get_job

    deadline = time.time() + timeout
    while time.time() < deadline:
        job = get_job(job_id)
        assert job is not None
        if job["status"] in TERMINAL_STATUSES:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_job_success_and_result():
    from pdf_capture_mcp.jobs import create_job

    job = create_job("test", lambda j: {"answer": 42}, params={"x": 1})
    assert job["status"] in ("queued", "running", "done")
    finished = _wait_terminal(job["job_id"])
    assert finished["status"] == "done"
    assert finished["result"] == {"answer": 42}
    assert finished["params"] == {"x": 1}


def test_job_failure_captures_error():
    from pdf_capture_mcp.jobs import create_job

    def _boom(job):
        raise ValueError("kaput")

    job = create_job("test", _boom)
    finished = _wait_terminal(job["job_id"])
    assert finished["status"] == "failed"
    assert "ValueError" in finished["error"]
    assert "kaput" in finished["error"]


def test_job_stage_updates_are_persisted(tmp_path):
    from pdf_capture_mcp.jobs import _job_path, create_job, update_stage

    def _staged(job):
        update_stage(job, "extracting")
        return {"ok": True}

    job = create_job("test", _staged)
    finished = _wait_terminal(job["job_id"])
    assert finished["status"] == "done"
    # Terminal state must be on disk
    on_disk = json.loads(_job_path(job["job_id"]).read_text(encoding="utf-8"))
    assert on_disk["status"] == "done"


def test_get_job_from_disk_marks_stale_running_as_failed():
    from pdf_capture_mcp.jobs import _jobs, _persist, get_job

    # Simulate a job left 'running' by a previous server process.
    stale = {
        "job_id": "deadbeef0000",
        "kind": "test",
        "status": "running",
        "stage": "extracting",
        "params": {},
        "created_at": time.time(),
        "started_at": time.time(),
        "finished_at": None,
        "error": None,
        "result": None,
    }
    _persist(stale)
    _jobs.pop("deadbeef0000", None)  # not in memory -> disk path

    job = get_job("deadbeef0000")
    assert job is not None
    assert job["status"] == "failed"
    assert "restarted" in job["error"]


def test_get_unknown_job_returns_none():
    from pdf_capture_mcp.jobs import get_job

    assert get_job("nope") is None


def test_list_recent_orders_newest_first():
    from pdf_capture_mcp.jobs import create_job, list_recent

    first = create_job("test", lambda j: {})
    _wait_terminal(first["job_id"])
    time.sleep(0.05)  # ensure distinct mtimes
    second = create_job("test", lambda j: {})
    _wait_terminal(second["job_id"])

    recent = list_recent()
    ids = [j["job_id"] for j in recent]
    assert ids.index(second["job_id"]) < ids.index(first["job_id"])


def test_public_view_is_compact():
    from pdf_capture_mcp.jobs import create_job, public_view

    job = create_job("test", lambda j: {"markdown_path": "/tmp/x.md"})
    finished = _wait_terminal(job["job_id"])
    view = public_view(finished)
    assert set(view) == {
        "job_id",
        "kind",
        "status",
        "stage",
        "elapsed_seconds",
        "error",
        "params",
        "result",
    }
    assert view["elapsed_seconds"] is not None


# ── v0.9.5: locking consistency (W7) + GC (W8) ──────────────────────────────


def test_concurrent_stage_updates_never_tear(tmp_path):
    """W7: hammer update_stage from the worker while polling get_job —
    readers must always see a consistent snapshot (done implies result)."""
    from pdf_capture_mcp.jobs import create_job, get_job, update_stage

    def target(job):
        for i in range(200):
            update_stage(job, f"stage-{i}", tick=i)
        return {"final": True}

    job = create_job("hammer", target)
    saw_done_without_result = False
    for _ in range(500):
        snap = get_job(job["job_id"])
        assert snap is not None
        if snap["status"] == "done" and snap.get("result") is None:
            saw_done_without_result = True
        if snap["status"] == "done":
            break
        time.sleep(0.001)
    finished = _wait_terminal(job["job_id"])
    assert not saw_done_without_result
    assert finished["result"] == {"final": True}
    # get_job returns copies, never the live dict
    a = get_job(job["job_id"])
    b = get_job(job["job_id"])
    assert a == b


def _make_stale_file(jobs_dir, name: str, age_s: float) -> None:
    import os

    p = jobs_dir / f"{name}.json"
    p.write_text(json.dumps({"job_id": name, "status": "done"}))
    old = time.time() - age_s
    os.utime(p, (old, old))


def _reset_gc_throttle():
    import pdf_capture_mcp.jobs as jobs_mod

    jobs_mod._last_gc = 0.0


def test_gc_removes_expired_files():
    from pdf_capture_mcp.jobs import _gc_jobs, _jobs_dir

    _reset_gc_throttle()
    d = _jobs_dir()
    _make_stale_file(d, "ancient", 31 * 24 * 3600)
    _make_stale_file(d, "fresh", 60)
    _gc_jobs()
    assert not (d / "ancient.json").exists()
    assert (d / "fresh.json").exists()


def test_gc_enforces_file_cap():
    import pdf_capture_mcp.jobs as jobs_mod
    from pdf_capture_mcp.jobs import _gc_jobs, _jobs_dir

    _reset_gc_throttle()
    d = _jobs_dir()
    # newest-first retention: create beyond the cap with distinct mtimes
    original_cap = jobs_mod._GC_MAX_FILES
    jobs_mod._GC_MAX_FILES = 5
    try:
        for i in range(8):
            _make_stale_file(d, f"j{i:02d}", age_s=(8 - i) * 10)
        _gc_jobs()
        remaining = sorted(p.stem for p in d.glob("*.json"))
        assert len(remaining) == 5
        assert remaining == [f"j{i:02d}" for i in range(3, 8)]  # newest kept
    finally:
        jobs_mod._GC_MAX_FILES = original_cap


def test_gc_spares_live_registry_jobs():
    import pdf_capture_mcp.jobs as jobs_mod
    from pdf_capture_mcp.jobs import _gc_jobs, _jobs_dir, _lock

    _reset_gc_throttle()
    d = _jobs_dir()
    _make_stale_file(d, "live-one", 31 * 24 * 3600)
    with _lock:
        jobs_mod._jobs["live-one"] = {"job_id": "live-one", "status": "running"}
    try:
        _gc_jobs()
        assert (d / "live-one.json").exists()  # in-memory job is immune
    finally:
        with _lock:
            jobs_mod._jobs.pop("live-one", None)


def test_gc_is_throttled():
    from pdf_capture_mcp.jobs import _gc_jobs, _jobs_dir

    _reset_gc_throttle()
    d = _jobs_dir()
    _gc_jobs()  # arms the throttle
    _make_stale_file(d, "late-expired", 31 * 24 * 3600)
    _gc_jobs()  # within the hourly window: must be a no-op
    assert (d / "late-expired.json").exists()
