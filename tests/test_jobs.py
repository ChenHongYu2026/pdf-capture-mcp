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
