"""Tests for v0.9.3 scale-resilience: segmented extraction, service
hygiene, per-file circuit breaker.

Every case encodes a pilot-batch forensic finding (245-page overload with
queue contagion; a 115-minute single document eating the night budget).
"""

from __future__ import annotations

from pathlib import Path

from pdf_capture_mcp.engines import marker_engine
from pdf_capture_mcp.server import (
    _extract_segmented,
    _run_pipeline_isolated,
    _run_segment_isolated,
)
from pdf_capture_mcp.types import ExtractReport


def _inline_runner(eng, pdf, seg_dir, kwargs, timeout_s):
    """Synchronous in-process runner for engine stubs (no spawn)."""
    return eng.extract(pdf, seg_dir, **kwargs)


class _FakeEngine:
    """Segment-aware engine stub: emits one md + one image per segment."""

    name = "fake"

    def __init__(self, fail_segments: set[int] | None = None) -> None:
        self.page_ranges: list[str] = []
        self.fail = fail_segments or set()
        self.attempts: dict[int, int] = {}

    def extract(self, pdf: Path, out_dir: Path, **kw) -> ExtractReport:
        k = int(Path(out_dir).name.split("_")[1])
        self.page_ranges.append(kw["page_range"])
        self.attempts[k] = self.attempts.get(k, 0) + 1
        if k in self.fail and self.attempts[k] == 1:
            return ExtractReport(ok=False, engine="fake", error="inference timeout")
        out = Path(out_dir)
        (out / "images").mkdir(parents=True, exist_ok=True)
        # Same bare name in every segment — the merge MUST de-collide these.
        (out / "images" / "_page_0_Figure_0.jpeg").write_bytes(b"x")
        (out / "full_text.md").write_text(
            f"# Segment {k}\n\n![](images/_page_0_Figure_0.jpeg)\n", encoding="utf-8"
        )
        return ExtractReport(
            ok=True,
            engine="fake",
            full_text_md=str(out / "full_text.md"),
            image_count=1,
            elapsed_seconds=1.0,
        )


def test_segmented_merge_order_and_image_namespacing(tmp_path):
    eng = _FakeEngine()
    report = _extract_segmented(
        eng,
        tmp_path / "big.pdf",
        tmp_path / "ex",
        200,
        enable_formula=True,
        stage_cb=lambda s: None,
        _runner=_inline_runner,
    )
    assert report.ok and report.metadata["segments"] == 3
    assert eng.page_ranges == ["0-79", "80-159", "160-199"]
    md = Path(report.full_text_md).read_text()
    # Segments merged in order
    assert md.index("# Segment 0") < md.index("# Segment 1") < md.index("# Segment 2")
    # Identically-named per-segment images de-collided with s<k>_ prefixes
    images = sorted(p.name for p in (tmp_path / "ex" / "images").iterdir())
    assert images == [
        "s0__page_0_Figure_0.jpeg",
        "s1__page_0_Figure_0.jpeg",
        "s2__page_0_Figure_0.jpeg",
    ]
    # References rewritten to the new names
    for name in images:
        assert f"images/{name}" in md
    assert "](images/_page_0_Figure_0.jpeg)" not in md


def test_segmented_retries_after_service_restart(tmp_path, monkeypatch):
    restarts: list[int] = []
    monkeypatch.setattr(
        marker_engine, "restart_inference_services", lambda: restarts.append(1) or 1
    )
    eng = _FakeEngine(fail_segments={1})
    report = _extract_segmented(
        eng,
        tmp_path / "big.pdf",
        tmp_path / "ex",
        200,
        enable_formula=True,
        stage_cb=lambda s: None,
        _runner=_inline_runner,
    )
    assert report.ok  # segment 1 succeeded on retry
    assert restarts == [1]  # exactly one hygiene restart
    assert eng.attempts[1] == 2


def test_segmented_reports_segment_on_double_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(marker_engine, "restart_inference_services", lambda: 0)

    class _AlwaysFail(_FakeEngine):
        def extract(self, pdf, out_dir, **kw):
            return ExtractReport(ok=False, engine="fake", error="dead service")

    report = _extract_segmented(
        _AlwaysFail(),
        tmp_path / "big.pdf",
        tmp_path / "ex",
        200,
        enable_formula=True,
        stage_cb=lambda s: None,
        _runner=_inline_runner,
    )
    assert not report.ok
    assert "Segment 1/3" in report.error and "pages 1-80" in report.error


# ── Circuit breaker ─────────────────────────────────────────────────────────


def test_isolated_run_round_trip(tmp_path):
    """Child converts a nonexistent pdf -> clean error dict comes back
    through the subprocess pipe (proves the full spawn round trip)."""
    result = _run_pipeline_isolated(
        tmp_path / "missing.pdf",
        {"engine": "auto", "enable_formula": False, "out_dir": str(tmp_path), "package": False},
        timeout_s=120,
    )
    assert isinstance(result, dict)
    assert result.get("timed_out") is not True  # it finished, albeit with ok=False


def test_isolated_run_timeout_terminates_and_restarts(monkeypatch, tmp_path):
    restarts: list[int] = []
    monkeypatch.setattr(
        marker_engine, "restart_inference_services", lambda: restarts.append(1) or 1
    )
    result = _run_pipeline_isolated(
        tmp_path / "missing.pdf",
        {"engine": "auto", "enable_formula": False, "out_dir": str(tmp_path), "package": False},
        timeout_s=0.05,  # expires while the child is still spawning
    )
    assert result["ok"] is False and result["timed_out"] is True
    assert restarts == [1]  # hygiene restart fired in the parent


# ── Service hygiene ─────────────────────────────────────────────────────────


def test_restart_inference_services_signals_patterns(monkeypatch):
    calls: list[list[str]] = []

    class _R:
        returncode = 0

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return _R()

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("time.sleep", lambda s: None)
    killed = marker_engine.restart_inference_services()
    assert killed == 2
    assert any("llama-server" in " ".join(c) for c in calls)
    assert any("surya" in " ".join(c) for c in calls)
    # v0.9.5 blast-radius contract: every kill is scoped to the current user
    import os as _os

    uid = str(_os.getuid())
    for c in calls:
        assert "-u" in c and uid in c, c


def test_restart_falls_back_when_anchored_misses(monkeypatch):
    """Anchored llama pattern missing -> broad (still user-scoped) fallback."""
    calls: list[list[str]] = []

    class _R:
        def __init__(self, rc):
            self.returncode = rc

    def fake_run(cmd, **kw):
        calls.append(cmd)
        pattern = cmd[-1]
        # anchored llama pattern (contains a path prefix) misses; broad hits
        if "llama-server" in pattern and "(" in pattern:
            return _R(1)
        return _R(0)

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert marker_engine.restart_inference_services() == 2
    llama_calls = [c for c in calls if "llama-server" in c[-1]]
    assert len(llama_calls) == 2  # anchored attempt, then broad fallback


# ── v0.9.4: segment breaker + honest degradation ────────────────────────────


def test_segment_degrades_to_fallback_engine(tmp_path, monkeypatch):
    """marker times out twice -> pymupdf fallback rescues the segment and
    the degradation is recorded (never silent)."""
    restarts: list[int] = []
    monkeypatch.setattr(
        marker_engine, "restart_inference_services", lambda: restarts.append(1) or 1
    )

    def timeout_runner(eng, pdf, seg_dir, kwargs, timeout_s):
        k = int(Path(seg_dir).name.split("_")[1])
        if k == 1:
            return None  # segment 1 wedges the OCR service both times
        return eng.extract(pdf, seg_dir, **kwargs)

    fallback = _FakeEngine()  # stands in for pymupdf

    def fb_extract(pdf, seg_dir, **kw):
        out = Path(seg_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "full_text.md").write_text("# Degraded segment text\n", encoding="utf-8")
        return ExtractReport(ok=True, engine="pymupdf", full_text_md=str(out / "full_text.md"))

    fallback.extract = fb_extract  # type: ignore[method-assign]

    report = _extract_segmented(
        _FakeEngine(),
        tmp_path / "big.pdf",
        tmp_path / "ex",
        200,
        enable_formula=True,
        stage_cb=lambda s: None,
        _runner=timeout_runner,
        _fallback=fallback,
    )
    assert report.ok
    assert report.metadata["degraded_segments"] == [2]  # human-numbered
    md = Path(report.full_text_md).read_text()
    assert "# Degraded segment text" in md  # content preserved
    assert "# Segment 0" in md and "# Segment 2" in md  # others full quality
    assert len(restarts) == 2  # retry hygiene + pre-degradation hygiene


def test_real_segment_isolation_round_trip(tmp_path):
    """True subprocess path with the registered pymupdf engine."""
    import fitz

    pdf = tmp_path / "t.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "hello segmented world with enough text to extract")
    doc.save(str(pdf))
    doc.close()

    class _Named:
        name = "pymupdf"

    report = _run_segment_isolated(
        _Named(),
        pdf,
        tmp_path / "seg_0",
        {"enable_formula": False, "enable_table": False, "pages": [0]},
        timeout_s=120,
    )
    assert report is not None and report.ok


def test_real_segment_isolation_timeout(tmp_path):
    class _Named:
        name = "pymupdf"

    report = _run_segment_isolated(
        _Named(),
        tmp_path / "missing.pdf",
        tmp_path / "seg_0",
        {"enable_formula": False},
        timeout_s=0.05,
    )
    assert report is None  # timed out while the child was spawning
