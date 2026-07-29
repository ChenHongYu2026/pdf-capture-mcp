"""Tests for v0.9.3 scale-resilience: segmented extraction, service
hygiene, per-file circuit breaker.

Every case encodes a pilot-batch forensic finding (245-page overload with
queue contagion; a 115-minute single document eating the night budget).
"""

from __future__ import annotations

import os
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
        # Body sized to clear the v0.10.0 near-empty guard (>=50 chars/page).
        body = "# Degraded segment text\n\n" + ("lorem ipsum text layer content\n" * 200)
        (out / "full_text.md").write_text(body, encoding="utf-8")
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
    """Timeout before the ready sentinel (spawn cannot finish in 50 ms)."""

    class _Named:
        name = "pymupdf"

    report = _run_segment_isolated(
        _Named(),
        tmp_path / "missing.pdf",
        tmp_path / "seg_0",
        {"enable_formula": False},
        timeout_s=120,
        ready_timeout_s=0.05,
    )
    assert report is None  # timed out while the child was spawning


def test_segment_isolation_timeout_after_ready(tmp_path):
    """Ready sentinel arrives, then the extraction itself blows the OCR
    budget — the parent must terminate the child and return None."""
    import fitz

    pdf = tmp_path / "big.pdf"
    doc = fitz.open()
    for _ in range(300):
        page = doc.new_page()
        page.insert_text((72, 100), "segment body text " * 20)
    doc.save(str(pdf))
    doc.close()

    class _Named:
        name = "pymupdf"

    report = _run_segment_isolated(
        _Named(),
        pdf,
        tmp_path / "seg_0",
        {"enable_formula": False, "enable_table": False},
        timeout_s=0.001,  # sentinel budget is generous; OCR budget is not
    )
    assert report is None


def test_segment_isolation_child_failure_before_ready(tmp_path):
    """A child that dies before models warm sends its report instead of
    the sentinel — the parent must hand it through, not hang."""

    class _Named:
        name = "no_such_engine"

    report = _run_segment_isolated(
        _Named(),
        tmp_path / "missing.pdf",
        tmp_path / "seg_0",
        {"enable_formula": False},
        timeout_s=120,
    )
    assert report is not None and not report.ok
    assert report.error


# ── v0.10.0: scanned documents — honest loss, half-window retry, checkpoints ─


def _make_scanned_pdf(path: Path, pages: int = 200) -> None:
    """Image-only stand-in: pages with no text layer at all."""
    import fitz

    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    doc.save(str(path))
    doc.close()


def _make_text_pdf(path: Path, pages: int = 200) -> None:
    import fitz

    doc = fitz.open()
    line = "digital text layer content for coverage probing here"
    for _ in range(pages):
        page = doc.new_page()
        for row in range(6):
            page.insert_text((36, 60 + 14 * row), line)
    doc.save(str(path))
    doc.close()


def test_scanned_segment_loss_is_explicit_not_fake_degraded(tmp_path, monkeypatch):
    """Scanned window fails OCR twice + both halves -> NO pymupdf fallback,
    an explicit placeholder, and metadata['missing_segments'] (v0.10.0:
    empty text must never impersonate preserved content)."""
    monkeypatch.setattr(marker_engine, "restart_inference_services", lambda: 0)
    pdf = tmp_path / "scan.pdf"
    _make_scanned_pdf(pdf)

    fb_calls: list[int] = []

    class _FallbackSpy:
        def extract(self, *a, **kw):
            fb_calls.append(1)
            return ExtractReport(ok=False, engine="pymupdf", error="never called")

    def runner(eng, pdf_, seg_dir, kwargs, timeout_s):
        if Path(seg_dir).name.startswith("seg_1"):
            return None  # main window and both halves wedge
        return eng.extract(pdf_, seg_dir, **kwargs)

    report = _extract_segmented(
        _FakeEngine(),
        pdf,
        tmp_path / "ex",
        200,
        enable_formula=True,
        stage_cb=lambda s: None,
        force_ocr=True,
        _runner=runner,
        _fallback=_FallbackSpy(),
    )
    assert report.ok
    assert fb_calls == []  # pymupdf can not rescue image-only pages
    assert report.metadata["degraded_segments"] == []
    assert report.metadata["missing_segments"] == [
        {"segment": 2, "pages": [81, 120]},
        {"segment": 2, "pages": [121, 160]},
    ]
    md = Path(report.full_text_md).read_text()
    assert md.count("content NOT captured") == 2
    assert "# Segment 0" in md and "# Segment 2" in md  # neighbours intact


def test_scanned_half_window_partial_rescue(tmp_path, monkeypatch):
    """Main window fails, first half succeeds, second half fails -> only
    the failed half is reported missing; rescued content is merged."""
    monkeypatch.setattr(marker_engine, "restart_inference_services", lambda: 0)
    pdf = tmp_path / "scan.pdf"
    _make_scanned_pdf(pdf)

    def runner(eng, pdf_, seg_dir, kwargs, timeout_s):
        name = Path(seg_dir).name
        if name == "seg_1" or name == "seg_1_h1":
            return None
        return eng.extract(pdf_, seg_dir, **kwargs)

    eng = _FakeEngine()
    report = _extract_segmented(
        eng,
        pdf,
        tmp_path / "ex",
        200,
        enable_formula=True,
        stage_cb=lambda s: None,
        force_ocr=True,
        _runner=runner,
    )
    assert report.ok
    assert report.metadata["missing_segments"] == [{"segment": 2, "pages": [121, 160]}]
    md = Path(report.full_text_md).read_text()
    assert md.count("content NOT captured") == 1
    # Rescued half merged with its own image namespace
    assert "s1h0__page_0_Figure_0.jpeg" in md
    # force_ocr wired through to every engine call
    assert eng.page_ranges  # sanity


def test_force_ocr_propagates_to_segment_kwargs(tmp_path):
    seen: list[dict] = []

    class _Recorder(_FakeEngine):
        def extract(self, pdf, out_dir, **kw):
            seen.append(kw)
            return super().extract(pdf, out_dir, **kw)

    _extract_segmented(
        _Recorder(),
        tmp_path / "big.pdf",
        tmp_path / "ex",
        200,
        enable_formula=True,
        stage_cb=lambda s: None,
        force_ocr=True,
        _runner=_inline_runner,
    )
    assert seen and all(kw.get("force_ocr") is True for kw in seen)


def test_segment_checkpoint_reuse_and_invalidation(tmp_path, monkeypatch):
    """Interrupted run leaves finished-segment checkpoints; the rerun
    reuses them (zero re-extraction) unless the kwargs hash changed."""
    monkeypatch.setattr(marker_engine, "restart_inference_services", lambda: 0)
    pdf = tmp_path / "text.pdf"
    _make_text_pdf(pdf)
    ex = tmp_path / "ex"

    # Run 1: segment 1 fails twice AND its pymupdf fallback fails ->
    # the whole run errors out, but seg_0's checkpoint must survive.
    class _DeadFallback:
        def extract(self, *a, **kw):
            return ExtractReport(ok=False, engine="pymupdf", error="dead")

    r1 = _extract_segmented(
        _FakeEngine(fail_segments={1}),
        pdf,
        ex,
        200,
        enable_formula=True,
        stage_cb=lambda s: None,
        _runner=lambda eng, p, d, kw, t: (
            None if Path(d).name == "seg_1" else eng.extract(p, d, **kw)
        ),
        _fallback=_DeadFallback(),
    )
    assert not r1.ok
    assert (ex / "seg_0" / ".seg_meta.json").exists()

    # Run 2: same kwargs -> seg_0 is reused, runner only sees seg_1/seg_2.
    ran: list[str] = []

    def counting_runner(eng, p, d, kw, t):
        ran.append(Path(d).name)
        return eng.extract(p, d, **kw)

    r2 = _extract_segmented(
        _FakeEngine(),
        pdf,
        ex,
        200,
        enable_formula=True,
        stage_cb=lambda s: None,
        _runner=counting_runner,
    )
    assert r2.ok
    assert "seg_0" not in ran and set(ran) == {"seg_1", "seg_2"}
    md = Path(r2.full_text_md).read_text()
    assert "# Segment 0" in md  # checkpoint content made it into the merge
    # Successful merge cleans the checkpoints up
    assert not (ex / "seg_0").exists()

    # Run 3: different kwargs hash (force_ocr flipped) -> full re-run.
    ran.clear()
    (ex / "full_text.md").unlink()
    r3 = _extract_segmented(
        _FakeEngine(),
        pdf,
        ex,
        200,
        enable_formula=True,
        stage_cb=lambda s: None,
        force_ocr=True,
        _runner=counting_runner,
    )
    assert r3.ok and set(ran) == {"seg_0", "seg_1", "seg_2"}


def test_near_empty_fallback_is_missing_not_degraded(tmp_path, monkeypatch):
    """Cheap guard: a fallback product under 50 chars/page is loss, not a
    degradation (text-layer window variant)."""
    monkeypatch.setattr(marker_engine, "restart_inference_services", lambda: 0)
    pdf = tmp_path / "text.pdf"
    _make_text_pdf(pdf)  # window HAS a text layer -> legacy fallback path

    class _EmptyFallback:
        def extract(self, pdf_, seg_dir, **kw):
            out = Path(seg_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "full_text.md").write_text("x\n", encoding="utf-8")
            return ExtractReport(ok=True, engine="pymupdf", full_text_md=str(out / "full_text.md"))

    report = _extract_segmented(
        _FakeEngine(),
        pdf,
        tmp_path / "ex",
        200,
        enable_formula=True,
        stage_cb=lambda s: None,
        _runner=lambda eng, p, d, kw, t: (
            None if Path(d).name == "seg_1" else eng.extract(p, d, **kw)
        ),
        _fallback=_EmptyFallback(),
    )
    assert report.ok
    assert report.metadata["degraded_segments"] == []
    assert report.metadata["missing_segments"] == [{"segment": 2, "pages": [81, 160]}]
    assert "content NOT captured" in Path(report.full_text_md).read_text()


def test_segment_timeout_env_override(monkeypatch):
    import importlib

    import pdf_capture_mcp.server as srv

    monkeypatch.setenv("PDF_CAPTURE_SEGMENT_TIMEOUT_S", "777")
    importlib.reload(srv)
    assert srv.SEGMENT_TIMEOUT_S == 777
    monkeypatch.delenv("PDF_CAPTURE_SEGMENT_TIMEOUT_S")
    importlib.reload(srv)
    assert srv.SEGMENT_TIMEOUT_S == 1200


# ── v0.11.1: field lessons from the 902-page scanned-book run ──────────────


def test_watch_parent_exits_on_reparent(monkeypatch):
    """Reparenting (parent died) must trigger a hard exit."""
    import threading
    import time

    from pdf_capture_mcp.server import _watch_parent

    exited = threading.Event()
    monkeypatch.setattr(os, "_exit", lambda code: exited.set())
    # A parent pid that is guaranteed NOT ours -> loop exits immediately.
    _watch_parent(parent_pid=os.getppid() + 99999, interval_s=0.01)
    for _ in range(100):
        if exited.is_set():
            break
        time.sleep(0.01)
    assert exited.is_set()


def test_watch_parent_stays_quiet_while_parent_alive(monkeypatch):
    import threading
    import time

    from pdf_capture_mcp.server import _watch_parent

    exited = threading.Event()
    monkeypatch.setattr(os, "_exit", lambda code: exited.set())
    _watch_parent(parent_pid=os.getppid(), interval_s=0.01)
    time.sleep(0.2)
    assert not exited.is_set()


def test_ensure_healthy_stdin_repairs_dead_fd0():
    """Closing fd 0 then calling the guard must restore a stat-able fd 0."""
    from pdf_capture_mcp.server import _ensure_healthy_stdin

    saved = os.dup(0)
    try:
        os.close(0)
        _ensure_healthy_stdin()
        os.fstat(0)  # must not raise
    finally:
        os.dup2(saved, 0)
        os.close(saved)


def test_ensure_healthy_stdin_noop_on_healthy_fd0():
    from pdf_capture_mcp.server import _ensure_healthy_stdin

    before = os.fstat(0)
    _ensure_healthy_stdin()
    after = os.fstat(0)
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)


def test_sync_inference_timeout_sets_and_respects_env(monkeypatch):
    from pdf_capture_mcp.server import _sync_inference_timeout

    monkeypatch.delenv("SURYA_INFERENCE_TIMEOUT_SECONDS", raising=False)
    _sync_inference_timeout(10800)
    assert os.environ["SURYA_INFERENCE_TIMEOUT_SECONDS"] == "10800"

    # Explicit user setting always wins.
    monkeypatch.setenv("SURYA_INFERENCE_TIMEOUT_SECONDS", "1234")
    _sync_inference_timeout(10800)
    assert os.environ["SURYA_INFERENCE_TIMEOUT_SECONDS"] == "1234"

    # Never below surya's own default.
    monkeypatch.delenv("SURYA_INFERENCE_TIMEOUT_SECONDS", raising=False)
    _sync_inference_timeout(30)
    assert os.environ["SURYA_INFERENCE_TIMEOUT_SECONDS"] == "600"

    # Zero/unset budget: leave the environment alone.
    monkeypatch.delenv("SURYA_INFERENCE_TIMEOUT_SECONDS", raising=False)
    _sync_inference_timeout(0)
    assert "SURYA_INFERENCE_TIMEOUT_SECONDS" not in os.environ


def test_merge_skips_appledouble_junk(tmp_path):
    """macOS ._* metadata must never be renamed into the shared images dir."""

    class _JunkEngine(_FakeEngine):
        def extract(self, pdf, out_dir, **kw):
            report = super().extract(pdf, out_dir, **kw)
            (Path(out_dir) / "images" / "._page_0_Figure_0.jpeg").write_bytes(b"j")
            return report

    report = _extract_segmented(
        _JunkEngine(),
        tmp_path / "big.pdf",
        tmp_path / "ex",
        200,
        enable_formula=True,
        stage_cb=lambda s: None,
        _runner=_inline_runner,
    )
    assert report.ok
    names = sorted(p.name for p in (tmp_path / "ex" / "images").iterdir())
    assert not any("._" in n for n in names), names
    assert report.image_count == 3  # junk not counted as content


def test_ensure_healthy_stdin_does_not_leak_fds():
    from pdf_capture_mcp.server import _ensure_healthy_stdin

    def fd_count() -> int:
        return len(os.listdir("/dev/fd"))

    saved = os.dup(0)
    try:
        os.close(0)
        before = fd_count()
        _ensure_healthy_stdin()
        # Exactly ONE new descriptor (the repaired fd 0) — the scratch
        # /dev/null fd must have been closed after dup2.
        assert fd_count() == before + 1
    finally:
        os.dup2(saved, 0)
        os.close(saved)
