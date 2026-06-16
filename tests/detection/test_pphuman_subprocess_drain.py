"""Tests for robust stdout/stderr draining in PPHumanPipelineSubprocessManager.

Per FixReports/UNIFIED_STREAM_2026-06-14.md §5 + the operator's
2026-06-14 directive to "fix the PP-Human subprocess hang":

  * PP-Human writes substantial output to stdout (Python banners,
    config dumps, model-load progress). The Linux default pipe
    buffer is 64 KiB; once full, the next ``print()`` in the
    child blocks, deadlocking the inference threads (which need
    the GIL held by the main thread).
  * PATCH-051 added a stderr-tap monitor but did NOT add a
    stdout-tap. This module pins the contract that BOTH pipes
    are drained continuously, that neither pipe can deadlock the
    subprocess on its own, and that the captured tails are
    surfaced in stall / crash logs.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_subprocess_via_manager(manager, cam_id: str) -> None:
    """Start the manager's monitor for ``cam_id`` and wait for subprocess exit.

    Used to drive the manager against a fake Popen that the test
    controls. We do not call ``manager.start()`` (which launches a
    real PP-Human) — we splice in our own Popen via the manager's
    internal dict.
    """
    raise NotImplementedError  # placeholder; see dedicated test bodies


def _make_chatty_python_script(
    out_lines: int = 5000,
    err_lines: int = 0,
) -> str:
    """Return a Python script that prints ``out_lines`` lines to stdout
    (and optionally ``err_lines`` to stderr) and exits 0.

    Each line is ~80 bytes. ``out_lines=5000`` yields ~400 KiB of
    stdout — well over the 64 KiB Linux pipe buffer — so a parent
    that does NOT drain stdout will deadlock on read.
    """
    body = textwrap.dedent(
        f"""
        import sys, time
        for i in range({out_lines}):
            sys.stdout.write(f"stdout-{{i:06d}} " + "x" * 60 + "\\n")
            sys.stdout.flush()
        for i in range({err_lines}):
            sys.stderr.write(f"stderr-{{i:06d}} " + "y" * 60 + "\\n")
            sys.stderr.flush()
        sys.exit(0)
        """
    ).strip()
    return body


# ---------------------------------------------------------------------------
# 1. stdout and stderr are BOTH drained
# ---------------------------------------------------------------------------


def test_manager_drains_stdout_and_stderr(tmp_path: Path) -> None:
    """When a subprocess writes heavily to BOTH stdout and stderr,
    the manager must drain both and the subprocess must complete
    within a few seconds (not deadlock on pipe-full)."""
    from app.detection.pphuman_pipeline import PPHumanPipelineSubprocessManager
    from app.detection.pphuman_pipeline import PPHumanDetectorAdapter

    # Use a stub adapter — we don't want to launch a real pipeline.
    adapter = PPHumanDetectorAdapter()
    mgr = PPHumanPipelineSubprocessManager(
        adapter=adapter,
        cameras=[("CAM_01", "ignored.mp4")],
        output_root=str(tmp_path),
    )

    script = _make_chatty_python_script(out_lines=2000, err_lines=2000)
    script_path = tmp_path / "chatty.py"
    script_path.write_text(script)

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Splice into the manager's state and start a monitor manually,
    # so we can assert on stdout_logs / stderr_logs without
    # ``manager.start()`` (which would try to launch a real pipeline).
    mgr._procs["CAM_01"] = proc
    mgr._stdout_logs = {"CAM_01": []}
    mgr._stderr_logs = {"CAM_01": []}
    mon = threading.Thread(
        target=mgr._monitor_subprocess,
        args=("CAM_01", "stub.mp4", tmp_path, proc),
        daemon=True,
    )
    mon.start()

    # Subprocess must finish in well under 30s — if stdout isn't
    # drained, this hangs forever on a 64 KiB buffer.
    mon.join(timeout=30)
    assert not mon.is_alive(), (
        "monitor thread is still alive — stdout/stderr drain is "
        "deadlocked on pipe-full"
    )
    proc.wait(timeout=5)
    assert proc.returncode == 0

    stdout_lines = mgr._stdout_logs["CAM_01"]
    stderr_lines = mgr._stderr_logs["CAM_01"]
    # The ring buffer caps at 200 lines; the subprocess produced
    # 2000, so we expect the buffer to be full (200) with the
    # *last* 200 lines captured (i.e. "stdout-001800" through
    # "stdout-001999"). The contract under test is "the child
    # did not deadlock on pipe-full and the buffer is the most
    # recent 200 lines, not the first 200".
    assert len(stdout_lines) == 200, (
        f"expected stdout ring buffer to be full (200 lines), "
        f"got {len(stdout_lines)}"
    )
    assert len(stderr_lines) == 200, (
        f"expected stderr ring buffer to be full (200 lines), "
        f"got {len(stderr_lines)}"
    )
    # The most recent line should be the last one the script wrote.
    assert stdout_lines[-1].startswith("stdout-001999"), (
        f"expected most recent stdout line to be stdout-001999, "
        f"got {stdout_lines[-1]!r}"
    )
    assert stderr_lines[-1].startswith("stderr-001999"), (
        f"expected most recent stderr line to be stderr-001999, "
        f"got {stderr_lines[-1]!r}"
    )


# ---------------------------------------------------------------------------
# 2. stdout-only spam cannot deadlock the monitor
# ---------------------------------------------------------------------------


def test_stdout_only_spam_does_not_deadlock(tmp_path: Path) -> None:
    """A child that writes heavily to stdout but nothing to stderr
    must still complete. (PATCH-051 would pass this — it drains
    stderr. The new contract: stdout is drained too.)"""
    from app.detection.pphuman_pipeline import PPHumanPipelineSubprocessManager
    from app.detection.pphuman_pipeline import PPHumanDetectorAdapter

    adapter = PPHumanDetectorAdapter()
    mgr = PPHumanPipelineSubprocessManager(adapter=adapter, cameras=[], output_root=str(tmp_path))

    script_path = tmp_path / "stdout_only.py"
    script_path.write_text(_make_chatty_python_script(out_lines=2000, err_lines=0))

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    mgr._procs["CAM_01"] = proc
    mgr._stdout_logs = {"CAM_01": []}
    mgr._stderr_logs = {"CAM_01": []}
    mon = threading.Thread(target=mgr._monitor_subprocess, args=("CAM_01", "stub.mp4", tmp_path, proc), daemon=True)
    mon.start()
    mon.join(timeout=30)
    assert not mon.is_alive(), "stdout-only spam deadlocked the monitor"
    proc.wait(timeout=5)
    assert proc.returncode == 0
    assert len(mgr._stdout_logs["CAM_01"]) == 200
    assert mgr._stdout_logs["CAM_01"][-1].startswith("stdout-001999")


# ---------------------------------------------------------------------------
# 3. stderr-only spam cannot deadlock the monitor
# ---------------------------------------------------------------------------


def test_stderr_only_spam_does_not_deadlock(tmp_path: Path) -> None:
    """A child that writes heavily to stderr but nothing to stdout
    must still complete. (PATCH-051 covers this; this test pins it
    against future regressions.)"""
    from app.detection.pphuman_pipeline import PPHumanPipelineSubprocessManager
    from app.detection.pphuman_pipeline import PPHumanDetectorAdapter

    adapter = PPHumanDetectorAdapter()
    mgr = PPHumanPipelineSubprocessManager(adapter=adapter, cameras=[], output_root=str(tmp_path))

    script_path = tmp_path / "stderr_only.py"
    script_path.write_text(_make_chatty_python_script(out_lines=0, err_lines=2000))

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    mgr._procs["CAM_01"] = proc
    mgr._stdout_logs = {"CAM_01": []}
    mgr._stderr_logs = {"CAM_01": []}
    mon = threading.Thread(target=mgr._monitor_subprocess, args=("CAM_01", "stub.mp4", tmp_path, proc), daemon=True)
    mon.start()
    mon.join(timeout=30)
    assert not mon.is_alive(), "stderr-only spam deadlocked the monitor"
    proc.wait(timeout=5)
    assert proc.returncode == 0
    assert len(mgr._stderr_logs["CAM_01"]) == 200
    assert mgr._stderr_logs["CAM_01"][-1].startswith("stderr-001999")


# ---------------------------------------------------------------------------
# 4. non-zero exit includes stdout tail and stderr tail
# ---------------------------------------------------------------------------


def test_nonzero_exit_surfaces_stdout_and_stderr_tails(tmp_path: Path) -> None:
    """A child that exits non-zero must have its stdout AND stderr
    tails available via ``manager.stdout_logs`` /
    ``manager.stderr_logs`` so the operator report shows what
    killed it."""
    from app.detection.pphuman_pipeline import PPHumanPipelineSubprocessManager
    from app.detection.pphuman_pipeline import PPHumanDetectorAdapter

    adapter = PPHumanDetectorAdapter()
    mgr = PPHumanPipelineSubprocessManager(adapter=adapter, cameras=[], output_root=str(tmp_path))

    script_path = tmp_path / "crashy.py"
    script_path.write_text(textwrap.dedent("""
        import sys
        sys.stdout.write("out-line-A\\n")
        sys.stdout.write("out-line-B\\n")
        sys.stdout.flush()
        sys.stderr.write("err-line-A\\n")
        sys.stderr.write("err-line-B\\n")
        sys.stderr.flush()
        sys.exit(7)
    """).strip())

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    mgr._procs["CAM_01"] = proc
    mgr._stdout_logs = {"CAM_01": []}
    mgr._stderr_logs = {"CAM_01": []}
    mon = threading.Thread(target=mgr._monitor_subprocess, args=("CAM_01", "stub.mp4", tmp_path, proc), daemon=True)
    mon.start()
    mon.join(timeout=15)
    assert not mon.is_alive()
    proc.wait(timeout=5)
    assert proc.returncode == 7

    out = mgr.stdout_logs.get("CAM_01", [])
    err = mgr.stderr_logs.get("CAM_01", [])
    assert "out-line-A" in out
    assert "out-line-B" in out
    assert "err-line-A" in err
    assert "err-line-B" in err


# ---------------------------------------------------------------------------
# 5. stall watchdog includes stdout/stderr tails
# ---------------------------------------------------------------------------


def test_stall_diagnosis_includes_pipe_tails(tmp_path: Path, caplog) -> None:
    """When the manager detects a non-zero exit, the captured
    stdout tail AND stderr tail MUST appear in the log line so the
    operator can diagnose without ``docker exec``."""
    from app.detection.pphuman_pipeline import PPHumanPipelineSubprocessManager
    from app.detection.pphuman_pipeline import PPHumanDetectorAdapter

    adapter = PPHumanDetectorAdapter()
    mgr = PPHumanPipelineSubprocessManager(adapter=adapter, cameras=[], output_root=str(tmp_path))

    script_path = tmp_path / "crashy2.py"
    script_path.write_text(textwrap.dedent("""
        import sys
        sys.stdout.write("OUT_TAIL_MARKER\\n"); sys.stdout.flush()
        sys.stderr.write("ERR_TAIL_MARKER\\n"); sys.stderr.flush()
        sys.exit(3)
    """).strip())

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    mgr._procs["CAM_01"] = proc
    mgr._stdout_logs = {"CAM_01": []}
    mgr._stderr_logs = {"CAM_01": []}

    import logging
    with caplog.at_level(logging.ERROR):
        mon = threading.Thread(
            target=mgr._monitor_subprocess, args=("CAM_01", "stub.mp4", tmp_path, proc), daemon=True,
        )
        mon.start()
        mon.join(timeout=15)
    proc.wait(timeout=5)

    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "OUT_TAIL_MARKER" in log_text, (
        f"stdout tail missing from crash log:\n{log_text}"
    )
    assert "ERR_TAIL_MARKER" in log_text, (
        f"stderr tail missing from crash log:\n{log_text}"
    )


# ---------------------------------------------------------------------------
# 6. watchdog fails clearly if no frames within timeout
# ---------------------------------------------------------------------------


def test_watchdog_fails_clearly_on_no_frame_timeout() -> None:
    """A StreamWatchdog constructed and not fed any frame signals
    must report healthy=False with a clear reason."""
    from app.detection.pphuman_pipeline import StreamWatchdog

    wd = StreamWatchdog(stall_timeout_seconds=0.1)
    # No note_frame() ever called.
    assert wd.healthy is False
    assert wd.stall_reason  # non-empty
    assert "no_frames" in wd.stall_reason or "stall" in wd.stall_reason.lower()


# ---------------------------------------------------------------------------
# 7. PP-Human direct push mode does not start old ffmpeg streamer
# ---------------------------------------------------------------------------


def test_unified_stream_does_not_start_old_ffmpeg_streamer() -> None:
    """In unified-stream mode the operator's ffmpeg streamer is
    disabled. This is enforced by .env (see
    test_unified_stream_mediamtx_streamer_disabled). Pinned here
    again at the runtime side: ``PPHumanPipelineSubprocessManager``
    must not import or instantiate the legacy streamer module."""
    import app.detection.pphuman_pipeline as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    forbidden = (
        "ffmpeg_streamer",
        "ffmpeg_subprocess",
        "LegacyFFmpegStreamer",
        "MtxStreamer",
    )
    for token in forbidden:
        assert token not in src, (
            f"pphuman_pipeline must not reference the legacy streamer "
            f"module token {token!r} in unified-stream mode"
        )


# ---------------------------------------------------------------------------
# 8. expected MediaMTX paths remain CAM_01 / CAM_02
# ---------------------------------------------------------------------------


def test_expected_publish_paths_remain_cam_basename() -> None:
    """After the drain fix, the basename contract must be unchanged:
    CAM_01.mp4 → sota-paddle-mtmc/CAM_01, etc."""
    from app.detection.pphuman_pipeline import expected_publish_path

    pushurl = "rtsp://198.51.100.20:8554/sota-paddle-mtmc/"
    assert (
        expected_publish_path(pushurl, "/data/smoke/CAM_01.mp4")
        == "rtsp://198.51.100.20:8554/sota-paddle-mtmc/CAM_01"
    )
    assert (
        expected_publish_path(pushurl, "/data/smoke/CAM_02.mp4")
        == "rtsp://198.51.100.20:8554/sota-paddle-mtmc/CAM_02"
    )


# ---------------------------------------------------------------------------
# 9. PYTHONUNBUFFERED is set in the subprocess env
# ---------------------------------------------------------------------------


def test_subprocess_env_sets_pythonunbuffered(tmp_path: Path) -> None:
    """The subprocess env must set ``PYTHONUNBUFFERED=1`` so the
    child flushes its stdout in (near) real time, not in 4 KiB
    blocks. This reduces the chance of a pipe-full deadlock even
    if the drain thread is briefly slow."""
    from app.detection.pphuman_pipeline import PPHumanDetectorAdapter

    adapter = PPHumanDetectorAdapter(
        pipeline_path=str(tmp_path / "no_such_pipeline.py"),  # don't actually launch
        config_path=str(tmp_path / "no_such_cfg.yml"),
    )

    captured: dict = {}

    def spy_popen(cmd, *args, **kwargs):
        captured["env"] = kwargs.get("env", os.environ)
        captured["cmd"] = cmd
        # Return a fake process so the caller's wait() etc. don't crash.
        class _Fake:
            def terminate(self): pass
            def kill(self): pass
            def wait(self, timeout=None): return 0
        return _Fake()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(subprocess, "Popen", spy_popen)
        # Also bypass the FileNotFoundError in build_pipeline_command
        # by writing a stub pipeline file.
        (tmp_path / "no_such_pipeline.py").write_text("# stub")
        (tmp_path / "no_such_cfg.yml").write_text("# stub")
        try:
            adapter.run_pipeline(
                camera_id="CAM_01",
                video_file="/tmp/x.mp4",
                output_dir=str(tmp_path / "out"),
                pushurl="rtsp://x:8554/y/",
            )
        except Exception:
            pass  # we only care that Popen was called

    env = captured.get("env") or {}
    assert env.get("PYTHONUNBUFFERED") == "1", (
        f"PYTHONUNBUFFERED must be '1' in the subprocess env, got "
        f"{env.get('PYTHONUNBUFFERED')!r}"
    )


# ---------------------------------------------------------------------------
# 10. stdout ring buffer is bounded
# ---------------------------------------------------------------------------


def test_stdout_ring_buffer_is_bounded(tmp_path: Path) -> None:
    """The captured stdout tail must be bounded (≤200 lines) to
    avoid memory blowup if the child is very chatty over a long
    run."""
    from app.detection.pphuman_pipeline import PPHumanPipelineSubprocessManager
    from app.detection.pphuman_pipeline import PPHumanDetectorAdapter

    adapter = PPHumanDetectorAdapter()
    mgr = PPHumanPipelineSubprocessManager(adapter=adapter, cameras=[], output_root=str(tmp_path))

    # 5000 lines → bounded to last 200.
    script_path = tmp_path / "huge_stdout.py"
    script_path.write_text(_make_chatty_python_script(out_lines=5000, err_lines=0))

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    mgr._procs["CAM_01"] = proc
    mgr._stdout_logs = {"CAM_01": []}
    mgr._stderr_logs = {"CAM_01": []}
    mon = threading.Thread(target=mgr._monitor_subprocess, args=("CAM_01", "stub.mp4", tmp_path, proc), daemon=True)
    mon.start()
    mon.join(timeout=30)
    assert not mon.is_alive()
    proc.wait(timeout=5)

    assert len(mgr._stdout_logs["CAM_01"]) <= 200, (
        f"stdout ring buffer must be ≤200 lines, got "
        f"{len(mgr._stdout_logs['CAM_01'])}"
    )


# ---------------------------------------------------------------------------
# 11. PATCH-052 cache priming override (UNIFIED_STREAM_2026-06-14 K.6)
# ---------------------------------------------------------------------------


def test_prime_model_cache_maps_reid_url_to_operator_subdir(tmp_path: Path) -> None:
    """The cfg's REID URL is ``reid_model.zip`` → cache dir
    ``reid_model/``, but the operator's actual ReID model is at
    ``strongbaseline_r50_30e_pa100k/``. The priming must create a
    symlink at ``cache_root/reid_model`` pointing at the
    operator's strongbaseline subdir, NOT an empty dir."""
    from app.detection.pphuman_pipeline import PPHumanDetectorAdapter

    adapter = PPHumanDetectorAdapter()
    model_dir = tmp_path / "models" / "pphuman"
    model_dir.mkdir(parents=True)
    # Operator has strongbaseline but NOT reid_model.
    (model_dir / "strongbaseline_r50_30e_pa100k").mkdir()
    (model_dir / "strongbaseline_r50_30e_pa100k" / "infer_cfg.yml").write_text("model: ReID")
    cache_root = tmp_path / "cache"
    adapter._prime_model_cache(cache_root=cache_root, model_dir=str(model_dir))

    # The cache entry ``reid_model`` should be a symlink (or
    # directory symlink) to the operator's strongbaseline dir.
    reid_link = cache_root / "reid_model"
    assert reid_link.exists(), "reid_model cache entry must exist"
    target = reid_link.resolve()
    assert (target / "infer_cfg.yml").exists(), (
        f"reid_model cache entry should resolve to the operator's "
        f"strongbaseline subdir; resolved to {target}"
    )


def test_prime_model_cache_repairs_broken_symlink(tmp_path: Path) -> None:
    """If the cache already has a *broken* symlink (target removed
    or renamed), the priming must remove the broken link and
    re-create a working one. Otherwise PP-Human would silently
    load an empty dir."""
    from app.detection.pphuman_pipeline import PPHumanDetectorAdapter

    adapter = PPHumanDetectorAdapter()
    model_dir = tmp_path / "models" / "pphuman"
    model_dir.mkdir(parents=True)
    (model_dir / "mot_ppyoloe_l_36e_pipeline").mkdir()
    (model_dir / "mot_ppyoloe_l_36e_pipeline" / "infer_cfg.yml").write_text("model: MOT")
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    # Plant a broken symlink (target doesn't exist).
    (cache_root / "mot_ppyoloe_l_36e_pipeline").symlink_to(
        tmp_path / "nonexistent" / "mot",
    )
    # Now prime — should remove the broken link and re-create.
    adapter._prime_model_cache(cache_root=cache_root, model_dir=str(model_dir))
    mot_link = cache_root / "mot_ppyoloe_l_36e_pipeline"
    assert mot_link.is_symlink(), "expected the broken link to be replaced"
    target = mot_link.resolve()
    assert (target / "infer_cfg.yml").exists()
