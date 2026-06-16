"""Tests for the real-PP-Human benchmark bridge (PATCH-051).

The production benchmark in ``scripts/benchmark_t4.py`` relies on
a three-layer chain:

    PPHumanDetectorAdapter   (subprocess launcher)
        → PPHumanPipelineSubprocessManager  (per-camera procs)
            → PPHumanFrameStateAdapter      (MOT tailer)
                → MultiCameraRunner  (per-camera worker)

A previous audit found that the subprocess-side of the chain had
three latent bugs that combined to make the benchmark silently
report ``detector_backend=real_pphuman, workers_crashed=false``
even when the official ``pipeline.py`` crashed on import. These
tests pin the new contract:

  1. The command builder uses ``sys.executable`` (the current
     Python interpreter) so the subprocess inherits the project
     venv with paddle / scipy installed.
  2. A subprocess that exits non-zero is added to
     ``crashed_cameras`` within seconds, not silently ignored.
  3. The stderr-tap monitor drains the child stderr so a chatty
     pipeline never deadlocks on a full pipe.
  4. ``PPHumanFrameStateAdapter.crashed_cameras`` is a UNION of
     the manager's crashed set and the tailer's own crashed set,
     so the benchmark sees subprocess crashes via the same
     accessor the tailer uses.

The tests are designed to run in CI without paddle installed —
they synthesize a fake ``pipeline.py`` script that exits non-zero
or writes MOT output as needed.
"""

from __future__ import annotations

import sys
import textwrap
import time
from pathlib import Path


from app.detection.pphuman_pipeline import (
    PPHumanDetectorAdapter,
    PPHumanFrameStateAdapter,
    PPHumanPipelineSubprocessManager,
)
from app.core.runtime_mode import RuntimeMode


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _write_fake_pipeline(
    path: Path,
    *,
    exit_code: int = 0,
    mot_lines: list[str] | None = None,
    sleep_seconds: float = 0.0,
) -> None:
    """Write a tiny stand-in for the official ``pipeline.py``.

    The stand-in honours the same CLI surface (``--config``,
    ``--video_file``, ``--output_dir``, ``--camera_id``) and
    either:
      * exits ``exit_code`` immediately (or after ``sleep_seconds``),
      * or writes ``mot_lines`` to
        ``{output_dir}/mot_results/{camera_id}.txt`` then exits 0.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mot_block = ""
    if mot_lines is not None:
        # emit a representative MOT file (frame,id,x,y,w,h,score,-1,-1)
        mot_block = textwrap.dedent(
            f"""
            import os, sys, time
            out_dir = sys.argv[sys.argv.index('--output_dir') + 1]
            cam_id  = sys.argv[sys.argv.index('--camera_id')  + 1]
            mot_dir = os.path.join(out_dir, 'mot_results')
            os.makedirs(mot_dir, exist_ok=True)
            with open(os.path.join(mot_dir, cam_id + '.txt'), 'w') as f:
                for line in {mot_lines!r}:
                    f.write(line + '\\n')
            """
        )
    script = textwrap.dedent(
        f"""
        #!/usr/bin/env python3
        {mot_block}
        if {sleep_seconds} > 0:
            import time
            time.sleep({sleep_seconds})
        sys.exit({exit_code})
        """
    )
    path.write_text(script)
    path.chmod(0o755)


def _make_adapter(pipeline_path: Path) -> PPHumanDetectorAdapter:
    """Construct a PPHumanDetectorAdapter pointed at a fake pipeline.

    We force ``mode=PRODUCTION`` so ``load()`` does not raise
    ProductionSafetyError; the adapter is in-memory only and
    never actually imports paddle here.
    """
    return PPHumanDetectorAdapter(
        pipeline_path=str(pipeline_path),
        config_path="/tmp/nonexistent_infer_cfg.yml",
        model_dir="/tmp/nonexistent_model_dir",
        device="cpu",
        run_mode="paddle",
        mode=RuntimeMode.PRODUCTION,
    )


# ----------------------------------------------------------------------------
# 1. Command uses sys.executable
# ----------------------------------------------------------------------------


def test_build_pipeline_command_uses_sys_executable(tmp_path: Path) -> None:
    """The command must use the *current* Python interpreter.

    Inside the production container ``/usr/bin/python`` may not have
    paddle installed; only the venv's ``sys.executable`` does. The
    previous hard-coded ``"python"`` silently resolved to a system
    interpreter that crashed on import.
    """
    pipeline = tmp_path / "pipeline.py"
    _write_fake_pipeline(pipeline, exit_code=0)
    adapter = _make_adapter(pipeline)
    cmd = adapter.build_pipeline_command(
        camera_id="CAM_TEST",
        video_file="/tmp/in.mp4",
        output_dir=str(tmp_path / "out"),
    )
    assert cmd[0] == sys.executable, (
        f"Expected sys.executable={sys.executable!r}, got {cmd[0]!r}. "
        "The PP-Human subprocess would inherit the wrong interpreter."
    )


# ----------------------------------------------------------------------------
# 1b. Command translates --camera_id to int (PATCH-051)
# ----------------------------------------------------------------------------


def test_build_pipeline_command_camera_id_is_int(tmp_path: Path) -> None:
    """The official pipeline.py declares ``--camera_id type=int``
    (see ``deploy/pipeline/cfg_utils.py`` line 80). Passing a
    string like ``"CAM_01"`` made argparse raise
    ``invalid int value: 'CAM_01'`` and the subprocess died.
    """
    pipeline = tmp_path / "pipeline.py"
    _write_fake_pipeline(pipeline, exit_code=0)
    adapter = _make_adapter(pipeline)
    cmd = adapter.build_pipeline_command(
        camera_id="CAM_TEST",
        video_file="/tmp/in.mp4",
        output_dir=str(tmp_path / "out"),
    )
    # --camera_id is the last arg-pair before the test ends.
    assert "--camera_id" in cmd, "command must include --camera_id"
    cam_idx = cmd.index("--camera_id")
    cam_value = cmd[cam_idx + 1]
    # Either the operator passed an int (rare) or we hashed it to int.
    assert cam_value.isdigit(), (
        f"--camera_id must be a string of digits (got {cam_value!r}); "
        "the official pipeline refuses non-int values."
    )
    # And two distinct cameras must map to distinct ints (no
    # accidental collision from the hash modulo 1000).
    cmd_a = adapter.build_pipeline_command(
        camera_id="CAM_01",
        video_file="/tmp/a.mp4",
        output_dir=str(tmp_path / "out_a"),
    )
    cmd_b = adapter.build_pipeline_command(
        camera_id="CAM_02",
        video_file="/tmp/b.mp4",
        output_dir=str(tmp_path / "out_b"),
    )
    assert cmd_a[cmd_a.index("--camera_id") + 1] != cmd_b[cmd_b.index("--camera_id") + 1], (
        "Two distinct cameras must hash to distinct --camera_id ints"
    )


# ----------------------------------------------------------------------------
# 2. Non-zero exit is surfaced as a crash
# ----------------------------------------------------------------------------


def test_subprocess_nonzero_exit_marks_camera_crashed(tmp_path: Path) -> None:
    """A subprocess that exits non-zero must appear in
    ``crashed_cameras`` within a few seconds.
    """
    pipeline = tmp_path / "pipeline.py"
    _write_fake_pipeline(pipeline, exit_code=2, sleep_seconds=0.0)
    adapter = _make_adapter(pipeline)
    out_root = tmp_path / "pphuman_out"
    mgr = PPHumanPipelineSubprocessManager(
        adapter,
        [("CAM_A", str(tmp_path / "in.mp4"))],
        output_root=str(out_root),
    )
    mgr.start()
    # Give the monitor thread a moment to notice the exit.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if mgr.crashed_cameras:
            break
        time.sleep(0.05)
    mgr.stop()
    assert "CAM_A" in mgr.crashed_cameras, (
        "A non-zero subprocess exit must be reported in "
        "crashed_cameras; otherwise the benchmark would silently "
        "treat the run as 'workers_crashed=false'."
    )


# ----------------------------------------------------------------------------
# 3. Frame-state adapter unions manager crashes
# ----------------------------------------------------------------------------


def test_frame_state_adapter_unions_manager_crashes(tmp_path: Path) -> None:
    """``PPHumanFrameStateAdapter.crashed_cameras`` must include
    crashes detected by the underlying ``PPHumanPipelineSubprocessManager``.

    The benchmark reads the adapter's set to set ``workers_crashed``;
    the previous implementation only saw tailer-thread crashes, not
    subprocess crashes. This test pins the union.
    """
    pipeline = tmp_path / "pipeline.py"
    _write_fake_pipeline(pipeline, exit_code=3, sleep_seconds=0.0)
    adapter = _make_adapter(pipeline)
    out_root = tmp_path / "pphuman_out"
    mgr = PPHumanPipelineSubprocessManager(
        adapter,
        [("CAM_B", str(tmp_path / "in.mp4"))],
        output_root=str(out_root),
    )
    state = PPHumanFrameStateAdapter(manager=mgr)
    state.start()
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if "CAM_B" in state.crashed_cameras:
            break
        time.sleep(0.05)
    crashed = state.crashed_cameras
    state.stop()
    mgr.stop()
    assert "CAM_B" in crashed, (
        "PPHumanFrameStateAdapter.crashed_cameras must include "
        "the manager's crashes; otherwise the benchmark cannot "
        "see subprocess failures."
    )


# ----------------------------------------------------------------------------
# 4. Stderr is drained (no deadlock)
# ----------------------------------------------------------------------------


def test_stderr_is_drained_no_deadlock(tmp_path: Path) -> None:
    """A chatty subprocess that emits > 64 KiB of stderr must not
    deadlock the parent. We write a 100 KiB line to stderr and
    assert the process exits and the monitor drains the line.
    """
    pipeline = tmp_path / "pipeline.py"
    # 100 KiB of stderr text; well over the default 64 KiB pipe buffer.
    big = "X" * 100_000
    script = (
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"sys.stderr.write({big!r}); sys.stderr.write('\\n'); sys.stderr.flush()\n"
        "sys.exit(0)\n"
    )
    pipeline.write_text(script)
    pipeline.chmod(0o755)
    adapter = _make_adapter(pipeline)
    out_root = tmp_path / "pphuman_out"
    mgr = PPHumanPipelineSubprocessManager(
        adapter,
        [("CAM_C", str(tmp_path / "in.mp4"))],
        output_root=str(out_root),
    )
    mgr.start()
    # Wait for the subprocess to exit.
    proc = mgr._procs["CAM_C"]
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    mgr.stop()
    # Process must have exited cleanly (rc=0), with the stderr
    # already drained (otherwise the child would still be alive
    # blocked on a full pipe).
    assert proc.poll() is not None, "Subprocess did not exit"
    assert proc.returncode == 0, f"Unexpected rc={proc.returncode}"
    # Stderr log captured at least one (possibly truncated) line.
    logs = mgr.stderr_logs.get("CAM_C", [])
    assert logs, "Monitor did not capture any stderr lines"


# ----------------------------------------------------------------------------
# 5. Successful subprocess is NOT marked crashed
# ----------------------------------------------------------------------------


def test_clean_exit_is_not_marked_crashed(tmp_path: Path) -> None:
    """A subprocess that exits 0 must not appear in crashed_cameras.

    This is the negative-side pin: a working PP-Human pipeline
    must not pollute the crashed set, otherwise the benchmark
    would refuse READY_FOR_LIMITED_PRODUCTION for a healthy run.
    """
    pipeline = tmp_path / "pipeline.py"
    mot_lines = ["1,1,10.0,20.0,100.0,200.0,0.91,-1,-1"]
    _write_fake_pipeline(pipeline, exit_code=0, mot_lines=mot_lines)
    adapter = _make_adapter(pipeline)
    out_root = tmp_path / "pphuman_out"
    mgr = PPHumanPipelineSubprocessManager(
        adapter,
        [("CAM_D", str(tmp_path / "in.mp4"))],
        output_root=str(out_root),
    )
    mgr.start()
    proc = mgr._procs["CAM_D"]
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    # Give the monitor a beat to record the exit code.
    time.sleep(0.2)
    mgr.stop()
    assert "CAM_D" not in mgr.crashed_cameras


# ----------------------------------------------------------------------------
# 6. Subprocess env overrides HOME / MPLCONFIGDIR
# ----------------------------------------------------------------------------


def test_subprocess_env_overrides_home(tmp_path: Path, monkeypatch) -> None:
    """The subprocess must see ``HOME`` redirected to a writable
    directory so PaddleDetection's ``expanduser("~/.cache/paddle")``
    doesn't resolve to a read-only /app/.cache.

    PATCH-051: the unprivileged ``app`` user cannot create
    ``/app/.cache``; without this override the subprocess exits
    with ``PermissionError: '/app/.cache'`` before producing any
    MOT output.
    """
    pipeline = tmp_path / "pipeline.py"
    script = (
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        f"open({str(tmp_path / 'home.log')!r}, 'w').write(os.environ.get('HOME', ''))"
    )
    pipeline.write_text(script)
    pipeline.chmod(0o755)
    adapter = _make_adapter(pipeline)
    out_root = tmp_path / "pphuman_out"
    mgr = PPHumanPipelineSubprocessManager(
        adapter,
        [("CAM_E", str(tmp_path / "in.mp4"))],
        output_root=str(out_root),
    )
    mgr.start()
    proc = mgr._procs["CAM_E"]
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    mgr.stop()
    home_log = tmp_path / "home.log"
    assert home_log.exists(), "Subprocess did not run"
    written = home_log.read_text().strip()
    # The subprocess HOME must be a writable scratch path, NOT
    # the production container's /app (which the app user cannot
    # write to).
    assert written and written != "/app", (
        f"Subprocess HOME is {written!r}; expected a writable "
        "scratch path under /tmp, not /app (unprivileged user)."
    )
    assert Path(written).is_dir() or written == "/tmp/pphuman_home", (
        f"Subprocess HOME {written!r} should exist as a directory"
    )


# ----------------------------------------------------------------------------
# 7. Model cache priming (avoid 30+ s redownload)
# ----------------------------------------------------------------------------


def test_model_cache_is_primed_with_local_model_dir(tmp_path: Path) -> None:
    """The subprocess must see the operator's baked model dir
    symlinked into PaddleDetection's ``~/.cache/paddle/infer_weights/``
    so it does NOT redownload 30+ MB of weights on every run.

    PATCH-051: PaddleDetection's ``auto_download_model`` only
    knows about its CDN URL. Operators who bake the model into
    the image (per the Dockerfile Layer 3) save 30+ s and avoid
    a hard internet dependency.

    The symlink target must be the SPECIFIC model subdir
    (e.g. ``mot_ppyoloe_l_36e_pipeline/``), not the parent —
    PaddleDetection's ``det_infer.py`` opens
    ``model_dir/infer_cfg.yml`` and only the subdir has that
    file.
    """
    pipeline = tmp_path / "pipeline.py"
    _write_fake_pipeline(pipeline, exit_code=0)
    adapter = _make_adapter(pipeline)
    # Set up a fake ``PPHUMAN_MODEL_DIR`` with the canonical
    # subdir layout (one subdir per PaddleDetection model).
    fake_model_dir = tmp_path / "fake_models"
    mot_subdir = fake_model_dir / "mot_ppyoloe_l_36e_pipeline"
    mot_subdir.mkdir(parents=True)
    (mot_subdir / "infer_cfg.yml").write_text("placeholder")
    adapter.model_dir = str(fake_model_dir)
    cache_root = tmp_path / "cache" / ".cache" / "paddle" / "infer_weights"
    adapter._prime_model_cache(
        cache_root=cache_root,
        model_dir=str(fake_model_dir),
    )
    # The symlink for the canonical PaddleDetection model name must exist.
    link = cache_root / "mot_ppyoloe_l_36e_pipeline"
    assert link.exists() or link.is_symlink(), (
        f"Expected symlink at {link!s}, found: {list(cache_root.iterdir()) if cache_root.exists() else 'cache_root does not exist'}"
    )
    # And the symlink must point to the SPECIFIC subdir, not the parent.
    assert link.resolve() == mot_subdir.resolve(), (
        f"Symlink {link!s} -> {link.resolve()!s} does not match "
        f"the subdir {mot_subdir.resolve()!s}. The previous version "
        "pointed at the parent; that triggered "
        "``FileNotFoundError: infer_cfg.yml`` in det_infer.py."
    )
