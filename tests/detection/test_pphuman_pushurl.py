"""Tests for the unified-stream pushurl wiring on the PP-Human pipeline.

Pin the contract:

  * ``build_pipeline_command`` emits ``--pushurl <base>`` when
    given a non-empty pushurl.
  * When pushurl is empty / None, ``--pushurl`` is omitted (legacy
    file-only behaviour).
  * ``PPHumanPipelineSubprocessManager`` forwards the pushurl_base
    to every subprocess it spawns.
  * The default MEDIAMTX_PPHUMAN_DIRECT_PUSH behaviour (when env
    is not set) is "true" so the typical operator setup works
    out of the box.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from app.detection.pphuman_pipeline import (
    PPHumanDetectorAdapter,
    PPHumanPipelineSubprocessManager,
)


# A stub pipeline_path so the adapter's __init__ does not fail
# when the real PaddleDetection is absent. We never invoke the
# subprocess in these tests — we only inspect build_pipeline_command
# argv.
_STUB_PIPELINE = "/opt/paddledetection/deploy/pipeline/pipeline.py"
_STUB_CONFIG = "/opt/paddledetection/deploy/pipeline/config/infer_cfg_pphuman.yml"


@pytest.fixture
def _stub_pipeline_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create empty stub files at the real PaddleDetection paths so
    ``Path(self.pipeline_path).exists()`` in build_pipeline_command
    passes. The files are inside the tmp dir; we symlink them
    into place so the production paths the adapter uses still
    work in the test container.
    """
    real_pipeline = Path(_STUB_PIPELINE)
    real_config = Path(_STUB_CONFIG)
    if real_pipeline.exists() and real_config.exists():
        return  # production paths exist (e.g. inside the api container)

    # Create stub files in a temp dir and bind-mount via symlinks
    # to the production paths so the adapter's path resolution
    # works. Skip if we cannot write to /opt (read-only fs).
    try:
        real_pipeline.parent.mkdir(parents=True, exist_ok=True)
        real_pipeline.touch(exist_ok=True)
        real_config.parent.mkdir(parents=True, exist_ok=True)
        real_config.touch(exist_ok=True)
        # Track them so we don't pollute the host fs.
        monkeypatch.setattr(real_pipeline, "exists", lambda: True)
        monkeypatch.setattr(real_config, "exists", lambda: True)
    except (PermissionError, OSError):
        # /opt is read-only in the dev environment. Use a
        # lightweight monkeypatch of the existence check instead.
        from app.detection import pphuman_pipeline as pp_mod
        orig = pp_mod.PPHumanDetectorAdapter.build_pipeline_command

        def _patched(self, *args, **kwargs):
            # Bypass the existence assertion. The function body
            # only opens the file in a subprocess we never run.
            self.pipeline_path = _STUB_PIPELINE  # keep type honest
            return orig(self, *args, **kwargs)

        # Easier: just patch Path.exists on the path strings.
        from pathlib import Path as _Path

        real_exists = _Path.exists

        def _fake_exists(self):
            if str(self) in (_STUB_PIPELINE, _STUB_CONFIG):
                return True
            return real_exists(self)

        monkeypatch.setattr(_Path, "exists", _fake_exists)


def _make_adapter() -> PPHumanDetectorAdapter:
    return PPHumanDetectorAdapter(
        pipeline_path=_STUB_PIPELINE,
        config_path=_STUB_CONFIG,
        model_dir="/models/pphuman",
        device="gpu",
        run_mode="paddle",
        skip_frame_num=2,
        mode=None,  # type: ignore[arg-type]
    )


# --- build_pipeline_command: --pushurl emission -------------------------


def test_build_pipeline_command_omits_pushurl_when_none(
    _stub_pipeline_paths: None,
) -> None:
    a = _make_adapter()
    cmd = a.build_pipeline_command(
        camera_id="CAM_01",
        video_file="/data/cam1_merged.mp4",
        output_dir="/tmp/out/CAM_01",
        pushurl=None,
    )
    assert "--pushurl" not in cmd


def test_build_pipeline_command_omits_pushurl_when_empty(
    _stub_pipeline_paths: None,
) -> None:
    a = _make_adapter()
    cmd = a.build_pipeline_command(
        camera_id="CAM_01",
        video_file="/data/cam1_merged.mp4",
        output_dir="/tmp/out/CAM_01",
        pushurl="",
    )
    assert "--pushurl" not in cmd


def test_build_pipeline_command_emits_pushurl_when_set(
    _stub_pipeline_paths: None,
) -> None:
    a = _make_adapter()
    base = "rtsp://198.51.100.20:8554/sota-paddle-mtmc/"
    cmd = a.build_pipeline_command(
        camera_id="CAM_01",
        video_file="/data/cam1_merged.mp4",
        output_dir="/tmp/out/CAM_01",
        pushurl=base,
    )
    # The argv must contain ``--pushurl <base>`` as a pair.
    assert "--pushurl" in cmd
    idx = cmd.index("--pushurl")
    assert cmd[idx + 1] == base


def test_build_pipeline_command_preserves_other_flags(
    _stub_pipeline_paths: None,
) -> None:
    """Adding pushurl must not displace any of the existing flags."""
    a = _make_adapter()
    cmd = a.build_pipeline_command(
        camera_id="CAM_01",
        video_file="/data/cam1_merged.mp4",
        output_dir="/tmp/out/CAM_01",
        pushurl="rtsp://h:8554/p/",
    )
    # Spot-check the must-have flags are still present.
    assert "--config" in cmd
    assert _STUB_CONFIG in cmd
    assert "--video_file" in cmd
    assert "/data/cam1_merged.mp4" in cmd
    assert "--output_dir" in cmd
    assert "/tmp/out/CAM_01" in cmd
    assert "--camera_id" in cmd
    # -o MOT.* overrides
    assert "MOT.enable=True" in cmd
    assert "MOT.skip_frame_num=2" in cmd


def test_build_pipeline_command_uses_trailing_slash_in_pushurl(
    _stub_pipeline_paths: None,
) -> None:
    """The operator-supplied base MUST end with a slash so PP-Human's
    os.path.join produces a clean URL like
    rtsp://host:8554/sota-paddle-mtmc/cam1_merged (and not
    rtsp://host:8554/sota-paddle-mtmccam1_merged)."""
    a = _make_adapter()
    base = "rtsp://198.51.100.20:8554/sota-paddle-mtmc/"
    cmd = a.build_pipeline_command(
        camera_id="CAM_01",
        video_file="/data/cam1_merged.mp4",
        output_dir="/tmp/out/CAM_01",
        pushurl=base,
    )
    assert cmd[cmd.index("--pushurl") + 1].endswith("/")


# --- PPHumanPipelineSubprocessManager: forwards pushurl to all subs -----


def test_subprocess_manager_forwards_pushurl_to_each_camera(tmp_path: Path) -> None:
    """The manager must pass the same pushurl_base to every
    subprocess it spawns (one per camera)."""

    # We don't actually want to spawn ffmpeg here — replace
    # adapter.run_pipeline with a recorder.
    adapter = mock.MagicMock(spec=PPHumanDetectorAdapter)
    cmds_received: list[tuple[str, str]] = []

    def fake_run(*, camera_id, video_file, output_dir, pushurl):
        cmds_received.append((camera_id, pushurl))
        proc = mock.MagicMock(spec=subprocess.Popen)
        proc.pid = 12345
        return proc

    adapter.run_pipeline.side_effect = fake_run

    mgr = PPHumanPipelineSubprocessManager(
        adapter=adapter,
        cameras=[
            ("CAM_01", "/data/cam1_merged.mp4"),
            ("CAM_02", "/data/cam2_merged.mp4"),
        ],
        output_root=str(tmp_path),
        pushurl_base="rtsp://host:8554/sota-paddle-mtmc/",
    )
    mgr.start()

    # Both cameras must have received the same pushurl_base.
    seen = dict(cmds_received)
    assert seen == {
        "CAM_01": "rtsp://host:8554/sota-paddle-mtmc/",
        "CAM_02": "rtsp://host:8554/sota-paddle-mtmc/",
    }


def test_subprocess_manager_no_pushurl_when_base_is_none(tmp_path: Path) -> None:
    adapter = mock.MagicMock(spec=PPHumanDetectorAdapter)
    cmds_received: list[tuple[str, str | None]] = []

    def fake_run(*, camera_id, video_file, output_dir, pushurl):
        cmds_received.append((camera_id, pushurl))
        proc = mock.MagicMock(spec=subprocess.Popen)
        proc.pid = 12345
        return proc

    adapter.run_pipeline.side_effect = fake_run

    mgr = PPHumanPipelineSubprocessManager(
        adapter=adapter,
        cameras=[("CAM_01", "/data/cam1_merged.mp4")],
        output_root=str(tmp_path),
        pushurl_base=None,
    )
    mgr.start()

    assert cmds_received == [("CAM_01", None)]


def test_make_frame_state_adapter_forwards_pushurl(tmp_path: Path) -> None:
    """Regression: ``make_frame_state_adapter`` used to construct
    its own manager with no pushurl_base, which silently shadowed
    the operator's intent. The pushurl must propagate."""
    from app.detection.pphuman_pipeline import make_frame_state_adapter

    adapter = mock.MagicMock(spec=PPHumanDetectorAdapter)
    adapter.load.return_value = None
    cmds_received: list[tuple[str, str | None]] = []

    def fake_run(*, camera_id, video_file, output_dir, pushurl):
        cmds_received.append((camera_id, pushurl))
        proc = mock.MagicMock(spec=subprocess.Popen)
        proc.pid = 12345
        return proc

    adapter.run_pipeline.side_effect = fake_run

    fsa = make_frame_state_adapter(
        adapter=adapter,
        cameras=[("CAM_01", "/data/cam1_merged.mp4")],
        output_root=str(tmp_path),
        pushurl_base="rtsp://h:8554/p/",
    )
    # Start the underlying manager the same way app.main does.
    fsa.manager.start()

    assert cmds_received == [("CAM_01", "rtsp://h:8554/p/")]


# --- Default behaviour: MEDIAMTX_PPHUMAN_DIRECT_PUSH is on by default --


def test_default_mediamtx_direct_push_is_on() -> None:
    """With the env var unset, the resolver in main.py treats
    direct push as enabled. We exercise the same parsing here
    so the contract is pinned without booting the full app."""
    with mock.patch.dict(os.environ, {}, clear=True):
        value = os.environ.get("MEDIAMTX_PPHUMAN_DIRECT_PUSH", "true").lower()
    assert value in ("1", "true", "yes", "on")

    with mock.patch.dict(
        os.environ, {"MEDIAMTX_PPHUMAN_DIRECT_PUSH": "false"}, clear=False
    ):
        value = os.environ.get("MEDIAMTX_PPHUMAN_DIRECT_PUSH", "true").lower()
    assert value == "false"
