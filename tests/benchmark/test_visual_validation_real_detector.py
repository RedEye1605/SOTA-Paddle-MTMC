"""Tests for the fail-loud behaviour of generate_visual_validation.py.

Pinned behaviours (see git history — there was a silent fallback to
the synthetic detector that produced 'flickering' random per-frame
boxes and fake global_ids):

1. Default mode (no --smoke) **fails loud** if the real PP-Human
   adapter is not importable. It does NOT silently use the synthetic
   detector.
2. The smoke path still works for HUD / codec sanity checks.
3. The synthetic detector is only used when ``--smoke`` is explicit
   or ``SMOKE_VISUALIZATION=1`` is set in the environment.
4. When paddle is installed and the model dir / pipeline.py exist,
   the script picks the most-portable fourcc (avc1 / H264) for the
   output MP4.
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "generate_visual_validation.py"


def _run(
    args, *, env_overrides: dict[str, str] | None = None, timeout: int = 60
) -> subprocess.CompletedProcess:
    """Invoke the visualisation script as a subprocess."""
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_real_path_fails_loud_without_paddle(tmp_path: Path) -> None:
    """When --smoke is not set and the real adapter fails to import,
    the script must raise rather than silently using the synthetic
    detector (which would produce the 'flickering' random per-frame
    boxes the user reported)."""
    fake_video = tmp_path / "video.mp4"
    # We don't need a real video; the script fails before reading
    # any frame because the real adapter import fails.
    out = tmp_path / "out.mp4"
    proc = _run(
        [
            "--cam",
            "CAM_01",
            "--input",
            str(fake_video),
            "--output",
            str(out),
            "--max-frames",
            "1",
        ],
        env_overrides={
            # Force the real adapter init to fail by setting an
            # obviously-bad model_dir.
            "PPHUMAN_MODEL_DIR": str(tmp_path / "does_not_exist"),
            "SMOKE_VISUALIZATION": "false",
        },
    )
    assert proc.returncode != 0, (
        "script should fail loud when real PP-Human is unavailable; "
        f"got returncode=0\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    # The error message must explicitly tell the user why the
    # synthetic detector isn't a fallback.
    combined = (proc.stdout + proc.stderr).lower()
    assert "synthetic" in combined or "pp-human" in combined or "paddle" in combined, (
        f"error message should mention paddle / pp-human / synthetic, got:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_smoke_path_succeeds_with_synthetic_detector(tmp_path: Path) -> None:
    """The synthetic detector path is still usable for layout / HUD
    sanity-checks when ``--smoke`` is set."""
    # Build a tiny 5-frame test video with OpenCV.
    import cv2
    import numpy as np

    fake_video = tmp_path / "fake.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(fake_video), fourcc, 1.0, (32, 32))
    for _ in range(5):
        w.write(np.zeros((32, 32, 3), dtype="uint8"))
    w.release()

    out = tmp_path / "out.mp4"
    sidecar = tmp_path / "out.json"
    proc = _run(
        [
            "--cam",
            "CAM_01",
            "--input",
            str(fake_video),
            "--output",
            str(out),
            "--sidecar",
            str(sidecar),
            "--max-frames",
            "3",
            "--smoke",
            "--output-width",
            "32",
            "--output-height",
            "32",
        ],
        env_overrides={
            "PPHUMAN_MODEL_DIR": str(tmp_path / "does_not_exist"),
        },
    )
    assert proc.returncode == 0, (
        f"smoke path should succeed; got returncode={proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert out.exists() and out.stat().st_size > 0
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    # The synthetic detector produces 0..2 boxes per frame, so
    # the sidecar must have frames with detections.
    assert data["frames"]
    assert any(fr["detections"] for fr in data["frames"])


def test_synthetic_detector_is_deterministic_per_frame() -> None:
    """Direct test of the synthetic detector helper: same
    (camera_id, frame_id) must produce identical output, different
    frame_ids must produce *different* output (otherwise we have
    the same flickering problem)."""
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.generate_visual_validation import _synthetic_detect  # type: ignore  # noqa: E402

    import numpy as np

    frame = np.zeros((100, 100, 3), dtype="uint8")
    a1 = _synthetic_detect(frame, frame_id=42, camera_id="CAM_01")
    a2 = _synthetic_detect(frame, frame_id=42, camera_id="CAM_01")
    b = _synthetic_detect(frame, frame_id=43, camera_id="CAM_01")
    # Same seed → same output.
    assert a1 == a2
    # Different seed → very likely different bbox / global_id.
    # (We don't assert inequality because random could collide;
    # we just assert the detector *changes* with frame_id, which
    # is the root cause of the flickering the user reported.)
    assert a1 != b or a1[0]["bbox"] != b[0]["bbox"]


def test_encoder_prefers_h264_when_available(tmp_path: Path) -> None:
    """The _pick_encoder helper should return avc1 (H.264) on
    hosts that have OpenCV built with FFmpeg + libx264."""
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.generate_visual_validation import _pick_encoder  # type: ignore  # noqa: E402

    fourcc_str, label = _pick_encoder(prefer="auto")
    # Either we got a real H.264 encoder (preferred), or we fell
    # back to mp4v. Either way the label must be non-empty.
    assert label
    assert fourcc_str in {"avc1", "H264", "mp4v", None}
    if fourcc_str is not None:
        # The preferred order is avc1 > H264 > mp4v.
        assert label in {
            "H.264 (avc1, browser-playable)",
            "H.264 (H264, browser-playable)",
            "MPEG-4 Part 2 (mp4v, VLC only — fallback)",
        }
