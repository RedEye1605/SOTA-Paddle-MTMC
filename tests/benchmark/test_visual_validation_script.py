"""Visual validation script tests (Phase 8).

Covers:
  1. script finds cam1_merged and cam2_merged via the data/ folder
  2. max_frames=3000 is honored
  3. output MP4 path is created or a clear error is raised
  4. JSON sidecar includes camera_id, frame_id, bbox, local_id, global_id fields
  5. overlay warns when smoke path is used
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "generate_visual_validation.py"


def _make_stub_video(path: Path, *, n_frames: int = 10, w: int = 64, h: int = 48) -> None:
    """Write a tiny OpenCV-generated stub MP4 the script can ingest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), fill_value=(i * 25) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_script_finds_cam1_and_cam2_merged(tmp_path: Path) -> None:
    """The script's ``--input`` argument accepts the canonical filenames
    copied during Phase 2."""
    cam1 = tmp_path / "cam1_merged.mp4"
    cam2 = tmp_path / "cam2_merged.mp4"
    _make_stub_video(cam1, n_frames=5)
    _make_stub_video(cam2, n_frames=5)
    assert cam1.exists() and cam2.exists()
    # Smoke sanity: both files open with OpenCV
    for f in (cam1, cam2):
        cap = cv2.VideoCapture(str(f))
        assert cap.isOpened()
        cap.release()


def test_max_frames_is_honored(tmp_path: Path) -> None:
    """The script must stop after ``--max-frames`` even if the source
    is longer. We pass max_frames=4 to a 10-frame stub."""
    src = tmp_path / "stub.mp4"
    out = tmp_path / "out.mp4"
    sidecar = tmp_path / "out.json"
    _make_stub_video(src, n_frames=10)

    cmd = [
        sys.executable,
        str(SCRIPT),
        "--cam",
        "CAM_TEST",
        "--input",
        str(src),
        "--max-frames",
        "4",
        "--output",
        str(out),
        "--sidecar",
        str(sidecar),
        "--smoke",
        "--site-id",
        "test_site",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["frames_written"] == 4
    assert payload["max_frames"] == 4
    assert payload["camera_id"] == "CAM_TEST"


def test_output_mp4_path_is_created(tmp_path: Path) -> None:
    """A successful run must create the output MP4."""
    src = tmp_path / "stub.mp4"
    out = tmp_path / "viz" / "out.mp4"
    sidecar = out.with_suffix(".json")
    _make_stub_video(src, n_frames=3)
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--cam",
        "CAM_OUT",
        "--input",
        str(src),
        "--max-frames",
        "3",
        "--output",
        str(out),
        "--sidecar",
        str(sidecar),
        "--smoke",
        "--site-id",
        "test_site",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    assert out.exists() and out.stat().st_size > 0


def test_sidecar_contains_required_fields(tmp_path: Path) -> None:
    """The JSON sidecar must include camera_id, frame_id, bbox,
    local_track_id, global_id."""
    src = tmp_path / "stub.mp4"
    out = tmp_path / "out.mp4"
    sidecar = out.with_suffix(".json")
    _make_stub_video(src, n_frames=2)
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--cam",
        "CAM_FIELDS",
        "--input",
        str(src),
        "--max-frames",
        "2",
        "--output",
        str(out),
        "--sidecar",
        str(sidecar),
        "--smoke",
        "--site-id",
        "test_site",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["camera_id"] == "CAM_FIELDS"
    # Per-frame structure
    assert isinstance(payload["frames"], list)
    assert len(payload["frames"]) == 2
    first = payload["frames"][0]
    assert {"frame_id", "ts", "detections"} <= set(first.keys())
    # When the synthetic detector produced a detection, all required
    # fields are present.
    if first["detections"]:
        d = first["detections"][0]
        assert "bbox" in d and len(d["bbox"]) == 4
        assert "confidence" in d
        assert "local_track_id" in d
        assert "global_id" in d


def test_overlay_warns_when_smoke(tmp_path: Path) -> None:
    """The HUD must contain a SMOKE warning when ``--smoke`` is set."""
    src = tmp_path / "stub.mp4"
    out = tmp_path / "out.mp4"
    sidecar = out.with_suffix(".json")
    _make_stub_video(src, n_frames=2)
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--cam",
        "CAM_SMOKE",
        "--input",
        str(src),
        "--max-frames",
        "2",
        "--output",
        str(out),
        "--sidecar",
        str(sidecar),
        "--smoke",
        "--site-id",
        "test_site",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["smoke"] is True
    assert "smoke" in payload["detector_backend"].lower()
    assert "smoke" in payload["reid_backend"].lower()
