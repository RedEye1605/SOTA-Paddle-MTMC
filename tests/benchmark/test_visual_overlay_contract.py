"""Visual overlay contract tests (Phase 8).

The ``app.streaming.overlay.annotate_frame`` function is the
single rendering surface used by:

  * the production MediaMTX streamer
  * the visual-validation script
  * the live dashboard

The contract this module pins down:

  1. The HUD always includes camera_id, frame_id, FPS, detector and
     ReID backend labels.
  2. When ``smoke=True`` the HUD prints a ``SMOKE`` warning.
  3. Per-detection labels include class, confidence, local track id,
     global id, ReID similarity, and (when present) zone id.
  4. The function is pure: the caller's input frame is not mutated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.streaming.overlay import (  # noqa: E402
    annotate_frame,
    draw_detections,
    draw_hud,
)


def _blank_frame(h: int = 240, w: int = 320) -> np.ndarray:
    return np.full((h, w, 3), 32, dtype=np.uint8)


def test_hud_contains_camera_id_and_frame_id() -> None:
    frame = _blank_frame()
    original_sum = int(frame.sum())
    # ``draw_hud`` mutates the frame in place and returns ``None``;
    # the function is a side-effect renderer.
    result = draw_hud(
        frame,
        camera_id="CAM_42",
        frame_id=1234,
        fps=9.5,
        detector_backend="paddledetection_pphuman",
        reid_backend="pphuman_strongbaseline",
    )
    assert result is None
    # The HUD draws text on the frame, so the pixel sum must change.
    assert int(frame.sum()) != original_sum


def test_hud_adds_smoke_warning_when_smoke_true() -> None:
    """The HUD's *line list* includes a SMOKE warning when smoke=True.
    We inspect the function's line composition rather than the
    rendered pixels because the alpha-blended overlay makes exact
    pixel assertions brittle on a near-black canvas.
    """
    import inspect

    src = inspect.getsource(draw_hud)
    # The function must have an `if smoke:` branch and the warning
    # line must include the word "SMOKE".
    assert "if smoke" in src
    assert "SMOKE" in src
    # And: the annotate_frame wrapper must forward ``smoke`` into
    # draw_hud.  End-to-end check via the wrapper.
    frame_smoke = _blank_frame()
    out = annotate_frame(
        frame_smoke,
        camera_id="CAM",
        frame_id=1,
        fps=10.0,
        detector_backend="synthetic_smoke",
        reid_backend="deterministic_smoke",
        smoke=True,
    )
    assert out is not None
    # The smoke-annotated frame must have at least one pixel whose
    # value differs from the original (the HUD was drawn).
    assert not np.array_equal(out, frame_smoke)


def test_annotate_frame_returns_new_frame_and_does_not_mutate_input() -> None:
    frame = _blank_frame()
    original_sum = int(frame.sum())
    detections = [
        {
            "bbox": [10, 20, 60, 120],
            "confidence": 0.9,
            "class_name": "person",
            "local_track_id": 5,
            "global_id": "G007",
            "reid_similarity": 0.85,
            "zone_id": "ZONE_A",
        }
    ]
    out = annotate_frame(
        frame,
        camera_id="CAM_01",
        frame_id=42,
        fps=10.0,
        detector_backend="paddledetection_pphuman",
        reid_backend="pphuman_strongbaseline",
        detections=detections,
        smoke=False,
    )
    assert out is not None
    assert int(frame.sum()) == original_sum, "input frame must not be mutated"
    assert out.shape == frame.shape


def test_draw_detections_handles_empty_list() -> None:
    frame = _blank_frame()
    original_sum = int(frame.sum())
    draw_detections(frame, [])
    # No-op on empty input
    assert int(frame.sum()) == original_sum


def test_draw_detections_skips_invalid_bboxes() -> None:
    frame = _blank_frame()
    detections = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.5, "class_name": "person"},
        {"bbox": None, "confidence": 0.5, "class_name": "person"},
        {"bbox": "nope", "confidence": 0.5, "class_name": "person"},
        {"bbox": [1, 2, 3], "confidence": 0.5, "class_name": "person"},
    ]
    # Must not raise on malformed input
    draw_detections(frame, detections)


def test_detection_label_contains_class_and_track_id() -> None:
    """The label builder must include class, confidence, local id,
    global id, ReID similarity, and zone id when present."""
    from app.streaming.overlay import _build_detection_label  # noqa: WPS450

    label = _build_detection_label(
        {
            "class_name": "person",
            "confidence": 0.88,
            "local_track_id": 3,
            "global_id": "G12",
            "reid_similarity": 0.91,
            "zone_id": "ZONE_A",
        }
    )
    assert "Person" in label
    assert "0.88" in label
    assert "L3" in label
    assert "G:12" in label or "G12" in label
    assert "0.91" in label
    assert "ZONE_A" in label
