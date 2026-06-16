"""Annotated-frame overlay for the visual validation pipeline (Phase 8/9).

Pure drawing helpers — stateless. Adapted from
``Service/offline-people-counting/app/engine/overlay.py``. The
overlay draws:

* A box per detection with the detector class + confidence.
* The local track id, the assigned global_id, and the ReID
  similarity (when available).
* A top-left HUD with camera id, frame number, wall-clock time,
  detector / ReID backends, and a ``smoke`` warning when the
  pipeline ran in smoke mode.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Color palette (BGR — OpenCV convention)
# ---------------------------------------------------------------------------
PERSON_COLOR = (56, 229, 77)
SMOKE_COLOR = (80, 200, 255)
WARN_COLOR = (50, 50, 255)
TEXT_COLOR = (245, 250, 255)
HUD_BG_COLOR = (8, 16, 24)

THICKNESS_DIVISOR = 3600
CORNER_MIN = 6
CORNER_MAX = 14
CORNER_RATIO = 0.06
PAD_X = 6
PAD_Y = 4
LABEL_BG_ALPHA = 0.70
FONT_SCALE_DIVISOR = 2400


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip_bbox(frame: np.ndarray, bbox: Sequence[float]) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(x1, 0, max(w - 1, 0)))
    y1 = int(np.clip(y1, 0, max(h - 1, 0)))
    x2 = int(np.clip(x2, 0, max(w - 1, 0)))
    y2 = int(np.clip(y2, 0, max(h - 1, 0)))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def _draw_corner_box(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    thickness: int,
    corner: int,
) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), HUD_BG_COLOR, max(1, thickness // 2))
    pts = [
        ((x1, y1), (x1 + corner, y1)),
        ((x1, y1), (x1, y1 + corner)),
        ((x2, y1), (x2 - corner, y1)),
        ((x2, y1), (x2, y1 + corner)),
        ((x1, y2), (x1 + corner, y2)),
        ((x1, y2), (x1, y2 - corner)),
        ((x2, y2), (x2 - corner, y2)),
        ((x2, y2), (x2, y2 - corner)),
    ]
    for start, end in pts:
        cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)


def _draw_label(
    frame: np.ndarray,
    *,
    x: int,
    y: int,
    label: str,
    color: tuple[int, int, int],
) -> None:
    if not label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(frame.shape[1], frame.shape[0]) / FONT_SCALE_DIVISOR)
    text_thickness = max(1, round(font_scale * 2))
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    pad_x, pad_y = PAD_X, PAD_Y
    label_h = text_h + baseline + pad_y * 2
    top = max(0, y - label_h - 4)
    bottom = top + label_h
    right = min(frame.shape[1] - 1, x + text_w + pad_x * 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, top), (right, bottom), HUD_BG_COLOR, -1)
    cv2.addWeighted(overlay, LABEL_BG_ALPHA, frame, 1 - LABEL_BG_ALPHA, 0, frame)
    cv2.rectangle(frame, (x, top), (right, bottom), color, 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (x + pad_x, bottom - pad_y - baseline),
        font,
        font_scale,
        TEXT_COLOR,
        text_thickness,
        cv2.LINE_AA,
    )


def _build_detection_label(det: dict[str, Any]) -> str:
    parts: list[str] = []
    cls = det.get("class_name") or "person"
    parts.append(str(cls).title())
    conf = det.get("confidence")
    if conf is not None:
        parts.append(f"{_safe_float(conf):.2f}")
    if det.get("local_track_id") is not None:
        parts.append(f"L{det.get('local_track_id')}")
    gid = det.get("global_id")
    if gid:
        parts.append(f"G:{gid}")
    sim = det.get("reid_similarity")
    if sim is not None:
        parts.append(f"R{_safe_float(sim):.2f}")
    zone = det.get("zone_id")
    if zone:
        parts.append(f"Z:{zone}")
    return " ".join(parts)


def draw_detections(frame: np.ndarray, detections: list[dict[str, Any]]) -> None:
    """Draw every detection on *frame* in place."""
    h, w = frame.shape[:2]
    for det in detections:
        bbox = det.get("bbox")
        # Strict guard: bbox must be a 4-element list/tuple of numbers.
        # Anything else (None, str, wrong length) is silently dropped
        # to keep the rendering pipeline robust against malformed
        # payloads.
        if (
            not isinstance(bbox, (list, tuple))
            or len(bbox) != 4
            or not all(isinstance(v, (int, float)) and v == v for v in bbox)
        ):
            continue
        x1, y1, x2, y2 = _clip_bbox(frame, bbox)
        thickness = max(1, round(min(h, w) / THICKNESS_DIVISOR))
        corner = max(CORNER_MIN, min(CORNER_MAX, int((x2 - x1) * CORNER_RATIO)))
        _draw_corner_box(frame, x1, y1, x2, y2, PERSON_COLOR, thickness, corner)
        _draw_label(frame, x=x1, y=y1, label=_build_detection_label(det), color=PERSON_COLOR)


def draw_hud(
    frame: np.ndarray,
    *,
    camera_id: str,
    frame_id: int,
    fps: float,
    detector_backend: str,
    reid_backend: str,
    smoke: bool = False,
    site_id: str = "default_site",
    timestamp: Optional[float] = None,
) -> None:
    """Top-left status panel.

    Includes the camera id, frame number, FPS, detector/ReID
    backends, the site id, and a wall-clock timestamp. A `SMOKE`
    warning is added when ``smoke=True``.
    """
    h, w = frame.shape[:2]
    lines = [
        f"Cam: {camera_id}",
        f"Frame: {frame_id}",
        f"FPS: {fps:.1f}",
        f"Detector: {detector_backend}",
        f"ReID: {reid_backend}",
        f"Site: {site_id}",
    ]
    if smoke:
        lines.append("WARNING: SMOKE-TEST BACKEND")
    if timestamp is None:
        timestamp = time.time()
    ts_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(f"Time: {ts_str}")

    n_lines = len(lines)
    hud_height = 20 + n_lines * 22 + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (16, 16), (520, 16 + hud_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    y_pos = 42
    for i, line in enumerate(lines):
        color = WARN_COLOR if (smoke and "SMOKE" in line) else TEXT_COLOR
        cv2.putText(
            frame,
            line,
            (28, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        y_pos += 22


def annotate_frame(
    frame: np.ndarray,
    *,
    camera_id: str,
    frame_id: int,
    fps: float,
    detector_backend: str,
    reid_backend: str,
    detections: Optional[list[dict[str, Any]]] = None,
    smoke: bool = False,
    site_id: str = "default_site",
    timestamp: Optional[float] = None,
) -> np.ndarray:
    """Return a new annotated copy of *frame* (caller's frame is untouched)."""
    out = frame.copy()
    if detections:
        draw_detections(out, detections)
    draw_hud(
        out,
        camera_id=camera_id,
        frame_id=frame_id,
        fps=fps,
        detector_backend=detector_backend,
        reid_backend=reid_backend,
        smoke=smoke,
        site_id=site_id,
        timestamp=timestamp,
    )
    return out
