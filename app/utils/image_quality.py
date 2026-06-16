"""Image-quality scoring for crop filtering.

ReID quality is a combination of:

- bbox height (small = far/occluded = poor ReID signal)
- blur (Laplacian variance)
- brightness (too dark or too bright = poor)
- occlusion (heuristic: bbox aspect ratio + center)
- frame-boundary cut (bbox touches image edge)

Score range: 0.0 (rejected) to 1.0 (perfect).
"""

from __future__ import annotations


import cv2
import numpy as np


def bbox_height(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(0.0, y2 - y1))


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def laplacian_variance(gray: np.ndarray) -> float:
    """Higher = sharper. 0 = uniform; <50 = very blurry."""
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray: np.ndarray) -> float:
    """Mean pixel intensity in [0, 255]. Target 50..200."""
    if gray.size == 0:
        return 0.0
    return float(gray.mean())


def is_cut_by_frame(
    bbox: tuple[float, float, float, float],
    frame_shape: tuple[int, int, int],
    margin_px: float = 2.0,
) -> bool:
    """True if the bbox touches the frame edge within `margin_px`."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    return x1 <= margin_px or y1 <= margin_px or x2 >= w - margin_px or y2 >= h - margin_px


def crop_quality_score(
    crop_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    min_height_px: float = 60.0,
) -> float:
    """Combine multiple quality signals into [0, 1].

    Returns 0.0 if the crop should be rejected (too small, blurry, very dark
    or very bright, or cut by frame).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0

    h = bbox_height(bbox)
    if h < min_height_px:
        return 0.0

    # height_score: 1.0 at h >= 4*min_height, 0.5 at h == min_height
    height_score = float(np.clip((h - min_height_px) / (3.0 * min_height_px) + 0.5, 0.0, 1.0))

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    lv = laplacian_variance(gray)
    # Sharpness: <50 reject, 50..150 ramp, >150 good
    sharp_score = float(np.clip((lv - 50.0) / 100.0, 0.0, 1.0))

    b = brightness(gray)
    # Brightness: 50..200 OK, <30 or >235 reject
    if b < 30.0 or b > 235.0:
        bright_score = 0.0
    else:
        # Triangle peaked at 128
        bright_score = float(1.0 - abs(b - 128.0) / 128.0)

    if is_cut_by_frame(bbox, crop_bgr.shape):
        cut_score = 0.0
    else:
        cut_score = 1.0

    # Weighted average; reject if any component is 0
    if 0.0 in (sharp_score, bright_score, cut_score):
        return 0.0

    score = 0.40 * height_score + 0.30 * sharp_score + 0.20 * bright_score + 0.10 * cut_score
    return float(np.clip(score, 0.0, 1.0))


def is_acceptable(score: float, threshold: float = 0.5) -> bool:
    return score >= threshold
