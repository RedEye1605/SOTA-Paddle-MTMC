"""Cropping and embedding-normalization helpers."""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np


def crop_with_padding(
    frame: np.ndarray,
    bbox: Tuple[float, float, float, float],
    pad_ratio: float = 0.10,
) -> np.ndarray:
    """Crop a person from `frame` with a small `pad_ratio` margin.

    ReID models (PP-Human, TransReID) expect ~10% padding around the bounding
    box to capture head/feet. Returns BGR crop.
    """
    if frame is None or frame.size == 0:
        raise ValueError("empty frame")
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px = bw * pad_ratio
    py = bh * pad_ratio
    x1p = int(max(0, math.floor(x1 - px)))
    y1p = int(max(0, math.floor(y1 - py)))
    x2p = int(min(w, math.ceil(x2 + px)))
    y2p = int(min(h, math.ceil(y2 + py)))
    if x2p <= x1p or y2p <= y1p:
        # bbox invalid; return a tiny placeholder so the caller can score 0
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame[y1p:y2p, x1p:x2p].copy()


def resize_keep_aspect(
    image: np.ndarray,
    target_size: Tuple[int, int],  # (W, H)
) -> np.ndarray:
    """Resize so the longer side fits `target_size`, then pad with zeros."""
    if image is None or image.size == 0:
        raise ValueError("empty image")
    tw, th = target_size
    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((th, tw, 3), dtype=image.dtype)
    x_off = (tw - new_w) // 2
    y_off = (th - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a 1D or 2D vector along the last axis."""
    v = np.asarray(vec, dtype=np.float32)
    if v.ndim == 1:
        n = float(np.linalg.norm(v))
        if n < eps:
            return v
        return v / n
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (v / norms).astype(np.float32)


def mean_normalized(vectors: np.ndarray) -> np.ndarray:
    """Mean of L2-normalized vectors, then re-normalized."""
    if vectors is None or len(vectors) == 0:
        raise ValueError("empty vectors")
    normed = np.stack([l2_normalize(v) for v in vectors], axis=0)
    mean = normed.mean(axis=0)
    return l2_normalize(mean)
