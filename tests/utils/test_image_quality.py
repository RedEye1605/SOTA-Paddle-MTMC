"""Image quality — crop filter."""

from __future__ import annotations

import numpy as np

from app.utils.crop import crop_with_padding, l2_normalize, mean_normalized, resize_keep_aspect
from app.utils.image_quality import crop_quality_score, is_acceptable


def test_crop_quality_too_small_rejected() -> None:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    q = crop_quality_score(img, bbox=(100, 100, 130, 130), min_height_px=60)
    assert q == 0.0


def test_crop_quality_normal_image_has_positive_score() -> None:
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    # Add some high-frequency content for sharpness
    rng = np.random.default_rng(42)
    img = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    q = crop_quality_score(img, bbox=(100, 100, 300, 400), min_height_px=60)
    assert 0.0 < q <= 1.0


def test_crop_quality_too_dark_rejected() -> None:
    img = np.full((480, 640, 3), 5, dtype=np.uint8)  # almost black
    q = crop_quality_score(img, bbox=(100, 100, 300, 400), min_height_px=60)
    assert q == 0.0


def test_crop_quality_too_bright_rejected() -> None:
    img = np.full((480, 640, 3), 250, dtype=np.uint8)  # blown out
    q = crop_quality_score(img, bbox=(100, 100, 300, 400), min_height_px=60)
    assert q == 0.0


def test_is_acceptable() -> None:
    assert is_acceptable(0.7) is True
    assert is_acceptable(0.4) is False


def test_l2_normalize_1d() -> None:
    v = np.array([3.0, 4.0], dtype=np.float32)
    n = l2_normalize(v)
    assert float(np.linalg.norm(n)) == 1.0
    assert abs(n[0] - 0.6) < 1e-6


def test_l2_normalize_2d() -> None:
    m = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    n = l2_normalize(m)
    assert float(np.linalg.norm(n[0])) == 1.0
    assert float(np.linalg.norm(n[1])) == 0.0


def test_mean_normalized() -> None:
    rng = np.random.default_rng(0)
    v1 = rng.standard_normal(128).astype(np.float32)
    v2 = rng.standard_normal(128).astype(np.float32)
    m = mean_normalized([v1, v2])
    assert m.shape == (128,)
    assert abs(float(np.linalg.norm(m)) - 1.0) < 1e-5


def test_crop_with_padding_adds_margin() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = crop_with_padding(img, bbox=(50, 25, 100, 75), pad_ratio=0.1)
    # crop must include bbox area + margin
    assert crop.shape[0] >= 50
    assert crop.shape[1] >= 50


def test_resize_keep_aspect_pads_to_target() -> None:
    img = np.full((300, 200, 3), 64, dtype=np.uint8)
    out = resize_keep_aspect(img, (256, 128))
    assert out.shape == (128, 256, 3)
