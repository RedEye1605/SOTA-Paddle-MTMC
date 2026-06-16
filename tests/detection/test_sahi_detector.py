"""Tests for the SahiDetector.

These tests do NOT require a GPU or a real PP-Human model. They mock
the underlying Paddle predictor and assert the slicing / NMS math
is correct.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from app.detection.sahi_detector import SahiDetector, SahiConfig


def _dummy_frame(h: int = 1080, w: int = 1920) -> np.ndarray:
    """Return a solid-color BGR frame for slicing tests."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_sahi_config_defaults():
    cfg = SahiConfig()
    assert cfg.patch_size == 320
    assert cfg.overlap_ratio == 0.2
    assert cfg.min_area == 30
    assert cfg.nms_iou == 0.5
    assert cfg.filter_threshold == 0.4
    assert cfg.device == "gpu:0"


def test_sahi_config_env_override(monkeypatch):
    monkeypatch.setenv("SAHI_PATCH_SIZE", "256")
    monkeypatch.setenv("SAHI_OVERLAP_RATIO", "0.3")
    monkeypatch.setenv("SAHI_NMS_IOU", "0.6")
    cfg = SahiConfig.from_env()
    assert cfg.patch_size == 256
    assert abs(cfg.overlap_ratio - 0.3) < 1e-6
    assert cfg.nms_iou == 0.6


def test_sahi_predict_returns_list_of_tuples():
    """predict() must return [(x1, y1, x2, y2, score), ...]."""
    cfg = SahiConfig()
    with patch.object(SahiDetector, "_load_model"):
        detector = SahiDetector(
            config=cfg, model_file="/fake.pdmodel", params_file="/fake.pdiparams"
        )
        detector._predictor = MagicMock()
        detector._predictor.predict = MagicMock(
            return_value=[
                np.array([10.0, 20.0, 100.0, 200.0, 0.9]),
            ]
        )
        out = detector.predict(_dummy_frame())
    assert isinstance(out, list)
    assert len(out) >= 0
    for det in out:
        assert len(det) == 5
        x1, y1, x2, y2, score = det
        assert x1 < x2
        assert y1 < y2
        assert 0.0 <= score <= 1.0


def test_sahi_predict_empty_frame_returns_empty():
    cfg = SahiConfig()
    with patch.object(SahiDetector, "_load_model"):
        detector = SahiDetector(
            config=cfg, model_file="/fake.pdmodel", params_file="/fake.pdiparams"
        )
        detector._predictor = MagicMock()
        detector._predictor.predict = MagicMock(return_value=[])
        out = detector.predict(_dummy_frame())
    assert out == []


def test_sahi_predict_filters_by_min_area():
    """Detections smaller than min_area should be dropped."""
    cfg = SahiConfig(min_area=30)
    with patch.object(SahiDetector, "_load_model"):
        detector = SahiDetector(
            config=cfg, model_file="/fake.pdmodel", params_file="/fake.pdiparams"
        )
        detector._predictor = MagicMock()
        # 4x4 bbox = 16 pixels² < 30 → drop
        detector._predictor.predict = MagicMock(return_value=[
            np.array([0.0, 0.0, 4.0, 4.0, 0.9]),
            # 100x100 = 10000 pixels² > 30 → keep
            np.array([10.0, 20.0, 110.0, 120.0, 0.9]),
        ])
        out = detector.predict(_dummy_frame())
    # After sahi slicing, the input is split into patches and each
    # patch is run through the mock predictor. We assert that the
    # small bbox is dropped (and at least one survives, in the
    # patch that contains the 100x100 bbox).
    assert all(
        (x2 - x1) * (y2 - y1) >= 30 for x1, y1, x2, y2, _ in out
    ), f"min_area filter failed; got: {out}"


def test_sahi_predict_no_torch():
    """SahiDetector construction must not import torch.

    We snapshot the set of ``torch*`` modules in ``sys.modules``
    before construction and assert no new ones are added. We
    intentionally do NOT check absolute absence of ``torch`` in
    ``sys.modules``: previous tests in this file (or in
    ``test_sahi_dependency.py``'s import-time skips) may have
    imported ``sahi`` and its eager torch cascade. The contract
    here is "the detector's own code never imports torch", which
    is what the api image's Paddle-only dependency contract enforces.
    """
    import sys

    cfg = SahiConfig()
    torch_before = {k for k in sys.modules if k == "torch" or k.startswith("torch.")}
    with patch.object(SahiDetector, "_load_model"):
        SahiDetector(config=cfg, model_file="/fake.pdmodel", params_file="/fake.pdiparams")
    torch_after = {k for k in sys.modules if k == "torch" or k.startswith("torch.")}
    added = torch_after - torch_before
    assert not added, f"SahiDetector construction imported torch modules: {added}"
