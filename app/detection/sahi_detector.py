"""SAHI (Slicing Aided Hyper Inference) detector.

Wraps the same PaddleDetection pedestrian model that
``paddledetection_pipeline.py`` uses. Slices a BGR frame into smaller
overlapping patches, runs the detector on each, and applies
class-agnostic NMS across patches.

This class lives in the api image (Paddle-only, no torch). It is
loaded once at process startup and re-used across frames.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SahiConfig:
    """All SAHI tunables. Constructed from env vars via ``from_env``."""

    patch_size: int = 320
    overlap_ratio: float = 0.2
    min_area: int = 30
    nms_iou: float = 0.5
    filter_threshold: float = 0.4
    device: str = "gpu:0"

    @classmethod
    def from_env(cls) -> "SahiConfig":
        return cls(
            patch_size=int(os.environ.get("SAHI_PATCH_SIZE", "320")),
            overlap_ratio=float(os.environ.get("SAHI_OVERLAP_RATIO", "0.2")),
            min_area=int(os.environ.get("SAHI_MIN_AREA", "30")),
            nms_iou=float(os.environ.get("SAHI_NMS_IOU", "0.5")),
            filter_threshold=float(os.environ.get("SAHI_FILTER_THRESHOLD", "0.4")),
            device=os.environ.get("SAHI_DEVICE", "gpu:0"),
        )


# A detection is (x1, y1, x2, y2, score) in full-frame pixel coords.
Detection = Tuple[float, float, float, float, float]


class SahiDetector:
    """Slices a BGR frame, runs the predictor on each patch, NMS-merges."""

    def __init__(
        self,
        *,
        config: SahiConfig,
        model_file: str,
        params_file: str,
    ) -> None:
        self._config = config
        self._model_file = model_file
        self._params_file = params_file
        self._predictor = None  # lazy-loaded on first predict()
        self._load_model()

    def _load_model(self) -> None:
        """Load the Paddle predictor.

        Honors ``self._config.device``: ``"gpu:N"`` (default ``"gpu:0"``)
        or ``"cpu"``. Unknown values fall back to ``gpu:0`` with a warning.

        Uses the same Config() + create_predictor() pattern as
        ``app/reid/pphuman_adapter.py``. We accept this method being
        patched in tests (the test suite mocks it).

        PATCH (2026-06-16, root-cause fix): the previous code passed
        ``Config(self._infer_config_path, self._model_dir)`` where
        ``_infer_config_path`` was the PP-Human *pipeline* YAML and
        ``_model_dir`` was the parent model directory. paddle.inference.Config
        expects ``(model.pdmodel, model.pdiparams)`` of a single
        inference model — not a pipeline YAML and a parent dir. The
        misconfigured config caused Paddle's AnalysisPredictor::Init
        to segfault in NaiveExecutor::CreateVariables on the api
        process startup, taking down the entire chain. The fix:
        accept explicit ``model_file`` and ``params_file`` paths (the
        DET model files, e.g.
        ``/models/pphuman/mot_ppyoloe_l_36e_pipeline/model.pdmodel``
        and ``model.pdiparams``).
        """
        # Imported lazily so the test suite can construct SahiDetector
        # without paddle installed.
        from paddle.inference import Config, create_predictor  # noqa: WPS433

        cfg = Config(self._model_file, self._params_file)
        device = self._config.device
        if device.startswith("gpu:"):
            gpu_id = int(device.split(":", 1)[1])
            cfg.enable_use_gpu(1000, gpu_id)
        elif device == "cpu":
            cfg.disable_gpu()
        else:
            # Unknown device string; fall back to GPU 0 with a warning.
            logger.warning(
                "SahiDetector unknown device=%r; falling back to gpu:0",
                device,
            )
            cfg.enable_use_gpu(1000, 0)
        cfg.set_cpu_math_library_num_threads(1)
        cfg.disable_glog_info()
        cfg.switch_ir_optim(False)
        self._predictor = create_predictor(cfg)
        logger.info(
            "SahiDetector loaded device=%s model_file=%s params_file=%s",
            device,
            self._model_file,
            self._params_file,
        )

    def predict(self, bgr: np.ndarray) -> List[Detection]:
        """Slice ``bgr`` into patches, run predictor on each, NMS-merge.

        Returns a list of ``(x1, y1, x2, y2, score)`` in full-frame coords.
        Empty list on error.
        """
        if bgr is None or bgr.size == 0:
            return []
        try:
            return self._predict_sliced(bgr)
        except Exception as e:  # noqa: BLE001
            logger.error("SahiDetector.predict error: %s", e, exc_info=False)
            return []

    def _predict_sliced(self, bgr: np.ndarray) -> List[Detection]:
        all_dets: List[np.ndarray] = []
        for x0, y0, patch in self._slice_frame(bgr):
            if patch is None or patch.size == 0:
                continue
            preds = self._predictor.predict([patch])  # may return list[ndarray]
            for arr in preds or []:
                if arr is None or len(arr) == 0:
                    continue
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.shape[1] < 5:
                    continue
                # Translate patch-coord bboxes to full-frame coords.
                translated = arr.copy()
                translated[:, 0] += x0
                translated[:, 2] += x0
                translated[:, 1] += y0
                translated[:, 3] += y0
                all_dets.append(translated)
        if not all_dets:
            return []
        merged = np.concatenate(all_dets, axis=0)
        return self._nms_filter(merged)

    def _slice_frame(self, bgr: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
        """Return overlapping image patches without importing external SAHI.

        The upstream ``sahi`` package eagerly imports model backends from
        ``sahi.__init__``; several of those require torch. That violates
        this API image's Paddle-only runtime contract. We only need the
        deterministic slicing grid, so keep it local and dependency-free.
        """
        h, w = bgr.shape[:2]
        patch = max(1, int(self._config.patch_size))
        overlap = min(max(float(self._config.overlap_ratio), 0.0), 0.95)
        step = max(1, int(round(patch * (1.0 - overlap))))

        def _starts(length: int) -> list[int]:
            if length <= patch:
                return [0]
            starts = list(range(0, max(1, length - patch + 1), step))
            last = length - patch
            if starts[-1] != last:
                starts.append(last)
            return starts

        slices: list[tuple[int, int, np.ndarray]] = []
        for y0 in _starts(h):
            for x0 in _starts(w):
                slices.append((x0, y0, bgr[y0 : y0 + patch, x0 : x0 + patch]))
        return slices

    def _nms_filter(self, dets: np.ndarray) -> List[Detection]:
        """Class-agnostic NMS + min_area + score filter.

        Implemented in pure numpy (no torch) to honor the api image's
        Paddle-only contract.
        """
        if dets.size == 0:
            return []
        # Score filter.
        mask = dets[:, 4] >= self._config.filter_threshold
        dets = dets[mask]
        if dets.size == 0:
            return []
        # Min-area filter.
        areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        dets = dets[areas >= self._config.min_area]
        if dets.size == 0:
            return []
        # Class-agnostic NMS (pure numpy, no torch).
        keep = self._class_agnostic_nms(
            dets[:, :4],
            dets[:, 4],
            iou_threshold=self._config.nms_iou,
        )
        if len(keep) == 0:
            return []
        kept = dets[keep]
        return [
            (float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]))
            for r in kept
        ]

    @staticmethod
    def _class_agnostic_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> List[int]:
        """Pure-numpy class-agnostic non-max suppression.

        Returns the indices of boxes to keep, ordered by descending
        score. Standard greedy NMS: pick highest score, suppress
        boxes with IoU >= threshold, repeat.
        """
        if boxes.shape[0] == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou < iou_threshold]
        return keep
