"""PP-Human StrongBaseline ReID adapter.

EXPERIMENTAL — OFF BY DEFAULT (operator spec, 2026-06-15).

The active production model is **TransReID MSMT17** (see
``app.reid.transreid_adapter``). The PP-Human StrongBaseline ReID was
the original plan but the operator dropped it: the Paddle 2.x
``strongbaseline_r50_30e_pa100k`` weights do not load under Paddle 3.x
(the ``model.json`` shape mismatches), and the api image bundles
Paddle 3.3.1 with no TransReID / torch deps.

This module is kept as a plug-in shell so operators who want to vendor
the upstream StrongBaseline ReID can do so via the
``PPHUMAN_REID_INFERENCE_FN`` env var. It is **NOT** instantiated by
``app.main`` (see ``select_reid_adapter()`` in ``app.main``) and the
default ``active_model`` in ``configs/app.yaml`` is
``transreid_msmt``.

Activation:
  * Set ``reid.active_model: pphuman_strongbaseline`` in ``app.yaml``
    (or ``SOTA_REID_MODEL=pphuman_strongbaseline``).
  * Provide a real paddle.inference predictor via the plug-in env var
    (see below) — the deterministic histogram fallback is forbidden
    in production.

Production path: a paddle.inference predictor loaded against
``/models/pphuman/strongbaseline_r50_30e_pa100k`` (the model the
PP-Human pipeline uses in its ATTR block; the SAME 256-dim StrongBaseline
features are exposed per-frame in the official PP-Human pipeline, so
using the ATTR head here is consistent with the pipeline's own per-frame
ReID embedding). The operator can also plug in a custom inference
callable via ``PPHUMAN_REID_INFERENCE_FN=module:fn``.

Smoke-test path: deterministic 256-dim histogram feature. Active only in
``RuntimeMode.SMOKE_TEST``; in production ``load()`` raises
``ProductionSafetyError`` if no real path is configured.

This module is intentionally import-friendly when ``paddle`` is not
installed: the paddle import is lazy. The vendored path uses
``paddle.inference`` rather than the higher-level ``paddle.jit`` API
because the StrongBaseline model is exported with Paddle's
``tools/export_model.py`` and ships as a static ``inference.pdmodel`` /
``inference.pdiparams`` pair — the inference engine is the canonical
way to load it (per the official PaddleDetection deployment docs).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from ..core.runtime_mode import (
    PPHUMAN_INFER_FN_ENV,
    RuntimeMode,
    assert_production_safe,
    get_inference_callable,
    resolve_runtime_mode,
    smoke_log,
)
from ..utils.crop import l2_normalize, resize_keep_aspect
from .base import ReIDAdapter, ReIDConfig

logger = logging.getLogger(__name__)

# StrongBaseline preprocessing per the official PaddleReID config.
PIXEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PIXEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PPHumanReIDAdapter(ReIDAdapter):
    def __init__(
        self,
        config: ReIDConfig,
        weight_dir: str | None = None,
        device: str = "gpu",
        use_fp16: bool = True,
        mode: Optional[RuntimeMode] = None,
    ) -> None:
        self.config = config
        self._weight_dir = weight_dir or os.environ.get(
            "PPHUMAN_REID_MODEL_DIR",
            os.environ.get("PPHUMAN_MODEL_DIR", "/models/pphuman/strongbaseline_r50_30e_pa100k"),
        )
        self._device_name = device
        self._use_fp16 = use_fp16
        self._mode = mode or resolve_runtime_mode()
        self._model = None
        self._predictor = None
        self._fallback_active = False

    def load(self) -> None:
        # 1) Plug-in path: PPHUMAN_REID_INFERENCE_FN=module:fn (operator-defined).
        plug_in = get_inference_callable(PPHUMAN_INFER_FN_ENV)
        if plug_in is not None:
            self._model = plug_in()
            self._fallback_active = False
            return
        # 2) Real path: paddle.inference predictor against weight_dir.
        try:
            self._try_load_paddle()
        except Exception as e:  # noqa: BLE001
            if self._mode == RuntimeMode.SMOKE_TEST:
                logger.error(
                    "PP-Human StrongBaseline weights not loaded (%s). Falling "
                    "back to deterministic 256-dim feature. SMOKE-TEST ONLY.",
                    e,
                )
                self._fallback_active = True
                self._model = None
            else:
                assert_production_safe(
                    mode=self._mode,
                    component="PPHumanReIDAdapter",
                    condition=f"missing real model (weight_dir={self._weight_dir!r})",
                )

    def warmup(self) -> None:
        if self._model is not None and not self._fallback_active:
            try:
                self.extract([np.zeros((128, 64, 3), dtype=np.uint8)])
            except Exception:  # noqa: BLE001
                pass

    def extract(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        if self._fallback_active:
            return self._deterministic_fallback(crops)
        if self._model is None:
            # PATCH-003 / BUG-003 / PATCH-040: refuse to operate in
            # production with no real model. Smoke mode allows the
            # deterministic fallback so the existing test suite
            # continues to work; production raises
            # ProductionSafetyError so a misconfigured deploy can
            # never silently run on histogram features.
            assert_production_safe(
                mode=self._mode,
                component="PPHumanReIDAdapter",
                condition="deterministic fallback because the model is not loaded",
            )
        return self._extract_paddle(crops)

    # ---- private ----
    def _try_load_paddle(self) -> None:
        weight_dir = Path(self._weight_dir)
        if not weight_dir.exists():
            raise FileNotFoundError(f"PPHuman model dir not found: {self._weight_dir}")
        # The official PaddleDetection export tool produces
        # ``inference.pdmodel`` + ``inference.pdiparams``, but the
        # BCE Bos pipeline zip (mot_ppyoloe_l_36e_pipeline.zip,
        # strongbaseline_r50_30e_pa100k.zip) ships as
        # ``model.pdmodel`` + ``model.pdiparams``. Accept either.
        # Operators can override with PPHUMAN_REID_MODEL_BASENAME.
        candidates = [
            os.environ.get("PPHUMAN_REID_MODEL_BASENAME", "").strip(),
            "inference",
            "model",
        ]
        model_file: Path | None = None
        params_file: Path | None = None
        for stem in candidates:
            if not stem:
                continue
            m = weight_dir / f"{stem}.pdmodel"
            p = weight_dir / f"{stem}.pdiparams"
            if m.exists() and p.exists():
                model_file, params_file = m, p
                break
        if model_file is None or params_file is None:
            raise FileNotFoundError(
                f"PPHuman StrongBaseline export not found in {self._weight_dir}: "
                f"expected (inference|model).pdmodel + matching .pdiparams "
                f"(use PaddleDetection tools/export_model.py to export, or unpack "
                f"strongbaseline_r50_30e_pa100k.zip from BCE Bos).",
            )
        try:
            import paddle  # type: ignore
            from paddle import inference  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"paddlepaddle not installed: {e}")
        config = inference.Config(str(model_file), str(params_file))
        trt_requested = False
        # Device selection — gpu/cpu/xpu/npu as the operator wants.
        if self._device_name == "gpu" and paddle.device.is_compiled_with_cuda():
            config.enable_use_gpu(100, 0)
            if self._use_fp16:
                try:
                    config.enable_tensorrt_engine(
                        workspace_size=1 << 30,
                        max_batch_size=16,
                        min_subgraph_size=3,
                        precision_mode=inference.PrecisionType.Half,
                        use_static=False,
                        use_calib_mode=False,
                    )
                    trt_requested = True
                except Exception:  # noqa: BLE001
                    logger.warning("TensorRT engine not enabled; using paddle FP16")
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(8)
        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)
        try:
            predictor = inference.create_predictor(config)
        except Exception as e:  # noqa: BLE001
            if not trt_requested:
                raise
            # The TRT shared library (libnvinfer.so) isn't installed in
            # the runtime image, but Paddle only attempts to dlopen it
            # at create_predictor time (not at enable_tensorrt_engine
            # time).  Fall back to native Paddle inference so the
            # adapter still loads — operators who need TRT must add
            # the TensorRT runtime to the image.
            logger.warning(
                "TensorRT predictor create failed (%s); rebuilding "
                "config WITHOUT TRT and retrying with native Paddle",
                str(e).splitlines()[0],
            )
            config = inference.Config(str(model_file), str(params_file))
            if self._device_name == "gpu" and paddle.device.is_compiled_with_cuda():
                config.enable_use_gpu(100, 0)
            else:
                config.disable_gpu()
                config.set_cpu_math_library_num_threads(8)
            config.switch_use_feed_fetch_ops(False)
            config.switch_ir_optim(True)
            predictor = inference.create_predictor(config)
        logger.info(
            "PP-Human StrongBaseline loaded: dir=%s device=%s fp16=%s",
            self._weight_dir,
            self._device_name,
            self._use_fp16,
        )
        self._predictor = predictor
        self._model = predictor
        self._fallback_active = False

    def _preprocess(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Convert BGR crops to a float32 NCHW tensor of shape (N, 3, 256, 128)."""
        out: list[np.ndarray] = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            try:
                resized = resize_keep_aspect(crop, (128, 256))
            except Exception:
                continue
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            normalized = (rgb - PIXEL_MEAN) / PIXEL_STD
            chw = normalized.transpose(2, 0, 1)  # (3, 256, 128)
            out.append(chw)
        if not out:
            return np.zeros((0, 3, 256, 128), dtype=np.float32)
        return np.stack(out, axis=0).astype(np.float32)

    def _extract_paddle(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        x = self._preprocess(crops)
        if x.shape[0] == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        # The exported StrongBaseline model has input name "x" and output
        # name "feature" / "logits" depending on the export. The standard
        # PaddleReID export uses input "image" and output "embedding".
        predictor = self._predictor
        input_names = predictor.get_input_names()
        if not input_names:
            raise RuntimeError("PP-Human predictor has no input names; cannot run inference")
        input_name = input_names[0]
        predictor.run()
        predictor.set_feed(input_name, x)
        # The predict() helper triggers the forward pass; then we read the
        # output. The exported StrongBaseline has a single output tensor
        # named "embedding" or "features". We read it by index 0.
        output_names = predictor.get_output_names()
        if not output_names:
            raise RuntimeError("PP-Human predictor has no output names; cannot read result")
        feat = predictor.get_fetch_handle(output_names[0]).copy_to_cpu()
        feat = np.asarray(feat, dtype=np.float32)
        if feat.ndim == 1:
            feat = feat[None, :]
        # L2-normalize per the StrongBaseline TEST config.
        return l2_normalize(feat)

    def _deterministic_fallback(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Stable 256-dim feature for smoke tests. NOT a real ReID feature."""
        smoke_log(
            "PPHumanReIDAdapter",
            "deterministic 256-dim histogram fallback (SMOKE-TEST)",
        )
        out = np.zeros((len(crops), self.embedding_dim), dtype=np.float32)
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                out[i] = np.zeros(self.embedding_dim, dtype=np.float32)
                continue
            try:
                resized = resize_keep_aspect(crop, self.config.input_size)
            except Exception:
                out[i] = np.zeros(self.embedding_dim, dtype=np.float32)
                continue
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            feats: list[float] = []
            for ch in range(3):
                hist, _ = np.histogram(hsv[:, :, ch], bins=32, range=(0, 256))
                feats.extend((hist / max(1, hist.sum())).tolist())
            small = cv2.resize(resized, (4, 4))
            feats.extend((small.astype(np.float32) / 255.0).flatten().tolist())
            v = np.array(feats, dtype=np.float32)
            if v.size < self.embedding_dim:
                v = np.concatenate([v, np.zeros(self.embedding_dim - v.size, dtype=np.float32)])
            else:
                v = v[: self.embedding_dim]
            out[i] = l2_normalize(v)
        return out


def make_default(mode: Optional[RuntimeMode] = None) -> PPHumanReIDAdapter:
    return PPHumanReIDAdapter(
        config=ReIDConfig(
            name="pphuman_strongbaseline",
            embedding_dim=256,
            qdrant_collection="person_reid_pphuman",
        ),
        mode=mode,
    )
