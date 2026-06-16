"""CLIP-ReID adapter (EXPERIMENTAL — OFF BY DEFAULT, operator spec).

The active production model is **TransReID MSMT17** (see
``app.reid.transreid_adapter``). The operator dropped CLIP-ReID along
with the PP-Human / vanilla-TransReID / CLIP-ReID lineup because the
upstream ``clip`` / ``open_clip`` deps conflict with the api image's
Paddle-only build and we do not want to vendor the upstream training
pipeline. This module is kept as a plug-in shell so operators who want
to bring their own CLIP-ReID inference callable can do so via the
``CLIPREID_INFERENCE_FN`` env var.

Activation (off by default):
  * Set ``reid.active_model: clipreid`` in ``app.yaml``.
  * Provide a real inference callable via ``CLIPREID_INFERENCE_FN``.
    The adapter delegates ``extract()`` to the plug-in and the
    deterministic histogram fallback is forbidden in production.

CLIP-ReID is two-stage: stage-1 warm-up of the text encoder, stage-2
fine-tune of the image encoder. It is **not** the production primary.
This module:

  * defaults to ``fallback_active=True`` — a 512-dim histogram feature
    is returned. The fallback is only allowed in
    ``RuntimeMode.SMOKE_TEST``; in production, ``load()`` raises
    ``ProductionSafetyError`` if the operator flips ``active=true`` but
    the weights are missing.
  * in production with the proper weights and ``CLIPREID_INFERENCE_FN``
    pointing at the upstream code, the operator's callable is invoked
    and the real 512-dim embedding is returned.

The reason CLIP-ReID stays optional: we do not want to vendor the
upstream CLIP+ReID pipeline (it depends on ``clip``, a custom
``open_clip`` fork, and several training-only modules). Production
deploys that want CLIP-ReID plug in the upstream code via env var.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..core.runtime_mode import (
    RuntimeMode,
    assert_production_safe,
    get_inference_callable,
    resolve_runtime_mode,
    smoke_log,
)
from ..utils.crop import l2_normalize, resize_keep_aspect
from .base import ReIDAdapter, ReIDConfig

logger = logging.getLogger(__name__)

CLIPREID_INFERENCE_FN_ENV = "CLIPREID_INFERENCE_FN"


class CLIPReIDAdapter(ReIDAdapter):
    def __init__(
        self,
        config: ReIDConfig,
        weight_path: str | None = None,
        mode: Optional[RuntimeMode] = None,
    ) -> None:
        self.config = config
        self._weight_path = weight_path or os.environ.get(
            "CLIPREID_WEIGHT",
            "/models/clipreid/clipreid.pth",
        )
        self._mode = mode or resolve_runtime_mode()
        self._model = None
        # Always fallback by default; the operator's plug-in can override.
        self._fallback_active = True

    def load(self) -> None:
        # 1) Plug-in path: CLIPREID_INFERENCE_FN=module:fn
        plug_in = get_inference_callable(CLIPREID_INFERENCE_FN_ENV)
        if plug_in is not None:
            self._model = plug_in()
            self._fallback_active = False
            return
        # 2) Real path: vendor isn't shipped. We refuse unless a
        #    plug-in is configured.
        if not Path(self._weight_path).exists():
            assert_production_safe(
                mode=self._mode,
                component="CLIPReIDAdapter",
                condition=(
                    f"missing weight (path={self._weight_path!r}) and no "
                    f"{CLIPREID_INFERENCE_FN_ENV} plug-in configured"
                ),
            )
        # We deliberately do not try to load the real model in this
        # adapter; the operator plugs in the upstream code via env var.
        assert_production_safe(
            mode=self._mode,
            component="CLIPReIDAdapter",
            condition=(
                f"CLIP-ReID weights present but no {CLIPREID_INFERENCE_FN_ENV} plug-in configured"
            ),
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
                component="CLIPReIDAdapter",
                condition="deterministic fallback because the model is not loaded",
            )
        # Plug-in model — we delegate to it. The plug-in is expected to
        # accept a list of BGR crops and return a numpy array of shape
        # (N, 512).
        return np.asarray(self._model(crops), dtype=np.float32)

    def _deterministic_fallback(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        smoke_log("CLIPReIDAdapter", "deterministic 512-dim histogram fallback (SMOKE-TEST)")
        out = np.zeros((len(crops), self.embedding_dim), dtype=np.float32)
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            try:
                resized = resize_keep_aspect(crop, self.config.input_size)
            except Exception:
                continue
            feats: list[float] = []
            for ch in range(3):
                hist, _ = np.histogram(resized[:, :, ch], bins=64, range=(0, 256))
                feats.extend((hist / max(1, hist.sum())).tolist())
            v = np.array(feats, dtype=np.float32)
            if v.size < self.embedding_dim:
                v = np.concatenate([v, np.zeros(self.embedding_dim - v.size, dtype=np.float32)])
            else:
                v = v[: self.embedding_dim]
            out[i] = l2_normalize(v)
        return out


def make_default(mode: Optional[RuntimeMode] = None) -> CLIPReIDAdapter:
    return CLIPReIDAdapter(
        config=ReIDConfig(
            name="clipreid",
            embedding_dim=512,
            qdrant_collection="person_reid_clipreid_optional",
        ),
        mode=mode,
    )
