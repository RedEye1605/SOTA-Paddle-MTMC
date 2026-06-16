"""TransReID adapter (vit_base_patch16_224_TransReID, 768-dim).

Production path: the real TransReID backbone from
``app.reid._transreid_native`` (a minimal vendor of the official
``damo-cv/TransReID`` inference path). Loads the official checkpoint
with ``weights_only=True``, switches to FP16 on GPU, and returns
L2-normalized 5x768 JPM features.

Smoke-test path: deterministic 768-dim histogram feature. The smoke
path is only active in ``RuntimeMode.SMOKE_TEST``; in production
``load()`` raises ``ProductionSafetyError`` if the weight is missing.

Security: we always load with ``weights_only=True`` to refuse
arbitrary pickle execution. This is verified by the
``test_dangerous_weights_refused`` architecture guard.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from ..core.runtime_mode import (
    RuntimeMode,
    TRANSREID_MODEL_FN_ENV,
    assert_production_safe,
    get_inference_callable,
    resolve_runtime_mode,
    smoke_log,
)
from ..utils.crop import l2_normalize, resize_keep_aspect
from .base import ReIDAdapter, ReIDConfig

logger = logging.getLogger(__name__)

# Constants from the official config — kept here so the adapter
# does not depend on a third-party yacs file.
PIXEL_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
PIXEL_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


# Profile table — matches scripts/inspect_transreid_checkpoint.py.
# PATCH-011: each profile pins a num_class that must match the
# on-disk classifier head.
TRANSREID_PROFILES: dict[str, dict[str, int]] = {
    "market1501": {"num_class": 751, "embedding_dim": 3840},
    "msmt17": {"num_class": 1041, "embedding_dim": 3840},
    "duke": {"num_class": 702, "embedding_dim": 3840},
    "veri": {"num_class": 776, "embedding_dim": 3840},
}


def _resolve_transreid_profile(
    profile: str,
    *,
    num_class: Optional[int] = None,
    embedding_dim: Optional[int] = None,
) -> tuple[int, int]:
    """Return ``(num_class, embedding_dim)`` for a profile.

    ``custom`` requires the operator to supply both values.
    """
    if profile in TRANSREID_PROFILES:
        info = TRANSREID_PROFILES[profile]
        return info["num_class"], info["embedding_dim"]
    if profile == "custom":
        if num_class is None or embedding_dim is None:
            raise ValueError(
                "profile=custom requires num_class and embedding_dim",
            )
        return int(num_class), int(embedding_dim)
    raise ValueError(
        f"Unknown TransReID profile {profile!r}; "
        f"choose from {list(TRANSREID_PROFILES) + ['custom']}",
    )


def _inspect_classifier_shape(state: dict) -> dict[str, Any] | None:
    """Return a dict describing the classifier-head shape in `state`.

    Detects CLS-only (`classifier.weight`) and JPM (`classifier.0.weight`
    ... `classifier.4.weight`). Returns None if no classifier head.
    """
    # Strip DataParallel prefix.
    s = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    if "classifier.weight" in s:
        shape = tuple(s["classifier.weight"].shape)
        return {
            "kind": "cls_only",
            "num_class": int(shape[0]),
            "embedding_dim": int(shape[1]) if len(shape) > 1 else None,
        }
    jpm = [k for k in s if k.startswith("classifier.") and k.endswith(".weight")]
    if jpm:
        shape = tuple(s[jpm[0]].shape)
        return {
            "kind": "jpm",
            "num_linears": len(jpm),
            "num_class": int(shape[0]) if shape else None,
            "embedding_dim": int(shape[1]) if len(shape) > 1 else None,
        }
    return None


class TransReIDAdapter(ReIDAdapter):
    def __init__(
        self,
        config: ReIDConfig,
        weight_path: str | None = None,
        weights_only: bool = True,
        num_class: int = 751,
        camera_num: int = 0,  # SIE is training-only; disabled for inference
        view_num: int = 0,
        device: str = "cuda",
        mode: Optional[RuntimeMode] = None,
        profile: str = "market1501",
        ignore_classifier_head: bool = True,
        require_checkpoint_in_production: bool = True,
    ) -> None:
        self.config = config
        self._weight_path = weight_path or os.environ.get(
            "TRANSREID_WEIGHT",
            "/models/transreid/transformer_120.pth",
        )
        self._weights_only = weights_only
        # Resolve profile → (num_class, embedding_dim).
        self._profile = profile
        profile_nc, profile_ed = _resolve_transreid_profile(
            profile,
            num_class=num_class,
            embedding_dim=config.embedding_dim,
        )
        self._num_class = profile_nc
        # If the operator passed an explicit embedding_dim that differs
        # from the profile default, honor it (custom override).
        if config.embedding_dim and config.embedding_dim != profile_ed:
            self._expected_embedding_dim = int(config.embedding_dim)
        else:
            self._expected_embedding_dim = profile_ed
        self._ignore_classifier_head = bool(ignore_classifier_head)
        self._require_checkpoint_in_production = bool(require_checkpoint_in_production)
        self._camera_num = camera_num
        self._view_num = view_num
        self._device_name = device
        self._mode = mode or resolve_runtime_mode()
        self._model = None
        self._has_jpm = True
        self._fallback_active = False
        self._inspect_result: dict[str, Any] = {}

    # ---- public ----
    def load(self) -> None:
        # 1) Plug-in path: TRANSREID_MODEL_FN=module:callable is preferred
        #    (the operator can vendor the upstream repo independently).
        plug_in = get_inference_callable(TRANSREID_MODEL_FN_ENV)
        if plug_in is not None:
            self._model = plug_in()
            self._fallback_active = False
            return
        # 2) Real-path: load the vendored backbone + the on-disk checkpoint.
        try:
            self._try_load_real()
        except Exception as e:  # noqa: BLE001
            if self._mode == RuntimeMode.SMOKE_TEST:
                logger.error(
                    "TransReID weights not loaded (%s). Falling back to "
                    "deterministic %d-dim feature. SMOKE-TEST ONLY.",
                    e,
                    self._expected_embedding_dim,
                )
                self._fallback_active = True
                self._model = None
            elif not self._require_checkpoint_in_production:
                # PATCH-011: operator has explicitly opted out of the
                # "must have a checkpoint file" rule. This is the
                # "feature-extractor mode" used when the operator
                # vendors the upstream model externally and the
                # vendored inference path is not in use. We log a
                # loud warning and fall back to the histogram.
                logger.error(
                    "TransReID checkpoint missing AND "
                    "require_checkpoint_in_production=False. "
                    "Falling back to deterministic %d-dim feature. "
                    "This is INTENDED for feature-extractor mode only; "
                    "the resolver will run on histogram features until "
                    "the operator wires a real model. weight=%r",
                    self._expected_embedding_dim,
                    self._weight_path,
                )
                self._fallback_active = True
                self._model = None
            else:
                # Production: refuse to start.
                assert_production_safe(
                    mode=self._mode,
                    component="TransReIDAdapter",
                    condition=f"missing real model (weight={self._weight_path!r})",
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
                component="TransReIDAdapter",
                condition="deterministic fallback because the model is not loaded",
            )
        return self._extract_real(crops)

    @property
    def profile(self) -> str:
        return self._profile

    @property
    def inspect_result(self) -> dict[str, Any]:
        """Return the latest checkpoint-inspection result (PATCH-011)."""
        return dict(self._inspect_result)

    # ---- private ----
    def _try_load_real(self) -> None:
        if not Path(self._weight_path).exists():
            raise FileNotFoundError(f"TransReID weight not found: {self._weight_path}")
        try:
            import torch  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"torch not installed: {e}")
        # Resolve device
        device = self._device_name
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available; falling back to CPU for TransReID")
            device = "cpu"
        use_fp16 = self.config.use_fp16 and device.startswith("cuda")
        # PATCH-011: preflight the checkpoint vs. profile. We refuse to
        # load if the classifier-head shape is incompatible AND the
        # operator has not opted into ``ignore_classifier_head``.
        self._inspect_result = self._preflight_checkpoint()
        if not self._inspect_result["ok"]:
            reason = self._inspect_result.get("reason")
            raise RuntimeError(
                f"TransReID checkpoint preflight failed "
                f"(profile={self._profile}, weight={self._weight_path!r}): "
                f"{reason}: expected num_class={self._num_class}, "
                f"got {self._inspect_result.get('detected_num_class')}",
            )
        from ._transreid_native import (
            build_transreid_model,
            load_transreid_checkpoint,
        )

        model, has_jpm = build_transreid_model(
            num_class=self._num_class,
            camera_num=self._camera_num,
            view_num=self._view_num,
            stride_size=12,
            sie_xishu=3.0,
            jpm=True,
            use_fp16=use_fp16,
            device=device,
        )
        summary = load_transreid_checkpoint(
            model,
            self._weight_path,
            device=device,
            weights_only=self._weights_only,
        )
        logger.info(
            "TransReID loaded: profile=%s weight=%s missing=%d unexpected=%d "
            "device=%s fp16=%s ignore_classifier_head=%s",
            self._profile,
            self._weight_path,
            len(summary["missing_keys"]),
            len(summary["unexpected_keys"]),
            device,
            use_fp16,
            self._ignore_classifier_head,
        )
        self._model = model
        self._has_jpm = has_jpm
        self._device = device
        self._use_fp16 = use_fp16
        self._fallback_active = False

    def _preflight_checkpoint(self) -> dict[str, Any]:
        """Inspect the on-disk checkpoint and verify profile compatibility.

        PATCH-011: the operator selects a profile (market1501 / msmt17
        / custom) and we verify the classifier head matches. If
        ``ignore_classifier_head=True`` the mismatch is allowed and
        logged (recommended for inference). If False, the preflight
        fails.
        """
        result: dict[str, Any] = {
            "path": self._weight_path,
            "profile": self._profile,
            "expected_num_class": self._num_class,
            "ignore_classifier_head": self._ignore_classifier_head,
        }
        try:
            import torch  # type: ignore
        except ImportError:
            result["ok"] = False
            result["reason"] = "torch_not_installed"
            return result
        try:
            ckpt = torch.load(
                self._weight_path, map_location="cpu", weights_only=self._weights_only
            )
        except FileNotFoundError as e:
            result["ok"] = False
            result["reason"] = "checkpoint_missing"
            result["error"] = str(e)
            return result
        except Exception as e:  # noqa: BLE001
            result["ok"] = False
            result["reason"] = "load_failed"
            result["error"] = str(e)
            return result
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            result["ok"] = False
            result["reason"] = "not_a_torch_checkpoint"
            return result
        cls = _inspect_classifier_shape(state)
        if cls is None:
            # No classifier head in the checkpoint — this is fine if
            # we explicitly allow it (feature-extractor only).
            if self._ignore_classifier_head:
                result["ok"] = True
                result["reason"] = "no_classifier_head_ignored"
                return result
            result["ok"] = False
            result["reason"] = "no_classifier_head"
            return result
        result["classifier"] = cls
        detected = cls.get("num_class")
        result["detected_num_class"] = detected
        if detected is None:
            result["ok"] = False
            result["reason"] = "classifier_shape_unreadable"
            return result
        if detected != self._num_class:
            if self._ignore_classifier_head:
                logger.warning(
                    "TransReID classifier-head mismatch: profile=%s expected "
                    "num_class=%d got=%d. ignore_classifier_head=true → "
                    "loading feature extractor only (the BNNeck and "
                    "classifier head will be ignored).",
                    self._profile,
                    self._num_class,
                    detected,
                )
                result["ok"] = True
                result["reason"] = "classifier_mismatch_ignored"
                return result
            result["ok"] = False
            result["reason"] = "classifier_mismatch"
            return result
        result["ok"] = True
        result["reason"] = "compatible"
        return result

    def _preprocess(self, crops: Sequence[np.ndarray]) -> "torch.Tensor":  # noqa: F821
        import torch  # type: ignore

        tensors: list[torch.Tensor] = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            # Resize to (H=256, W=128) per official config.
            try:
                resized = resize_keep_aspect(crop, (128, 256))
            except Exception:
                continue
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            normalized = (rgb - PIXEL_MEAN) / PIXEL_STD
            # HWC -> CHW
            chw = normalized.transpose(2, 0, 1)
            tensors.append(torch.from_numpy(chw))
        if not tensors:
            return torch.zeros((0, 3, 256, 128))
        return torch.stack(tensors, dim=0).to(self._device)

    def _extract_real(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        import torch  # type: ignore
        from ._transreid_native import extract_inference_feature

        x = self._preprocess(crops)
        if x.shape[0] == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        if self._use_fp16 and self._device.startswith("cuda"):
            x = x.half()
        B = x.shape[0]
        cam_labels = torch.zeros(B, dtype=torch.long, device=self._device)
        view_labels = torch.zeros(B, dtype=torch.long, device=self._device)
        feat = extract_inference_feature(
            self._model,
            has_jpm=self._has_jpm,
            images=x,
            cam_labels=cam_labels,
            view_labels=view_labels,
            neck_feat="before",
            l2_normalize=True,
        )
        return feat.float().cpu().numpy().astype(np.float32)

    def _deterministic_fallback(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Stable 768-dim feature for smoke tests. NOT a real ReID feature."""
        smoke_log("TransReIDAdapter", "deterministic 768-dim histogram fallback (SMOKE-TEST)")
        out = np.zeros((len(crops), self.embedding_dim), dtype=np.float32)
        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue
            try:
                resized = resize_keep_aspect(crop, self.config.input_size)
            except Exception:
                continue
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            feats: list[float] = []
            for ch in range(3):
                hist, _ = np.histogram(hsv[:, :, ch], bins=64, range=(0, 256))
                feats.extend((hist / max(1, hist.sum())).tolist())
            small = cv2.resize(resized, (8, 8))
            feats.extend((small.astype(np.float32) / 255.0).flatten().tolist())
            v = np.array(feats, dtype=np.float32)
            if v.size < self.embedding_dim:
                v = np.concatenate([v, np.zeros(self.embedding_dim - v.size, dtype=np.float32)])
            else:
                v = v[: self.embedding_dim]
            out[i] = l2_normalize(v)
        return out


def make_default(mode: Optional[RuntimeMode] = None) -> TransReIDAdapter:
    return TransReIDAdapter(
        config=ReIDConfig(
            name="transreid",
            embedding_dim=768,
            qdrant_collection="person_reid_transreid",
        ),
        mode=mode,
    )
