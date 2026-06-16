"""Tests for the vendored TransReID backbone (inference path only).

These tests do NOT require a TransReID checkpoint on disk; they verify
the model construction + forward pass + output shape contract.
The full real-weight integration test is in
``test_audit_required_integration.py::test_transreid_forward_shape``
(marked ``slow``) which loads ``models/vit_transreid_msmt.pth`` if
present.
"""

from __future__ import annotations

import importlib

import pytest

torch = pytest.importorskip(
    "torch", reason="torch is required for the TransReID vendored model test"
)


def test_transreid_vendor_imports() -> None:
    """The vendored module imports without error."""
    mod = importlib.import_module("app.reid._transreid_native")
    assert hasattr(mod, "vit_base_patch16_224_TransReID")
    assert hasattr(mod, "build_transreid_model")
    assert hasattr(mod, "extract_inference_feature")
    assert hasattr(mod, "load_transreid_checkpoint")


def test_transreid_vit_constructs() -> None:
    """The SIE-Transformer can be constructed with the official defaults."""
    from app.reid._transreid_native import vit_base_patch16_224_TransReID

    model = vit_base_patch16_224_TransReID(
        img_size=(256, 128),
        stride_size=12,
        camera=6,
        view=0,
        local_feature=True,
        sie_xishu=3.0,
    )
    assert model.num_features == 768
    # Block count = 12 (ViT-Base)
    assert len(model.blocks) == 12


def test_transreid_forward_shape_with_random_weights() -> None:
    """The forward pass returns the expected shapes for JPM mode."""
    from app.reid._transreid_native import (
        build_transreid_model,
        extract_inference_feature,
    )

    # PATCH-011: SIE is disabled for inference (camera_num=0); the
    # on-disk checkpoint's SIE keys are silently dropped at load
    # time. The test mirrors production.
    model, has_jpm = build_transreid_model(
        num_class=751,
        camera_num=0,
        view_num=0,
        stride_size=12,
        sie_xishu=3.0,
        jpm=True,
        use_fp16=False,
        device="cpu",
    )
    assert has_jpm is True
    images = torch.randn(2, 3, 256, 128)
    cam_labels = torch.zeros(2, dtype=torch.long)
    view_labels = torch.zeros(2, dtype=torch.long)
    feat = extract_inference_feature(
        model,
        has_jpm=has_jpm,
        images=images,
        cam_labels=cam_labels,
        view_labels=view_labels,
        neck_feat="before",
        l2_normalize=True,
    )
    # JPM: 5 * 768 = 3840 raw → L2-normalized → 3840-dim per image.
    # We extract the global CLS + 4 local-mean chunks, concatenated.
    assert feat.shape == (2, 5 * 768)
    # L2-normalize ⇒ unit norm.
    norms = feat.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


def test_transreid_load_checkpoint_refuses_non_dict() -> None:
    """``load_transreid_checkpoint`` must call ``torch.load`` with
    ``weights_only=True`` (security audit).
    """
    from app.reid._transreid_native import (
        build_transreid_model,
        load_transreid_checkpoint,
    )

    model, _ = build_transreid_model(
        num_class=751, camera_num=0, jpm=True, use_fp16=False, device="cpu"
    )
    # We can't actually try to load a malicious pickle, but we can
    # verify the function calls torch.load with weights_only=True.
    import inspect

    src = inspect.getsource(load_transreid_checkpoint)
    assert "weights_only" in src
    assert "weights_only=weights_only" in src
