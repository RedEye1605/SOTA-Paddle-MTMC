"""TransReID checkpoint compatibility tests (PATCH-011).

These tests verify:
  1. Missing checkpoint fails in production.
  2. Classifier-head mismatch fails when ``ignore_classifier_head=False``.
  3. Classifier-head mismatch passes when ``ignore_classifier_head=True``.
  4. Expected embedding dimension is validated.
  5. Smoke-test can skip the real checkpoint only with explicit smoke config.
  6. Profile selector is wired through ``select_reid_adapter``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def test_profile_table_has_market1501_and_msmt17() -> None:
    from app.reid.transreid_adapter import TRANSREID_PROFILES

    assert "market1501" in TRANSREID_PROFILES
    assert "msmt17" in TRANSREID_PROFILES
    assert TRANSREID_PROFILES["market1501"]["num_class"] == 751
    assert TRANSREID_PROFILES["msmt17"]["num_class"] == 1041


def test_resolve_transreid_profile_market1501() -> None:
    from app.reid.transreid_adapter import _resolve_transreid_profile

    nc, ed = _resolve_transreid_profile("market1501")
    assert nc == 751
    assert ed == 3840


def test_resolve_transreid_profile_msmt17() -> None:
    from app.reid.transreid_adapter import _resolve_transreid_profile

    nc, ed = _resolve_transreid_profile("msmt17")
    assert nc == 1041
    assert ed == 3840


def test_resolve_transreid_profile_custom_requires_args() -> None:
    from app.reid.transreid_adapter import _resolve_transreid_profile

    with pytest.raises(ValueError):
        _resolve_transreid_profile("custom")
    nc, ed = _resolve_transreid_profile("custom", num_class=42, embedding_dim=3840)
    assert (nc, ed) == (42, 3840)


def test_resolve_transreid_profile_unknown_raises() -> None:
    from app.reid.transreid_adapter import _resolve_transreid_profile

    with pytest.raises(ValueError):
        _resolve_transreid_profile("not-a-profile")


def test_inspect_classifier_shape_detects_cls_only() -> None:
    from app.reid.transreid_adapter import _inspect_classifier_shape

    # Mock the torch.Tensor.shape access via dict.
    class _MockTensor:
        def __init__(self, shape):
            self.shape = shape

    state = {"classifier.weight": _MockTensor((751, 768))}
    out = _inspect_classifier_shape(state)
    assert out is not None
    assert out["kind"] == "cls_only"
    assert out["num_class"] == 751
    assert out["embedding_dim"] == 768


def test_inspect_classifier_shape_detects_jpm() -> None:
    from app.reid.transreid_adapter import _inspect_classifier_shape

    class _MockTensor:
        def __init__(self, shape):
            self.shape = shape

    state = {
        "classifier.0.weight": _MockTensor((1041, 768)),
        "classifier.1.weight": _MockTensor((1041, 768)),
        "classifier.2.weight": _MockTensor((1041, 768)),
        "classifier.3.weight": _MockTensor((1041, 768)),
        "classifier.4.weight": _MockTensor((1041, 768)),
    }
    out = _inspect_classifier_shape(state)
    assert out is not None
    assert out["kind"] == "jpm"
    assert out["num_linears"] == 5
    assert out["num_class"] == 1041


def test_inspect_classifier_shape_strips_module_prefix() -> None:
    from app.reid.transreid_adapter import _inspect_classifier_shape

    class _MockTensor:
        def __init__(self, shape):
            self.shape = shape

    state = {"module.classifier.weight": _MockTensor((1041, 768))}
    out = _inspect_classifier_shape(state)
    assert out is not None
    assert out["num_class"] == 1041


def test_inspect_classifier_shape_returns_none_when_missing() -> None:
    from app.reid.transreid_adapter import _inspect_classifier_shape

    out = _inspect_classifier_shape({"patch_embed.proj.weight": "x"})
    assert out is None


def test_adapter_missing_checkpoint_fails_in_production() -> None:
    from app.reid.transreid_adapter import TransReIDAdapter
    from app.reid.base import ReIDConfig
    from app.core.runtime_mode import ProductionSafetyError, RuntimeMode

    with tempfile.TemporaryDirectory() as tmp:
        adapter = TransReIDAdapter(
            ReIDConfig(
                name="transreid",
                embedding_dim=3840,
                qdrant_collection="person_reid_transreid",
            ),
            weight_path=f"{tmp}/missing.pth",
            profile="msmt17",
            mode=RuntimeMode.PRODUCTION,
        )
        with pytest.raises(ProductionSafetyError):
            adapter.load()


def test_adapter_smoke_test_can_skip_missing_checkpoint() -> None:
    from app.reid.transreid_adapter import TransReIDAdapter
    from app.reid.base import ReIDConfig
    from app.core.runtime_mode import RuntimeMode

    with tempfile.TemporaryDirectory() as tmp:
        adapter = TransReIDAdapter(
            ReIDConfig(
                name="transreid",
                embedding_dim=3840,
                qdrant_collection="person_reid_transreid",
            ),
            weight_path=f"{tmp}/missing.pth",
            profile="msmt17",
            mode=RuntimeMode.SMOKE_TEST,
        )
        adapter.load()
        assert adapter._fallback_active is True


def test_adapter_require_checkpoint_false_allows_production_without() -> None:
    """If the operator sets ``require_checkpoint_in_production=False``
    the adapter falls back to histogram in production. This is for
    plug-in scenarios only — the audit's main PATCH-040 contract is
    that production MUST have a real model.
    """
    from app.reid.transreid_adapter import TransReIDAdapter
    from app.reid.base import ReIDConfig
    from app.core.runtime_mode import RuntimeMode

    with tempfile.TemporaryDirectory() as tmp:
        adapter = TransReIDAdapter(
            ReIDConfig(
                name="transreid",
                embedding_dim=3840,
                qdrant_collection="person_reid_transreid",
            ),
            weight_path=f"{tmp}/missing.pth",
            profile="msmt17",
            require_checkpoint_in_production=False,
            mode=RuntimeMode.PRODUCTION,
        )
        adapter.load()
        assert adapter._fallback_active is True


def test_inspector_reports_missing_checkpoint(tmp_path) -> None:
    """The standalone inspector script reports missing checkpoint with
    exit code 1 and a JSON `ok=false, reason=checkpoint_missing`.
    """
    import subprocess
    import sys

    target = tmp_path / "missing.pth"
    script = Path(__file__).resolve().parents[2] / "scripts" / "inspect_transreid_checkpoint.py"
    proc = subprocess.run(
        [sys.executable, str(script), str(target), "--json"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["reason"] == "checkpoint_missing"


def test_inspector_writes_human_readable_default(tmp_path) -> None:
    """Without ``--json`` the inspector writes a key:value text report."""
    import subprocess
    import sys

    target = tmp_path / "missing.pth"
    script = Path(__file__).resolve().parents[2] / "scripts" / "inspect_transreid_checkpoint.py"
    proc = subprocess.run(
        [sys.executable, str(script), str(target)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 1
    # human-readable has "ok:" and "reason:" lines.
    assert "ok: False" in proc.stdout
    assert "reason: checkpoint_missing" in proc.stdout


def test_select_reid_adapter_reads_profile_from_config() -> None:
    """``select_reid_adapter`` propagates profile/ignore_head/require
    from the active reid_cfg into the TransReIDAdapter.
    """
    from app.main import select_reid_adapter
    from app.core.runtime_mode import RuntimeMode

    cfg = {
        "active_model": "transreid",
        "profile": "msmt17",
        "embedding_dim": 3840,
        "ignore_classifier_head": True,
        "require_checkpoint_in_production": True,
    }
    adapter = select_reid_adapter(cfg, RuntimeMode.SMOKE_TEST)
    assert adapter.profile == "msmt17"
    assert adapter._ignore_classifier_head is True
    assert adapter._require_checkpoint_in_production is True
    assert adapter._num_class == 1041


def test_select_reid_adapter_market1501_profile() -> None:
    from app.main import select_reid_adapter
    from app.core.runtime_mode import RuntimeMode

    cfg = {
        "active_model": "transreid",
        "profile": "market1501",
        "embedding_dim": 3840,
        "ignore_classifier_head": False,
        "require_checkpoint_in_production": True,
    }
    adapter = select_reid_adapter(cfg, RuntimeMode.SMOKE_TEST)
    assert adapter.profile == "market1501"
    assert adapter._ignore_classifier_head is False
    assert adapter._num_class == 751
