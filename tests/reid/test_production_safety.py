"""Production safety — refuse to start with fake models.

These tests verify PATCH-003, PATCH-040, and the audit's overall
"production paths are deterministic fallbacks" risk.

Hard rule:
  * In ``production`` mode (the default), the ReID adapter MUST refuse
    to load if the real model is missing.
  * In ``smoke_test`` mode, the deterministic / histogram fallback is
    allowed and the output is clearly marked as non-production.
  * The runtime mode resolver correctly maps ``SOTA_RUNTIME_MODE`` /
    ``ALLOW_SYNTHETIC_SMOKE_TEST`` to the right RuntimeMode.
"""

from __future__ import annotations


import pytest

from app.core.runtime_mode import (
    ProductionSafetyError,
    RuntimeMode,
    assert_production_safe,
    resolve_runtime_mode,
)


def test_runtime_mode_from_string_production() -> None:
    assert RuntimeMode.from_string("production") == RuntimeMode.PRODUCTION
    assert RuntimeMode.from_string("multi_rtsp") == RuntimeMode.PRODUCTION
    assert RuntimeMode.from_string("") == RuntimeMode.PRODUCTION
    assert RuntimeMode.from_string(None) == RuntimeMode.PRODUCTION


def test_runtime_mode_from_string_smoke_test() -> None:
    assert RuntimeMode.from_string("smoke_test") == RuntimeMode.SMOKE_TEST
    assert RuntimeMode.from_string("single_cam_smoke") == RuntimeMode.SMOKE_TEST
    assert RuntimeMode.from_string("smoke") == RuntimeMode.SMOKE_TEST


def test_runtime_mode_from_string_benchmark() -> None:
    assert RuntimeMode.from_string("benchmark") == RuntimeMode.BENCHMARK


def test_runtime_mode_from_string_unknown_raises() -> None:
    with pytest.raises(ValueError):
        RuntimeMode.from_string("not-a-mode")


def test_runtime_mode_allows_synthetic_only_in_smoke() -> None:
    assert RuntimeMode.SMOKE_TEST.allows_synthetic is True
    assert RuntimeMode.PRODUCTION.allows_synthetic is False
    assert RuntimeMode.BENCHMARK.allows_synthetic is False


def test_assert_production_safe_passes_in_smoke() -> None:
    # In smoke mode the assertion is a no-op.
    assert_production_safe(
        mode=RuntimeMode.SMOKE_TEST,
        component="X",
        condition="anything goes",
    )


def test_assert_production_safe_raises_in_production() -> None:
    with pytest.raises(ProductionSafetyError):
        assert_production_safe(
            mode=RuntimeMode.PRODUCTION,
            component="X",
            condition="synthetic fallback",
        )


def test_resolve_runtime_mode_default_is_production(monkeypatch) -> None:
    monkeypatch.delenv("SOTA_RUNTIME_MODE", raising=False)
    monkeypatch.delenv("ALLOW_SYNTHETIC_SMOKE_TEST", raising=False)
    assert resolve_runtime_mode() == RuntimeMode.PRODUCTION


def test_resolve_runtime_mode_from_env(monkeypatch) -> None:
    monkeypatch.setenv("SOTA_RUNTIME_MODE", "smoke_test")
    assert resolve_runtime_mode() == RuntimeMode.SMOKE_TEST


def test_resolve_runtime_mode_allow_synthetic(monkeypatch) -> None:
    monkeypatch.delenv("SOTA_RUNTIME_MODE", raising=False)
    monkeypatch.setenv("ALLOW_SYNTHETIC_SMOKE_TEST", "true")
    assert resolve_runtime_mode() == RuntimeMode.SMOKE_TEST


# --- Real adapter load tests (in-process) ---


def test_transreid_adapter_refuses_in_production_without_weights(tmp_path) -> None:
    """The TransReID adapter MUST raise ProductionSafetyError when
    runtime mode is production and the weight file is missing.
    """
    from app.reid.transreid_adapter import TransReIDAdapter
    from app.reid.base import ReIDConfig

    adapter = TransReIDAdapter(
        ReIDConfig(name="transreid", embedding_dim=768, qdrant_collection="person_reid_transreid"),
        weight_path=str(tmp_path / "missing.pth"),
        mode=RuntimeMode.PRODUCTION,
    )
    with pytest.raises(ProductionSafetyError):
        adapter.load()


def test_transreid_adapter_refuses_extract_when_unloaded_in_production() -> None:
    """A caught production load failure must not leave histogram fallback
    available through extract().
    """
    from app.reid.transreid_adapter import TransReIDAdapter
    from app.reid.base import ReIDConfig
    import numpy as np

    adapter = TransReIDAdapter(
        ReIDConfig(name="transreid", embedding_dim=768, qdrant_collection="person_reid_transreid"),
        weight_path="/tmp/does-not-matter.pth",
        mode=RuntimeMode.PRODUCTION,
    )
    with pytest.raises(ProductionSafetyError):
        adapter.extract([np.zeros((64, 32, 3), dtype=np.uint8)])


def test_transreid_adapter_falls_back_in_smoke_test(tmp_path) -> None:
    """In smoke-test mode the adapter allows the histogram fallback."""
    from app.reid.transreid_adapter import TransReIDAdapter
    from app.reid.base import ReIDConfig

    adapter = TransReIDAdapter(
        ReIDConfig(name="transreid", embedding_dim=768, qdrant_collection="person_reid_transreid"),
        weight_path=str(tmp_path / "missing.pth"),
        mode=RuntimeMode.SMOKE_TEST,
    )
    adapter.load()
    assert adapter._fallback_active is True
    import numpy as np

    out = adapter.extract([np.zeros((64, 32, 3), dtype=np.uint8)])
    assert out.shape == (1, 768)
    # The fallback emits a L2-normalized vector (norm ≈ 1).
    import numpy as np

    norm = float(np.linalg.norm(out[0]))
    assert abs(norm - 1.0) < 1e-3 or norm == 0.0


def test_pphuman_adapter_refuses_in_production_without_weights(tmp_path) -> None:
    from app.reid.pphuman_adapter import PPHumanReIDAdapter
    from app.reid.base import ReIDConfig

    adapter = PPHumanReIDAdapter(
        ReIDConfig(
            name="pphuman_strongbaseline",
            embedding_dim=256,
            qdrant_collection="person_reid_pphuman",
        ),
        weight_dir=str(tmp_path / "missing"),
        mode=RuntimeMode.PRODUCTION,
    )
    with pytest.raises(ProductionSafetyError):
        adapter.load()


def test_pphuman_worker_refuses_production_without_detector() -> None:
    """``PPHumanWorker`` refuses to run in production if no detector is
    passed. The worker is the production gate.
    """
    import numpy as np
    from app.workers.pphuman_worker import PPHumanWorker

    def _stub_reader():
        for frame_id in range(2):
            yield frame_id, 0.0, np.zeros((64, 64, 3), dtype=np.uint8)

    worker = PPHumanWorker(
        camera_id="CAM_01",
        frame_reader=_stub_reader(),
        smoke_test_mode=False,
        mode=RuntimeMode.PRODUCTION,
    )
    with pytest.raises(ProductionSafetyError):
        list(worker.run())


def test_pphuman_worker_allows_synthetic_in_smoke() -> None:
    import numpy as np
    from app.workers.pphuman_worker import PPHumanWorker

    def _stub_reader():
        for frame_id in range(2):
            yield frame_id, 0.0, np.zeros((64, 64, 3), dtype=np.uint8)

    worker = PPHumanWorker(
        camera_id="CAM_01",
        frame_reader=_stub_reader(),
        smoke_test_mode=True,  # implies SMOKE_TEST mode
        mode=RuntimeMode.SMOKE_TEST,
    )
    out = list(worker.run())
    assert len(out) == 2
