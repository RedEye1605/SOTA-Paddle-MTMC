"""Regression tests for low-risk targeted improvements (FixReports/26).

These tests cover the three targeted improvements applied in Phase 12:

1. ``MultiCameraRunner._run_worker``: drop_newest policy now also
   records the second drop when the eviction+retry path fails.
2. ``scripts.benchmark_t4._write_reports``: now writes via
   ``*.tmp`` + ``os.replace`` so a SIGKILL mid-write does not
   leave a truncated JSON / Markdown report.
3. ``scripts.readiness_preflight._check_infra_env``: the required
   env vars now include ``QDRANT_HOST``.
"""

from __future__ import annotations

import json
from queue import Full
from pathlib import Path
from unittest.mock import MagicMock


# ----------------------------------------------------------------------------
# 1. drop_newest second-drop accounting regression
# ----------------------------------------------------------------------------


def test_drop_newest_records_second_drop_when_retry_also_fails(monkeypatch) -> None:
    """Simulate a fully-saturated queue where put_nowait fails BOTH
    on the first attempt AND on the post-eviction retry.  The
    previous code observed only one drop; the fix now observes two.
    """
    from app.workers.multi_camera_runner import MultiCameraRunner, CameraSource

    # Build a runner but don't start it — we drive _run_worker directly.
    runner = MultiCameraRunner(
        [CameraSource("CAM_FAKE", "stub://x", 8, 8, 5)],
        smoke_test_mode=True,
        drop_policy="drop_newest",
        frame_queue_maxsize=1,
    )

    # A queue that always raises Full on put_nowait.
    class _AlwaysFullQueue:
        def qsize(self) -> int:
            return 1

        def put_nowait(self, _item) -> None:
            raise Full()

        def get_nowait(self) -> None:
            # Eviction "succeeds" but the next put still fails.
            return None

        def put(self, _item, timeout=None) -> None:  # noqa: ARG002
            raise Full()

    q = _AlwaysFullQueue()

    # Single-frame "worker" that emits one FrameResult.
    class _StubResult:
        camera_id = "CAM_FAKE"
        ts = 0.0
        frame = None
        detections: list = []

    class _StubWorker:
        def run(self):
            yield _StubResult()
            yield _StubResult()

    drop_counter = {"n": 0}

    class _StubMetrics:
        def set_status(self, *_a, **_kw) -> None:
            pass

        def observe_frame_latency(self, *_a, **_kw) -> None:
            pass

        def observe_frame(self, *_a, **_kw) -> None:
            pass

        def observe_queue_depth(self, *_a, **_kw) -> None:
            pass

        def observe_drop(self) -> None:
            drop_counter["n"] += 1

    monkeypatch.setattr(
        "app.workers.multi_camera_runner.PER_CAMERA",
        MagicMock(for_camera=lambda _c: _StubMetrics()),
    )

    runner._run_worker(_StubWorker(), q, "CAM_FAKE")

    # 2 frames * 2 drops each (initial + retry) = 4 observed drops.
    assert drop_counter["n"] == 4


# ----------------------------------------------------------------------------
# 2. Atomic benchmark report write
# ----------------------------------------------------------------------------


def test_write_reports_is_atomic_via_tmp_and_replace(tmp_path: Path) -> None:
    """The JSON/Markdown reports must be written via a ``*.tmp``
    sibling and ``os.replace``.  A truncated ``*.tmp`` left behind
    by a crashed previous run must not corrupt the final file.
    """
    from scripts.benchmark_t4 import _write_reports

    report = {
        "mode": "smoke_benchmark",
        "started_at": "20260101T000000Z",
        "dataset_name": "atomic_write_test",
        "cameras": ["CAM_A"],
        "duration_seconds": 1.0,
        "total_analytics_fps": 0.0,
        "per_camera_analytics_fps": {"CAM_A_fps": 0.0},
    }

    # Pre-seed a stale *.tmp file from a "previous crash" — the new
    # write must overwrite it cleanly.
    stale_tmp = tmp_path / "benchmark_20260101T000000Z.json.tmp"
    stale_tmp.write_text("THIS IS GARBAGE FROM A PRIOR CRASH")

    _write_reports(report, tmp_path)

    final_json = tmp_path / "benchmark_20260101T000000Z.json"
    final_md = tmp_path / "benchmark_20260101T000000Z.md"
    assert final_json.exists()
    assert final_md.exists()

    # The *.tmp must NOT exist after a successful write — it was
    # renamed via os.replace.
    assert not (tmp_path / "benchmark_20260101T000000Z.json.tmp").exists()
    assert not (tmp_path / "benchmark_20260101T000000Z.md.tmp").exists()

    # The final file is well-formed JSON.
    loaded = json.loads(final_json.read_text())
    assert loaded["mode"] == "smoke_benchmark"
    assert loaded["dataset_name"] == "atomic_write_test"


# ----------------------------------------------------------------------------
# 3. Preflight required env vars now include QDRANT_HOST
# ----------------------------------------------------------------------------


def test_preflight_infra_env_requires_qdrant_host(monkeypatch) -> None:
    from scripts.readiness_preflight import _check_infra_env

    monkeypatch.setenv("POSTGRES_HOST", "relation-store")
    monkeypatch.setenv("POSTGRES_USER", "yamaha")
    monkeypatch.setenv("POSTGRES_PASSWORD", "real_pw_2026")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "dev_minio_access_key_2026")
    monkeypatch.setenv("MINIO_SECRET_KEY", "dev_minio_secret_key_2026")
    monkeypatch.setenv("REDIS_HOST", "message-bus")
    # QDRANT_HOST deliberately unset.
    monkeypatch.delenv("QDRANT_HOST", raising=False)

    out = _check_infra_env()
    assert out["ok"] is False
    assert "QDRANT_HOST" in out["reason"]


def test_preflight_infra_env_passes_with_qdrant_host(monkeypatch) -> None:
    from scripts.readiness_preflight import _check_infra_env

    monkeypatch.setenv("POSTGRES_HOST", "relation-store")
    monkeypatch.setenv("POSTGRES_USER", "yamaha")
    monkeypatch.setenv("POSTGRES_PASSWORD", "real_pw_2026")
    monkeypatch.setenv("QDRANT_HOST", "vector-store")
    monkeypatch.setenv("MINIO_ENDPOINT", "minio.example.invalid")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "dev_minio_access_key_2026")
    monkeypatch.setenv("MINIO_SECRET_KEY", "dev_minio_secret_key_2026")
    monkeypatch.setenv("REDIS_HOST", "message-bus")

    out = _check_infra_env()
    assert out["ok"] is True


# ----------------------------------------------------------------------------
# 4. Hard-rule enforcement: production_benchmark requires real metrics
# ----------------------------------------------------------------------------


def test_promotion_gate_rejects_production_benchmark_without_real_metrics() -> None:
    """Task-spec hard rule #8:

        Do not claim READY_FOR_LIMITED_PRODUCTION unless real model
        + real recorded multi-camera benchmark actually pass.

    The promotion gate must refuse a production_benchmark report
    that omits the accuracy metrics, even if FPS and GPU pass.
    Without this, a benchmark that ran without label-checking
    silently promotes to LIMITED_PRODUCTION.
    """
    from app.improvement.promotion_gate import GateThresholds, PromotionGate

    bench_without_metrics = {
        "mode": "production_benchmark",
        "cameras": ["CAM_01", "CAM_02"],
        "duration_seconds": 30.0,
        "total_analytics_fps": 1700.0,
        "per_camera_analytics_fps": {"CAM_01_fps": 121.6, "CAM_02_fps": 1585.1},
        "gpu_memory_used_mb_max": 224.0,
        # NO false_merge_rate, cross_camera_match_accuracy, id_fragmentation_rate
    }
    result = PromotionGate(GateThresholds()).check(bench_without_metrics)
    assert result.passed is False
    assert any("missing required real-model metrics" in f for f in result.failures)


def test_promotion_gate_accepts_production_benchmark_with_real_metrics() -> None:
    from app.improvement.promotion_gate import GateThresholds, PromotionGate

    bench_with_metrics = {
        "mode": "production_benchmark",
        "cameras": ["CAM_01", "CAM_02"],
        "duration_seconds": 1800.0,
        "total_analytics_fps": 12.0,
        "per_camera_analytics_fps": {"CAM_01_fps": 6.0, "CAM_02_fps": 6.0},
        "gpu_memory_used_mb_max": 8000.0,
        "false_merge_rate": 0.01,
        "cross_camera_match_accuracy": 0.93,
        "id_fragmentation_rate": 0.08,
    }
    result = PromotionGate(GateThresholds()).check(bench_with_metrics)
    assert result.passed is True
    assert result.failures == []


def test_promotion_gate_smoke_benchmark_not_required_to_have_real_metrics() -> None:
    """A smoke_benchmark report (mode != 'production_benchmark') is
    not subject to the require_real_metrics check — it can be
    missing the accuracy keys and the gate still passes (caller
    then caps the verdict at READY_FOR_SHADOW_TEST).
    """
    from app.improvement.promotion_gate import GateThresholds, PromotionGate

    bench = {
        "mode": "smoke_benchmark",
        "cameras": ["CAM_01"],
        "duration_seconds": 10.0,
        "per_camera_analytics_fps": {"CAM_01_fps": 100.0},
        # No accuracy metrics — that's OK for smoke_benchmark.
    }
    result = PromotionGate(GateThresholds()).check(bench)
    assert result.passed is True


def test_promotion_gate_require_real_metrics_can_be_disabled() -> None:
    """Operators who deliberately want to test the perf path
    without real accuracy labels can opt out — but the default
    is strict."""
    from app.improvement.promotion_gate import GateThresholds, PromotionGate

    bench = {
        "mode": "production_benchmark",
        "cameras": ["CAM_01"],
        "duration_seconds": 30.0,
        "per_camera_analytics_fps": {"CAM_01_fps": 100.0},
        # No accuracy metrics.
    }
    strict = PromotionGate(GateThresholds(require_real_metrics=True)).check(bench)
    assert strict.passed is False

    relaxed = PromotionGate(GateThresholds(require_real_metrics=False)).check(bench)
    assert relaxed.passed is True
