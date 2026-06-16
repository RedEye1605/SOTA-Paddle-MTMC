"""Tests for the improvement-loop skeleton (Phase 11)."""

from __future__ import annotations


def test_evidence_sampler_should_sample_at_n() -> None:
    from app.improvement.evidence_sampler import EvidenceSampler

    # Use a fake MinioStore — the constructor takes the real one but
    # ``should_sample`` doesn't touch it.
    class _FakeMinio:
        bucket = "evidence"

    s = EvidenceSampler(source_minio=_FakeMinio(), sample_every_n=3)
    seq = [s.should_sample() for _ in range(7)]
    # Should be True exactly at indices 2, 5 (the 3rd, 6th call).
    assert seq == [False, False, True, False, False, True, False]


def test_dataset_manifest_round_trip() -> None:
    from app.improvement.dataset_manifest import (
        CameraClip,
        DatasetManifest,
    )

    m = DatasetManifest(
        name="test",
        version="1",
        created_at="2026-06-12T00:00:00Z",
        cameras=[CameraClip("CAM_01", "/tmp/x.mp4", 0.0, 100.0)],
        labels="/tmp/labels.json",
        site_id="showroom_a",
    )
    raw = m.to_json()
    parsed = DatasetManifest.from_json(raw)
    assert parsed.name == m.name
    assert len(parsed.cameras) == 1
    assert parsed.cameras[0].camera_id == "CAM_01"


def test_promotion_gate_passes_clean_metrics() -> None:
    from app.improvement.promotion_gate import PromotionGate

    gate = PromotionGate()
    report = {
        "metrics": {
            "false_merge_rate": 0.01,
            "cross_camera_match_accuracy": 0.90,
            "id_fragmentation_rate": 0.05,
            "per_camera_analytics_fps": {"CAM_01": 8.0},
            "gpu_memory_used_mb": 8000,
            "qdrant_query_latency_p99_ms": 100,
            "postgres_write_latency_p99_ms": 30,
        }
    }
    r = gate.check(report)
    assert r.passed is True, r.failures


def test_promotion_gate_fails_high_false_merge() -> None:
    from app.improvement.promotion_gate import PromotionGate

    gate = PromotionGate()
    report = {
        "metrics": {
            "false_merge_rate": 0.20,  # way over 0.05
            "cross_camera_match_accuracy": 0.90,
        }
    }
    r = gate.check(report)
    assert r.passed is False
    assert any("false_merge_rate" in f for f in r.failures)


def test_promotion_gate_fails_low_fps() -> None:
    from app.improvement.promotion_gate import PromotionGate

    gate = PromotionGate()
    report = {
        "metrics": {
            "per_camera_analytics_fps": {"CAM_01": 2.0},  # below 5.0
        }
    }
    r = gate.check(report)
    assert r.passed is False
    assert any("fps[CAM_01]" in f for f in r.failures)


def test_promotion_gate_fails_high_gpu_memory() -> None:
    from app.improvement.promotion_gate import PromotionGate

    gate = PromotionGate()
    report = {
        "metrics": {
            "gpu_memory_used_mb": 14000,  # over 12000
        }
    }
    r = gate.check(report)
    assert r.passed is False
    assert any("gpu_memory" in f for f in r.failures)


def test_offline_evaluator_runs_against_synthetic_data() -> None:
    """The offline evaluator produces the 22 metrics. We use a
    mock resolver that always returns 'match'.
    """
    from app.improvement.dataset_manifest import (
        CameraClip,
        DatasetManifest,
    )
    from app.improvement.offline_evaluator import OfflineEvaluator
    from app.identity.resolver import GlobalIdentityResolver

    class _FakeResolver(GlobalIdentityResolver):
        def __init__(self):
            # Skip the real __init__ — we override resolve() entirely.
            pass

        def resolve(self, **kwargs):
            return {
                "decision": "match",
                "assigned_global_id": kwargs["tracklet_id"].split("-")[0],
                "confidence_state": "firm",
            }

    evaluator = OfflineEvaluator(resolver=_FakeResolver())
    manifest = DatasetManifest(
        name="test",
        version="1",
        created_at="2026-06-12T00:00:00Z",
        cameras=[CameraClip("CAM_01", "/tmp/x.mp4", 0.0, 100.0)],
    )
    tracklets = [
        {"tracklet_id": f"TL-{i}", "camera_id": "CAM_01", "ts": 1.0, "embedding": [0.0] * 768}
        for i in range(3)
    ]
    ground_truth = {f"TL-{i}": f"TL-{i}".split("-")[0] for i in range(3)}
    report = evaluator.evaluate(manifest, tracklets=tracklets, ground_truth=ground_truth)
    d = report.to_dict()
    assert d["matches"] == 3
    assert d["metrics"]["cross_camera_match_accuracy"] == 1.0
    assert d["total_tracklets"] == 3
