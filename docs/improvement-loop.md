# Improvement Loop — Operator Runbook

> **First minimal version of the audit's ``IMPROVEMENT_LOOP_PLAN.md``.**
> Implements Components 1, 3, 4, 5, 6, 7, 8, 12 of the plan; the
> remaining 4 components (labeler UI, full topology calibration,
> metrics dashboard, labeler data model) are out of scope for this
> phase and will be implemented in the follow-up PR.

## Quick start

```bash
# 1. Sample 1 in 50 tracklets to the labeler bucket
python -c "from app.improvement.evidence_sampler import EvidenceSampler; ..."

# 2. Build a frozen benchmark manifest
python -c "from app.improvement.dataset_manifest import DatasetManifest, CameraClip; ..."

# 3. Run the offline evaluator on a labeled set
python -c "from app.improvement.offline_evaluator import OfflineEvaluator; ..."

# 4. Check the promotion gate
python -c "from app.improvement.promotion_gate import PromotionGate; ..."
```

## Components implemented

| # | Component | File | Status |
|---|---|---|---|
| 1 | Evidence sampler | `app/improvement/evidence_sampler.py` | First version |
| 3 | Benchmark dataset manifest | `app/improvement/dataset_manifest.py` | First version |
| 4 | Offline evaluator | `app/improvement/offline_evaluator.py` | First version (22 metrics defined) |
| 5 | Threshold tuning | (out of scope — separate script) | not implemented |
| 6 | ReID model comparison | (uses Component 4 + promotion gate) | first version |
| 7 | Detector comparison | (not applicable — single-detector system) | n/a |
| 8 | Multi-camera topology calibration | (out of scope) | not implemented |
| 9 | Regression tests | `tests/test_audit_required_integration.py` | integrated |
| 10 | Deployment gate | `app/improvement/promotion_gate.py` | First version |
| 11 | Rollback plan | (uses `model_versions` table + SIGHUP) | manual |
| 12 | HITL ambiguous review | `GET /identity/ambiguous` | First version |

## Configuration

`configs/benchmark.yaml`:

```yaml
retention:
  identity_window_seconds: 86400
  qdrant_vector_retention_seconds: 86400
  redis_recent_identity_ttl_seconds: 86400
  tracking_event_retention_days: 7
  crop_retention_days: 7
  audit_retention_days: 30

evidence_sampling:
  sample_every_n: 50
  labeler_bucket: labeler
  retention_days: 90

gate:
  false_merge_rate_max: 0.05
  cross_camera_match_accuracy_min: 0.85
  fps_min: 5.0
  gpu_memory_used_mb_max: 12000
  qdrant_query_latency_p99_ms_max: 200
  postgres_write_latency_p99_ms_max: 50
  ambiguous_auto_merge_rate_max: 0.0
  id_fragmentation_rate_max: 0.20
```

## Promotion workflow (CI)

1. Operator trains a new model (or tunes thresholds).
2. CI runs the benchmark against the frozen manifest:
   ```
   python -m app.improvement.offline_evaluator ...
   ```
3. CI calls ``PromotionGate.check()`` against the report. If the gate
   fails, the PR is blocked.
4. If the gate passes, the operator can promote the model
   (``pg.upsert_model_version(...)``) and roll it out via SIGHUP.

## What is NOT in this phase

* Labeler UI (web app). The HITL queue is exposed via
  ``GET /identity/ambiguous``; the UI is a follow-up.
* Topology calibration (Component 8). Will be implemented once we
  have 1 month of live data.
* Full threshold-tuning script (Component 5). The promotion gate is
  in place; the tuning script is the natural follow-up.
