# Improvement Loop Plan — SOTA-Paddle-MTMC

> **Phase 10 — Improvement loop design.** A practical loop that
> takes the system from "structurally complete" to "self-tuning
> in production", with human-in-the-loop for ambiguous
> decisions.

## Loop architecture

```
Live system
    │  (frame stream → tracklets → decisions → telemetry)
    ▼
Evidence sampler ── privacy-safe crop retention (TTL-bounded)
    │
    ▼
Labeled benchmark set (held-out, not auto-trained)
    │
    ▼
Offline evaluator (re-runs the same resolver logic against
ground-truth, reports metrics)
    │
    ▼
Metrics report
    │
    ▼
Threshold / model candidate
    │
    ▼
Shadow-mode deployment (real pipeline, but decisions are
written to a *shadow* table, not the live `global_identities`)
    │
    ▼
A/B comparison (live vs shadow)
    │
    ▼
Promotion OR rollback (deployment gate)
    │
    ▼
Human-in-the-loop review for ambiguous merges
```

## Component 1: Data capture

### Evidence sampler

- Every Nth tracklet's `best_crop_uri` is sampled (default
  N=50, configurable per camera).
- Sampled crops are re-keyed to a labeler bucket in MinIO
  (e.g. `s3://labeler-bucket/{site_id}/{camera_id}/{yyyy}/{mm}/{dd}/{tracklet_id}.jpg`).
- The labeler bucket is *separate* from the production
  evidence bucket. Different retention (90 days vs 7 days
  for production evidence).
- An `evidence_sampler` row is written to PG:
  `evidence_sampler(tracklet_id, bucket, key, sampled_at,
  label_status)`.

### Privacy-safe retention

- Sampler crops are deleted after 90 days
  (configurable). A nightly job enforces.
- The `evidence` (production) bucket is deleted after
  7 days (configurable).
- Qdrant points have no automatic TTL; the resolver's
  `timestamp_gte` filter naturally excludes old points
  from search. A separate cleanup job periodically calls
  `qdrant.delete(…, points_selector=Filter(must=[Range("timestamp", lt=now-N_days)]))`.

## Component 2: Ground-truth labeling workflow

### Labeler tool

- A simple web UI (out of scope here; documented in
  `Docs/labeler_design.md` to be written).
- For each sampled (camera, tracklet) pair, the operator
  assigns:
  - The "true" `global_id` (or "new person" if unknown).
  - The cross-camera link: which `global_id` this tracklet
    is the same person as, on other cameras.
  - Optional: bounding-box correction, occlusion flag.

### Labeler data model

```sql
CREATE TABLE labeler_assignments (
    assignment_id    BIGSERIAL PRIMARY KEY,
    tracklet_id      TEXT NOT NULL REFERENCES tracklets(tracklet_id),
    labeler_global_id TEXT NOT NULL,           -- "LBL-..." (not the real GID)
    is_new_person    BOOLEAN NOT NULL DEFAULT FALSE,
    cross_camera_links JSONB,                 -- [tracklet_id, …] on other cameras
    confidence       DOUBLE PRECISION,         -- labeler confidence 0..1
    labeler_user     TEXT NOT NULL,
    labeled_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### Labeler workflow integration

- The `global_identity_resolver` writes to
  `identity_decisions.assigned_global_id`.
- The labeler writes to `labeler_assignments.labeler_global_id`.
- A nightly job computes the
  `live_decision ↔ labeler_groundtruth` join and produces
  metrics.

## Component 3: Benchmark dataset creation

- For each site, the operator picks:
  - 1 "easy" hour (normal traffic).
  - 1 "hard" hour (peak hours, occlusion).
  - 1 "long" 24 h period (rare re-identification).
- These are stored as a frozen manifest
  `benchmark_manifest.json` with a SHA-256 hash.
- The benchmark dataset is *immutable* during a model
  release cycle.

## Component 4: Metrics

### Detection (per camera)

- `person_recall = TP / (TP + FN)` — at IoU ≥ 0.5 vs
  ground-truth bboxes.
- `person_precision = TP / (TP + FP)`.
- `small_person_recall` — same, restricted to
  `bbox.height < 80 px`.
- `false_positive_per_hour` — total FP / wall-clock hours.

### Single-camera tracking

- `id_switches_per_tracklet` — count of times a
  `local_track_id` changes mid-track.
- `track_fragmentation` — number of local tracks per
  ground-truth person.
- `track_purity` — fraction of frames in a local track
  belonging to the most-common ground-truth person.
- `local_id_stability` — `1 - (id_switches / frames)`.

### Cross-camera ReID

- `cross_camera_match_accuracy` — fraction of true
  cross-camera merges that the resolver did correctly.
- `false_merge_rate` — fraction of "assigned_global_id" decisions
  that the labeler says are wrong. **Critical: this is the
  one the deployment gate watches.**
- `id_fragmentation_rate` — fraction of ground-truth
  persons who have > 1 `global_id`. Inverse of the
  match-recall.
- `ambiguous_decision_rate` — fraction of decisions
  with `decision_type='ambiguous'`.
- `top_1_accuracy` — fraction of labeler-matched
  tracklets where the resolver's top-1 was correct.
- `top_5_accuracy` — same with top-5.

### Operations

- `per_camera_analytics_fps` — gauge (per camera).
- `total_analytics_fps` — gauge.
- `gpu_memory_used_mb` — gauge.
- `cpu_usage_percent` — gauge.
- `qdrant_query_latency_p50_p95_p99_ms` — histogram.
- `postgres_write_latency_p50_p95_p99_ms` — histogram.
- `redis_stream_backlog{stream_name}` — gauge.
- `minio_upload_latency_p99_ms` — histogram.
- `mqtt_publish_failures_total` — counter.
- `rtsp_reconnect_total` — counter.

## Component 5: Threshold tuning

- Per `Docs/reid_threshold_tuning.md`, the deployment
  starts with:
  - `auto_match_threshold = 0.82`
  - `candidate_threshold = 0.72`
  - `ambiguous_margin = 0.04`
  - `weights = {reid: 0.55, temporal: 0.20, camera: 0.15, quality: 0.05, zone: 0.05}`
- After 1 week of labeler data, the offline evaluator
  plots:
  - `final_score` histogram, split by `is_match=true/false`.
  - ROC curve.
  - The "knee" is selected as the new `auto_match_threshold`.
- The new value is staged to shadow mode, not auto-promoted.

## Component 6: ReID model comparison

- Run the existing benchmark `scripts/benchmark_t4.py`
  (after it's been implemented; currently a stub) with:
  - `pphuman_strongbaseline` (256-d, Paddle)
  - `transreid` (768-d, default)
  - `transreid + JPM` (5*768-d concat, optional)
  - `clipreid` (512-d, off by default)
- For each model, the offline evaluator reports
  `cross_camera_match_accuracy`, `false_merge_rate`,
  `id_fragmentation_rate`, and `top_1/5_accuracy`.
- The chosen model is the one with the best
  `false_merge_rate` (i.e. lowest false-merge) at
  acceptable `id_fragmentation_rate ≤ 0.20` and
  `fps ≥ target_fps`.

## Component 7: Detector comparison

- The system uses Paddle PP-Human
  (`mot_ppyoloe_l_36e_pipeline`) as the only detector.
  No comparison is currently planned.
- For future, the comparison set is:
  - `mot_ppyoloe_l_36e_pipeline` (lightweight, 31.4 FPS)
  - `mot_ppyoloe_l_36e_pipeline` high-precision variant
  - RF-DETR (only for the `Service/` comparison, not for
    SOTA-Paddle-MTMC's own pipeline).

## Component 8: Multi-camera topology calibration

- The `camera_links` table is initially seeded from a
  human survey ("CAM_01 → CAM_02 is reachable in 10-90 s").
- After 1 month, the offline evaluator computes
  `min/max_travel_seconds` from observed `global_id`
  transitions:
  - For each `global_id` that was seen on multiple
    cameras, log the elapsed time between the last
    sighting on `from_cam` and the first sighting on
    `to_cam`.
  - The empirical `min = p05`, `max = p95` (5th-95th
    percentile) of the observed distribution becomes
    the new `min_travel_seconds` and `max_travel_seconds`.
- Discrepancies between the human survey and the
  empirical distribution are flagged for review.

## Component 9: Regression tests

- Every promotion candidate must pass:
  - The 68 existing tests.
  - The 15 new tests in `TEST_QUALITY_AUDIT.md`.
  - The benchmark on the frozen `benchmark_manifest.json`.
- A regression is recorded if:
  - `false_merge_rate` increases by > 0.01 absolute.
  - `id_fragmentation_rate` increases by > 0.05 absolute.
  - `fps` drops below the target.
  - `gpu_memory_used_mb` increases by > 1 GB.

## Component 10: Deployment gates

A promotion candidate (new model, new threshold, new
config) is rejected if:

1. `false_merge_rate` > 0.05 (or > baseline + 0.01)
2. `cross_camera_match_accuracy` < 0.85 (or < baseline - 0.02)
3. `small_person_recall` < 0.60 (or < baseline - 0.05)
4. `fps < 5.0` per camera at peak load
5. `gpu_memory_used_mb` > 12 GB (T4 budget)
6. `qdrant_query_latency_p99_ms` > 200 ms
7. `postgres_write_latency_p99_ms` > 50 ms
8. `ambiguous_auto_merge_rate` > 0 (the system should
   never auto-merge ambiguous decisions; this is a
   binary invariant)

The gate is a Python script in `scripts/deploy_gate.py`
that reads the `benchmark.json` and exits non-zero if
any condition fails.

## Component 11: Rollback plan

- Every model version is registered in the
  `model_versions` table (already in `001_init.sql`).
- `pg.upsert_model_version()` is called on every
  promotion.
- Rollback is a single SQL:
  ```sql
  UPDATE model_versions
     SET is_active=FALSE, deactivated_at=now()
   WHERE model_id = (SELECT model_id FROM model_versions
                     WHERE is_active=TRUE
                       AND task='reid'
                     LIMIT 1);
  ```
- The `MultiCameraRunner` reloads on SIGHUP; the new
  active model is loaded.

## Component 12: Human-in-the-loop for ambiguous decisions

- "Ambiguous" decisions are stored in
  `identity_decisions.decision_type='ambiguous'`.
- An API endpoint `GET /identity/ambiguous?limit=50`
  returns the most recent ambiguous decisions.
- A future UI (out of scope) lists them and lets an
  operator click "merge" or "split".
- The merge action writes:
  ```sql
  INSERT INTO identity_merge_audit (old_global_id, new_global_id, operator, reason, score)
  VALUES (?, ?, 'operator:<username>', 'manual_review', ?);
  ```
- The split action creates a new `global_id` and writes
  the same `identity_merge_audit` row.
- **Original decision evidence is never deleted.** The
  audit table preserves the full decision history.

## Operational cadence

| Cadence | Activity |
|---|---|
| Real-time | Live system writes `identity_decisions`, `tracklet_embeddings`, `zone_events`. |
| Daily | Review `ambiguous_decision_rate`; if > 0.10, run the threshold-tuning report. |
| Weekly | Run the offline evaluator on the latest labeler batch. Update `benchmark.json`. |
| Monthly | Re-calibrate `camera_links` from observed transitions. |
| Quarterly | Re-bake `benchmark_manifest.json` with new labeler data. |
| Per-promotion | Run `scripts/deploy_gate.py`; gate must pass. |

## Documentation artifacts (to be written)

- `Docs/labeler_design.md` — UI for the labeler.
- `Docs/benchmark_methodology.md` — how to compute the
  22 metrics above from PG + Qdrant.
- `Docs/calibration_playbook.md` — how to update
  thresholds, weights, and camera_links.
- `scripts/calibrate_thresholds.py` — implements
  Component 5.
- `scripts/calibrate_topology.py` — implements
  Component 8.
- `scripts/deploy_gate.py` — implements Component 10.
- `scripts/retention_worker.py` — nightly PG + Qdrant +
  MinIO retention.

## Improvement loop verdict

**Design: complete.** Every component from the audit brief
is mapped to a specific artifact, a specific SQL, or a
specific script.

**Implementation: 0/12.** The loop is currently a paper
design. The labeler, the offline evaluator, the
threshold-tuning script, the topology-calibration script,
the deploy gate, the retention worker — all are
"documented but not written".

To get to a self-tuning system, the team must implement
at minimum: 1 (evidence sampler), 4 (metrics), 6 (model
comparison), 10 (deployment gate), 12 (HITL ambiguous
queue).

The other components (labeler UI, topology calibration,
retention) are important but can be implemented in later
phases.
