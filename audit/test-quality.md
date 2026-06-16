# Test Quality Audit — SOTA-Paddle-MTMC

> **Phase 9 — Test quality audit.** The brief says "do not only
> count tests". Each test is evaluated for what it actually
> proves, what it relies on (mock vs real), and what it leaves
> unverified.

## Headline numbers

- **68 tests** pass in 0.73 s.
- **0 tests** require a GPU.
- **0 tests** require a real Paddle model.
- **0 tests** require a real TransReID model.
- **0 tests** require a real Qdrant.
- **0 tests** require a real PostgreSQL.
- **0 tests** require a real Redis.
- **0 tests** require a real MinIO.
- **0 tests** require real RTSP or recorded video.
- **0 tests** cover multi-camera end-to-end (real pipeline).
- **0 tests** cover 24h retrieval.
- **0 tests** cover production-fallback blocking.
- **0 tests** cover the synthetic detector.
- **0 tests** cover Docker Compose startup.

## Per-file assessment

| File | Tests | What it actually proves | Mocks? | Real? | Gaps |
|---|---|---|---|---|---|
| `test_ambiguity.py` | 4 | The `decide_ambiguity` function maps (top1, top2, threshold, margin, is_known_link) to {"match", "candidate", "ambiguous", "new"} correctly. | Pure-Python, no I/O. | Real (logic is testable without infra). | Does not test that the resolver CALLS `decide_ambiguity` with the right args from the 5-factor score. |
| `test_architecture_guards.py` | 4 | (a) `Service/` dir exists. (b) No string `'Service/...'` writes in SOTA code. (c) No imports of `rfdetr/botsort/boxmot/youtureid`. (d) No secrets in repo. (e) Dockerfile has no `change_me_in_production`. (f) `.env` has no literal secrets. | File-system scan + regex. | Real. | Does NOT enforce "one model instance shared", "ReID only on stable tracklets", "no per-frame ReID", "global gallery shared", "camera_links honored". These are claimed in README but unverified. |
| `test_camera_topology.py` | 4 | `CameraTopology.is_known_link` returns `True/False/None` per the enabled flag. `is_within_travel_window` enforces min/max. `candidate_cameras_for` filters by enabled. | Pure-Python. | Real. | OK. |
| `test_config_loading.py` | 2 | `load_all_configs` reads 4 YAMLs and returns a dict-of-dicts. Default thresholds in `app.yaml` are 0.82 / 0.72 / 0.04. | Pure-Python. | Real. | Does not validate `cameras.yaml` against the schema (e.g. `width` is a positive int). Does not validate `zones.yaml` polygons are well-formed (CCW, non-self-intersecting). |
| `test_db_schema.py` | 5 | All 13 required tables exist in the SQL migrations. Cameras, global_identities, identity_decisions have the expected columns. Seed `camera_links.sample.sql` has a `FALSE` row. | File-system scan. | Real. | Does not check FK constraints. Does not check that the seed files match the YAMLs (e.g. `cameras.yaml` has 5 cameras, but `cameras.sample.sql` only has 5 — OK, but no automated cross-check). |
| `test_dwell.py` | 4 | `DwellBookkeeper.on_event` handles open/close/dup/stale correctly. | Pure-Python. | Real. | Does not test the PG upsert of dwell_sessions. Does not test the telemetry_worker → dwell pipeline. |
| `test_identity_scoring.py` | 7 | `temporal_score` is Gaussian. `camera_topology_score` is 1/0/0.5. `zone_transition_score` is 1/0.5. `weights.total() == 1`. `final_score ∈ [0, 1]`. `score_breakdown` returns all 6 components. `decide_ambiguity` matches the spec. | Pure-Python. | Real. | Does not test that the resolver actually uses `final_score` for the decision (it doesn't — see BUG-008). |
| `test_image_quality.py` | 7 | `crop_quality_score` rejects too-small/too-dark/too-bright, accepts normal. `l2_normalize` and `mean_normalized` produce unit vectors. `crop_with_padding` adds margin. `resize_keep_aspect` pads to target. | numpy. | Real. | Does not test the actual quality on real CCTV footage (laplacian variance distribution). |
| `test_multi_camera_identity.py` | 3 | Disabled link → "new". `candidate_cameras_for` filters disabled. Empty topology → no candidates. | Pure-Python. | Real. | Misleadingly named: does NOT test multi-camera end-to-end. Just tests `decide_ambiguity` + `CameraTopology`. No resolver, no Qdrant, no PG. |
| `test_pphuman_worker.py` | 3 | Worker emits `LocalTrack` from synthetic detector. Local IDs are camera-local. Frame-skip ratio is correct. | Stub frame reader, no real model. | **Mocked** — the "detector" is the synthetic one. | Does not test with a real Paddle detector. Does not test the worker against the OC-SORT config. |
| `test_qdrant_store.py` | 4 | `search()` short-circuits on empty candidates (no Qdrant call). `search()` always emits a Filter with timestamp/camera/quality/model. Collections have dim ∈ {256, 768, 512}. Payload index fields are complete. | **Mock client** (`_FakeClient`). | **Mocked** — no real Qdrant. | Does not test the real `qdrant_client` API call (e.g. `qdrant_client.models.Range(gte=…)` vs `gte=value`). |
| `test_thingsboard_payload.py` | 4 | Payload shape `{ts, values}` with global_id, zone_event, dwell. Default `ts` is in ms. | Pure-Python. | Real. | Does not test against a real MQTT broker. |
| `test_zone_assignment.py` | 5 | Point-in-polygon, centroid, parse_zones, JSON error, bbox assignment. | Pure-Python. | Real. | Does not test against real zone polygons from `zones.yaml`. |
| **TOTAL** | **68** | **Mixed.** Logic tests are real; integration tests are mocked. |  |  |  |

## Tests that should be added before production

In priority order:

### 1. Real Paddle integration test
- Mark as `slow`, gated by `pytest -m slow`.
- Load `models/pphuman/mot_ppyoloe_l_36e_pipeline` (or skip if missing).
- Run inference on a recorded 30 s video clip.
- Assert `len(detections) > 0` and `len(detections) < 200` (sanity).
- Assert `REGISTRY.reid_extractions_total > 0` after the run.

### 2. Real TransReID integration test
- Load `models/vit_transreid_msmt.pth` (correct path).
- Run `extract()` on a single 256×128 crop.
- Assert the output is 768-dim and L2-normalized.

### 3. Real Qdrant integration test
- Spin up Qdrant via `docker compose up -d vector-store`.
- `init_collections()`; assert 3 collections exist with right dims.
- Upsert 100 points; assert search returns the closest.
- Test that a `MatchAny(camera_ids=[…])` filter excludes other cameras.

### 4. Real PostgreSQL integration test
- Spin up PG via `docker compose up -d relation-store`.
- Run migrations.
- Insert a tracklet; assert roundtrip with the inserted row.
- Test the `identity_decisions` INSERT with all 18 fields.
- Test the `expire_old_identities` UPDATE with a sentinel row.

### 5. Real Redis integration test
- Spin up Redis via `docker compose up -d message-bus`.
- `set_active`, `get_active`, `mark_recent`, `mark_camera_last_seen`.
- Verify TTLs (`ttl < 60` for active, `ttl < 86400` for recent).
- `xadd` and `xreadgroup` roundtrip.

### 6. Multi-camera end-to-end test
- Recorded dataset: 2 cameras, 1 person walking from CAM_01 to CAM_02.
- Run for 60 s.
- Assert the resolver assigns the same `global_id` to both
  cameras' tracklets.
- Assert the 5-factor score is in `[0.82, 1.0]` (auto-match).

### 7. Impossible-transition test
- Recorded dataset: 1 person on CAM_01 and CAM_04 within 5 s
  (impossible per topology).
- Assert the resolver returns "new" for the second camera
  (no merge).
- Assert the `identity_decisions.reason` mentions
  `camera_topology=False`.

### 8. 24h retrieval test
- Insert a tracklet embedding with `timestamp = now - 23h` into Qdrant.
- Insert the corresponding `global_identities` row in PG.
- Insert a fresh tracklet at `now`.
- Assert the resolver retrieves the 23h-old embedding.

### 9. Production-fallback-blocking test
- Set `mode = multi_rtsp`, `_fallback_active = True`.
- Assert `main()` raises `RuntimeError` at startup.

### 10. One-model-per-process test
- Construct a `MultiCameraRunner` with 4 cameras and a
  `model=…` argument.
- Assert all 4 `PPHumanWorker`s share the same model
  instance (`id(worker.model) == id(model)`).

### 11. Synthetic-detector-blocked-in-production test
- `mode=multi_rtsp`, `detector_factory=None`.
- Assert `main()` refuses to start.

### 12. Docker Compose integration test
- `docker compose up -d` all services.
- `docker compose run --rm db-migrator` succeeds.
- `docker compose run --rm detect-pipeline python scripts/init_qdrant.py` succeeds.
- `curl localhost:8000/health` returns `{"status": "ok"}`.

### 13. Stream consumer wiring test
- Publish 100 messages to `stream:tracklets`.
- Assert `ReIDWorker` consumes all 100.
- Manually publish 100 messages to `stream:embeddings`.
- Assert a stream consumer for the resolver processes them all.
- (Today: this fails because no resolver consumer exists.)

### 14. Retention test
- Insert a `global_identities` row with `last_seen_at = now - 25h`.
- Call `expire_old_identities(86400)`.
- Assert `status = 'expired'`.

### 15. ReID-batch stress test
- Publish 10 000 tracklets in a burst.
- Assert the ReID worker's `stream_backlog` gauge returns to
  0 within 60 s.

## Test verdict

**Existing tests are honest and well-named.** They test the
*logic* of the code paths that are wired. They do NOT test
the production model paths, because those paths are not
wired.

**To claim production-readiness, the team must add at minimum
tests 1, 2, 6, 7, 8, 9, 10, 11, 13** — those are the gap
between "structurally complete" and "production-ready".

**68 passing tests is a respectable number for a
structurally-complete skeleton, but the test pyramid is
inverted**: too many logic tests at the base, almost no
integration tests at the top.
