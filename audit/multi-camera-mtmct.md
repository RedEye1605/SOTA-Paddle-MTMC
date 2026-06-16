# Multi-Camera MTMCT Audit — SOTA-Paddle-MTMC

> **Phase 4 — Multi-camera MTMCT audit.** Verifies that the
> implementation is truly multi-camera, that the local→global
> hierarchy is enforced, and that cross-camera matching uses
> topology correctly.

## Confirmed behavior (verified in source)

| # | Behavior | Evidence |
|---|---|---|
| 1 | Local tracker state is per-camera (each `PPHumanWorker` has its own `_tracks` dict and `_next_local_id` counter). | `app/workers/pphuman_worker.py:74-82` |
| 2 | Two cameras can have `local_track_id=1` simultaneously without collision. | `tests/test_pphuman_worker.py:43-58` |
| 3 | `local_track_id` is camera-local and ephemeral; it never becomes `global_id` directly. | `app/identity/resolver.py:223-232` (resolver is the only `mint_global_id` caller); `tests/test_multi_camera_identity.py` |
| 4 | `global_id` is minted as `GID-{8hex}-{cam_short}`. The `8hex` is a UUID, so collisions are statistically impossible. | `app/identity/session.py:12-17` |
| 5 | `camera_links` is loaded once at startup into `CameraTopology`; same instance is shared across all resolver calls. | `app/main.py:108-110`, `app/identity/camera_topology.py:32-74` |
| 6 | Disabled `camera_links` (e.g. CAM_01→CAM_04) is a hard block. | `app/identity/ambiguity.py:57-58`; `tests/test_ambiguity.py:38-44` |
| 7 | Enabled `camera_links` does not auto-merge by itself; the resolver still requires 5-factor score. | `app/identity/resolver.py:185-204` |
| 8 | Qdrant collection is per-model, not per-camera. All cameras share the same `person_reid_{model}` collection. | `app/storage/qdrant_store.py:24-28` |
| 9 | PostgreSQL tracks `tracklets(camera_id, local_track_id)` with FK to `cameras(camera_id)`. | `db/migrations/002_identity_tables.sql:28-43` |
| 10 | Each camera has its own active local-track binding in Redis (`active:{cam_id}:{local_id}`). | `app/storage/redis_state.py:67-90` |
| 11 | Telemetry publishes per-camera payloads via `build_global_count_payload(camera_id=…)`. | `app/telemetry/thingsboard_payload.py:7-29` |
| 12 | `Redis Streams` use consumer groups so multiple ReID workers can scale horizontally. | `app/storage/redis_state.py:139-144` |

## Failed / partial behavior (with evidence)

| # | Behavior | Evidence |
|---|---|---|
| F1 | **The "one model instance shared across all cameras" rule is not enforced.** `MultiCameraRunner.__init__` does not accept a `model` argument; each `PPHumanWorker` is constructed without one. The README's claim and the test_architecture_guards claim are unverified. | `app/workers/multi_camera_runner.py:79-94` |
| F2 | **Cross-camera retrieval doesn't filter by travel-time window.** The Qdrant search filter uses `timestamp_gte = ts - 86400` (24h), but topology says `min_travel_seconds=10, max_travel_seconds=90`. A 23h-old CAM_01 candidate for a CAM_02 tracklet is still returned. | `app/identity/resolver.py:111-119` |
| F3 | **The 5-factor `final_score` is computed but not used in the decision.** `decide_ambiguity` only checks `top1.score` (ReID cosine) and the margin. Topology is a pre-check; the weighted `final_score` is essentially advisory. | `app/identity/resolver.py:185-204` vs `app/identity/ambiguity.py:34-71` |
| F4 | **The 24h fallback (Stage 3) is not implemented.** `_candidate_cameras()` only returns Stage 1 (same-cam) and Stage 2 (topology-linked). The "Stage 3: 24 h fallback" described in the docstring and `Docs/architecture.md` is missing. | `app/identity/resolver.py:79-99` |
| F5 | **Ambiguous decisions do not produce a stored ambiguity record.** When `decide_ambiguity` returns `"ambiguous"`, the resolver writes a row to `identity_decisions` but creates no `global_id` and does not enqueue anything for an operator to review later. The "ambiguous" decision is silently dropped — the tracklet effectively has no global_id until it re-arrives. | `app/identity/resolver.py:219-223` |
| F6 | **The `stream:zone_events` and `stream:identity_decisions` consumers have no publishers.** `TelemetryWorker` reads from these streams but nothing publishes. Dwell sessions and identity-decision telemetry are dead. | `app/workers/telemetry_worker.py:78-89` |
| F7 | **Camera-level rate limiting is absent.** A fast camera floods the collector; a slow camera starves. No FPS target per camera. | `app/workers/multi_camera_runner.py:101-114` |
| F8 | **No GPU memory accounting across cameras.** `gpu_memory_used_mb()` is a global `nvidia-smi` read; the runner does not know per-model VRAM. If Paddle + TransReID + CLIP-ReID were ever all loaded, OOM is unguarded. | `app/utils/gpu.py:15-33` |
| F9 | **Per-camera QoS / back-pressure is absent.** The `Queue(maxsize=64)` per worker means a fast producer can backpressure, but if the collector is slow, items are silently dropped. There is no `Queue.full()` check or dead-letter. | `app/workers/multi_camera_runner.py:87` |
| F10 | **No reconnect logic for RTSP.** `cv2.VideoCapture` on an RTSP stream that drops will return `ok=False` once, then the loop breaks. The worker dies and the camera is silent forever. | `app/workers/multi_camera_runner.py:38-55` |

## Tests that prove multi-camera correctness (and gaps)

| Test | What it covers | Gaps |
|---|---|---|
| `test_pphuman_worker_local_track_id_is_camera_local` (test_pphuman_worker.py:43) | Two workers have independent `local_track_id` spaces. | Does not test the reid_worker + resolver chain. |
| `test_no_topology_link_blocks_match` (test_multi_camera_identity.py:15) | Disabled `camera_links` → "new". | Does not test enabled but topology-impossible travel time. |
| `test_candidate_cameras_respect_enabled_only` (test_multi_camera_identity.py:33) | Disabled links are not in candidate set. | Does not test travel-time window. |
| `test_24h_fallback_not_in_topology` (test_multi_camera_identity.py:48) | Empty `camera_links` → empty candidates. | Misleadingly named — there is no 24h fallback test at all. |

## Test gaps — recommended additional tests

1. **Two-camera same-person test (real)**: feed the same person's
   crop to CAM_01 and CAM_02; assert the resolver returns
   `assigned_global_id` equal across both. Requires a recorded
   multi-camera video (e.g. a person walking from CAM_01 to CAM_02
   FOV).

2. **Two-camera different-people test**: two unrelated people in
   the two cameras at the same time. Assert no false merge.
   Asserts the `camera_topology_score=False → "new"` path.

3. **Impossible travel-time test**: same person in CAM_01, then in
   CAM_02 within 1 second (less than `min_travel_seconds=10`).
   Assert the resolver returns "new", not "match", even when
   cosine is 0.99.

4. **24h re-identification test**: store a tracklet's embedding
   with `timestamp = now - 23h`, mint a new tracklet at
   `timestamp = now`. Assert the resolver retrieves the old
   embedding via Qdrant, scores it, and assigns the same
   `global_id`.

5. **ID fragmentation test**: same physical person, two local
   tracklets on the same camera (because of a 2-frame gap in
   OC-SORT). Assert a single `global_id` after the resolver.

6. **Two-camera same-`local_track_id=1`**: feed each camera one
   track. Assert the resolver sees two different `tracklet_id`s
   and assigns the *same* `global_id` (if the visual match is
   strong) without conflict.

7. **Dwell-sessions test**: zone-enter on CAM_01, then
   zone-exit on CAM_01; assert one open + one closed row in
   `dwell_sessions`.

8. **Stream backlog test**: publish 10 000 fake tracklets to
   `stream:tracklets`; assert the ReID worker drains them all
   without memory blowup; assert `stream_backlog` gauge
   reports 0 at the end.

9. **Paddle real-model integration test**: load the actual
   Paddle PP-Human model and run inference on a real
   `1920x1080` video file; assert
   `REGISTRY.reid_extractions_total > 0` and that
   `extract()` returns a 256-dim vector whose first 8 values
   are not a histogram of a flat image (sanity check).

10. **TransReID real-model integration test**: same as above,
    768-dim output, JPM with 5×768 = 3840 raw features reduced
    to 768 by `neck_feat: 'before'`.

## Multi-camera verdict

The **plumbing** is multi-camera: per-camera workers, shared
resolver, shared Qdrant, per-camera Redis active bindings. The
**logic** is multi-camera: 5-factor score, camera topology,
ambiguity gating. The **production** wiring is single-camera:
the synthetic detector and histogram ReID produce effectively
random embeddings per crop.

**Verdict:** **STRUCTURALLY MULTI-CAMERA, NOT PRODUCTION-READY.**

The system can be run as multi-camera today; it will produce
*consistent* (but meaningless) global_ids. It will not produce
*correct* global_ids without real Paddle + real TransReID.
