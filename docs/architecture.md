# Architecture — SOTA-Paddle-MTMC

## High-level pipeline

```
RTSP streams (CAM_01 .. CAM_N)
     │
     ▼
Multi-camera runtime (single Python process, shared GPU)
     │
     ├── per-camera DecodeWorker ── FFmpeg reader
     │
     ├── per-camera TrackerWorker (one PP-Human instance, batched)
     │      - Person detection  (mot_ppyoloe_l_36e_pipeline, TensorRT FP16)
     │      - Single-camera tracking (OC-SORT, official config)
     │      - Emits: local_track_id, bbox, score, frame_id
     │
     ▼
Tracklet Collector (async, single instance)
     - waits for track to be stable (≥ min_track_age_frames)
     - collects 5–15 candidate crops
     - applies image-quality filter (height, blur, brightness, occlusion, frame cut)
     - uploads best crop to MinIO
     - emits tracklet to Redis Stream `stream:tracklets`
     │
     ▼
ReID Worker (async, single instance, GPU)
     - pulls tracklets from `stream:tracklets`
     - runs active ReID model (PP-Human StrongBaseline / TransReID / CLIP)
     - normalizes embeddings
     - upserts vectors + payload to Qdrant (`person_reid_{model}`)
     - emits embedding row to `stream:embeddings`
     │
     ▼
Global Identity Resolver (async, single instance)
     - pulls embeddings from `stream:embeddings`
     - staged Qdrant search:
         1. same-camera recent (last 60 s)
         2. linked cameras (camera_links, valid travel window)
         3. 24 h fallback (low-confidence only)
     - applies 5-factor weighted score
     - assigns / creates / holds global_id
     - persists to PostgreSQL `identity_decisions`, `global_identities`
     │
     ▼
Zone & Dwell Module
     - assigns tracklet to polygon zones
     - emits zone_events (entry/exit)
     - opens/closes dwell_sessions
     │
     ▼
Telemetry Worker
     - publishes MQTT ThingsBoard `{ts, values}` payloads
     - exposes REST API (FastAPI) for dashboard

     ▼
Optional: MediaMTX annotated stream
     - skipped in Phase 2 baseline to keep T4 cycles free
```

## Identity hierarchy (3 tiers)

| Tier | Scope | Lifetime | Source |
|---|---|---|---|
| `local_track_id` | single camera | ephemeral (~30 s) | OC-SORT/PP-Human |
| `tracklet_id` | single camera | until stable/closed | tracklet_collector |
| `global_id` | all cameras | up to 24 h | global_identity_resolver |

Hard rule: a `global_id` is NEVER created from a `local_track_id` directly.

## Storage ownership

| Store | Owns | Why |
|---|---|---|
| **PostgreSQL** | source of truth: cameras, zones, links, identities, decisions, audit | durable, auditable, JOINable |
| **Qdrant** | vectors + searchable metadata | HNSW index, payload filter push-down |
| **Redis** | active binding (cam+local→global), TTL'd cache, streams | low-latency, ephemeral |
| **MinIO** | evidence crops | S3-compatible, deterministic paths |

## Hard guarantees (architecture-guard tests)

1. No code writes into `Service/`.
2. No secrets in repo.
3. One model instance shared across all cameras (verified by guard test).
4. ReID runs only on stable tracklets (no per-frame ReID).
5. `global_id` is never assigned from `local_track_id` alone.
6. Cosine similarity alone does NOT decide identity.
7. Qdrant search ALWAYS uses payload filters.
8. Ambiguous candidates are NOT auto-merged.
9. PostgreSQL never silently falls back to JSON.
10. Global gallery is shared across cameras.
11. `camera_links` is honored for cross-camera matching.
