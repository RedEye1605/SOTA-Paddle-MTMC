# Comparison with existing `Service/`

This document is the architectural diff. It is read-only — `Service/` is not modified.

## `Service/` (offline-people-counting) — what it is

- **Detector**: RF-DETR (transformer-based, SOTA accuracy for medium-to-large people)
- **Tracker**: BoT-SORT (boxmot) with internal ReID **disabled** (`with_reid=False`)
- **ReID**: YouTuReID (Tencent Youtu Lab, 768-dim ONNX, ~3 ms on T4 FP16)
- **Resolver**: per-camera `GlobalIdentityResolver` with two-threshold policy
  (`same_camera_match_threshold=0.80`, `cross_camera_match_threshold=0.55`).
  Single-cosine decision, no margin/zone/topology gating.
- **Storage**:
  - **PostgreSQL + pgvector** for embeddings (in-DB vectors)
  - **JSON** fallback when DB disabled (we forbid this in SOTA-Paddle-MTMC)
  - **MinIO** for evidence crops
  - **RAM** as the active matching cache
- **Streaming**: per-camera threads, FFmpeg → MediaMTX annotated stream
- **MQTT**: ThingsBoard `{ts, values}` format
- **Camera concurrency**: threads per camera, detector model is shared but
  `BotSortTracker` is instantiated per camera (one per thread, not per process).

## SOTA-Paddle-MTMC — what is different

| Aspect | `Service/` | SOTA-Paddle-MTMC |
|---|---|---|
| Detector | RF-DETR (proprietary fine-tune) | **PP-Human** (`mot_ppyoloe_l_36e_pipeline`) — official PaddleDetection |
| Tracker | BoT-SORT (boxmot) | **OC-SORT** (Paddle official config) when available, else DeepSORT |
| ReID (default) | YouTuReID (CVPR-W 2021 winner) | **PP-Human StrongBaseline** (Paddle) → **TransReID** (ICCV 2021) |
| ReID (optional) | — | **CLIP-ReID** (opt-in, off by default) |
| Resolver | per-camera resolver, single-threshold | **single shared global resolver** with 5-factor score |
| Identity decision | cosine ≥ threshold → match | reid+temporal+topology+quality+zone weighted + margin check |
| Cross-camera gate | none (only same/different cam threshold) | **camera_links + travel time + zone transition** |
| Vector store | pgvector (in-DB) | **Qdrant** (dedicated, HNSW + payload indexes) |
| State cache | RAM (process-local) | **Redis** (TTL'd, cross-process) |
| Queue | in-process queue | **Redis Streams** (durable, consumer groups) |
| Multi-camera topol. | none | **camera_links** table with min/max travel seconds |
| Decision audit | implicit | **identity_decisions** + **identity_merge_audit** tables |
| Tracker model | BoT-SORT has CMC (sof) | OC-SORT has native motion compensation (better for occlusions) |
| TensorRT FP16 | ✅ partial (via RF-DETR fp16 flag) | ✅ official Paddle `--run_mode=trt_fp16` |
| Single-cam mode | production | **smoke test only** |
| Frame-skip | manual | config-driven, off by default |

## Why SOTA-Paddle-MTMC is the research-backed choice

1. **PP-Human MTMCT is the official PaddleDetection multi-camera solution**
   (verified at `deploy/pipeline/docs/tutorials/pphuman_mtmct_en.md`). Using it
   gives us a maintained, documented, versioned pipeline.
2. **OC-SORT is more recent than BoT-SORT** (CVPR 2023 vs 2022) and is designed
   for non-linear motion in crowded scenes.
3. **TransReID is a Transformer ReID with SIE (camera-aware embeddings)** —
   natively cross-camera-aware. YouTuReID is CNN-only.
4. **Qdrant is purpose-built for vector search** with payload filters and HNSW.
   pgvector is general-purpose and slower on large-scale filtered search.
5. **Redis Streams + TTLs** give us a durable, observable async pipeline, not
   a hidden in-process queue.
6. **camera_links + travel time** are first-class in the schema. `Service/`
   has no concept of camera topology — cross-camera matches are made by cosine
   alone, which is the most common source of false merges in production MTMCT.

## What SOTA-Paddle-MTMC inherits from `Service/`

- ThingsBoard `{ts, values}` MQTT payload
- MinIO evidence bucket pattern (deterministic paths)
- Python 3.12 + NVIDIA CUDA
- T4 hardware target
- `visual=False` default for production

## What SOTA-Paddle-MTMC explicitly does NOT inherit

- RF-DETR (phase 1); can be added as a comparison bridge in phase 2 if needed
- BoT-SORT (phase 1)
- YouTuReID (phase 1)
- pgvector as primary vector store (we use Qdrant)
- Per-camera resolver pattern
- JSON fallback when DB is down
- RAM as the only active cache (we still have RAM caches for hot data, but
  Redis is the canonical active cache)
