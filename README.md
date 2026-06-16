# SOTA-Paddle-MTMC

> **Multi-camera MTMCT (Multi-Target Multi-Camera Tracking) system for
> Yamaha showrooms**, built on **PaddleDetection PP-Human** as the
> primary baseline, with **TransReID MSMT17** as the production ReID
> model.

> ⚠️ **This is a multi-camera MTMCT system. Single-camera mode is
> only a smoke test.** Global identity is shared across cameras.
> Local track IDs are temporary and camera-local. Persistent ID is
> resolved by the global identity service. Cross-camera matching uses
> **ReID + camera topology + travel time + zone transition**.

## Why this exists

`Service/` is a working, production-grade people-counting service for
Yamaha showrooms, using RF-DETR + BoT-SORT + YouTuReID. This system
exists to **evaluate an alternative, research-backed stack** with
stricter guarantees:

- **Stricter identity decisions** (5-factor weighted score, not
  cosine alone)
- **Camera topology as a first-class constraint** (impossible
  transitions are hard-blocked, not just statistically unlikely)
- **24 h persistent global identity** with staged retrieval
- **Transformer ReID (TransReID MSMT17)** as the production default
- **Qdrant for vector search** with payload filters push-down
- **Redis Streams** for durable async pipeline

See `docs/comparison-legacy-service.md` for the full diff.

## Stack

| Component | Choice | Why |
|---|---|---|
| Detector | PP-Human (`mot_ppyoloe_l_36e_pipeline`) | Official Paddle, joint det+embed, T4-friendly |
| Tracker | OC-SORT (Paddle official config) | CVPR 2023, robust to non-linear motion |
| ReID (production) | **TransReID MSMT17** | Transformer + SIE + JPM, SOTA |
| Vector store | Qdrant | HNSW + payload filters |
| State cache | Redis | TTLs + Streams |
| Durable store | PostgreSQL | source of truth |
| Evidence | MinIO | S3-compatible |
| Telemetry | MQTT (ThingsBoard) | `{ts, values}` format |
| Hardware | 1 × NVIDIA T4, 8 cores, 32 GB RAM | production target |

## Quickstart

```bash
# 1. Configure
cp .env.example .env
# Edit .env with your credentials (PostgreSQL, Qdrant, Redis,
# MinIO, MQTT broker). The detect-pipeline image is Paddle-only;
# the embedding-sidecar image (built locally) adds torch.
# MinIO points at the operator's external cluster
# (minio.example.invalid:9000) via MINIO_ENDPOINT in .env.

# 2. Bring up infra
docker compose up -d relation-store vector-store message-bus

# 3. Migrate
docker compose run --rm db-migrator

# 4. Initialize Qdrant collections + payload indexes
docker compose run --rm detect-pipeline python scripts/init_qdrant.py

# 5. (one-time) Build the embedding-sidecar image
docker compose build embedding-sidecar

# 6. Bring up the full pipeline
docker compose up -d detect-pipeline embedding-sidecar
# The detect-pipeline connects to Postgres / Qdrant / Redis /
# external MinIO. The embedding-sidecar consumes stream:tracklets
# and writes real TransReID embeddings to Qdrant.

# 7. Smoke test
docker compose exec detect-pipeline python -m app.main --mode single_cam_smoke
```

## Layout

```
SOTA-Paddle-MTMC/
├── README.md
├── Dockerfile                  # eval + sidecar image (torch-enabled)
├── docker-compose.yaml
├── .env.example
├── requirements.txt            # production runtime deps
├── pyproject.toml              # build metadata only
├── app/                        # source code
│   ├── main.py
│   ├── api/                    # FastAPI server
│   ├── cli/                    # args + config + logging
│   ├── core/                   # runtime mode + safety gates
│   ├── detection/              # PP-Human + SAHI
│   ├── identity/               # resolver + scoring + topology
│   ├── reid/                   # ReID adapters (TransReID MSMT17)
│   ├── seed/                   # YAML → Postgres seeder
│   ├── storage/                # postgres + qdrant + redis + external minio
│   ├── streaming/              # mediaMTX + overlay
│   ├── telemetry/              # mqtt + metrics
│   ├── utils/                  # crop + frame_buffer + image_quality
│   ├── workers/                # runner + workers
│   └── zones/                  # polygon + dwell
├── tests/                      # organized by subject
│   ├── conftest.py
│   ├── architecture/           # guards
│   ├── api/                    # health
│   ├── benchmark/              # benchmark + readiness
│   ├── config/                 # config loading + bucket paths
│   ├── detection/              # PP-Human + SAHI
│   ├── identity/               # resolver + scoring + topology
│   ├── integration/            # legacy compat + parity
│   ├── reid/                   # adapter tests
│   ├── storage/                # pg + qdrant
│   ├── streaming/              # mediaMTX + ffmpeg + RTSP
│   ├── telemetry/              # mqtt + metrics + thingsboard
│   ├── utils/                  # crop + frame_buffer
│   └── workers/                # runner + worker tests
├── configs/                    # runtime config (YAML)
│   ├── app.yaml
│   ├── cameras.yaml
│   ├── zones.yaml
│   ├── camera_links.yaml
│   ├── legacy/                 # legacy Service compat
│   └── pphuman/                # PaddleDetection pipeline config
├── scripts/                    # operator scripts
│   ├── init_qdrant.py
│   ├── retention_worker.py
│   ├── readiness_preflight.py
│   ├── readiness_gate.py
│   ├── inspect_transreid_checkpoint.py
│   ├── run_transreid_sidecar.py
│   ├── generate_visual_validation.py
│   ├── download_pphuman_models.sh
│   ├── download_transreid_models.sh
│   ├── benchmark_t4.py
│   ├── compare_with_service_baseline.py
│   ├── run_local_video_test.sh
│   └── run_multi_rtsp_test.sh
├── docs/                       # operator + architecture docs
├── audit/                      # pre-launch audit findings
└── history/                    # chronological change log
```

## Hard rules (enforced by tests)

1. ❌ Do not modify `Service/`.
2. ❌ Do not commit secrets.
3. ✅ One model instance is shared across all cameras.
4. ✅ ReID runs only on stable tracklets (no per-frame ReID).
5. ✅ `global_id` is never assigned from `local_track_id` alone.
6. ✅ Cosine similarity alone does NOT decide identity.
7. ✅ Qdrant search ALWAYS uses payload filters.
8. ✅ Ambiguous candidates are NOT auto-merged.
9. ✅ PostgreSQL never silently falls back to JSON.
10. ✅ Global gallery is shared across cameras.
11. ✅ `camera_links` is honored for cross-camera matching.
12. ✅ Production refuses to start with histogram / synthetic
    fallbacks (see `tests/architecture/`).

## Documentation

- `docs/architecture.md` — high-level pipeline diagram
- `docs/comparison-legacy-service.md` — vs `Service/`
- `docs/database-design.md` — PostgreSQL, Qdrant, Redis, MinIO
- `docs/t4-optimization.md` — T4-specific tuning
- `docs/reid-thresholds.md` — 5-factor score, decision policy
- `docs/mtmct-operations.md` — day-1, day-2, failure modes
- `docs/benchmark-plan.md` — benchmark scenarios
- `docs/operator-runbook.md` — operator day-to-day
- `docs/post-revert-remediation.md` — per-file change log
  (2026-06-17 audit remediation)

## Audit

The pre-launch audit findings live in `audit/` (lowercase). The
remediation that brought the system to its current production-ready
state is documented in `history/` (one file per change / fix).

## Container architecture

The system runs in two images:

1. **detect-pipeline image** (prebuilt
   `sota-paddle-mtmct:paddle33-numpy126-b2-api`):
   Paddle 3.3.1 + NumPy 1.26.4 + full detect-pipeline service
   deps. Paddle-only (no torch, no TransReID). Hardcoded in
   `docker-compose.yaml` — no rebuild required.
2. **sidecar / replay-eval image** (this repo's `Dockerfile`):
   extends the detect-pipeline image, adds torch + torchvision +
   TransReID in the `sidecar` target. Built locally by
   `docker compose build embedding-sidecar`.

Why two images: Paddle 2.6.x's `fused_conv2d_add_act_kernel` (PPYOLOE
head) was compiled against cuDNN 8.6. PyTorch 2.4+ transitively
pulls cuDNN 9.x — they cannot coexist in one venv. The runtime
separation is a safe feature split enforced by
`tests/architecture/test_runtime_separation.py`.
