# Model Selection

## Detector: PP-Human pedestrian tracking (official PaddleDetection)

- **Checkpoint family**: `mot_ppyoloe_l_36e_pipeline`
- **Variant (default)**: `pedestrian_tracking_lightweight`
- **Variant (fallback)**: `pedestrian_tracking_high_precision` (31.4 FPS on T4 FP16, 182 MB)
- **Why**: jointly trained detector + embedder, official Paddle model, documented
  for T4 TensorRT FP16 in the PaddleDetection README.

## Tracker: OC-SORT via Paddle pipeline

- **Config**: `deploy/pipeline/config/tracker_config.yml` from PaddleDetection
- **Why OC-SORT over BoT-SORT**: OC-SORT (CVPR 2023) handles non-linear motion
  via Observation-Centric Re-Update; better for CCTV-style crowded scenes.
- **Fallback**: DeepSORT (also in Paddle's tracker_config.yml).

## ReID (3-tier priority, configurable)

### Tier 1 (default): PP-Human StrongBaseline

- Official Paddle ReID baseline, joint training with the PP-Human detector.
- Embedding dim: 256 (verified via PaddleDetection docs).
- ~5–10 ms per crop on T4 FP16.
- **Use for**: phase 4 baseline, smoke tests, low-risk deployments.

### Tier 2 (recommended): TransReID (ICCV 2021)

- **Why TransReID**: first Transformer-based ReID with SIE (camera-aware
  embeddings) and JPM (jigsaw patch module) — natively cross-camera-aware.
- **Checkpoint**: `transformer_120.pth` on Market-1501 (ranking 1 on Market-1501
  leaderboard at time of paper release).
- **Embedding dim**: 768.
- **Use for**: production deployments where accuracy matters.
- **Loading safety**: always pass `weights_only=True` (or
  `torch.serialization.safe_globals` allowlist) — see
  `app/reid/transreid_adapter.py`.

### Tier 3 (optional benchmark): CLIP-ReID

- Off by default.
- Why optional: ~440M parameters, two-stage training, ~6 GB VRAM on T4.
- Use only when comparing ReID benchmarks, not in production.

## Decision policy

| Active model | Qdrant collection | Embedding dim | Used when |
|---|---|---|---|
| PP-Human | `person_reid_pphuman` | 256 | default, smoke tests |
| TransReID | `person_reid_transreid` | 768 | production |
| CLIP-ReID | `person_reid_clipreid_optional` | 512 | opt-in benchmark only |

A single deployment uses ONE active model. Switching models requires a backfill
job (out of scope for phase 5; covered by `scripts/benchmark_t4.py`).
