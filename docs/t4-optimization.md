# T4 Optimization

## Hardware constraints

- 1 × NVIDIA T4 (16 GB VRAM, 2.5 TFLOPS FP32, 5 TFLOPS FP16 with Tensor Cores)
- 8 CPU cores
- 32 GB RAM

## Default config

```yaml
runtime:
  device: gpu
  gpu_id: 0
  run_mode: trt_fp16       # TensorRT FP16
  visual: false             # no overlay in production
  cpu_threads: 8

streams:
  mode: multi_rtsp
  max_initial_cameras: 2    # start with 2, scale to 4 after benchmark
  target_analytics_fps: 5
  skip_frame_num: 2         # process every 3rd frame
```

## Hard rules (enforced by architecture-guard tests)

1. One model instance shared across all cameras.
2. `run_mode=trt_fp16` for the detector (mandatory).
3. ReID only on stable tracklets (min track age 10 frames).
4. No per-frame ReID.
5. No CLIP-ReID by default.
6. No SAHI by default.

## Batching strategy

- **Detector**: batch size = 1 (RTSP streams are 25–30 FPS each; a single T4
  comfortably runs 4 cameras at 5 FPS analytics). Batched inference requires
  synchronized batches across cameras which adds latency.
- **ReID**: batch 16–32 crops per forward pass (cosine similarity is parallelized
  on GPU; this saturates the SMs without exhausting VRAM).
- **Resolver**: pure CPU + Qdrant + Redis — no GPU work.
- **Telemetry**: pure CPU — no GPU work.

## VRAM budget

| Component | VRAM |
|---|---|
| PP-Human detector (TensorRT FP16) | ~2.5 GB |
| PP-Human ReID (256-d) | ~0.4 GB |
| TransReID (768-d, ViT-Base) | ~1.1 GB |
| CUDA context + driver overhead | ~0.5 GB |
| Headroom | ~11 GB |
| **Total (default: PP-Human + ReID)** | **~3.5 GB used, 12.5 GB free** |

This leaves ample headroom for a second ReID model (e.g. running both PP-Human
and TransReID in parallel during a benchmark) or a CLIP-ReID comparison run.

## Performance logging

`app/telemetry/metrics.py` exposes a Prometheus-compatible `/metrics` endpoint.
Required metrics:

- `analytics_fps_per_camera{camera_id}` — gauge
- `gpu_memory_used_bytes` — gauge
- `qdrant_query_latency_seconds` — histogram
- `postgres_write_latency_seconds` — histogram
- `reid_extractions_total{model_name}` — counter
- `identity_decisions_total{decision_type}` — counter
- `tracklet_buffer_size{camera_id}` — gauge
- `stream_backlog{stream_name}` — gauge

## Backpressure

If the `stream:tracklets` backlog exceeds 5,000 entries, the collector pauses
new tracking until the ReID worker drains the queue. If Qdrant p99 latency
exceeds 200 ms, the resolver logs a warning and falls back to a relaxed search
(more results, lower quality threshold for the candidate pool).
