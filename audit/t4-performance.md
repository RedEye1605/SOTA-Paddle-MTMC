# T4 Performance Audit — SOTA-Paddle-MTMC

> **Phase 7 — T4 performance audit.** Hardware target: 1× NVIDIA
> T4 (16 GB), 8 CPU cores, 32 GB RAM. Verifies that the
> implementation does not waste GPU cycles, does not block on
> network I/O, and exposes real performance metrics.

## Audit table

| # | Concern | Claim | Code evidence | Verdict |
|---|---|---|---|---|
| T1 | One-model-per-camera avoided | README "one model instance shared across all cameras" | `MultiCameraRunner.start()` constructs one `PPHumanWorker` per camera, no shared `model=…` argument. The "production load" is commented out. | **FAIL — not wired.** With the synthetic detector this is moot; with real Paddle, the architecture allows per-camera model load. |
| T2 | TensorRT FP16 | `app.yaml: runtime.run_mode: trt_fp16` | `app/main.py` never builds or loads a Paddle engine. `run_mode` is a config string that is read but unused. | **FAIL — text only.** |
| T3 | Visual rendering off in production | `app.yaml: visual: false` | The `PPHumanWorker.run()` yields `frame` (numpy array) but never draws on it. ✅ No `cv2.putText` or `cv2.rectangle` calls. | **PASS.** |
| T4 | ReID only on stable tracklets | `reid.run_on_stable_tracklet_only: true` | `TrackletCollector.on_frame` only collects after `tr.age_frames >= min_track_age_frames=10`. ✅ `ReIDWorker` consumes `stream:tracklets`, not the frame stream. | **PASS.** |
| T5 | Transformer ReID not per-frame | ReID worker is async on a separate thread | Yes, the `ReIDWorker` runs in a background `threading.Thread` (line 228-231 of `main.py`). | **PASS — by architecture.** |
| T6 | CLIP-ReID off by default | `app.yaml: reid.optional_benchmark.clipreid.active=false` (implied) | `CLIPReIDAdapter._fallback_active = True` is set in `__init__`. There is no path to disable the fallback. | **PASS — by config.** |
| T7 | SAHI off by default | Not mentioned in code | No `sahi` import. ✅ | **PASS by absence.** |
| T8 | Queue backpressure | `app.yaml: queues.tracklet_buffer_capacity: 5000` | The config value is never read. `MultiCameraRunner` uses `Queue(maxsize=64)` hard-coded. | **FAIL — config ignored.** |
| T9 | GPU memory logging | `app/main.py:268-272` calls `gpu_memory_used_mb()` and sets `REGISTRY.gpu_memory_used.set(mb * 1024 * 1024)` every 10 s | The `gpu_memory_used_mb()` shim calls `nvidia-smi` via subprocess (2s timeout). ✅ | **PASS — but the metric is observed every 10s, not per-frame.** |
| T10 | FPS per camera logging | `REGISTRY.analytics_fps` exists | **Never set anywhere in the code.** `MultiCameraRunner` does not measure per-camera FPS. `TrackletCollector.on_frame` does not record frame times. | **FAIL — metric exists, value is always 0.** |
| T11 | Qdrant latency logging | `REGISTRY.qdrant_query_latency` exists | Logged at DEBUG in `QdrantStore.search()`, but the histogram is never `observe()`d. | **FAIL — metric exists, value is always empty.** |
| T12 | PostgreSQL write latency logging | `REGISTRY.postgres_write_latency` exists | `timed_execute` returns the latency in seconds, but the histogram is never `observe()`d. | **FAIL — metric exists, value is always empty.** |
| T13 | Benchmark measures real runtime | `scripts/benchmark_t4.py` is supposed to measure 5 scenarios | The script is a stub: `run_scenario` records `elapsed_seconds=0` and writes `"note": "Skeleton."` | **FAIL — not implemented.** |
| T14 | Docker resources reasonable | `docker-compose.yaml` has `deploy.resources.reservations.devices: [nvidia gpu count=1]` | The `api` service has GPU reservation, but no `cpus: 8` or `memory: 32G` limit. | **PARTIAL.** |
| T15 | CPU decode bottleneck considered | The decoder is `cv2.VideoCapture` in `make_frame_reader` | `cv2.VideoCapture` with H.264 RTSP requires CPU decode (no NVDEC on T4 in container). At 5 cameras × 25 FPS = 125 FPS decode, the 8-core CPU may be saturated. **No `ffmpeg` subprocess is used despite the comment** (`"Production should use FFmpeg for RTSP streams (more robust reconnection)"`). | **FAIL — CPU decode bottleneck, no FFmpeg fallback.** |

## FPS / FPS-based decisions

- The runner does NOT enforce a per-camera FPS target.
  `CameraSource.fps_target` is stored but never used to
  rate-limit the worker.
- `target_analytics_fps: 5` in `app.yaml` is never enforced.
- The ReID worker `consume(count=4, block_ms=1000)` polls
  every 1s — at peak load (4 tracklets/s), this saturates
  immediately. With 5 cameras at 5 FPS analytics and 1
  tracklet/frame, the peak is 25 tracklets/s. The ReID
  worker's 4-msg-poll-1000ms-cycle can process at most ~4/s
  per worker thread.

## T4 VRAM budget (per `Docs/t4_optimization.md`)

| Component | Claimed | Real (today) |
|---|---|---|
| PP-Human detector (TensorRT FP16) | ~2.5 GB | 0 GB (no Paddle) |
| PP-Human ReID (256-d) | ~0.4 GB | 0 GB (fallback) |
| TransReID (768-d, ViT-Base) | ~1.1 GB | 0 GB (fallback) |
| CUDA context | ~0.5 GB | 0.5 GB (driver) |
| Headroom | ~11 GB | 15.5 GB |

**In production today, the T4 is mostly idle.** When Paddle
is wired, the budget is correct.

## ReID batch sizing

- `app/reid/base.py:17`: `batch_size: int = 16`
- `configs/reid/pphuman_reid.yaml:11`: `batch_size: 16`
- `configs/reid/transreid.yaml:11`: `batch_size: 16`
- `TrackletCollector` caps `max_crops_per_tracklet = 15`. So
  a single tracklet batches 15 crops. Multiple tracklets
  in the queue are processed serially by `ReIDWorker`. No
  cross-tracklet batching. **Could batch up to 16-32 crops
  per forward pass for ~2× throughput.**

## T4 verdict

| Concern | Status |
|---|---|
| Architecture for performance | ✅ Mostly correct (one-process, shared-model intent) |
| Performance metrics exposed | ❌ Three of four metrics are always 0 or empty |
| Queue / backpressure | ❌ Config ignored; no per-camera rate limit |
| Real benchmark | ❌ Stub |
| Decode bottleneck | ❌ `cv2.VideoCapture` only; no FFmpeg |
| Production ReID cadence | ✅ |
| Production ReID batch | ⚠️ Could be larger; not pipelined |
| GPU model sharing | ❌ Not enforced |

**T4 is ready as an *architecture sketch*.** To make it
production-ready:
1. Wire Paddle + TransReID real model loading.
2. Add per-camera FPS measurement (track wall-clock between
   `FrameResult` emissions).
3. Fix `gpu_memory_used`, `qdrant_query_latency`,
   `postgres_write_latency`, `analytics_fps`,
   `tracklet_buffer_size`, `stream_backlog` to actually
   be observed.
4. Implement `scripts/benchmark_t4.py` to load the real
   model and run scenarios 1-5 against a recorded dataset.
5. Add FFmpeg subprocess for RTSP decode; do not depend on
   `cv2.VideoCapture`'s H.264 path.
6. Implement the `queues.tracklet_buffer_capacity` and
   `backpressure_pause_threshold` config keys.
