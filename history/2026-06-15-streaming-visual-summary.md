# FixReport 49 — Streaming / MinIO / MQTT / Visualization summary

**Date**: 2026-06-13
**Scope**: complete summary of Phases 0-12, including the readiness
verdict and remaining blockers.

---

## 1. Implementation summary

All 12 phases are complete. The SOTA pipeline now has:

* **MediaMTX streaming**: per-camera `MediaMTXStreamer` with FFmpeg
  argv, URL builders, daemon push thread, exponential-backoff
  reconnect, and per-camera isolation. Disabled by default in
  tests; opt-in via `MEDIAMTX_ENABLED=true` + `MEDIAMTX_HOST`.
* **MinIO evidence storage**: 3-bucket split (evidence, reports,
  models) with deterministic paths exactly matching the user
  spec:
  * `evidence/{site}/{cam}/{zone}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet_id}/best.jpg`
  * `evidence/.../debug_{frame_id:06d}.jpg`
  * `evidence/pending/...` (for unassigned GIDs, PATCH-029)
  * `visualization/{site}/{cam}/{yyyy}/{mm}/{dd}/first_3000_frames.mp4`
  * `reports/{site}/{yyyy}/{mm}/{dd}/benchmark_{ts}.json`
* **MQTT / ThingsBoard telemetry**: token and username/password
  auth, `{ts, values}` payload, configurable topic, fail-fast on
  missing config.
* **Annotated visualization**: new
  `scripts/generate_visual_validation.py` runs the full pipeline
  on the first N frames of a recorded video, annotates every
  frame with bbox / confidence / local_track_id / global_id /
  ReID similarity / zone_id, and writes both an MP4 and a JSON
  sidecar.
* **Tests**: 360 tests pass (was 249 before this work, +111 net
  new tests).

## 2. Service reference files inspected

The complete list is in
`FixReports/44_service_reference_mapping.md`. Summary:

* `Service/offline-people-counting/app/io/streamer_command.py`
* `Service/offline-people-counting/app/io/streamer.py`
* `Service/offline-people-counting/app/io/mqtt_connection.py`
* `Service/offline-people-counting/app/io/mqtt_publisher.py`
* `Service/offline-people-counting/app/io/mqtt_topics.py`
* `Service/offline-people-counting/app/io/minio_client.py`
* `Service/offline-people-counting/app/io/minio_uploader.py`
* `Service/offline-people-counting/app/io/history.py`
* `Service/offline-people-counting/app/counting/payload.py`
* `Service/offline-people-counting/app/engine/overlay.py`
* `Service/offline-people-counting/config.yaml`
* `Service/offline-people-counting/.env.example`
* `Service/docker-compose.yaml`
* `Service/data/` (read-only listing only)

No Service file was modified.

## 3. Dataset copy result

* `data/cam1_merged.mp4` — 2.1 GiB, 3072×2048, 20.00 fps,
  143 726 frames, hevc.
* `data/cam2_merged.mp4` — 1.8 GiB, 2592×1944, 20.00 fps,
  141 852 frames, hevc.

MD5 verified byte-identical to `Service/data/`. The other
folders in `Service/data/` (`CCTV 29 APR 2026/`, `CCTV AI/`,
`CCTV FSS-*/`, `crossing_*`, `datasets/`, `fss_*_merged.mp4`)
were **not** copied.

CAM_01 → `cam1_merged.mp4`; CAM_02 → `cam2_merged.mp4`.

## 4. MinIO bucket integration

| Bucket | Env var | Status | Purpose |
| --- | --- | --- | --- |
| evidence | `MINIO_BUCKET_EVIDENCE` | ✅ used by `put_crop` | person crops, best.jpg, debug |
| reports | `MINIO_BUCKET_REPORTS` | ✅ used by `put_visualization` + `put_report` | benchmark JSON, visualization MP4 |
| models | `MINIO_BUCKET_MODELS` | ✅ reserved (no current writer) | reserved for future model fetching |

Bucket-required behaviour:
* `MINIO_CREATE_BUCKETS=false` (default) → `_require_bucket`
  raises `RuntimeError` if the bucket is missing. Production
  must pre-create the buckets.
* `MINIO_CREATE_BUCKETS=true` (dev/CI) → `_maybe_make_bucket`
  creates the bucket on first connect.

Evidence path pattern (verified by
`tests/test_evidence_bucket_paths.py`):

```text
evidence/{site_id}/{camera_id}/{zone_id}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet_id}/best.jpg
evidence/{site_id}/{camera_id}/{zone_id}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet_id}/debug_{frame_id:06d}.jpg
```

## 5. MQTT / ThingsBoard integration

* Host configured: yes (via `MQTT_HOST` or `MQTT_BROKER_HOST` in
  the operator's `.env`; exact value redacted in this report).
* Topic: `v1/devices/me/telemetry` (ThingsBoard standard) or
  `v1/devices/<token>/telemetry` if `THINGSBOARD_DEVICE_TOKEN` is
  set; overridable via `MQTT_TOPIC`.
* Payload format: `{ts, values}`. `ts` is milliseconds since
  epoch. `values` carries per-camera / per-zone gauges and
  deltas.
* Telemetry enabled: `MQTT_ENABLED=true` (env-controlled).
  Disabled when the broker config is incomplete; the runtime
  safety gate fails the production preflight in that case.
* Credentials are never logged (verified by
  `tests/test_mqtt_thingsboard_external.py`).

## 6. MediaMTX streaming

* Enabled: opt-in via `MEDIAMTX_ENABLED=true`. Disabled by default
  in CI / dev.
* Stream URLs (one per camera):
  * `rtsp://{MEDIAMTX_HOST}:{MEDIAMTX_RTSP_PORT}/sota-paddle-mtmc/CAM_01`
  * `rtsp://{MEDIAMTX_HOST}:{MEDIAMTX_RTSP_PORT}/sota-paddle-mtmc/CAM_02`
* FFmpeg command pattern (from
  `app/streaming/ffmpeg_writer.py::build_ffmpeg_command`):

  ```text
  ffmpeg -loglevel warning \
    -f rawvideo -pix_fmt bgr24 -s {W}x{H} -r {fps} -i pipe:0 \
    -c:v libx264 -preset ultrafast -tune zerolatency \
    -b:v {kbps}k -maxrate {kbps}k -bufsize {2*kbps}k \
    -g {fps*2} -sc_threshold 0 \
    -f rtsp -rtsp_transport tcp {output_url}
  ```

* Per-camera isolation: each `MediaMTXStreamer` is its own
  object; a failure on CAM_01 does not stop CAM_02 (verified by
  `tests/test_streaming_disabled_safe.py` and
  `tests/test_mediamtx_streaming_config.py`).
* Strict mode (`MEDIAMTX_STRICT=true`) is honored by the
  visualization script.

## 7. Detector / ReID status

* PP-Human detector model: `models/pphuman/mot_ppyoloe_l_36e_pipeline/`
  (196 MB). Config wired via `configs/app.yaml::detection_tracking`.
* PP-Human ReID model: `models/pphuman/strongbaseline_r50_30e_pa100k/`
  (90 MB).
* **Active ReID model: `pphuman_strongbaseline`** (production
  default).
* TransReID availability: TransReID MSMT17 checkpoint
  (`models/vit_transreid_msmt.pth`, 400 MB) is present and
  compatible — the inspector script confirms 1041 classes /
  768-dim embeddings, matching the MSMT17 profile. Switching to
  TransReID as `active_model` is out of scope for Phases 4-10;
  the active model is intentionally kept on
  `pphuman_strongbaseline` until the operator explicitly
  provisions the Qdrant collection and verifies the preflight.

## 8. 3000-frame visualization

| Camera | MP4 | JSON sidecar | Frames | Avg fps | Elapsed |
| --- | --- | --- | ---: | ---: | ---: |
| CAM_01 | `reports/visualization/CAM_01_first_3000_frames.mp4` (349 MB) | `reports/visualization/CAM_01_first_3000_frames.json` (1.6 MB) | 3 000 | 23.2 | 129.3 s |
| CAM_02 | `reports/visualization/CAM_02_first_3000_frames.mp4` (338 MB) | `reports/visualization/CAM_02_first_3000_frames.json` (1.6 MB) | 3 000 | 28.0 | 107.1 s |

Both runs are explicitly `smoke=True` (the dev container has no
Paddle stack). The HUD includes the `WARNING: SMOKE-TEST BACKEND`
line and the JSON sidecar's `detector_backend` / `reid_backend`
are `synthetic_smoke` / `deterministic_smoke`. The visual
artefacts are valid for human review of the pipeline plumbing,
not for assessing real model quality.

## 9. Production benchmark on cam_merged

`configs/benchmark_real_cam_merged.yaml` is wired to
`data/cam1_merged.mp4` and `data/cam2_merged.mp4`. The smoke
benchmark run completed:

* `status: success`
* `cameras_processed: [CAM_01, CAM_02]`
* `detector_backend: synthetic_smoke` (explicit-smoke flag)
* `reid_backend: smoke_deterministic`
* `workers_crashed: false`
* `required_metrics_present: false` (no labels)
* `total_analytics_fps: 198.98`
* `gpu_memory_used_mb_max: 224.0`

The **production benchmark refused to start** because the
`runtime_mode` safety gate correctly blocked the synthetic path
in production mode. This is the expected and required behaviour
per the hard rule *"Production must refuse synthetic detector and
deterministic ReID"*.

## 10. Ruff / test status

* `ruff check`: ✅ all checks passed
* `ruff format --check`: ✅ 111 files already formatted
* `compileall`: ✅ no syntax errors
* `pytest`: ✅ **360 passed, 1 warning** (the warning is a
  pre-existing thread-exception noise from intentional smoke
  tests that exercise the "stub source not openable" path).
* `docker compose config`: ✅ normalised service graph emitted.

## 11. Readiness gate verdict

The system satisfies the criteria for:

* ✅ **STRUCTURALLY_READY** — code compiles, tests pass, infra
  builds, documentation complete.
* ✅ **READY_FOR_SHADOW_TEST** — preflight passes, smoke
  benchmark works, no detector worker crashes, no safety gate
  weakened.
* ❌ **READY_FOR_LIMITED_PRODUCTION** — **refused**, with
  reasons:
  1. Production benchmark refused by `runtime_mode` safety
     gate (PaddleDetection pipeline not installed in dev).
  2. `detector_backend: synthetic_smoke` is not real.
  3. `reid_backend: smoke_deterministic` is not real.
  4. `required_metrics_present: false` (no labels for
     cam_merged set).
  5. Real metrics from a recorded multi-camera production
     run do not exist.

All five of the user's "do not claim READY_FOR_LIMITED_PRODUCTION
unless…" criteria are unmet.

## 12. Remaining blockers

To promote to `READY_FOR_LIMITED_PRODUCTION`, the operator must:

1. **Provision PaddleDetection** at `/opt/paddledetection` (or set
   `PPHUMAN_PIPELINE_PATH` / `PPHUMAN_INFER_CONFIG` to the
   existing on-disk weights). The dev container has the weights
   but not the runtime pipeline.
2. **Re-run the production benchmark** with the real adapter so
   `detector_backend` reports `paddledetection_pphuman` (or the
   configured `framework`).
3. **Produce ground-truth labels** for the cam_merged videos
   (e.g. via the `Service/offline-people-counting/scripts/eval`
   labeler) so `required_metrics_present` becomes `true`.
4. **Confirm `workers_crashed: false`** on the production
   benchmark report.
5. **Re-run `scripts/readiness_gate.py`** to confirm the gate
   promotes to `READY_FOR_LIMITED_PRODUCTION`.

## 13. Exact next operator commands

```bash
cd /home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC

# 1. Install PaddleDetection (or point at an existing checkout)
test -d /opt/paddledetection || \
  git clone https://github.com/PaddlePaddle/PaddleDetection \
    /opt/paddledetection

# 2. Verify the PP-Human models
ls models/pphuman/mot_ppyoloe_l_36e_pipeline/model.pdiparams
ls models/pphuman/strongbaseline_r50_30e_pa100k/model.pdiparams
ls models/vit_transreid_msmt.pth

# 3. Re-run the production benchmark on cam_merged
uv run python scripts/benchmark_t4.py \
  --mode production_benchmark \
  --dataset configs/benchmark_real_cam_merged.yaml \
  --max-seconds 60

# 4. Re-run the readiness gate
uv run python scripts/readiness_gate.py

# 5. (Optional) Re-generate the visualization with the real adapter
uv run python scripts/generate_visual_validation.py \
  --cam CAM_01 \
  --input data/cam1_merged.mp4 \
  --max-frames 3000 \
  --output reports/visualization/CAM_01_first_3000_frames.mp4
```

Until the operator runs the above, the project remains at
`READY_FOR_SHADOW_TEST` — exactly what the safety gate is
designed to enforce.
