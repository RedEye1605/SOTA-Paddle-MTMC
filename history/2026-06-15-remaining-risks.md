# Remaining Risks

> **Honest assessment of the gaps between the current implementation
> and a fully production-ready deployment.** Each risk is sized with
> the operator work needed to close it.

## 1. Operator-side setup (1-2 days)

These items block production deploy but are well-documented in
`Docs/official_paddle_integration.md` §7 and `README.md`.

| # | Risk | Severity | Owner work | Documentation |
|---|---|---|---|---|
| 1.1 | PaddleDetection repo not cloned | HIGH | `git clone PaddleDetection /opt/paddledetection` | `Docs/official_paddle_integration.md` §1 |
| 1.2 | PP-Human MOT model not downloaded | HIGH | download from BCE (~200 MB) and unzip | `Docs/official_paddle_integration.md` §1.1 |
| 1.3 | TensorRT engine not built | HIGH | `paddle.tools.trt --run_mode trt_fp16` (~10-30 min build) | `Docs/official_paddle_integration.md` §7 |
| 1.4 | TransReID weight shape mismatch | MEDIUM | the on-disk weight is MSMT17 (`num_class=1041`); the config says Market-1501 (`num_class=751`). Either download the Market-1501 weight OR update the config to `num_class=1041`. | `Configs/reid/transreid.yaml` |
| 1.5 | `SOTA_API_TOKEN` not set | HIGH | the server refuses to start in production without it | `app/api/server.py:1-90` |
| 1.6 | `.env` uses `change_me_in_production` defaults | MEDIUM | the audit's S6 finding; the system logs a warning at startup but does not refuse. The fix is operator hygiene. | `docker-compose.yaml` |

## 2. Incomplete audits (3-5 days)

These are documented as PATCH-PARTIAL in the FIX_SUMMARY. They are
not blocking but should be closed before a multi-week deploy.

| # | Risk | Severity | Remaining work |
|---|---|---|---|
| 2.1 | PATCH-016 travel-time filter not in Qdrant payload | LOW | the topology filter is strict (same cam + linked cams only); a CAM_01 candidate older than the max travel window is filtered by the 24h persistence window. The strict travel-time filter is in the `docs/architecture.md` plan but not yet pushed to Qdrant payload. |
| 2.2 | PATCH-018 per-camera FPS not logged | LOW | the `analytics_fps_per_camera` gauge exists; per-camera wall-clock wiring in `MultiCameraRunner.stream` is a 1-hour change. |
| 2.3 | PATCH-029 best.jpg re-keyed by global_id | LOW | the best.jpg is now uploaded next to the debug crops; the re-keying after global_id assignment is documented but not implemented. A race-condition with two ReID workers re-keying the same tracklet is the reason for the `evidence_rekey` dedup table (also a follow-up). |
| 2.4 | PATCH-031 backpressure / QoS | LOW | the `Queue(maxsize=64)` is bounded; the explicit `backpressure_pause_threshold` from `app.yaml` is not read. A 1-hour change in `MultiCameraRunner`. |
| 2.5 | PATCH-032 RTSP reconnect | LOW | `make_frame_reader` does not reconnect on EOF for RTSP. A real T4 deploy should use FFmpeg subprocess (the audit's T4 audit flagged this). The interface is clean; the fix is local. |
| 2.6 | PATCH-048/49 `benchmark_t4.py` skeleton | MEDIUM | the script structure is in place; running it against a real model + recorded dataset is operator work. |
| 2.7 | PATCH-047 Docker api HEALTHCHECK | LOW | the infrastructure services have healthchecks; the `api` service does not. A 1-line change in `docker-compose.yaml`. |

## 3. Architecture choices (1-2 weeks)

These are decisions baked into the architecture; they are not bugs
but they may not match the operator's preference.

| # | Choice | Alternative | Why we chose this |
|---|---|---|---|
| 3.1 | One PP-Human subprocess per camera | Single subprocess with multi-stream API | the official PaddleDetection pipeline.py reads one `--video_file` at a time; multi-stream = multi-process is the documented pattern |
| 3.2 | Vendor minimal TransReID backbone (~250 lines) | pip-install `damo-cv/TransReID` | the upstream repo has no PyPI release; pulling the full repo would also pull yacs, the custom `config` module, training-only data loaders, etc. The vendor is inference-only. |
| 3.3 | StrongBaseline via `paddle.inference` | Use the PP-Human pipeline's ATTR block directly | the StrongBaseline 256-dim output is what the PP-Human pipeline uses for per-frame ReID; we want the offline ReID to be consistent with the pipeline's per-frame ReID |
| 3.4 | Stage 3 24h fallback with higher threshold | Always-on Stage 3 | the audit's "Stage 3 is low-confidence only" — we raise the threshold to 0.92 for Stage 3 to avoid false merges |
| 3.5 | HITL review via `GET /identity/ambiguous` | Web UI | UI is out of scope for this phase; the API endpoint is the foundation |

## 4. Operational risks (1-2 weeks of operation data)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 4.1 | Threshold tuning | MEDIUM | the deploy starts with `auto_match_threshold=0.82`; the first 1-2 weeks of labeler data will retune it via the promotion gate |
| 4.2 | Topology calibration | MEDIUM | the camera_links are initially seeded from a human survey; the IMPROVEMENT_LOOP Component 8 re-calibrates them from observed transitions after 1 month |
| 4.3 | ID fragmentation | LOW | the 5-factor `final_score` is the threshold variable (PATCH-008); ambiguous candidates are not auto-merged; the staging scheme is conservative |
| 4.4 | False merge | HIGH | the topology hard-block + Stage 3 higher threshold + ambiguous-hold policy are the safeguards. The promotion gate watches `false_merge_rate`. |

## 5. Things explicitly NOT changed

* **`Service/`** — out of scope; the hard rule is preserved.
* **`compare_with_service_baseline.py`** — stub; not touched.
* **The MediaMTX optional service** — the audit's "P2 — MediaMTX
  optional" item is not in the original audit's PATCH_PLAN; not touched.
* **Operator merge UI** — the audit's "P2" item; the
  `GET /identity/ambiguous` endpoint is the foundation, the UI is a
  follow-up.

## 6. What blocks the verdict from being "READY FOR LIMITED PRODUCTION"

1. The operator must:
   - Clone PaddleDetection to `/opt/paddledetection` (or set `PPHUMAN_PIPELINE_PATH`).
   - Download the PP-Human MOT model + unzip to `/models/pphuman`.
   - Build the TensorRT engine (`paddle.tools.trt --run_mode trt_fp16`).
   - Set `SOTA_API_TOKEN` in `.env`.
2. The on-disk TransReID weight (MSMT17) must be aligned with the config
   (`num_class=1041` for MSMT, or download Market-1501 weight).
3. The T4 GPU + CUDA 12.4 environment must be present.
4. A recorded multi-camera test dataset must be available for the
   smoke benchmark.

Once those four items are confirmed, the system is
**READY FOR LIMITED PRODUCTION** (one showroom, one site).

For multi-site deployment, additional work is needed (cameras
calibration, topology survey, labeler pipeline, dashboard).
