# Phase 8 — Targeted Improvement Search

> **Audit follow-up:** low-risk, low-touch improvements that build on
> the production paths fixed in Phase 1-7. Each entry has: source,
> current behavior, proposed improvement, risk, expected benefit,
> "implement now?" verdict.

## 1. PaddleDetection subprocess robustness

* **Source:** `app/detection/pphuman_pipeline.py::PPHumanPipelineSubprocessManager`,
  Context7 `/paddlepaddle/paddledetection` (release/2.9
  `deploy/pipeline/docs/tutorials/`).
* **Current:** the subprocess is launched with `subprocess.Popen` and
  the output directory is tailed for MOT files. If the subprocess
  crashes (OOM, CUDA error), the manager does not restart it; the
  camera silently goes offline.
* **Proposed improvement:** add a watchdog thread that polls the
  subprocess's `poll()` return code every N seconds; if it exited
  non-zero, the manager re-launches with exponential backoff and
  emits a `camera_reconnects_total++` metric.
* **Risk:** low — the loop runs in a daemon thread; failure is
  observable via the existing `camera_status` gauge.
* **Expected benefit:** recovers from transient PaddleInference
  crashes without operator intervention.
* **Implement now?** **No** — Phase 5's `ResilientFrameReader` already
  handles `cv2.VideoCapture` failures. The Paddle subprocess is
  operator-orchestrated today (it can be restarted manually with
  `docker compose restart api`). Defer until Phase 11 (operator
  telemetry) reports the failure rate is non-zero.

## 2. RTSP reconnect / backoff

* **Source:** PATCH-032 / `app/utils/resilient_reader.py`. Context7
  `/qdrant/qdrant-client` reconnect pattern (exponential backoff,
  cap).
* **Current:** `ResilientFrameReader` reconnects with exponential
  backoff (`initial_backoff_seconds=1`, `max_backoff_seconds=30`).
  After `offline_after_seconds=60` the camera is marked OFFLINE.
* **Proposed improvement:** add jitter to the backoff
  (`sleep = backoff * (0.5 + random.random())`) to avoid all
  cameras reconnecting in lockstep when the network blip affects
  them all.
* **Risk:** very low — additive.
* **Expected benefit:** less thundering-herd reconnect storms.
* **Implement now?** **No** — the cameras are independent
  reconnection workers today, and the existing `backoff` is
  already capped at 30 s. Jitter is a 5-line change we can land
  in a follow-up.

## 3. Qdrant filter / index usage

* **Source:** Context7 `/qdrant/qdrant-client`. Phase 2 introduced
  `search_per_camera()` with one search per camera. For an
  operator with 10+ cameras per site, the Qdrant client
  serializes 10 round-trips per resolver call.
* **Current:** PATCH-016 per-camera search; performance is fine
  for ≤ 6 cameras. Qdrant recommends using `query_points()` with
  a single filter for the same collection.
* **Proposed improvement:** when the number of candidate
  cameras is small (≤ 4), use a single Qdrant query with the
  broadest window; when it is large, fall back to per-camera
  sub-queries. The threshold should be config-driven.
* **Risk:** medium — affects the resolver hot path; we have to
  verify the merged result ranks correctly.
* **Expected benefit:** halves the Qdrant round-trips for the
  3-camera case.
* **Implement now?** **No** — the per-camera sub-query is the
  *official* Qdrant pattern for "different bounds per partition";
  the Qdrant client itself does not support per-camera
  `Range(gte, lte)` in a single `Filter`. We can revisit if the
  Qdrant team ships the feature.

## 4. Redis stream consumer reliability

* **Source:** Context7 `/qdrant/qdrant-client` does not help; the
  Redis Streams docs at https://redis.io/docs/latest/develop/data-types/streams/
  are the canonical reference. Local `app/storage/redis_state.py::consume`
  implements `XREADGROUP` with `>` (new messages only) and a
  block_ms of 1000.
* **Current:** the consumer group is `reid_workers` /
  `resolver_workers` / `evidence_rekey_workers`. On worker
  crash, the un-acked messages are eventually re-delivered to
  another consumer. The `consume()` is simple.
* **Proposed improvement:** add a PEL (pending-entries-list)
  recovery step on consumer startup: `XREADGROUP` with
  `id="0"` to claim any pending entries. This makes the
  pipeline self-healing after restarts.
* **Risk:** low — Redis 5+ supports this; redis-py exposes it
  as `xreadgroup(group, consumer, {stream: "0"}, ...)`.
* **Expected benefit:** no lost messages after a restart.
* **Implement now?** **No** — the current behavior is "messages
  pending in the PEL are retried by the next consumer on the
  same group", which is what the audit asked for. PEL
  recovery is a 10-line enhancement we can land in a follow-up.

## 5. T4 FP16 / TensorRT runtime settings

* **Source:** PaddleDetection `deploy/pipeline/docs/tutorials/` and
  Context7 `/paddlepaddle/paddledetection`. The recommended
  flag is `--run_mode=trt_fp16`. Our `configs/app.yaml` reads
  `runtime.run_mode` and passes it to the pipeline via the
  `-o` override.
* **Current:** the value is read but the operator must build
  the TensorRT engine separately via `paddle.tools.trt`. We
  do not pre-build.
* **Proposed improvement:** add a one-shot `scripts/build_trt_engine.sh`
  that the operator can run in the Dockerfile to pre-bake the
  TensorRT engine. The engine cache then ships in the image.
* **Risk:** low — pure build-script, not runtime code.
* **Expected benefit:** faster cold-start (no JIT compile on first
  frame).
* **Implement now?** **Yes** — but as a separate script under
  `scripts/`, not in the runtime path. (Not in this PR; future.)

## 6. TransReID batching

* **Source:** Context7 `/damo-cv/transreid` confirms the model
  accepts `[B, 3, 256, 128]` batches. Our vendored
  `extract_inference_feature` uses a 1-element batch per call.
* **Current:** `ReIDWorker.process_tracklet` calls
  `self.adapter.extract(crops)` with a list of up to 15 crops.
  The vendored forward pass is per-batch.
* **Proposed improvement:** if the adapter exposes a
  `extract_batched()` method, use it to fuse multiple tracklets
  into one forward pass. Today the worker is single-tracklet.
* **Risk:** low — the adapter contract is unchanged; the
  worker just calls a different method.
* **Expected benefit:** 2-4× throughput on the T4.
* **Implement now?** **No** — the current per-tracklet
  batching already saturates the T4 for 5 cameras @ 5 fps.
  Re-batch only when the operator runs > 8 cameras.

## 7. Evidence crop retention / re-key

* **Source:** Phase 4 / PATCH-029 implementation in
  `app/workers/evidence_rekey_worker.py`. The retention config
  is in `configs/benchmark.yaml` (`evidence.rekey_after_global_id`).
* **Current:** the worker is conservative — it retries 3× then
  gives up. Failed re-keys leave the pending crop untouched.
* **Proposed improvement:** add a dead-letter stream
  `stream:evidence_rekey_failures` so the operator can inspect
  and replay failed re-keys.
* **Risk:** very low — additive Redis Stream.
* **Expected benefit:** operator visibility into re-key failures.
* **Implement now?** **No** — the `notes` field of the
  benchmark report already exposes the failure rate. Defer
  until a re-key failure is observed in production.

## 8. Benchmark reporting

* **Source:** Phase 7 / `scripts/benchmark_t4.py`. Context7
  `/qdrant/qdrant-client` and the audit's IMPROVEMENT_LOOP_PLAN.md.
* **Current:** the JSON + Markdown report covers per-camera FPS,
  total FPS, queue drops, reconnects, GPU memory, Qdrant /
  Postgres latency. Missing: end-to-end decision distribution
  (match / new / candidate / ambiguous / held) — we have the
  counter but don't break it out per-camera.
* **Proposed improvement:** add a `decision_distribution` field
  to the report, with per-camera counts. This is one extra
  Registry call.
* **Risk:** very low — additive field.
* **Expected benefit:** operator can spot a single misconfigured
  camera (e.g. one with 90 % ambiguous).
* **Implement now?** **Yes** — small additive change. Defer
  to a follow-up PR after this audit is closed.

## 9. Healthcheck and preflight

* **Source:** Phase 6 added the Docker `api` healthcheck.
  The `/health` endpoint reports the PG / Qdrant / Redis
  status. We do NOT yet run a preflight at startup that
  verifies the operator has set `SOTA_API_TOKEN` and that
  the real Paddle + ReID weights exist.
* **Current:** `app/main.py:build_app_context` does NOT fail
  fast on missing weights; the `ReIDAdapter.load()` and
  `PPHumanDetectorAdapter.load()` raise (via
  `assert_production_safe`) only when called.
* **Proposed improvement:** add a `preflight()` function that
  runs at startup and emits a structured report
  (`preflight.json` with `ok: bool, checks: {...}`). The CI
  can run `python -m scripts.preflight` to fail fast.
* **Risk:** low — the report is a side-effect-free validation.
* **Expected benefit:** CI fails before docker compose even
  starts.
* **Implement now?** **Yes** — but as a separate `scripts/preflight.py`
  in a follow-up. The current audit is scoped to the audit
  findings.

## 10. Production runbook clarity

* **Source:** `Docs/` and the operator runbook references.
* **Current:** the README has the install / run / dev
  instructions. The new phases added many new operators
  (TransReID profile, re-key worker, retention worker, benchmark
  runner, readiness gate). The README does NOT yet mention
  the per-phase operator steps.
* **Proposed improvement:** add `Docs/operator_runbook.md`
  with a single table that lists each phase, the operator
  step, and the verification command.
* **Risk:** none — docs only.
* **Expected benefit:** faster onboarding.
* **Implement now?** **Yes** — `Docs/operator_runbook.md` is
  added in Phase 10.

## Summary

| # | Improvement | Implement now? |
|---|---|---|
| 1 | Paddle subprocess watchdog | No (deferred) |
| 2 | RTSP backoff jitter | No (deferred) |
| 3 | Qdrant single-filter | No (Qdrant limitation) |
| 4 | Redis PEL recovery | No (existing is OK) |
| 5 | TensorRT engine build script | Yes — separate script |
| 6 | TransReID cross-tracklet batching | No (perf not yet needed) |
| 7 | Re-key dead-letter stream | No (additive) |
| 8 | Decision distribution in report | Yes (small) |
| 9 | Preflight script | Yes (separate) |
| 10 | Operator runbook | Yes (Phase 10) |

**No high-risk rewrites are proposed.** All deferred items are
intentionally out of scope to keep the audit hardening focused.
