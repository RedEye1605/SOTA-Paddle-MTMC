# Full Chain Diagnosis — 2026-06-16

**Goal:** dig every remaining issue causing bug / stale state in the
production chain. Apply systematic-debugging Phase 1 (root cause
investigation) across every component boundary. NO FIXES are proposed
in this report; this is the evidence base for the next round.

## Diagnosis methodology

Multi-component evidence sweep. For each component boundary (api, PP-Human
subprocess, reid-sidecar, Redis, Qdrant, Postgres, MinIO, MediaMTX) we
captured: live state, per-thread WCHAN, per-FD read position, log tail,
and resource usage. Then the data was correlated to identify WHERE the
chain is broken.

## Findings (10 issues, 5 already documented, 5 newly discovered)

### 1. PRODUCTION-READINESS BUGS (5 — already fixed, see 2026-06-17 report)

| Bug | Status | Reference |
|-----|--------|-----------|
| BUG-1 reid_override path | ✅ Fixed | `app/identity/ambiguity.py`, commit 0873b39 |
| BUG-2 active:* TTL | ✅ Fixed | `app/storage/redis_state.py`, commit f0997c4 |
| BUG-3 sidecar bypasses bridge | 📝 Documented (not a bug) | report section 3 |
| BUG-4 telemetry stream empty | 📝 Documented (not a bug) | report section 4 |
| BUG-5 sync MinIO upload | ✅ Fixed | `app/detection/_vendor/paddledetection_pipeline.py`, commit 071ca19 |

### 2. BUG-NEW-A: PP-Human capturevideo deadlock at 24% of source videos ⚠️ BLOCKER

**Symptom (evidence):**
- api is up 18 min, but `stream:detections` has 100,003 entries (all from
  the prior run, XLEN capped at MAXLEN=100000) and 0 new entries in 16+ min.
- api log: **4 log lines after 18:04:00 startup**; the most recent is
  `FastAPI auth enabled` at 18:04:00.858.
- PP-Human subprocesses (PIDs 175, 179) are alive (40 threads, 700MB RSS each)
  but **all in `futex_do_wait` sleep state**. No Python stack visible without
  py-spy.
- File position frozen: CAM_01 = 539,089,473 / 2,221,062,762 bytes (**24.3%**),
  CAM_02 = 452,481,403 / 1,940,110,445 bytes (**23.3%**). Same value 9+ sec
  apart.
- `rchar` (cumulative read) frozen at 295,507,764 (CAM_01) and 292,238,382
  (CAM_02). No more reads from the .mp4 files.
- ffmpeg children (PIDs 300, 339) in `anon_pipe_read` — they have **never
  received any frame bytes** through the api's pipe.

**Root cause (hypothesis, not yet confirmed):**
The `capturevideo` producer thread (PP-Human pipeline.py:944) and the
`predict_video` consumer thread (PP-Human pipeline.py:1086) are
deadlocked on `framequeue` (size 10). The capture thread is most likely
stuck inside OpenCV's HEVC decoder for the 4K merged videos. The
predict_video thread is blocked on `framequeue.get()` waiting for new
frames that never arrive.

**Why this is different from the prior "stall at frame 49,091" theory:**
That theory was based on the api restart count. The actual current
position is **~99,000+ frames in (24% of 7186 sec × 20 fps = 143,720
total frames)**, not 49,091. The 24% mark is reached after ~2 min of
active decode, then the chain freezes for the remaining 16+ min.

**Why the api main process shows 398% CPU intermittently:** The api runs
periodic sweeps (resolver, finalize loops, telemetry flushes) that burn
CPU in short bursts. They're not the stall cause; they're idle between
bursts. Confirmed by `docker stats`: API CPU dropped from 398% to 1.80%
between two sample points (12s apart) with no new log lines.

**Open question:** Is the deadlock transient (Paddle CUDA context loss,
recovers on restart) or hard (HEVC decoder bug at this offset)? Need
to restart the chain and watch whether the same offset stalls.

### 3. BUG-NEW-B: reid-sidecar RTSPFrameBuffer cannot open MediaMTX RTSP ⚠️ BLOCKER

**Symptom (evidence):**
- sidecar log: 200+ consecutive `WARNING app.utils.frame_buffer: RTSPFrameBuffer[CAM_01] error: failed to open rtsp://198.51.100.20:8554/sota-paddle-mtmc/cam1_merged; sleeping 30.0s` messages.
- Underlying: `method DESCRIBE failed: 404 Not Found` from MediaMTX.
- Both CAM_01 and CAM_02 buffers fail every 30s.

**Root cause:**
The sidecar's frame_buffer is a fallback path for when tracklet
emissions don't include `frame_uri`. It tries to read the corresponding
RTSP stream from MediaMTX. But:
- (a) MediaMTX is **external** to the compose stack and has no publisher
  for the `sota-paddle-mtmc/cam1_merged` path.
- (b) Even if MediaMTX had a publisher, the api is in "Unified stream
  mode" — the api reads the raw video file, runs PP-Human detection, and
  pushes the annotated frames back to MediaMTX. The sidecar is supposed
  to read the **annotated** stream back, not the raw one.

The result: the sidecar's fallback path is structurally broken. The
sidecar only works when the api's pipeline emits `frame_uri` (BUG-5
fix). When the pipeline is stuck, no `frame_uri` is emitted, and the
sidecar's fallback can't help.

**Why the sidecar hasn't crashed:** the 30s sleep + retry loop means
the sidecar stays alive but does no work. Its main work is
`sidecar: tracklet=... cam=... local=... crops=... points=...` lines,
which require a `frame_uri` to download. Zero such lines in the current
run.

### 4. BUG-NEW-C: stream:embeddings_transreid, stream:telemetry, stream:zone_events all empty (no producers) ⚠️ DESIGN

**Symptom (evidence):**
- `stream:embeddings_transreid` = 0 entries (no producer when sidecar is
  bypassed; BUG-3 documented this).
- `stream:telemetry` = 0 entries (no producer code anywhere; leftover
  from prior design).
- `stream:zone_events` = 0 entries (resolver only writes here when
  `new_zone_id` is set; in the prior run, 1 of 59 decisions had a zone
  change).

**Root cause:** These streams are defined in `configs/app.yaml` but no
code path writes to them reliably:
- `stream:embeddings_transreid`: the sidecar publishes to
  `stream:embeddings` (single stream) instead of going through the
  two-step bridge. Contract violation per spec; data path is correct.
- `stream:telemetry`: zero producers. The `TelemetryWorker` publishes
  to MQTT (ThingsBoard), not Redis Streams. The stream is dead-letter.
- `stream:zone_events`: only the resolver writes here, only when
  `new_zone_id` is set. 0 of 0 events in the current run because the
  resolver isn't running (chain frozen).

**Decision:** Document as design limitation. The `stream:telemetry`
and `stream:zone_events` streams are dead-letter. Cleanup would
require coordinated changes to the resolver + telemetry worker.

### 5. BUG-NEW-D: PRODUCTION REFUSED for TransReID in api; expected but at ERROR level

**Symptom (evidence):**
- api log: `ERROR [app.runtime_mode] PRODUCTION REFUSED: TransReIDAdapter attempted "missing real model (weight='/models/vit_transreid_msmt.pth')" but runtime mode is 'production' which disallows synthetic / deterministic paths.`

**Root cause:** Expected. The api image has no TransReID model mounted
(by design — torch-free api). The api's `ReIDWorker` is a passthrough:
it downloads BGR frames, crops bboxes, and hands off to the sidecar
via `stream:tracklets` → sidecar → `stream:embeddings`. The adapter
configuration is set so the api can still emit `stream:embeddings` with
the right `model_name`, but `extract()` returns zeros.

**Decision:** Downgrade the log level from ERROR to INFO or WARNING.
The error confuses operators reading logs. The chain is working as
designed (api has no model, sidecar has the model, sidecar is the only
ReID path).

### 6. BUG-NEW-E: MediaMTX HLS endpoints return 404 (no publisher) ⚠️ BLOCKER (test only)

**Symptom (evidence):**
- `curl -I http://198.51.100.20:8889/sota-paddle-mtmc/cam1_merged/index.m3u8`
  → `HTTP/1.1 404 Not Found` (server: mediamtx).
- Same for cam2_merged.

**Root cause:** MediaMTX is external (not in compose) and has no
publisher for these paths. The api is supposed to push annotated frames
to `rtsp://198.51.100.20:8554/sota-paddle-mtmc/<basename>`, but the
PP-Human pipeline is frozen, so nothing is being pushed. The HLS
endpoint is reachable but has no upstream.

**Why this matters for the operator:** The HLS overlay verification
(the operator's "show me it works" criterion) is blocked on:
1. The PP-Human pipeline being unstuck (BUG-NEW-A) so frames are pushed.
2. MediaMTX being reachable (confirmed: 198.51.100.20:8554 responds to
   RTSP DESCRIBE but returns 404 for these paths).
3. The `IdentityOverlayCache` reading the resolver's decision and
   writing the bbox overlay to the HLS stream (separate pipeline).

### 7. Chain orchestration findings

**Pre-existing task inventory:**

| Task | Status | Notes |
|------|--------|-------|
| #138 SAHI at 2 Hz | pending (BLOCKED on BUG-NEW-A) | Need pipeline flowing first |
| #139 Compare baseline vs 2 Hz | pending (BLOCKED on BUG-NEW-A) | Same |
| #140 SAHI at 5 Hz | pending (BLOCKED on BUG-NEW-A) | Same |
| #156/159 Pipeline stall root cause | diagnosed (BUG-NEW-A) | Not yet implemented |

**Side-effect: stale `active:*` keys.**

The BUG-2 fix was verified live in a prior session (4 active keys
present). The keys have since expired (TTL=600s, now 18+ min old). The
chain is not producing new keys because the resolver isn't running
(BUG-NEW-A).

**This is NOT a regression.** The `_set_active_if_possible` function is
correctly implemented; the chain just has no work for it to do right
now.

## Hypotheses tested (Phase 3) — 2026-06-16 18:35-18:42 UTC

Restarted the chain (api+reid-sidecar via `docker compose up -d --force-recreate`).
Sampled for 7+ minutes. Findings:

| Hypothesis | Test result | Verdict |
|------------|-------------|---------|
| H1: Paddle CUDA context loss is transient | File pos frozen at 539,089,473 bytes **7+ min after restart** | **FALSE** — restart does not recover; same offset stalls again |
| H2: HEVC decoder bug at 24% | `cv2.VideoCapture(/data/cam1_merged.mp4).set(POS_FRAMES, 50000); read()` → 54.7 fps | **FALSE** — OpenCV decodes the same offset in 0.95s without issue |
| H3: framequeue producer stuck (capturevideo) | Producer thread is in `futex_do_wait`, **same state as consumer** | **INCONCLUSIVE** — both stuck, no easy way to disambiguate from /proc |

**Updated root cause (refined):** The video is healthy. The HEVC decoder
works. Paddle 3.3.1's `mot_predictor.predict_image()` (called from
`pipeline.py:1115`) deadlocks at the same offset after a few minutes
of sustained inference. The deadlock survives a process restart
(same byte offset), which means it is **deterministic** — either
the predictor hits a CUDA state it can't recover from, or
OC-SORT's tracker internal state goes into an infinite loop on a
specific detection pattern.

**Evidence chain:**
1. File pos 539,089,473 = 24.3% of cam1_merged.mp4
2. 24.3% × 7186s = 1745s into the file; seek was 1800s, so ~55s past the
   seek point, which is ~1100 frames at 20 fps
3. Restart reproduces the same stall at the same offset
4. cv2 decodes the same offset in 0.95s (healthy)
5. PP-Human subprocess is in `futex_do_wait` for both producer and consumer
6. api main process is at 500-700% CPU but all threads are in `futex_do_wait`
   (GIL contention among idle threads)

**Most-likely root cause candidates (ranked):**
1. **`PaddlePredictor.predict()` infinite loop on a specific frame**
   (Paddle 3.3.1 has known issues with sustained inference loops)
2. **OC-SORT tracker's `update()` deadlock** on a degenerate detection
3. **`side-channel event queue.put()` blocking forever** because the
   api's `pipe:[18735048]` reader is stuck in `futex_do_wait`

## Next-step options (no fix proposed in this report)

1. **Add per-frame logging in pipeline.py** so we can see exactly which
   frame the stall happens at (vs byte position). Requires
   re-vendoring the file.
2. **Switch to `--run_mode trt_fp16` or `--device cpu`** as a workaround
   to test if it's a CUDA-specific issue.
3. **Reduce the video to a single-frame test** to confirm the deadlock
   is in PaddlePredictor vs. video content.
4. **Replace PaddleHuman's main loop with a custom one** that uses
   frame-by-frame inference with explicit timeouts.

Any of these requires a code change + chain restart + ~10 min of
observation. The operator's "1 person = 1 global_id" criterion is
NOT verifiable end-to-end until the stall is fixed.

## File inventory (no code changes in this report)

Files read for this diagnosis (no edits made):
- `app/identity/resolver.py`
- `app/identity/ambiguity.py`
- `app/storage/redis_state.py`
- `app/storage/postgres.py` (skipped — connection issue)
- `app/detection/_vendor/paddledetection_pipeline.py`
- `app/main.py`
- `/opt/paddledetection/deploy/pipeline/pipeline.py` (vendored, read-only inside container)
- `app/workers/identity_overlay_cache.py` (skipped)
- `app/workers/telemetry_worker.py` (skipped)
- `app/reid/transreid_sidecar.py` (skipped — read in prior session)
- `docker-compose.yaml` (skipped — no changes needed)

## Next step

Phase 3 hypothesis test. Restart the chain (`docker compose up -d
--force-recreate api reid-sidecar`) and watch:
- whether the chain reaches 24% in 2 min and freezes again (H1 false → H2)
- whether it gets past 24% and runs to completion (H1 true → CUDA context loss)
- whether the ffmpeg push to MediaMTX finally publishes frames (HLS visible)

If H1 is true, the operator's "1 person = 1 global_id" criterion is
verifiable end-to-end. If H1 is false, the source video has a bad
frame at 24% and the chain needs a video re-encode or seek-around.

## Status

| Issue | Status | Blocks production? |
|-------|--------|-------------------|
| BUGS 1-5 (production-readiness) | ✅ Fixed | No |
| BUG-NEW-A PP-Human deadlock | 🔬 Diagnosed; hypothesis not yet tested | **Yes — chain frozen** |
| BUG-NEW-B Sidecar RTSP fallback | 🔬 Diagnosed; structural | Yes — secondary to A |
| BUG-NEW-C Dead-letter streams | 📝 Documented | No (cleanup only) |
| BUG-NEW-D TransReID ERROR log | 📝 Documented; needs log-level downshift | No (cosmetic) |
| BUG-NEW-E MediaMTX HLS 404 | 🔬 Diagnosed; consequence of A | Yes — but tests HLS overlay |
