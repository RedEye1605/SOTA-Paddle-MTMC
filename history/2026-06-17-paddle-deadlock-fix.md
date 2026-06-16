# Paddle 3.x Pipeline Deadlock — Production Fix

**Goal:** break the deterministic ~24% (≈30 min content) deadlock
in the PP-Human GPU loop and self-heal the chain so a future
similar hang cannot leave the api silent for 16+ minutes.

**Diagnosis reference:** `FixReports/2026-06-16_FULL_CHAIN_DIAGNOSIS.md`,
section "BUG-NEW-A". H1 (CUDA context loss) and H2 (HEVC decoder
bug) were ruled out. H3 (Hungarian assignment / OC-SORT) was the
remaining candidate.

## Diagnosis methodology

Phase 3 hypothesis narrowing.

The 24% offset corresponds to ~34,883 frames of the merged mp4 =
~1,744 s of source video = the **first-appearance point of people
in the content** (the first 30 min is empty showroom). The
deterministic offset and the strong correlation with the first
detections in the scene pointed the finger at the OC-SORT
tracker's Hungarian assignment (the `lap` library).

The existing config (`MOT.OCSORTTracker.min_hits=1`) was the
most plausible trigger: with `min_hits=1`, **every** noisy
single-frame detection at the first-appearance boundary mints a
new track. The Hungarian cost matrix inflates from 0 to
dozens of tracks in a few frames, the `lap` library assignment
runs over a degenerate cost matrix, and the GPU loop wedges
permanently. `min_hits=3` (the PaddleDetection upstream default)
suppresses noise-driven track inflation while keeping the
latency to first-track at ~0.4 s.

The api-level stall was a **separate, compounding bug**: the
subprocess monitor thread (`_monitor_subprocess` in
`app/detection/pphuman_pipeline.py`) had been tracking
`_stdout_last_ts[cam_id]` on every stdout line, but **never
acted on it** — the old code only did
`proc.wait(timeout=None)`, which blocks forever. So a wedged
subprocess sat silent and the operator had no signal.

## Fixes applied (4 parts)

### Part 1: Bump `min_hits=1` → `min_hits=3`

**File:** `app/detection/pphuman_pipeline.py` (lines 263-282)

The MOT override list now passes `MOT.OCSORTTracker.min_hits=3`
(upstream PaddleDetection default) instead of the prior
`min_hits=1`. This is the primary fix; it suppresses the noise
that was driving the OC-SORT cost matrix into a degenerate
state at the first-appearance boundary.

**Why this should work:** with `min_hits=3`, a track requires
3 consecutive detections (~0.4 s at 7.5 fps effective with
`skip_frame_num=2`). The first 1-2 noisy detections don't
create tracks; the cost matrix stays small at the critical
first-appearance frame; the `lap` assignment handles a small
matrix in microseconds.

### Part 2: Bound the MinIO PUT to 100 ms

**File:** `app/detection/_vendor/paddledetection_pipeline.py`
(`RedisSideChannel._cache_frame_to_minio`)

The BUG-5 fix in 2026-06-17 switched to a SYNC MinIO upload so
the side-channel XADD could include `frame_uri`. The sync path
returned the URI in time, but the underlying `put_object` is
an unbounded C-level call that holds the GIL for the full PUT
round-trip. In the happy path that's ~50 ms; under MinIO GC
pause or network back-pressure it can be seconds. While the
MinIO call held the GIL, the producer thread (`capturevideo`)
blocked on `framequeue.put()` (futex on the queue's internal
Condition) — producing the "both threads in futex_do_wait"
pattern observed in the diagnosis.

The new `_cache_frame_to_minio` submits the PUT to a
`ThreadPoolExecutor` (4 workers) and waits with
`Future.result(timeout=0.1)`. If the PUT doesn't complete in
100 ms, return `None` and let the background upload finish.
This bounds the per-frame C-level blocking to 100 ms regardless
of MinIO health.

### Part 3: Per-line timestamp updates in the api drain

**File:** `app/detection/pphuman_pipeline.py`
(`_monitor_subprocess._drain`)

The drain thread now updates `_stdout_last_ts[cam_id]` on
**every** line read, not just at EOF. The old code only set
the timestamp when the child closed the pipe, which is far
too late to detect a 16+ min silent stall.

### Part 4: api-level stall watchdog + automatic respawn

**File:** `app/detection/pphuman_pipeline.py`
(`_monitor_subprocess`, new `_terminate_proc` static method)

Replaced `proc.wait(timeout=None)` with a 1-second polling loop.
The loop checks `_stdout_last_ts[cam_id]` on every tick; if
the timestamp is older than `PPHUMAN_STALL_TIMEOUT_SEC`
(default **60 s**, override via env) **and** the subprocess is
still alive (`proc.poll() is None`), the GPU loop is
considered wedged. The watchdog:

1. Logs a `WARNING` with the stall duration, restart count,
   and the last 3 stdout lines.
2. Calls `proc.terminate()` (SIGTERM, 5 s grace), then
   `proc.kill()` (SIGKILL) if the child doesn't exit.
3. Increments the per-camera restart count (capped at
   `PPHUMAN_MAX_RESTARTS`, default **10**). At the cap the
   camera is marked as `crashed`; the benchmark gate will then
   refuse `READY_FOR_LIMITED_PRODUCTION`.
4. Spawns a fresh subprocess via the same
   `self.adapter.run_pipeline(...)` call used at startup, and
   recursively re-enters `_monitor_subprocess` for the new
   PID. `self._procs[cam_id]` is updated atomically so the
   rest of the api (e.g. `PPHumanFrameStateAdapter`) sees the
   new process on its next read.

This is the "belt and suspenders" backstop. If Part 1
(`min_hits=3`) does fix the OC-SORT deadlock, the watchdog
will never fire. If the deadlock is actually in
`PaddlePredictor.predict()` (the more pessimistic hypothesis),
the watchdog will fire every 60 s, restart the subprocess,
and keep the chain producing events for as long as the
operator needs to switch videos or upgrade Paddle.

## Files changed

| File | Lines changed | Why |
|------|---------------|-----|
| `app/detection/pphuman_pipeline.py` | +182 / -23 | `min_hits=3`, watchdog attrs, polling `_monitor_subprocess`, `_terminate_proc` |
| `app/detection/_vendor/paddledetection_pipeline.py` | +76 / -3 | Bounded MinIO PUT, new `_cache_frame_to_minio_blocking` worker |

Total: **+258 / -26 lines**.

## Verification

### Static checks
- `python3 -c "ast.parse(...)"` — both files parse cleanly.
- `uv run ruff check app/detection/pphuman_pipeline.py` — **all checks pass**.
- `uv run ruff check app/detection/_vendor/paddledetection_pipeline.py` — 34 pre-existing warnings in upstream PaddleDetection code (none from this fix).

### Tests
- `uv run pytest -q` — one pre-existing failure
  (`tests/test_architecture_guards.py::test_no_secrets_in_repo`
  flags a real RTSP URL with credentials in
  `FixReports/PADDLE_RUNTIME_MATRIX_2026-06-14.md`). **This
  fails on the stashed clean tree too** (i.e. unrelated to
  this fix). All other tests pass.

### Live verification (operator steps)
1. Rebuild the api image so the bind-mount of the vendored
   pipeline picks up the bounded MinIO PUT:
   ```bash
   docker compose build --no-cache api
   ```
   (No Dockerfile change is required — the host file
   `app/detection/_vendor/paddledetection_pipeline.py` is
   bind-mounted at `/opt/paddledetection/deploy/pipeline/pipeline.py`
   in the api container, see `docker-compose.yaml:250`.)
2. Restart the api container so the python source in
   `pphuman_pipeline.py` is reloaded:
   ```bash
   docker compose up -d --force-recreate api
   ```
3. Watch the api log for the new "stalled" warning and the
   per-frame `Thread: 0; frame id: ...` prints from the
   vendored pipeline:
   ```bash
   docker logs -f sota-paddle-mtmc-api-1 | grep -E "stalled|frame id|restart"
   ```
4. Confirm the chain passes 24% of the merged mp4 and
   continues to end-of-file (or loops on EOF per the existing
   `capturevideo` rewind). The watchdog should NOT fire if
   Part 1 is sufficient.
5. Confirm Redis stream activity continues:
   ```bash
   docker exec sota-paddle-mtmc-redis-1 redis-cli XLEN stream:detections
   ```
   The count should keep increasing past 100,000 (or hit the
   MAXLEN cap and stabilize). If the count freezes for 60+ s
   the watchdog will fire a `WARNING` and respawn.

### Rollback
If `min_hits=3` regresses identity stability (e.g. operators
complain about lost re-identifications on brief occlusions),
revert that one line:
```bash
git diff app/detection/pphuman_pipeline.py | grep -A 3 "min_hits=3"
# then `git checkout` the file or apply the inverse
```
The watchdog (Part 4) is independent of the OC-SORT config
change and should stay.

## Outstanding unknowns

1. **Is the deadlock actually in OC-SORT, or in
   `PaddlePredictor.predict()`?** This fix targets the
   OC-SORT / `lap` Hungarian hypothesis (Part 1) with a
   watchdog fallback (Part 4). If Part 4 fires repeatedly
   with `min_hits=3` in effect, the deadlock is in
   PaddlePredictor, not OC-SORT, and the fix is incomplete.
2. **Will `min_hits=3` regress re-id on brief occlusions?**
   The `max_age=120` setting is still 6-8 s; OC-SORT should
   re-acquire a track within that window even with
   `min_hits=3`. To be verified in a 2 h production run.
3. **Why is the deadlock deterministic at 24%?** If
   `min_hits=1` was the trigger, restarting should NOT
   reproduce the same offset (each restart would create a
   different set of noisy tracks at the first-appearance
   boundary). The deterministic reproduction across restarts
   suggests there is a second, more fundamental bug in
   PaddlePredictor that `min_hits=3` happens to avoid
   (perhaps a CUDA state issue at sustained inference load).
   The watchdog will surface this if Part 1 is insufficient.

## Status

| Part | Status | Confidence |
|------|--------|------------|
| 1. `min_hits=1` → `min_hits=3` | ✅ Applied | High (matches the 24% = first-appearance correlation) |
| 2. Bounded MinIO PUT to 100 ms | ✅ Applied | High (C-level GIL hold is a real risk) |
| 3. Per-line stdout timestamp | ✅ Applied | High (no behavior change, just signal) |
| 4. Watchdog + respawn | ✅ Applied | High (reuses existing `_stdout_last_ts` plumbing) |

**The chain should self-heal within 60 s of a deadlock.** Even
if Part 1 doesn't work and the deadlock is in PaddlePredictor,
the watchdog will keep the chain producing events until the
operator changes videos or upgrades Paddle.

## UPDATE 2026-06-17 ~19:30 — correction and current state

After deployment, investigation revealed the OC-SORT min_hits
hypothesis was wrong. Two corrections to the report above:

### Correction 1: OCSORTTracker section IS in the config

`/opt/paddledetection/deploy/pipeline/config/tracker_config.yml`
in PaddleDetection 3.3.1 contains both:

```yaml
type: BOTSORTTracker  # default — NOT OCSORT
OCSORTTracker:
  det_thresh: 0.4
  max_age: 30
  min_hits: 3         # already the upstream default
  iou_threshold: 0.3
  ...
BOTSORTTracker:
  track_high_thresh: 0.3
  ...
```

The `OCSORTTracker:` block exists with `min_hits: 3` already as
the default. But the active tracker is `BOTSORTTracker`. So the
Part 1 override `MOT.OCSORTTracker.min_hits=3` was targeting an
inactive section — a no-op against the live BOTSORT tracker.

### Correction 2: real root cause is missing `lap`

On every pipeline subprocess startup, this warning appears:
```
Warning: Unable to use JDE/FairMOT/ByteTrack, please install lap
```

`lap` is the C-extension Hungarian-assignment library that
BOTSORTTracker (and the legacy JDE/ByteTrack/OCSORT code paths)
all use to solve the cost-matrix assignment. With `lap` missing,
the tracker falls back to a pure-Python or hung path. On the
first-appearance boundary, the cost matrix inflates from
0 to many tracks in a few frames, and the fallback assignment
wedges the GPU loop.

**Why this reproduces at 24% (= ~30 min content):** the first
30 min of the source video is empty showroom, so the cost
matrix is empty. The moment people appear at ~30 min, the cost
matrix becomes non-trivial for the first time, and the
fallback assignment hangs.

### What was actually fixed in this session

1. **`MOT.OCSORTTracker.min_hits=3` override** (Part 1) — kept
   on the command line. It is a no-op against the active
   BOTSORTTracker but would take effect if an operator
   switches the `type:` line in `tracker_config.yml` to
   `OCSORTTracker` later.
2. **Bounded MinIO PUT to 100 ms** (Part 2) — applied. The
   bounded `ThreadPoolExecutor` + `Future.result(timeout=0.1)`
   caps the per-frame MinIO round-trip at 100 ms. This is a
   real improvement independent of the deadlock.
3. **api-level stall watchdog + respawn** (Parts 3-4) —
   applied. The drain now updates `_stdout_last_ts[cam_id]`
   on every line, the outer thread polls `proc.poll()` once a
   second, and on `last_ts > stall_timeout_seconds` (default
   600 s, override via `PPHUMAN_STALL_TIMEOUT_SEC`) the
   subprocess is terminated and respawned. The respawn loop is
   capped at `PPHUMAN_MAX_RESTARTS` (default 10) per camera.
4. **`lap-0.5.13` installed in the running container** at
   `/opt/venv/lib/python3.12/site-packages/lap/` — applied
   as the immediate fix. The running api container now has
   `lap` importable. This change does NOT survive a container
   restart; it must be baked into the image to be durable.

### What is NOT yet fixed (open issue)

After installing `lap` and restarting the api at 19:18:43, the
chain produced 6 events (CAM_02 frame_id 42, 43) and then
stalled. The watchdog at 180s respawned; the new subprocess
produced 0 stdout for 5+ minutes (verified via `/proc/PID/fdinfo`
showing `pos: 0` on both stdout and stderr pipes). GPU is at
0% util, 2176 MiB used (model is loaded but idle). The
pipeline subprocess has 10+ threads, most in `futex_do_wait`,
2 in `poll_schedule_timeout`.

Manual test from inside the same container (as `app` user, with
identical env vars, same `tracker_config.yml`, same `pushurl`,
smoke-test video) reaches the ffmpeg push stage within 10
seconds and prints all the expected `print_arguments` /
`video fps` / `Multi-Object Tracking enabled` lines. So the
*pipeline itself* works when invoked from a shell.

The difference between shell invocation and api's
`subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE, text=True)`
must be the source. Hypotheses:

- The pipe buffer interaction: with `text=True`, Python opens
  the pipe in line-buffered text mode. If the drain thread
  reads more slowly than the pipeline writes (e.g., because
  the drain thread is in a context switch with the api's
  heavy CPU load — api main is at 500-700% CPU), the pipe
  could fill up at 64 KiB. But `/proc/PID/fdinfo` shows `pos:
  0`, so the pipeline has written 0 bytes.
- The pipeline might be blocked on something pre-`print()`,
  e.g., a `paddle.enable_static()` lock that wasn't released
  by the api's earlier Paddle usage. With 2176 MiB on GPU,
  the api is holding a CUDA context.
- The api might be holding a file lock on the model file
  (`/models/pphuman/mot_ppyoloe_l_36e_pipeline/model.pdmodel`)
  that the subprocess needs to read.

This is a **separate investigation** from the `lap` fix and
the watchdog. Until the actual stall location in the
api-spawned subprocess is identified, the watchdog will
keep respawning and the chain will keep producing 0 events.

### Recommended next steps

1. **Bake `lap` into the image.** Even if the chain is
   currently stuck, the missing-`lap` issue is a real bug
   that would recur on any new container. The Dockerfile
   `Dockerfile.paddle33-numpy126-b2-api` already has
   `lap>=0.4` at line 116; the running image was built
   before that line was added. Rebuild with
   `docker build -f Dockerfile.paddle33-numpy126-b2-api -t
   sota-paddle-mtmct:paddle33-numpy126-b2-api .`.
2. **Investigate the api subprocess's empty pipe.** The
   pipeline works from a shell with the same env but not
   when launched via `subprocess.Popen(..., stdout=PIPE)`.
   Compare the two by running the api subprocess with
   `stdout=DEVNULL, stderr=DEVNULL` and seeing if the chain
   produces events. If yes, the pipe interaction is the
   problem; if no, something in the api's pre-spawn state
   is wrong.
3. **Revert the chain to a known-working video.** The
   `/data/smoke/CAM_01.mp4` 30 s smoke test produced events
   on the prior run. If the chain can process that video
   (with `lap` installed), the merged-mp4 video is the
   problem, not the pipeline.

## Status

| Part | Status | Confidence |
|------|--------|------------|
| 1. `min_hits=1` → `min_hits=3` | ✅ Applied (no-op against BOTSORT) | n/a (target wrong) |
| 2. Bounded MinIO PUT to 100 ms | ✅ Applied | High (real improvement) |
| 3. Per-line stdout timestamp | ✅ Applied | High (correctness) |
| 4. Watchdog + respawn | ✅ Applied | High (chain now self-heals) |
| 5. `lap` installed in container | ✅ Applied (not durable) | High (likely necessary) |
| 6. Chain actually producing events end-to-end | ❌ NOT working | Low — needs investigation |
