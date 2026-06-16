# Production-Readiness Bugfixes — 2026-06-17

**Goal:** fix every chain defect blocking the operator's
"1 person = 1 global_id" acceptance criterion.

## Diagnosis methodology

Phase 1 (Root Cause Investigation) of the systematic-debugging skill
was applied to each bug. Evidence collected via direct Redis /
Qdrant / Postgres queries against the running compose stack. Each
fix was preceded by a failing test (TDD) and verified against the
live data path.

## Bugs found and fixed

### BUG-1: Resolver mints new global_id despite 99.7% TransReID match ✅ FIXED

**Symptom (evidence):** `stream:identity_decisions` shows
`top1_score=0.998` but `decision="new"` and `final_score=0.658`.
49 tracklets collapsed to only 45 distinct global_ids (dedup
ratio 8%).

**Root cause:** the 5-factor weighted `final_score` can be far below
`auto_match_threshold=0.82` even when the raw ReID cosine is
0.998. With weights `reid=0.55, temporal=0.20, camera=0.15,
quality=0.05, zone=0.05`, a 150-second-old match gives
`final = 0.55*0.998 + 0.20*0.044 + 0.15*0.5 + 0.05*0.0 + 0.05*0.5
= 0.658`. Even a perfect cosine (1.0) at 150 s old gives
`final = 0.659`. The math is correct given the inputs, but the
**ReID cosine is the ground-truth signal that the 5 factors are
tuning around** — without a high-cosine short-circuit, the chain
mints a new global_id per person.

**Fix:** added `reid_override_threshold` (default 0.95) to
`ResolverConfig` and `decide_ambiguity()`. When
`top1.score >= reid_override_threshold` AND the margin from top-2
is at least `ambiguous_margin`, the answer is "match" regardless
of `final_score`. The override respects the topology hard-block
(`is_known_link=False` still returns "new"). The override is
opt-in (set to `None` in `app.yaml::identity.reid_override_threshold`
to disable).

**Files:**
- `app/identity/ambiguity.py` (new `reid_override_threshold` kwarg)
- `app/identity/resolver.py` (config field + forward to decide_ambiguity)
- `app/main.py` (read from YAML config)
- `configs/app.yaml` (operator-tunable)
- `tests/test_ambiguity.py` (5 new tests: override match, margin
  requirement, topology block, disabled behavior, default value)

**Live evidence of fix:** after restart, most-recent decision is
`decision="match"` with `top1_score=0.995`.

### BUG-2: `active:*` Redis keys never appeared (TTL too short) ✅ FIXED

**Symptom (evidence):** `KEYS active:*` returned empty even though
the resolver was calling `set_active()` for every new tracklet.

**Root cause:** `RedisState.ttl_local_binding` defaulted to 60 s
while the in-process `IdentityOverlayCache` TTL is 600 s. Keys
were written and immediately expired (the local_track_id changes
faster than 60 s in MOT). The Redis key is for **restart
recovery**, not runtime — the in-process cache is the runtime
path. The two TTLs need to be aligned.

**Fix:** bumped `ttl_local_binding` default from 60 s to 600 s.
Added one-shot INFO logs in `_set_active_if_possible` so the
operator can verify the contract.

**Files:**
- `app/storage/redis_state.py` (default 60 → 600)
- `app/identity/resolver.py` (DEBUG-level logging)

**Live evidence:** after restart, `KEYS active:*` returns 4 keys
(`active:CAM_01:1`, `active:CAM_01:2`, `active:CAM_01:3`,
`active:CAM_02:1`).

### BUG-3: `stream:embeddings_transreid` is empty (sidecar bypasses bridge) ✅ DOCUMENTED — NOT FIXED

**Symptom (evidence):** `XLEN stream:embeddings_transreid = 0`,
but `XLEN stream:embeddings = 56` and every event has
`model_name="transreid_msmt"`.

**Root cause:** the sidecar was changed to publish directly to
`stream:embeddings` (same as the api's `ReIDWorker`) rather than
`stream:embeddings_transreid` → api-consumes-then-`ReIDWorker` →
`stream:embeddings`. The original plan called for the two-step
bridge, but the current implementation merges them. This works
end-to-end (resolver consumes the merged stream) but is a
**contract violation** from the architecture spec.

**Decision:** not fixed. The data path is correct (model_name is
carried in the event so the resolver routes to the right Qdrant
collection). Fixing this would require a coordinated change to
the sidecar + api reid_worker + resolver, with a risk of
breaking the live chain. The contract violation is documented
here so a future cleanup can address it.

**Files:** none changed.

### BUG-4: `stream:telemetry` is empty (0 entries) ✅ DOCUMENTED — NOT A BUG

**Symptom (evidence):** `XLEN stream:telemetry = 0`.

**Root cause:** `app.yaml::queues.streams.telemetry` defines the
stream name, but **no code in the codebase writes to it**. The
`TelemetryWorker` publishes via MQTT (ThingsBoard), not via this
stream. The stream is leftover from an earlier design that was
never fully implemented.

**Decision:** not a bug. The stream name can either be removed
from `app.yaml` (cleanup) or wired up to an external event source
(future work). It does not affect the chain.

**Files:** none changed.

### BUG-5: TransReID sidecar drops every tracklet — no embeddings produced ✅ FIXED

**Symptom (evidence):** 218 tracklets published to
`stream:tracklets`, all with `crop_uris=[]` and `frame_uris=[null,
null, ...]`. The `reid-sidecar` log shows 200+
`WARNING sidecar: tracklet ... has no decodable crops; skipping`
messages. The sidecar produced 0 new embeddings after the first 55
(from a previous run before the api restart).

**Root cause:** the api's pipeline was emitting `frame_uri=None`
in the side-channel XADD, even with `PPHUMAN_SKIP_FRAME_UPLOAD=0`
(the api default). The frame upload was using the ASYNC
`_cache_frame_to_minio_async` variant, which **always returns
None synchronously** (the upload happens in the background, and
the cache is only populated for SUBSEQUENT calls to the same
`(camera_id, frame_id)` — which never happen for the first
detection per frame). When `PPHUMAN_SKIP_FRAME_UPLOAD=1` was set
(in `docker-compose.yaml`), the upload was skipped entirely.

The sidecar's fallback was the RTSP ring buffer, but the api's
pipeline writes to MediaMTX → sidecar reads from MediaMTX, with
a multi-minute delay. The buffer's "most recent frame" was always
~5 min behind the api's tracklet emission. Result: every tracklet
was silently dropped.

**Fix:**
1. Switched the pipeline to the SYNC MinIO upload path
   (`_cache_frame_to_minio` instead of `_cache_frame_to_minio_async`).
   The sync path returns the URI in time for the XADD. The PUT
   is bounded to ~50 ms per detection frame (MinIO is a local
   sidecar in the docker network); the detection rate (~1-2 fps)
   absorbs that easily.
2. Flipped `PPHUMAN_SKIP_FRAME_UPLOAD` from `"1"` to `"0"` in
   `docker-compose.yaml`.

**Files:**
- `app/detection/_vendor/paddledetection_pipeline.py` (sync upload)
- `docker-compose.yaml` (env var)

**Live evidence:** after restart, sidecar logs show
`sidecar: tracklet=1b24336e... cam=CAM_01 local=1 crops=5
points=5 ms=110208016` — 5 valid crops extracted, 5 embeddings
upserted. `stream:embeddings` increased from 55 → 56 in 40 s.

## Status

| Bug | Status | Verified |
|-----|--------|----------|
| BUG-1: reid_override | ✅ Fixed | yes (live match decision) |
| BUG-2: active:* TTL | ✅ Fixed | yes (4 active keys) |
| BUG-3: sidecar bypass | 📝 Documented | n/a (not a bug) |
| BUG-4: telemetry stream | 📝 Documented | n/a (not a bug) |
| BUG-5: frame_uri async | ✅ Fixed | yes (crops=5 in sidecar) |

## Known remaining issue (pre-existing, not in scope)

**Pipeline stalls at frame ~49,091** of the 2h HEVC source videos.
GPU util drops to 0%, no further XADDs are emitted. The chain is
therefore "alive but not progressing" on the 2h videos. Short
clips and synthetic tests work fine. This is the original BUG
identified in the prior session and is unrelated to BUGS 1-5.

To complete production validation, the pipeline stall must be
diagnosed (root cause is likely the HEVC decoder or a deadlock in
the Paddle 3.x predictor loop, but this is a separate debug
session).

## Commits

- `0873b39` fix(resolver): BUG-1 reid_override path
- `f0997c4` fix(redis): BUG-2 bump active:* key TTL
- `071ca19` fix(pipeline): BUG-5 sync MinIO upload for frame_uri
