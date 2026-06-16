# ReID 24-Hour Audit — SOTA-Paddle-MTMC

> **Phase 5 — ReID 24h audit.** Verifies persistent ReID across
> a 24-hour window, including staged retrieval, fallback handling,
> and audit records.

## What "24h persistence" means in this codebase

1. `Qdrant` is the gallery. Embeddings have a `timestamp` payload
   field (Unix seconds, int). The resolver search filter
   constrains `timestamp >= ts - 86400`.
2. `PostgreSQL.global_identities` is the source of truth. The
   `expire_old_identities(older_than_seconds)` method marks
   identities older than the window as `status='expired'`.
3. `Redis` has a 24h TTL on `recent:global:{global_id}` keys
   (86400 s) and `camera:last_seen:{camera_id}`. After 24h, the
   key expires; the resolver does not consult Redis for old
   identities (it only uses Redis for "last seen" and
   "active binding").

## Confirmed behavior (with evidence)

| # | Behavior | Evidence |
|---|---|---|
| C1 | Qdrant search uses `timestamp_gte` (24h window). | `app/storage/qdrant_store.py:111-138` |
| C2 | Qdrant search requires `model_name` and `model_version` filters (prevents mixing embeddings from incompatible models). | `app/storage/qdrant_store.py:135-138` |
| C3 | Qdrant search short-circuits on empty `candidate_camera_ids` (no global blind scan). | `app/storage/qdrant_store.py:127-128` |
| C4 | Each ReID model has its own Qdrant collection with the right `embedding_dim` (256 / 768 / 512). | `app/storage/qdrant_store.py:24-28` |
| C5 | Embeddings are L2-normalized (`l2_normalize` in `app/utils/crop.py`) so cosine distance is meaningful. | `app/utils/crop.py:57-67` |
| C6 | The mean-of-tracklet-embeddings is re-normalized (`mean_normalized`). | `app/utils/crop.py:70-76` |
| C7 | Low-quality crops are rejected by `crop_quality_score` returning 0 → no upload, no embedding. | `app/utils/image_quality.py:61-104` |
| C8 | Every identity decision is logged to `identity_decisions` (full 5-factor breakdown). | `app/storage/postgres.py:275-303` |
| C9 | Manual merge is supported via `identity_merge_audit` table (operator-driven). | `db/migrations/002_identity_tables.sql:121-132` |
| C10 | Identity expiration is supported via `expire_old_identities(older_than_seconds)`. | `app/storage/postgres.py:351-363` |
| C11 | Ambiguous candidates are NOT auto-merged (decision = "ambiguous" → `assigned_global_id=None`). | `app/identity/resolver.py:219-223` |
| C12 | Production ReID is *distinguishable* from fallback: `_fallback_active` is a flag on the adapter, but it is logged at WARNING. (See BUG-033 — should be ERROR.) | `app/reid/transreid_adapter.py:42-45` |

## Failed / partial behavior (with evidence)

| # | Behavior | Evidence |
|---|---|---|
| F1 | **The 24h fallback Stage 3 of staged retrieval is NOT implemented.** `_candidate_cameras()` returns only the source camera (Stage 1) and topology-linked cameras (Stage 2). The docstring and `Docs/architecture.md` claim a Stage 3 "24h fallback" that searches the global gallery outside topology. | `app/identity/resolver.py:79-99`; `Docs/architecture.md:39-42` |
| F2 | **Cross-camera search is not filtered by travel-time window.** A 23h-old CAM_01 candidate is returned for a CAM_02 tracklet even when `min_travel_seconds=10, max_travel_seconds=90` says it's impossible. | `app/identity/resolver.py:111-119` |
| F3 | **All ReID extractions in the test/audit paths are histogram features, not real ReID.** A "feature" derived from `quality_score` and HSV histograms has zero discriminative power between visually similar people. The 24h persistence is technically correct (the feature is stored for 24h) but practically useless. | `app/reid/transreid_adapter.py:88-111`; `app/workers/reid_worker.py:62-69` |
| F4 | **The `transient_fallback` is logged at WARNING, not ERROR.** A operator asleep at the wheel will not notice. | `app/reid/transreid_adapter.py:42-45` |
| F5 | **The architecture has no startup guard that refuses to start in `mode=multi_rtsp` if `_fallback_active` is True.** | `app/main.py:111-114` |
| F6 | **Embedding dimension mismatch is not checked at write time.** If someone misconfigures `active_model=transreid` but the Qdrant collection `person_reid_transreid` was never initialized, the resolver will silently write to the wrong collection. (Actually: `_qdrant_collection_for()` returns the right collection per model_name. OK.) | `app/identity/resolver.py:121-127` |
| F7 | **The shipped TransReID weight is MSMT17 (`vit_transreid_msmt.pth`) but the `download_transreid_models.sh` script downloads Market-1501 (`transformer_120.pth`).** Different `num_class` heads. | `models/`, `scripts/download_transreid_models.sh:19-21` |
| F8 | **`reid_extractions_total` counter is never incremented.** No `REGISTRY.reid_extractions.inc()` call exists anywhere. The `metrics` endpoint reports 0 forever. | `app/telemetry/metrics.py:122-124` |
| F9 | **`qdrant_query_latency_seconds` and `postgres_write_latency_seconds` histograms are never observed.** | `app/storage/qdrant_store.py:152-153`; `app/storage/postgres.py:97-102` |
| F10 | **Identity fragmentation is not analysed.** No metric for "same person, multiple GIDs across 24h" is computed anywhere. The `benchmark_plan.md` says `id_fragmentation_rate` is a required metric, but the benchmark script is a stub. | `scripts/benchmark_t4.py:21-43` |
| F11 | **The tracklet collector emits a `mean_vec` to `stream:embeddings` as a JSON list of floats.** This is large (~6 KB per embedding for 768-dim). It bloats Redis memory and serializes slowly. | `app/workers/reid_worker.py:113-115` |
| F12 | **The ReID worker's `process_tracklet` is called from a `stream:tracklets` consumer, but the resolver is an in-process function that is NOT called from the stream.** So the resolver is never actually run end-to-end in production. The code path is `for msg in msgs: tl = …; self.process_tracklet(tl); ack` — but the resolver is never invoked. | `app/workers/reid_worker.py:124-148` |

## PPHUMAN_INFER_FN plug-in path

**Status: not implemented.**

The README and `Docs/research_sources.md` mention
`PPHUMAN_REID_INFERENCE_FN` and `TRANSREID_MODEL_FN` as env vars
that the operator can set to plug in a custom inference function.
**Neither env var is read anywhere in the codebase.** A
`grep -r "PPHUMAN_REID_INFERENCE_FN\|TRANSREID_MODEL_FN" SOTA-Paddle-MTMC/`
returns zero hits.

This means the operator CANNOT plug in a real PaddleInference
session. The fallback is permanent.

## Deterministic fallback embeddings

**Status: always active in any environment that does not have the
real model files at exactly the right paths.**

`PPHumanReIDAdapter._try_load_paddle`:
```python
weight_dir = self._weight_dir or os.environ.get("PPHUMAN_MODEL_DIR", "/models/pphuman")
if not Path(weight_dir).exists():
    raise FileNotFoundError(f"PPHuman model dir not found: {weight_dir}")
try:
    import paddle
except Exception as e:
    raise RuntimeError(f"paddlepaddle not installed: {e}")
raise RuntimeError("PP-Human inference function not configured. Set PPHUMAN_REID_INFERENCE_FN to a Python callable, or accept the deterministic fallback (smoke tests only).")
```

Every exception path leads to `self._fallback_active = True`.
The only way to avoid the fallback is:
1. Install `paddlepaddle-gpu` (not in `requirements.txt`).
2. Provide `/models/pphuman/mot_ppyoloe_l_36e_pipeline` (the
   inference config expects this exact directory).
3. Set `PPHUMAN_REID_INFERENCE_FN` to a Python callable (which
   is never read).

In a fresh environment, even with the model weights, the
fallback is the only path.

## Synthetic detector in smoke test

**Status: ALWAYS the production path, because the model-load
is commented out and the `detector_factory` is never wired.**

`multi_camera_runner.py:80-86` constructs:
```python
worker = PPHumanWorker(
    camera_id=cam.camera_id,
    frame_reader=reader,
    skip_frame_num=self.skip_frame_num,
    smoke_test_mode=self._smoke_test_mode,
)
```
There is no `detector=…` parameter. `smoke_test_mode=True` is
the default. The synthetic detector is the *only* detector.

## Production startup rejection of fallback mode

**Status: not implemented.**

`build_app_context()` always loads the adapter, logs WARNING on
fallback, and continues. There is no `if smoke and
adapter._fallback_active: return; if not smoke and
adapter._fallback_active: raise RuntimeError("ReID fallback
in production — abort")`.

## Log loudness on fallback

**Status: WARNING (should be ERROR, see BUG-033).**

```python
logger.warning("TransReID weights not loaded (%s). Using
deterministic 768-dim fallback. This is for smoke tests ONLY.", e)
```

A grep for `fallback` in the running logs (if you have them)
will find this; an operator who is not actively reading logs
will not.

## ReID 24h verdict

**24h persistence as a *database concept* is correct and complete.**
The schema, the Qdrant filters, the Redis TTLs, the audit
tables are all in place. A test running against synthetic
histogram features will pass; a test running against real
Paddle+TransReID will produce meaningful results.

**24h persistence as a *production feature* is not implemented.**
The ReID adapter falls back to a histogram. The detector falls
back to a random box. The reid_worker fabricates crops. The
resolver is never invoked from the stream consumer (the
"identity_decisions" stream has no publisher).

**Verdict: STRUCTURALLY READY, NOT PRODUCTION-READY.**

To make it production-ready:
1. Implement real Paddle PP-Human + TransReID inference.
2. Implement `ReIDWorker` → `Resolver` → `PG/Redis/Qdrant` chain
   end-to-end (the resolver is currently a class but not
   wired into the stream consumer).
3. Add Stage 3 (24h global fallback outside topology).
4. Add travel-time filter in Qdrant payload filter.
5. Block startup if `smoke=False and adapter._fallback_active`.
6. Add operator-review queue for "ambiguous" decisions.
