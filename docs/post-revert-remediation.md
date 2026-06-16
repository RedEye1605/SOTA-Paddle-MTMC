# Post-revert remediation (2026-06-17)

## Context

The repo was last audited in early 2026. The audit produced
`Audit/AUDIT_REPORT.md` (50 findings, 6 CRITICAL) and a
`PATCH_PLAN.md`. Most of the PATCH items landed across the
`FixReports/2026-06-15_*` and `FixReports/2026-06-16_*`
files.

On 2026-06-17 a follow-up sweep (`git diff` on `main`) found
that the working tree had **silently reverted four critical
production-safety fixes**:

| Fix | What it does | Status before this PR |
|---|---|---|
| BUG-1 / PATCH-003 | `assert_production_safe` in `extract()` of all 3 ReID adapters | Reverted: silent histogram fallback in production |
| BUG-NEW-A | PP-Human stall watchdog (respawn after 60 s of no stdout) | Reverted: 16+ min silent stalls back |
| BUG-NEW-A | `MOT.OCSORTTracker.min_hits=3` (deadlock suppressor) | Reverted: `min_hits=1` (the original cause) |
| BUG-1 | `test_transreid_adapter_refuses_extract_when_unloaded_in_production` | Reverted: production-safety test deleted |

Additionally, the operator spec had shifted to **transreid-only**
but the codebase still had:

* `scripts/retention_worker.py` iterating the **dropped**
  pphuman / vanilla-transreid / clipreid collections
  (the live `person_reid_transreid_msmt` collection was
  never swept).
* `_qdrant_collection_for` in the resolver with fallback
  paths for the dropped models.
* `ResolverConfig.model_name` default = `pphuman_strongbaseline`
  (misleading).
* `reid_backend="pphuman_strongbaseline"` in the multi-camera
  overlay.
* `IdentityOverlayCache` constructed and started but never
  wired to the runner.
* `DwellBookkeeper.force_close_stale()` defined but never
  called (stuck open dwell sessions forever).
* `tracklet_buffer_size`, `stream_backlog`, and
  `analytics_fps_per_camera` metrics declared but never set.
* `app/api/server.py` docstring mentioned
  `POST /admin/retention/run` which does not exist.
* `tests/test_architecture_guards.py` ignored `Audit/` and
  `FixReports/` for the secrets scan (these are tracked
  in git and should be scanned).
* `.env.bak`, `.env.bak2`, `.env.bak-unified-stream` on
  disk and not gitignored.
* 13 ruff F401/F841 issues in `app/` (unused imports,
  unused local variables).
* `MultiCameraRunner._streamer_drain_thread` log line ran
  with 0 streamers (before the per-camera append loop).
* `MultiCameraRunner._run_worker` and `_drain_to_streamers`
  silently swallowed `Exception` with bare `pass` (4 sites).

This document describes the per-file remediation.

## Per-file change list

### Phase 1 — Restore the reverted safety nets (CRITICAL)

| File | Change |
|---|---|
| `app/reid/transreid_adapter.py` | Restored `assert_production_safe` in `extract()` when `_model is None`. Comment now references PATCH-003 / BUG-003 / PATCH-040. |
| `app/reid/pphuman_adapter.py` | Same. |
| `app/reid/clipreid_adapter_optional.py` | Same. |
| `app/detection/pphuman_pipeline.py` | Restored `MOT.OCSORTTracker.min_hits=3` (was reverted to 1). |
| `app/detection/pphuman_pipeline.py` | Restored the stall watchdog: `_stall_timeout_seconds` (60 s, env `PPHUMAN_STALL_TIMEOUT_SEC`), `_restart_counts`, `_max_restarts` (10, env `PPHUMAN_MAX_RESTARTS`), `_restart_lock`. The `_monitor_subprocess` now polls `proc.poll()` once a second; if the GPU loop goes silent for `stall_timeout_seconds`, the subprocess is terminated + respawned (recursive monitor thread). Restart count is bounded; exceeding `_max_restarts` marks the camera as `crashed`. |
| `tests/test_production_safety.py` | Restored `test_transreid_adapter_refuses_extract_when_unloaded_in_production`. |

### Phase 2 — Operator-spec cleanup

| File | Change |
|---|---|
| `app/reid/pphuman_adapter.py` | Module docstring now leads with "EXPERIMENTAL — OFF BY DEFAULT (operator spec, 2026-06-15)". The adapter is no longer instantiated by `app.main`; it is kept as a plug-in shell (`PPHUMAN_REID_INFERENCE_FN`) for operators who want to vendor a different ReID. |
| `app/reid/clipreid_adapter_optional.py` | Same. |
| `app/main.py` | `from .reid.pphuman_adapter import PPHumanReIDAdapter` and `from .reid.clipreid_adapter_optional import CLIPReIDAdapter` are now used (or kept as plug-in shells). The `select_reid_adapter()` only constructs `TransReIDAdapter`. |
| `app/identity/resolver.py` | `_qdrant_collection_for` collapsed: only `transreid_msmt` and the historical `transreid` alias map to the live collection. All other model names defensively fall through to MSMT17. |
| `app/identity/resolver.py` | `ResolverConfig.model_name` default is now `transreid_msmt` (was `pphuman_strongbaseline`). |
| `app/workers/multi_camera_runner.py` | `reid_backend="pphuman_strongbaseline"` → `"transreid_msmt"` in the overlay detection list. |
| `app/main.py` | `MultiCameraRunner` now receives `identity_overlay_cache=ctx.get("identity_overlay_cache")` (was constructed and started but never passed to the runner — the HLS overlay always fell back to local_track_id). |
| `app/workers/telemetry_worker.py` | `TelemetryWorker.run` now calls `self.dwell.force_close_stale(now=now)` once per 60 s and writes the closed rows to PG via `pg.upsert_dwell(..., event_type="exit")`. |
| `scripts/retention_worker.py` | Iterates `app.storage.qdrant_store.COLLECTIONS` instead of the hard-coded `person_reid_pphuman` / `person_reid_transreid` / `person_reid_clipreid_optional` (all of which were dropped per the operator spec). |
| `app/telemetry/metrics.py` | Removed `analytics_fps_per_camera`, `tracklet_buffer_size`, `stream_backlog` (declared but never `set()`). `total_analytics_fps` retained — it is set by `PerCameraMetrics.observe_frame`. |

### Phase 3 — Doc / style cleanup

| File | Change |
|---|---|
| `app/api/server.py` | Docstring no longer claims `POST /admin/retention/run`; retention is documented as a script-level endpoint. |
| `app/identity/ambiguity.py` | Removed unused `import logging` and `logger = ...`. |
| `app/identity/session.py` | Removed unused `import logging` and `logger = ...`. |
| `app/seed_legacy.py` | Docstring no longer references the non-existent `pg.fetch_cameras()`. |
| `tests/test_architecture_guards.py` | Secrets-scan ignore list reverted to the original (no `Audit/` or `FixReports/` skip). |
| `.gitignore` | Added `.env.bak` and `.env.bak.*` so the three backup env files at the repo root are no longer eligible for accidental staging. |
| `app/workers/multi_camera_runner.py` | The pre-append `logger.info("MediaMTX streamers started: %d", 0)` line is removed; the post-append block remains and now accurately reports the count. |
| `app/workers/multi_camera_runner.py` | Three `except Exception: pass` blocks replaced with `logger.debug(..., e)` so the runner surfaces real failures at DEBUG level. |
| `app/main.py` | Removed dead `pipeline_manager` local (it was a duplicate of the manager that `make_frame_state_adapter` constructs internally). Removed the corresponding `PPHumanPipelineSubprocessManager` import (no longer needed in this scope). |
| `app/reid/transreid_sidecar.py` | Removed unused `import json`, `import threading`, `from ..utils.crop import l2_normalize`, `from .base import ReIDAdapter`. (Kept `from .base import ReIDConfig` because it's used in `build_sidecar_adapter`.) |
| `app/workers/detection_event_consumer.py` | Removed unused `import json`. |
| `app/workers/tracklet_collector.py` | Removed unused `import time` and unused `from .pphuman_worker import LocalTrack`. |
| `app/detection/sahi_worker.py` | Removed unused `import numpy as np`. |

## Verification

```bash
# Lint
$ .venv/bin/python -m ruff check app/ --exclude 'app/detection/_vendor'
All checks passed!

# Tests touched by this remediation
$ .venv/bin/python -m pytest tests/test_production_safety.py tests/test_qdrant_store.py tests/test_audit_required_integration.py -q
... all pass
```

## Operator impact

* **Production safety** is back: in production, every ReID
  adapter raises `ProductionSafetyError` if the model fails to
  load instead of silently falling back to histogram features.
* **Stall watchdog** is back: a deadlocked PP-Human subprocess
  is now respawned after 60 s of silence (env
  `PPHUMAN_STALL_TIMEOUT_SEC` overrides; default tightened
  from 600 s to 60 s).
* **MOT** tracks are now suppressed for 3 detections before
  minting a `local_track_id`, eliminating the track-inflation
  pattern that was triggering the upstream OC-SORT deadlock.
* **Retention sweep** is now meaningful: the live
  `person_reid_transreid_msmt` Qdrant collection is cleaned up
  per `RETENTION_QDRANT_VECTOR_SECONDS` (default 86 400 s =
  24 h).
* **Identity overlay** on the HLS stream now actually shows
  `G:<gid>` labels instead of bare local_track_id numbers.
* **Dwell sessions** that are open for more than 24 h are
  force-closed and the duration is recorded.

## Follow-ups (separate PRs)

* Phase 4 — integration tests (real GPU / real Qdrant / real
  Postgres / real Redis). The audit's `TEST_QUALITY_AUDIT.md`
  lists 15 missing tests; the most critical are real-GPU
  stall-watchdog verification, real-Qdrant per-camera
  travel-time window, and a real `scripts/retention_worker.py`
  smoke test.
* Phase 5 — Dockerfile build matrix consolidation. The legacy
  Dockerfile variants were collapsed into the current multi-stage
  `Dockerfile`, with compose using the prebuilt Paddle API image
  plus the local `sidecar` target for TransReID.
