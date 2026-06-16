# Phase 0 — Current State Verification

**Date:** 2026-06-13
**Project root:** `/home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC`

## Verification commands run

```text
uv run python -m pytest tests/ -q
uv run python -m compileall app scripts tests
docker compose config

uv run python -c "from app.storage.postgres import PostgresStore"
uv run python -c "from app.runtime_mode import RuntimeMode"
uv run python -c "from app.reid.transreid_adapter import TransReIDAdapter"
uv run python -c "from app.reid._transreid_native import vit_base_patch16_224_TransReID"
uv run python scripts/inspect_transreid_checkpoint.py models/vit_transreid_msmt.pth --profile msmt17
```

## Results

| Check                            | Result                                | Expected          | Pass |
| -------------------------------- | ------------------------------------- | ----------------- | ---- |
| pytest count                     | 214 passed, 3 warnings, 29.21s        | 214               | YES  |
| compileall                       | Clean (no errors)                     | Clean             | YES  |
| docker compose config            | Parses successfully                   | Parses            | YES  |
| PostgresStore import             | `postgres ok`                         | Imports           | YES  |
| RuntimeMode import               | `runtime_mode ok` + enum listing      | Imports           | YES  |
| TransReIDAdapter import          | `transreid_adapter ok`                | Imports           | YES  |
| vit_base_patch16_224_TransReID   | `native transreid ok`                 | Imports           | YES  |
| TransReID checkpoint inspector   | `ok: True / reason: compatible`       | Compatible        | YES  |

## TransReID checkpoint details (verified)

```text
path: models/vit_transreid_msmt.pth
exists: True
size_bytes: 419206362  (~419 MB)
expected_profile: msmt17
num_tensors: 211
has_backbone: True
classifier: {kind: cls_only, shape: [1041, 768], num_class: 1041, embedding_dim: 768}
detected_num_class: 1041
detected_profile: msmt17
ok: True
reason: compatible
```

## Notes

- The inspector script takes the path as a positional argument, not `--checkpoint`. The first run failed with
  `unrecognized arguments: --checkpoint` and the corrected invocation succeeded.
- Test count is exactly 214, matching the previous operator handoff.
- The 3 warnings are not test failures:
  1. `StarletteDeprecationWarning` for `httpx` (from fastapi TestClient).
  2. and 3. `PytestUnhandledThreadExceptionWarning` from `test_benchmark_smoke_runs_with_fake_sources` — these are
     expected because the test intentionally uses `stub://cam01` and `stub://cam02` URLs that cannot be opened as
     real video sources. The exceptions are caught inside the worker and do not fail the test.

## Verdict

Phase 0 verification: PASS. No drift from prior state. Proceeding to Phase 1.
