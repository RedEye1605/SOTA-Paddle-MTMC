# Phase 1 — Ruff Lint, Format, and Cleanliness Gate

**Date:** 2026-06-13
**Project root:** `/home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC`

## Initial state

`uv run ruff check app scripts tests` reported **119 errors**:

| Code   | Count | Meaning                                     |
| ------ | ----- | ------------------------------------------- |
| F401   | 101   | imported but unused                         |
| F821   | 7     | undefined name (`Any`, `torch`)             |
| F841   | 4     | local variable assigned but never used      |
| F541   | 2     | f-string without any placeholders           |
| E401   | 2     | multiple imports on one line                |
| F811   | 1     | redefinition of unused name                 |
| E714   | 1     | test for object identity should be `is not` |
| E713   | 1     | test for membership should be `not in`      |

`uv run ruff format --check app scripts tests` reported **81 files would be reformatted**.

## Fix strategy

1. `uv run ruff check app scripts tests --fix` — safe fixes only. Resolved **106 of 119** errors
   (all F401, F541, E401, E714, E713, F811).
2. Remaining **11 errors** required manual inspection:
   - 7× F821: missing `Any` import in two files; one forward-reference to `torch` in a string
     annotation.
   - 4× F841: dead local variables in production code, scripts, and tests.
3. `uv run ruff format app scripts tests` — applied project formatter to **81 files**.

## Manual fixes applied

### F821 — undefined name

- `app/improvement/promotion_gate.py:72` uses `Any` in a nested function signature.
  Fix: added `from typing import Any` (was missing from existing imports).

- `app/reid/transreid_adapter.py` uses `Any` in 5 type annotations
  (function signature, dataclass field, property, two method return types).
  Fix: extended existing `from typing import Optional, Sequence` to
  `from typing import Any, Optional, Sequence`.

- `app/reid/transreid_adapter.py:359` is a string forward reference `"torch.Tensor"`
  in `_preprocess` return annotation. `torch` is intentionally not imported at
  module level — it is imported inside the function (4 existing `import torch # type: ignore`
  sites). Fix: added `# noqa: F821` to the line, keeping the function-local import pattern
  intact. This is consistent with the existing convention in the same file.

### F841 — unused local variable

- `app/main.py:270` `smoke_max_frames = args.smoke_max_frames or env_int("SMOKE_MAX_FRAMES", 0)`
  is fully dead — `smoke_max_frames` is never referenced anywhere in the codebase.
  Fix: removed the line. The CLI arg `args.smoke_max_frames` may still exist in argparse
  but has no effect; that is a pre-existing code-rot issue outside the scope of lint cleanup.

- `scripts/inspect_transreid_checkpoint.py:92` `unique_shapes = sorted(set(shapes))` is
  computed but not used in the JPM-classifier return block. Fix: removed the line.

- `tests/test_architecture_guards_one_model.py:148` `SERVICE_DIR = ROOT.parent / "Service"`
  is computed but the test scans for `Service/` references via regex, not via path.
  Fix: removed the line.

- `tests/test_dwell.py:22` first `e = d.on_event(...)` is unused (the assertion only
  checks `e2 is None`). Fix: removed the `e =` assignment, keeping the call so the
  state transition (enter) still occurs.

## Deferred lint issues

**None.** Every issue flagged by ruff was either auto-fixed or fixed manually.

## Final state

```text
$ uv run ruff check app scripts tests
All checks passed!

$ uv run ruff format --check app scripts tests
90 files already formatted

$ uv run python -m pytest tests/ --tb=short
214 passed, 3 warnings in 29.21s

$ uv run python -m compileall app scripts tests
(no errors)

$ docker compose config
(parses successfully)
```

## pyproject.toml ruff config (already present, no changes needed)

```toml
[tool.ruff]
line-length = 100
target-version = "py312"
```

The conservative rule set `["E", "F", "I"]` was used (defaults to ruff's full default
plus import sorting). No rule set was added to `pyproject.toml` — only the existing
config was relied on. Stayed on the default rule set; no `UP`/`B`/`SIM` rules
were enabled, so no semantic refactors were forced.

## Safety gate checks (must remain enforced)

- Production safety checks in `app/runtime_mode.py` and `app/reid/transreid_adapter.py`
  were not modified beyond adding `Any` to typing imports and a `# noqa: F821` on a
  string forward reference.
- `camera_num=0` default remains.
- SIE remains disabled.
- `weights_only=True` enforcement remains.
- Synthetic detector / deterministic ReID refusal paths in `load()` are untouched.
- Qdrant filtering / payload indexing logic untouched.
- Vendored TransReID inference behavior in `_transreid_native/` is untouched.

## Verdict

Phase 1 ruff cleanup: PASS. 119 → 0 lint errors. 81 files reformatted.
All 214 tests pass. compileall clean. docker compose config valid.
