# Ruff quality gate

> **Lint + format gate.  Run before every commit, in every CI
> step, and after any rebase.  All checks ship clean as of
> 2026-06-13.**

## Commands

```bash
uv run ruff check app scripts tests
uv run ruff format --check app scripts tests
```

## Auto-fix

```bash
uv run ruff check app scripts tests --fix
uv run ruff format app scripts tests
```

## Verified clean state (2026-06-13)

```text
uv run ruff check app scripts tests
  -> All checks passed!

uv run ruff format --check app scripts tests
  -> 90 files already formatted
```

## What was fixed in Phase 1

Initial state of the repo: **119 ruff errors**:

| Code   | Count | Meaning                                     |
| ------ | ----- | ------------------------------------------- |
| F401   | 101   | unused import                               |
| F821   | 7     | undefined name                              |
| F841   | 4     | local variable assigned but never used      |
| E713   | 2     | `not in` test                               |
| E714   | 2     | `is not` identity test                      |
| E401   | 2     | multiple imports on one line                |
| W292   | 1     | missing trailing newline                    |

All resolved with a mix of `ruff check --fix` (autofix) and
hand edits for the `F821` / `F841` cases that needed an
intentional import (typing.Any) or removal of dead variables.

## Config

`pyproject.toml` ships ruff settings:

```toml
[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

Add to this only with a written justification.

## Conventions enforced

* Imports: ruff's default sort order (stdlib → third-party →
  local), one import per line, no `from X import *`.
* Line length: 100.  Wrap long strings via implicit
  string concatenation.
* Trailing newline on every file.
* No unused imports.  Use `# noqa: F401` for re-export shims.
* No unused local variables.  Either consume the value or drop
  the assignment.

## Forward-looking deferrals

The current ruleset deliberately does NOT enable:

```text
ANN  (type annotations)        -> high churn, low immediate value
PT   (pytest conventions)      -> existing tests use multiple styles
N    (PEP 8 naming)            -> some legacy names matched against
                                  PaddleDetection naming
TRY  (try-except style)        -> would require auditing every
                                  noqa: BLE001 site
```

If a future phase decides to enable any of these, the migration
must:

1. Land in its own PR with a clear before/after diff.
2. Include an updated `pyproject.toml`.
3. Pass `uv run ruff check --select <new>` cleanly on the
   first try.

## Pre-commit (optional)

The repo does not ship a `pre-commit` config.  Operators who
want one can add:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.31
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
```
