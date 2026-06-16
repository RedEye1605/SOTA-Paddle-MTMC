"""Hard contract: SAHI tiling must not pull an external DL stack.

The API image stays Paddle-only. The local ``SahiDetector`` implements
frame slicing directly, so runtime manifests must not include the
external ``sahi`` package or torch-family dependencies.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_RUNTIME_DEPS = {"sahi", "torch", "torchvision", "tensorflow", "jax"}


def _dependency_name(spec: str) -> str | None:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", spec)
    if match is None:
        return None
    return match.group(1).lower().replace("_", "-")


def test_runtime_manifests_do_not_pull_external_sahi_stack() -> None:
    names: set[str] = set()

    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = _dependency_name(line)
        if name is not None:
            names.add(name)

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    for spec in pyproject["project"]["dependencies"]:
        name = _dependency_name(spec)
        if name is not None:
            names.add(name)
    for extra_specs in pyproject["project"].get("optional-dependencies", {}).values():
        for spec in extra_specs:
            name = _dependency_name(spec)
            if name is not None:
                names.add(name)

    assert not (names & FORBIDDEN_RUNTIME_DEPS)


def test_internal_sahi_import_does_not_import_external_stack() -> None:
    watched = {"sahi", "torch", "torchvision", "tensorflow", "jax"}
    before = {
        name
        for name in sys.modules
        if any(name == root or name.startswith(f"{root}.") for root in watched)
    }

    from app.detection.sahi_detector import SahiDetector  # noqa: F401

    after = {
        name
        for name in sys.modules
        if any(name == root or name.startswith(f"{root}.") for root in watched)
    }
    assert not (after - before)
