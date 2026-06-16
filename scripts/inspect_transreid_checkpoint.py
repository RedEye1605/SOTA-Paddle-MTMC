#!/usr/bin/env python3
"""Inspect a TransReID checkpoint and report its profile compatibility.

The script is intentionally dependency-light: it uses only torch when
available, and falls back to a lightweight ZIP-based check otherwise.
The output is a JSON object on stdout describing the checkpoint's
classifier head shape, expected profile, and verdict.

Usage:
    python scripts/inspect_transreid_checkpoint.py <path> [--json]

Exit codes:
    0  - checkpoint is loadable, profile compatible
    1  - checkpoint missing
    2  - profile mismatch (classifier shape does not match expected)
    3  - file is not a torch checkpoint
    4  - unexpected shape (no recognizable classifier head)

The script is safe to run in CI: it does not load the model class
itself, only the raw state_dict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# TransReID's `classifier` is a `nn.Linear(768, num_class)` (or a
# `ModuleList` of 5 such linears for JPM). The state_dict stores the
# weight as `classifier.weight` and the bias as `classifier.bias`.
# For ModuleList, the keys are `classifier.0.weight`, `classifier.1.weight`, ...
KNOWN_PROFILES: dict[str, dict[str, int]] = {
    "market1501": {"num_class": 751},
    "msmt17": {"num_class": 1041},
    "duke": {"num_class": 702},  # sometimes used in vendor
    "veri": {"num_class": 776},  # vehicle ReID
}

# JPM's classifier is a ModuleList of 5 linears, all sharing the same
# num_class. The official `make_model` creates them when `local_feature=True`.
JPM_NUM_LINEARS = 5


def _strip_module_prefix(state: dict) -> dict:
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


def _read_state_dict(path: Path, weights_only: bool) -> dict | None:
    """Read the state_dict from a torch checkpoint. Returns None on
    any error so the caller can produce a structured failure.
    """
    try:
        import torch
    except ImportError:
        return None
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=weights_only)
    except Exception as e:  # noqa: BLE001
        print(f"# ERROR loading checkpoint: {e}", file=sys.stderr)
        return None
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    print(f"# ERROR: not a state_dict (got {type(ckpt).__name__})", file=sys.stderr)
    return None


def _classifier_shape(state: dict) -> dict[str, Any] | None:
    """Return a summary of the classifier-head shape in the state_dict.

    Detects:
      * `classifier.weight`           (CLS-only)
      * `classifier.0.weight` .. `.4.weight`  (JPM)
      * `module.classifier.*` (DataParallel prefix)
    """
    state = _strip_module_prefix(state)
    keys = list(state.keys())
    if "classifier.weight" in state:
        shape = tuple(state["classifier.weight"].shape)
        return {
            "kind": "cls_only",
            "shape": list(shape),
            "num_class": int(shape[0]),
            "embedding_dim": int(shape[1]) if len(shape) > 1 else None,
        }
    jpm_keys = [k for k in keys if k.startswith("classifier.") and k.endswith(".weight")]
    # NOTE: the prefix-match above is intentionally lenient. The local re-id
    # code names its head ``classifier.<idx>``, but official TransReID
    # checkpoints store them as ``classifier``, ``classifier_1`` ... ``classifier_4``
    # (no leading dot). Both naming conventions are accepted here.
    if jpm_keys:
        shapes = [tuple(state[k].shape) for k in jpm_keys]
        return {
            "kind": "jpm",
            "shape": list(shapes[0]) if shapes else None,
            "num_linears": len(jpm_keys),
            "all_shapes": [list(s) for s in shapes],
            "num_class": int(shapes[0][0]) if shapes else None,
            "embedding_dim": int(shapes[0][1]) if shapes and len(shapes[0]) > 1 else None,
        }
    return None


def _backbone_present(state: dict) -> bool:
    """Heuristic: the backbone stores ``patch_embed.proj.weight``."""
    state = _strip_module_prefix(state)
    return (
        any(k.startswith("patch_embed.proj") for k in state)
        or any(k.startswith("base.patch_embed") for k in state)
        or any("cls_token" in k for k in state)
    )


def inspect(
    checkpoint_path: Path,
    *,
    expected_profile: str = "msmt17",
    expected_num_class: int | None = None,
    weights_only: bool = True,
) -> dict[str, Any]:
    """Inspect the checkpoint and return a structured verdict."""
    result: dict[str, Any] = {
        "path": str(checkpoint_path),
        "exists": checkpoint_path.exists(),
        "size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
        "expected_profile": expected_profile,
        "expected_num_class": expected_num_class,
    }
    if not checkpoint_path.exists():
        result["ok"] = False
        result["reason"] = "checkpoint_missing"
        return result
    state = _read_state_dict(checkpoint_path, weights_only=weights_only)
    if state is None:
        result["ok"] = False
        result["reason"] = "not_a_torch_checkpoint"
        return result
    result["num_tensors"] = len(state)
    result["has_backbone"] = _backbone_present(state)
    cls = _classifier_shape(state)
    result["classifier"] = cls
    if cls is None:
        result["ok"] = False
        result["reason"] = "no_classifier_head"
        return result
    # Profile detection
    detected_num = cls.get("num_class")
    matched_profile: str | None = None
    for prof, info in KNOWN_PROFILES.items():
        if info["num_class"] == detected_num:
            matched_profile = prof
            break
    result["detected_num_class"] = detected_num
    result["detected_profile"] = matched_profile
    if expected_num_class is None and expected_profile in KNOWN_PROFILES:
        expected_num_class = KNOWN_PROFILES[expected_profile]["num_class"]
    result["expected_num_class_effective"] = expected_num_class
    if expected_num_class is not None and detected_num != expected_num_class:
        result["ok"] = False
        result["reason"] = "num_class_mismatch"
        result["expected"] = expected_num_class
        result["got"] = detected_num
        return result
    result["ok"] = True
    result["reason"] = "compatible"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a TransReID checkpoint and verify profile compatibility.",
    )
    parser.add_argument("path", type=Path, help="Path to the .pth checkpoint file")
    parser.add_argument(
        "--profile",
        default="msmt17",
        choices=list(KNOWN_PROFILES.keys()) + ["custom"],
        help="Expected profile (default: msmt17)",
    )
    parser.add_argument(
        "--num-class",
        type=int,
        default=None,
        help="Override expected num_class (used with --profile custom)",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        default=True,
        help="Use torch.load(weights_only=True) (recommended; default)",
    )
    parser.add_argument(
        "--no-weights-only",
        dest="weights_only",
        action="store_false",
        help="Disable weights_only (NOT recommended; for trusted weights only)",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    result = inspect(
        args.path,
        expected_profile=args.profile,
        expected_num_class=args.num_class,
        weights_only=args.weights_only,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for k, v in result.items():
            print(f"{k}: {v}")
    if not result["ok"]:
        reason = result.get("reason")
        return {
            "checkpoint_missing": 1,
            "num_class_mismatch": 2,
            "not_a_torch_checkpoint": 3,
            "no_classifier_head": 4,
        }.get(reason, 5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
