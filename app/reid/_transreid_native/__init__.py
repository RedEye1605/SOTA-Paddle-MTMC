"""Marker module for the vendored TransReID backbone.

The public API is re-exported from ``.model`` at import time, so
importing this module requires ``torch`` to be installed. Operators
on a CPU-only dev box get a clear error message; the Dockerfile
``sidecar`` target brings torch in for production ReID.
"""

from __future__ import annotations

try:
    from .model import (
        BNNeck,
        JPM,
        build_transreid_model,
        extract_inference_feature,
        load_transreid_checkpoint,
        vit_base_patch16_224_TransReID,
    )
except ImportError as e:
    raise ImportError(
        "TransReID vendor requires torch; use the Dockerfile `sidecar` "
        "target or install the matching torch wheel in your dev venv "
        "(or set PATCH_TORCH_REQUIRED=0 to skip). "
        f"Underlying error: {e}",
    ) from e

__all__ = [
    "BNNeck",
    "JPM",
    "build_transreid_model",
    "extract_inference_feature",
    "load_transreid_checkpoint",
    "vit_base_patch16_224_TransReID",
]
