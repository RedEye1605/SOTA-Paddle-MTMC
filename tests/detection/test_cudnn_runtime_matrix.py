"""Pinned: the api container's runtime cuDNN is in the 8.x family.

Why this matters
----------------
PaddlePaddle 2.6.2 is compiled against cuDNN 8.6 + CUDA 11.8
(``paddle.version.cudnn() == 8.6.0``). The paddlepaddle-gpu wheel
transitively pins ``nvidia-cudnn-cu12==9.19.0.56`` which is loaded
at runtime via the dynamic loader, but PaddleDetection's
``ppyoloe_head.forward_eval`` calls a C++ kernel
(``fused_conv2d_add_act_kernel.cu:610``) that uses the legacy
``cudnnConvolutionBiasActivationForward`` path — which cuDNN 9.x
removed from the legacy API. The result is::

    ExternalError: CUDNN error(3000), CUDNN_STATUS_NOT_SUPPORTED.
      [operator < fused_conv2d_add_act > error]

and MOT init aborts before any frame is emitted. This bug does not
appear in ``F.conv2d(x, w, bias=b)`` (the basic conv-bias path uses
the cudnn 9 frontend correctly), only in the legacy fused conv-bias-
activation call that PaddleDetection's PPYOLOE head uses.

The fix is to pin the runtime cuDNN to the 8.x family
(``nvidia-cudnn-cu12==8.9.7.29`` is the last 8.x release, Dec 2023),
matching the cuDNN ABI that paddle 2.6.2's
``fused_conv2d_add_act_kernel`` was compiled against.

This test asserts:

  1. The api container loads cuDNN in the 8.x family at runtime
     (reads the wheel dist-info).
  2. The base image is replaced with the runtime cuDNN at the
     loader level (the libcudnn.so symlink points at a libcudnn.so.8).

If either fails, MOT init crashes with CUDNN error(3000) and
PP-Human emits no frames.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

DIST_INFO_GLOB = "/opt/venv/lib/python3.12/site-packages/nvidia_cudnn_cu12-*.dist-info"


def _read_runtime_cudnn_version() -> str | None:
    """Read the paddle wheel's transitive nvidia-cudnn-cu12 dist-info
    to find which cuDNN version is actually on the loader path."""
    import glob
    matches = sorted(glob.glob(DIST_INFO_GLOB))
    if not matches:
        return None
    # nvidia_cudnn_cu12-9.19.0.56.dist-info  -> "9.19.0.56"
    m = re.match(r"nvidia_cudnn_cu12-([0-9.]+)\.dist-info", os.path.basename(matches[0]))
    return m.group(1) if m else None


def test_runtime_cudnn_is_8x_family() -> None:
    """RED: the container currently ships nvidia-cudnn-cu12==9.19.0.56
    (the paddle 2.6.2 wheel's transitive dep). Paddle 2.6.2's
    fused_conv2d_add_act_kernel was compiled against cuDNN 8.6 and
    rejects the cuDNN 9.x ABI. The fix is to force cuDNN 8.x at
    runtime via the GPU dependency pin.
    """
    if not Path("/opt/venv").exists():
        pytest.skip("not running inside the api container")
    version = _read_runtime_cudnn_version()
    if version is None:
        pytest.skip(
            "nvidia-cudnn-cu12 dist-info not found; "
            "this image may not have a GPU paddle wheel installed"
        )
    major = int(version.split(".")[0])
    assert major == 8, (
        f"runtime cuDNN is {version!r} (major={major}); "
        f"PaddlePaddle 2.6.2's fused_conv2d_add_act_kernel was "
        f"compiled against cuDNN 8.6 and rejects cuDNN 9.x with "
        f"CUDNN error(3000), CUDNN_STATUS_NOT_SUPPORTED at "
        f"fused_conv2d_add_act_kernel.cu:610. The runtime cuDNN "
        f"must be in the 8.x family. Pin "
        f"nvidia-cudnn-cu12==8.9.7.29 in the GPU dependency set and rebuild."
    )


def test_libcudnn_so_symlink_points_to_libcudnn_so_8() -> None:
    """RED: the Dockerfile PATCH-051 creates
    ``/usr/lib/x86_64-linux-gnu/libcudnn.so`` as a symlink to
    ``libcudnn.so.9``. With cuDNN 8.9 installed, the symlink must
    point at ``libcudnn.so.8`` so the dynamic loader resolves to
    the 8.x ABI that paddle 2.6.2 expects."""
    if not Path("/opt/venv").exists():
        pytest.skip("not running inside the api container")
    symlink = Path("/usr/lib/x86_64-linux-gnu/libcudnn.so")
    if not symlink.is_symlink():
        pytest.skip(
            "libcudnn.so is not a symlink in this image; "
            "the symlink contract only applies to images that "
            "follow the PATCH-051 pattern"
        )
    target = os.readlink(symlink)
    # Resolve any chained symlinks relative to the symlink's dir.
    if not os.path.isabs(target):
        target = os.path.normpath(os.path.join(symlink.parent, target))
    if os.path.islink(target):
        target = os.readlink(target)
        if not os.path.isabs(target):
            target = os.path.normpath(os.path.join(symlink.parent, target))
    assert target.endswith("libcudnn.so.8") or "libcudnn.so.8." in os.path.basename(target), (
        f"libcudnn.so -> {target}; expected a libcudnn.so.8 target. "
        f"With cuDNN 9.x at the loader path, Paddle 2.6.2's "
        f"fused_conv2d_add_act_kernel fails with CUDNN error(3000)."
    )
