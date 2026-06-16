"""GPU telemetry helpers.

A lightweight shim so the rest of the codebase can call `gpu_memory_used_mb()`
without importing torch/nvidia-smi at import time.
"""

from __future__ import annotations

import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)


def gpu_memory_used_mb() -> Optional[float]:
    """Returns the used VRAM in MB for GPU 0, or None if nvidia-smi is missing."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        text = out.decode("utf-8", errors="ignore").strip().splitlines()
        if not text:
            return None
        return float(text[0])
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def gpu_memory_total_mb() -> Optional[float]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        text = out.decode("utf-8", errors="ignore").strip().splitlines()
        if not text:
            return None
        return float(text[0])
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
