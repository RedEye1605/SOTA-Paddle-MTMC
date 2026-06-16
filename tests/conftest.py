"""Pytest configuration — add project root to sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Register custom pytest marks used in test_persistent_id_architecture.py
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "hls: live HLS probes (skipped unless --run-hls is passed)",
    )
    config.addinivalue_line(
        "markers",
        "gpu_required: tests that need a CUDA/GPU host; skipped in CPU-only runs",
    )
