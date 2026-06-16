"""Session helpers for 24 h rolling windows."""

from __future__ import annotations

import uuid

from ..utils.time import now_ts, session_id_for_window


def mint_global_id(camera_id: str, ts: float | None = None) -> str:
    """Mint a fresh global_id like 'GID-<8hex>-<camera_short>'."""
    if ts is None:
        ts = now_ts()
    short_cam = camera_id.split("_")[-1] if "_" in camera_id else camera_id
    return f"GID-{uuid.uuid4().hex[:8].upper()}-{short_cam}"


def current_session_id(ts: float | None = None) -> str:
    if ts is None:
        ts = now_ts()
    return session_id_for_window(ts, window_seconds=86_400)
