"""Argparse — small, surface-stable.

`main.py` calls :func:`parse_args` once at startup. Anything that can be
configured in YAML is preferred over a flag; flags are for operational
overrides only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sota-paddle-mtmct",
        description=(
            "Multi-camera MTMCT system. PP-Human baseline + TransReID, "
            "Qdrant, Redis, PostgreSQL, MinIO, ThingsBoard MQTT."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/app.yaml"),
        help="Path to the runtime YAML config (default: configs/app.yaml).",
    )
    parser.add_argument(
        "--cameras-config",
        type=Path,
        default=Path("configs/cameras.yaml"),
        help="Path to the cameras YAML config.",
    )
    parser.add_argument(
        "--zones-config",
        type=Path,
        default=Path("configs/zones.yaml"),
        help="Path to the zones YAML config.",
    )
    parser.add_argument(
        "--links-config",
        type=Path,
        default=Path("configs/camera_links.yaml"),
        help="Path to the camera_links YAML config.",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "production",
            "smoke_test",
            "benchmark",
            # Legacy aliases kept for backward compat.
            "multi_rtsp",
            "single_cam_smoke",
        ],
        default=None,
        help=(
            "Runtime mode. If unset, uses SOTA_RUNTIME_MODE env or "
            "defaults to `production`. `smoke_test` (or the legacy "
            "`single_cam_smoke`) is for dev/CI only; production mode "
            "refuses to start without real Paddle + TransReID models."
        ),
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default=None,
        help="For single_cam_smoke mode, which camera to use (e.g. CAM_01).",
    )
    parser.add_argument(
        "--video-file",
        type=str,
        default=None,
        help=(
            "For single_cam_smoke mode, a local video file path. Overrides the camera's RTSP URL."
        ),
    )
    parser.add_argument(
        "--smoke-max-frames",
        type=int,
        default=None,
        help="Override SMOKE_MAX_FRAMES env var.",
    )
    parser.add_argument(
        "--smoke-max-seconds",
        type=int,
        default=None,
        help="Override SMOKE_MAX_SECONDS env var.",
    )
    return parser.parse_args(argv)
