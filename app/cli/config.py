"""Config loading and validation.

The loader is intentionally strict: missing required keys fail loud at startup.
We never silently fall back to JSON or a default value if a critical config is
missing — see ``app/storage/postgres.py`` and ``app/storage/qdrant_store.py``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ----- Typed config dataclasses (subset; most code uses dict access) -----
@dataclass
class RuntimeConfig:
    device: str
    gpu_id: int
    run_mode: str
    cpu_threads: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RuntimeConfig":
        return cls(
            device=d.get("device", "gpu"),
            gpu_id=int(d.get("gpu_id", 0)),
            run_mode=d.get("run_mode", "trt_fp16"),
            cpu_threads=int(d.get("cpu_threads", 8)),
        )


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file; raise loud if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_all_configs(
    config_path: Path,
    cameras_path: Path,
    zones_path: Path,
    links_path: Path,
) -> dict[str, dict[str, Any]]:
    """Load the four top-level YAML files and return a dict of dicts.

    Returns:
        {
            "app": {...},            # configs/app.yaml
            "cameras": {...},        # configs/cameras.yaml
            "zones":  {...},         # configs/zones.yaml
            "links":  {...},         # configs/camera_links.yaml
        }
    """
    configs = {
        "app": load_yaml(config_path),
        "cameras": load_yaml(cameras_path),
        "zones": load_yaml(zones_path),
        "links": load_yaml(links_path),
    }
    logger.info(
        "Loaded configs: app=%s cameras=%d zones=%d links=%d",
        configs["app"].get("app", {}).get("name", "?"),
        len(configs["cameras"].get("cameras", [])),
        len(configs["zones"].get("zones", [])),
        len(configs["links"].get("camera_links", [])),
    )
    return configs


def env_bool(key: str, default: bool = False) -> bool:
    v = os.environ.get(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        logger.warning("env %s=%r is not an int, falling back to %d", key, v, default)
        return default


def env_str(key: str, default: str = "") -> str:
    v = os.environ.get(key)
    return v if v is not None else default


def get_active_cameras(cameras_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only cameras marked is_active=true.

    RTSP URL is resolved lazily from the env var named in
    `rtsp_url_env_key` — the URL itself is never in the YAML.
    """
    out: list[dict[str, Any]] = []
    for cam in cameras_cfg.get("cameras", []):
        if not cam.get("is_active", True):
            continue
        out.append(cam)
    return out


def resolve_rtsp_url(camera_cfg: dict[str, Any]) -> str:
    """Read the env var named in `rtsp_url_env_key`.

    Accepts any of:

    * ``rtsp://`` / ``rtsps://`` / ``rtmp://`` / ``http(s)://`` / ``tcp://`` /
      ``udp://`` — live network streams.
    * ``/abs/path/to/file`` / ``./relative/path`` / ``~/path`` — local video
      file. The file is validated to exist (only for the path form, not
      for ``file://`` URIs which OpenCV handles internally).
    * ``file:///abs/path`` — local file URI form.

    The returned string is the value as-is; the runner's
    ``_is_live_stream`` decides whether to use the resilient reader
    (live) or the simple reader (file).
    """
    env_key = camera_cfg["rtsp_url_env_key"]
    url = os.environ.get(env_key, "")
    if not url:
        raise RuntimeError(
            f"Camera {camera_cfg['camera_id']} requires env var "
            f"{env_key!r} to be set (do not put RTSP URLs in YAML)."
        )
    # If the value looks like a local path (not a network scheme),
    # expand ~ and ensure the file exists. This catches typos in
    # the operator's .env early instead of letting the runner
    # crash with "Cannot open video source: …" 30 seconds in.
    if not _looks_like_network_url(url):
        from pathlib import Path

        expanded = Path(url).expanduser()
        if not expanded.exists():
            raise RuntimeError(
                f"Camera {camera_cfg['camera_id']} video source "
                f"{url!r} (expanded: {expanded}) does not exist. "
                f"Either fix {env_key} in .env or supply an "
                f"rtsp://... URL."
            )
    return url


def _looks_like_network_url(value: str) -> bool:
    """True if *value* is a network stream URL (rtsp/http/...)."""
    v = value.strip().lower()
    return (
        v.startswith("rtsp://")
        or v.startswith("rtsps://")
        or v.startswith("rtmp://")
        or v.startswith("http://")
        or v.startswith("https://")
        or v.startswith("tcp://")
        or v.startswith("udp://")
        or v.startswith("file://")
    )
