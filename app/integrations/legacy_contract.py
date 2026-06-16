"""Legacy Service/offline-people-counting external-integration contract.

This module centralises every external-facing value the legacy
``Service/offline-people-counting`` pipeline publishes.  The values
are loaded from ``configs/legacy/offline_people_counting.yaml`` and
exposed through small helper functions:

* :func:`legacy_device_config` — per-camera device config (the
  ``cam_1`` / ``cam_2`` naming, ``site_id``, etc.) that the legacy
  ThingsBoard payload uses.
* :func:`legacy_camera_topic` — the MQTT topic for a given legacy
  channel (``summary``, ``attributes``, ``event``, ``status``).
* :func:`normalize_legacy_camera_id` — ``CAM_01`` → ``cam1``.
* :func:`legacy_client_id` — the
  ``people_counter_<device_name>_<epoch>_<rand>`` client_id.
* :func:`legacy_evidence_key` — the
  ``{prefix}/{cam_id}/{zone_slug}/{date}/{ts}_{pid}.jpg`` MinIO
  object path.
* :func:`legacy_publish_url` / :func:`legacy_hls_url` /
  :func:`legacy_webrtc_url` — the exact stream URLs the legacy
  pipeline builds.
* :func:`legacy_roi_zones` — the original-coordinate ROI polygons
  for CAM_01 / CAM_02, loaded from the legacy YAML.
* :func:`flag_enabled` / :func:`all_flags` — the new feature
  toggles (``ENABLE_SEND_MQTT``, ``SHOW_*``, …).

The module is intentionally pure-Python and free of side effects so
the regression tests can import it without bringing up the full
pipeline.  All public helpers accept an explicit ``cfg`` to make
tests deterministic.
"""

from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "legacy" / "offline_people_counting.yaml"
)
_CONFIG_CACHE: dict[str, Any] | None = None


def _load_config(path: Path | None = None) -> dict[str, Any]:
    global _CONFIG_CACHE
    p = path or _CONFIG_PATH
    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if path is None:
        _CONFIG_CACHE = cfg
    return cfg


def reset_config_cache() -> None:
    """Drop the cached config. Tests that mutate the YAML call this."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


# ---------------------------------------------------------------------------
# Topic + camera-id normalization
# ---------------------------------------------------------------------------


# Same normalisation rules as Service/.../app/io/mqtt_topics.py
def normalize_legacy_camera_id(camera_id: str, device_name: str | None = None) -> str:
    """Return the legacy camera id used in MQTT topics.

    Examples::

        normalize_legacy_camera_id("CAM_01", "cam_1")  -> "cam1"
        normalize_legacy_camera_id("cam1",  "cam1")    -> "cam1"
        normalize_legacy_camera_id("CAM_02", "cam_2")  -> "cam2"
    """
    raw = str(device_name or camera_id or "unknown")
    if raw.startswith("cam_"):
        raw = raw.replace("cam_", "cam", 1)
    return raw.replace("_", "-")


def _strip_topic_slash(s: str) -> str:
    return str(s).strip("/")


def legacy_camera_topic(
    channel: str,
    camera_id: str,
    device_name: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> str:
    """Return the legacy MQTT topic for *channel* on *camera_id*.

    Channels: ``"telemetry" | "attributes" | "events" | "status" | "command"``.
    The returned topic matches
    ``Service/offline-people-counting/app/io/mqtt_topics.py::generate_topics``
    exactly.
    """
    cfg = cfg or _load_config()
    mqtt = cfg.get("mqtt", {})
    base = _strip_topic_slash(mqtt.get("topic_base", "ai/yamaha/people-detection"))
    cid = normalize_legacy_camera_id(camera_id, device_name)
    channel_map = {
        "telemetry": mqtt.get("channels", {}).get("telemetry", "summary"),
        "attributes": mqtt.get("channels", {}).get("attributes", "attributes"),
        "events": mqtt.get("channels", {}).get("events", "event"),
        "status": mqtt.get("channels", {}).get("status", "status"),
        "command": "+",  # wildcard form: {topic_base}/+/command
    }
    suffix = channel_map.get(channel)
    if suffix is None:
        raise ValueError(f"Unknown MQTT channel: {channel!r}")
    if channel == "command":
        return f"{base}/+/command"
    return f"{base}/{cid}/{suffix}"


def legacy_client_id(
    device_name: str,
    *,
    now_epoch: float | None = None,
    rand: int | None = None,
) -> str:
    """Return the legacy MQTT client_id for a given *device_name*.

    Mirrors ``MQTTConnection.__init__`` in the legacy pipeline::

        people_counter_{device_name}_{int(time.time())}_{randint(1000,9999)}
    """
    cfg = _load_config()
    prefix = cfg.get("mqtt", {}).get("client_id_prefix", "people_counter")
    epoch = int(now_epoch if now_epoch is not None else time.time())
    r = rand if rand is not None else random.randint(1000, 9999)
    return f"{prefix}_{device_name}_{epoch}_{r}"


# ---------------------------------------------------------------------------
# Device config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegacyDeviceConfig:
    camera_id: str  # e.g. "CAM_01"
    device_name: str  # e.g. "cam_1"
    device_type: str
    location: str
    category: str
    integration: str
    subsystem: str
    site_id: str

    def topic_camera_id(self) -> str:
        """The camera id used in the MQTT topic (normalized)."""
        return normalize_legacy_camera_id(self.camera_id, self.device_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_name": self.device_name,
            "device_type": self.device_type,
            "location": self.location,
            "category": self.category,
            "integration": self.integration,
            "subsystem": self.subsystem,
            "site_id": self.site_id,
            "camera_id": self.camera_id,
        }


def legacy_device_config(
    camera_id: str,
    cfg: dict[str, Any] | None = None,
) -> LegacyDeviceConfig:
    """Return the legacy device config for *camera_id* (e.g. ``CAM_01``)."""
    cfg = cfg or _load_config()
    dev = cfg.get("devices", {}).get(camera_id)
    if not dev:
        # Fallback: synthesise a default that matches the legacy
        # DEFAULT_DEVICE_CONFIG in mqtt_topics.py. Used by tests when
        # the operator hasn't added a custom entry.
        device_name = "cam_1" if camera_id in {"CAM_01", "cam1", "cam_1"} else "cam_2"
        return LegacyDeviceConfig(
            camera_id=camera_id,
            device_name=device_name,
            device_type="People Counting",
            location=f"Main Entrance - {camera_id}",
            category="ymh",
            integration="yamaha",
            subsystem="demo",
            site_id="site_001",
        )
    return LegacyDeviceConfig(
        camera_id=camera_id,
        device_name=str(dev.get("device_name", "")),
        device_type=str(dev.get("device_type", "People Counting")),
        location=str(dev.get("location", "")),
        category=str(dev.get("category", "ymh")),
        integration=str(dev.get("integration", "yamaha")),
        subsystem=str(dev.get("subsystem", "demo")),
        site_id=str(dev.get("site_id", "site_001")),
    )


def all_legacy_devices(cfg: dict[str, Any] | None = None) -> list[LegacyDeviceConfig]:
    cfg = cfg or _load_config()
    return [legacy_device_config(cid, cfg) for cid in cfg.get("devices", {}).keys()]


# ---------------------------------------------------------------------------
# MinIO object keys
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify_zone(zone: str | None, fallback: str) -> str:
    """Same slug rule as Service/.../minio_uploader.py::slugify_zone."""
    value = str(zone or fallback or "unknown-zone").strip().lower()
    value = value.replace("&", " and ")
    value = _SLUG_RE.sub("-", value).strip("-")
    return value or "unknown-zone"


def legacy_evidence_key(
    *,
    camera_id: str,
    zone: str | None,
    person_id: int,
    timestamp_epoch: float | None = None,
    cfg: dict[str, Any] | None = None,
) -> str:
    """Build the legacy MinIO object key for a person-crop evidence file.

    The returned key matches
    ``Service/offline-people-counting/app/io/minio_uploader.py::build_object_name``
    exactly::

        {object_prefix}/{camera_id}/{zone_slug}/{date}/{epoch_ms}_{person_id}.jpg
    """
    cfg = cfg or _load_config()
    m = cfg.get("minio", {})
    object_prefix = m.get("object_prefix", "people-detection")
    date_format = m.get("object_date_format", "%Y-%m-%d")
    location_title = m.get("location_title", "main_hallway")
    from datetime import datetime, timezone as _tz

    ts = (
        datetime.fromtimestamp(timestamp_epoch, tz=_tz.utc)
        if timestamp_epoch is not None
        else datetime.now(tz=_tz.utc)
    )
    epoch_ms = int(ts.timestamp() * 1000)
    date_part = ts.strftime(date_format)
    zone_folder = _slugify_zone(zone, location_title)
    parts = [
        object_prefix,
        camera_id,
        zone_folder,
        date_part,
        f"{epoch_ms}_{person_id}.jpg",
    ]
    return "/".join(p.strip("/") for p in parts if p)


# ---------------------------------------------------------------------------
# Streaming URLs
# ---------------------------------------------------------------------------


def legacy_publish_url(
    host: str,
    camera_id: str,
    cfg: dict[str, Any] | None = None,
) -> str:
    """RTSP publish URL — matches ``build_publish_url`` in legacy.

    The legacy pipeline derives the camera id segment from
    ``device_name`` (e.g. ``cam_1`` → ``cam1``).  We honour the same
    mapping by looking up the new id (CAM_01) in the legacy device
    config; if the operator has supplied a custom device config we
    use that, otherwise the canonical ``cam1`` / ``cam2`` is used.
    """
    cfg = cfg or _load_config()
    s = cfg.get("streaming", {})
    tpl = s.get("publish_url", "rtsp://{host}:{rtsp_port}/{camera_id}/live")
    return tpl.format(
        host=host,
        rtsp_port=int(s.get("rtsp_port", 8554)),
        camera_id=_resolved_legacy_topic_id(camera_id, cfg),
    )


def legacy_hls_url(
    host: str,
    camera_id: str,
    cfg: dict[str, Any] | None = None,
) -> str:
    cfg = cfg or _load_config()
    s = cfg.get("streaming", {})
    tpl = s.get("hls_url", "http://{host}:{hls_port}/{camera_id}/live/index.m3u8")
    return tpl.format(
        host=host,
        hls_port=int(s.get("hls_port", 8889)),
        camera_id=_resolved_legacy_topic_id(camera_id, cfg),
    )


def legacy_webrtc_url(
    host: str,
    camera_id: str,
    cfg: dict[str, Any] | None = None,
) -> str:
    cfg = cfg or _load_config()
    s = cfg.get("streaming", {})
    tpl = s.get("webrtc_url", "http://{host}:{webrtc_port}/{camera_id}/live")
    return tpl.format(
        host=host,
        webrtc_port=int(s.get("webrtc_port", 8890)),
        camera_id=_resolved_legacy_topic_id(camera_id, cfg),
    )


def _resolved_legacy_topic_id(camera_id: str, cfg: dict[str, Any]) -> str:
    """Map *camera_id* to the legacy normalized id used in URLs.

    Looks up the new id (CAM_01 / CAM_02) in the legacy device
    config; if found, uses that device_name for normalization.
    Otherwise falls back to the standard normalization
    (``cam_1`` → ``cam1``).
    """
    dev = cfg.get("devices", {}).get(camera_id)
    if dev:
        device_name = str(dev.get("device_name", camera_id))
        return normalize_legacy_camera_id(camera_id, device_name)
    return normalize_legacy_camera_id(camera_id)


# ---------------------------------------------------------------------------
# ROI / zone config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegacyRoi:
    name: str
    # Original (source) video pixel coordinates, exactly as in the
    # legacy config.yaml.
    points: list[tuple[int, int]]


@dataclass(frozen=True)
class LegacyCameraRois:
    camera_id: str
    video_path: str
    original_size: tuple[int, int]
    display_size: tuple[int, int]
    fps: int
    rois: list[LegacyRoi]
    zone_colors: dict[str, tuple[int, int, int]]

    def to_processing_polygons(
        self, processing_size: tuple[int, int]
    ) -> list[tuple[str, list[tuple[int, int]]]]:
        """Scale the source-coordinate ROIs to *processing_size*.

        Mirrors ``Service/.../app/engine/roi.py::scale_roi_config``.
        """
        out: list[tuple[str, list[tuple[int, int]]]] = []
        orig_w, orig_h = self.original_size
        tgt_w, tgt_h = processing_size
        sx = tgt_w / orig_w
        sy = tgt_h / orig_h
        for r in self.rois:
            scaled = [(int(round(x * sx)), int(round(y * sy))) for (x, y) in r.points]
            out.append((r.name, scaled))
        return out


def legacy_roi_zones(
    camera_id: str,
    cfg: dict[str, Any] | None = None,
) -> LegacyCameraRois:
    """Return the legacy ROI / video config for *camera_id*.

    ``camera_id`` is the new-pipeline id (``CAM_01`` / ``CAM_02``) OR
    the legacy id (``cam1`` / ``cam2``).
    """
    cfg = cfg or _load_config()
    cams = cfg.get("cameras", {})
    # Try both the new id (CAM_01) and the legacy id (cam1).
    candidates = [camera_id]
    if camera_id.startswith("CAM_"):
        candidates.append("cam" + camera_id.removeprefix("CAM_").lstrip("0").zfill(0))
        # Use the canonical mapping: CAM_01 -> cam1
        digits = camera_id.removeprefix("CAM_").lstrip("0") or "0"
        candidates.append(f"cam{digits}")
    for key in candidates:
        if key in cams:
            entry = cams[key]
            video = entry.get("video", {})
            original = tuple(int(v) for v in video.get("original_size", [0, 0]))
            display = tuple(int(v) for v in video.get("display_size", [0, 0]))
            rois: list[LegacyRoi] = []
            for r in entry.get("rois", []):
                pts = [(int(p[0]), int(p[1])) for p in r.get("points", [])]
                rois.append(LegacyRoi(name=str(r.get("name", "")), points=pts))
            colors = {
                str(name): tuple(int(v) for v in rgb)
                for name, rgb in entry.get("visual", {}).get("zone_colors", {}).items()
            }
            return LegacyCameraRois(
                camera_id=key,
                video_path=str(video.get("path", "")),
                original_size=original,
                display_size=display,
                fps=int(video.get("fps", 15)),
                rois=rois,
                zone_colors=colors,
            )
    raise KeyError(f"No legacy ROI config for camera_id={camera_id!r}")


def all_legacy_cameras(cfg: dict[str, Any] | None = None) -> list[LegacyCameraRois]:
    cfg = cfg or _load_config()
    return [legacy_roi_zones(c, cfg) for c in cfg.get("cameras", {}).keys()]


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


def legacy_thresholds(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or _load_config()
    return dict(cfg.get("thresholds", {}))


# ---------------------------------------------------------------------------
# Feature toggles
# ---------------------------------------------------------------------------

_TOGGLE_NAMES: tuple[str, ...] = (
    "ENABLE_SEND_MQTT",
    "ENABLE_MINIO_UPLOAD",
    "ENABLE_TRACK_ID",
    "SHOW_ROI_ZONES",
    "SHOW_CONFIDENCE_SCORE",
    "SHOW_TRACK_ID",
    "SHOW_DETECTION_BOX",
    "SHOW_CAMERA_LABEL",
    "SHOW_COUNTING_OVERLAY",
)


def _flag_source(cfg: dict[str, Any] | None = None) -> dict[str, bool]:
    cfg = cfg or _load_config()
    toggles = dict(cfg.get("toggles", {}) or {})
    # Env vars override YAML when set.
    for name in _TOGGLE_NAMES:
        env_val = os.environ.get(name)
        if env_val is not None and env_val != "":
            toggles[name] = _to_bool(env_val, toggles.get(name, True))
    return toggles


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def flag_enabled(name: str, cfg: dict[str, Any] | None = None) -> bool:
    """Return True if the named feature toggle is on.

    Resolution order: env var > YAML > default-true. Unknown toggle
    names return True (we treat "unknown == on" to be lenient — the
    new pipeline only adds toggles, never removes a default-true
    behaviour).
    """
    if name not in _TOGGLE_NAMES:
        return True
    src = _flag_source(cfg)
    return _to_bool(src.get(name, True), True)


def all_flags(cfg: dict[str, Any] | None = None) -> dict[str, bool]:
    """Return a dict of all toggle states."""
    src = _flag_source(cfg)
    return {name: _to_bool(src.get(name, True), True) for name in _TOGGLE_NAMES}


def toggle_names() -> tuple[str, ...]:
    return _TOGGLE_NAMES
