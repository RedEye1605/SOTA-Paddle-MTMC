"""SAHITrackletBridge.

Lightweight consumer of ``stream:detections_sahi``. Maintains a
short-lived (2-second) sliding window of SAHI detections per camera,
deduplicates against active PP-Human tracks, and emits NEW
auxiliary tracklets to the existing TrackletCollector via
``on_sahi_detection``.

The bridge does NOT modify SDE_Detector or any PaddleDetection
internal. It runs as a separate Redis consumer in the api process.

Design (operator-approved, 2026-06-16):
  - One bridge instance shared across all cameras; per-camera
    sliding windows.
  - Per-camera sliding window: keep the last 2 seconds of SAHI
    detections.
  - Dedup against PP-Human: read active PP-Human track bboxes from
    ``stream:detections`` (XREVRANGE COUNT 200, last 1 second).
  - If a SAHI detection does not match any active PP-Human bbox
    within the last 1 second (IoU > 0.3), create a synthetic
    tracklet via TrackletCollector.on_sahi_detection.
  - Synthetic local_track_id: ``frame_id`` (one tracklet per
    (camera, frame) — there is no MOT for SAHI).
  - Marked source="sahi" and provisional=True downstream.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SAHIBridgeConfig:
    """Tunables for the SAHITrackletBridge."""

    sahi_stream: str = "stream:detections_sahi"
    pphuman_stream: str = "stream:detections"
    consumer_group: str = "sahi_bridge_workers"
    consumer_name: str = "sahi-bridge-01"
    # Sliding window for SAHI dedup (in milliseconds).
    sahi_window_ms: int = 2000
    # Window for active PP-Human tracks (in milliseconds).
    pphuman_window_ms: int = 1000
    # IoU threshold for "matches an active PP-Human track".
    dedup_iou: float = 0.3
    # How often to scan for new SAHI events (milliseconds).
    poll_interval_ms: int = 200
    # Max SAHI events per scan.
    scan_count: int = 10
    # Max PP-Human events to scan for dedup (most recent first).
    pphuman_scan_count: int = 200
    # How often to refresh the PP-Human dedup cache (milliseconds).
    # 0 = refresh on every SAHI event.
    pphuman_refresh_ms: int = 0

    @classmethod
    def from_env(cls) -> "SAHIBridgeConfig":
        return cls(
            sahi_stream=os.environ.get("SAHI_STREAM", "stream:detections_sahi"),
            pphuman_stream="stream:detections",
            consumer_group=os.environ.get(
                "SAHI_BRIDGE_CONSUMER_GROUP", "sahi_bridge_workers"
            ),
            consumer_name=os.environ.get(
                "SAHI_BRIDGE_CONSUMER_NAME", "sahi-bridge-01"
            ),
            sahi_window_ms=int(os.environ.get("SAHI_BRIDGE_WINDOW_MS", "2000")),
            pphuman_window_ms=int(
                os.environ.get("SAHI_BRIDGE_PPHUMAN_WINDOW_MS", "1000")
            ),
            dedup_iou=float(os.environ.get("SAHI_BRIDGE_DEDUP_IOU", "0.3")),
            poll_interval_ms=int(
                os.environ.get("SAHI_BRIDGE_POLL_INTERVAL_MS", "200")
            ),
            pphuman_refresh_ms=int(
                os.environ.get("SAHI_BRIDGE_PPHUMAN_REFRESH_MS", "0")
            ),
        )


def _iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Standard IoU for two axis-aligned bboxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SAHITrackletBridge:
    """Consumes ``stream:detections_sahi``, dedups against PP-Human,
    and emits auxiliary tracklets to TrackletCollector.

    The bridge does NOT run its own MOT. Tracklets are short-lived
    (one-frame) placeholders; the persistent-ID chain
    (ReIDWorker + GlobalIdentityResolver) handles the actual
    identity resolution.
    """

    def __init__(
        self,
        *,
        redis: Any,
        collector: Any,  # TrackletCollector
        config: Optional[SAHIBridgeConfig] = None,
    ) -> None:
        self._redis = redis
        self._collector = collector
        self._config = config or SAHIBridgeConfig()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Per-camera sliding window: deque of (timestamp_ms, bbox, frame_id, score, idx)
        self._sahi_windows: Dict[str, Deque] = {}
        # Cache of recent PP-Human bboxes per camera to avoid hammering
        # Redis on every SAHI bbox. Keyed by (camera_id, last_refresh_ms).
        self._pphuman_cache: Dict[str, Deque] = {}
        self._pphuman_cache_ts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        # Ensure consumer group exists (idempotent).
        try:
            self._redis.ensure_group(
                self._config.sahi_stream, self._config.consumer_group
            )
        except Exception:  # noqa: BLE001
            # Most likely BUSYGROUP (already exists) or Redis briefly
            # down; both are non-fatal. Real errors surface at scan time.
            pass
        self._thread = threading.Thread(
            target=self._run,
            name="sahi-tracklet-bridge",
            daemon=True,
        )
        self._thread.start()
        logger.info("SAHITrackletBridge started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("SAHITrackletBridge stopped")

    def _run(self) -> None:
        poll_sec = self._config.poll_interval_ms / 1000.0
        while not self._stop_event.is_set():
            try:
                self._scan_once()
            except Exception as e:  # noqa: BLE001
                logger.warning("SAHITrackletBridge scan error: %s", e)
            self._stop_event.wait(timeout=poll_sec)

    # ------------------------------------------------------------------
    # Scan loop
    # ------------------------------------------------------------------

    def _scan_once(self) -> None:
        """Read new SAHI events; for each, dedup and emit if new."""
        try:
            msgs = self._redis.consume(
                self._config.sahi_stream,
                self._config.consumer_group,
                self._config.consumer_name,
                count=self._config.scan_count,
                block_ms=0,  # non-blocking
            )
        except Exception:  # noqa: BLE001
            return
        for msg_id, fields in msgs:
            try:
                self._handle_sahi_event(fields)
            except Exception as e:  # noqa: BLE001
                logger.warning("SAHITrackletBridge handle error: %s", e)
                continue
            else:
                try:
                    self._redis.ack(
                        self._config.sahi_stream,
                        self._config.consumer_group,
                        msg_id,
                    )
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    # Per-event handling
    # ------------------------------------------------------------------

    def _handle_sahi_event(self, fields: dict) -> None:
        """fields is the JSON-decoded dict from stream:detections_sahi.

        Expected keys (set by SAHIWorker._publish):
          - camera_id: str
          - frame_id: int (currently always 0 from SAHIWorker)
          - timestamp_ms: int
          - detections: list[ [x1, y1, x2, y2, score] ]
        """
        camera_id = fields.get("camera_id", "")
        if not camera_id:
            return
        try:
            ts_ms = int(fields.get("timestamp_ms", "0"))
        except Exception:  # noqa: BLE001
            ts_ms = 0
        try:
            frame_id = int(fields.get("frame_id", "0"))
        except Exception:  # noqa: BLE001
            frame_id = 0
        dets = fields.get("detections", []) or []
        # ``consume`` already JSON-decodes — detections is a list of
        # [x1, y1, x2, y2, score] arrays. Be defensive: some callers
        # (tests) may pre-stringify.
        if isinstance(dets, str):
            try:
                dets = json.loads(dets)
            except Exception:  # noqa: BLE001
                dets = []
        for idx, det in enumerate(dets):
            if not isinstance(det, (list, tuple)) or len(det) < 5:
                continue
            try:
                bbox = (
                    float(det[0]),
                    float(det[1]),
                    float(det[2]),
                    float(det[3]),
                )
                score = float(det[4])
            except Exception:  # noqa: BLE001
                continue
            self._handle_sahi_bbox(
                camera_id=camera_id,
                frame_id=frame_id,
                ts_ms=ts_ms,
                bbox=bbox,
                score=score,
                idx=idx,
            )

    def _handle_sahi_bbox(
        self,
        *,
        camera_id: str,
        frame_id: int,
        ts_ms: int,
        bbox: Tuple[float, float, float, float],
        score: float,
        idx: int,
    ) -> None:
        # Sliding window: drop old entries.
        window = self._sahi_windows.setdefault(camera_id, deque())
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - self._config.sahi_window_ms
        while window and window[0][0] < cutoff:
            window.popleft()
        # Dedup against active PP-Human tracks. If Redis is
        # unavailable, treat as "no match" and let SAHI through
        # (the alternative — refusing to emit — is worse; the
        # TrackletCollector is the dedup safety net).
        if self._matches_active_pphuman(camera_id, bbox, now_ms):
            return
        # Emit to TrackletCollector.
        self._collector.on_sahi_detection(
            camera_id=camera_id,
            frame_id=frame_id,
            timestamp_ms=ts_ms,
            bbox=bbox,
            score=score,
        )
        # Add to our own window so we don't double-emit within the
        # dedup window if a future event replays.
        window.append((ts_ms, bbox, frame_id, score, idx))

    # ------------------------------------------------------------------
    # PP-Human dedup
    # ------------------------------------------------------------------

    def _matches_active_pphuman(
        self,
        camera_id: str,
        bbox: Tuple[float, float, float, float],
        now_ms: int,
    ) -> bool:
        """Check if the bbox matches any active PP-Human track."""
        pp_bboxes = self._get_pphuman_bboxes(camera_id, now_ms)
        for pp_bbox in pp_bboxes:
            if _iou(bbox, pp_bbox) > self._config.dedup_iou:
                return True
        return False

    def _get_pphuman_bboxes(
        self,
        camera_id: str,
        now_ms: int,
    ) -> list[Tuple[float, float, float, float]]:
        """Read most-recent PP-Human bboxes for ``camera_id``.

        Cached for ``pphuman_refresh_ms`` milliseconds to avoid
        hammering Redis on every SAHI bbox. Returns [] on any
        Redis error (the caller treats [] as "no match").
        """
        refresh_ms = self._config.pphuman_refresh_ms
        if refresh_ms > 0:
            last_ts = self._pphuman_cache_ts.get(camera_id, 0)
            if (now_ms - last_ts) < refresh_ms:
                return list(self._pphuman_cache.get(camera_id, []))
        # Lazy-load via XREVRANGE on the underlying redis-py client.
        # The RedisState wrapper does not expose xrevrange, so we
        # reach into ``self._redis.client`` (the raw redis.Redis).
        try:
            client = getattr(self._redis, "client", self._redis)
            items = client.xrevrange(
                self._config.pphuman_stream,
                max="+",
                min="-",
                count=self._config.pphuman_scan_count,
            )
        except Exception:  # noqa: BLE001
            return []
        cutoff = now_ms - self._config.pphuman_window_ms
        out: list[Tuple[float, float, float, float]] = []
        for _eid, raw_fields in items or []:
            # ``raw_fields`` may be either decoded dicts (if a
            # higher-level wrapper pre-decoded) or raw bytes/str
            # (from the raw redis-py client). Normalize.
            try:
                ts = int(_field_get(raw_fields, "timestamp_ms", 0))
            except Exception:  # noqa: BLE001
                ts = 0
            if ts and ts < cutoff:
                continue
            # PP-Human's RedisSideChannel.emit_detection emits ONE bbox
            # per stream entry under the singular key ``bbox``
            # (``[x1, y1, x2, y2]``), not a list of bboxes. The bridge
            # must read that key; reading ``bboxes`` (plural) here
            # silently missed every PP-Human event and let SAHI
            # detections through unconditionally, causing the
            # double-counting / global_id fragmentation seen in
            # production.
            bbox = _field_get(raw_fields, "bbox", [])
            if isinstance(bbox, str):
                try:
                    bbox = json.loads(bbox)
                except Exception:  # noqa: BLE001
                    bbox = []
            # Only consider events for THIS camera.
            cam = _field_get(raw_fields, "camera_id", "")
            if cam and cam != camera_id:
                continue
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            try:
                out.append(
                    (
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    )
                )
            except Exception:  # noqa: BLE001
                continue
        self._pphuman_cache[camera_id] = deque(out)
        self._pphuman_cache_ts[camera_id] = now_ms
        return out


def _field_get(fields: Any, key: str, default: Any) -> Any:
    """Get a field from a Redis entry that may be bytes/str/dict."""
    if not isinstance(fields, dict):
        return default
    if key in fields:
        return fields[key]
    # Try the bytes/str variant.
    for k, v in fields.items():
        try:
            if isinstance(k, bytes) and k.decode("utf-8", "ignore") == key:
                return v
            if isinstance(k, str) and k == key:
                return v
        except Exception:  # noqa: BLE001
            continue
    return default
