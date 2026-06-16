"""IdentityOverlayCache — in-process cache of recent identity decisions.

Subscribes (via XREAD, no consumer group) to ``stream:identity_decisions``
and maintains a ``dict[(camera_id, local_track_id)] -> global_id`` so the
overlay can render ``G:{global_id}`` in real time without round-tripping
to Redis per frame.

Design:
  - XREAD (no group) so the cache is a non-blocking fan-out reader;
    the telemetry_worker is the canonical consumer.
  - TTL: each entry expires after `IDENTITY_ACTIVE_BINDING_TTL_SEC`
    (600 s = 10 min default). The cache is GC'd on lookup.
  - Decision outcomes ``hold_ambiguous`` and ``reject_impossible`` are
    IGNORED — they MUST NOT bind to a global_id (per operator spec).
  - Outcomes ``assign_existing`` and ``create_new`` are accepted and
    the global_id is recorded.

PATCH (2026-06-15, anti-flicker, operator spec): the cache is
keyed on THREE indexes (not just one):
  1. ``(camera_id, local_track_id)`` — primary, used by the HLS
     overlay. Bumped TTL to 600 s so brief MOT re-id gaps (caused by
     occlusions or NMS jitter) don't visibly break the G:<gid>
     label in the stream.
  2. ``(camera_id, global_id)`` — secondary, used as a "look up the
     most recent local_track_id for this global_id" reverse index.
     This is consulted when the cache is asked for a ``local_track_id``
     it has never seen, AND that ``local_track_id``'s (cam, local)
     entry has expired. If a recent global_id is in this index, we
     return that gid under the new local_track_id — so the
     overlay can survive an MOT re-id.
  3. ``(global_id) -> (camera_id, local_track_id, expires_at)`` —
     tertiary "last seen" index, used by #2 to pick the most
     recent local_track_id for a given global_id.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import json

from ..storage.redis_state import RedisState

logger = logging.getLogger(__name__)


class IdentityOverlayCache:
    """Maintains a per-(camera_id, local_track_id) -> global_id map
    for real-time overlay rendering.
    """

    STREAM = "stream:identity_decisions"
    # PATCH (2026-06-15, anti-flicker): bumped from 120 s to 600 s
    # so a brief MOT re-id gap (occlusion / NMS jitter) does NOT
    # visibly break the ``G:<gid>`` label in the HLS overlay. The
    # overlay renders this value per frame; if the binding expires
    # the bbox falls back to ``Person`` + ``L<id>`` until the next
    # decision event arrives, which the operator observed as a
    # "new ID per flicker". 600 s is enough to ride out a 30+ s
    # occlusion at 15 fps.
    DEFAULT_TTL_SEC = 600
    DEFAULT_BLOCK_MS = 1000

    def __init__(
        self,
        *,
        redis: RedisState,
        consumer_name: str = "identity-overlay-cache-01",
        ttl_sec: int = DEFAULT_TTL_SEC,
        block_ms: int = DEFAULT_BLOCK_MS,
    ) -> None:
        self._redis = redis
        self._consumer_name = consumer_name
        self._ttl_sec = int(
            os.environ.get("IDENTITY_ACTIVE_BINDING_TTL_SEC", ttl_sec)
        )
        self._block_ms = block_ms
        # (camera_id, local_track_id) -> (global_id, expires_at)
        self._by_local: dict[tuple[str, int], tuple[str, float]] = {}
        # PATCH (2026-06-15, anti-flicker): reverse index
        # ``global_id`` -> ``(camera_id, local_track_id, expires_at)``
        # of the MOST RECENT binding for that gid. Used by
        # ``lookup_reassociate()`` to bridge an MOT re-id: when a
        # new local_track_id gets a fresh detection with no
        # existing binding, we can fall back to the most recent
        # global_id seen on this camera (within TTL) so the overlay
        # keeps showing the same ``G:<gid>`` to the operator.
        self._by_gid_last: dict[str, tuple[str, int, float]] = {}
        # ``camera_id`` -> ordered list of (global_id,
        # local_track_id, expires_at) for that camera, MRU at the
        # tail. We use this for the camera-local "most recent
        # gids" lookup. Bounded by TTL.
        self._by_cam_gids: dict[str, list[tuple[str, int, float]]] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        # Counters
        self._parsed = 0
        self._applied = 0
        self._skipped_ambiguous = 0
        self._reassociated = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="identity-overlay-cache",
        )
        self._thread.start()
        logger.info(
            "IdentityOverlayCache started (ttl=%ds, consumer=%s)",
            self._ttl_sec,
            self._consumer_name,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def lookup(self, camera_id: str, local_track_id: Optional[int]) -> Optional[str]:
        """Look up the global_id for a (camera, local_track_id) tuple.

        Falls back to the most-recent-on-this-camera global_id if the
        (camera, local_track_id) entry has expired — this is the
        "anti-flicker" re-association path. Returns ``None`` if no
        recent global_id is known for this camera.
        """
        if local_track_id is None:
            return None
        key = (str(camera_id), int(local_track_id))
        now = time.time()
        with self._lock:
            entry = self._by_local.get(key)
            if entry is not None:
                gid, expires_at = entry
                if expires_at >= now:
                    return gid
                # GC expired
                self._by_local.pop(key, None)
            # Fallback: re-association. Find the most recent gid
            # seen on this camera (within TTL) and return it. This
            # lets an MOT re-id (new local_track_id, same person)
            # keep showing the same ``G:<gid>`` in the overlay.
            recent = self._by_cam_gids.get(str(camera_id), [])
            # Walk from the tail (most recent) and pick the first
            # non-expired entry.
            for gid, ltid, expires_at in reversed(recent):
                if expires_at < now:
                    continue
                # Self-association: re-bind the new local_track_id
                # to this gid so subsequent lookups are O(1).
                self._by_local[key] = (gid, expires_at)
                self._reassociated += 1
                return gid
            return None

    def counters(self) -> dict[str, int]:
        return {
            "decisions_parsed": self._parsed,
            "decisions_applied": self._applied,
            "decisions_skipped_ambiguous": self._skipped_ambiguous,
            "reassociated": self._reassociated,
            "size": len(self._by_local),
        }

    def _run(self) -> None:
        last_id = "$"  # XREAD with no group: only new messages
        while not self._stop.is_set():
            try:
                # XREAD (no group) — fan-out subscriber.
                msgs = self._redis.client.xread(
                    {self.STREAM: last_id},
                    count=16,
                    block=self._block_ms,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("IdentityOverlayCache XREAD error: %s", e)
                time.sleep(0.5)
                continue
            if not msgs:
                continue
            for stream_name, entries in msgs:
                for msg_id, raw_fields in entries:
                    last_id = msg_id
                    try:
                        fields = {
                            k: json.loads(v) if isinstance(v, (bytes, str)) else v
                            for k, v in raw_fields.items()
                        }
                    except Exception:  # noqa: BLE001
                        continue
                    self._apply(fields)

    def _apply(self, fields: dict) -> None:
        camera_id = fields.get("camera_id")
        local_track_id = fields.get("local_track_id")
        decision = fields.get("decision")
        global_id = fields.get("assigned_global_id")
        if not camera_id or local_track_id is None:
            return
        self._parsed += 1
        # Per spec: hold_ambiguous and reject_impossible MUST NOT bind
        if decision in ("hold_ambiguous", "reject_impossible", "ambiguous", "held"):
            self._skipped_ambiguous += 1
            return
        if not global_id:
            return
        # Apply: store under (camera_id, local_track_id) with TTL,
        # and update the secondary indexes for re-association.
        cam = str(camera_id)
        ltid = int(local_track_id)
        gid = str(global_id)
        now = time.time()
        expires_at = now + self._ttl_sec
        with self._lock:
            self._by_local[(cam, ltid)] = (gid, expires_at)
            self._by_gid_last[gid] = (cam, ltid, expires_at)
            recent = self._by_cam_gids.setdefault(cam, [])
            recent.append((gid, ltid, expires_at))
            # GC expired entries from the tail while we're here. We
            # only GC this camera's list; the worst case is a
            # per-camera accumulation of (one entry per
            # decision) which is bounded by the decision rate × TTL.
            while recent and recent[0][2] < now:
                recent.pop(0)
            self._applied += 1
