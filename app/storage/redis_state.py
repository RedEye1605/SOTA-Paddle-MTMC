"""Redis — active state, recent cache, streams."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import redis

logger = logging.getLogger(__name__)


class RedisState:
    """Thin wrapper around redis-py.

    Keys:
      active:{camera_id}:{local_track_id}        -> global_id (TTL 600s)
      recent:global:{global_id}                  -> last_seen (TTL 86400s)
      camera:last_seen:{camera_id}               -> last_seen (TTL 86400s)
      tracklet_buffer:{camera_id}:{local_track_id} -> list of crop URIs (TTL 300s)

    PATCH (2026-06-17, BUG-2): the previous 60 s TTL on
    ``active:{camera_id}:{local_track_id}`` was too short for the
    restart-recovery path: a fresh IdentityOverlayCache, on api
    restart, XADDs a recovery query and most keys were already gone.
    The IdentityOverlayCache's in-process TTL is 600 s (10 min) per
    the anti-flicker spec; the Redis key should outlive at least
    one full in-process cache cycle so a restart doesn't lose the
    binding. Bump to 600 s.
    """

    def __init__(
        self,
        host: str,
        port: int,
        db: int = 0,
        password: str = "",
        ttl_local_binding: int = 600,  # PATCH (2026-06-17, BUG-2): 60s -> 600s
        ttl_tracklet_buffer: int = 300,
        ttl_recent_identity: int = 86_400,
    ) -> None:
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self.ttl_local_binding = ttl_local_binding
        self.ttl_tracklet_buffer = ttl_tracklet_buffer
        self.ttl_recent_identity = ttl_recent_identity
        self._client: Optional[redis.Redis] = None

    def connect(self) -> None:
        if self._client is not None:
            return
        self._client = redis.Redis(
            host=self._host,
            port=self._port,
            db=self._db,
            password=self._password or None,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        logger.info("Redis ready: %s:%d db=%d", self._host, self._port, self._db)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    @property
    def client(self) -> redis.Redis:
        assert self._client is not None, "RedisState.connect() first"
        return self._client

    # ---- key helpers ----
    def _active_key(self, camera_id: str, local_track_id: int) -> str:
        return f"active:{camera_id}:{local_track_id}"

    def _recent_key(self, global_id: str) -> str:
        return f"recent:global:{global_id}"

    def _last_seen_key(self, camera_id: str) -> str:
        return f"camera:last_seen:{camera_id}"

    def _buffer_key(self, camera_id: str, local_track_id: int) -> str:
        return f"tracklet_buffer:{camera_id}:{local_track_id}"

    # ---- active binding ----
    def set_active(self, camera_id: str, local_track_id: int, global_id: str) -> None:
        self.client.setex(
            self._active_key(camera_id, local_track_id),
            self.ttl_local_binding,
            global_id,
        )

    def get_active(self, camera_id: str, local_track_id: int) -> Optional[str]:
        return self.client.get(self._active_key(camera_id, local_track_id))

    def delete_active(self, camera_id: str, local_track_id: int) -> None:
        self.client.delete(self._active_key(camera_id, local_track_id))

    # ---- recent identity ----
    def mark_recent(self, global_id: str, ts: float, camera_id: str) -> None:
        self.client.setex(
            self._recent_key(global_id),
            self.ttl_recent_identity,
            json.dumps({"last_seen": ts, "camera_id": camera_id}),
        )

    def get_recent(self, global_id: str) -> Optional[dict[str, Any]]:
        raw = self.client.get(self._recent_key(global_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    # ---- camera last seen ----
    def mark_camera_last_seen(self, camera_id: str, ts: float) -> None:
        self.client.setex(self._last_seen_key(camera_id), self.ttl_recent_identity, ts)

    def get_camera_last_seen(self, camera_id: str) -> Optional[float]:
        raw = self.client.get(self._last_seen_key(camera_id))
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    # ---- tracklet buffer ----
    def append_crop(self, camera_id: str, local_track_id: int, crop_uri: str) -> int:
        """Returns the new buffer length (or 0 if expired)."""
        key = self._buffer_key(camera_id, local_track_id)
        pipe = self.client.pipeline()
        pipe.rpush(key, crop_uri)
        pipe.expire(key, self.ttl_tracklet_buffer)
        results = pipe.execute()
        return int(results[0])

    def fetch_buffer(self, camera_id: str, local_track_id: int) -> list[str]:
        return list(self.client.lrange(self._buffer_key(camera_id, local_track_id), 0, -1))

    def clear_buffer(self, camera_id: str, local_track_id: int) -> None:
        self.client.delete(self._buffer_key(camera_id, local_track_id))

    # ---- streams ----
    def ensure_group(self, stream: str, group: str, start_id: str = "0") -> None:
        try:
            self.client.xgroup_create(stream, group, id=start_id, mkstream=True)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def publish(
        self,
        stream: str,
        payload: dict[str, Any],
        maxlen: int = 1_000_000,
    ) -> str:
        """XADD with MAXLEN ~ cap to keep streams bounded.

        A 24h+ session can produce tens of millions of detection
        events; without a MAXLEN the stream keys grow without bound
        and Redis memory pressure follows. Per Redis docs,
        ``MAXLEN ~ N`` is O(1) per XADD so this is safe on the hot
        path. Callers can override ``maxlen`` per stream (e.g.
        detections at 2M, embeddings at 250K) by passing the
        argument.
        """
        return self.client.xadd(
            stream,
            {k: json.dumps(v) for k, v in payload.items()},
            maxlen=maxlen,
            approximate=True,
        )

    def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 10,
        block_ms: int = 1000,
    ) -> list[tuple[str, dict[str, Any]]]:
        resp = self.client.xreadgroup(
            group,
            consumer,
            {stream: ">"},
            count=count,
            block=block_ms,
        )
        out: list[tuple[str, dict[str, Any]]] = []
        for _, entries in resp or []:
            for msg_id, fields in entries:
                decoded = {k: json.loads(v) for k, v in fields.items()}
                out.append((msg_id, decoded))
        return out

    def ack(self, stream: str, group: str, *msg_ids: str) -> None:
        if msg_ids:
            self.client.xack(stream, group, *msg_ids)

    def stream_len(self, stream: str) -> int:
        try:
            return int(self.client.xlen(stream))
        except redis.ResponseError:
            return 0

    # ---- health ----
    def healthcheck(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception as e:  # noqa: BLE001
            logger.error("Redis healthcheck failed: %s", e)
            return False


def from_env() -> RedisState:
    return RedisState(
        host=os.environ.get("REDIS_HOST", "message-bus"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        db=int(os.environ.get("REDIS_DB", "0")),
        password=os.environ.get("REDIS_PASSWORD", ""),
    )
