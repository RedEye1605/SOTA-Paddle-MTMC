"""PELSweeper — reclaim stuck pending entries from dead consumers.

When a consumer in a Redis Streams consumer group reads a message via
``XREADGROUP`` and dies before ``XACK``-ing it, the message stays in
that consumer's Pending Entry List (PEL) forever. No other consumer
in the same group will see it (the canonical failure mode of
fire-and-forget streaming).

This sweeper periodically:
  1. ``SCAN`` for ``stream:*`` keys.
  2. For each stream, ``XINFO GROUPS`` to enumerate the groups.
  3. For each (stream, group) pair, ``XAUTOCLAIM`` with a long
     ``min_idle_time`` to claim messages whose owner has been silent.
  4. Re-append the claimed payloads to the same stream, then ``XACK``
     the originals. Downstream workers consume only new ``>`` entries,
     so claiming without requeueing would strand the work under the
     sweeper consumer, and ACKing without requeueing would lose it.

The sweeper is a daemon thread. It does NOT participate in graceful
shutdown of the api process beyond ``thread.join(timeout=2)``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

from ..storage.redis_state import RedisState

logger = logging.getLogger(__name__)


STREAM_KEY_PATTERN = "stream:*"
SCAN_BATCH = 200
PEL_SWEEPER_NAME = "pel-sweeper"


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("env %s=%r is not an int, falling back to %d", key, raw, default)
        return default


class _Metrics:
    def __init__(self) -> None:
        self.pel_swept_total = 0
        self.pel_claimed_total = 0
        self.pel_deleted_total = 0
        self.pel_idle_seconds = 0.0
        self.pel_errors_total = 0
        self.pel_streams_scanned = 0
        self.pel_groups_scanned = 0
        self.pel_sweeps_total = 0
        self.pel_sweep_seconds = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "pel_swept_total": self.pel_swept_total,
            "pel_claimed_total": self.pel_claimed_total,
            "pel_deleted_total": self.pel_deleted_total,
            "pel_idle_seconds": self.pel_idle_seconds,
            "pel_errors_total": self.pel_errors_total,
            "pel_streams_scanned": self.pel_streams_scanned,
            "pel_groups_scanned": self.pel_groups_scanned,
            "pel_sweeps_total": self.pel_sweeps_total,
            "pel_sweep_seconds": self.pel_sweep_seconds,
        }


class PELSweeper:
    """Background sweeper that reclaims stuck PEL entries.

    Args:
        redis: RedisState instance to use for SCAN / XINFO / XAUTOCLAIM.
        interval_seconds: How often to run the sweep.
        min_idle_seconds: Only claim messages idle longer than this.
        claim_batch_size: Per XAUTOCLAIM call (the ``count`` argument).
    """

    def __init__(
        self,
        *,
        redis: RedisState,
        interval_seconds: int = 60,
        min_idle_seconds: int = 60,
        claim_batch_size: int = 100,
        consumer_name: str = PEL_SWEEPER_NAME,
    ) -> None:
        self._redis = redis
        self._interval_seconds = max(1, int(interval_seconds))
        self._min_idle_seconds = max(0, int(min_idle_seconds))
        self._claim_batch_size = max(1, int(claim_batch_size))
        self._consumer_name = consumer_name
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._metrics = _Metrics()
        self._min_idle_ms = self._min_idle_seconds * 1000

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="pel-sweeper",
        )
        self._thread.start()
        logger.info(
            "PELSweeper started (interval=%ds, min_idle=%ds, batch=%d, consumer=%s)",
            self._interval_seconds,
            self._min_idle_seconds,
            self._claim_batch_size,
            self._consumer_name,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def metrics(self) -> dict[str, float]:
        return self._metrics.as_dict()

    def sweep_once(self) -> None:
        """Run a single sweep. Exposed for tests."""
        self._do_sweep()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._do_sweep()
            except Exception as e:  # noqa: BLE001
                self._metrics.pel_errors_total += 1
                logger.warning("PELSweeper sweep error (continuing): %s", e)
            self._stop.wait(self._interval_seconds)

    def _do_sweep(self) -> None:
        started = time.monotonic()
        self._metrics.pel_sweeps_total += 1
        r = self._redis.client
        streams = self._scan_streams(r)
        self._metrics.pel_streams_scanned += len(streams)
        max_idle_ms = 0
        for stream in streams:
            groups = self._list_groups(r, stream)
            self._metrics.pel_groups_scanned += len(groups)
            for group in groups:
                idle_ms = self._sweep_group(r, stream, group)
                if idle_ms > max_idle_ms:
                    max_idle_ms = idle_ms
        if max_idle_ms > 0:
            self._metrics.pel_idle_seconds = max_idle_ms / 1000.0
        elapsed = time.monotonic() - started
        self._metrics.pel_sweep_seconds = elapsed
        if elapsed > 0.5:
            logger.info(
                "PELSweeper sweep took %.3fs (streams=%d)",
                elapsed,
                len(streams),
            )

    def _scan_streams(self, r: Any) -> list[str]:
        out: list[str] = []
        cursor = 0
        while True:
            cursor, keys = r.scan(
                match=STREAM_KEY_PATTERN,
                count=SCAN_BATCH,
            )
            out.extend(keys)
            if cursor == 0:
                break
        return out

    def _list_groups(self, r: Any, stream: str) -> list[str]:
        try:
            groups = r.xinfo_groups(stream)
        except Exception as e:  # noqa: BLE001
            self._metrics.pel_errors_total += 1
            logger.debug("XINFO GROUPS %s failed: %s", stream, e)
            return []
        out: list[str] = []
        for g in groups:
            name = g.get("name") if isinstance(g, dict) else None
            if name:
                out.append(name)
        return out

    def _sweep_group(self, r: Any, stream: str, group: str) -> int:
        """Sweep one (stream, group). Returns the max idle ms observed."""
        sample_idle_ms = self._sample_max_idle_ms(r, stream, group)
        cursor = "0-0"
        first = True
        while first or cursor != "0-0":
            first = False
            try:
                result = r.xautoclaim(
                    name=stream,
                    groupname=group,
                    consumername=self._consumer_name,
                    min_idle_time=self._min_idle_ms,
                    start_id=cursor,
                    count=self._claim_batch_size,
                )
            except Exception as e:  # noqa: BLE001
                self._metrics.pel_errors_total += 1
                logger.warning(
                    "XAUTOCLAIM %s/%s failed: %s", stream, group, e
                )
                return sample_idle_ms
            if not result:
                return sample_idle_ms
            next_cursor = result[0]
            claimed = result[1] if len(result) > 1 else []
            deleted = result[2] if len(result) > 2 else []
            if deleted:
                self._metrics.pel_deleted_total += len(deleted)
                logger.info(
                    "PELSweeper: %d deleted message(s) in %s/%s: %s",
                    len(deleted),
                    stream,
                    group,
                    list(deleted),
                )
            if claimed:
                self._metrics.pel_claimed_total += len(claimed)
                self._metrics.pel_swept_total += len(claimed)
                requeued = self._requeue_claimed(r, stream, group, claimed)
                logger.info(
                    "PELSweeper: claimed %d and requeued %d message(s) from %s/%s",
                    len(claimed),
                    requeued,
                    stream,
                    group,
                )
            cursor = next_cursor
            if isinstance(cursor, (bytes, bytearray)):
                cursor = cursor.decode()
        return sample_idle_ms

    def _sample_max_idle_ms(self, r: Any, stream: str, group: str) -> int:
        """Sample the max idle time of claimable messages in a group's PEL.

        Returns 0 if the PEL is empty or the call fails. We pass
        ``idle=self._min_idle_ms`` so we only see messages we'd
        actually claim on this sweep.
        """
        try:
            entries = r.xpending_range(
                name=stream,
                groupname=group,
                min="-",
                max="+",
                count=self._claim_batch_size,
                idle=self._min_idle_ms,
            )
        except Exception as e:  # noqa: BLE001
            self._metrics.pel_errors_total += 1
            logger.debug("XPENDING_RANGE %s/%s failed: %s", stream, group, e)
            return 0
        if not entries:
            return 0
        max_ms = 0
        for entry in entries:
            idle = entry.get("time_since_delivered", 0) if isinstance(entry, dict) else 0
            if idle > max_ms:
                max_ms = idle
        return int(max_ms)

    def _requeue_claimed(
        self,
        r: Any,
        stream: str,
        group: str,
        claimed: list[Any],
    ) -> int:
        msg_ids: list[str] = []
        requeued = 0
        for entry in claimed:
            if not entry or not entry[0]:
                continue
            msg_id = entry[0]
            fields = entry[1] if len(entry) > 1 else None
            if not fields:
                continue
            try:
                r.xadd(stream, dict(fields))
            except Exception as e:  # noqa: BLE001
                self._metrics.pel_errors_total += 1
                logger.warning(
                    "XADD requeue %s/%s (id=%s) failed: %s",
                    stream,
                    group,
                    msg_id,
                    e,
                )
                continue
            msg_ids.append(msg_id)
            requeued += 1
        if not msg_ids:
            return requeued
        try:
            r.xack(stream, group, *msg_ids)
        except Exception as e:  # noqa: BLE001
            self._metrics.pel_errors_total += 1
            logger.warning(
                "XACK %s/%s (ids=%s) failed: %s", stream, group, msg_ids, e
            )
        return requeued


def from_env(redis: RedisState) -> PELSweeper:
    return PELSweeper(
        redis=redis,
        interval_seconds=_env_int("PEL_SWEEP_INTERVAL_SECONDS", 60),
        min_idle_seconds=_env_int("PEL_MIN_IDLE_SECONDS", 60),
        claim_batch_size=_env_int("PEL_CLAIM_BATCH_SIZE", 100),
    )
