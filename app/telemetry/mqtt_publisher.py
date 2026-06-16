"""Async, non-blocking MQTT periodic publisher (Phase 5).

Adaptation of the upstream ``Service/offline-people-counting`` pattern
(:mod:`app.io.mqtt_publisher`). The publisher owns:

* a bounded ``queue.Queue`` of pending payloads;
* a daemon retry thread that drains the queue;
* an *optional* periodic loop that calls a user-supplied
  ``data_callback()`` every ``interval_seconds`` to produce
  per-camera / per-zone summary payloads.

Hard rules:
  1. The publisher NEVER blocks the caller. If the queue is full or
     the client is not connected, ``publish_telemetry`` returns
     ``False`` and logs a single warning.
  2. Smoke-test mode is a hard short-circuit. The publisher
     refuses to enqueue authoritative telemetry in
     ``RuntimeMode.SMOKE_TEST``. Operators who want to
     smoke-test the wire can use ``force_smoke_telemetry=true`` to
     opt in (still tagged as smoke in the message).
  3. The retry thread is daemonised; closing the publisher stops
     it within ``stop_timeout_seconds``.
  4. Credentials are never logged. We log only broker host / topic
     / queue size.

The publisher is best used through ``TelemetryWorker`` which calls
``publish_telemetry(payload)`` from the per-decision and per-zone-event
hooks.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Callable, Optional

from ..core.runtime_mode import RuntimeMode, resolve_runtime_mode, smoke_log
from .mqtt_client import MqttPublisher

logger = logging.getLogger(__name__)


# Defaults; can be overridden per-instance.
QUEUE_MAXSIZE_DEFAULT = 1000
QUEUE_GET_TIMEOUT_SECONDS = 1.0
RECONNECT_REQUEUE_DELAY_SECONDS = 1.0
RETRY_LOOP_ERROR_DELAY_SECONDS = 1.0
MAX_PUBLISH_ATTEMPTS = 3
MAX_PUBLISH_BACKOFF_SECONDS = 30.0
STOP_TIMEOUT_SECONDS = 2.0
MILESTONE_LOG_INTERVAL = 100


class AsyncMqttPublisher:
    """Async, non-blocking, periodic MQTT publisher.

    Parameters
    ----------
    client
        A connected :class:`MqttPublisher` (we never construct the
        paho client ourselves).
    queue_maxsize
        Bound for the pending-publishes queue.
    mode
        Runtime mode; in ``SMOKE_TEST`` authoritative publishes are
        short-circuited unless ``force_smoke_telemetry`` is set.
    force_smoke_telemetry
        Allow the publisher to forward payloads in smoke-test mode.
        The payload values still include a ``smoke=True`` marker
        so downstream consumers can filter it out.
    """

    def __init__(
        self,
        client: MqttPublisher,
        *,
        queue_maxsize: int = QUEUE_MAXSIZE_DEFAULT,
        mode: Optional[RuntimeMode] = None,
        force_smoke_telemetry: bool = False,
    ) -> None:
        self._client = client
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=max(1, queue_maxsize))
        self._stop_event = threading.Event()
        self._retry_thread: Optional[threading.Thread] = None
        self._periodic_thread: Optional[threading.Thread] = None
        self._mode = mode or resolve_runtime_mode()
        self._force_smoke_telemetry = bool(force_smoke_telemetry)
        self.publish_count = 0
        self.last_publish_time: Optional[float] = None
        self.dropped_count = 0

    # ---- lifecycle ----
    def start(self) -> None:
        if self._retry_thread and self._retry_thread.is_alive():
            return
        self._stop_event.clear()
        self._retry_thread = threading.Thread(
            target=self._retry_loop,
            daemon=True,
            name="mqtt-publisher",
        )
        self._retry_thread.start()
        logger.info(
            "mqtt async publisher started | broker=%s | topic=%s | mode=%s",
            self._client.broker,
            self._client.topic,
            self._mode.value,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._retry_thread and self._retry_thread.is_alive():
            self._retry_thread.join(timeout=STOP_TIMEOUT_SECONDS)
        self._retry_thread = None
        if self._periodic_thread and self._periodic_thread.is_alive():
            self._periodic_thread.join(timeout=STOP_TIMEOUT_SECONDS)
        self._periodic_thread = None

    # ---- enqueue ----
    def publish_telemetry(self, data: dict, *, smoke_override: bool = False) -> bool:
        """Enqueue *data* for asynchronous publish.

        Returns ``True`` if the message was enqueued, ``False`` if it
        was dropped (queue full, smoke gate closed, client not
        connected).
        """
        if self._mode == RuntimeMode.SMOKE_TEST and not (
            self._force_smoke_telemetry or smoke_override
        ):
            smoke_log(
                "AsyncMqttPublisher",
                "refusing to enqueue authoritative telemetry in SMOKE_TEST",
            )
            self.dropped_count += 1
            return False
        if self._client is None or self._client._client is None:  # noqa: SLF001
            logger.debug("mqtt client not connected; dropping payload")
            self.dropped_count += 1
            return False
        try:
            self._queue.put_nowait(dict(data))
            return True
        except queue.Full:
            self.dropped_count += 1
            logger.warning(
                "mqtt publish queue full; dropping payload (dropped=%d)",
                self.dropped_count,
            )
            return False

    # ---- periodic loop ----
    def start_periodic(
        self,
        data_callback: Callable[[], Optional[dict]],
        interval_seconds: float = 5.0,
    ) -> None:
        """Run *data_callback* every *interval_seconds* and publish
        the resulting payload. The callback must return ``None`` to
        skip a tick (e.g. when no fresh data is available).
        """
        if self._periodic_thread and self._periodic_thread.is_alive():
            return
        stop = self._stop_event

        def _loop() -> None:
            while not stop.is_set():
                try:
                    payload = data_callback()
                except Exception as e:  # noqa: BLE001
                    logger.debug("periodic data_callback error: %s", e)
                    payload = None
                if payload is not None:
                    self.publish_telemetry(payload)
                stop.wait(timeout=max(0.1, interval_seconds))

        self._periodic_thread = threading.Thread(target=_loop, daemon=True, name="mqtt-periodic")
        self._periodic_thread.start()
        logger.info("mqtt periodic publish started | interval=%.1fs", interval_seconds)

    # ---- internals ----
    def _retry_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                try:
                    item = self._queue.get(timeout=QUEUE_GET_TIMEOUT_SECONDS)
                except queue.Empty:
                    continue
                self._attempt_publish(item)
            except Exception as e:  # noqa: BLE001
                logger.error("mqtt retry loop error: %s", e)
                time.sleep(RETRY_LOOP_ERROR_DELAY_SECONDS)

    def _attempt_publish(self, item: dict) -> None:
        body = json.dumps(item, default=str)
        for attempt in range(MAX_PUBLISH_ATTEMPTS):
            try:
                with self._client._lock:  # noqa: SLF001
                    info = self._client._client.publish(  # noqa: SLF001
                        self._client.topic, body, qos=self._client._qos
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("mqtt publish attempt failed (attempt=%d): %s", attempt, e)
                time.sleep(min(2**attempt, MAX_PUBLISH_BACKOFF_SECONDS))
                continue
            if info is not None and getattr(info, "rc", 0) == 0:
                self.publish_count += 1
                self.last_publish_time = time.time()
                if self.publish_count == 1 or self.publish_count % MILESTONE_LOG_INTERVAL == 0:
                    logger.info(
                        "mqtt publish milestone | count=%d | topic=%s",
                        self.publish_count,
                        self._client.topic,
                    )
                return
            time.sleep(min(2**attempt, MAX_PUBLISH_BACKOFF_SECONDS))
        self.dropped_count += 1
        logger.error("mqtt publish failed after %d attempts; dropping", MAX_PUBLISH_ATTEMPTS)
