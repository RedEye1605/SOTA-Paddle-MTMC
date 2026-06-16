"""Legacy-compatible MQTT publisher (Phase 5b).

Wraps the project's :class:`app.telemetry.mqtt_client.MqttPublisher` so
that outbound telemetry matches the legacy
``Service/offline-people-counting`` contract:

* topic = ``ai/yamaha/people-detection/{cam1|cam2}/summary``
* payload = ``{ts, values}`` (ThingsBoard)
* client_id = ``people_counter_{device_name}_{epoch}_{rand}``
* QoS = 1, retain = False, publish_interval = 3s
* never blocks the caller; bounded queue + retry thread

This module does NOT change the *internal* telemetry surfaces
(``TelemetryWorker`` still uses the existing MQTT client). It is
an opt-in shim for code paths that must emit the legacy topic /
payload shape — primarily the regression tests and the visual
validation script that pushes data-point summaries to ThingsBoard.

The new pipeline can also use this publisher end-to-end by setting
``MQTT_TOPIC=""`` and the env var ``USE_LEGACY_MQTT=true``; the
existing :class:`MqttPublisher` still works for the
``v1/devices/<token>/telemetry`` channel.
"""

from __future__ import annotations

import json
import logging
import queue
import time
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Callable, Optional

from ..telemetry.mqtt_client import MqttPublisher
from .legacy_contract import (
    flag_enabled,
    legacy_camera_topic,
    legacy_client_id,
    legacy_device_config,
)

logger = logging.getLogger(__name__)

MAX_PUBLISH_ATTEMPTS = 3
MAX_PUBLISH_BACKOFF_SECONDS = 30.0
QUEUE_GET_TIMEOUT_SECONDS = 1.0
RECONNECT_REQUEUE_DELAY_SECONDS = 1.0
RETRY_LOOP_ERROR_DELAY_SECONDS = 1.0
PUBLISH_MILESTONE_LOG_INTERVAL = 100


@dataclass
class _PendingPublish:
    data: dict
    attempt: int
    timestamp: float = field(default_factory=time.time)


class LegacyMqttPublisher:
    """Async, non-blocking publisher with the legacy topic + payload contract.

    Parameters
    ----------
    client
        A connected :class:`MqttPublisher`.
    camera_id
        New-pipeline camera id, e.g. ``"CAM_01"``.
    queue_maxsize
        Bounded queue length.
    enabled_override
        Force the ``ENABLE_SEND_MQTT`` toggle value (used by tests).
    """

    def __init__(
        self,
        client: MqttPublisher,
        *,
        camera_id: str,
        queue_maxsize: int = 1000,
        enabled_override: Optional[bool] = None,
    ) -> None:
        self._camera_id = camera_id
        self._device = legacy_device_config(camera_id)
        self._topic = legacy_camera_topic("telemetry", camera_id, self._device.device_name)
        self._client = client
        self._queue: queue.Queue[_PendingPublish] = queue.Queue(maxsize=max(1, queue_maxsize))
        self._pending: dict[int, _PendingPublish] = {}
        self._pending_lock = Lock()
        self._stop_event = Event()
        self._retry_thread: Optional[Thread] = None
        self._periodic_thread: Optional[Thread] = None
        self.publish_count = 0
        self.last_publish_time: Optional[float] = None
        self._enabled = (
            enabled_override if enabled_override is not None else flag_enabled("ENABLE_SEND_MQTT")
        )
        if not self._enabled:
            logger.info(
                "legacy MQTT publisher disabled by ENABLE_SEND_MQTT=false | camera=%s",
                camera_id,
            )

    @property
    def topic(self) -> str:
        """The legacy MQTT telemetry topic for this publisher."""
        return self._topic

    @property
    def device_name(self) -> str:
        return self._device.device_name

    @property
    def is_enabled(self) -> bool:
        return bool(self._enabled)

    def start(self) -> None:
        if not self._enabled:
            return
        if self._retry_thread and self._retry_thread.is_alive():
            return
        self._stop_event.clear()
        self._retry_thread = Thread(
            target=self._retry_loop, daemon=True, name="legacy-mqtt-publisher"
        )
        self._retry_thread.start()
        logger.info(
            "legacy MQTT publisher started | topic=%s | camera=%s | client_id=%s",
            self._topic,
            self._camera_id,
            legacy_client_id(self._device.device_name),
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._retry_thread and self._retry_thread.is_alive():
            self._retry_thread.join(timeout=2.0)
        self._retry_thread = None
        if self._periodic_thread and self._periodic_thread.is_alive():
            self._periodic_thread.join(timeout=2.0)
        self._periodic_thread = None

    # ---- producer side ----
    def publish_telemetry(self, data: dict) -> bool:
        if not self._enabled:
            logger.debug(
                "legacy MQTT publish skipped (disabled) | topic=%s | camera=%s",
                self._topic,
                self._camera_id,
            )
            return False
        client = getattr(self._client, "_client", None)
        if client is None:
            return False
        try:
            self._queue.put_nowait(_PendingPublish(data=dict(data), attempt=0))
            return True
        except queue.Full:
            logger.error("legacy mqtt publish queue full; dropping message")
            return False

    def start_periodic(
        self,
        data_callback: Callable[[], Optional[dict]],
        interval_seconds: int = 3,
    ) -> None:
        """Mirror the legacy MQTTPublisher.start_periodic(interval=3)."""
        if not self._enabled:
            return
        if self._periodic_thread and self._periodic_thread.is_alive():
            return
        stop = self._stop_event

        def _loop() -> None:
            while not stop.is_set():
                try:
                    payload = data_callback()
                    if payload:
                        self.publish_telemetry(payload)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("legacy mqtt periodic publish error: %s", exc)
                stop.wait(timeout=interval_seconds)

        self._periodic_thread = Thread(target=_loop, daemon=True, name="legacy-mqtt-periodic")
        self._periodic_thread.start()
        logger.info(
            "legacy mqtt periodic started | topic=%s | interval=%ds",
            self._topic,
            interval_seconds,
        )

    # ---- internals ----
    def _retry_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                try:
                    item = self._queue.get(timeout=QUEUE_GET_TIMEOUT_SECONDS)
                except queue.Empty:
                    continue
                client = getattr(self._client, "_client", None)
                if client is None:
                    self._requeue(item)
                    time.sleep(RECONNECT_REQUEUE_DELAY_SECONDS)
                    continue
                self._attempt_publish(item)
            except Exception as exc:  # noqa: BLE001
                logger.error("legacy mqtt retry processor error: %s", exc)
                time.sleep(RETRY_LOOP_ERROR_DELAY_SECONDS)

    def _attempt_publish(self, item: _PendingPublish) -> None:
        client = getattr(self._client, "_client", None)
        if client is None:
            self._requeue(item)
            return
        body = json.dumps(item.data, default=str)
        try:
            result = client.publish(
                self._topic,
                body,
                qos=self._client._qos,
                retain=False,  # noqa: SLF001
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("legacy mqtt publish call failed: %s", exc)
            if item.attempt < MAX_PUBLISH_ATTEMPTS - 1:
                self._requeue(_PendingPublish(data=item.data, attempt=item.attempt + 1))
                time.sleep(min(2**item.attempt, MAX_PUBLISH_BACKOFF_SECONDS))
            return
        # On paho-mqtt v2 the result has .rc (int) and .mid (int).
        rc = getattr(result, "rc", 0)
        mid = getattr(result, "mid", None)
        if rc == 0 and mid is not None:
            with self._pending_lock:
                self._pending[mid] = item
            return
        logger.error("legacy mqtt publish failed | rc=%s | attempt=%d", rc, item.attempt)
        if item.attempt < MAX_PUBLISH_ATTEMPTS - 1:
            time.sleep(min(2**item.attempt, MAX_PUBLISH_BACKOFF_SECONDS))
            self._requeue(_PendingPublish(data=item.data, attempt=item.attempt + 1))

    def _requeue(self, item: _PendingPublish) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logger.error("legacy mqtt publish queue full on reconnect; dropping")
