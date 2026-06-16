"""Telemetry worker.

Consumes `stream:identity_decisions` and `stream:zone_events` and
publishes MQTT ThingsBoard `{ts, values}` payloads. Also writes dwell
sessions to PostgreSQL.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from ..storage.postgres import PostgresStore
from ..storage.redis_state import RedisState
from ..telemetry.mqtt_client import MqttPublisher
from ..telemetry.thingsboard_payload import build_global_count_payload
from ..zones.dwell import DwellBookkeeper

logger = logging.getLogger(__name__)


class TelemetryWorker:
    def __init__(
        self,
        *,
        pg: PostgresStore,
        redis: RedisState,
        mqtt: Optional[MqttPublisher] = None,
        site_id: str = "default_site",
    ) -> None:
        self.pg = pg
        self.redis = redis
        self.mqtt = mqtt
        self.site_id = site_id
        self.dwell = DwellBookkeeper()

    def on_identity_decision(self, decision: dict[str, Any]) -> None:
        """Stream `stream:identity_decisions` -> no-op for telemetry payload;
        decisions are persisted by the resolver. We may publish a per-identity
        state if the GID was newly created."""
        if decision.get("decision") == "new" and decision.get("assigned_global_id"):
            # PATCH (2026-06-15): include local_track_id only when
            # SHOW_LOCAL_TRACK_ID=true (operator's spec). The default
            # is False — local_track_id is debug-only.
            extra: dict = {}
            if os.environ.get("SHOW_LOCAL_TRACK_ID", "false").lower() == "true":
                lid = decision.get("local_track_id")
                if lid is not None:
                    extra["local_track_id"] = int(lid)
            payload = build_global_count_payload(
                global_id=decision["assigned_global_id"],
                camera_id=decision.get("camera_id"),
                site_id=self.site_id,
                extra_values=extra or None,
            )
            if self.mqtt is not None:
                self._publish(decision.get("camera_id"), payload)

    def on_zone_event(self, event: dict[str, Any]) -> None:
        self.pg.insert_zone_event(
            global_id=event["global_id"],
            tracklet_id=event["tracklet_id"],
            camera_id=event["camera_id"],
            zone_id=event["zone_id"],
            event_type=event["event_type"],
            ts=float(event.get("timestamp") or time.time()),
            confidence=event.get("confidence"),
        )
        # Open/close dwell
        result = self.dwell.on_event(
            global_id=event["global_id"],
            zone_id=event["zone_id"],
            camera_id=event["camera_id"],
            event_type=event["event_type"],
            ts=float(event.get("timestamp") or time.time()),
        )
        if result is not None:
            self.pg.upsert_dwell(
                global_id=result["global_id"],
                zone_id=result["zone_id"],
                camera_id=result["camera_id"],
                ts=float(result.get("entered_at") or result.get("exited_at") or time.time()),
                event_type="enter" if result["kind"] == "open" else "exit",
            )

    def _publish(self, camera_id: str | None, payload: dict[str, Any]) -> None:
        """Route a payload to the right per-camera topic.

        In legacy contract mode the publisher switches topic to
        ``{topic_base}/{cam1|cam2}/summary`` per call. In new mode
        (THINGSBOARD_DEVICE_TOKEN set) the camera id is ignored and
        the single configured topic is used.
        """
        if self.mqtt is None:
            return
        if getattr(self.mqtt, "legacy_contract", False) and camera_id:
            self.mqtt.publish_for_camera(camera_id, payload)
        else:
            self.mqtt.publish(payload)

    def run(self, stop_event=None) -> None:
        self.redis.ensure_group("stream:identity_decisions", "telemetry_workers")
        self.redis.ensure_group("stream:zone_events", "telemetry_workers")
        # PATCH (2026-06-17, dwell force-close): ``DwellBookkeeper``
        # has a ``force_close_stale()`` method that closes any open
        # dwell session older than ``max_open_seconds`` (24 h by
        # default). Without this periodic call, a person who leaves
        # the showroom without triggering an ``exit`` event (a
        # dropped zone transition, a camera outage) leaves the dwell
        # row in ``status='open'`` forever, and the
        # ``/dwell/summary`` endpoint reports ever-growing durations.
        # Run the sweep once per minute.
        last_force_close = 0.0
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            now = time.time()
            if (now - last_force_close) > 60.0:
                for closed in self.dwell.force_close_stale(now=now):
                    self.pg.upsert_dwell(
                        global_id=closed["global_id"],
                        zone_id=closed["zone_id"],
                        camera_id=closed["camera_id"],
                        ts=float(closed.get("exited_at") or now),
                        event_type="exit",
                    )
                last_force_close = now
            for stream, group, consumer, handler in [
                (
                    "stream:identity_decisions",
                    "telemetry_workers",
                    "telemetry-worker-01",
                    self.on_identity_decision,
                ),
                (
                    "stream:zone_events",
                    "telemetry_workers",
                    "telemetry-worker-zone-01",
                    self.on_zone_event,
                ),
            ]:
                msgs = self.redis.consume(
                    stream,
                    group,
                    consumer,
                    count=8,
                    block_ms=500,
                )
                for msg_id, fields in msgs:
                    try:
                        handler(fields)
                    except Exception as e:  # noqa: BLE001
                        logger.exception(
                            "telemetry handler failed for %s/%s: %s",
                            stream,
                            msg_id,
                            e,
                        )
                        continue
                    else:
                        self.redis.ack(stream, group, msg_id)
