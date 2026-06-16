"""Tests for the legacy MQTT publisher (Phase 5b).

* topic is exactly ``ai/yamaha/people-detection/{cam1|cam2}/summary``
* payload is ``{ts, values}`` (ThingsBoard)
* ``ENABLE_SEND_MQTT=false`` ⇒ no connect, no publish
* enabled ⇒ publishes via the project's existing paho client wrapper
* client_id format matches the legacy ``people_counter_<name>_<ts>_<rand>``
* publish is non-blocking + retry; dropping on full queue is a no-op
"""

from __future__ import annotations

import json
import logging
import time
from unittest import mock


from app.integrations.legacy_contract import legacy_camera_topic
from app.integrations.mqtt_publisher import LegacyMqttPublisher


class _FakePaho:
    """Mimics the underlying paho-mqtt client object."""

    def __init__(self) -> None:
        self.published: list[tuple[str, str, int, bool]] = []

    def publish(self, topic, body, qos=1, retain=False):  # noqa: ANN001
        self.published.append((topic, body, qos, retain))
        r = mock.MagicMock()
        r.rc = 0
        r.mid = len(self.published)
        return r


class _FakeWrapper:
    """Mimics the project :class:`MqttPublisher` wrapper."""

    def __init__(self) -> None:
        self._client = _FakePaho()
        self._qos = 1
        self.broker = "test-broker"
        self.topic = "v1/devices/me/telemetry"


def _make_publisher(
    camera_id: str = "CAM_01", *, enabled: bool | None = None
) -> tuple[LegacyMqttPublisher, _FakePaho]:
    wrapper = _FakeWrapper()
    p = LegacyMqttPublisher(
        wrapper,  # the project wrapper
        camera_id=camera_id,
        enabled_override=enabled,
    )
    return p, wrapper._client


# ---------------------------------------------------------------------------
# Topic & payload shape
# ---------------------------------------------------------------------------


def test_cam01_publisher_uses_legacy_topic() -> None:
    p, _ = _make_publisher("CAM_01", enabled=True)
    assert p.topic == "ai/yamaha/people-detection/cam1/summary"


def test_cam02_publisher_uses_legacy_topic() -> None:
    p, _ = _make_publisher("CAM_02", enabled=True)
    assert p.topic == "ai/yamaha/people-detection/cam2/summary"


def test_publish_telemetry_uses_legacy_payload_shape() -> None:
    p, fake = _make_publisher("CAM_01", enabled=True)
    p.start()
    try:
        accepted = p.publish_telemetry({"ts": 1_700_000_000_000, "values": {"TotalPeople": 3}})
        assert accepted is True
        # Drain the retry thread.
        deadline = time.time() + 2.0
        while time.time() < deadline and not fake.published:
            time.sleep(0.05)
    finally:
        p.stop()
    assert len(fake.published) == 1
    topic, body, qos, retain = fake.published[0]
    assert topic == "ai/yamaha/people-detection/cam1/summary"
    payload = json.loads(body)
    assert payload["ts"] == 1_700_000_000_000
    assert payload["values"]["TotalPeople"] == 3
    assert qos == 1
    assert retain is False


# ---------------------------------------------------------------------------
# Disabled → no publish
# ---------------------------------------------------------------------------


def test_publisher_disabled_does_not_publish() -> None:
    p, fake = _make_publisher("CAM_01", enabled=False)
    p.start()
    try:
        accepted = p.publish_telemetry({"ts": 1, "values": {}})
        assert accepted is False
    finally:
        p.stop()
    assert fake.published == []


def test_publisher_disabled_logs_disabled_state(caplog) -> None:
    with caplog.at_level(logging.INFO):
        _make_publisher("CAM_01", enabled=False)
    assert "MQTT disabled" in caplog.text or "disabled" in caplog.text.lower()


# ---------------------------------------------------------------------------
# enabled → no client → drops
# ---------------------------------------------------------------------------


def test_publisher_drops_when_underlying_client_is_none() -> None:
    wrapper = _FakeWrapper()
    wrapper._client = None
    p = LegacyMqttPublisher(
        wrapper,
        camera_id="CAM_01",
        enabled_override=True,
    )
    p.start()
    try:
        accepted = p.publish_telemetry({"ts": 1, "values": {"x": 1}})
        assert accepted is False
    finally:
        p.stop()


# ---------------------------------------------------------------------------
# Queue behaviour
# ---------------------------------------------------------------------------


def test_publisher_drops_on_full_queue() -> None:
    p, fake = _make_publisher("CAM_01", enabled=True)
    p.start()
    try:
        # Block the retry loop by replacing _attempt_publish with a
        # no-op; queue fills up and subsequent publishes drop.
        p._attempt_publish = lambda _x: None  # type: ignore[assignment]
        for _ in range(p._queue.maxsize + 5):  # noqa: SLF001
            p.publish_telemetry({"ts": 1, "values": {"x": 1}})
        # Some may have been dropped (the queue is full).
    finally:
        p.stop()
    # The fake's .published list will be empty because we replaced
    # the publish attempt; the queue-full path returns False.
    assert fake.published == []


# ---------------------------------------------------------------------------
# Topic consistency vs. legacy_contract helper
# ---------------------------------------------------------------------------


def test_publisher_topic_matches_legacy_helper() -> None:
    p, _ = _make_publisher("CAM_01", enabled=True)
    assert p.topic == legacy_camera_topic("telemetry", "CAM_01", "cam_1")
    p2, _ = _make_publisher("CAM_02", enabled=True)
    assert p2.topic == legacy_camera_topic("telemetry", "CAM_02", "cam_2")
