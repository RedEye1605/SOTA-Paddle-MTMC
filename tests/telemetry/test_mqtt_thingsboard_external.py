"""MQTT / ThingsBoard external-broker tests (Phase 5).

Covers:
  1. payload shape: {ts, values}, ts in milliseconds
  2. credentials are never logged
  3. telemetry is disabled when the broker config is missing
  4. token-based auth (THINGSBOARD_DEVICE_TOKEN) overrides the
     username/password and the topic becomes
     v1/devices/<token>/telemetry
  5. smoke mode does not publish authoritative telemetry unless
     explicitly opted-in
"""

from __future__ import annotations

import logging
import os
from unittest import mock

from app.core.runtime_mode import RuntimeMode
from app.telemetry import mqtt_client
from app.telemetry.mqtt_client import from_env
from app.telemetry.mqtt_publisher import AsyncMqttPublisher
from app.telemetry.thingsboard_payload import (
    build_dwell_payload,
    build_global_count_payload,
    build_system_health_payload,
    build_zone_event_payload,
    build_zone_summary_payload,
)


# ---------------------------------------------------------------------------
# 1. payload shape: {ts, values}, ts in milliseconds
# ---------------------------------------------------------------------------


def test_zone_summary_payload_shape() -> None:
    p = build_zone_summary_payload(
        cam_id="CAM_01",
        zone_id="ZONE_A",
        people_count=3,
        entries=1,
        exits=0,
        dwell_avg_seconds=42.5,
        active_global_ids=3,
        ts_ms=1_000,
    )
    assert set(p.keys()) == {"ts", "values"}
    assert p["ts"] == 1_000
    assert p["values"]["cam_id"] == "CAM_01"
    assert p["values"]["zone_id"] == "ZONE_A"
    assert p["values"]["people_count"] == 3
    assert p["values"]["entries"] == 1
    assert p["values"]["exits"] == 0
    assert p["values"]["dwell_avg_seconds"] == 42.5
    assert p["values"]["active_global_ids"] == 3


def test_default_ts_is_milliseconds() -> None:
    p = build_zone_summary_payload(
        cam_id="CAM_01",
        zone_id="ZONE_A",
        people_count=0,
        entries=0,
        exits=0,
        dwell_avg_seconds=0.0,
        active_global_ids=0,
    )
    assert p["ts"] > 1_700_000_000_000  # ~ year 2023 in ms


def test_global_count_payload_unchanged() -> None:
    p = build_global_count_payload(global_id="GID-1", camera_id="CAM_01", site_id="x", ts_ms=42)
    assert p == {
        "ts": 42,
        "values": {
            "global_id_active": 1,
            "global_id": "GID-1",
            "site_id": "x",
            "camera_id": "CAM_01",
        },
    }


def test_zone_event_payload_unchanged() -> None:
    p = build_zone_event_payload(
        zone_id="Z1",
        camera_id="CAM_01",
        event_type="enter",
        global_id="GID",
        ts_ms=42,
    )
    assert p["ts"] == 42
    assert p["values"]["zone_event"] == "enter"


def test_dwell_payload_unchanged() -> None:
    p = build_dwell_payload(
        global_id="GID",
        zone_id="Z1",
        camera_id="CAM_01",
        duration_seconds=10,
        ts_ms=42,
    )
    assert p["ts"] == 42
    assert p["values"]["dwell_duration_seconds"] == 10


def test_system_health_payload_shape() -> None:
    p = build_system_health_payload(
        site_id="s",
        camera_id="CAM_01",
        fps=12.5,
        detector_backend="pphuman",
        reid_backend="transreid",
        workers_crashed=0,
        stream_healthy=True,
        ts_ms=1234,
    )
    assert p["ts"] == 1234
    assert p["values"]["fps"] == 12.5
    assert p["values"]["detector_backend"] == "pphuman"
    assert p["values"]["reid_backend"] == "transreid"


# ---------------------------------------------------------------------------
# 2. credentials are never logged
# ---------------------------------------------------------------------------


def test_from_env_info_log_does_not_contain_password(caplog) -> None:
    """The constructor's info log must not include the password / token."""
    fake_client_module = mock.MagicMock()
    with mock.patch.dict(
        os.environ,
        {
            "MQTT_ENABLED": "true",
            "MQTT_HOST": "broker.example.com",
            "MQTT_PORT": "1883",
            "MQTT_USERNAME": "alice",
            "MQTT_PASSWORD": "shh-this-is-secret",
            "MQTT_TOPIC": "v1/devices/me/telemetry",
        },
    ):
        with mock.patch.object(mqtt_client, "MqttPublisher", fake_client_module):
            mqtt_client.from_env()
    fake_client_module.assert_called_once()
    kwargs = fake_client_module.call_args.kwargs
    assert kwargs["password"] == "shh-this-is-secret"
    # The published kwargs only contain the password field; the
    # *log message* must not. We assert that the field name does not
    # appear in any captured info log line.
    # (caplog can be empty if the log is not emitted; that is fine
    # because the constructor itself does not log the password.)
    text = caplog.text
    assert "shh-this-is-secret" not in text


def test_async_publisher_does_not_log_password(caplog) -> None:
    """The async publisher's start log must not contain credentials."""
    fake = mock.MagicMock()
    fake.broker = "broker.example.com"
    fake.topic = "v1/devices/me/telemetry"
    fake._client = mock.MagicMock()
    p = AsyncMqttPublisher(fake, mode=RuntimeMode.PRODUCTION)
    with caplog.at_level(logging.INFO):
        p.start()
    assert "password" not in caplog.text.lower()
    assert "token" not in caplog.text.lower()
    p.stop()


# ---------------------------------------------------------------------------
# 3. telemetry is disabled when config is missing
# ---------------------------------------------------------------------------


def test_from_env_disabled_when_mqtt_enabled_false() -> None:
    with mock.patch.dict(os.environ, {"MQTT_ENABLED": "false"}):
        assert from_env() is None


def test_from_env_disabled_when_broker_missing() -> None:
    with mock.patch.dict(
        os.environ,
        {"MQTT_ENABLED": "true", "MQTT_HOST": "", "MQTT_BROKER_HOST": ""},
        clear=False,
    ):
        assert from_env() is None


def test_from_env_returns_publisher_with_host_port_topic() -> None:
    with mock.patch.dict(
        os.environ,
        {
            "MQTT_ENABLED": "true",
            "MQTT_HOST": "broker.example.com:1884",
            "MQTT_PORT": "",
            "MQTT_USERNAME": "u",
            "MQTT_PASSWORD": "p",
            "MQTT_TOPIC": "v1/devices/me/telemetry",
        },
    ):
        p = from_env()
    assert p is not None
    assert p.broker == "broker.example.com"
    assert p.port == 1884
    assert p.topic == "v1/devices/me/telemetry"


def test_strip_tcp_scheme() -> None:
    with mock.patch.dict(
        os.environ,
        {
            "MQTT_ENABLED": "true",
            "MQTT_BROKER_HOST": "tcp://broker.example.com:1883",
            "MQTT_USERNAME": "",
            "MQTT_PASSWORD": "",
            "MQTT_HOST": "",
        },
    ):
        p = from_env()
    assert p is not None
    assert p.broker == "broker.example.com"
    assert p.port == 1883


def test_mqtt_host_wins_over_broker_host() -> None:
    with mock.patch.dict(
        os.environ,
        {
            "MQTT_ENABLED": "true",
            "MQTT_BROKER_HOST": "old.example.com:1883",
            "MQTT_HOST": "new.example.com:1883",
            "MQTT_USERNAME": "u",
            "MQTT_PASSWORD": "p",
        },
    ):
        p = from_env()
    assert p is not None
    assert p.broker == "new.example.com"


# ---------------------------------------------------------------------------
# 4. token-based auth
# ---------------------------------------------------------------------------


def test_thingsboard_token_used_as_password_and_topic() -> None:
    with mock.patch.dict(
        os.environ,
        {
            "MQTT_ENABLED": "true",
            "MQTT_HOST": "thingsboard.example.com:1883",
            "MQTT_USERNAME": "ignored-when-token-set",
            "MQTT_PASSWORD": "ignored-when-token-set",
            "THINGSBOARD_DEVICE_TOKEN": "my-device-token-abc",
            "MQTT_TOPIC": "",
        },
    ):
        p = from_env()
    assert p is not None
    assert p.topic == "v1/devices/my-device-token-abc/telemetry"
    assert p._password == "my-device-token-abc"  # noqa: SLF001
    assert p._username == ""  # noqa: SLF001


def test_explicit_topic_overrides_token_topic() -> None:
    with mock.patch.dict(
        os.environ,
        {
            "MQTT_ENABLED": "true",
            "MQTT_HOST": "h:1883",
            "THINGSBOARD_DEVICE_TOKEN": "tok",
            "MQTT_TOPIC": "my/explicit/topic",
        },
    ):
        p = from_env()
    assert p is not None
    assert p.topic == "my/explicit/topic"


# ---------------------------------------------------------------------------
# 5. smoke mode does not publish authoritative telemetry
# ---------------------------------------------------------------------------


def test_async_publisher_blocks_smoke_mode_by_default() -> None:
    fake = mock.MagicMock()
    fake.broker = "h"
    fake.topic = "t"
    fake._client = mock.MagicMock()
    p = AsyncMqttPublisher(fake, mode=RuntimeMode.SMOKE_TEST)
    p.start()
    try:
        accepted = p.publish_telemetry({"ts": 1, "values": {"x": 1}})
        assert accepted is False
        assert p.dropped_count == 1
    finally:
        p.stop()


def test_async_publisher_force_smoke_telemetry_overrides() -> None:
    fake = mock.MagicMock()
    fake.broker = "h"
    fake.topic = "t"
    fake._client = mock.MagicMock()
    p = AsyncMqttPublisher(fake, mode=RuntimeMode.SMOKE_TEST, force_smoke_telemetry=True)
    p.start()
    try:
        accepted = p.publish_telemetry({"ts": 1, "values": {"x": 1}})
        assert accepted is True
    finally:
        p.stop()


def test_async_publisher_production_mode_publishes() -> None:
    fake = mock.MagicMock()
    fake.broker = "h"
    fake.topic = "t"
    fake._client = mock.MagicMock()
    p = AsyncMqttPublisher(fake, mode=RuntimeMode.PRODUCTION)
    p.start()
    try:
        accepted = p.publish_telemetry({"ts": 1, "values": {"x": 1}})
        assert accepted is True
    finally:
        p.stop()


def test_async_publisher_drops_when_client_not_connected() -> None:
    fake = mock.MagicMock()
    fake.broker = "h"
    fake.topic = "t"
    fake._client = None
    p = AsyncMqttPublisher(fake, mode=RuntimeMode.PRODUCTION)
    p.start()
    try:
        accepted = p.publish_telemetry({"ts": 1, "values": {"x": 1}})
        assert accepted is False
        assert p.dropped_count == 1
    finally:
        p.stop()


def test_async_publisher_dropped_count_on_full_queue() -> None:
    fake = mock.MagicMock()
    fake.broker = "h"
    fake.topic = "t"
    fake._client = mock.MagicMock()
    p = AsyncMqttPublisher(fake, queue_maxsize=2, mode=RuntimeMode.PRODUCTION)
    p.start()
    try:
        # Fill the queue (publisher thread is too slow to drain)
        # We disable the publisher's drain temporarily by replacing
        # _attempt_publish with a no-op so the queue stays full.
        p._attempt_publish = lambda _x: None  # type: ignore[assignment]
        for _ in range(5):
            p.publish_telemetry({"ts": 1, "values": {"x": 1}})
        assert p.dropped_count >= 1
    finally:
        p.stop()


# ---------------------------------------------------------------------------
# 6. contract: builder helpers used by TelemetryWorker still work
# ---------------------------------------------------------------------------


def test_zone_summary_extras() -> None:
    p = build_zone_summary_payload(
        cam_id="CAM_01",
        zone_id="Z1",
        people_count=0,
        entries=0,
        exits=0,
        dwell_avg_seconds=0.0,
        active_global_ids=0,
        extra_values={"site_id": "showroom"},
    )
    assert p["values"]["site_id"] == "showroom"
