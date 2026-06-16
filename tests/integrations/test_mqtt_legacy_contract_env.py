"""End-to-end tests for the legacy MQTT contract via the project's
existing :class:`MqttPublisher` (Phase 5b integration).

These tests pin the behaviour ``from_env()`` must exhibit when
``USE_LEGACY_MQTT_CONTRACT=true`` (default) AND no
``THINGSBOARD_DEVICE_TOKEN`` is set: the per-camera publish must
go to the legacy topic shape.
"""

from __future__ import annotations

import os
from unittest import mock


from app.telemetry.mqtt_client import MqttPublisher, from_env


def _patched_env(env: dict[str, str]):
    return mock.patch.dict(os.environ, env, clear=False)


def test_from_env_legacy_topic_base_for_legacy_broker() -> None:
    with mock.patch.dict(
        os.environ,
        {
            "MQTT_ENABLED": "true",
            "MQTT_BROKER_HOST": "mqtt.example.invalid",
            "MQTT_USERNAME": "<MQTT_CREDENTIAL>",
            "MQTT_PASSWORD": "<MQTT_CREDENTIAL>",
            "MQTT_TOPIC_BASE": "ai/yamaha/people-detection",
            "USE_LEGACY_MQTT_CONTRACT": "true",
            "THINGSBOARD_DEVICE_TOKEN": "",
        },
        clear=False,
    ):
        p = from_env()
    assert p is not None
    assert p.broker == "mqtt.example.invalid"
    assert p.topic == "ai/yamaha/people-detection"
    assert p.legacy_contract is True


def test_from_env_legacy_disabled_when_token_set() -> None:
    with mock.patch.dict(
        os.environ,
        {
            "MQTT_ENABLED": "true",
            "MQTT_BROKER_HOST": "mqtt.example.invalid",
            "MQTT_USERNAME": "ignored",
            "MQTT_PASSWORD": "ignored",
            "THINGSBOARD_DEVICE_TOKEN": "my-device-token",
            "USE_LEGACY_MQTT_CONTRACT": "true",  # ignored when token is set
        },
        clear=False,
    ):
        p = from_env()
    assert p is not None
    # The token path is the v1/devices/<token>/telemetry channel.
    assert p.topic == "v1/devices/my-device-token/telemetry"
    assert p.legacy_contract is False


def test_from_env_legacy_disabled_by_env_var() -> None:
    with mock.patch.dict(
        os.environ,
        {
            "MQTT_ENABLED": "true",
            "MQTT_BROKER_HOST": "mqtt.example.invalid",
            "USE_LEGACY_MQTT_CONTRACT": "false",
        },
        clear=False,
    ):
        p = from_env()
    assert p is not None
    assert p.legacy_contract is False


# ---------------------------------------------------------------------------
# publish_for_camera
# ---------------------------------------------------------------------------


def test_publish_for_camera_appends_legacy_suffix() -> None:
    """In legacy mode, publish_for_camera('CAM_01', ...) goes to
    ai/yamaha/people-detection/cam1/summary."""
    p = MqttPublisher(
        broker_host="mqtt.example.invalid",
        port=1883,
        username="user",  # test fixture
        password="x",  # test fixture (1-char placeholder, never used)
        topic_base="ai/yamaha/people-detection",
        legacy_contract=True,
    )
    fake_client = mock.MagicMock()
    p._client = fake_client  # noqa: SLF001
    p.publish_for_camera("CAM_01", {"ts": 1, "values": {"x": 1}})
    fake_client.publish.assert_called_once()
    args, kwargs = fake_client.publish.call_args
    assert args[0] == "ai/yamaha/people-detection/cam1/summary"
    assert kwargs.get("qos") == 1  # qos passed as kwarg


def test_publish_for_camera_cam02() -> None:
    p = MqttPublisher(
        broker_host="h",
        port=1883,
        topic_base="ai/yamaha/people-detection",
        legacy_contract=True,
    )
    fake_client = mock.MagicMock()
    p._client = fake_client  # noqa: SLF001
    p.publish_for_camera("CAM_02", {"ts": 1, "values": {"x": 1}})
    args, _ = fake_client.publish.call_args
    assert args[0] == "ai/yamaha/people-detection/cam2/summary"


def test_publish_for_camera_new_contract_ignores_camera_id() -> None:
    """In new (ThingsBoard device-channel) mode the camera id is
    ignored and the single configured topic is used."""
    p = MqttPublisher(
        broker_host="h",
        port=1883,
        topic_base="v1/devices/me/telemetry",
        legacy_contract=False,
    )
    fake_client = mock.MagicMock()
    p._client = fake_client  # noqa: SLF001
    p.publish_for_camera("CAM_01", {"ts": 1, "values": {"x": 1}})
    args, _ = fake_client.publish.call_args
    assert args[0] == "v1/devices/me/telemetry"


def test_publish_drops_when_not_connected() -> None:
    p = MqttPublisher(
        broker_host="h",
        port=1883,
        topic_base="ai/yamaha/people-detection",
        legacy_contract=True,
    )
    # No client connect; publish should silently drop.
    p.publish_for_camera("CAM_01", {"ts": 1, "values": {}})  # no error
