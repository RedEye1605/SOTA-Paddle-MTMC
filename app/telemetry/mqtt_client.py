"""MQTT client for ThingsBoard telemetry.

Phase 5 changes:
* Token-based auth (``THINGSBOARD_DEVICE_TOKEN``) is supported
  alongside username/password. The token, when set, becomes the MQTT
  password and the publish topic becomes
  ``v1/devices/<token>/telemetry``.
* An explicit ``MQTT_HOST`` env var wins over ``MQTT_BROKER_HOST``
  (the older ``tcp://host:port`` style is still supported).
* ``MQTT_TOPIC`` overrides the per-device topic.
* Credentials are **never** logged. The constructor receives them
  via kwargs; only the broker host, port, and resolved topic name
  appear in the info log.

Phase 5b changes (legacy Service/offline-people-counting contract):
* When ``USE_LEGACY_MQTT_CONTRACT=true`` (default) AND no
  ``THINGSBOARD_DEVICE_TOKEN`` is set, the publisher falls back to
  the legacy contract:

      topic_base = "ai/yamaha/people-detection"

  per-camera publishes use :meth:`publish_for_camera`, which appends
  the legacy suffix ``/{cam1|cam2}/summary`` (and the legacy client
  id format ``people_counter_<device_name>_<ts>_<rand>``).
* The class itself is a thin wrapper around paho-mqtt. The async,
  non-blocking, periodic publisher is in
  :mod:`app.telemetry.mqtt_publisher` (Phase 5).
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _strip_scheme(value: str) -> str:
    """Strip the ``tcp://`` / ``ssl://`` / ``mqtts://`` scheme prefix."""
    for prefix in ("ssl://", "mqtts://", "tcp://", "mqtt://"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _split_host_port(value: str, default_port: int = 1883) -> tuple[str, int]:
    """Parse ``host[:port]``. Bare ``host`` → ``(host, default_port)``."""
    value = _strip_scheme(value)
    if ":" in value:
        host, port_s = value.rsplit(":", 1)
        try:
            return host, int(port_s)
        except ValueError:
            return host, default_port
    return value, default_port


def _is_truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _legacy_client_id(device_name: str | None) -> str:
    """Return the legacy ``people_counter_<name>_<ts>_<rand>`` client id."""
    name = (device_name or "unknown").strip() or "unknown"
    return f"people_counter_{name}_{int(time.time())}_{random.randint(1000, 9999)}"


def _legacy_camera_topic_id(camera_id: str, device_name: str | None = None) -> str:
    """``CAM_01`` / ``cam_1`` → ``cam1`` (legacy topic id).

    Mirrors ``Service/offline-people-counting/app/io/mqtt_topics.py::
    generate_topics``: the camera id segment of the topic is
    derived from the device_name (``cam_1`` → ``cam1``) by stripping
    the leading ``cam_`` prefix and replacing underscores with
    dashes.  When *device_name* is supplied we use it; otherwise
    we fall back to normalizing *camera_id* directly.
    """
    if device_name:
        raw = str(device_name)
        if raw.lower().startswith("cam_"):
            raw = raw[4:]
        return ("cam" + raw).replace("_", "-")
    raw = str(camera_id or "unknown")
    if raw[:4].lower() == "cam_":
        raw = raw[4:]
    elif raw[:3].upper() == "CAM" and len(raw) > 3 and raw[3] == "_":
        raw = raw[4:]
    return "cam" + raw.replace("_", "-")


def _legacy_device_name_for(camera_id: str) -> str:
    """Best-effort device_name for a camera id, used in client_id / topic."""
    cid = str(camera_id or "")
    if cid.startswith("CAM_"):
        digits = cid.removeprefix("CAM_").lstrip("0") or "0"
        return f"cam_{digits}"
    return cid


class MqttPublisher:
    def __init__(
        self,
        *,
        broker_host: str,
        port: int = 1883,
        username: str = "",
        password: str = "",
        client_id: str = "sota-paddle-mtmct",
        topic_base: str = "v1/devices/me/telemetry",
        qos: int = 1,
        tls_enabled: bool = False,
        tls_ca_cert: Optional[str] = None,
        tls_certfile: Optional[str] = None,
        tls_keyfile: Optional[str] = None,
        legacy_contract: bool = False,
    ) -> None:
        self._broker = broker_host
        self._port = port
        self._username = username
        self._password = password
        self._client_id = client_id
        self._topic_base = topic_base
        self._qos = qos
        self._tls_enabled = tls_enabled
        self._tls_ca_cert = tls_ca_cert
        self._tls_certfile = tls_certfile
        self._tls_keyfile = tls_keyfile
        self._legacy_contract = bool(legacy_contract)
        self._client = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        try:
            import paho.mqtt.client as mqtt
        except Exception as e:  # noqa: BLE001
            logger.error("paho-mqtt not installed: %s", e)
            return
        self._client = mqtt.Client(client_id=self._client_id, clean_session=True)
        if self._username or self._password:
            self._client.username_pw_set(self._username, self._password)
        if self._tls_enabled:
            tls_kwargs: dict[str, Any] = {}
            if self._tls_ca_cert:
                tls_kwargs["ca_certs"] = self._tls_ca_cert
            if self._tls_certfile:
                tls_kwargs["certfile"] = self._tls_certfile
            if self._tls_keyfile:
                tls_kwargs["keyfile"] = self._tls_keyfile
            self._client.tls_set(**tls_kwargs)
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)
        try:
            self._client.connect(self._broker, self._port, keepalive=60)
            self._client.loop_start()
            # NB: never log the password / token.
            logger.info(
                "MQTT connected: %s:%d topic_base=%s legacy_contract=%s tls=%s",
                self._broker,
                self._port,
                self._topic_base,
                self._legacy_contract,
                self._tls_enabled,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("MQTT connect failed: %s", e)

    def publish(self, payload: dict[str, Any]) -> None:
        """Publish on the default ``topic_base`` (no camera suffix)."""
        self._publish_to(self._topic_base, payload)

    def publish_for_camera(
        self,
        camera_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Publish a payload addressed to a specific camera.

        In legacy contract mode the topic becomes
        ``{topic_base}/{cam1|cam2}/summary`` (the suffix matches the
        legacy ``ai/yamaha/people-detection`` shape).  In new
        (ThingsBoard device-channel) mode the camera id is ignored
        and the default topic is used.
        """
        if not self._legacy_contract:
            self._publish_to(self._topic_base, payload)
            return
        # The legacy pipeline derives the topic camera id from the
        # ``device_name`` (e.g. ``cam_2`` → ``cam2``).  When the
        # caller passes the new-pipeline id (``CAM_02``) we first
        # try the legacy device config; if that is unavailable we
        # fall back to the simpler normalization.
        topic_id = self._resolve_legacy_topic_camera_id(camera_id)
        topic = f"{self._topic_base.rstrip('/')}/{topic_id}/summary"
        self._publish_to(topic, payload)

    def _resolve_legacy_topic_camera_id(self, camera_id: str) -> str:
        """Map a new-pipeline camera id to the legacy MQTT topic id.

        The legacy pipeline normalizes the camera id segment of the
        MQTT topic from the device_name (``cam_2`` → ``cam2``).  We
        look up the device_name via the legacy contract config so
        callers can pass either ``CAM_02`` or ``cam_2`` and the
        resulting topic is identical.
        """
        try:
            from app.integrations.legacy_contract import legacy_device_config

            dev = legacy_device_config(camera_id)
            if dev and dev.device_name:
                return _legacy_camera_topic_id(camera_id, dev.device_name)
        except Exception:  # noqa: BLE001
            pass
        return _legacy_camera_topic_id(camera_id)

    def _publish_to(self, topic: str, payload: dict[str, Any]) -> None:
        if self._client is None:
            logger.debug("MQTT client not connected; dropping payload")
            return
        body = json.dumps(payload, default=str)
        with self._lock:
            self._client.publish(topic, body, qos=self._qos)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    @property
    def broker(self) -> str:
        return self._broker

    @property
    def port(self) -> int:
        return self._port

    @property
    def topic(self) -> str:
        return self._topic_base

    @property
    def legacy_contract(self) -> bool:
        return self._legacy_contract


def _resolve_topic(
    *,
    explicit_topic: str,
    thingsboard_token: str,
    fallback: str,
) -> str:
    """Pick the publish topic.

    Order of precedence:
    1. ``MQTT_TOPIC`` (explicit_topic) if non-empty
    2. ``v1/devices/<token>/telemetry`` if THINGSBOARD_DEVICE_TOKEN
    3. ``MQTT_TOPIC_BASE`` (fallback) — default ``v1/devices/me/telemetry``
    """
    if explicit_topic:
        return explicit_topic
    if thingsboard_token:
        return f"v1/devices/{thingsboard_token}/telemetry"
    return fallback


def from_env() -> Optional["MqttPublisher"]:
    """Build an :class:`MqttPublisher` from env vars.

    Returns ``None`` if ``MQTT_ENABLED=false`` or if the broker
    configuration is missing.
    """
    if os.environ.get("MQTT_ENABLED", "true").lower() not in {"1", "true", "yes", "on"}:
        return None
    explicit_host = os.environ.get("MQTT_HOST", "").strip()
    legacy_broker = os.environ.get("MQTT_BROKER_HOST", "").strip()
    chosen = explicit_host or legacy_broker
    if not chosen:
        logger.warning(
            "MQTT enabled but neither MQTT_HOST nor MQTT_BROKER_HOST is set; telemetry is disabled"
        )
        return None
    host, port = _split_host_port(chosen, default_port=1883)
    # Optional explicit port override.
    port_env = os.environ.get("MQTT_PORT", "").strip()
    if port_env:
        try:
            port = int(port_env)
        except ValueError:
            pass
    token = os.environ.get("THINGSBOARD_DEVICE_TOKEN", "").strip()
    username = os.environ.get("MQTT_USERNAME", "").strip()
    password = os.environ.get("MQTT_PASSWORD", "").strip()
    # If a ThingsBoard token is set, the broker authenticates via
    # the token (used as the password) and the topic must be the
    # device-specific one. ``username`` is empty in token mode.
    if token:
        username = ""
        password = token
    # Legacy contract is on by default (matches the legacy Service
    # pipeline). Setting THINGSBOARD_DEVICE_TOKEN automatically
    # disables it so the v1/devices/<token>/telemetry path works.
    legacy = not token and _is_truthy(os.environ.get("USE_LEGACY_MQTT_CONTRACT"), default=True)
    topic = _resolve_topic(
        explicit_topic=os.environ.get("MQTT_TOPIC", "").strip(),
        thingsboard_token=token,
        fallback=os.environ.get("MQTT_TOPIC_BASE", "v1/devices/me/telemetry").strip()
        or "v1/devices/me/telemetry",
    )
    # When using the legacy contract, the topic is the *base* — the
    # per-camera suffix is appended at publish time via
    # :meth:`MqttPublisher.publish_for_camera`.
    if legacy and not os.environ.get("MQTT_TOPIC", "").strip():
        topic = (
            os.environ.get("MQTT_TOPIC_BASE", "ai/yamaha/people-detection").strip()
            or "ai/yamaha/people-detection"
        )
    client_id = os.environ.get("MQTT_CLIENT_ID", "sota-paddle-mtmct").strip() or "sota-paddle-mtmct"
    return MqttPublisher(
        broker_host=host,
        port=port,
        username=username,
        password=password,
        client_id=client_id,
        topic_base=topic,
        qos=int(os.environ.get("MQTT_QOS", "1")),
        tls_enabled=os.environ.get("MQTT_TLS_ENABLED", "false").lower() in {"1", "true", "yes"},
        tls_ca_cert=os.environ.get("MQTT_TLS_CA_CERT") or None,
        tls_certfile=os.environ.get("MQTT_TLS_CERTFILE") or None,
        tls_keyfile=os.environ.get("MQTT_TLS_KEYFILE") or None,
        legacy_contract=legacy,
    )
