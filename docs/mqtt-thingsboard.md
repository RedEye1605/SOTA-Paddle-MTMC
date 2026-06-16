# MQTT / ThingsBoard setup

SOTA-Paddle-MTMC publishes ThingsBoard-compatible analytics to the
operator's external MQTT broker.

## 1. Topics

Two topic modes are supported:

1. **Token-based** (default in production):
   - `v1/devices/<THINGSBOARD_DEVICE_TOKEN>/telemetry`
2. **Username-based** (fallback):
   - `MQTT_TOPIC`, or `MQTT_TOPIC_BASE` (`v1/devices/me/telemetry` by default)

If `THINGSBOARD_DEVICE_TOKEN` is set, the publisher uses it both as
the password and as part of the topic. Otherwise, the publisher uses
`MQTT_USERNAME` / `MQTT_PASSWORD` and the topic is `MQTT_TOPIC` (or
`MQTT_TOPIC_BASE`).

## 2. Payload shape

ThingsBoard telemetry RPC requires the `{ts, values}` envelope.

### 2.1. Per-camera summary

```json
{
  "ts": 1710000000000,
  "values": {
    "cam_id": "CAM_01",
    "zone_id": "ZONE_A",
    "people_count": 3,
    "entries": 1,
    "exits": 0,
    "dwell_avg_seconds": 42.5,
    "active_global_ids": 3
  }
}
```

### 2.2. Identity decision

```json
{
  "ts": 1710000000000,
  "values": {
    "global_id_active": 1,
    "global_id": "GID_CAM01_0001",
    "site_id": "yamaha_showroom",
    "camera_id": "CAM_01"
  }
}
```

### 2.3. Zone event

```json
{
  "ts": 1710000000000,
  "values": {
    "zone_event": "enter",
    "zone_id": "ZONE_A",
    "camera_id": "CAM_01",
    "global_id": "GID_CAM01_0001"
  }
}
```

### 2.4. Dwell

```json
{
  "ts": 1710000000000,
  "values": {
    "dwell_duration_seconds": 42,
    "zone_id": "ZONE_A",
    "camera_id": "CAM_01",
    "global_id": "GID_CAM01_0001"
  }
}
```

## 3. Auth

| Method | Env vars |
| --- | --- |
| Username / password | `MQTT_USERNAME`, `MQTT_PASSWORD` |
| ThingsBoard token | `THINGSBOARD_DEVICE_TOKEN` (used as password) |
| TLS | `MQTT_TLS_ENABLED=true`, `MQTT_TLS_CA_CERT`/`MQTT_TLS_CERTFILE`/`MQTT_TLS_KEYFILE` |

Passwords and tokens are **never logged**. The publisher only ever
logs the broker host, port, and topic name.

## 4. Disable / fail-fast

- `MQTT_ENABLED=false` (env) → no client is constructed; the
  telemetry worker is a no-op.
- If MQTT is enabled but the broker is unreachable, the
  `MqttPublisher` keeps trying with exponential backoff (min 1 s,
  max 30 s). The publisher is **non-blocking** — telemetry
  publishes are dropped with a warning when the queue is full.
- Production preflight refuses `READY_FOR_LIMITED_PRODUCTION` if
  telemetry is required (e.g. shadow test) but the broker is
  unreachable.

## 5. Smoke-test isolation

Smoke tests must not write authoritative telemetry. The smoke
worker uses `RUNTIME_MODE=SMOKE_TEST`; the publisher detects this
and short-circuits — see `app/telemetry/mqtt_publisher.py` for the
gate. The gate is enforced by the test
`test_telemetry_disabled_in_smoke_mode`.

## 6. Quick start

```bash
# 1. Verify env
grep -E "^(MQTT_BROKER_HOST|MQTT_HOST|MQTT_USERNAME|MQTT_PASSWORD|THINGSBOARD_DEVICE_TOKEN|MQTT_TOPIC)=" .env

# 2. Run a one-off publish to verify connectivity
uv run python - <<'PY'
import os, time, json
os.environ.setdefault("MQTT_HOST", "mqtt.example.invalid")
os.environ.setdefault("MQTT_USERNAME", "<MQTT_CREDENTIAL>")
os.environ.setdefault("MQTT_PASSWORD", "<MQTT_CREDENTIAL>")
from app.telemetry.mqtt_client import MqttPublisher
p = MqttPublisher.from_env()
if p is None:
    raise SystemExit("MQTT disabled by env")
p.connect()
time.sleep(1.0)
p.publish({"ts": int(time.time() * 1000),
           "values": {"cam_id": "CAM_01", "people_count": 0}})
p.close()
PY
```

## 7. References

- `app/telemetry/mqtt_client.py` — paho-mqtt v2 client.
- `app/telemetry/mqtt_publisher.py` — async, non-blocking publisher.
- `app/telemetry/thingsboard_payload.py` — payload builders.
- `app/workers/telemetry_worker.py` — stream consumer that wires
  resolver decisions and zone events to the publisher.
- `Docs/external_services_setup.md` — cross-service overview.
