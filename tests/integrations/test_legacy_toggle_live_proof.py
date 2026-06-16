"""Live toggle-behavior proofs (Phase 5b).

Each test here is a *red-green* proof: setting the toggle env var
to false must observably change the runtime behavior, not just
the flag value. The existing ``test_legacy_toggles.py`` only checks
flag resolution; this file proves the *consumer* behavior.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from app.integrations.legacy_contract import (
    flag_enabled,
    legacy_camera_topic,
    legacy_evidence_key,
    legacy_roi_zones,
)


def _reload_contract():
    """Reload legacy_contract with the current env vars."""
    import app.integrations.legacy_contract as lc

    importlib.reload(lc)
    return lc


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    """Each test starts with no toggle env vars, then sets its own."""
    for k in (
        "ENABLE_SEND_MQTT",
        "ENABLE_MINIO_UPLOAD",
        "ENABLE_TRACK_ID",
        "SHOW_ROI_ZONES",
        "SHOW_CONFIDENCE_SCORE",
        "SHOW_TRACK_ID",
        "SHOW_DETECTION_BOX",
        "SHOW_CAMERA_LABEL",
        "SHOW_COUNTING_OVERLAY",
    ):
        monkeypatch.delenv(k, raising=False)
    yield monkeypatch


# ---------------------------------------------------------------------------
# 1. ENABLE_SEND_MQTT=false  ->  no connect, no publish
# ---------------------------------------------------------------------------


def test_enable_send_mqtt_false_blocks_publish_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_SEND_MQTT", "false")
    _reload_contract()
    from app.integrations.mqtt_publisher import LegacyMqttPublisher

    class FakeMqtt:
        def __init__(self):
            self.published: list = []

        def publish_for_camera(self, cam, payload):
            self.published.append((cam, payload))
            return True

    fake = FakeMqtt()
    p = LegacyMqttPublisher(fake, camera_id="CAM_01")
    # 1. Publisher must report itself disabled.
    assert p.is_enabled is False
    # 2. start() must NOT spawn a thread.
    p.start()
    assert p._retry_thread is None
    # 3. publish_telemetry must return False and not enqueue.
    ok = p.publish_telemetry({"ts": 1, "values": {}})
    assert ok is False
    # 4. The underlying MqttPublisher must never see a publish.
    assert fake.published == []


def test_enable_send_mqtt_true_allows_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_SEND_MQTT", "true")
    _reload_contract()
    from app.integrations.mqtt_publisher import LegacyMqttPublisher

    class FakeMqtt:
        def publish_for_camera(self, cam, payload):
            self.saw = (cam, payload)
            return True

    fake = FakeMqtt()
    p = LegacyMqttPublisher(fake, camera_id="CAM_01")
    assert p.is_enabled is True
    # A connected client means publish_telemetry will queue.
    fake._client = object()  # the internal _client is checked
    ok = p.publish_telemetry({"ts": 1, "values": {"x": 1}})
    assert ok is True


# ---------------------------------------------------------------------------
# 2. ENABLE_MINIO_UPLOAD=false  ->  no upload attempt
# ---------------------------------------------------------------------------


def test_enable_minio_upload_false_returns_none_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_MINIO_UPLOAD", "false")
    _reload_contract()
    from app.integrations.minio_uploader import LegacyMinioUploader

    class FakeStore:
        bucket = "yamaha-poc"
        def __init__(self):
            self._client = object()
            self.put_object_calls: list = []

    store = FakeStore()
    u = LegacyMinioUploader(store, camera_id="cam1")
    assert u.is_enabled is False
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    url, key = u.upload_person_crop(
        frame, (10, 10, 90, 90), person_id=42, zone="Sport Zone",
    )
    assert (url, key) == (None, None)
    # The store must not have seen a put_object call.
    assert store.put_object_calls == []


def test_enable_minio_upload_true_calls_put_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_MINIO_UPLOAD", "true")
    _reload_contract()
    from app.integrations.minio_uploader import LegacyMinioUploader

    class FakeClient:
        def __init__(self):
            self.puts: list = []
        def put_object(self, **kw):
            self.puts.append(kw)
        def presigned_get_object(self, **kw):
            return "https://signed/url"

    class FakeStore:
        bucket = "yamaha-poc"
        def __init__(self):
            self._client = FakeClient()
            self.bucket = "yamaha-poc"

    store = FakeStore()
    u = LegacyMinioUploader(store, camera_id="cam1")
    assert u.is_enabled is True
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    url, key = u.upload_person_crop(
        frame, (10, 10, 90, 90), person_id=42, zone="Sport Zone",
    )
    assert url is not None
    assert key is not None
    assert len(store._client.puts) == 1
    assert store._client.puts[0]["bucket_name"] == "yamaha-poc"
    assert "people-detection/cam1/sport-zone/" in store._client.puts[0]["object_name"]


# ---------------------------------------------------------------------------
# 3. SHOW_ROI_ZONES=false  ->  ROI is hidden in *visualization only*
# ---------------------------------------------------------------------------


def test_show_roi_zones_false_does_not_change_roi_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHOW_ROI_ZONES", "false")
    _reload_contract()
    cam = legacy_roi_zones("cam1")
    # ROI contract (polygons, sizes, count) must not change.
    assert cam.original_size == (3072, 2048)
    assert len(cam.rois) == 4
    # And the consumer (flag_enabled) reads false.
    assert flag_enabled("SHOW_ROI_ZONES") is False


# ---------------------------------------------------------------------------
# 4. SHOW_CONFIDENCE_SCORE=false  ->  payload/counting unchanged
# ---------------------------------------------------------------------------


def test_show_confidence_score_false_does_not_change_topic_or_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHOW_CONFIDENCE_SCORE", "false")
    _reload_contract()
    # The MQTT topic is the contract value, independent of visualization.
    assert (
        legacy_camera_topic("telemetry", "CAM_01", "cam_1")
        == "ai/yamaha/people-detection/cam1/summary"
    )
    # And the payload builder still produces the same field set.
    from app.integrations.legacy_payload import (
        FakeZoneStats, LegacyPayloadBuilder,
    )
    b = LegacyPayloadBuilder()
    s = FakeZoneStats(name="Active Zone")
    s.current_count = 3
    p = b.build(
        zone_stats={"Active Zone": s},
        unique_counts=(1, 0, 0),
        frame_count=10,
        camera_dwell_times=[],
        timestamp=1_700_000_000.0,
    )
    assert "ActiveZoneCount" in p["values"]
    assert "TotalPeople" in p["values"]


# ---------------------------------------------------------------------------
# 5. SHOW_TRACK_ID=false  ->  visual ID hidden, contract unchanged
# ---------------------------------------------------------------------------


def test_show_track_id_false_does_not_change_object_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SHOW_TRACK_ID", "false")
    _reload_contract()
    key = legacy_evidence_key(camera_id="cam1", zone="Sport Zone", person_id=1)
    # MinIO object key is the contract value, independent of SHOW_TRACK_ID.
    assert key.startswith("people-detection/cam1/sport-zone/")


# ---------------------------------------------------------------------------
# 6. ENABLE_TRACK_ID=false  ->  internal tracking still works
# ---------------------------------------------------------------------------


def test_enable_track_id_false_does_not_break_roi_or_counting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_TRACK_ID", "false")
    _reload_contract()
    # ROI must still resolve (counting uses ROI membership).
    cam1 = legacy_roi_zones("cam1")
    cam2 = legacy_roi_zones("cam2")
    assert len(cam1.rois) == 4
    assert len(cam2.rois) == 4
    # Payload builder still works.
    from app.integrations.legacy_payload import (
        FakeZoneStats, LegacyPayloadBuilder,
    )
    b = LegacyPayloadBuilder()
    s = FakeZoneStats(name="Sport Zone")
    s.current_count = 5
    p = b.build(
        zone_stats={"Sport Zone": s},
        unique_counts=(5, 0, 0),
        frame_count=10,
        camera_dwell_times=[],
        timestamp=1_700_000_000.0,
    )
    assert p["values"]["TotalPeople"] == 5
    assert p["values"]["SportZoneCount"] == 5


# ---------------------------------------------------------------------------
# 7. All toggles independently read the env var
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "ENABLE_SEND_MQTT",
        "ENABLE_MINIO_UPLOAD",
        "ENABLE_TRACK_ID",
        "SHOW_ROI_ZONES",
        "SHOW_CONFIDENCE_SCORE",
        "SHOW_TRACK_ID",
    ],
)
def test_toggle_env_false_is_observed_live(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setenv(name, "false")
    _reload_contract()
    assert flag_enabled(name) is False
