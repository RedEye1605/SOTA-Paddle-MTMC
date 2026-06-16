"""Dump new SOTA payload + topic + minio key to JSON on stdout."""
import json
import os
import sys
from collections import deque
from copy import deepcopy
from types import SimpleNamespace

os.chdir('/home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC')
sys.path.insert(0, '/home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC')

from app.integrations.legacy_contract import legacy_camera_topic, legacy_evidence_key  # type: ignore
from app.integrations.legacy_payload import LegacyPayloadBuilder  # type: ignore


def make_stats(zones, counts):
    out = {}
    for z in zones:
        s = SimpleNamespace()
        s.current_count = counts.get(z, 0)
        s.total_entered = 100
        s.total_exited = 80
        s.dwell_times = deque([10.0, 20.0, 30.0, 25.0], maxlen=1000)
        s.valid_entries = deque(
            [
                {"dwell_time": 10.0, "person_id": 1},
                {"dwell_time": 20.0, "person_id": 2},
                {"dwell_time": 30.0, "person_id": 3},
            ],
            maxlen=1000,
        )
        s.pending_entries = deque(maxlen=1000)
        s.total_entered_hourly = 20
        s.total_entered_daily = 40
        s.total_exited_hourly = 15
        s.total_exited_daily = 30
        s.dwell_times_hourly = deque([10.0, 20.0], maxlen=1000)
        s.dwell_times_daily = deque([10.0, 20.0, 30.0, 25.0], maxlen=1000)
        s.valid_entries_hourly = deque(
            [{"dwell_time": 10.0}, {"dwell_time": 20.0}], maxlen=1000
        )
        s.valid_entries_daily = deque(
            [
                {"dwell_time": 10.0},
                {"dwell_time": 20.0},
                {"dwell_time": 30.0},
            ],
            maxlen=1000,
        )
        out[z] = s
    return out


CAM_01 = ["Fazzio & Filano Zone", "Active Zone", "Dealing Zone 1", "Island Zone"]
CAM_01_C = {"Fazzio & Filano Zone": 2, "Active Zone": 3, "Dealing Zone 1": 1, "Island Zone": 1}
CAM_02 = ["Sport Zone", "Premium Zone", "Dealing Zone 2", "Island Zone"]
CAM_02_C = {"Sport Zone": 4, "Premium Zone": 2, "Dealing Zone 2": 1, "Island Zone": 0}
DWELL = [10.0, 20.0, 30.0, 25.0, 50.0, 60.0]
UNIQUE = (7, 3, 1)
FRAMES = 180
TS = 1700000000.0

pb = LegacyPayloadBuilder()
cam1 = pb.build(
    zone_stats=deepcopy(make_stats(CAM_01, CAM_01_C)),
    unique_counts=UNIQUE,
    frame_count=FRAMES,
    camera_dwell_times=list(DWELL),
    timestamp=TS,
)
pb.reset()
cam2 = pb.build(
    zone_stats=deepcopy(make_stats(CAM_02, CAM_02_C)),
    unique_counts=UNIQUE,
    frame_count=FRAMES,
    camera_dwell_times=list(DWELL),
    timestamp=TS,
)

t1 = legacy_camera_topic("telemetry", "CAM_01", "cam_1")
t2 = legacy_camera_topic("telemetry", "CAM_02", "cam_2")

minio_key = legacy_evidence_key(
    camera_id="cam1",
    zone="Active Zone",
    person_id=42,
    timestamp_epoch=1749926400.0,
)

out = {
    "cam1_payload": cam1,
    "cam2_payload": cam2,
    "cam1_topic": t1,
    "cam2_topic": t2,
    "minio_key": minio_key,
}
print(json.dumps(out))
