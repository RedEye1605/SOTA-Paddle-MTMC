"""Dump legacy Service payload + topic + minio key to JSON on stdout."""
import json
import os
import sys
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone

os.chdir('/home/rhendy/Projects/yamaha/Service/offline-people-counting')
sys.path.insert(0, '/home/rhendy/Projects/yamaha/Service/offline-people-counting')

from app.counting.aggregator import _ZoneStats  # type: ignore
from app.counting.payload import PayloadBuilder  # type: ignore
from app.io.mqtt_topics import generate_topics  # type: ignore
from app.io.minio_uploader import build_object_name  # type: ignore


def make_stats(zones, counts):
    out = {}
    for z in zones:
        s = _ZoneStats()
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

pb = PayloadBuilder()
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

t1 = generate_topics({"device_name": "cam_1", "camera_id": "cam_1"}, "ai/yamaha/people-detection")["telemetry"]
t2 = generate_topics({"device_name": "cam_2", "camera_id": "cam_2"}, "ai/yamaha/people-detection")["telemetry"]

minio_key = build_object_name(
    object_prefix="people-detection",
    camera_id="cam1",
    zone="Active Zone",
    location_title="main_hallway",
    date_format="%Y-%m-%d",
    person_id=42,
    timestamp=datetime.fromtimestamp(1749926400.0, tz=timezone.utc),
)

out = {
    "cam1_payload": cam1,
    "cam2_payload": cam2,
    "cam1_topic": t1,
    "cam2_topic": t2,
    "minio_key": minio_key,
}
print(json.dumps(out))
