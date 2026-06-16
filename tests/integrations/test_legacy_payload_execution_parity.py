"""Cross-pipeline execution parity test (D-leg).

Closes the loop beyond A/B/C:

    A. new code vs hardcoded expectations   (not what we want)
    B. new code vs YAML                     (test_legacy_contract.py)
    C. YAML vs parsed legacy source         (test_legacy_yaml_vs_legacy_source.py)
    D. *running* the new code vs *running* the legacy code on the
       same inputs and diffing the resulting MQTT payloads
                                                       (this file)

Strategy
--------
The SOTA project and the legacy Service project both contain a top-level
``app/`` package. They cannot be imported into the same Python process
without conflicts, so this test shells out to two independent subprocess
runs (one per venv) to dump their payloads + topics + MinIO keys to a
temp directory, then diffs the resulting JSON files.

If ``Service/offline-people-counting`` is missing (e.g. CI that checked
out only SOTA), the tests are skipped — the contract is still pinned
by the A/B/C legs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

SERVICE_ROOT = Path("/home/rhendy/Projects/yamaha/Service/offline-people-counting")
SOTA_ROOT = Path("/home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC")
HAS_SERVICE = (
    SERVICE_ROOT.is_dir()
    and (SERVICE_ROOT / "app" / "counting" / "payload.py").is_file()
    and (SERVICE_ROOT / ".venv" / "bin" / "python").is_file()
)
HAS_SOTA = (
    SOTA_ROOT.is_dir()
    and (SOTA_ROOT / ".venv" / "bin" / "python").is_file()
    and (SOTA_ROOT / "app" / "integrations" / "legacy_payload.py").is_file()
)

pytestmark = pytest.mark.skipif(
    not (HAS_SERVICE and HAS_SOTA),
    reason="Need both Service/offline-people-counting and SOTA-Paddle-MTMC venvs in workspace",
)


SCRIPTS = Path(__file__).resolve().parent / "_parity_assets"
SERVICE_VENV_PY = SERVICE_ROOT / ".venv" / "bin" / "python"
SOTA_VENV_PY = SOTA_ROOT / ".venv" / "bin" / "python"
SERVICE_DUMP = SCRIPTS / "service_dump.py"
SOTA_DUMP = SCRIPTS / "sota_dump.py"


def _ensure_assets():
    """Create the dump scripts once on first test run."""
    SERVICE_DUMP.parent.mkdir(parents=True, exist_ok=True)
    if not SERVICE_DUMP.exists():
        SERVICE_DUMP.write_text(_SERVICE_DUMP_SOURCE)
    if not SOTA_DUMP.exists():
        SOTA_DUMP.write_text(_SOTA_DUMP_SOURCE)


def _run_dump(py: Path, script: Path, cwd: Path) -> dict:
    proc = subprocess.run(
        [str(py), str(script)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"dump script failed (rc={proc.returncode}):\n"
            f"  stdout: {proc.stdout[:500]}\n"
            f"  stderr: {proc.stderr[:500]}"
        )
    return json.loads(proc.stdout)


def _diff(label: str, a, b) -> int:
    if a == b:
        return 0
    print(f"  DIFF {label}: legacy={a!r}  new={b!r}")
    return 1


def _diff_payloads(label: str, a: dict, b: dict) -> int:
    if a == b:
        return 0
    print(f"  DIFF {label}:")
    keys_a, keys_b = set(a.keys()), set(b.keys())
    n = 0
    for k in sorted(keys_a & keys_b):
        if a[k] != b[k]:
            print(f"    {k}: legacy={a[k]!r}  new={b[k]!r}")
            n += 1
    if keys_a - keys_b:
        print(f"    ONLY_LEGACY: {sorted(keys_a - keys_b)}")
        n += len(keys_a - keys_b)
    if keys_b - keys_a:
        print(f"    ONLY_NEW: {sorted(keys_b - keys_a)}")
        n += len(keys_b - keys_a)
    return n


@pytest.fixture(scope="module")
def parity_dump() -> dict:
    """Run both dump scripts once and return the combined result."""
    _ensure_assets()
    legacy = _run_dump(SERVICE_VENV_PY, SERVICE_DUMP, SERVICE_ROOT)
    new = _run_dump(SOTA_VENV_PY, SOTA_DUMP, SOTA_ROOT)
    return {"legacy": legacy, "new": new}


def test_cam01_payload_byte_equal_to_legacy(parity_dump):
    legacy, new = parity_dump["legacy"], parity_dump["new"]
    n_diffs = _diff_payloads(
        "CAM_01 payload", legacy["cam1_payload"]["values"], new["cam1_payload"]["values"]
    )
    assert n_diffs == 0, f"CAM_01 payload has {n_diffs} field differences"
    assert legacy["cam1_payload"]["ts"] == new["cam1_payload"]["ts"]


def test_cam02_payload_byte_equal_to_legacy(parity_dump):
    legacy, new = parity_dump["legacy"], parity_dump["new"]
    n_diffs = _diff_payloads(
        "CAM_02 payload", legacy["cam2_payload"]["values"], new["cam2_payload"]["values"]
    )
    assert n_diffs == 0, f"CAM_02 payload has {n_diffs} field differences"
    assert legacy["cam2_payload"]["ts"] == new["cam2_payload"]["ts"]


def test_mqtt_topics_match(parity_dump):
    legacy, new = parity_dump["legacy"], parity_dump["new"]
    assert legacy["cam1_topic"] == new["cam1_topic"] == "ai/yamaha/people-detection/cam1/summary"
    assert legacy["cam2_topic"] == new["cam2_topic"] == "ai/yamaha/people-detection/cam2/summary"


def test_minio_object_keys_match(parity_dump):
    legacy, new = parity_dump["legacy"], parity_dump["new"]
    assert legacy["minio_key"] == new["minio_key"]


# ---------------------------------------------------------------------------
# Dump scripts — run in subprocesses with the project's own venvs so the
# two `app/` packages never collide in the same process.
# ---------------------------------------------------------------------------

_SERVICE_DUMP_SOURCE = '''\
"""Dump legacy Service payload + topic + minio key to JSON on stdout."""
import json
import os
import sys
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone

os.chdir({SERVICE_ROOT!r})
sys.path.insert(0, {SERVICE_ROOT!r})

from app.counting.aggregator import _ZoneStats  # type: ignore
from app.counting.payload import PayloadBuilder  # type: ignore
from app.io.mqtt_topics import generate_topics  # type: ignore
from app.io.minio_uploader import build_object_name  # type: ignore


def make_stats(zones, counts):
    out = {{}}
    for z in zones:
        s = _ZoneStats()
        s.current_count = counts.get(z, 0)
        s.total_entered = 100
        s.total_exited = 80
        s.dwell_times = deque([10.0, 20.0, 30.0, 25.0], maxlen=1000)
        s.valid_entries = deque(
            [
                {{"dwell_time": 10.0, "person_id": 1}},
                {{"dwell_time": 20.0, "person_id": 2}},
                {{"dwell_time": 30.0, "person_id": 3}},
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
            [{{"dwell_time": 10.0}}, {{"dwell_time": 20.0}}], maxlen=1000
        )
        s.valid_entries_daily = deque(
            [
                {{"dwell_time": 10.0}},
                {{"dwell_time": 20.0}},
                {{"dwell_time": 30.0}},
            ],
            maxlen=1000,
        )
        out[z] = s
    return out


CAM_01 = ["Fazzio & Filano Zone", "Active Zone", "Dealing Zone 1", "Island Zone"]
CAM_01_C = {{"Fazzio & Filano Zone": 2, "Active Zone": 3, "Dealing Zone 1": 1, "Island Zone": 1}}
CAM_02 = ["Sport Zone", "Premium Zone", "Dealing Zone 2", "Island Zone"]
CAM_02_C = {{"Sport Zone": 4, "Premium Zone": 2, "Dealing Zone 2": 1, "Island Zone": 0}}
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

t1 = generate_topics({{"device_name": "cam_1", "camera_id": "cam_1"}}, "ai/yamaha/people-detection")["telemetry"]
t2 = generate_topics({{"device_name": "cam_2", "camera_id": "cam_2"}}, "ai/yamaha/people-detection")["telemetry"]

minio_key = build_object_name(
    object_prefix="people-detection",
    camera_id="cam1",
    zone="Active Zone",
    location_title="main_hallway",
    date_format="%Y-%m-%d",
    person_id=42,
    timestamp=datetime.fromtimestamp(1749926400.0, tz=timezone.utc),
)

out = {{
    "cam1_payload": cam1,
    "cam2_payload": cam2,
    "cam1_topic": t1,
    "cam2_topic": t2,
    "minio_key": minio_key,
}}
print(json.dumps(out))
'''.format(SERVICE_ROOT=str(SERVICE_ROOT))


_SOTA_DUMP_SOURCE = '''\
"""Dump new SOTA payload + topic + minio key to JSON on stdout."""
import json
import os
import sys
from collections import deque
from copy import deepcopy
from types import SimpleNamespace

os.chdir({SOTA_ROOT!r})
sys.path.insert(0, {SOTA_ROOT!r})

from app.integrations.legacy_contract import legacy_camera_topic, legacy_evidence_key  # type: ignore
from app.integrations.legacy_payload import LegacyPayloadBuilder  # type: ignore


def make_stats(zones, counts):
    out = {{}}
    for z in zones:
        s = SimpleNamespace()
        s.current_count = counts.get(z, 0)
        s.total_entered = 100
        s.total_exited = 80
        s.dwell_times = deque([10.0, 20.0, 30.0, 25.0], maxlen=1000)
        s.valid_entries = deque(
            [
                {{"dwell_time": 10.0, "person_id": 1}},
                {{"dwell_time": 20.0, "person_id": 2}},
                {{"dwell_time": 30.0, "person_id": 3}},
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
            [{{"dwell_time": 10.0}}, {{"dwell_time": 20.0}}], maxlen=1000
        )
        s.valid_entries_daily = deque(
            [
                {{"dwell_time": 10.0}},
                {{"dwell_time": 20.0}},
                {{"dwell_time": 30.0}},
            ],
            maxlen=1000,
        )
        out[z] = s
    return out


CAM_01 = ["Fazzio & Filano Zone", "Active Zone", "Dealing Zone 1", "Island Zone"]
CAM_01_C = {{"Fazzio & Filano Zone": 2, "Active Zone": 3, "Dealing Zone 1": 1, "Island Zone": 1}}
CAM_02 = ["Sport Zone", "Premium Zone", "Dealing Zone 2", "Island Zone"]
CAM_02_C = {{"Sport Zone": 4, "Premium Zone": 2, "Dealing Zone 2": 1, "Island Zone": 0}}
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

out = {{
    "cam1_payload": cam1,
    "cam2_payload": cam2,
    "cam1_topic": t1,
    "cam2_topic": t2,
    "minio_key": minio_key,
}}
print(json.dumps(out))
'''.format(SOTA_ROOT=str(SOTA_ROOT))
