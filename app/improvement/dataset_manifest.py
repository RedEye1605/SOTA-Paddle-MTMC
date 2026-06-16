"""Benchmark dataset manifest.

A frozen JSON manifest that lists the recorded clips used for offline
evaluation. The manifest's ``sha256`` is the audit trail: the
benchmark is reproducible against this exact set, no drift.

Format:

    {
        "name": "showroom_a_2026-05-12",
        "version": "1",
        "created_at": "2026-05-12T08:30:00Z",
        "cameras": [
            {"camera_id": "CAM_01", "video": "s3://benchmark/cam01.mp4",
             "start_ts": 1715520000, "end_ts": 1715550000,
             "sha256": "..."},
            ...
        ],
        "labels": "s3://benchmark/labels.json",
        "site_id": "showroom_a"
    }
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class CameraClip:
    camera_id: str
    video: str  # local path or s3 URI
    start_ts: float
    end_ts: float
    sha256: str = ""  # computed at construction time

    def compute_sha256(self, path: Optional[Path] = None) -> str:
        if not self.video.startswith("s3://") and path is None:
            path = Path(self.video)
        if path is None or not path.exists():
            return self.sha256
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        self.sha256 = h.hexdigest()
        return self.sha256


@dataclass
class DatasetManifest:
    name: str
    version: str
    created_at: str
    cameras: list[CameraClip] = field(default_factory=list)
    labels: str = ""  # path / s3 URI to ground-truth labels
    site_id: str = "default_site"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write(self, path: Path) -> None:
        path.write_text(self.to_json())

    @classmethod
    def from_json(cls, raw: str) -> "DatasetManifest":
        d = json.loads(raw)
        cams = [CameraClip(**c) for c in d.pop("cameras", [])]
        return cls(cameras=cams, **d)
