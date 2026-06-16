"""DetectionUnionMerger.

Merges raw PP-Human detections with raw SAHI detections via
class-agnostic NMS. Reads SAHI detections non-blockingly (GET on
``sahi:latest:{camera_id}`` + XREVRANGE COUNT 1 on
``stream:detections_sahi``). The pre-MOT path never uses the
blocking XREAD-with-0-ms-timeout pattern.

Joins on ``timestamp_ms`` (primary) with a configurable slack window;
falls back to ``frame_id`` if the SAHI event has no timestamp.

Called by the vendored PP-Human pipeline subprocess immediately
before OC-SORT.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UnionConfig:
    """All union-merger tunables."""

    union_nms_iou: float = 0.5
    union_min_conf: float = 0.4
    stale_ms: int = 400
    timestamp_slack_ms: int = 50
    stream: str = "stream:detections_sahi"
    latest_key_prefix: str = "sahi:latest:"

    @classmethod
    def from_env(cls) -> "UnionConfig":
        return cls(
            union_nms_iou=float(os.environ.get("SAHI_UNION_NMS_IOU", "0.5")),
            union_min_conf=float(os.environ.get("SAHI_UNION_MIN_CONF", "0.4")),
            stale_ms=int(os.environ.get("SAHI_STALE_MS", "400")),
            timestamp_slack_ms=int(
                os.environ.get("SAHI_TIMESTAMP_SLACK_MS", "50")
            ),
            stream=os.environ.get("SAHI_STREAM", "stream:detections_sahi"),
            latest_key_prefix=os.environ.get(
                "SAHI_LATEST_KEY_PREFIX", "sahi:latest:"
            ),
        )


@dataclass(frozen=True)
class MergedDetection:
    """One detection in the union output, tagged by source."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source: str  # "pphuman" | "sahi" | "merged"

    def to_dict(self) -> dict:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "score": self.score,
            "source": self.source,
        }


class DetectionUnionMerger:
    """Non-blocking merger of PP-Human + SAHI raw detections."""

    def __init__(
        self,
        *,
        redis: Any,
        config: UnionConfig,
        camera_id: str,
    ) -> None:
        self._redis = redis
        self._config = config
        self._camera_id = camera_id

    def merge(
        self,
        pphuman_dets: Sequence[dict],
        *,
        current_timestamp_ms: int,
        current_frame_id: int,
    ) -> List[MergedDetection]:
        """Merge PP-Human + SAHI bboxes via class-agnostic NMS.

        Returns tagged ``MergedDetection`` list. Source tag is
        ``"merged"`` if the surviving bbox was the higher-score member
        of a (pphuman, sahi) pair that NMS collapsed.
        """
        try:
            sahi_event = self._read_sahi_event()
        except Exception as e:  # noqa: BLE001
            logger.warning("DetectionUnionMerger redis read error: %s", e)
            return self._tag_pphuman(pphuman_dets)

        if sahi_event is None:
            return self._tag_pphuman(pphuman_dets)

        # Filter by camera_id and timestamp.
        if sahi_event.get("camera_id") and sahi_event["camera_id"] != self._camera_id:
            return self._tag_pphuman(pphuman_dets)
        ts_ms = sahi_event.get("timestamp_ms")
        if ts_ms is None:
            # Fallback: frame_id match within ±1.
            if abs(int(sahi_event.get("frame_id", -1)) - current_frame_id) > 1:
                return self._tag_pphuman(pphuman_dets)
        else:
            ts_ms = int(ts_ms)
            if abs(ts_ms - current_timestamp_ms) > self._config.timestamp_slack_ms:
                return self._tag_pphuman(pphuman_dets)
            if (current_timestamp_ms - ts_ms) > self._config.stale_ms:
                return self._tag_pphuman(pphuman_dets)

        # Parse SAHI bboxes.
        raw_dets = sahi_event.get("detections", [])
        if isinstance(raw_dets, str):
            try:
                raw_dets = json.loads(raw_dets)
            except Exception:  # noqa: BLE001
                raw_dets = []
        sahi_dets = [
            d for d in (raw_dets or [])
            if len(d) >= 5 and float(d[4]) >= self._config.union_min_conf
        ]

        if not sahi_dets:
            return self._tag_pphuman(pphuman_dets)

        # NMS over the union.
        pphuman_norm = [
            (
                float(d["x1"]),
                float(d["y1"]),
                float(d["x2"]),
                float(d["y2"]),
                float(d.get("score", 1.0)),
            )
            for d in pphuman_dets
        ]
        all_dets = [(b[0], b[1], b[2], b[3], b[4], "pphuman") for b in pphuman_norm] + [
            (float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4]), "sahi")
            for d in sahi_dets
        ]
        # Sort by score desc.
        all_dets.sort(key=lambda x: -x[4])

        # Compute IoU pairs to know which pphuman+sahi pairs collapsed.
        pair_collapsed = {}  # id(all_dets[idx]) -> True
        for i in range(len(all_dets)):
            for j in range(i + 1, len(all_dets)):
                if all_dets[i][5] != all_dets[j][5] and self._iou(
                    all_dets[i], all_dets[j]
                ) > self._config.union_nms_iou:
                    pair_collapsed[id(all_dets[i])] = True
                    pair_collapsed[id(all_dets[j])] = True

        keep: List[tuple] = []
        suppressed = [False] * len(all_dets)
        for i in range(len(all_dets)):
            if suppressed[i]:
                continue
            best = all_dets[i]
            keep.append(best)
            for j in range(i + 1, len(all_dets)):
                if suppressed[j]:
                    continue
                if self._iou(best, all_dets[j]) > self._config.union_nms_iou:
                    suppressed[j] = True

        # Tag survivors.
        out: List[MergedDetection] = []
        for det in keep:
            x1, y1, x2, y2, score, src = det
            if id(det) in pair_collapsed:
                tag = "merged"
            else:
                tag = src
            out.append(
                MergedDetection(
                    x1=x1, y1=y1, x2=x2, y2=y2, score=score, source=tag
                )
            )
        return out

    def _read_sahi_event(self) -> Optional[dict]:
        """Non-blocking: GET first, then XREVRANGE COUNT 1 as fallback.

        Staleness filtering against ``current_timestamp_ms`` is the
        caller's responsibility (see ``merge``); this method only
        parses and returns the most recent event.
        """
        try:
            latest_raw = self._redis.get(self._config.latest_key_prefix + self._camera_id)
        except Exception:  # noqa: BLE001
            latest_raw = None
        event: Optional[dict] = None
        if latest_raw:
            try:
                event = json.loads(latest_raw)
            except Exception:  # noqa: BLE001
                event = None
        if event is None:
            try:
                items = self._redis.xrevrange(
                    self._config.stream, max="+", min="-", count=1
                )
            except Exception:  # noqa: BLE001
                return None
            if not items:
                return None
            _eid, fields = items[0]
            try:
                ts_ms = int(fields.get("timestamp_ms", "0"))
            except Exception:  # noqa: BLE001
                ts_ms = 0
            dets_raw = fields.get("detections", "[]")
            if isinstance(dets_raw, str):
                try:
                    dets_raw = json.loads(dets_raw)
                except Exception:  # noqa: BLE001
                    dets_raw = []
            event = {
                "camera_id": fields.get("camera_id", ""),
                "frame_id": fields.get("frame_id", ""),
                "timestamp_ms": ts_ms,
                "detections": dets_raw,
            }
        return event

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
        bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _tag_pphuman(pphuman_dets: Sequence[dict]) -> List[MergedDetection]:
        return [
            MergedDetection(
                x1=float(d["x1"]),
                y1=float(d["y1"]),
                x2=float(d["x2"]),
                y2=float(d["y2"]),
                score=float(d.get("score", 1.0)),
                source="pphuman",
            )
            for d in pphuman_dets
        ]
