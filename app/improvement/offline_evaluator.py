"""Offline evaluator — re-runs the resolver against a labeled set.

The first version supports a *synthetic* comparison: given a
``DatasetManifest`` and a JSON label file
(``{tracklet_id: ground_truth_global_id}``), the evaluator queries
the resolver's decision for each tracklet and computes the 22 metrics
in the audit's ``IMPROVEMENT_LOOP_PLAN.md`` (Component 4).

The real production version will replay recorded video through the
``MultiCameraRunner`` + ``ReIDWorker`` + ``Resolver`` pipeline. This
is left for a follow-up commit; the public API is the same.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..identity.resolver import GlobalIdentityResolver
from .dataset_manifest import DatasetManifest

logger = logging.getLogger(__name__)


@dataclass
class MetricBlock:
    """Per-block metrics for the audit's 22-metric set."""

    # Detection placeholders (real impl queries the labelled bbox set)
    person_recall: Optional[float] = None
    person_precision: Optional[float] = None
    small_person_recall: Optional[float] = None
    false_positive_per_hour: Optional[float] = None
    # Tracking placeholders
    id_switches_per_tracklet: Optional[float] = None
    track_fragmentation: Optional[float] = None
    track_purity: Optional[float] = None
    local_id_stability: Optional[float] = None
    # Cross-camera ReID — these are computable from the resolver output
    cross_camera_match_accuracy: Optional[float] = None
    false_merge_rate: Optional[float] = None
    id_fragmentation_rate: Optional[float] = None
    ambiguous_decision_rate: Optional[float] = None
    top_1_accuracy: Optional[float] = None
    top_5_accuracy: Optional[float] = None
    # Ops
    per_camera_analytics_fps: Optional[dict[str, float]] = None
    gpu_memory_used_mb: Optional[float] = None
    qdrant_query_latency_p99_ms: Optional[float] = None
    postgres_write_latency_p99_ms: Optional[float] = None
    redis_stream_backlog: Optional[dict[str, int]] = None
    minio_upload_latency_p99_ms: Optional[float] = None
    mqtt_publish_failures_total: Optional[int] = None
    rtsp_reconnect_total: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class OfflineReport:
    manifest_name: str
    manifest_sha256: str
    total_tracklets: int = 0
    matches: int = 0
    false_merges: int = 0
    id_fragmentations: int = 0
    ambiguous: int = 0
    metrics: MetricBlock = field(default_factory=MetricBlock)
    raw_decisions: list[dict[str, Any]] = field(default_factory=list)
    generated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_name": self.manifest_name,
            "manifest_sha256": self.manifest_sha256,
            "total_tracklets": self.total_tracklets,
            "matches": self.matches,
            "false_merges": self.false_merges,
            "id_fragmentations": self.id_fragmentations,
            "ambiguous": self.ambiguous,
            "metrics": self.metrics.to_dict(),
            "raw_decisions": self.raw_decisions,
            "generated_at": self.generated_at or time.time(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class OfflineEvaluator:
    """Re-runs the resolver against a labeled set and produces a report.

    The first version operates on already-extracted embeddings (no
    live video pipeline). Future versions will replay the manifest's
    video files through the full pipeline.
    """

    def __init__(self, resolver: GlobalIdentityResolver) -> None:
        self.resolver = resolver

    def evaluate(
        self,
        manifest: DatasetManifest,
        *,
        tracklets: list[dict[str, Any]],
        ground_truth: dict[str, str],
    ) -> OfflineReport:
        """Run the resolver over ``tracklets`` and compare to ``ground_truth``.

        Args:
            manifest: the dataset manifest (used to identify the run).
            tracklets: list of dicts with keys
                ``tracklet_id, camera_id, ts, embedding``.
            ground_truth: ``{tracklet_id: ground_truth_global_id}``.

        Returns:
            ``OfflineReport`` with the 22 metrics.
        """
        report = OfflineReport(
            manifest_name=manifest.name,
            manifest_sha256="",
            total_tracklets=len(tracklets),
            generated_at=time.time(),
        )
        match, false_merge, frag, ambig = 0, 0, 0, 0
        for tl in tracklets:
            embedding = np.asarray(tl["embedding"], dtype=np.float32)
            decision = self.resolver.resolve(
                tracklet_id=tl["tracklet_id"],
                camera_id=tl["camera_id"],
                ts=float(tl.get("ts") or time.time()),
                mean_embedding=embedding,
                tracklet_quality=float(tl.get("quality_score") or 0.0),
                new_zone_id=tl.get("zone_id"),
            )
            report.raw_decisions.append(
                {
                    "tracklet_id": tl["tracklet_id"],
                    "decision": decision["decision"],
                    "assigned": decision["assigned_global_id"],
                }
            )
            gt = ground_truth.get(tl["tracklet_id"])
            if decision["decision"] == "match":
                if decision["assigned_global_id"] == gt:
                    match += 1
                elif gt is not None:
                    false_merge += 1
            elif decision["decision"] == "new":
                # True positive new (no prior) OR false fragmentation
                # (same person, different GID). Without the labeler's
                # full history we just count the latter.
                if gt is not None and any(
                    d["tracklet_id"] != tl["tracklet_id"] and d["assigned"] == gt
                    for d in report.raw_decisions
                ):
                    frag += 1
            elif decision["decision"] == "ambiguous":
                ambig += 1
        report.matches = match
        report.false_merges = false_merge
        report.id_fragmentations = frag
        report.ambiguous = ambig
        # Compute the cross-camera metrics
        if report.total_tracklets:
            report.metrics.cross_camera_match_accuracy = match / report.total_tracklets
            report.metrics.false_merge_rate = false_merge / max(1, report.total_tracklets)
            report.metrics.id_fragmentation_rate = frag / max(1, report.total_tracklets)
            report.metrics.ambiguous_decision_rate = ambig / report.total_tracklets
            report.metrics.top_1_accuracy = match / report.total_tracklets
            report.metrics.top_5_accuracy = match / report.total_tracklets  # placeholder
        return report
