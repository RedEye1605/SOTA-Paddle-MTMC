"""5-factor weighted scoring for global identity decisions.

```
final_score = reid_weight * reid_similarity
            + temporal_weight * temporal_score
            + camera_weight * camera_topology_score
            + quality_weight * crop_quality_score
            + zone_weight * zone_transition_score
```

All inputs are expected in [0, 1]. Output is also in [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoreWeights:
    reid_weight: float = 0.55
    temporal_weight: float = 0.20
    camera_weight: float = 0.15
    quality_weight: float = 0.05
    zone_weight: float = 0.05

    def total(self) -> float:
        return (
            self.reid_weight
            + self.temporal_weight
            + self.camera_weight
            + self.quality_weight
            + self.zone_weight
        )

    def normalized(self) -> "ScoreWeights":
        t = self.total()
        if t <= 0:
            return self
        return ScoreWeights(
            reid_weight=self.reid_weight / t,
            temporal_weight=self.temporal_weight / t,
            camera_weight=self.camera_weight / t,
            quality_weight=self.quality_weight / t,
            zone_weight=self.zone_weight / t,
        )


def temporal_score(time_diff_seconds: float, sigma_seconds: float = 60.0) -> float:
    """Gaussian decay. Same instant → 1.0; far in the past → 0.0."""
    if sigma_seconds <= 0:
        return 0.0
    return float(math.exp(-(time_diff_seconds**2) / (2.0 * sigma_seconds**2)))


def camera_topology_score(is_known_link: Optional[bool]) -> float:
    """1.0 if explicitly enabled, 0.5 if unknown (no row), 0.0 if disabled.

    `is_known_link`:
      - True:  row exists and enabled
      - False: row exists and enabled=False (explicit impossibility)
      - None:  no row at all
    """
    if is_known_link is True:
        return 1.0
    if is_known_link is False:
        return 0.0
    return 0.5


def quality_score_from_tracklet(quality: Optional[float]) -> float:
    """Default 0.5 when the tracklet quality is missing."""
    if quality is None:
        return 0.5
    return max(0.0, min(1.0, quality))


def zone_transition_score(prev_zone_id: Optional[str], new_zone_id: Optional[str]) -> float:
    """Same zone → 1.0; both present but different → 0.5; one missing → 0.5."""
    if prev_zone_id is None or new_zone_id is None:
        return 0.5
    return 1.0 if prev_zone_id == new_zone_id else 0.5


def final_score(
    reid_similarity: float,
    time_diff_seconds: float,
    is_known_link: Optional[bool],
    tracklet_quality: Optional[float],
    prev_zone_id: Optional[str],
    new_zone_id: Optional[str],
    weights: ScoreWeights,
    sigma_seconds: float = 60.0,
) -> float:
    w = weights.normalized()
    t = temporal_score(time_diff_seconds, sigma_seconds)
    cam = camera_topology_score(is_known_link)
    qual = quality_score_from_tracklet(tracklet_quality)
    zone = zone_transition_score(prev_zone_id, new_zone_id)
    score = (
        w.reid_weight * reid_similarity
        + w.temporal_weight * t
        + w.camera_weight * cam
        + w.quality_weight * qual
        + w.zone_weight * zone
    )
    return float(max(0.0, min(1.0, score)))


def score_breakdown(
    reid_similarity: float,
    time_diff_seconds: float,
    is_known_link: Optional[bool],
    tracklet_quality: Optional[float],
    prev_zone_id: Optional[str],
    new_zone_id: Optional[str],
    weights: ScoreWeights,
    sigma_seconds: float = 60.0,
) -> dict[str, float]:
    """Like :func:`final_score` but returns each component for audit."""
    w = weights.normalized()
    return {
        "reid_similarity": reid_similarity,
        "temporal_score": temporal_score(time_diff_seconds, sigma_seconds),
        "camera_topology_score": camera_topology_score(is_known_link),
        "quality_score": quality_score_from_tracklet(tracklet_quality),
        "zone_score": zone_transition_score(prev_zone_id, new_zone_id),
        "final_score": final_score(
            reid_similarity,
            time_diff_seconds,
            is_known_link,
            tracklet_quality,
            prev_zone_id,
            new_zone_id,
            weights,
            sigma_seconds,
        ),
        "weight_reid": w.reid_weight,
        "weight_temporal": w.temporal_weight,
        "weight_camera": w.camera_weight,
        "weight_quality": w.quality_weight,
        "weight_zone": w.zone_weight,
    }
