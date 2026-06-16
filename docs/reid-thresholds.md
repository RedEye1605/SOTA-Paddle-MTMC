# ReID Threshold Tuning

## Initial weights (from task spec)

```yaml
reid_weight:        0.55
temporal_weight:    0.20
camera_weight:      0.15
quality_weight:     0.05
zone_weight:        0.05
```

## Initial thresholds

```yaml
auto_match_threshold:   0.82
candidate_threshold:    0.72
ambiguous_margin:       0.04
```

## Decision policy

```
final_score =
    reid_weight * reid_similarity
  + temporal_weight * temporal_score
  + camera_weight  * camera_topology_score
  + quality_weight * crop_quality_score
  + zone_weight    * zone_transition_score
```

```
if final_score >= auto_match_threshold
   and (top1 - top2) > ambiguous_margin
   and camera_topology_valid:
       assign existing global_id

elif candidate_threshold <= final_score < auto_match_threshold:
       store candidate, do NOT auto-merge

elif top1 and top2 too close (margin < ambiguous_margin):
       mark ambiguous, create new global_id or hold

else:
       create new global_id
```

## What each factor measures

- `reid_similarity` — cosine similarity between normalized tracklet embedding
  and candidate global identity's mean embedding. Range: [0, 1].
- `temporal_score` — Gaussian decay of the time difference between the new
  tracklet's timestamp and the candidate's `last_seen_at`. Sigma = 60 s.
  Range: [0, 1].
- `camera_topology_score` — 1.0 if `camera_links` allows the transition,
  0.5 if `camera_links` does not exist (unknown transition), 0.0 if
  `camera_links.enabled = false` for this link.
- `crop_quality_score` — combined score of bbox height, blur, brightness,
  occlusion. Range: [0, 1].
- `zone_transition_score` — 1.0 if the candidate's last zone and the new
  tracklet's zone are physically adjacent (or the same), 0.5 if not adjacent,
  0.0 if one of the two zones is missing.

## Tuning recommendations (out of scope, documented for ops)

1. **Start with the spec defaults** and capture a 24 h ground-truth log.
2. **Plot the final_score histogram** for true matches and true non-matches.
3. **Find the knee** where false merges start appearing.
4. **Lower the auto_match_threshold** if you have too many new IDs.
5. **Raise it** if you see the same physical person get 2+ global_ids.

## Anti-patterns to avoid

- ❌ Lowering threshold to "make matches" → false merges dominate.
- ❌ Disabling topology check → matches across impossible camera pairs.
- ❌ Disabling margin check → top-1/top-2 collisions get auto-merged.
- ❌ Single-factor decision (cosine alone) → ignores time, geometry, quality.
