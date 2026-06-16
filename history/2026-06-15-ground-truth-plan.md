# Phase 6 — Ground truth label plan for cam_merged

Date: 2026-06-13

## Status: PLAN READY, LABELS NOT YET PRODUCED

The plan for labelling the cam_merged validation set is
written; **no actual labels exist yet** for cam_merged. This
is consistent with the spec ("If no labels exist, do not try
to force LIMITED_PRODUCTION.").

## Document added

`Docs/ground_truth_labeling_cam_merged.md` — extends the
umbrella plan in `Docs/ground_truth_labeling_plan.md` with
a concrete JSON schema for the cam_merged set, the four
required metrics, the operator labelling procedure, and the
gating logic that turns a label file into
`READY_FOR_LIMITED_PRODUCTION`.

## Schema (top-level shape)

```json
{
  "dataset_name": "cam_merged_validation",
  "site_id": "yamaha_showroom",
  "schema_version": "1.0.0",
  "fps": 10,
  "timezone": "Asia/Jakarta",
  "labellers": ["operator-A", "operator-B"],
  "labelled_at": "2026-06-15T08:00:00Z",
  "global_persons": [
    {
      "global_person_id": "P001",
      "notes": "Tall man, blue jacket",
      "segments": [
        {
          "camera_id": "CAM_01",
          "start_frame": 100,
          "end_frame": 500,
          "start_time": 10.0,
          "end_time": 50.0,
          "bbox_track": "optional_or_external",
          "tracklet_id_hint": 7,
          "quality_flag": "ok"
        },
        {
          "camera_id": "CAM_02",
          "start_frame": 800,
          "end_frame": 1200,
          "start_time": 80.0,
          "end_time": 120.0,
          "bbox_track": "optional_or_external",
          "tracklet_id_hint": null,
          "quality_flag": "occluded_in_middle"
        }
      ]
    }
  ]
}
```

## Required metrics (the four §3 of the doc)

| Metric | Definition | Threshold |
|---|---|---|
| `false_merge_rate` | fraction of distinct global_persons for which the system emits fewer `global_id`s than segments | ≤ 0.05 |
| `cross_camera_match_accuracy` | fraction of cross-camera persons for which the system emits one `global_id` for all segments | ≥ 0.85 |
| `id_fragmentation_rate` | fraction of segments for which the system emits > 1 `global_id` within a single segment | ≤ 0.20 |
| `ambiguous_decision_rate` | fraction of resolver decisions that landed in the ambiguous band | = 0.0 |

## Where labels live

```
data/labels/cam_merged_validation.json
```

The benchmark dataset manifest
(`configs/benchmark_real_cam_merged.yaml`) already has the
`labels.optional_ground_truth_path` field; setting it to
`null` (as it is now) makes the benchmark record
`required_metrics_present=false`.

## Operator procedure

1. Watch `data/cam1_merged.mp4` and `data/cam2_merged.mp4`
   in VLC/mpv. Use the
   `Service/offline-people-counting/scripts/eval/build_labeler.py`
   HTML contact-sheet for visual review.
2. Identify cross-camera transitions; allocate
   `global_person_id` per distinct person.
3. Record segments (camera, start/end frame, start/end
   wall-clock, `quality_flag`).
4. Write the JSON; validate with `jq`.
5. Re-run the benchmark; the report includes the four
   metrics.

## Tooling status

- **Ready:** the umbrella
  `Docs/ground_truth_labeling_plan.md` and the
  `Service/offline-people-counting/scripts/eval/build_labeler.py`
  contact-sheet generator.
- **To do (out of scope here):**
  `scripts/inspect_labels.py` and
  `scripts/compute_label_metrics.py` standalone tools.
  The benchmark_t4 script inlines the metric logic, so
  the standalone tools are nice-to-have for QA reviewers,
  not blocking.

## Files added

- `Docs/ground_truth_labeling_cam_merged.md` — new.

## Verdict

Plan is in place. The labels themselves remain operator
work. Until they are produced **and** the production
benchmark runs end-to-end on the labelled video with
the four §3 metrics within thresholds, the readiness
gate will not move past `READY_FOR_SHADOW_TEST`.
