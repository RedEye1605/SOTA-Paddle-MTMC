# Ground-truth labelling plan

This document specifies the labelled multi-camera dataset
required to drive the production benchmark to
`READY_FOR_LIMITED_PRODUCTION`. The plan is intentionally
minimal — it pins the dataset size, the label format, the
metrics computed from the labels, and the integration points
into `scripts/benchmark_t4.py` and
`scripts/readiness_gate.py`. Until this plan (or an equivalent
labelled dataset) exists, the readiness gate caps at
`READY_FOR_SHADOW_TEST`.

## 1. Minimum dataset duration

* **At least 30 minutes** of synchronised video.
* **At least 2 cameras** covering the same physical space
  with overlapping fields of view.
* Cameras must be **time-synchronised** (NTP or hardware sync,
  ±100 ms tolerance). The label schema uses a single global
  timeline; mis-synchronised video makes cross-camera
  transitions impossible to label.
* The recorded video format is the same one the benchmark
  consumes today: an mp4 readable by OpenCV's
  `VideoCapture`, with H.264/AAC and 5–10 fps. The benchmark
  does not need 30 fps; the 5–10 fps cadence matches the
  `MultiCameraRunner`'s default `skip_frame_num=2`.

## 2. Required labels

For every cross-camera transition of a single person, the
labeller records the following fields:

| Field | Type | Description |
|---|---|---|
| `global_id` | int | Globally unique person id (1..N). |
| `appearances` | list | One entry per (camera, tracklet, time-range). |
| `appearances[].camera_id` | string | The camera (e.g. `CAM_01`). |
| `appearances[].tracklet_id` | int | The local track id within that camera. |
| `appearances[].frame_range` | [int, int] | Inclusive frame range where the tracklet is visible. |
| `appearances[].time_range` | [float, float] | Wall-clock seconds (from video start). |
| `appearances[].bbox_samples` | list | (Optional) 5 sample bboxes for QA. |

The label format **must** distinguish between
*cross-camera transitions* (a person leaves one camera's
view and enters another's) and *re-identification* (a person
briefly disappears and re-appears in the same camera). The
label tool is the operator's choice; the format is JSON for
machine readability.

## 3. Metrics to compute

The benchmark loads the labels and the per-camera MOT
detections, then computes:

| Metric | Definition |
|---|---|
| `false_merge_rate` | Fraction of decisions where two different `global_id`s were merged into one. |
| `cross_camera_match_accuracy` | Fraction of cross-camera transitions where the resolver assigned the same `global_id` to both appearances. |
| `id_fragmentation_rate` | Fraction of true `global_id`s that the resolver split into ≥2 distinct identities. |
| `ambiguous_decision_rate` | Fraction of decisions marked `ambiguous` (the resolver refused to merge and parked the tracklet for re-id at stage 3). |

The exact formulas are in
`app/improvement/promotion_gate.py` and the
`compute_*` helpers the benchmark would call. The integration
contract is that **all four metrics must be present in the
report** for `READY_FOR_LIMITED_PRODUCTION`.

## 4. Label format

JSON. The top-level shape:

```json
{
  "dataset": {
    "name": "yamaha_showroom_30min",
    "site_id": "yamaha_demo",
    "fps": 5,
    "cameras": [
      {"camera_id": "CAM_01", "video_path": "/data/cam01.mp4"},
      {"camera_id": "CAM_02", "video_path": "/data/cam02.mp4"}
    ]
  },
  "global_identities": [
    {
      "global_id": 1,
      "appearances": [
        {
          "camera_id": "CAM_01",
          "tracklet_id": 17,
          "frame_range": [120, 480],
          "time_range": [24.0, 96.0]
        },
        {
          "camera_id": "CAM_02",
          "tracklet_id": 9,
          "frame_range": [510, 870],
          "time_range": [102.0, 174.0]
        }
      ]
    }
  ]
}
```

CSV is acceptable for small datasets (one row per
appearance, with `global_id` repeated) but JSON is the
canonical format and what `benchmark_t4.py` consumes by
default.

## 5. How `benchmark_t4.py` consumes labels

`benchmark_t4.py` reads the manifest's
`labels.optional_ground_truth_path` and (when present) calls
the labelling-aware accuracy helpers. The current code at
`_maybe_load_labels_and_score` already records `labels_loaded`
and a note; the next iteration is to add a real
`compute_metrics_from_labels(decisions_json, labels_json)`
that produces the four metrics above.

A sketch of the integration:

```python
def _compute_accuracy_metrics(report, dataset, decisions):
    labels = _load_labels(dataset["labels"]["optional_ground_truth_path"])
    if labels is None:
        return {}
    return {
        "false_merge_rate": compute_false_merge_rate(decisions, labels),
        "cross_camera_match_accuracy": compute_xc_match_accuracy(decisions, labels),
        "id_fragmentation_rate": compute_id_fragmentation(decisions, labels),
        "ambiguous_decision_rate": compute_ambiguous_rate(decisions),
    }
```

`report["status"]` becomes `success` only when all four
metrics are non-null and the benchmark ran for at least 2
cameras.

## 6. How `readiness_gate.py` uses the metrics

`readiness_gate.py` delegates the metrics gate to
`app.improvement.promotion_gate.PromotionGate` with
`require_real_metrics=True` (the default). When
`production_benchmark`'s `required_metrics_present` is
`False`, the gate caps the verdict at
`READY_FOR_SHADOW_TEST` and the operator's next step is
to provide labels.

When all four metrics are present and within thresholds
(see `configs/benchmark.yaml`'s `gate` block), the gate
promotes to `READY_FOR_LIMITED_PRODUCTION`.

## 7. Tooling

The labeler can be a manual spreadsheet exported to JSON,
or a custom tool. The reference service's
`Service/offline-people-counting/scripts/eval/build_labeler.py`
is a working starting point. The labeler's output schema
matches the JSON above; the benchmark consumes it directly.

## 8. Privacy scope

The labels contain pseudonymous person identities
(`global_id` is a small integer). They MUST NOT contain
names, faces, biometrics, demographics, or emotions. The
`PII` rule (hard rule #8 in the project rules) is enforced
by a test in
`tests/test_architecture_guards_one_model.py` — that test
MUST be updated to assert that the label JSON has no
forbidden fields (e.g. `name`, `face_embedding`,
`gender`, `age`, `emotion`).
