# Ground-truth labelling for the cam_merged set

This document extends the umbrella plan in
[`Docs/ground_truth_labeling_plan.md`](ground_truth_labeling_plan.md)
with a **concrete JSON schema** for the multi-hour
`data/cam1_merged.mp4` + `data/cam2_merged.mp4` set, plus a
**metrics mapping** that the readiness gate consumes to
move from `READY_FOR_SHADOW_TEST` to
`READY_FOR_LIMITED_PRODUCTION`.

## Why a separate document

The umbrella plan describes the abstract label schema and the
minimum dataset duration (30 min × 2 cameras). This document
pins the **on-disk shape** that `scripts/benchmark_t4.py`
expects, with one example entry per cross-camera transition
type. Operators who follow this document end up with a file
they can drop into `data/labels/cam_merged_validation.json`
and re-run the benchmark without code changes.

## 1. Where labels live

```
data/labels/
  cam_merged_validation.json   # the canonical label set
  cam_merged_validation.schema.json   # JSON Schema (optional)
```

The benchmark dataset manifest
(`configs/benchmark_real_cam_merged.yaml`) points at this
file via the `labels.optional_ground_truth_path` key:

```yaml
dataset:
  name: cam_merged_validation
  cameras:
    - camera_id: CAM_01
      video_path: data/cam1_merged.mp4
    - camera_id: CAM_02
      video_path: data/cam2_merged.mp4
  labels:
    optional_ground_truth_path: data/labels/cam_merged_validation.json
```

If the file is missing or the path is `null`, the benchmark
records `required_metrics_present=false` and the readiness
gate caps the verdict at `READY_FOR_SHADOW_TEST`.

## 2. JSON schema (concrete example)

The label file is a JSON object. The top-level shape is:

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
      "notes": "Tall man, blue jacket; appears 4 times across 2 cameras",
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

### Field reference

| Path | Type | Required | Description |
|---|---|---|---|
| `dataset_name` | string | yes | Matches the manifest's `dataset.name`. |
| `site_id` | string | yes | Matches the manifest's `dataset.site_id`. |
| `schema_version` | string | yes | SemVer. Bump on backwards-incompatible changes. |
| `fps` | number | yes | Source video fps. Used to convert frame ↔ time. |
| `timezone` | string | yes | IANA timezone, used for wall-clock human review. |
| `labellers` | list[string] | yes | Operator names (or anonymous IDs). |
| `labelled_at` | string (ISO-8601) | yes | When the labels were produced. |
| `global_persons` | list[object] | yes | One entry per **global person id**. |
| `global_persons[].global_person_id` | string | yes | Stable id, e.g. `P001`. The benchmark never derives these — the labeller is the source of truth. |
| `global_persons[].notes` | string | no | Free-form description for human reviewers. |
| `global_persons[].segments` | list[object] | yes | At least one entry. |
| `global_persons[].segments[].camera_id` | string | yes | The camera this segment was seen on. |
| `global_persons[].segments[].start_frame` | int | yes | Inclusive first frame of the segment. |
| `global_persons[].segments[].end_frame` | int | yes | Inclusive last frame of the segment. |
| `global_persons[].segments[].start_time` | float | yes | Wall-clock seconds from video start. |
| `global_persons[].segments[].end_time` | float | yes | Wall-clock seconds from video start. |
| `global_persons[].segments[].bbox_track` | string | no | Either `"optional_or_external"`, or a path to a JSON file with the per-frame bboxes (e.g. `data/labels/P001_CAM_01.json`). |
| `global_persons[].segments[].tracklet_id_hint` | int \| null | no | Operator's best guess at the local track id. The benchmark does **not** use this for matching — it is for the QA reviewer. |
| `global_persons[].segments[].quality_flag` | enum | yes | One of `ok`, `occluded_part`, `occluded_in_middle`, `partial_exit`, `re_entry_uncertain`. |

### `quality_flag` semantics

| Flag | Effect on metric calculation |
|---|---|
| `ok` | The segment is fully visible. Counts fully. |
| `occluded_part` | The person is visible for ≥ 80% of the segment; occluded briefly. Counts in ID fragmentation but with a `±10%` tolerance. |
| `occluded_in_middle` | The person is visible at the segment endpoints; occluded in the middle. Counts in ID fragmentation but is **not** used to detect false merges. |
| `partial_exit` | The person leaves the frame at one end. Counts in `id_fragmentation_rate` only if the system emits a new id for the partial segment. |
| `re_entry_uncertain` | The labeller could not be sure whether two segments are the same person. The benchmark counts both options and picks the more conservative metric. |

## 3. Required metrics

The benchmark consumes the labels to compute four
production-readiness metrics. The values are written to
`reports/benchmark_<timestamp>.json` under the top-level
`metrics` key, then read by `scripts/readiness_gate.py`.

### 3.1 `false_merge_rate`

**Definition:** the fraction of *distinct* global_persons in
the labels for which the system's resolver emitted **fewer**
global_ids than there are labellers' segments (i.e. the
system merged two segments of the same labeller identity
into one — a true cross-camera merge was correct; a
*spurious* merge is when the system reports a single
`global_id` for two different `global_person_id`s in the
labels).

**Formula:**

```text
false_merge_rate = (false_merges) / (distinct_global_persons)
```

Where `false_merges` is the count of `global_person_id`s in
the labels for which the system's emitted `global_id` set
contains a value shared with at least one *other*
`global_person_id`.

**Threshold:** `false_merge_rate ≤ 0.05` (per
`configs/benchmark_real_cam_merged.yaml` gate).

### 3.2 `cross_camera_match_accuracy`

**Definition:** the fraction of *cross-camera* segments in
the labels (i.e. `global_person_id`s that appear on more
than one camera) for which the system's resolver emits the
**same** `global_id` for all segments of that person.

**Formula:**

```text
cross_camera_match_accuracy = (correct_xcam_merges) / (xcam_global_persons)
```

**Threshold:** `cross_camera_match_accuracy ≥ 0.85`.

### 3.3 `id_fragmentation_rate`

**Definition:** the fraction of labelled segments for which
the system emitted **more than one** `global_id` within that
single `(camera_id, start_frame, end_frame)` segment — i.e.
the system broke the person's continuous appearance into
multiple ids.

**Formula:**

```text
id_fragmentation_rate = (fragmented_segments) / (total_segments)
```

Segments with `quality_flag ∈ {occluded_in_middle,
partial_exit}` are excluded from the denominator (the system
has legitimate reason to re-id the person).

**Threshold:** `id_fragmentation_rate ≤ 0.20`.

### 3.4 `ambiguous_decision_rate`

**Definition:** the fraction of resolver decisions that
landed in the `ambiguous` band (similarity within
`candidate_threshold` and `auto_match_threshold`) and were
therefore deferred to a human review queue. The benchmark
counts these from the runtime telemetry
(`identity_decisions` table where `decision='ambiguous'`).

**Threshold:** `ambiguous_decision_rate = 0.0` (no
auto-resolution in the ambiguous band; the system must
escalate).

## 4. Labelling procedure (operator steps)

1. **Watch the merged mp4s** in
   `data/cam1_merged.mp4` and `data/cam2_merged.mp4`. They
   are 2 GB and 1.8 GB respectively. Use VLC or mpv with the
   `--pause-on-frame-change` flag, or the
   `scripts/build_labeler.py` HTML tool from `Service/`.
2. **Identify cross-camera transitions** — a person leaves
   one camera's view and enters another's. For each
   transition, allocate a fresh `global_person_id` (e.g.
   `P001`, `P002`, …).
3. **Record segments** — for each appearance of that
   person, record the camera id, the start frame, the end
   frame, the start/end wall-clock times, and a
   `quality_flag`.
4. **Write the JSON** in the shape of §2.
5. **Validate** with `jq`:
   ```bash
   jq -e '.global_persons | length' data/labels/cam_merged_validation.json
   ```
6. **Re-run the benchmark** with the labels:
   ```bash
   docker compose run --rm -w /app api python scripts/benchmark_t4.py \
     --mode production_benchmark \
     --dataset configs/benchmark_real_cam_merged.yaml \
     --max-seconds 600
   ```
   The report will include
   `required_metrics_present=true` and the four metrics
   listed in §3.

## 5. Tooling

- **Service/'s `build_labeler.py`** generates an HTML
  contact sheet for visual review. The `eval/inspector.html`
  produced by that tool can be opened in a browser; frames
  are tiled per camera with frame numbers overlaid.
- **`scripts/inspect_labels.py`** (TBD) will be a small
  validator that checks the JSON shape against the schema
  in §2 and prints per-`global_person_id` summaries.
- **`scripts/compute_label_metrics.py`** (TBD) will compute
  the four §3 metrics from a label file and a benchmark
  report. The benchmark_t4 script inlines this logic but
  the standalone tool is useful for QA.

## 6. What's missing for the cam_merged set

**No manual labels exist yet.** This document is the
**plan**; the labels themselves are operator work. The
cudnn 9 + paddle 2.6.2 blocker (Phase 4 §"Remaining
blockers") must also be resolved before the labels become
useful — without a working production benchmark, the
labels are inert.

## 7. Verdict

This document unblocks the operator's labelling work. The
gate will move to `READY_FOR_LIMITED_PRODUCTION` only
**after** the labels exist **and** the production benchmark
runs end-to-end with the real detector on the labelled
video **and** all four §3 metrics are within thresholds.
