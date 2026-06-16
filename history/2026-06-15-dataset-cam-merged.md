# Phase 2 — Dataset Copy (cam_merged)

Date: 2026-06-13
Source: `/home/rhendy/Projects/yamaha/Service/data/`
Destination: `/home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC/data/`
Pattern: `cam1_merged*` and `cam2_merged*` only.

## 1. Files matched in Service/data (maxdepth 1)

```text
./cam1_merged.mp4
./cam2_merged.mp4
```

Only these two files were matched; the rest of `Service/data/`
(`CCTV 29 APR 2026/`, `CCTV AI/`, `CCTV FSS-20260429T020634Z-3-001/`,
`crossing_*.mp4`, `datasets/`, `fss_*_merged.mp4`, …) was explicitly
excluded by the glob.

## 2. Copy command

```bash
cp -n /home/rhendy/Projects/yamaha/Service/data/cam1_merged.mp4 \
      /home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC/data/

cp -n /home/rhendy/Projects/yamaha/Service/data/cam2_merged.mp4 \
      /home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC/data/
```

`cp -n` is used to refuse overwriting existing files. Both files were
successfully copied (the destination did not have them prior to
Phase 2).

## 3. Resulting `data/` listing

```text
data/
├── cam1_merged.mp4      2.1G  (ISO Media, MP4 Base Media v1)
├── cam2_merged.mp4      1.9G  (ISO Media, MP4 Base Media v1)
├── cam01.mp4            39K   (pre-existing, smoke test stub)
└── cam02.mp4            39K   (pre-existing, smoke test stub)
```

The two `cam01.mp4` / `cam02.mp4` stub files are unrelated to the
newly copied `cam1_merged.mp4` / `cam2_merged.mp4`. They are the
short OpenCV-generated stubs used by the smoke benchmark.

## 4. Detected video/container format

`file` command output:

```text
data/cam1_merged.mp4: ISO Media, MP4 Base Media v1 [ISO 14496-12:2003]
data/cam2_merged.mp4: ISO Media, MP4 Base Media v1 [ISO 14496-12:2003]
```

OpenCV probe (`cv2.VideoCapture`):

| File | Frames | Resolution | FPS | Duration | FourCC |
| --- | ---: | --- | ---: | ---: | --- |
| `cam1_merged.mp4` | 143 726 | 3072 × 2048 | 20.00 | 7 186.3 s ≈ 1 h 59 m | `hevc` |
| `cam2_merged.mp4` | 141 852 | 2592 × 1944 | 20.00 | 7 092.6 s ≈ 1 h 58 m | `hevc` |

Both videos are well over the 3 000-frame visualization budget; the
visualization script will stop after 3 000 frames (or earlier on
`--max-frames`).

## 5. CAM_01 / CAM_02 mapping

Per the task spec:

| SOTA camera id | Source file |
| --- | --- |
| `CAM_01` | `data/cam1_merged.mp4` |
| `CAM_02` | `data/cam2_merged.mp4` |

The mapping is enforced in `configs/cameras.yaml` (the existing entry
already uses `cam01` / `cam02` aliases; the visual-validation script
will use the `CAM_01` / `CAM_02` labels as the *operator-facing* name
while resolving the underlying file via the `cam1_merged` /
`cam2_merged` pattern).

## 6. Files NOT copied (explicit exclusion)

- `CCTV 29 APR 2026/` — full production capture (~600 MiB) — NOT
  copied because the user explicitly scoped the import to
  `cam1_merged*` / `cam2_merged*`.
- `CCTV AI/`, `CCTV FSS-…/`, `datasets/` — not relevant to the
  validation scope.
- `crossing_test_30s.mp4`, `crossing_test_30s_hard.mp4` — single-shot
  tests, not multi-camera validation material.
- `crossing_ground_truth_hard.json` — out of scope.
- `fss_cam1_merged.mp4`, `fss_cam2_merged.mp4` — alternative merged
  sets; only the canonical `cam1_merged.mp4` / `cam2_merged.mp4`
  pairs are imported.

## 7. Verdict

The minimal dataset required for visual validation is in place. The
files are large (~4 GiB total), so the visual-validation script must
be efficient (frame skipping, no per-frame model re-init) and the
operator should pre-allocate at least 8 GiB of free disk for the
output MP4s.

## 8. Phase 2 re-verification (2026-06-13)

Re-verified via `md5sum` — both files are byte-identical to the
Service originals:

```text
e7bcb383bec88fb27971e39d14701216  data/cam1_merged.mp4
8d7704bf4ead7b10498f79ad588b8ee8  data/cam2_merged.mp4
```

The pre-existing `data/cam01.mp4` and `data/cam02.mp4` (39 KB
OpenCV-generated stubs) remain in place and are used by the smoke
benchmark; they are not the same as the 2-3 hour multi-camera
captures.
