# cam_merged dataset setup

The yamaha showroom pipeline validates against two multi-hour
recorded videos: `cam1_merged.mp4` (camera 1) and
`cam2_merged.mp4` (camera 2). They were merged from the per-day
recordings and live in `Service/data/` (the upstream system); the
SOTA pipeline keeps a local copy under `SOTA-Paddle-MTMC/data/`.

## Copying the dataset

```bash
# Only these two files are required.
cp -n /path/to/Service/data/cam1_merged.mp4 \
      SOTA-Paddle-MTMC/data/

cp -n /path/to/Service/data/cam2_merged.mp4 \
      SOTA-Paddle-MTMC/data/
```

`cp -n` refuses to overwrite. Re-run is safe.

## Verifying integrity

```bash
md5sum /path/to/Service/data/cam1_merged.mp4 \
       SOTA-Paddle-MTMC/data/cam1_merged.mp4
md5sum /path/to/Service/data/cam2_merged.mp4 \
       SOTA-Paddle-MTMC/data/cam2_merged.mp4
```

The MD5s must match.

## Properties

| File | Resolution | FPS | Frames | FourCC | Size |
| --- | --- | ---: | ---: | --- | ---: |
| `cam1_merged.mp4` | 3072 × 2048 | 20.00 | 143 726 | `hevc` | ~2.1 GiB |
| `cam2_merged.mp4` | 2592 × 1944 | 20.00 | 141 852 | `hevc` | ~1.8 GiB |

Both videos are well over the 3000-frame visualization budget; the
visualization script stops after 3000 frames.

## Wiring into a benchmark manifest

The shipped manifest `configs/benchmark_real_cam_merged.yaml`
points at both files. To use it:

```bash
uv run python scripts/benchmark_t4.py \
  --mode smoke_benchmark \
  --dataset configs/benchmark_real_cam_merged.yaml \
  --max-seconds 30
```

For the production benchmark, the PaddleDetection stack must be
installed (see `Docs/pphuman_model_setup.md`). The
`runtime_mode` safety gate will refuse to start a production
benchmark without the real adapter.

## What is *not* copied

Only `cam1_merged.mp4` and `cam2_merged.mp4` are copied. The
upstream `Service/data/` folder also contains:
`CCTV 29 APR 2026/`, `CCTV AI/`, `CCTV FSS-*/`,
`crossing_test_30s*.mp4`, `crossing_ground_truth_hard.json`,
`datasets/`, and `fss_*_merged.mp4` — none of which are part of
the cam_merged validation scope.
