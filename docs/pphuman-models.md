# PP-Human model setup

> **Exact commands that produced the on-disk PP-Human artifacts
> this repo has been validated against on host
> `/home/rhendy/paddledetection/`.**

## What we need

| Artifact                                | Used for          | Size   |
| --------------------------------------- | ----------------- | ------ |
| PaddleDetection git clone               | pipeline.py       | ~50 MB |
| `mot_ppyoloe_l_36e_pipeline`            | MOT detector      | ~150 MB |
| `strongbaseline_r50_30e_pa100k`         | PP-Human ReID     | ~80 MB |

## PaddleDetection clone

```bash
git clone --depth 1 \
    https://github.com/PaddlePaddle/PaddleDetection.git \
    /home/$USER/paddledetection

# Verify the two files the pipeline needs:
test -f /home/$USER/paddledetection/deploy/pipeline/pipeline.py
test -f /home/$USER/paddledetection/deploy/pipeline/config/infer_cfg_pphuman.yml

# Record the path in .env:
echo "PPHUMAN_PIPELINE_PATH=/home/$USER/paddledetection/deploy/pipeline/pipeline.py" >> .env
```

> On hosts without `sudo` (this host has no root access), do NOT
> use `/opt/paddledetection`.  Use a user-owned path under
> `$HOME` and point `PPHUMAN_PIPELINE_PATH` at it.

## PP-Human models

The official PP-Human v2.4 zips that used to live under
`https://paddledet.bj.bcebos.com/mot/v2.4/*.tar.gz` now return
**HTTP 403** and **must not be used**.  The current source of
truth is the BCE Bos `pipeline/` directory:

```text
https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip
https://bj.bcebos.com/v1/paddledet/models/pipeline/strongbaseline_r50_30e_pa100k.zip
```

The vendored script `scripts/download_pphuman_models.sh` uses the
correct URLs:

```bash
mkdir -p models/pphuman
bash scripts/download_pphuman_models.sh
```

The script writes:

```text
models/pphuman/mot_ppyoloe_l_36e_pipeline/
    inference.pdmodel
    inference.pdiparams
    infer_cfg.yml
models/pphuman/strongbaseline_r50_30e_pa100k/
    inference.pdmodel
    inference.pdiparams
    infer_cfg.yml
```

## Runtime envs

```text
PPHUMAN_PIPELINE_PATH=/home/$USER/paddledetection/deploy/pipeline/pipeline.py
PPHUMAN_MODEL_DIR=models/pphuman               # absolute is also fine
PPHUMAN_DEVICE=gpu                              # T4 → gpu
PPHUMAN_RUN_MODE=trt_fp16                       # T4 → fp16 TRT
```

## Verify on this host

```bash
uv run python -c "
from app.detection.pphuman_pipeline import PPHumanDetectorAdapter
adapter = PPHumanDetectorAdapter(
    pipeline_path='/home/rhendy/paddledetection/deploy/pipeline/pipeline.py',
    config_path='/home/rhendy/paddledetection/deploy/pipeline/config/infer_cfg_pphuman.yml',
    model_dir='models/pphuman',
    device='gpu',
    run_mode='trt_fp16',
)
print('pphuman adapter wired ok')
"
```

## Hard rules

```text
1. /mot/v2.4/*.tar.gz URLs are dead.  Do not re-introduce them.
2. The PP-Human ReID is `strongbaseline_r50_30e_pa100k` (256-d).
   Do not silently substitute another ReID model.
3. trt_fp16 is the T4 default.  trt_int8 requires a calibration
   cache that is NOT shipped.
4. In production mode the synthetic / heuristic detector path
   is refused (RuntimeMode.PRODUCTION).
```
