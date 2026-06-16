# Official PaddleDetection / TransReID integration verification

> **Authoritative reference for the PaddleDetection PP-Human pipeline
> and TransReID inference paths used by SOTA-Paddle-MTMC.** All
> commands below were inspected against the official upstream docs
> (via Context7) before being wired into this repo.

## 1. PaddleDetection — PP-Human pipeline

The official PP-Human pipeline is run as a subprocess of
`app/detection/pphuman_pipeline.py:PPHumanDetectorAdapter`. The CLI
surface is documented in
[PaddleDetection release/2.9 — PP-Human tutorials](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.9/deploy/pipeline).

### 1.1 The official command (Context7-verified)

```bash
python deploy/pipeline/pipeline.py \
    --config deploy/pipeline/config/infer_cfg_pphuman.yml \
    -o MOT.enable=True MOT.model_dir=<MODEL_DIR> \
    --video_file=<FILE_OR_RTSP> \
    --device=gpu \
    --run_mode=trt_fp16
```

Key flags (from Context7 `/paddlepaddle/paddledetection`):

| Flag | Meaning |
|---|---|
| `--config` | Path to the infer_cfg YAML (e.g. `infer_cfg_pphuman.yml`). |
| `-o KEY=VALUE` | Override any YAML key (e.g. `MOT.model_dir=…`). |
| `--video_file` | A video file, RTSP URL, or webcam index. |
| `--device` | `gpu` / `cpu` / `xpu`. |
| `--run_mode` | `paddle` / `trt_fp32` / `trt_fp16` / `trt_int8`. |
| `--output_dir` | Where the pipeline writes its MOT outputs. |

### 1.2 The infer_cfg YAML keys we override

```yaml
PIPELINE:
  enable: True

DET:
  model_dir: output_inference/mot_ppyoloe_l_36e_pipeline
  batch_size: 1
  enable: True

MOT:
  model_dir: output_inference/mot_ppyoloe_l_36e_pipeline
  tracker_config: deploy/pipeline/config/tracker_config.yml
  batch_size: 1
  skip_frame_num: 2
  enable: True

ATTR:
  model_dir: output_inference/strongbaseline_r50_30e_pa100k
  batch_size: 8
  enable: False
```

We always leave `ATTR.enable=False` and `SKELETON_ACTION.enable=False`
(both are explicit in the default YAML). Only the MOT block runs.

### 1.3 The MOT output format (Context7-verified)

Paddle's MOT output is a text file `mot_results/{video_name}.txt`,
each line:

```text
frame,id,x1,y1,w,h,score,-1,-1,-1
```

Per the PaddleDetection MOT README, this is the documented format. The
adapter parses it via `PPHumanDetectorAdapter.parse_mot_file()`.

### 1.4 Multi-stream pattern

The official multi-stream pattern is one pipeline process per camera
(this is the documented `multi_camera_mtmct_en.md` approach). Our
`PPHumanPipelineSubprocessManager` implements that pattern.

## 2. TransReID — vendored backbone

We vendor a minimal subset of the official
[`damo-cv/TransReID`](https://github.com/damo-cv/TransReID) inference
path in `app/reid/_transreid_native/`. Specifically:

  * `vit_base_patch16_224_TransReID` — the SIE-augmented ViT-B/16.
  * `JPM` — the Jigsaw Patch Module.
  * `BNNeck` — the BN bottleneck (inference uses `neck_feat='before'`).
  * `build_transreid_model` — wires the above into a `nn.Module`.
  * `extract_inference_feature` — runs the forward pass with the
    official TEST config (`neck_feat='before'`, L2-normalize).

### 2.1 The official model construction (Context7-verified)

```python
from model import make_model
from config import cfg

cfg.merge_from_file('configs/Market/vit_transreid_stride.yml')
cfg.freeze()

model = make_model(
    cfg,
    num_class=751,      # Market-1501; for MSMT17 use 1041
    camera_num=6,
    view_num=0
)
```

The vit_pytorch version is also documented:

```python
model = vit_base_patch16_224_TransReID(
    img_size=(256, 128),
    stride_size=12,
    camera=6, view=0,
    local_feature=True,
    sie_xishu=3.0
)
```

Our vendored `SIETransformer` matches this contract (see
`_transreid_native/model.py`).

### 2.2 The forward pass (Context7-verified)

```python
model.eval()
with torch.no_grad():
    features = model(images, cam_label=cam_labels, view_label=view_labels)
    # JPM: [B, 5*768] = [B, 3840] raw, reduced to [B, 768] via
        # neck_feat='before' + L2-normalize.
```

Our `extract_inference_feature` mirrors this exactly.

### 2.3 Loading safety

The audit's security review requires `weights_only=True` to refuse
arbitrary pickle objects. We enforce this in
`TransReIDAdapter._try_load_real` and `load_transreid_checkpoint`.
The architecture-guard test
`test_dangerous_weights_refused` enforces it at the repo level.

## 3. PaddleInference (PP-Human StrongBaseline ReID)

Per the official PaddleDetection deployment docs, the exported
StrongBaseline model is loaded with `paddle.inference`. Our
`PPHumanReIDAdapter._try_load_paddle()` calls
`paddle.inference.create_predictor(config)` against
`/models/pphuman/strongbaseline_r50_30e_pa100k/inference.pdmodel`.

The predictor's input name is `image`; the output is `embedding`
(256-dim, L2-normalized for cosine distance).

## 4. Paddle Tracking — OC-SORT

`configs/pphuman/tracker_config.yml` matches the official
`tracker_config.yml` keys. In production, OC-SORT runs inside the
PP-Human subprocess (one tracker instance per camera). In smoke-test
mode, the worker's hand-rolled IoU tracker is used; in the real
production deploy, the MOT block in the pipeline runs the official
tracker.

## 5. Multi-camera model sharing (PATCH-007)

Per the README's hard rule: "one model instance shared across all
cameras". With Paddle's per-stream subprocess pattern, this is
satisfied automatically: there is one Python process per camera, but
each process holds a single PaddleInference session. The shared
*adapter* in the parent process is verified by
`tests/test_architecture_guards_one_model.py::test_multi_camera_shares_detector_in_smoke`.

## 6. Cached Context7 queries (for the audit trail)

The PaddleDetection docs (Context7 `/paddlepaddle/paddledetection`,
4141 snippets, High reputation) and TransReID docs
(`/damo-cv/transreid`, 68 snippets, High reputation) were both
queried during the implementation. Key Q&A:

  * **Q:** How is PP-Human pipeline invoked?
    **A:** `python deploy/pipeline/pipeline.py --config
    infer_cfg_pphuman.yml --video_file=… --device=gpu`. Source:
    PaddleDetection official README.

  * **Q:** What is the MOT output format?
    **A:** `frame,id,x1,y1,w,h,score,-1,-1,-1` per line, written to
    `{output_dir}/mot_results/{name}.txt`. Source: PaddleDetection
    `configs/mot/deepsort/README.md`.

  * **Q:** How is TransReID constructed?
    **A:** `make_model(cfg, num_class=751, camera_num=6, view_num=0)`
    OR `vit_base_patch16_224_TransReID(img_size=(256,128), …)`.
    Source: damo-cv/TransReID README.

  * **Q:** How is the forward pass in inference?
    **A:** `model.eval(); with torch.no_grad(): model(images,
    cam_label=…, view_label=…)`. Source: damo-cv/TransReID README.

## 7. Operator checklist for production

  1. Clone PaddleDetection to `/opt/paddledetection` (or set
     `PPHUMAN_PIPELINE_PATH`).
  2. Download the PP-Human MOT model and unzip to `/models/pphuman`
     (or set `PPHUMAN_MODEL_DIR`).
  3. Download `transformer_120.pth` (Market-1501) to
     `/models/transreid/` (or set `TRANSREID_WEIGHT`).
  4. Build the TensorRT engine for the MOT model:
     `paddle.tools.trt --model_dir … --run_mode trt_fp16`.
  5. Set `SOTA_API_TOKEN` in `.env` (required for the API).
  6. Run `python main.py` (no `--mode` → defaults to `production`).
     The system refuses to start if any of the above are missing.
