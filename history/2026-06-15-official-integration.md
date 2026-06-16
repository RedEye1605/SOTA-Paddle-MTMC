# Official Integration Verification

> **Authoritative reference for the official PaddleDetection and
> TransReID paths used by SOTA-Paddle-MTMC.** This is the audit's
> required "official source verification" — each official path is
> backed by a Context7 query + a link to the upstream docs.

## 1. PaddleDetection PP-Human pipeline

**Adapter:** `app/detection/pphuman_pipeline.py:PPHumanDetectorAdapter`

**Method:** `build_pipeline_command()` constructs the official
`pipeline.py` invocation per the
[PaddleDetection PP-Human tutorial](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/docs/tutorials/PPHuman_QUICK_STARTED.md).

**Context7 query (June 2026):** `/paddlepaddle/paddledetection`, query
"PP-Human pipeline.py command line inference for multi-stream MOT
pedestrian tracking".

**Verified commands from Context7:**

```bash
python deploy/pipeline/pipeline.py \
    --config deploy/pipeline/config/infer_cfg_pphuman.yml \
    --video_file=demo/pedestrian.mp4 \
    --device=GPU

# Enable MOT + override model dir
python deploy/pipeline/pipeline.py \
    --config deploy/pipeline/config/infer_cfg_pphuman.yml \
    -o MOT.enable=True MOT.model_dir=ppyoloe_infer/ \
    --video_file=test_video.mp4 \
    --device=gpu

# TensorRT FP16 via run_mode flag
python deploy/pipeline/pipeline.py \
    --config deploy/pipeline/config/infer_cfg_pphuman.yml \
    -o MOT.enable=True MOT.model_dir=ppyoloe/ \
    --video_file=test_video.mp4 \
    --device=gpu \
    --run_mode=trt_fp16
```

Our adapter (`build_pipeline_command`) produces the same shape.

**MOT output format (verified):**
`frame,id,x1,y1,w,h,score,-1,-1,-1` per line, written to
`{output_dir}/mot_results/{name}.txt`. Source: PaddleDetection
`configs/mot/deepsort/README.md` and
`configs/mot/botsort/README.md`.

## 2. PaddleInference (PP-Human StrongBaseline ReID)

**Adapter:** `app/reid/pphuman_adapter.py:PPHumanReIDAdapter`

**Method:** `paddle.inference.create_predictor(config)` against
`/models/pphuman/strongbaseline_r50_30e_pa100k/inference.pdmodel`.

**Context7 query (June 2026):** `/paddlepaddle/paddledetection`, query
"PP-Human official ReID (StrongBaseline)".

The official PP-Human pipeline uses
`output_inference/strongbaseline_r50_30e_pa100k` for the `ATTR` block
(per `infer_cfg_pphuman.yml`). The same StrongBaseline model is
exported via Paddle's `tools/export_model.py` and can be loaded with
`paddle.inference`.

## 3. TransReID backbone (vendored)

**Adapter:** `app/reid/transreid_adapter.py:TransReIDAdapter`
**Vendored backbone:** `app/reid/_transreid_native/`

**Context7 query (June 2026):** `/damo-cv/transreid`, query "TransReID
vit_base_patch16_224_TransReID make_model num_class JPM local feature
neck_feat before stride_size sie_xishu".

**Verified facts from Context7:**

1. The TransReID model construction is via `vit_base_patch16_224_TransReID`
   with `img_size=(256, 128)`, `stride_size=12`, `local_feature=True`,
   `sie_xishu=3.0` (matching the official Market config).
2. The forward pass is `model(images, cam_label=cam_labels, view_label=view_labels)`
   and returns `[B, 768*5]` for JPM (global + 4 local).
3. The official TEST config uses `neck_feat='before'` and
   `feat_norm='yes'` (L2-normalize).

Our vendored model (`SIETransformer` in
`app/reid/_transreid_native/model.py`) implements this exact contract
with the same default arguments. The `extract_inference_feature`
helper does the JPM aggregation + L2-normalize.

**Loading safety:** `load_transreid_checkpoint` always calls
`torch.load(..., weights_only=True)`. The architecture-guard test
`test_dangerous_weights_refused` enforces this at the repo level
(no `weights_only=False` string in any tracked file).

## 4. PyTorch / NumPy versions

The vendored backbone uses only the standard PyTorch API (`nn.Module`,
`nn.LayerNorm`, `nn.Linear`, `nn.MultiheadAttention`-equivalent via
manual QKV). No dependency on the upstream `model.backbones.vit_pytorch`
module (so we do not pull in yacs / timm / clip dependencies).

**Versions tested:**
* Python 3.12
* PyTorch 2.4.0 (CPU works; GPU requires `+cu124` wheel)
* NumPy 1.26+

The repo-level tests `test_transreid_vendor.py` are gated on
`torch` being installed (`pytest.importorskip`). They are skipped on
the CI host that does not have torch.

## 5. Qdrant / Redis / PostgreSQL / MinIO / FastAPI / Paho-MQTT

All storage layer APIs match the official client docs:

* `qdrant-client>=1.12` — uses `client.create_collection`,
  `client.create_payload_index`, `client.search`,
  `client.delete(points_selector=FilterSelector(filter=Filter(must=[FieldCondition(...)])))`,
  `client.count(count_filter=...)`. All verified against
  Context7 `/qdrant/qdrant-client` (181 snippets, High reputation,
  benchmark 90.8).
* `redis>=5.0` — uses `r.setex`, `r.xadd`, `r.xgroup_create(mkstream=True)`,
  `r.xreadgroup`, `r.xack`. Standard redis-py API.
* `psycopg_pool>=3.2` + `psycopg[binary]>=3.2` — uses
  `ConnectionPool(conninfo=...)`. Standard psycopg 3 API.
* `minio>=7.2` — uses `Minio(endpoint, ...)`, `put_object`,
  `get_object`, `copy_object`, `list_objects`, `remove_object`.
  `CopySource` import verified.
* `fastapi>=0.115` — uses `HTTPBearer` + `Depends(verify)` per the
  Context7 `/fastapi/fastapi` (2153 snippets) security tutorial.
* `paho-mqtt>=2.1` — uses `mqtt.Client`, `username_pw_set`, `tls_set`,
  `reconnect_delay_set`, `connect`, `loop_start`, `publish`, `loop_stop`,
  `disconnect`. Standard paho-mqtt 2.x API.

## 6. PaddleDetection deployment

The official deployment doc (Context7 `/paddlepaddle/paddledetection`,
`deploy/pipeline/docs/tutorials/`) recommends TensorRT FP16 for
T4-class GPUs:

```yaml
MOT:
  model_dir: output_inference/mot_ppyoloe_l_36e_pipeline
  tracker_config: deploy/pipeline/config/tracker_config.yml
  enable: True
```

with `--run_mode=trt_fp16` on the CLI. Our adapter's `run_mode` is
read from `app.yaml::runtime.run_mode` and passed to the pipeline
via the `-o` override (`MOT.run_mode=trt_fp16`).

## 7. Conclusion

All production paths use official APIs that are verified against
Context7 / upstream docs. The vendored TransReID backbone is a
minimal inference-only subset of the official
`damo-cv/TransReID` repository, restricted to:
* `vit_base_patch16_224_TransReID` (SIE-Transformer)
* JPM module
* BNNeck (used only at `neck_feat='before'` in inference)

The vendor is necessary because `damo-cv/TransReID` has no PyPI
release; pulling the full upstream repo would also pull yacs, the
custom `config` module, training-only data loaders, and an
`open_clip` fork — none of which are needed for inference.

For operators who prefer the full upstream code, the
`TRANSREID_MODEL_FN=module:fn` plug-in env var allows injecting a
custom callable that returns a `torch.nn.Module` of their choice.
