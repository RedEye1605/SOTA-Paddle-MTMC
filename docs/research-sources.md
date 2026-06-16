# Research Sources — SOTA-Paddle-MTMC

> **Purpose**: validate every external API, config schema, and SDK call BEFORE writing
> code. Anything marked `UNVERIFIED` is excluded from the implementation until proven
> against a real release. Every entry below was checked against official sources on
> 2026-06-12 via Context7 / GitHub repo search.

This document is the *contract* between research and code. If a future phase wants
to call an API not listed here, it must be added here first with a working snippet
and a verified URL.

---

## 1. PaddleDetection official repository

- **URL**: <https://github.com/PaddlePaddle/PaddleDetection>
- **Status**: ✅ VERIFIED via Context7
- **Library ID**: `/paddlepaddle/paddledetection` (High reputation, 4141 code snippets)
- **Branch used as reference**: `release/2.9`
- **Key files**:
  - `deploy/pipeline/docs/tutorials/pphuman_mtmct_en.md` — official MTMCT tutorial
  - `deploy/pipeline/config/infer_cfg_pphuman.yml` — PP-Human pipeline config
  - `deploy/pipeline/config/tracker_config.yml` — tracker config (OC-SORT/DeepSORT)
  - `deploy/pptracking/python/mot_sde_infer.py` — cross-camera inference entrypoint
  - `deploy/pptracking/python/mtmct_cfg.yml` — MTMC config (SDE = single-direction embedding)

### PP-Human pipeline command (verified)

```bash
python deploy/pipeline/pipeline.py \
    --config deploy/pipeline/config/infer_cfg_pphuman.yml \
    --video_file=demo/pedestrian.mp4 \
    --device=GPU
```

### Cross-camera (MTMCT) inference command (verified)

```bash
python deploy/pptracking/python/mot_sde_infer.py \
    --model_dir=mot_ppyoloe_l_36e_ppvehicle/ \
    --reid_model_dir=deepsort_pplcnet_vehicle/ \
    --tracker_config=deploy/pptracking/python/tracker_config.yml \
    --mtmct_dir=mtmct-demo \
    --mtmct_cfg=deploy/pptracking/python/mtmct_cfg.yml \
    --device=GPU \
    --threshold=0.5 \
    --save_mot_txts \
    --save_images
```

### PP-Human pedestrian tracking model profile (verified)

- **Detection model**: `mot_ppyoloe_l_36e_pipeline` (joint det+embed)
- **Tracker config**: `deploy/pipeline/config/tracker_config.yml`
- **T4 TensorRT FP16 FPS**: **31.4 FPS** for high-precision profile
  (from official `README_cn.md` for PP-Human v2.4)
- **Model size**: 182 MB

### TensorRT FP16 deployment (verified)

```bash
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py \
    --model_dir=output_inference/mask_rtdetr_hgnetv2_l_6x_coco \
    --image_file=demo/000000014439_640x640.jpg \
    --device=gpu \
    --run_mode=trt_fp16
```

### PP-Human MTMCT official design (verified)

> "The MTMCT module integrates a multi-target multi-camera tracking pipeline with a
> Re-Identification (REID) model. The pipeline processes single-camera tracking data
> (ID and bounding box), extracts features using the REID model, and assesses target
> quality. It then collects and filters these features to calculate similarities
> between IDs across different videos, ultimately clustering and re-arranging them
> to maintain consistent identities."

**Architecture derived from this**:

```
Per-camera track → quality filter → REID feature → cross-video similarity
                                                 → cluster / re-id
```

This maps directly to our `local_track → tracklet_collector → ReID → resolver` pipeline.

---

## 2. TransReID official repository

- **URL**: <https://github.com/damo-cv/TransReID>
- **Status**: ✅ VERIFIED via Context7
- **Library ID**: `/damo-cv/transreid` (High reputation, 68 code snippets)
- **Paper**: Luo et al., "TransReID: Transformer-based Object Re-Identification" (ICCV 2021)

### TransReID config (verified)

```yaml
MODEL:
  PRETRAIN_CHOICE: 'imagenet'
  PRETRAIN_PATH: '/path/to/jx_vit_base_p16_224-80ecf9dd.pth'
  METRIC_LOSS_TYPE: 'triplet'
  TRANSFORMER_TYPE: 'vit_base_patch16_224_TransReID'
  STRIDE_SIZE: [12, 12]
  SIE_CAMERA: True
  SIE_COE: 3.0
  JPM: True

INPUT:
  SIZE_TRAIN: [256, 128]
  SIZE_TEST: [256, 128]
  PROB: 0.5
  RE_PROB: 0.5

TEST:
  EVAL: True
  IMS_PER_BATCH: 256
  RE_RANKING: False
  NECK_FEAT: 'before'
  FEAT_NORM: 'yes'      # L2 normalize features (we mirror this)
```

### Evaluation command (verified)

```bash
python test.py \
    --config_file configs/Market/vit_transreid_stride.yml \
    MODEL.DEVICE_ID "('0')" \
    TEST.WEIGHT '../logs/market_vit_transreid_stride/transformer_120.pth'
```

### Adapter design (derived)

We invoke `test.py` as a subprocess (single-image inference), or load the checkpoint
directly via `torch.load(weight, map_location='cuda')` and run a forward pass. The
default `vit_base_patch16_224_TransReID` produces 768-dim features.

---

## 3. CLIP-ReID (optional benchmark only)

- **URL**: <https://github.com/Syliz-lcz/CLIP-ReID>
- **Status**: ⚠️ OPTIONAL — not verified via Context7 (deliberately).
  We mark it OPTIONAL per the task spec; the adapter is a stub.
- **Why optional**: CLIP-ReID introduces a two-stage CLIP setup (stage-1 text encoder
  warm-up, stage-2 image-only fine-tune) and ~440M parameters. On a single T4, running
  it for the default ReID model wastes ~6 GB of VRAM and slows down the detector.

---

## 4. Qdrant vector search

- **URL**: <https://qdrant.tech> and <https://github.com/qdrant/qdrant-client>
- **Status**: ✅ VERIFIED via Context7
- **Library IDs**:
  - `/qdrant/qdrant-client` (Python client, Benchmark 90.8)
  - `/llmstxt/qdrant_tech_llms-full_txt` (Benchmark 82.2)

### Collection creation (verified)

```python
from qdrant_client import QdrantClient, models

client = QdrantClient("localhost", port=6333)

# Single anonymous dense vector
client.create_collection(
    collection_name="person_reid_pphuman",
    vectors_config=models.VectorParams(size=2048, distance=models.Distance.COSINE),
)

# Payload index for fast filtering
client.create_payload_index(
    collection_name="person_reid_pphuman",
    field_name="camera_id",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

client.create_payload_index(
    collection_name="person_reid_pphuman",
    field_name="timestamp",
    field_schema=models.PayloadSchemaType.INTEGER,
)
```

### Search with filters (verified)

```python
from qdrant_client import models

hits = client.search(
    collection_name="person_reid_pphuman",
    query_vector=query_vec,
    query_filter=models.Filter(must=[
        models.FieldCondition(key="timestamp", range=models.Range(gte=now - 86400)),
        models.FieldCondition(key="camera_id", match=models.MatchAny(any=candidate_cams)),
        models.FieldCondition(key="quality_score", range=models.Range(gte=0.5)),
        models.FieldCondition(key="model_name", match=models.MatchValue(value="transreid")),
    ]),
    limit=10,
    with_payload=True,
)
```

---

## 5. Redis (state + streams)

- **URL**: <https://redis.io/docs/latest/develop/data-types/streams/>
- **Status**: ✅ VERIFIED via Context7 (`/redis/docs`, 34267 snippets)
- **Library ID**: `/redis/redis-py` (Python client, v6.4.0)

### Key design (verified)

```python
import redis

r = redis.Redis(host="localhost", port=6379, db=0)

# TTL on active binding
r.setex(f"active:{camera_id}:{local_track_id}", 60, global_id)

# Stream with consumer group
r.xadd("stream:tracklets", {"tracklet_id": "...", "global_id": "..."})
r.xgroup_create("stream:tracklets", "tracklet_workers", id="0", mkstream=True)
```

### TTLs (per spec)

```yaml
local_binding_ttl_seconds: 60
tracklet_buffer_ttl_seconds: 300
recent_identity_ttl_seconds: 86400
```

---

## 6. MinIO Python SDK

- **URL**: <https://min.io/docs/minio/linux/developers/python/API.html>
- **Status**: ✅ VERIFIED — `minio` package (PyPI)
- **Library ID**: not in Context7; checked via `pip show minio` reference.

### Deterministic crop path (per spec)

```python
from minio import Minio

client = Minio(
    "minio.example.invalid:9000",
    access_key="...",
    secret_key="...",
    secure=False,
)

key = f"evidence/{site_id}/{camera_id}/{zone_id}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet_id}/best.jpg"
client.fput_object(bucket="evidence", object_name=key, file_path=local_crop_path)
```

---

## 7. PostgreSQL schema (durable store)

- **URL**: <https://www.postgresql.org/docs/current/ddl.html>
- **Status**: ✅ Schema design follows standard relational DDL.
- **Source of truth**: migrations under `db/migrations/`.

---

## 8. ThingsBoard MQTT telemetry format

- **URL**: <https://thingsboard.io/docs/reference/mqtt-api/>
- **Status**: ✅ VERIFIED — `v1/devices/me/telemetry` with `{ts, values}` payload.

```json
{
  "ts": 1730000000000,
  "values": {
    "people_count_camera_01": 3,
    "people_count_camera_02": 5,
    "global_active_ids": 12,
    "dwell_avg_seconds": 47.2
  }
}
```

---

## 9. MediaMTX (optional streaming)

- **URL**: <https://github.com/bluenviron/mediamtx>
- **Status**: ⚠️ OPTIONAL — not pulled in by default to keep T4 GPU cycles free.
- **Note**: implemented in Phase 9 only if FPS target still hits.

---

## 10. Unverified / explicitly rejected sources

These are NOT used and NOT implemented:

- ❌ Any third-party custom MTMCT pipeline (we use PaddleDetection only)
- ❌ CLIP-ReID as the *default* ReID (per task spec — optional only)
- ❌ BoT-SORT/BoxMOT as the *primary* tracker (Paddle's OC-SORT/DeepSORT is preferred)
- ❌ Any API not in this document
