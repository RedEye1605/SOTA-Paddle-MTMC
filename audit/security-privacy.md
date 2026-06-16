# Security & Privacy Audit — SOTA-Paddle-MTMC

> **Phase 8 — Security & privacy audit.** CCTV people-tracking
> system; scope is limited to people-counting/ReID analytics.
> Forbidden features (face recognition, demographic inference,
> emotion, naming) are explicitly out of scope.

## Forbidden features — verified absent

| Feature | Searched? | Found? | Evidence |
|---|---|---|---|
| Face recognition | `face`, `landmark`, `embedding` (in face context) | No | No `facenet`, `arcface`, `InsightFace`, `dlib`, or `face_recognition` import. No "face" string in any config or doc. ✅ |
| Name / person identification | `name=`, `person_name` | No | The `global_id` is `GID-{8hex}-{cam_short}` — pseudonymous. No first/last name fields anywhere. ✅ |
| Demographic inference (age/gender) | `age`, `gender`, `demographic` | No | The PP-Human `ATTR` block in `infer_cfg_pphuman_sota.yml` is enabled (Paddle's default). The `ATTR` block does pedestrian attribute recognition (clothing color, orientation), NOT age/gender, but **the architecture does run the StrongBaseline attribute head** when wired. The Paddle `ATTR` model (`strongbaseline_r50_30e_pa100k`) is a 26-attribute classifier trained on PA100K; it does not include age or gender. ✅ |
| Emotion detection | `emotion`, `affect` | No | The `SKELETON_ACTION` block is `enable: False` in the infer_cfg. ✅ |
| Identity naming | `name=`, `display_name`, `label=` | No | ✅ |
| Biometric identification beyond ReID | `face_recognition`, `fingerprint` | No | ✅ |

**Verdict on forbidden features:** all absent. The Paddle
StrongBaseline attribute head is a *clothing/attribute*
classifier, not a biometric. ✅

## ReID identity is pseudonymous

- `global_id = GID-{8hex}-{cam_short}` — random UUID.
- No mapping from `global_id` to any PII.
- No database table for `name ↔ global_id`.
- `tracklet_embeddings` stores only `tracklet_id, global_id,
  camera_id, model_name, model_version, vector_db_*,
  quality_score, created_at` — no PII.
- `identity_decisions` stores only decisions and scores.

✅ Pseudonymous identity, no PII stored.

## 24h retention

- **PostgreSQL:** `expire_old_identities(older_than_seconds)`
  exists but is not called by any code path. No scheduled
  job. **Effectively no retention.**
- **Qdrant:** No `delete_by_time` or `delete_by_id` method.
  No payload-based filter for cleanup. Points live forever.
- **MinIO:** No lifecycle policy. Evidence crops grow
  forever.
- **Redis:** `ttl_recent_identity = 86400` ✅ — the
  `recent:global:{id}` key expires.

**Verdict on retention:** only the Redis cache is TTL'd.
PostgreSQL, Qdrant, and MinIO retain indefinitely.
**Production-unsafe for a privacy-conscious deployment.**

Required fix: add a retention worker that calls
`expire_old_identities`, deletes Qdrant points older than N
days, and applies a MinIO lifecycle policy.

## Secret / credential leaks

### `.env` (example)

The `.env.example` file uses `change_me_in_production`
placeholders for:
- `POSTGRES_PASSWORD`
- `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`
- `MQTT_PASSWORD`

✅ Placeholders, not real values.

### `.env` (actual)

`cp .env.example .env` was required for `docker compose config`
to parse. The actual `.env` (not in repo) is what the
operator fills in.

### `docker-compose.yaml`

Same `change_me_in_production` placeholders, with
`${POSTGRES_PASSWORD:-change_me_in_production}` defaults.
If the operator forgets to set `.env`, the system starts
with the default password. **Fail-closed would refuse
to start.** This is fail-open.

### `Dockerfile`

`test_dockerfile_does_not_commit_secrets` asserts that
`change_me_in_production` is not in the Dockerfile. ✅
No `ENV` lines with secrets.

### RTSP passwords in `cameras.yaml` / `db/seed/cameras.sample.sql`

The RTSP URL is read from an env var named in
`rtsp_url_env_key` (e.g. `CAM_01_RTSP_URL`). The URL itself
is never in the YAML or the seed SQL. ✅

### Logs

- `app/workers/reid_worker.py` does not log the
  `crop_uri` (it includes the S3 path, no auth). ✅
- `app/workers/multi_camera_runner.py:48` does not log
  the source URL. ✅
- `app/main.py` does not log the
  `POSTGRES_PASSWORD` or `MINIO_SECRET_KEY`. ✅
- `app/telemetry/mqtt_client.py:71-72` publishes
  payload over TLS (if `tls_enabled`). The password
  is set via `username_pw_set` and is not logged. ✅

### Token / API key leaks

No `AKIA*` (AWS) or `ghp_*` (GitHub) or `sk-*` (OpenAI)
strings found in any tracked file.

## `.gitignore`

A `.gitignore` is missing from this directory. The
`scripts/init_qdrant.py` and other generated artifacts
could be committed accidentally. (Not checked in this
audit but worth flagging.)

## `architecture_guards/test_no_secrets_in_repo`

```python
SECRET_PATTERNS = [
    (r"password\s*=\s*['\"]\w{6,}",                 "password literal"),
    (r"rtsp://" + "[^/'\"]+:" + "[^@'\"]+@",         "RTSP URL with credentials"),
    (r"(?i)AKIA[0-9A-Z]{16}",                        "AWS access key id"),
    (r"(?i)aws_secret_access_key\s*=",              "AWS secret key"),
    (r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",  "private key"),
]
```

The regex `r"password\s*=\s*['\"]\w{6,}"` does not catch
`password=...` in YAML (e.g. `postgres_password: foo`).
**The scanner is partial.** ⚠️

## Image-content sensitivity

- The system uploads JPEG crops to MinIO. These are
  potentially-identifying images of people.
- The `evidence` bucket has no server-side encryption
  configuration (no `SSE-S3`, no `SSE-KMS`).
- The Qdrant embedding vectors do NOT allow reverse image
  generation, but combined with metadata
  (`camera_id, timestamp, quality_score`) they can be
  cross-referenced with the original MinIO crop.
- **No access control on `/identity/{global_id}` and
  `/identity/decisions` endpoints.** Anyone who can reach
  port 8000 can query. No auth middleware. ⚠️

## Network exposure

- `docker-compose.yaml` exposes `5432`, `6333`, `9000`,
  `9001` (MinIO console), `8000` (API) to `0.0.0.0`.
- In production, these should be on an internal network
  only. The compose file does not constrain them.
  **No `networks: internal:`** ⚠️

## Authentication / Authorization

- The `api` has **no authentication**. All endpoints are
  open.
- `mqtt_publisher.username_pw_set` uses
  `${MQTT_USERNAME}` / `${MQTT_PASSWORD}` from env. Good.
- PostgreSQL uses `${POSTGRES_USER}` / `${POSTGRES_PASSWORD}`.
  Good.
- MinIO uses `${MINIO_ACCESS_KEY}` / `${MINIO_SECRET_KEY}`.
  Good.

## TransReID weight loading safety

`TransReIDAdapter._try_load`:
```python
ckpt = torch.load(self._weight_path, map_location="cpu", weights_only=self._weights_only)
```

`weights_only=True` is set by default at three levels:
- Constructor default: `weights_only: bool = True` (line 30)
- Stored on self: `self._weights_only = weights_only` (line 34)
- Config: `weights_only: true` in `configs/reid/transreid.yaml:39`
- Documented in module docstring (line 6) and inline comment
  (line 56)

**Verified via literal grep**: zero `paddle.load`, `pickle.load`,
`paddle.jit.load`, or `weights_only=False` strings exist
anywhere in the project. ✅ This correctly refuses arbitrary
pickle execution. (See also BUG-034: the on-disk weight is the
MSMT17 checkpoint while the config points to the Market-1501
path; the weights_only flag still applies, but the wrong
weights will fail to load with a `size mismatch` rather than
a silent RCE — an acceptable fail-mode for now.)

## Paddle `paddlepaddle` is not in `requirements.txt`

This is more of a deploy issue than a security issue, but
note: if a real attacker added a malicious Paddle model
under `/models/pphuman/`, the system would load it without
verification. **No model signature / hash check.** ⚠️

## Findings

| # | Finding | Severity |
|---|---|---|
| S1 | No FastAPI auth on identity endpoints | HIGH |
| S2 | No model signature/hash check | MEDIUM |
| S3 | Compose exposes all infra ports to 0.0.0.0 | MEDIUM |
| S4 | MinIO evidence has no SSE | MEDIUM |
| S5 | No retention policy for evidence crops, Qdrant, PG | HIGH (privacy) |
| S6 | `POSTGRES_PASSWORD`, `MINIO_*_KEY`, `MQTT_PASSWORD` default to `change_me_in_production` if `.env` not set | HIGH |
| S7 | Secret scanner regex misses YAML key:value form | LOW |
| S8 | `.gitignore` is missing | LOW |
| S9 | `architecture_guards` does not enforce the "no face recognition" rule | LOW (out of scope, but worth adding a guard) |
| S10 | The Paddle `ATTR` block in `infer_cfg_pphuman_sota.yml` is `enable: True` — Paddle's StrongBaseline attribute head is a 26-attribute classifier (clothing color, orientation, accessories), not demographic. ✅ Acceptable. | NONE |

## Privacy verdict

**Privacy posture is acceptable for the intended use case
(people-counting with pseudonymous ReID) IF retention
policies are added before deployment.**

The biggest gaps are:
1. No FastAPI auth → anyone on the network can query
   global_id → tracklet → evidence chain.
2. No retention → CCTV data grows forever.
3. MinIO evidence is unencrypted at rest.

These are fixable but require work before production.

## Security verdict

**Security posture is acceptable for a private internal
network. Not acceptable for any internet-facing
deployment.** The missing FastAPI auth is the most
urgent fix.
