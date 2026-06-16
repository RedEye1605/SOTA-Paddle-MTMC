# MinIO bucket setup

The SOTA API stores evidence, reports, and (optionally) model
artefacts in **three separate buckets** in the operator's external
MinIO.

## 1. Bucket roles

| Env var | Default | Purpose |
| --- | --- | --- |
| `MINIO_BUCKET_EVIDENCE` | `evidence` | Person crops, debug frames, best.jpg per tracklet |
| `MINIO_BUCKET_REPORTS`  | `reports`  | Benchmark JSON, visualization MP4 sidecars |
| `MINIO_BUCKET_MODELS`   | `models`   | Reserved for model artefact sync (out of scope for Phase 3-9) |

A bucket can be left empty to disable uploads to that bucket
(evidence uploads will fail-fast with a clear error; reports and
models uploads will be skipped).

## 2. Object path scheme

The path scheme matches the upstream Service's pattern but is more
deeply nested so a per-day, per-tracklet listing is straightforward:

```text
evidence/{site_id}/{camera_id}/{zone_id}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet_id}/best.jpg
evidence/{site_id}/{camera_id}/{zone_id}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet_id}/debug_{frame_id:06d}.jpg
visualization/{site_id}/{camera_id}/{yyyy}/{mm}/{dd}/first_3000_frames.mp4
reports/{site_id}/{yyyy}/{mm}/{dd}/benchmark_{timestamp}.json
```

If a tracklet's `global_id` has not been assigned yet, the
`pending_evidence_key` helper routes the upload to:

```text
evidence/pending/{site_id}/{camera_id}/{tracklet_id}/best.jpg
```

The tracklet-collector's `emit_closed_tracklets` later server-side
copies the `best.jpg` to its final dated location once the resolver
assigns a real `global_id`.

## 3. Operator pre-flight

Before booting SOTA in production, the operator must confirm the
three buckets exist and that the API's `MINIO_ACCESS_KEY` /
`MINIO_SECRET_KEY` have at least `s3:GetObject`, `s3:PutObject`, and
`s3:ListBucket` on each bucket.

```bash
# Verify the bucket exists (using mc or the AWS CLI)
mc alias set prod https://minio.example.invalid <access> <secret>
mc ls prod/evidence
mc ls prod/reports
mc ls prod/models
```

If a bucket is missing, either create it manually:

```bash
mc mb prod/evidence
mc mb prod/reports
mc mb prod/models
```

or, on the dev host only, set `MINIO_CREATE_BUCKETS=true` in
`.env` to let the API lazily create missing buckets on connect. Do
NOT enable this in production unless the policy explicitly allows
it.

## 4. Configuration matrix

| Source | Bucket | Path | Caller |
| --- | --- | --- | --- |
| `TrackletCollector.on_frame` | `MINIO_BUCKET_EVIDENCE` | `evidence/...` | debug crops + best.jpg |
| `TrackletCollector.emit_closed_tracklets` | `MINIO_BUCKET_EVIDENCE` | `evidence/...` (final) | re-key after resolver |
| `scripts/generate_visual_validation.py` | `MINIO_BUCKET_REPORTS` | `visualization/...` | annotated MP4 sidecar |
| `scripts/benchmark_t4.py` (Phase 10) | `MINIO_BUCKET_REPORTS` | `reports/...` | benchmark JSON |
| future: `scripts/download_pphuman_models.sh` | `MINIO_BUCKET_MODELS` | `models/...` | model artefact sync |

## 5. Failure modes

| Failure | Behaviour |
| --- | --- |
| Bucket does not exist, `MINIO_CREATE_BUCKETS=true` | API lazily creates it on first use |
| Bucket does not exist, `MINIO_CREATE_BUCKETS=false` | First `put_*` call raises a clear error; the engine logs `minio evidence upload failed: bucket missing` |
| Network blip | The existing `MinioStore` retries via the official `minio` SDK; tracklets stay in PostgreSQL even if the upload fails |
| Wrong credentials | `connect()` raises; `is_available` is `False`; the engine continues but evidence is dropped with a warning |

## 6. Quick start (operator)

```bash
# 1. Confirm the three buckets exist
mc ls prod/evidence prod/reports prod/models

# 2. Set the env vars in .env
cat >> .env <<'EOF'
MINIO_BUCKET_EVIDENCE=evidence
MINIO_BUCKET_REPORTS=reports
MINIO_BUCKET_MODELS=models
EOF

# 3. Run the readiness preflight
uv run python scripts/readiness_preflight.py
```

## 7. References

- `app/storage/minio_store.py` — `MinioStore` with `evidence_key()`,
  `pending_evidence_key()`, `visualization_key()`, `report_key()`.
- `app/workers/tracklet_collector.py` — call sites for evidence
  upload.
- `scripts/generate_visual_validation.py` — call site for the
  visualization upload.
- `scripts/benchmark_t4.py` — call site for the benchmark report
  upload.
- `Docs/external_services_setup.md` — cross-service overview.
