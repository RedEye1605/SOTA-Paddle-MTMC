# Evidence Re-key

> **Operator runbook for PATCH-029.**
> How a tracklet's best crop moves from the *pending* path
> to the final dated path after the resolver assigns a
> global_id.

## TL;DR

When a tracklet is closed, the collector uploads the best
crop to `evidence/pending/{site}/{camera}/{tracklet}/best.jpg`
with `global_id="UNASSIGNED"`. The resolver then assigns a
global_id. The `EvidenceRekeyWorker` consumes
`stream:identity_decisions` and server-side copies the
pending crop to
`evidence/{site}/{camera}/{zone}/{yyyy}/{mm}/{dd}/{global_id}/{tracklet}/best.jpg`,
then updates `tracklets.best_crop_uri`. The pending copy
is deleted by default.

## Flow

```
Collector              Resolver              RekeyWorker           MinIO
   │                      │                      │                   │
   │ upload to pending/   │                      │                   │
   ├─────────────────────►│                      │                   │
   │                      │ resolve → new gid   │                   │
   │                      │ (publishes to       │                   │
   │                      │  stream:identity_   │                   │
   │                      │  decisions)         │                   │
   │                      ├─────────────────────►                   │
   │                      │                      │ consume           │
   │                      │                      │ copy pending→final│
   │                      │                      ├──────────────────►
   │                      │                      │ delete pending    │
   │                      │                      ├──────────────────►
   │                      │                      │ UPDATE best_crop_uri
   │                      │                      │ (in PG)           │
```

## Configuration

`configs/benchmark.yaml`:

```yaml
evidence:
  rekey_after_global_id: true
  keep_pending_copy: false
  rekey_retry_max: 3
  pending_prefix: "evidence/pending"
```

* `rekey_after_global_id`: enable the re-key worker
  entirely.
* `keep_pending_copy`: if true, the pending copy is kept
  for audit / debug. If false (default), it is deleted
  after a successful re-key.
* `rekey_retry_max`: number of retry attempts on transient
  copy failures.

## Failure handling

* A copy failure is retried up to `rekey_retry_max` times
  with linear backoff (0.5 s, 1.0 s, 1.5 s).
* If all retries fail, the worker logs an ERROR with the
  tracklet_id. The pending crop is left untouched. The
  next operator-driven `python -m scripts.ev_rekey_replay
  <tracklet_id>` re-attempts.
* The worker never blocks the resolver — the resolver hot
  path completes as soon as it has published the
  decision, and the re-key happens asynchronously.

## Tests

* `tests/test_evidence_rekey.py` covers the pending path,
  the copy, the delete, the retry, the disable flag, and
  the failure-doesn't-delete-pending case.
