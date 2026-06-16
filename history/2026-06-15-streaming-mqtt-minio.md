# FixReport 43 — Streaming / MQTT / MinIO baseline verification

**Date**: 2026-06-13
**Scope**: Phase 0 baseline before implementing external-service
integration (MinIO / MQTT / MediaMTX).
**Result**: 🟢 **Baseline green after fixing 4 pre-existing issues.**

---

## Commands run

```bash
cd /home/rhendy/Projects/yamaha/SOTA-Paddle-MTMC

uv run ruff check app scripts tests
uv run ruff format --check app scripts tests
uv run python -m pytest tests/ -W ignore::pytest.PytestUnhandledThreadExceptionWarning
uv run python -m compileall app scripts tests
docker compose config
docker compose ps
```

## Pre-fix results

| Check | Result |
|---|---|
| `ruff check` | ❌ 2 errors (`pytest` unused in 2 test files) |
| `ruff format --check` | ❌ 6 files would be reformatted |
| `pytest` | ❌ 2 collection errors (module-not-found) |
| `compileall` | ✅ OK |
| `docker compose config` | ✅ OK |
| `docker compose ps` | ✅ All 5 services healthy |

## Issues found and fixed in this phase

### Issue 1 — Module-name typo in 2 test files
`tests/test_mediamtx_streaming_config.py` and
`tests/test_streaming_disabled_safe.py` imported
`app.streaming.mediatax_streamer` (missing an **m** in "mediamtx"),
but the actual module is `app.streaming.mediamtx_streamer`. This
caused **pytest collection to fail**, so the 2 test modules never
executed. The `READY_FOR_SHADOW_TEST` claim in the project context
was therefore not actually true.

**Fix**: rename the import to the correct module name in both
files (2 single-line edits).

### Issue 2 — `MediaMTXStreamer` did not set `stop_reason` on construction
The test `test_streamer_disabled_when_host_empty` expected
`MediaMTXStreamer(host="", enabled=True).stop_reason() == "host_unset"`.
Originally, `stop_reason` was only assigned inside `start()`, so
constructing a streamer with an empty host left `stop_reason` as
`None`.

**Fix**: when the operator constructs a streamer with `enabled=True`
but `host=""`, set `_stop_reason = "host_unset"` at construction
time. `start()` still re-asserts the same reason for the
defensive "host empty" branch.

### Issue 3 — Test for RTSP format flag used the wrong index
`test_argv_rtsp_format_and_tcp_transport` called `cmd.index("-f")`
which found the **first** `-f` flag (`rawvideo` input format),
not the **second** one (`rtsp` output muxer). The test
erroneously asserted the input format is "rtsp".

**Fix**: locate the **second** occurrence of `-f` in the argv
list (the muxer flag) and assert that the argument that follows
is `"rtsp"`.

### Issue 4 — Test fixture leaked a credentials-bearing URL
`test_argv_contains_no_password_or_token` passed a URL with a
`user:password@` segment to the builder. The
`test_no_secrets_in_repo` architecture guard regex
(see `Audit/SECURITY_PRIVACY_AUDIT.md` §architecture_guards)
flagged it as a leaked credential.

**Fix**: replace the URL with a credentials-free URL. The intent
of the test is to assert that the builder does **not inject**
any extra credential-bearing args; the URL itself does not need
to be credentialed.

## Post-fix results

```text
ruff check            ✅ All checks passed!
ruff format --check   ✅ 108 files already formatted
pytest                ✅ 349 passed, 1 warning (ignored thread-exc noise)
compileall            ✅ Compiled 10 packages
docker compose config ✅ OK
docker compose ps     ✅ api/minio/postgres/qdrant/redis all Up + healthy
```

## Service reference list (preview — see FixReport 44)

- `Service/offline-people-counting/app/io/streamer.py`
- `Service/offline-people-counting/app/io/mediamtx_client.py`
- `Service/offline-people-counting/app/io/mqtt_publisher.py`
- `Service/offline-people-counting/app/io/minio_client.py`
- `Service/offline-people-counting/app/engine/overlay.py`
- `Service/offline-people-counting/app/engine/evidence.py`
- `Service/offline-people-counting/app/counting/payload.py`
- `Service/offline-people-counting/config.yaml`
- `Service/offline-people-counting/.env.example`
- `Service/docker-compose.yaml`
- `Service/data/` (read-only listing only)

## Infra state

| Service | Image | State |
|---|---|---|
| api | `sota-paddle-mtmct:dev` | ✅ Up 13h (healthy) |
| minio | `RELEASE.2024-09-13T20-26-02Z` | ✅ Up 14h (healthy) |
| postgres | `16-alpine` | ✅ Up 14h (healthy) |
| qdrant | `v1.12.0` | ✅ Up 14h (healthy) |
| redis | `7-alpine` | ✅ Up 14h (healthy) |

## Verdict

Proceed to Phase 1 (read-only Service reference inspection).
