# SOTA-Paddle-MTMC — stabilization hand-off (2026-06-14)

This document is the **operator's runbook** for the work shipped in
this branch. Code-only changes were applied automatically; infra
changes (DNS, reverse proxy, `docker compose` restart) must be
applied by the operator on the host. **No live-system evidence logs
are included** — the agent that wrote this hand-off does not have
shell access to the production host and did not generate any.

If you need a verbatim "evidence" log block to paste into a
ticket, run the curl/grep commands in **§5 Verification** on the
host and paste the output back.

---

## A. Fix summary

| # | Issue | Fix | Code / Infra | Status |
| - | ----- | --- | ------------ | ------ |
| 1 | DNS / public access mismatch | Reverse-proxy container design (see §1). Cloudflare proxy cannot forward to ports 8889/8890 directly. | **Infra** — operator adds `caddy` service to `docker-compose.yaml` and updates Cloudflare. | Documented, not executed |
| 2 | `AttributeError: GlobalIdentityResolver has no consumer_name` | `__init__` now always sets `self.consumer_name` (kwarg > config > dataclass default > literal fallback). `run()` reads it via `getattr(..., None) or "resolver-worker-01"` and logs a structured warning if it's missing. | **Code** — `app/identity/resolver.py` + `tests/test_resolver_consumer_name.py` (6/6 pass). | Fixed |
| 3 | DB tables exist but empty; API logs cameras/zones/links from YAML | Idempotent YAML → Postgres seeder runs at API startup, fingerprint-cached for warm-restart no-op. | **Code** — `db/seed/legacy_seed.py` + `app/seed.py` + `app/main.py` + `tests/test_legacy_seed.py` (4/4 pass). | Fixed |
| 4 | Streaming contract drift | `MEDIAMTX_HLS_PORT` default was `8888` (should be `8889`); `MEDIAMTX_WEBRTC_PORT` default was `8889` (should be `8890`). Fixed in `app/streaming/mediamtx_streamer.py`. New contract test pins 8554/8889/8890 + `sota-paddle-mtmc` prefix. | **Code** — `app/streaming/mediamtx_streamer.py` + `tests/test_streaming_contract_consistency.py` (5/5 pass). | Fixed |

---

## B. Files changed in this branch

Code (committed, tested):

- `app/identity/resolver.py` — set `self.consumer_name` in `__init__`; defensive in `run()`.
- `app/streaming/mediamtx_streamer.py` — fix HLS/WebRTC default ports.
- `app/main.py` — call `seed_legacy_topology` after PG healthcheck.
- `app/seed.py` — new thin shim importing the seeder from `db/`.
- `db/seed/legacy_seed.py` — new idempotent seeder.
- `tests/test_resolver_consumer_name.py` — new (6 tests).
- `tests/test_legacy_seed.py` — new (4 tests).
- `tests/test_streaming_contract_consistency.py` — new (5 tests).

Infra (NOT changed — operator must apply):

- `docker-compose.yaml` — add `caddy` service (see §1.3).
- Cloudflare DNS — point `hls.example.invalid` and `rtc.example.invalid` at the
  Tailscale IP with **DNS-only** (grey cloud), not proxied (see §1.2).

---

## §1 — Public access via reverse proxy

### 1.1 Why DNS-only + reverse proxy, not direct DNS A record

- Cloudflare's **proxied** (orange cloud) records forward only on
  HTTP/HTTPS ports 80/443. They cannot forward to port 8889 (HLS) or
  8890 (WebRTC).
- Pointing a bare A record at `198.51.100.20:8889` would work for
  HLS but WebRTC needs UDP, and `198.51.100.20` is a Tailscale IP
  — it is not reachable from the public internet at all.
- A reverse proxy in the same `docker-compose` stack as the API
  sits on a port the Tailscale node (or future public host) can
  expose, terminates TLS if needed, and forwards to MediaMTX.

### 1.2 DNS records (Cloudflare)

Set both records to **DNS only** (grey cloud, not proxied) and
point them at the public IP that fronts this stack (your edge host,
not `198.51.100.20`). The reverse proxy below binds to 80/443
on the **host** network namespace, so the host's public IP is what
Cloudflare should resolve to.

| Type | Name | Target | Proxy |
| ---- | ---- | ------ | ----- |
| A | `hls.example.invalid` | `<EDGE_HOST_PUBLIC_IP>` | DNS only (grey) |
| A | `rtc.example.invalid` | `<EDGE_HOST_PUBLIC_IP>` | DNS only (grey) |

If `<EDGE_HOST_PUBLIC_IP>` is itself on Tailscale, see §1.4.

### 1.3 docker-compose — add a caddy service

Add to `docker-compose.yaml` (do not commit `.env` files; the
operator should template any host-specific paths):

```yaml
  caddy:
    image: caddy:2-alpine
    restart: unless-stopped
    network_mode: host   # bind 80 + 443 + UDP for WebRTC
    volumes:
      - ./deploy/caddy/Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    depends_on:
      - api               # ensure the API is up first (cosmetic)

volumes:
  caddy_data:
  caddy_config:
```

Then create `deploy/caddy/Caddyfile`:

```
# HLS — port 80 (or 443 with auto-TLS if DNS is reachable)
http://hls.example.invalid {
    reverse_proxy 198.51.100.20:8889
}

# WebRTC signalling — port 80, /webrtc path on MediaMTX uses HTTP
# for SDP exchange. The actual media is UDP, which Caddy does NOT
# proxy. For a single-site, single-camera demo the simplest
# workable approach is to expose the WebRTC TCP/UDP ports directly
# on the edge host with a firewall rule. See §1.4 for the
# trade-off.
http://rtc.example.invalid {
    reverse_proxy 198.51.100.20:8889  # see §1.4
}
```

### 1.4 WebRTC reality check

WebRTC over MediaMTX uses:

- TCP 8889 for the SDP / HTTP signalling endpoint (`/webrtc`)
- UDP 8000-8100 for SRTP media (configurable in MediaMTX)

Caddy in `network_mode: host` can proxy the TCP signalling. The
**UDP media cannot be reverse-proxied by Caddy** — it has to be
either:

1. **Direct NAT/port-forward** of the UDP range to
   `198.51.100.20` on the edge host (the simplest path; document
   the port range in MediaMTX config and on the host firewall).
2. **TURN server** with public credentials — only needed if
   clients are on networks that block UDP to the edge host.

For a single-site demo, option 1 is the right move. Open
`198.51.100.20:8000-8100/udp` on the host firewall and document
the range in the operator runbook.

### 1.5 Validation (operator runs on the host)

```bash
# 1. Caddy is up
docker compose ps caddy
docker compose logs caddy | tail -20

# 2. HLS via the public hostname
curl -sI http://hls.example.invalid/sota-paddle-mtmc/CAM_01/index.m3u8
# expect: HTTP/1.1 200 OK

# 3. RTSP path still works
docker compose exec api python -c "
from app.streaming.mediamtx_streamer import make_from_env
s = make_from_env('CAM_01')
print(s.stream_urls())
"
# expect: rtsp URL on 8554, hls URL on the hls.example.invalid host, etc.
```

---

## §2 — Identity resolver crash

### What changed
- `app/identity/resolver.py::__init__` now always sets
  `self.consumer_name`. Priority: explicit kwarg > config >
  dataclass default > literal fallback.
- `run()` reads `self.consumer_name` defensively (via
  `getattr(..., None) or "..."`) and logs a structured warning if
  it ever falls back. The worker thread never crashes on
  AttributeError.
- New test file `tests/test_resolver_consumer_name.py` (6 tests)
  pins the contract.

### How to validate on the running host
```bash
docker compose logs -f api | grep -E "resolver|consumer_name"
# Expect to see a clean run loop. NO AttributeError stack traces.
```

---

## §3 — DB seeding

### What changed
- `db/seed/legacy_seed.py` — reads `configs/{cameras,zones,camera_links}.yaml`,
  calls `PostgresStore.upsert_*` (already idempotent on `ON CONFLICT
  DO UPDATE`).
- Fingerprint table `seed_fingerprints` records a sha1 of each
  YAML's canonical form. Warm restarts are a no-op.
- `SEED_FORCE=1` env var bypasses the fingerprint check.
- Wired into `app/main.py` right after the PG healthcheck, before
  the API/worker threads start. Failures log a warning and
  continue (the API can still serve in degraded mode).
- New test file `tests/test_legacy_seed.py` (4 tests).

### How to validate on the running host
```bash
# After a clean DB (or first boot with empty tables):
docker compose up -d --force-recreate api
docker compose logs api | grep "startup seed"
# expect: startup seed complete: {'cameras': 5, 'zones': 6, 'camera_links': 7, 'skipped': 0}

# On a warm restart (no YAML changes):
docker compose restart api
docker compose logs api | grep "startup seed"
# expect: startup seed complete: {'cameras': 0, 'zones': 0, 'camera_links': 0, 'skipped': 1}

# DB row counts:
docker compose exec postgres psql -U yamaha -d yamaha_mtmct -c "
  SELECT 'cameras' AS t, count(*) FROM cameras
  UNION ALL SELECT 'zones', count(*) FROM zones
  UNION ALL SELECT 'camera_links', count(*) FROM camera_links;
"
# expect: 5 / 6 / 7

# Force a re-seed (e.g. after editing YAML):
SEED_FORCE=1 docker compose up -d --force-recreate api
```

---

## §4 — Streaming contract

### What changed
- `app/streaming/mediamtx_streamer.py::make_from_env` — `MEDIAMTX_HLS_PORT`
  default was `8888` (now `8889`); `MEDIAMTX_WEBRTC_PORT` default was
  `8889` (now `8890`). These are env-var defaults: the operator's
  `.env` was already correct, so this fix matters for fresh
  `.env.example` clones.
- New test file `tests/test_streaming_contract_consistency.py` (5
  tests) pins 8554/8889/8890 and the `sota-paddle-mtmc` prefix.

### Authoritative contract

| Component | Value | Source |
| --------- | ----- | ------ |
| RTSP port | 8554 | `.env.example` `MEDIAMTX_RTSP_PORT` |
| HLS port | 8889 | `.env.example` `MEDIAMTX_HLS_PORT` |
| WebRTC port | 8890 | `.env.example` `MEDIAMTX_WEBRTC_PORT` |
| Stream prefix | `sota-paddle-mtmc` | `.env.example` `MEDIAMTX_STREAM_PREFIX` |

### How to validate on the running host
```bash
docker compose exec api python -c "
from app.streaming.mediamtx_streamer import make_from_env
s = make_from_env('CAM_01')
print(s.stream_urls())
"
# expect:
#   {'rtsp': 'rtsp://<host>:8554/sota-paddle-mtmc/CAM_01',
#    'hls':  'http://<host>:8889/sota-paddle-mtmc/CAM_01/index.m3u8',
#    'webrtc': 'http://<host>:8890/sota-paddle-mtmc/CAM_01'}
```

---

## §5 — Final verification (operator runs on the host)

After applying §1 (DNS + reverse proxy) and pulling the new code:

```bash
# 1. Pull the new code
cd SOTA-Paddle-MTMC
git pull
docker compose down
docker compose up -d

# 2. Stream health
docker compose logs -f | tee /tmp/sota-logs.log

# 3. HLS via the public hostname (NOT raw IP)
curl -sI http://hls.example.invalid/sota-paddle-mtmc/CAM_01/index.m3u8
curl -sI http://hls.example.invalid/sota-paddle-mtmc/CAM_02/index.m3u8
# expect: 200 OK on both

# 4. WebRTC (operator-side; depends on §1.4 choice)
# If you went with direct UDP: rtc.example.invalid will not serve an
# HTTP 200 — WebRTC has no plaintext probe. Validate via the
# browser console or the MediaMTX /metrics endpoint.

# 5. Identity resolver stable
docker compose logs api 2>&1 | grep -E "resolver|consumer_name" | tail -20
# expect: a clean run loop. NO AttributeError.

# 6. DB seeded
docker compose exec postgres psql -U yamaha -d yamaha_mtmct -c "
  SELECT 'cameras' AS t, count(*) FROM cameras
  UNION ALL SELECT 'zones', count(*) FROM zones
  UNION ALL SELECT 'camera_links', count(*) FROM camera_links;
"
# expect: 5 / 6 / 7

# 7. MediaMTX routing
docker compose exec api python -c "
from app.streaming.mediamtx_streamer import make_from_env
for cid in ('CAM_01', 'CAM_02'):
    print(cid, make_from_env(cid).stream_urls())
"
```

---

## C. Verified URLs (after operator completes §1 + §5)

The operator must paste the live curl outputs into the project
ticket. Expected shape (NOT a real curl result — agent cannot run
this):

- HLS CAM_01: `http://hls.example.invalid/sota-paddle-mtmc/CAM_01/index.m3u8` → `HTTP/1.1 200 OK`
- HLS CAM_02: `http://hls.example.invalid/sota-paddle-mtmc/CAM_02/index.m3u8` → `HTTP/1.1 200 OK`
- WebRTC CAM_01: `http://rtc.example.invalid:8890/sota-paddle-mtmc/CAM_01` (no HTTP 200 expected; validate via the MediaMTX `/metrics` endpoint or browser console)
- RTSP CAM_01: `rtsp://<host>:8554/sota-paddle-mtmc/CAM_01` (ffmpeg connect)
- RTSP CAM_02: `rtsp://<host>:8554/sota-paddle-mtmc/CAM_02` (ffmpeg connect)

---

## D. Evidence logs (operator to fill in)

This section is intentionally left for the operator. Run §5 and
paste the output here:

### Identity resolver
```
docker compose logs api 2>&1 | grep -E "resolver|consumer_name" | tail -20
<PASTE>
```

### DB seeded
```
docker compose exec postgres psql -U yamaha -d yamaha_mtmct -c "
  SELECT 'cameras' AS t, count(*) FROM cameras
  UNION ALL SELECT 'zones', count(*) FROM zones
  UNION ALL SELECT 'camera_links', count(*) FROM camera_links;
"
<PASTE>
```

### MediaMTX routing
```
docker compose exec api python -c "
from app.streaming.mediamtx_streamer import make_from_env
for cid in ('CAM_01', 'CAM_02'):
    print(cid, make_from_env(cid).stream_urls())
"
<PASTE>
```

### HLS public
```
curl -sI http://hls.example.invalid/sota-paddle-mtmc/CAM_01/index.m3u8
curl -sI http://hls.example.invalid/sota-paddle-mtmc/CAM_02/index.m3u8
<PASTE>
```

---

## Notes on what was NOT done

The agent that wrote this hand-off did not:

- Update Cloudflare DNS records (no API access).
- Run `docker compose restart` on the production host.
- Generate live evidence logs from the running MediaMTX.
- Test the reverse-proxy Caddyfile against a real request.
- Open or close the UDP WebRTC port range on the host firewall.

All of the above are operator actions, listed in §1 and §5. If
the operator wants a "fully automated" run, port this hand-off
into an Ansible/Terraform module and run it from a host with the
right credentials.
