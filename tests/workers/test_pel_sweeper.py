"""PELSweeper integration tests.

These tests use a real local Redis. They are skipped if ``REDIS_URL``
is not set or if Redis is unreachable. The tests share a single
client (per process) to avoid leaking stream keys across tests;
each test uses a unique stream key so the order does not matter.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Iterator

import pytest

from app.storage.redis_state import RedisState
from app.workers.pel_sweeper import PELSweeper


REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("PEL_TEST_REDIS_DB", "15"))


def _redis_available() -> bool:
    if not os.environ.get("REDIS_URL") and not _probe():
        return False
    return True


def _probe() -> bool:
    try:
        r = RedisState(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
        r.connect()
        ok = r.healthcheck()
        r.close()
        return ok
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _redis_available(),
    reason="local Redis not reachable; set REDIS_URL or run a redis on localhost:6379",
)


@pytest.fixture()
def redis_state() -> Iterator[RedisState]:
    r = RedisState(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    r.connect()
    r.client.flushdb()
    yield r
    r.client.flushdb()
    r.close()


def _unique_stream(prefix: str) -> str:
    return f"{prefix}:{uuid.uuid4().hex[:8]}"


def _seed_pending(redis: RedisState, stream: str, group: str, consumer: str, n: int = 3) -> list[str]:
    redis.client.xadd(stream, {"k": json.dumps("seed")})
    msg_ids: list[str] = []
    for _ in range(n - 1):
        msg_ids.append(redis.client.xadd(stream, {"k": json.dumps("seed")}))
    redis.ensure_group(stream, group, start_id="0")
    resp = redis.client.xreadgroup(
        group,
        consumer,
        {stream: ">"},
        count=n,
        block=100,
    )
    for _, entries in resp or []:
        for msg_id, _ in entries:
            msg_ids.append(msg_id)
    return msg_ids


def test_sweeper_claims_idle_messages(redis_state: RedisState) -> None:
    stream = _unique_stream("stream:sweeper_test")
    group = "g1"
    consumer = "dead-consumer"
    _seed_pending(redis_state, stream, group, consumer, n=3)

    pending = redis_state.client.xpending(stream, group)
    assert pending["pending"] == 3

    time.sleep(1.2)

    sweeper = PELSweeper(
        redis=redis_state,
        interval_seconds=60,
        min_idle_seconds=1,
        claim_batch_size=10,
    )
    try:
        sweeper.sweep_once()
    finally:
        sweeper.stop()

    pending_after = redis_state.client.xpending(stream, group)
    assert pending_after["pending"] == 0, pending_after
    metrics = sweeper.metrics()
    assert metrics["pel_swept_total"] == 3
    assert metrics["pel_claimed_total"] == 3
    assert redis_state.client.xlen(stream) == 6


def test_sweeper_requeues_claimed_messages(redis_state: RedisState) -> None:
    stream = _unique_stream("stream:sweeper_ack")
    group = "g1"
    consumer = "dead-consumer"
    _seed_pending(redis_state, stream, group, consumer, n=2)

    before = redis_state.client.xpending_range(
        name=stream, groupname=group, min="-", max="+", count=10
    )
    owners = {entry["consumer"] for entry in before}
    assert "dead-consumer" in owners

    time.sleep(1.2)
    sweeper = PELSweeper(
        redis=redis_state,
        interval_seconds=60,
        min_idle_seconds=1,
        claim_batch_size=10,
    )
    try:
        sweeper.sweep_once()
    finally:
        sweeper.stop()

    pending = redis_state.client.xpending(stream, group)
    assert pending["pending"] == 0

    resp = redis_state.client.xreadgroup(
        group,
        "live-consumer",
        {stream: ">"},
        count=10,
        block=100,
    )
    requeued = [msg_id for _, entries in resp or [] for msg_id, _ in entries]
    assert len(requeued) == 2


def test_sweeper_respects_min_idle_time(redis_state: RedisState) -> None:
    stream = _unique_stream("stream:sweeper_idle")
    group = "g1"
    consumer = "dead-consumer"
    _seed_pending(redis_state, stream, group, consumer, n=1)

    sweeper = PELSweeper(
        redis=redis_state,
        interval_seconds=60,
        min_idle_seconds=2,
        claim_batch_size=10,
    )
    try:
        sweeper.sweep_once()
        pending = redis_state.client.xpending(stream, group)
        assert pending["pending"] == 1, "fresh message should not be claimed"

        time.sleep(3.0)
        sweeper.sweep_once()
        pending = redis_state.client.xpending(stream, group)
        assert pending["pending"] == 0, "idle original should be requeued"
    finally:
        sweeper.stop()

    metrics = sweeper.metrics()
    assert metrics["pel_swept_total"] == 1
    assert metrics["pel_idle_seconds"] >= 2.0


def test_sweeper_handles_no_groups(redis_state: RedisState) -> None:
    stream = _unique_stream("stream:sweeper_nogrp")
    redis_state.client.xadd(stream, {"k": "v"})

    sweeper = PELSweeper(
        redis=redis_state,
        interval_seconds=60,
        min_idle_seconds=1,
        claim_batch_size=10,
    )
    try:
        sweeper.sweep_once()
    finally:
        sweeper.stop()

    metrics = sweeper.metrics()
    assert metrics["pel_swept_total"] == 0
    assert metrics["pel_errors_total"] == 0
    assert metrics["pel_groups_scanned"] == 0


def test_sweeper_stops_cleanly(redis_state: RedisState) -> None:
    sweeper = PELSweeper(
        redis=redis_state,
        interval_seconds=1,
        min_idle_seconds=1,
        claim_batch_size=10,
    )
    sweeper.start()
    time.sleep(0.1)
    started = time.monotonic()
    sweeper.stop()
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"stop() took {elapsed:.2f}s, expected <2s"
