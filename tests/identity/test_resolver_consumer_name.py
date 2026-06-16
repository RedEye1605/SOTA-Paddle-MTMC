"""Regression tests for the resolver consumer_name crash.

Bug: GlobalIdentityResolver.__init__ stored config.consumer_name on
self.config but never copied it to self.consumer_name. The run() loop
called ``self.redis.consume(..., self.consumer_name, ...)`` and
crashed the worker thread with::

    AttributeError: 'GlobalIdentityResolver' object has no attribute
                   'consumer_name'

These tests pin down:
  1. __init__ always sets a non-empty self.consumer_name.
  2. The default is "resolver-worker-01" when nothing is provided.
  3. An explicit kwarg wins.
  4. config.consumer_name is honored.
  5. run() does not raise AttributeError even if the attribute is
     forcibly deleted before the loop starts.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock


from app.identity.resolver import GlobalIdentityResolver, ResolverConfig


def _make_resolver(**overrides) -> GlobalIdentityResolver:
    """Build a resolver with stubbed storage. We only exercise
    __init__ and run()-time attribute access; the storage clients are
    not exercised in these tests."""
    pg = MagicMock(name="pg")
    qdrant = MagicMock(name="qdrant")
    redis = MagicMock(name="redis")
    redis.consume.return_value = []  # empty stream => loop exits on stop
    topology = MagicMock(name="topology")
    return GlobalIdentityResolver(
        pg=pg,
        qdrant=qdrant,
        redis=redis,
        topology=topology,
        **overrides,
    )


def test_consumer_name_default_is_set() -> None:
    r = _make_resolver()
    assert hasattr(r, "consumer_name")
    assert r.consumer_name == "resolver-worker-01"


def test_consumer_name_kwarg_wins() -> None:
    r = _make_resolver(consumer_name="custom-resolver-A")
    assert r.consumer_name == "custom-resolver-A"


def test_consumer_name_from_config_is_honored() -> None:
    cfg = ResolverConfig(consumer_name="resolver-from-yaml")
    r = _make_resolver(config=cfg)
    assert r.consumer_name == "resolver-from-yaml"


def test_kwarg_overrides_config() -> None:
    cfg = ResolverConfig(consumer_name="resolver-from-yaml")
    r = _make_resolver(config=cfg, consumer_name="resolver-override")
    assert r.consumer_name == "resolver-override"


def test_run_does_not_crash_when_attribute_missing() -> None:
    """Regression: run() must not raise AttributeError. The worker
    thread must keep polling the stream."""
    r = _make_resolver(consumer_name="resolver-ok")
    # Simulate the bug class: someone deletes the attribute.
    del r.consumer_name

    stop = threading.Event()
    stop.set()  # exit the loop on the first iteration

    # run() must return without raising.
    t0 = time.time()
    r.run(stop_event=stop)
    assert (time.time() - t0) < 2.0, "run() should exit promptly when stop is set"


def test_run_uses_resolved_consumer_name_in_redis_consume() -> None:
    """The name passed to redis.consume() must be the resolved one,
    not raise AttributeError and not silently use 'None'."""
    r = _make_resolver(consumer_name="resolver-prod-1")

    stop = threading.Event()
    # Schedule the stop after one consume() call so the loop exits.
    def _consume_then_stop(*args, **kwargs):
        stop.set()
        return []

    r.redis.consume.side_effect = _consume_then_stop
    r.run(stop_event=stop)

    assert r.redis.consume.called
    call_args = r.redis.consume.call_args
    # Third positional arg is the consumer name.
    assert call_args.args[2] == "resolver-prod-1"
