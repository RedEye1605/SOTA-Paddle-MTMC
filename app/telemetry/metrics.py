"""In-process metrics — minimal Prometheus-style text output.

We intentionally do NOT use the official `prometheus_client` library
to keep the dependency surface small. The metrics are exposed as
text from the FastAPI /metrics endpoint.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any


class Counter:
    def __init__(self, name: str, help_: str = "") -> None:
        self.name = name
        self.help = help_
        self._lock = threading.Lock()
        self._value: dict[tuple, float] = {}

    def inc(self, amount: float = 1.0, **labels: Any) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._value[key] = self._value.get(key, 0.0) + amount

    def value(self, **labels: Any) -> float:
        key = tuple(sorted(labels.items()))
        with self._lock:
            return self._value.get(key, 0.0)

    def render(self) -> str:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self._lock:
            items = list(self._value.items())
        for key, val in items:
            if not key:
                out.append(f"{self.name} {val}")
            else:
                labels_str = ",".join(f'{k}="{v}"' for k, v in key)
                out.append(f"{self.name}{{{labels_str}}} {val}")
        return "\n".join(out)


class Gauge:
    def __init__(self, name: str, help_: str = "") -> None:
        self.name = name
        self.help = help_
        self._lock = threading.Lock()
        self._value: dict[tuple, float] = {}

    def set(self, value: float, **labels: Any) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            self._value[key] = value

    def render(self) -> str:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        with self._lock:
            items = list(self._value.items())
        for key, val in items:
            if not key:
                out.append(f"{self.name} {val}")
            else:
                labels_str = ",".join(f'{k}="{v}"' for k, v in key)
                out.append(f"{self.name}{{{labels_str}}} {val}")
        return "\n".join(out)


class Histogram:
    def __init__(self, name: str, help_: str = "") -> None:
        self.name = name
        self.help = help_
        self._lock = threading.Lock()
        self._samples: dict[tuple, deque] = {}
        self._maxlen = 256

    def observe(self, value: float, **labels: Any) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            dq = self._samples.setdefault(key, deque(maxlen=self._maxlen))
            dq.append(float(value))

    def render(self) -> str:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} summary"]
        with self._lock:
            items = list(self._samples.items())
        for key, dq in items:
            vals = sorted(dq)
            if not vals:
                continue
            n = len(vals)
            p50 = vals[max(0, n // 2 - 1)]
            p95 = vals[min(n - 1, int(n * 0.95))]
            p99 = vals[min(n - 1, int(n * 0.99))]
            labels_str = (",".join(f'{k}="{v}"' for k, v in key) + ",") if key else ""
            out.append(f'{self.name}{{{labels_str}quantile="0.5"}} {p50}')
            out.append(f'{self.name}{{{labels_str}quantile="0.95"}} {p95}')
            out.append(f'{self.name}{{{labels_str}quantile="0.99"}} {p99}')
            out.append(f"{self.name}{{{labels_str}count}} {n}")
        return "\n".join(out)


# Process-wide metrics registry
class MetricsRegistry:
    def __init__(self) -> None:
        # PATCH-018: per-camera metrics. Each is labeled by camera_id
        # so /metrics shows the breakdown.
        self.camera_fps = Gauge(
            "camera_fps",
            "Per-camera analytics FPS (rolling average)",
        )
        self.camera_frame_latency_ms = Gauge(
            "camera_frame_latency_ms",
            "Per-camera frame latency in milliseconds (last frame)",
        )
        self.camera_decode_errors_total = Counter(
            "camera_decode_errors_total",
            "Per-camera decode/read errors",
        )
        self.camera_reconnects_total = Counter(
            "camera_reconnects_total",
            "Per-camera RTSP reconnect events",
        )
        self.camera_last_frame_timestamp = Gauge(
            "camera_last_frame_timestamp",
            "Per-camera last frame timestamp (epoch seconds)",
        )
        self.camera_queue_depth = Gauge(
            "camera_queue_depth",
            "Per-camera frame-queue depth (current items)",
        )
        self.camera_status = Gauge(
            "camera_status",
            "Per-camera status (0=offline, 1=degraded, 2=online)",
        )
        self.camera_drops_total = Counter(
            "camera_drops_total",
            "Per-camera queue drops (backpressure)",
        )
        # Top-level counters (no labels).
        self.gpu_memory_used = Gauge(
            "gpu_memory_used_bytes",
            "GPU memory used in bytes",
        )
        self.qdrant_query_latency = Histogram(
            "qdrant_query_latency_seconds",
            "Qdrant query latency distribution",
        )
        self.postgres_write_latency = Histogram(
            "postgres_write_latency_seconds",
            "PostgreSQL write latency distribution",
        )
        self.reid_extractions = Counter(
            "reid_extractions_total",
            "Total ReID extractions performed",
        )
        self.identity_decisions = Counter(
            "identity_decisions_total",
            "Total identity decisions made",
        )
        # Aggregated across cameras.
        self.total_fps = Gauge(
            "total_analytics_fps",
            "Total analytics FPS (sum across cameras)",
        )

    def render(self) -> str:
        return "\n\n".join(
            [
                self.camera_fps.render(),
                self.camera_frame_latency_ms.render(),
                self.camera_decode_errors_total.render(),
                self.camera_reconnects_total.render(),
                self.camera_last_frame_timestamp.render(),
                self.camera_queue_depth.render(),
                self.camera_status.render(),
                self.camera_drops_total.render(),
                self.gpu_memory_used.render(),
                self.qdrant_query_latency.render(),
                self.postgres_write_latency.render(),
                self.reid_extractions.render(),
                self.identity_decisions.render(),
                self.total_fps.render(),
            ]
        )


# Global singleton
REGISTRY = MetricsRegistry()
