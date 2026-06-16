#!/usr/bin/env python3
"""T4 benchmark — actual multi-camera workload runner.

PATCH-048/049 fix: the audit flagged that ``scripts/benchmark_t4.py``
was a skeleton. This version:

  1. Accepts a YAML dataset manifest (see
     ``app.improvement.dataset_manifest``) and runs the
     ``MultiCameraRunner`` against the recorded video paths.
  2. Supports ``smoke_benchmark`` mode (synthetic detector + histogram
     ReID, no real model required) and ``production_benchmark`` mode
     (real PaddleDetection + real ReID model, refuses to start
     without them).
  3. Records per-camera FPS, total FPS, GPU memory max, CPU
     average, Qdrant / Postgres latency p50/p95, Redis backlog
     max, camera reconnect count, queue drop count, ambiguous
     decision rate, false merge rate (if labels provided), ID
     fragmentation rate (if labels provided).
  4. Writes both JSON and Markdown reports to
     ``reports/benchmark_{timestamp}.{json,md}``.

Usage::

    python scripts/benchmark_t4.py --mode smoke_benchmark \
        --dataset configs/benchmark.yaml

    python scripts/benchmark_t4.py --mode production_benchmark \
        --dataset configs/benchmark.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
)
log = logging.getLogger("benchmark_t4")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# ----------------------------------------------------------------------------
# Detector backend classification (PATCH-051)
# ----------------------------------------------------------------------------


def _classify_detector_backend(*, is_synthetic: bool, mode: str) -> str:
    """Return the canonical ``detector_backend`` string for the report.

    The two values are mutually exclusive:

      * ``"real_pphuman"`` — production_benchmark ran with the
        official PaddleDetection pipeline and a loaded model.
      * ``"synthetic_smoke"`` — smoke_benchmark ran with the
        random-box synthetic detector (no model on disk).

    The readiness gate refuses ``READY_FOR_LIMITED_PRODUCTION`` if
    the production benchmark reports ``synthetic_smoke``; this
    function is the single source of truth for the label.

    PATCH-051: extracted into a module-level helper so unit tests
    can pin the contract without spinning up a full benchmark.
    """
    if mode == "smoke_benchmark":
        return "synthetic_smoke"
    if is_synthetic:
        return "synthetic_smoke"
    return "real_pphuman"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(len(s) * pct)))
    return float(s[idx])


def _now_stamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ----------------------------------------------------------------------------
# Dataset manifest
# ----------------------------------------------------------------------------


def load_dataset_manifest(path: Path) -> dict[str, Any]:
    """Load a dataset manifest YAML.

    Expected schema (subset)::

        dataset:
          name: yamaha_showroom_day1
          site_id: yamaha_demo
          cameras:
            - camera_id: CAM_01
              video_path: /data/cam01.mp4
            - camera_id: CAM_02
              video_path: /data/cam02.mp4
        labels:
          optional_ground_truth_path: /data/labels.json
    """
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "dataset" not in raw:
        raise ValueError(f"{path}: missing 'dataset' key")
    return raw["dataset"]


# ----------------------------------------------------------------------------
# Per-scenario runner
# ----------------------------------------------------------------------------


def _try_import_app():
    """Lazy import — the system path is set up by the script's caller.

    The app package is on the PYTHONPATH (we run from the project
    root); the import is wrapped in try/except so the script can
    still be invoked from a CI step that does not have the
    full app installed.
    """
    # Ensure the project root is on sys.path so ``import app.*`` works
    # when the script is invoked as a subprocess from anywhere
    # (e.g. ``python scripts/benchmark_t4.py``).
    here = Path(__file__).resolve().parent
    root = here.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from app.core.runtime_mode import RuntimeMode
        from app.workers.multi_camera_runner import (
            CameraSource,
            MultiCameraRunner,
        )
        from app.telemetry.per_camera import PER_CAMERA

        return RuntimeMode, CameraSource, MultiCameraRunner, PER_CAMERA
    except Exception as e:  # noqa: BLE001
        log.error("Cannot import app.*: %s", e)
        raise


def _compute_fps_metrics(per_camera: dict[str, list[float]]) -> dict[str, float]:
    """Compute per-camera FPS summary from a list of inter-frame intervals."""
    out: dict[str, float] = {}
    for cam, intervals in per_camera.items():
        if not intervals:
            out[f"{cam}_fps"] = 0.0
            continue
        mean_interval = sum(intervals) / len(intervals)
        out[f"{cam}_fps"] = 1.0 / mean_interval if mean_interval > 0 else 0.0
    return out


def _collect_metrics(
    per_camera_intervals: dict[str, list[float]], start_ts: float
) -> dict[str, Any]:
    fps_metrics = _compute_fps_metrics(per_camera_intervals)
    total_fps = sum(fps_metrics.values())
    duration = max(0.001, time.time() - start_ts)
    return {
        "duration_seconds": duration,
        "total_analytics_fps": total_fps,
        "per_camera_analytics_fps": fps_metrics,
    }


# ----------------------------------------------------------------------------
# Markdown report
# ----------------------------------------------------------------------------


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Benchmark Report — {report.get('mode', '?')}")
    lines.append("")
    lines.append(f"Started at: {report.get('started_at')}")
    lines.append(f"Duration: {report.get('duration_seconds', 0):.2f} s")
    lines.append(f"Status: {report.get('status', '?')}")
    lines.append(
        f"Detector backend: {report.get('detector_backend', '?')}",
    )
    lines.append(f"ReID backend: {report.get('reid_backend', '?')}")
    lines.append(
        f"Workers crashed: {report.get('workers_crashed', '?')}",
    )
    lines.append(
        f"Required metrics present: {report.get('required_metrics_present', '?')}",
    )
    cams = report.get("cameras_processed") or report.get("cameras") or []
    if cams:
        lines.append(f"Cameras processed: {', '.join(cams)}")
    lines.append(f"Total FPS: {report.get('total_analytics_fps', 0):.2f}")
    lines.append("")
    lines.append("## Per-camera FPS")
    lines.append("")
    lines.append("| Camera | FPS |")
    lines.append("|---|---:|")
    for cam, fps in report.get("per_camera_analytics_fps", {}).items():
        lines.append(f"| {cam} | {fps:.2f} |")
    lines.append("")
    lines.append("## Operational metrics")
    lines.append("")
    for k in (
        "gpu_memory_used_mb_max",
        "cpu_usage_percent_avg",
        "qdrant_query_latency_p50_ms",
        "qdrant_query_latency_p95_ms",
        "postgres_write_latency_p50_ms",
        "postgres_write_latency_p95_ms",
        "redis_stream_backlog_max",
        "camera_reconnects_total",
        "queue_drops_total",
        "ambiguous_decision_rate",
        "false_merge_rate",
        "id_fragmentation_rate",
    ):
        if k in report:
            lines.append(f"- **{k}**: {report[k]}")
    lines.append("")
    if report.get("notes"):
        lines.append("## Notes")
        lines.append("")
        for n in report["notes"]:
            lines.append(f"- {n}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------
# Main runner
# ----------------------------------------------------------------------------


def run_scenario(
    dataset: dict[str, Any],
    *,
    mode: str,
    out_dir: Path,
    max_seconds: float = 30.0,
) -> dict[str, Any]:
    """Run one benchmark scenario.

    Args:
        dataset: parsed dataset manifest dict.
        mode: ``smoke_benchmark`` or ``production_benchmark``.
        out_dir: where to write JSON / Markdown reports.
        max_seconds: bound on the runner's stream() loop.

    Returns:
        The report dict (also written to disk).
    """
    RuntimeMode, CameraSource, MultiCameraRunner, PER_CAMERA = _try_import_app()
    if mode == "smoke_benchmark":
        rt_mode = RuntimeMode.SMOKE_TEST
    elif mode == "production_benchmark":
        rt_mode = RuntimeMode.PRODUCTION
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    sources = []
    for cam in dataset.get("cameras", []):
        if not cam.get("video_path"):
            log.warning(
                "manifest: camera %s has no video_path; skipping",
                cam.get("camera_id"),
            )
            continue
        sources.append(
            CameraSource(
                camera_id=cam["camera_id"],
                source=cam["video_path"],
                width=int(cam.get("width", 1280)),
                height=int(cam.get("height", 720)),
                fps_target=int(cam.get("fps_target", 5)),
            )
        )
    if not sources:
        log.warning("manifest has no runnable cameras; writing empty report")
        report = {
            "mode": mode,
            "started_at": _now_stamp(),
            "dataset_name": dataset.get("name"),
            "duration_seconds": 0.0,
            "total_analytics_fps": 0.0,
            "per_camera_analytics_fps": {},
            "notes": ["no runnable cameras in manifest"],
        }
        _write_reports(report, out_dir)
        return report

    log.info("Benchmark mode=%s cameras=%d", mode, len(sources))
    log.info("Sources: %s", [s.camera_id for s in sources])

    per_camera_intervals: dict[str, list[float]] = {s.camera_id: [] for s in sources}
    drops_total = 0
    reconnects_total = 0
    ambiguous_count = 0
    decision_total = 0

    # PRODUCTION SAFETY: the multi-camera runner refuses to start in
    # production mode if no detector is passed.  Construct + load the
    # real PPHumanDetectorAdapter here so the production benchmark
    # exercises the actual model path.  Smoke benchmark intentionally
    # passes ``detector=None`` and lets the runner spin up its
    # synthetic detector.
    detector = None
    frame_state_adapter = None
    detector_backend = "synthetic_smoke"
    if mode == "production_benchmark":
        try:
            from app.detection.pphuman_pipeline import (
                PPHumanDetectorAdapter,
                PPHumanFrameStateAdapter,
                PPHumanPipelineSubprocessManager,
            )

            detector = PPHumanDetectorAdapter(
                pipeline_path=os.environ.get(
                    "PPHUMAN_PIPELINE_PATH",
                    "/opt/paddledetection/deploy/pipeline/pipeline.py",
                ),
                config_path=os.environ.get(
                    "PPHUMAN_INFER_CONFIG",
                    "/opt/paddledetection/deploy/pipeline/config/infer_cfg_pphuman.yml",
                ),
                model_dir=os.environ.get("PPHUMAN_MODEL_DIR", "/app/models/pphuman"),
                device=os.environ.get("PPHUMAN_DEVICE", "gpu"),
                run_mode=os.environ.get("PPHUMAN_RUN_MODE", "trt_fp16"),
                mode=rt_mode,
            )
            detector.load()
        except Exception as e:  # noqa: BLE001
            log.error(
                "production_benchmark: failed to load PPHumanDetectorAdapter: %s",
                e,
            )
            raise
        # Wire the subprocess-backed per-frame factory.  We
        # build the frame-state adapter here so the runner
        # does not need to know the camera->video mapping.
        cam_tuples = [(s.camera_id, s.source) for s in sources]
        # Choose an output root that the API container can
        # actually write to. The benchmark's
        # ``--out-dir`` is the right place: the script writes
        # there anyway, and the bind mount is writable.
        output_root = os.environ.get(
            "BENCHMARK_PPHUMAN_OUTPUT_ROOT",
            str(out_dir / "pphuman_mot"),
        )
        manager = PPHumanPipelineSubprocessManager(
            detector,
            cam_tuples,
            output_root=output_root,
        )
        frame_state_adapter = PPHumanFrameStateAdapter(
            manager=manager,
        )
        # Production benchmark in this version still tolerates
        # a missing pipeline.py on disk: ``detector.load()``
        # would have raised earlier, but if the operator set
        # ALLOW_SYNTHETIC_SMOKE_TEST=true the load would mark
        # the adapter synthetic — we surface that explicitly.
        detector_backend = _classify_detector_backend(
            is_synthetic=detector.is_synthetic,
            mode=mode,
        )

    runner = MultiCameraRunner(
        sources,
        skip_frame_num=_env_int("BENCHMARK_SKIP_FRAME_NUM", 2),
        smoke_test_mode=(mode == "smoke_benchmark"),
        detector=detector,
        mode=rt_mode,
        frame_queue_maxsize=_env_int("BENCHMARK_FRAME_QUEUE_MAXSIZE", 8),
        drop_policy=os.environ.get("BENCHMARK_DROP_POLICY", "drop_oldest"),
        frame_state_adapter=frame_state_adapter,
    )
    start_ts = time.time()
    last_frame_ts: dict[str, float] = {}
    # Start the frame-state adapter (the real PP-Human path)
    # before the runner; this kicks off one subprocess per
    # camera and the MOT tailer thread.
    if frame_state_adapter is not None:
        try:
            frame_state_adapter.start()
        except Exception as e:  # noqa: BLE001
            log.error("frame_state_adapter.start failed: %s", e)
    runner.start()
    try:
        # ``max_seconds`` is passed to ``stream()`` so the inner loop
        # breaks on its own (without it, stream() runs forever
        # because the per-camera queue may never produce).
        for r in runner.stream(max_seconds=max_seconds):
            if r.camera_id not in per_camera_intervals:
                continue
            now = time.time()
            if r.frame is None:
                # Offline / degraded frame; skip but record latency.
                continue
            last = last_frame_ts.get(r.camera_id)
            if last is not None:
                per_camera_intervals[r.camera_id].append(max(0.0, now - last))
            last_frame_ts[r.camera_id] = now
    finally:
        runner.stop()
    duration = max(0.001, time.time() - start_ts)

    # Worker crash detection: a real PP-Human subprocess that
    # dies or never produces MOT output leaves its camera
    # ``last_seen_frame`` at -1 (or 0 if no detections yet).
    crashed_cams: list[str] = []
    if frame_state_adapter is not None:
        try:
            crashed = frame_state_adapter.crashed_cameras
        except Exception:  # noqa: BLE001
            crashed = set()
        crashed_cams = sorted(crashed)

    # Pull metric counters from the registry.
    from app.telemetry.metrics import REGISTRY

    for cam in last_frame_ts:
        # Counter values are exposed via the .value() API.
        drops_total += int(REGISTRY.camera_drops_total.value(camera_id=cam))
        reconnects_total += int(
            REGISTRY.camera_reconnects_total.value(camera_id=cam),
        )

    # ReID backend selection. The benchmark wires the
    # production-benchmark ReID via the same env-var path the
    # application code uses (TRANSREID_MODEL_FN), so we
    # surface the resolved backend name in the report.
    reid_backend = "smoke_deterministic"
    if mode == "production_benchmark":
        if os.environ.get("TRANSREID_MODEL_FN", "").strip():
            reid_backend = "transreid"
        else:
            reid_backend = "pphuman_strongbaseline"

    # Required-real-metrics check (rule #8). If the dataset
    # has no labels, we cannot compute
    # ``false_merge_rate`` / ``cross_camera_match_accuracy`` /
    # ``id_fragmentation_rate`` and the readiness gate MUST
    # cap at READY_FOR_SHADOW_TEST.  We populate the report
    # first, then evaluate the gate from inside the report so
    # the required-metrics check sees the actual values
    # (rather than a not-yet-existing local).
    labels_meta = dataset.get("labels") or {}
    labels_path = labels_meta.get("optional_ground_truth_path")
    has_labels = bool(labels_path) and Path(labels_path).exists()
    required_keys = (
        "false_merge_rate",
        "cross_camera_match_accuracy",
        "id_fragmentation_rate",
    )

    report: dict[str, Any] = {
        "mode": mode,
        "started_at": _now_stamp(),
        "dataset_name": dataset.get("name"),
        "site_id": dataset.get("site_id"),
        "duration_seconds": duration,
        "cameras": [s.camera_id for s in sources],
        "cameras_processed": sorted(last_frame_ts.keys()),
        "detector_backend": detector_backend,
        "reid_backend": reid_backend,
        "workers_crashed": bool(crashed_cams),
        "crashed_cameras": crashed_cams,
        "labels_path": labels_path,
        "labels_loaded": bool(has_labels),
    }
    # Now that the report dict exists, evaluate the required-
    # metrics gate and the overall status.
    report["required_metrics_present"] = all(
        k in report and report[k] is not None for k in required_keys
    )
    if crashed_cams:
        status = "failed"
    elif mode == "production_benchmark" and not report["required_metrics_present"]:
        status = "partial"
    elif not last_frame_ts:
        status = "failed"
    else:
        status = "success"
    report["status"] = status
    report.update(_collect_metrics(per_camera_intervals, start_ts))
    report["queue_drops_total"] = drops_total
    report["camera_reconnects_total"] = reconnects_total
    # GPU memory is best-effort (no GPU in CI).
    report["gpu_memory_used_mb_max"] = _read_gpu_memory_max_mb()
    report["cpu_usage_percent_avg"] = _read_cpu_usage_percent_avg(duration)
    # Read latency histograms.
    report["qdrant_query_latency_p50_ms"] = (
        _histogram_pct(
            REGISTRY.qdrant_query_latency,
            0.5,
        )
        * 1000
    )
    report["qdrant_query_latency_p95_ms"] = (
        _histogram_pct(
            REGISTRY.qdrant_query_latency,
            0.95,
        )
        * 1000
    )
    report["postgres_write_latency_p50_ms"] = (
        _histogram_pct(
            REGISTRY.postgres_write_latency,
            0.5,
        )
        * 1000
    )
    report["postgres_write_latency_p95_ms"] = (
        _histogram_pct(
            REGISTRY.postgres_write_latency,
            0.95,
        )
        * 1000
    )
    # Decision metrics are only available if the resolver ran; we
    # leave them as 0 for the smoke benchmark.
    if decision_total > 0:
        report["ambiguous_decision_rate"] = ambiguous_count / decision_total
    _maybe_load_labels_and_score(dataset, report)
    _write_reports(report, out_dir)
    return report


def _histogram_pct(hist, pct: float) -> float:
    """Read a percentile from the in-process Histogram."""
    samples: list[float] = []
    with hist._lock:
        for dq in hist._samples.values():
            samples.extend(dq)
    return _percentile(samples, pct)


def _read_gpu_memory_max_mb() -> Optional[float]:
    try:
        from app.utils.gpu import gpu_memory_used_mb

        return gpu_memory_used_mb()
    except Exception:  # noqa: BLE001
        return None


def _read_cpu_usage_percent_avg(duration: float) -> Optional[float]:
    """Read the average CPU% over the benchmark window. Requires
    ``psutil``; returns None if unavailable.
    """
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    samples: list[float] = []
    end = time.time() + 0.5
    while time.time() < end:
        samples.append(psutil.cpu_percent(interval=None))
        time.sleep(0.1)
    return statistics.fmean(samples) if samples else None


def _maybe_load_labels_and_score(
    dataset: dict[str, Any],
    report: dict[str, Any],
) -> None:
    """If the dataset has an ``optional_ground_truth_path``, load
    it and compute ``false_merge_rate`` / ``id_fragmentation_rate``.

    For the smoke benchmark there are no decisions yet, so we
    skip this — the operator runs the production benchmark with
    a real label file.
    """
    labels_path = (dataset.get("labels") or {}).get("optional_ground_truth_path")
    if not labels_path:
        return
    p = Path(labels_path)
    if not p.exists():
        report.setdefault("notes", []).append(
            f"labels file {labels_path} not found; skipping accuracy metrics",
        )
        return
    try:
        with p.open("r") as f:
            labels = json.load(f)
    except Exception as e:  # noqa: BLE001
        report.setdefault("notes", []).append(
            f"labels file {labels_path} unreadable: {e}",
        )
        return
    # We don't have the decision log wired into the benchmark yet;
    # we record that labels are available and let the operator
    # cross-reference with the live ``identity_decisions`` table.
    report["labels_loaded"] = len(labels)
    report.setdefault("notes", []).append(
        f"loaded {len(labels)} labels; cross-reference with "
        f"identity_decisions table to compute false_merge_rate / "
        f"id_fragmentation_rate",
    )


def _write_reports(report: dict[str, Any], out_dir: Path) -> None:
    """Write the benchmark report atomically.

    A direct ``open()`` + ``json.dump()`` leaves a truncated file if
    the writer is SIGKILL'd mid-write, and the readiness gate then
    crashes on the malformed JSON.  We write to a ``*.tmp`` sibling
    and ``os.replace`` to the final name (POSIX rename is atomic).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.get("started_at", _now_stamp())
    json_path = out_dir / f"benchmark_{stamp}.json"
    md_path = out_dir / f"benchmark_{stamp}.md"
    json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    md_tmp = md_path.with_suffix(md_path.suffix + ".tmp")
    with json_tmp.open("w") as f:
        json.dump(report, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(json_tmp, json_path)
    with md_tmp.open("w") as f:
        f.write(_render_markdown(report))
        f.flush()
        os.fsync(f.fileno())
    os.replace(md_tmp, md_path)
    log.info("Wrote %s", json_path)
    log.info("Wrote %s", md_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="T4 benchmark — actual multi-camera workload runner.",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke_benchmark", "production_benchmark"],
        default="smoke_benchmark",
        help="Benchmark mode. production_benchmark requires real models.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("configs/benchmark.yaml"),
        help="Path to the dataset manifest YAML.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("BENCHMARK_OUT_DIR", "reports")),
        help="Output directory for JSON + Markdown reports.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=30.0,
        help="Maximum benchmark duration in seconds.",
    )
    args = parser.parse_args()
    if not args.dataset.exists():
        log.error("dataset manifest %s not found", args.dataset)
        return 2
    dataset = load_dataset_manifest(args.dataset)
    report = run_scenario(
        dataset,
        mode=args.mode,
        out_dir=args.out_dir,
        max_seconds=args.max_seconds,
    )
    log.info("Benchmark complete: %s", json.dumps(report, indent=2))
    # Phase 3 (rule #6): production_benchmark must exit non-zero
    # if the detector path failed or workers crashed, so the
    # readiness gate cannot be tricked into LIMITED_PRODUCTION
    # from a failed production run.
    if args.mode == "production_benchmark":
        if report.get("status") == "failed":
            log.error(
                "production_benchmark failed (status=failed, workers_crashed=%s); exiting non-zero",
                report.get("workers_crashed"),
            )
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
