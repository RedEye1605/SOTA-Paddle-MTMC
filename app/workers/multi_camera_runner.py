"""Multi-camera runner.

A single Python process hosts N per-camera workers. The detector is
shared across workers (the audit's "one model instance per process"
rule, PATCH-007). The runner yields `FrameResult` objects from all
cameras into a single async queue that the tracklet collector drains.

Hard rules (enforced by tests):
  - The detector (real or synthetic) is constructed ONCE in start() and
    passed to every worker.
  - The smoke test mode is the ONLY mode that allows the synthetic
    detector; in production the synthetic path is refused.
  - One model instance shared across all cameras (verified by an
    architecture-guard test that asserts `id(worker._detector) ==
    id(shared)`).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Iterator, Optional

import cv2
import numpy as np

from ..detection.pphuman_pipeline import (
    PPHumanDetectorAdapter,
    PPHumanFrameStateAdapter,
    PPHumanPipelineSubprocessManager,
)
from ..core.runtime_mode import (
    RuntimeMode,
    assert_production_safe,
    resolve_runtime_mode,
    smoke_log,
)
from ..telemetry.per_camera import (
    CAMERA_STATUS_OFFLINE,
    CAMERA_STATUS_ONLINE,
    PER_CAMERA,
)
from ..utils.resilient_reader import (
    ReconnectConfig,
    ResilientFrameReader,
    _is_live_stream,
    _normalize_video_source,
)
from ..streaming.mediamtx_streamer import make_from_env as make_streamer_from_env
from ..streaming.overlay import annotate_frame
from .pphuman_worker import FrameResult, PPHumanWorker


def _make_streamer_for_camera(cam: "CameraSource"):
    """Build a MediaMTX streamer for *cam* from env. Returns ``None``
    when the streamer is disabled (``MEDIAMTX_ENABLED=false`` or
    ``MEDIAMTX_HOST`` is empty) so dev environments without a
    streaming server don't fail."""
    try:
        return make_streamer_from_env(cam.camera_id)
    except Exception:  # noqa: BLE001
        return None

logger = logging.getLogger(__name__)


@dataclass
class CameraSource:
    camera_id: str
    source: str  # RTSP URL or local file path
    width: int
    height: int
    fps_target: int


def make_frame_reader(source: str, *, loop: bool = False):
    """A simple, blocking frame reader using OpenCV. Production should
    use FFmpeg for RTSP streams (more robust reconnection)."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break
        frame_id += 1
        ts = time.time()
        yield frame_id, ts, frame
    cap.release()


class MultiCameraRunner:
    """N per-camera workers, ONE shared detector (and ONE shared model)."""

    def __init__(
        self,
        cameras: list[CameraSource],
        *,
        skip_frame_num: int = 2,
        smoke_test_mode: bool = True,
        detector: Optional[PPHumanDetectorAdapter] = None,
        mode: Optional[RuntimeMode] = None,
        frame_reader_factory: Optional[callable] = None,
        frame_queue_maxsize: int = 64,
        drop_policy: str = "drop_oldest",
        reconnect: Optional[ReconnectConfig] = None,
        frame_state_adapter: Optional[PPHumanFrameStateAdapter] = None,
        identity_overlay_cache: Optional[object] = None,
    ) -> None:
        """Create a multi-camera runner.

        Args:
            cameras: list of ``CameraSource`` to run.
            skip_frame_num: process every Nth frame (1 = every other).
            smoke_test_mode: legacy flag — ``True`` implies SMOKE_TEST mode.
            detector: optional shared ``PPHumanDetectorAdapter`` (real
                or synthetic). Production: required. Smoke: optional.
            mode: explicit ``RuntimeMode``. If unset, falls back to
                ``smoke_test_mode``.
            frame_reader_factory: optional ``(source) -> Iterator`` factory
                for the per-camera frame reader. Used by tests to inject
                a fake reader. Defaults to ``make_frame_reader``.
            frame_queue_maxsize: per-camera frame-queue capacity
                (PATCH-031). Default 64 (matches the previous value).
            drop_policy: ``drop_oldest`` | ``drop_newest`` |
                ``block_with_timeout`` (PATCH-031).
            reconnect: optional PATCH-032 reconnect config. Applied
                when the runner auto-creates readers via the
                resilient path (the test-injected factory still wins).
            frame_state_adapter: optional pre-built
                :class:`PPHumanFrameStateAdapter`. When ``detector``
                is also passed, the adapter is preferred and the
                runner skips the synthetic / production detector
                negotiation. Used by ``benchmark_t4.py`` to inject
                a fake adapter for tests.
        """
        self.cameras = cameras
        self.skip_frame_num = skip_frame_num
        self._smoke_test_mode = smoke_test_mode
        self._explicit_detector = detector
        self._explicit_frame_state = frame_state_adapter
        # Backward compat: smoke_test_mode=True implies SMOKE_TEST.
        if mode is None:
            if smoke_test_mode:
                self._mode = RuntimeMode.SMOKE_TEST
            else:
                self._mode = resolve_runtime_mode()
        else:
            self._mode = mode
        self._frame_reader_factory = frame_reader_factory or self._default_factory
        # PATCH-031: configurable backpressure.
        if drop_policy not in {"drop_oldest", "drop_newest", "block_with_timeout"}:
            raise ValueError(
                f"Unknown drop_policy {drop_policy!r}; "
                f"expected drop_oldest | drop_newest | block_with_timeout",
            )
        self._drop_policy = drop_policy
        self._frame_queue_maxsize = max(1, int(frame_queue_maxsize))
        self._reconnect = reconnect or ReconnectConfig(enabled=True)
        # PATCH (2026-06-15): optional IdentityOverlayCache for real-
        # time ``G:{global_id}`` rendering in the HLS overlay. When
        # None (e.g. tests, eval profile), the overlay falls back to
        # the local_track_id label only.
        self._identity_overlay_cache = identity_overlay_cache
        self._workers: list[PPHumanWorker] = []
        self._queues: list[Queue] = []
        self._downstream_queues: list[Queue] = []
        self._streamers: list = []
        self._streamer_drain_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # The shared detector is constructed exactly once in start().
        # It is the same instance across all workers.
        self._shared_detector: Optional[PPHumanDetectorAdapter] = None
        # The frame-state adapter (real PP-Human subprocess path)
        # is constructed lazily in start() when a real detector
        # is provided and the caller did not inject one.
        self._frame_state: Optional[PPHumanFrameStateAdapter] = None
        # The optional subprocess manager — when a real detector is
        # provided, the runner can launch one PP-Human pipeline per
        # camera and tail their MOT output.
        self._pipeline_manager: Optional[PPHumanPipelineSubprocessManager] = None

    def _default_factory(self, cam: CameraSource) -> Iterator[tuple[int, float, np.ndarray]]:
        """Default per-camera reader factory.

        PATCH-032: live streams (RTSP / RTMP / HTTP / TCP / UDP) get
        the resilient reader with reconnect + degraded/offline
        state machine. Local file sources keep the simple
        ``make_frame_reader`` (no reconnect on EOF).
        """
        if _is_live_stream(cam.source):
            return ResilientFrameReader(
                cam.source,
                camera_id=cam.camera_id,
                config=self._reconnect,
            )
        # Local file source: normalize (file:// → /path, ~ → $HOME)
        # so OpenCV gets a real path on every platform.
        normalized = _normalize_video_source(cam.source)
        return make_frame_reader(normalized, loop=False)

    def start(self) -> None:
        # Build per-camera downstream queues BEFORE spawning the
        # drain thread (avoids a race where the drain thread
        # sees an empty ``self._queues`` and crashes with
        # ``IndexError``).
        # Always one downstream queue per camera, regardless of
        # whether a streamer is wired — the drain thread is the
        # sole consumer of the upstream queues in all cases.
        for _ in self.cameras:
            ds_q: Queue = Queue(maxsize=self._frame_queue_maxsize)
            self._downstream_queues.append(ds_q)
        # NOTE: streamers are appended to ``self._streamers`` later in
        # this method (per-camera, in the loop that builds workers).
        # The actual streamer start + drain-thread spawn happens
        # after that loop, when the list is populated. Logging "0
        # streamers" here was misleading (PATCH 2026-06-17).
        self._streamer_drain_thread: Optional[threading.Thread] = None
        # Idempotency — PATCH-025.
        if self._workers:
            return
        # The "one model instance" rule. Either the caller passed a
        # detector explicitly, or we construct a synthetic one in
        # smoke mode, or we fail-fast in production.
        if self._explicit_detector is not None:
            self._shared_detector = self._explicit_detector
            # If a real detector was passed, also wire the
            # subprocess-backed per-frame factory.  The factory
            # wraps one official PP-Human pipeline per camera
            # and tails their MOT output (PATCH-051).
            if self._explicit_frame_state is not None:
                self._frame_state = self._explicit_frame_state
            else:
                # Lazy-build a frame-state adapter. The runner
                # does NOT know each camera's video source here
                # (the underlying subprocess needs a video file
                # per camera), so the production path is
                # operator-supplied: the caller passes an
                # already-built frame_state_adapter when the
                # benchmark wants the real model wired in.
                smoke_log(
                    "MultiCameraRunner",
                    "real PPHumanDetectorAdapter passed without a "
                    "frame_state_adapter; the worker will fall back "
                    "to the per-frame factory from the adapter's "
                    "synthetic branch. Production must inject a "
                    "PPHumanFrameStateAdapter to avoid "
                    "NotImplementedError.",
                )
        elif self._mode == RuntimeMode.SMOKE_TEST:
            # Construct a synthetic detector (PPHumanDetectorAdapter
            # with no real pipeline; the worker's _synthetic branch
            # will fire). We use a per-instance flag — NOT a
            # per-worker one — so the architecture guard can verify
            # identity.
            self._shared_detector = None  # signals synthetic to the worker
            smoke_log(
                "MultiCameraRunner",
                f"no detector passed; will use synthetic (SMOKE-TEST) for "
                f"{len(self.cameras)} cameras",
            )
        else:
            assert_production_safe(
                mode=self._mode,
                component="MultiCameraRunner",
                condition="no detector adapter provided (refuse to start in production)",
            )

        for cam in self.cameras:
            reader = self._frame_reader_factory(cam)
            # If we have a real frame-state adapter, use a
            # per-camera factory; otherwise the worker falls
            # back to its synthetic branch (SMOKE-TEST only).
            detector_factory = None
            if self._frame_state is not None:
                detector_factory = self._frame_state.per_camera_detector_factory(
                    cam.camera_id,
                )
            worker = PPHumanWorker(
                camera_id=cam.camera_id,
                frame_reader=reader,
                skip_frame_num=self.skip_frame_num,
                smoke_test_mode=self._smoke_test_mode,
                detector=self._shared_detector,
                detector_factory=detector_factory,
                mode=self._mode,
            )
            # PATCH-031: configurable queue max size. The old hard-
            # coded value of 64 stays as the default.
            q: Queue = Queue(maxsize=self._frame_queue_maxsize)
            self._workers.append(worker)
            self._queues.append(q)
            # PATCH-018: per-camera metrics object.
            PER_CAMERA.for_camera(cam.camera_id)
            # PATCH-NNN fix: spin up one MediaMTX streamer per
            # camera so the annotated frames (with bbox overlays
            # + HUD) get pushed to RTSP for the HLS / WebRTC
            # consumers. The streamer's no-op when
            # ``MEDIAMTX_ENABLED=false`` or ``MEDIAMTX_HOST`` is
            # empty, so dev environments without MediaMTX still
            # work fine.
            streamer = _make_streamer_for_camera(cam)
            self._streamers.append(streamer)
            t = threading.Thread(
                target=self._run_worker,
                args=(worker, q, cam.camera_id),
                daemon=True,
                name=f"pphuman-{cam.camera_id}",
            )
            t.start()
        logger.info(
            "MultiCameraRunner started %d workers (shared detector=%s)",
            len(self._workers),
            "real" if self._shared_detector is not None else "synthetic",
        )
        # PATCH-FIX: start the per-camera MediaMTX streamers NOW that
        # they have been appended to ``self._streamers``. (The
        # previous start-loop ran against an empty list, before this
        # per-camera append, and was a silent no-op.) Each streamer
        # is itself a no-op when ``MEDIAMTX_ENABLED=false`` or
        # ``MEDIAMTX_HOST`` is empty.
        for streamer in self._streamers:
            if streamer is not None:
                try:
                    streamer.start()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("streamer start failed: %s", exc)
        # Now that ``self._queues`` is fully populated, start the
        # streamer drain thread. It captures references at
        # iteration time so it sees the freshly-built queues.
        self._streamer_drain_thread = threading.Thread(
            target=self._drain_to_streamers,
            daemon=True,
            name="streamer-drain",
        )
        self._streamer_drain_thread.start()

    def _run_worker(self, worker: PPHumanWorker, q: Queue, camera_id: str) -> None:
        """Per-worker producer loop.

        PATCH-018: emit per-camera metrics (frame latency, queue
        depth, drop count, decode errors, reconnects). PATCH-031:
        configurable drop policy (``drop_oldest`` / ``drop_newest``
        / ``block_with_timeout``).
        """
        m = PER_CAMERA.for_camera(camera_id)
        m.set_status(CAMERA_STATUS_ONLINE)
        last_put_wall = time.monotonic()
        try:
            for result in worker.run():
                if self._stop_event.is_set():
                    break
                now_wall = time.monotonic()
                # Frame latency = time elapsed since the previous emit.
                latency_ms = (now_wall - last_put_wall) * 1000.0
                last_put_wall = now_wall
                m.observe_frame_latency(latency_ms)
                m.observe_frame(result.ts)
                # Update queue depth BEFORE attempting the put so the
                # gauge reflects the worker's actual backlog.
                m.observe_queue_depth(q.qsize())
                try:
                    if self._drop_policy == "block_with_timeout":
                        q.put(result, timeout=0.5)
                    else:
                        q.put_nowait(result)
                except Exception:  # noqa: BLE001
                    m.observe_drop()
                    if self._drop_policy == "drop_newest":
                        # Newest (just-produced) frame is dropped; we
                        # try to evict the oldest and retry once.  If the
                        # retry also fails (queue is genuinely saturated),
                        # account for the second drop so the counter
                        # reflects the true number of frames not enqueued.
                        try:
                            q.get_nowait()
                        except Exception as e:  # noqa: BLE001
                            logger.debug(
                                "queue.get_nowait failed during "
                                "drop_newest evict: %s",
                                e,
                            )
                        try:
                            q.put_nowait(result)
                        except Exception as e:  # noqa: BLE001
                            logger.debug(
                                "queue.put_nowait failed during "
                                "drop_newest retry: %s",
                                e,
                            )
                            m.observe_drop()
                    # drop_oldest is the default; the put_nowait above
                    # raised Full and we silently drop the new frame.
                    continue
        except Exception as e:  # noqa: BLE001
            logger.exception("camera worker %s stopped after error: %s", camera_id, e)
            m.observe_decode_error()
            m.set_status(CAMERA_STATUS_OFFLINE)

    def stream(self, max_seconds: Optional[float] = None) -> Iterator[FrameResult]:
        """Yield results from all cameras in arrival order.

        PATCH-018: also feeds the per-camera queue-depth gauge. The
        actual frame observations are recorded by ``_run_worker``.
        """
        start = time.time()
        active = len(self._workers)
        # The drain thread is the sole consumer of the worker
        # queues (``self._queues``); it re-enqueues the same
        # FrameResult into the per-camera downstream queues
        # (``self._downstream_queues``) after pushing the
        # annotated frame to MediaMTX. ``stream()`` consumers
        # read from the downstream queues so they see the
        # frames *after* the MediaMTX push completes.
        while active > 0 and not self._stop_event.is_set():
            for cam, ds_q in zip(self.cameras, self._downstream_queues):
                PER_CAMERA.for_camera(cam.camera_id).observe_queue_depth(ds_q.qsize())
            for ds_q in self._downstream_queues:
                try:
                    item = ds_q.get(timeout=0.1)
                    yield item
                except Empty:
                    pass
            if max_seconds is not None and (time.time() - start) > max_seconds:
                break
            active = sum(1 for w in self._workers if w is not None)

    def stop(self) -> None:
        # PATCH-023 fix — signal the workers and join the threads.
        self._stop_event.set()
        for w in self._workers:
            try:
                # nothing to clean up on the worker itself; the
                # frame_reader is a generator closed by ``StopIteration``
                # when the underlying VideoCapture exhausts.
                pass
            except Exception:  # noqa: BLE001
                pass
        if self._frame_state is not None:
            try:
                self._frame_state.stop()
            except Exception:  # noqa: BLE001
                pass
        if self._pipeline_manager is not None:
            self._pipeline_manager.stop()
        # Stop the streamers (each one kills its ffmpeg subprocess).
        for streamer in self._streamers:
            if streamer is not None:
                try:
                    streamer.stop()
                except Exception:  # noqa: BLE001
                    pass
        self._workers.clear()
        self._queues.clear()
        self._streamers.clear()

    def _drain_to_streamers(self) -> None:
        """Background thread: pull frames from each camera's queue
        and push them (with overlay) into the MediaMTX streamer,
        then re-enqueue them to a per-camera downstream queue for
        :meth:`stream` consumers (the tracklet collector).

        We can't have two consumers of the same ``Queue`` —
        items would be split between them and the tracklet
        collector would see gaps. Instead the drain thread is
        the *sole* consumer; after pushing the annotated frame
        to the streamer, it re-enqueues the same ``FrameResult``
        (the numpy ``frame`` is shallow-copied — the next
        overlay step makes a new ndarray) to a per-camera
        downstream queue that :meth:`stream` reads.
        """
        # Per-camera downstream queue, indexed by camera_id. The
        # ``stream()`` method reads from this set instead of the
        # raw worker queues.
        # Map: camera_id -> (upstream queue, downstream queue, streamer).
        # We include every camera regardless of streamer presence
        # — the drain thread is the *sole* upstream consumer in
        # all cases. Streamer-less cameras just skip the push
        # step but still re-enqueue to the downstream queue.
        links: dict[str, tuple] = {}
        for cam, up_q, ds_q, streamer in zip(
            self.cameras, self._queues, self._downstream_queues, self._streamers
        ):
            links[cam.camera_id] = (up_q, ds_q, streamer)
        while not self._stop_event.is_set():
            for cam_id, (up_q, ds_q, streamer) in list(links.items()):
                try:
                    item = up_q.get(timeout=0.1)
                except Empty:
                    continue
                frame = getattr(item, "frame", None)
                # Push the annotated frame to MediaMTX (if a
                # streamer is wired and enabled). This is purely
                # visual; downstream consumers in :meth:`stream`
                # always get the original ``item`` regardless of
                # whether the streamer step succeeded.
                if frame is not None and streamer is not None and streamer.is_enabled():
                    try:
                        detections = [
                            {
                                "bbox": list(t.bbox),
                                "confidence": float(getattr(t, "score", 0.0)),
                                "class_name": "person",
                                "local_track_id": getattr(t, "track_id", None),
                                # PATCH (2026-06-15): look up global_id
                                # from the IdentityOverlayCache. The
                                # overlay renders ``G:{global_id}``
                                # when this is set; otherwise the
                                # existing fallback (local_track_id
                                # only) applies.
                                "global_id": (
                                    self._identity_overlay_cache.lookup(
                                        cam_id,
                                        getattr(t, "track_id", None),
                                    )
                                    if self._identity_overlay_cache is not None
                                    else None
                                ),
                            }
                            for t in getattr(item, "tracks", [])
                            if getattr(t, "bbox", None) is not None
                        ]
                        annotated = annotate_frame(
                            frame,
                            camera_id=cam_id,
                            frame_id=getattr(item, "frame_id", 0),
                            fps=0.0,
                            detector_backend="paddledetection_pphuman",
                            reid_backend="transreid_msmt",
                            detections=detections,
                            smoke=False,
                            site_id="yamaha_showroom",
                        )
                        streamer.push_frame(annotated)
                    except Exception as e:  # noqa: BLE001
                        logger.debug(
                            "streamer push_frame failed for %s: %s", cam_id, e
                        )
                # Re-enqueue for the tracklet collector (downstream
                # consumer in :meth:`stream`). ALWAYS — even when
                # the frame is ``None`` (e.g. EOF) so the
                # collector can update its state.
                try:
                    ds_q.put_nowait(item)
                except Exception as e:  # noqa: BLE001
                    logger.debug(
                        "downstream queue put_nowait failed for %s: %s", cam_id, e
                    )

    # ---- shared-model inspection (used by architecture-guard tests) ----
    def shared_detector(self) -> Optional[PPHumanDetectorAdapter]:
        """Return the single shared detector instance.

        Used by ``tests/test_architecture_guards.py`` to assert that all
        workers share the same model object (the audit's "one model
        instance per process" hard rule, PATCH-007 / PATCH-037).
        """
        return self._shared_detector

    def frame_state(self) -> Optional[PPHumanFrameStateAdapter]:
        """Return the per-frame adapter, if a real PP-Human
        pipeline is wired in. Used by
        ``tests/test_pphuman_detector_adapter.py`` and the
        benchmark report builder.
        """
        return self._frame_state
