"""Tests for PPHumanDetectorAdapter / PPHumanFrameStateAdapter.

These tests pin the new contract introduced by the
``PPHumanWorker`` ↔ ``PPHumanDetectorAdapter`` fix (PATCH-051):

  * ``PPHumanDetectorAdapter`` remains subprocess-only at the
    model layer; per-frame access is mediated by a
    ``PPHumanFrameStateAdapter`` that tails the subprocess's
    MOT output and exposes a per-camera, per-frame lookup.
  * The smoke path (synthetic detector) is **only** allowed
    when the runtime mode is ``SMOKE_TEST`` — production
    refuses to start without a real detector.
  * Detection records carry ``camera_id`` (via the adapter's
    per-camera factory), ``frame_id``, ``bbox``,
    ``confidence``, and an implicit ``class_id``/``class_name``
    in the worker's ``LocalTrack`` (default 0 / "person",
    matching PaddleDetection's PP-Human MOT format).
  * Local track IDs are camera-local — the adapter does NOT
    assign a global identity; the resolver does.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from app.detection.pphuman_pipeline import (
    Detection,
    PPHumanDetectorAdapter,
    PPHumanFrameStateAdapter,
)
from app.core.runtime_mode import (
    ProductionSafetyError,
    RuntimeMode,
)
from app.workers.pphuman_worker import PPHumanWorker


# ----------------------------------------------------------------------------
# Detection record shape
# ----------------------------------------------------------------------------


def test_detection_record_has_required_fields() -> None:
    """A Detection must carry camera_id (via the adapter path),
    frame_id, bbox, confidence, class_id (worker maps to 0).
    """
    d = Detection(
        frame_id=1,
        track_id=7,
        bbox=(10.0, 20.0, 110.0, 220.0),
        confidence=0.91,
    )
    assert d.frame_id == 1
    assert d.track_id == 7
    assert d.bbox == (10.0, 20.0, 110.0, 220.0)
    assert d.confidence == pytest.approx(0.91)


# ----------------------------------------------------------------------------
# PPHumanDetectorAdapter refuse-synthetic behaviour
# ----------------------------------------------------------------------------


def test_pphuman_detector_adapter_refuses_synthetic_in_production(tmp_path) -> None:
    """In production, ``PPHumanDetectorAdapter.load()`` must fail
    if the official pipeline is not installed (no synthetic
    fallback). The smoke path is the only place the synthetic
    flag is allowed.
    """
    adapter = PPHumanDetectorAdapter(
        pipeline_path=str(tmp_path / "missing-pipeline.py"),
        config_path=str(tmp_path / "missing-config.yml"),
        model_dir=str(tmp_path / "missing-model"),
        mode=RuntimeMode.PRODUCTION,
    )
    with pytest.raises(ProductionSafetyError):
        adapter.load()
    assert adapter.is_synthetic is False


def test_pphuman_detector_adapter_smoke_synthetic_fallback(tmp_path) -> None:
    """In SMOKE_TEST, the adapter may fall back to a synthetic
    detector when the pipeline is not on disk. The flag must be
    True after ``load()`` so the rest of the system can refuse
    to use it in production.
    """
    adapter = PPHumanDetectorAdapter(
        pipeline_path=str(tmp_path / "missing-pipeline.py"),
        config_path=str(tmp_path / "missing-config.yml"),
        model_dir=str(tmp_path / "missing-model"),
        mode=RuntimeMode.SMOKE_TEST,
    )
    adapter.load()
    assert adapter.is_synthetic is True


# ----------------------------------------------------------------------------
# PPHumanFrameStateAdapter: the per-frame bridge
# ----------------------------------------------------------------------------


class _FakeSubprocessManager:
    """Stand-in for :class:`PPHumanPipelineSubprocessManager`.

    Yields a deterministic stream of (camera_id, Detection)
    tuples without spawning subprocesses — the unit-test seam
    for the per-frame adapter.
    """

    def __init__(self, plan):
        # plan: list of (camera_id, n_frames, [(frame_id, x1, y1, x2, y2, conf), ...])

        # plan: list of (camera_id, n_frames, [(frame_id, x1, y1, x2, y2, conf), ...])
        self._plan = plan
        self._started = False
        self._stop = threading.Event()

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._stop.set()

    def stream(self):
        # Replay each camera's frames in increasing order;
        # yield per-detection. A short sleep is inserted to
        # give the tailer thread a chance to keep up — the
        # exact timing is irrelevant for the test, only the
        # ordering matters.
        for cam, _n, dets in self._plan:
            for d in dets:
                if self._stop.is_set():
                    return
                yield (
                    cam,
                    Detection(
                        frame_id=int(d[0]),
                        track_id=1,
                        bbox=(float(d[1]), float(d[2]), float(d[3]), float(d[4])),
                        confidence=float(d[5]),
                    ),
                )
                # Yield to the tailer.
                time.sleep(0.001)


def test_frame_state_adapter_buffers_per_frame() -> None:
    """The frame-state adapter must accumulate detections
    keyed by ``frame_id`` so a late-arriving subprocess write
    is still attributable to the correct frame.
    """
    mgr = _FakeSubprocessManager(
        [
            (
                "CAM_01",
                3,
                [
                    (1, 10.0, 20.0, 110.0, 220.0, 0.91),
                    (2, 30.0, 40.0, 130.0, 240.0, 0.88),
                    (3, 50.0, 60.0, 150.0, 260.0, 0.85),
                ],
            ),
        ],
    )
    adapter = PPHumanFrameStateAdapter(manager=mgr)  # type: ignore[arg-type]
    adapter.start()
    # Allow the tailer to drain the (tiny) plan.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if adapter.detections_for_frame("CAM_01", 3):
            break
        time.sleep(0.01)
    adapter.stop()
    assert adapter.detections_for_frame("CAM_01", 1) != []
    assert adapter.detections_for_frame("CAM_01", 2) != []
    assert adapter.detections_for_frame("CAM_01", 3) != []


def test_frame_state_adapter_returns_empty_for_unknown_camera() -> None:
    mgr = _FakeSubprocessManager([])
    adapter = PPHumanFrameStateAdapter(manager=mgr)  # type: ignore[arg-type]
    adapter.start()
    try:
        assert adapter.detections_for_frame("CAM_99", 0) == []
    finally:
        adapter.stop()


def test_frame_state_adapter_per_camera_factory_signature() -> None:
    """The per-camera factory accepts (frame, frame_id) and
    returns the MOT detections for that frame.
    """
    mgr = _FakeSubprocessManager(
        [
            (
                "CAM_01",
                2,
                [
                    (1, 10.0, 20.0, 110.0, 220.0, 0.91),
                    (2, 30.0, 40.0, 130.0, 240.0, 0.88),
                ],
            ),
        ],
    )
    adapter = PPHumanFrameStateAdapter(manager=mgr)  # type: ignore[arg-type]
    adapter.start()
    try:
        factory = adapter.per_camera_detector_factory("CAM_01")
        # Tags for introspection.
        assert getattr(factory, "camera_id", None) == "CAM_01"
        # The factory must NOT raise NotImplementedError on
        # a normal frame lookup; before the subprocess has
        # written anything it returns [].
        assert factory(np.zeros((480, 640, 3), dtype=np.uint8), 0) == []
    finally:
        adapter.stop()


# ----------------------------------------------------------------------------
# PPHumanWorker accepts real detector output
# ----------------------------------------------------------------------------


class _StubFrameReader:
    def __init__(self, n_frames: int = 3, shape=(480, 640, 3)):
        self._n = n_frames
        self._shape = shape
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= self._n:
            raise StopIteration
        self._i += 1
        return self._i, 1.0, np.zeros(self._shape, dtype=np.uint8)


def test_pphuman_worker_no_longer_raises_not_implemented_in_production() -> None:
    """Regression: production wiring used to crash the worker
    with NotImplementedError because the worker tried to call
    a non-callable PPHumanDetectorAdapter instance. With the
    per-frame factory wired in, the worker MUST accept real
    detector output without NotImplementedError.
    """
    real_factory = lambda frame, frame_id=0: [  # noqa: E731
        Detection(
            frame_id=frame_id,
            track_id=1,
            bbox=(10.0, 20.0, 110.0, 220.0),
            confidence=0.91,
        ),
    ]
    worker = PPHumanWorker(
        camera_id="CAM_01",
        frame_reader=_StubFrameReader(3),
        skip_frame_num=0,
        smoke_test_mode=False,
        detector=None,
        detector_factory=real_factory,
        mode=RuntimeMode.PRODUCTION,
    )
    frames = list(worker.run())
    assert len(frames) == 3
    # Every non-skipped frame should have a track because the
    # factory always returns one Detection.
    non_skipped = [f for f in frames if not f.skipped]
    assert non_skipped, "expected at least one non-skipped frame"
    for f in non_skipped:
        assert len(f.tracks) == 1
        t = f.tracks[0]
        assert t.camera_id == "CAM_01"
        assert t.bbox == (10.0, 20.0, 110.0, 220.0)
        assert t.confidence == pytest.approx(0.91)
        assert t.class_id == 0  # PP-Human MOT → class 0 (person)


def test_pphuman_worker_local_track_id_is_camera_local_with_real_factory() -> None:
    """The worker's local_track_id is still camera-local even
    with a real (subprocess-backed) factory.
    """
    real_factory = lambda frame, frame_id=0: [  # noqa: E731
        Detection(frame_id=frame_id, track_id=1, bbox=(0, 0, 10, 10), confidence=0.5),
    ]
    w = PPHumanWorker(
        camera_id="CAM_01",
        frame_reader=_StubFrameReader(2),
        skip_frame_num=0,
        smoke_test_mode=False,
        detector_factory=real_factory,
        mode=RuntimeMode.PRODUCTION,
    )
    frames = list(w.run())
    ids = [t.local_track_id for f in frames for t in f.tracks]
    # The worker re-uses the same local_track_id across frames
    # because the IoU tracker matches both frames to the
    # same track (same bbox). The point of this test is that
    # the IDs do not encode any global meaning.
    assert all(isinstance(i, int) for i in ids)
    assert len(set(ids)) == 1  # the IoU tracker kept one track
