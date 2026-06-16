"""PP-Human worker (per-camera).

In production, the worker delegates to a real :class:`PPHumanDetectorAdapter`
that runs the official PaddleDetection pipeline as a subprocess (per
the documented multi-stream pattern). The worker itself owns only the
local tracker state — the model lives in the subprocess, so the
subprocess-per-stream architecture keeps one model instance per
process (the audit's "one-model-per-process" rule is preserved: each
camera gets its own process, but the parent Python process contains
the orchestration and the resolver, not a duplicate model).

In smoke-test mode, the synthetic detector is used.

Hard guarantees (enforced by tests):
  - Local track ID is camera-local (no global meaning).
  - The detector factory is wired (real OR synthetic — production
    blocks the synthetic path).
  - ReID runs only on stable tracklets, never per-frame.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

import numpy as np

from ..detection.pphuman_pipeline import PPHumanDetectorAdapter
from ..core.runtime_mode import (
    RuntimeMode,
    assert_production_safe,
    resolve_runtime_mode,
    smoke_log,
)

logger = logging.getLogger(__name__)


@dataclass
class LocalTrack:
    camera_id: str
    local_track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    frame_id: int = 0
    ts: float = 0.0
    age_frames: int = 0
    is_confirmed: bool = False
    last_zone_id: Optional[str] = None


@dataclass
class FrameResult:
    camera_id: str
    frame_id: int
    ts: float
    frame: Optional[np.ndarray]
    tracks: list[LocalTrack] = field(default_factory=list)
    skipped: bool = False


class PPHumanWorker:
    """Per-camera worker. Owns a tracker state (NOT a model)."""

    def __init__(
        self,
        camera_id: str,
        frame_reader: Iterator[tuple[int, float, np.ndarray]],
        *,
        skip_frame_num: int = 0,
        conf_threshold: float = 0.4,
        smoke_test_mode: bool = True,
        tracker_factory: Optional[Callable[..., Any]] = None,
        detector: Optional[PPHumanDetectorAdapter] = None,
        detector_factory: Optional[Callable[..., Any]] = None,
        mode: Optional[RuntimeMode] = None,
    ) -> None:
        self.camera_id = camera_id
        self._frame_reader = frame_reader
        self.skip_frame_num = skip_frame_num
        self.conf_threshold = conf_threshold
        # Backward-compat: the existing API allows `smoke_test_mode=True`
        # to opt into the synthetic detector. In the new world that
        # implies `RuntimeMode.SMOKE_TEST`.
        if mode is None:
            if smoke_test_mode:
                self._mode = RuntimeMode.SMOKE_TEST
            else:
                self._mode = resolve_runtime_mode()
        else:
            self._mode = mode
        # The synthetic / real path is decided by `detector` and `mode`:
        #   detector is None        -> production: refuse, smoke: synthetic
        #   detector is provided    -> use it (real or smoke synthetic)
        self._detector = detector
        self._tracker_factory = tracker_factory
        self._detector_factory = detector_factory
        # If the operator did not pass a detector, and we are in smoke
        # mode, we use the synthetic fallback for backward compatibility
        # with the existing test suite.
        self._smoke_synthetic = (
            self._detector is None
            and self._detector_factory is None
            and self._mode == RuntimeMode.SMOKE_TEST
        )
        if self._smoke_synthetic:
            smoke_log(
                "PPHumanWorker", f"camera={self.camera_id} using synthetic detector (SMOKE-TEST)"
            )
        self._frame_counter = 0
        self._next_local_id = 1
        self._tracks: dict[int, LocalTrack] = {}
        self._last_emit_frame: dict[int, int] = {}

    def _next_id(self) -> int:
        i = self._next_local_id
        self._next_local_id += 1
        return i

    def _synthetic_detect(self, frame: np.ndarray) -> list[LocalTrack]:
        """Smoke-test detector. Mirrors the previous behaviour for
        backward compatibility with the existing test suite. NEVER used
        in production.
        """
        h, w = frame.shape[:2]
        out: list[LocalTrack] = []
        rng = np.random.default_rng(
            seed=hash((self.camera_id, self._frame_counter)) & 0xFFFFFFFF,
        )
        n = int(rng.integers(0, 3))
        for _ in range(n):
            x1 = float(rng.uniform(0.0, 0.7) * w)
            y1 = float(rng.uniform(0.0, 0.7) * h)
            x2 = x1 + float(rng.uniform(40, 120))
            y2 = y1 + float(rng.uniform(80, 240))
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            out.append(
                LocalTrack(
                    camera_id=self.camera_id,
                    local_track_id=self._next_id(),
                    bbox=(x1, y1, x2, y2),
                    confidence=float(rng.uniform(0.5, 0.95)),
                    class_id=0,
                    frame_id=self._frame_counter,
                    ts=time.time(),
                    age_frames=0,
                    is_confirmed=False,
                )
            )
        return out

    def _detector_dets(self, frame: np.ndarray) -> list[LocalTrack]:
        """Run the detector (real or synthetic) and produce LocalTracks.

        In production, ``self._detector_factory`` is the per-frame
        callable backed by a real :class:`PPHumanFrameStateAdapter`
        (which in turn tails the official PP-Human subprocess
        manager and returns MOT-format detections for the current
        frame). The ``self._detector`` slot is reserved for
        legacy / unit-test wiring; in production we always go
        through ``detector_factory``.

        In smoke-test mode without a real detector, we use
        :meth:`_synthetic_detect`.
        """
        if self._detector is not None and self._detector_factory is None:
            # Refuse rather than crash: a caller that passed a
            # bare ``PPHumanDetectorAdapter`` instance forgot to
            # provide the per-frame callable bridge. Production
            # refuses to start in this state.
            assert_production_safe(
                mode=self._mode,
                component="PPHumanWorker",
                condition=(
                    "PPHumanDetectorAdapter was passed without a "
                    "detector_factory; production must wire the "
                    "subprocess-backed per-frame factory."
                ),
            )
            smoke_log(
                "PPHumanWorker",
                "detector=adapter with no factory in SMOKE-TEST; "
                "falling back to synthetic (this hides a wiring bug; "
                "fix the call site).",
            )
            return self._synthetic_detect(frame)
        if self._detector_factory is not None:
            # Real path: pass the current frame_id so the
            # subprocess-backed adapter can look up the
            # corresponding MOT detections.
            try:
                raw = self._detector_factory(frame, self._frame_counter)
            except TypeError:
                # Backward compat: legacy single-arg factory.
                raw = self._detector_factory(frame)
            out: list[LocalTrack] = []
            for r in raw:
                out.append(
                    LocalTrack(
                        camera_id=self.camera_id,
                        local_track_id=self._next_id(),
                        bbox=tuple(r.bbox),
                        confidence=float(r.confidence),
                        class_id=0,
                        frame_id=self._frame_counter,
                        ts=time.time(),
                        age_frames=0,
                        is_confirmed=False,
                    )
                )
            return out
        # Smoke-test synthetic fallback.
        return self._synthetic_detect(frame)

    def _update_tracks(
        self,
        detections: list[LocalTrack],
        frame_id: int,
        ts: float,
    ) -> list[LocalTrack]:
        """Naive IoU tracker (replaced by OC-SORT in the subprocess path).

        In production with the real PP-Human pipeline subprocess, the
        tracker is OC-SORT and the MOT output already has stable
        ``track_id`` values. The parent's ``MultiCameraRunner`` matches
        those into :class:`LocalTrack` instances via
        ``detector_factory``. This IoU tracker remains the smoke-test
        default.
        """
        active_ids = set(self._tracks.keys())
        new_track_assignments: list[tuple[LocalTrack, int]] = []
        for det in detections:
            best_iou, best_id = 0.0, None
            for tid, tr in self._tracks.items():
                ax1, ay1, ax2, ay2 = tr.bbox
                bx1, by1, bx2, by2 = det.bbox
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
                area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
                union = area_a + area_b - inter
                iou = inter / max(union, 1e-6)
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_iou > 0.2:
                new_track_assignments.append((det, best_id))
            else:
                new_track_assignments.append((det, det.local_track_id))

        out: list[LocalTrack] = []
        used_ids: set[int] = set()
        for det, tid in new_track_assignments:
            if tid in active_ids and tid in used_ids:
                continue
            tr = self._tracks.get(tid)
            if tr is None:
                tr = LocalTrack(
                    camera_id=self.camera_id,
                    local_track_id=tid,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    frame_id=frame_id,
                    ts=ts,
                    age_frames=1,
                    is_confirmed=False,
                )
                self._tracks[tid] = tr
            else:
                tr.bbox = det.bbox
                tr.confidence = det.confidence
                tr.frame_id = frame_id
                tr.ts = ts
                tr.age_frames += 1
                if tr.age_frames >= 3:
                    tr.is_confirmed = True
            used_ids.add(tid)
            out.append(tr)
        for tid in list(self._tracks.keys()):
            if tid not in used_ids and (frame_id - self._tracks[tid].frame_id) > 30:
                del self._tracks[tid]
        return out

    def run(
        self,
        max_frames: Optional[int] = None,
        max_seconds: Optional[float] = None,
    ) -> Iterator[FrameResult]:
        """Yield FrameResult for each processed frame.

        Production mode refuses to start without a real detector.
        Smoke-test mode allows the synthetic fallback.

        PATCH-032: when the underlying frame reader yields a None
        frame (the camera is offline/degraded), we propagate a
        ``FrameResult(frame=None, tracks=[])`` so the consumer can
        update per-camera status without crashing.
        """
        if not self._smoke_synthetic and self._detector is None and self._detector_factory is None:
            assert_production_safe(
                mode=self._mode,
                component="PPHumanWorker",
                condition="no detector adapter or factory provided",
            )
        start = time.time()
        for frame_id, ts, frame in self._frame_reader:
            self._frame_counter += 1
            # PATCH-032: skip None frames (camera is offline) but
            # still emit a FrameResult so the consumer knows the
            # camera is alive (just frame-less this tick).
            if frame is None:
                yield FrameResult(
                    camera_id=self.camera_id,
                    frame_id=frame_id,
                    ts=ts,
                    frame=None,
                    tracks=[],
                    skipped=True,
                )
                continue
            if self.skip_frame_num and (self._frame_counter % (self.skip_frame_num + 1) != 0):
                yield FrameResult(
                    camera_id=self.camera_id,
                    frame_id=frame_id,
                    ts=ts,
                    frame=frame,
                    tracks=[],
                    skipped=True,
                )
                continue
            if self._smoke_synthetic or (self._detector is None and self._detector_factory is None):
                detections = self._synthetic_detect(frame)
            else:
                detections = self._detector_dets(frame)
            tracks = self._update_tracks(detections, frame_id, ts)
            yield FrameResult(
                camera_id=self.camera_id,
                frame_id=frame_id,
                ts=ts,
                frame=frame,
                tracks=tracks,
                skipped=False,
            )
            if max_frames is not None and self._frame_counter >= max_frames:
                break
            if max_seconds is not None and (time.time() - start) > max_seconds:
                break
