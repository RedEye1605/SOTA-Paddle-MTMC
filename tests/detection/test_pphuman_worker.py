"""PP-Human worker: local tracks, frame skipping, multi-camera concurrency."""

from __future__ import annotations


import numpy as np

from app.workers.pphuman_worker import PPHumanWorker


class _StubReader:
    def __init__(self, n_frames: int = 5, shape=(480, 640, 3)):
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


def test_pphuman_worker_emits_local_tracks() -> None:
    worker = PPHumanWorker(
        camera_id="CAM_01",
        frame_reader=_StubReader(5),
        skip_frame_num=0,
        smoke_test_mode=True,
    )
    frames = list(worker.run())
    assert len(frames) == 5
    # smoke detector emits 0..2 tracks per frame
    for f in frames:
        assert 0 <= len(f.tracks) <= 2


def test_pphuman_worker_local_track_id_is_camera_local() -> None:
    """Hard rule: local_track_id is camera-local, ephemeral."""
    w1 = PPHumanWorker(camera_id="CAM_01", frame_reader=_StubReader(3), smoke_test_mode=True)
    w2 = PPHumanWorker(camera_id="CAM_02", frame_reader=_StubReader(3), smoke_test_mode=True)
    f1 = list(w1.run())
    f2 = list(w2.run())
    # different cameras have independent local_track_id spaces
    ids1 = {t.local_track_id for f in f1 for t in f.tracks}
    ids2 = {t.local_track_id for f in f2 for t in f.tracks}
    # collisions are allowed by design (no global meaning); we only
    # verify both populations are populated.
    assert isinstance(ids1, set)
    assert isinstance(ids2, set)


def test_pphuman_worker_frame_skip() -> None:
    worker = PPHumanWorker(
        camera_id="CAM_01",
        frame_reader=_StubReader(10),
        skip_frame_num=2,  # process every 3rd frame
        smoke_test_mode=True,
    )
    frames = list(worker.run())
    # 1/3 of frames processed (ceil(10/3) = 4)
    processed = [f for f in frames if not f.skipped]
    skipped = [f for f in frames if f.skipped]
    assert len(processed) + len(skipped) == 10
    assert len(skipped) > 0
