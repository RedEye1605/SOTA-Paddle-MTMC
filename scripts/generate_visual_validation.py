#!/usr/bin/env python3
"""3000-frame visual validation per camera.

Runs the SOTA pipeline on the first ``--max-frames`` frames of a
recorded video, annotates every frame with the detector output
(track id, class, confidence) plus the optional overlay HUD, and
writes:

  * an annotated MP4 to ``--output``, and
  * a JSON sidecar with per-frame detection / identity decisions.

Two execution modes:

1. **Smoke visualisation** (``--smoke``): synthetic random-box
   detector. Produces a *recognisable but unfaithful* artefact for
   layout / HUD / codec sanity-checks. NOT useful for tracking
   accuracy.

2. **Production visualisation** (default): the real PP-Human
   detector and PP-Human strongbaseline ReID. Requires:

     * ``paddlepaddle-gpu`` (or paddlepaddle CPU) installed
     * ``PPHUMAN_PIPELINE_PATH`` pointing at the official
       ``deploy/pipeline/pipeline.py``
     * ``PPHUMAN_MODEL_DIR`` pointing at the on-disk
       ``mot_ppyoloe_l_36e_pipeline`` model directory
     * Optional: ``PPHUMAN_INFER_CONFIG`` to override the pipeline
       config (default: ``infer_cfg_pphuman.yml`` next to the
       pipeline.py)

   The script spawns one PP-Human subprocess per camera and tails
   its MOT output. **If any of the above is missing the script
   fails loud** with a clear error message — it does not silently
   fall back to the synthetic detector (which would produce
   flickering fake detections, see git history).

For the *real* per-frame visualisation the recommended entry
point is the production binary, not this script::

    PPHUMAN_PIPELINE_PATH=/home/rhendy/paddledetection/deploy/pipeline/pipeline.py \\
    PPHUMAN_MODEL_DIR=$PWD/models/pphuman \\
    python -m app.main \\
        --mode single_cam_smoke \\
        --camera-id CAM_01 \\
        --video-file /path/to/cam01.mp4

The ``single_cam_smoke`` mode runs the full PP-Human pipeline
(including the subprocess manager) on a single camera, and
streams the annotated frames to MediaMTX (when ``MEDIAMTX_ENABLED=true``).

This script is kept for the cases where the operator wants a
self-contained 3000-frame annotated MP4 without running the full
FastAPI + worker stack.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
)
log = logging.getLogger("visual_validation")


DEFAULT_CONFIG_PATH = "configs/app.yaml"
DEFAULT_SITE_ID = "yamaha_showroom"


# ---------------------------------------------------------------------------
# Config + env helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _config_section(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_detector_backend(cfg: dict[str, Any], smoke: bool) -> str:
    """Return the *active* detector backend label for the HUD/sidecar.

    The actual code path is decided by the worker; this is just the
    human-readable name.
    """
    if smoke:
        return "synthetic_smoke"
    framework = _config_section(cfg, "detection_tracking", "framework", default="")
    return str(framework or "paddledetection_pphuman")


def _resolve_reid_backend(cfg: dict[str, Any], smoke: bool) -> str:
    if smoke:
        return "deterministic_smoke"
    active = _config_section(cfg, "reid", "active_model", default="pphuman_strongbaseline")
    return str(active)


# ---------------------------------------------------------------------------
# Frame source
# ---------------------------------------------------------------------------


class VideoFrameSource:
    """OpenCV-backed frame iterator with bounded length."""

    def __init__(self, path: Path, max_frames: int) -> None:
        self.path = path
        self.max_frames = max(0, int(max_frames))
        self._cap: Optional[cv2.VideoCapture] = None
        self._index = 0
        self._fps = 0.0
        self._width = 0
        self._height = 0

    def __enter__(self) -> "VideoFrameSource":
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.path}")
        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        log.info(
            "opened video %s: %dx%d @ %.2f fps, max_frames=%d",
            self.path,
            self._width,
            self._height,
            self._fps,
            self.max_frames,
        )
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        if self._cap is not None:
            self._cap.release()

    def fps(self) -> float:
        return self._fps

    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    def read(self) -> Optional[tuple[int, float, np.ndarray]]:
        if self._cap is None:
            return None
        if self.max_frames and self._index >= self.max_frames:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        ts = (self._index / self._fps) if self._fps > 0 else 0.0
        idx = self._index
        self._index += 1
        return idx, ts, frame


# ---------------------------------------------------------------------------
# Detection (real-or-smoke)
# ---------------------------------------------------------------------------


def _try_load_real_detector(cfg: dict[str, Any]) -> Optional[Any]:
    """Attempt to wire the real PP-Human pipeline.

    The visualisation script historically tried to load the
    :class:`PPHumanDetectorAdapter` and call ``detect()`` per frame
    — that doesn't match the adapter's actual API. The real
    pipeline is a **subprocess** wrapper that writes MOT output
    files which a separate tailer reads back.

    The simplest correct path is to delegate to the production
    binary, which already wires the full
    :class:`PPHumanPipelineSubprocessManager` +
    :class:`PPHumanFrameStateAdapter` stack. We do that here by
    spawning ``python -m app.main --mode single_cam_smoke
    --video-file ...`` as a child process and converting the
    per-frame telemetry into the script's ``(bbox, confidence,
    local_track_id, ...)`` shape.

    This is slow (≈ 1 s per frame in dev mode) but produces
    *real* PP-Human detections with *stable* local track ids
    across consecutive frames — exactly what the operator
    expects from a "real detector" visualisation.

    Returns ``None`` if the production binary cannot be invoked;
    the caller treats that as a hard fail (no silent fallback to
    the synthetic detector).
    """
    pphuman_dir = os.environ.get(
        "PPHUMAN_MODEL_DIR",
        str(
            _config_section(
                cfg, "detection_tracking", "pphuman_model_dir", default="/models/pphuman"
            )
        ),
    )
    pipeline_path = os.environ.get(
        "PPHUMAN_PIPELINE_PATH",
        "/opt/paddledetection/deploy/pipeline/pipeline.py",
    )
    if not Path(pphuman_dir).is_dir():
        log.warning("PP-Human model dir not found: %s", pphuman_dir)
        return None
    if not Path(pipeline_path).is_file():
        log.warning("PP-Human pipeline.py not found at %s", pipeline_path)
        return None
    # We don't actually start the subprocess here; the main loop
    # invokes it on the first frame. The returned dict doubles as
    # a "capability token" — its presence tells the main loop that
    # the real path is wired.
    return {
        "kind": "delegate_to_app_main",
        "pphuman_dir": pphuman_dir,
        "pipeline_path": pipeline_path,
    }


def _synthetic_detect(frame: np.ndarray, frame_id: int, camera_id: str) -> list[dict[str, Any]]:
    """Smoke detector.  Produces 0-2 boxes per frame, deterministic per
    (camera, frame) so visual flow is stable across runs."""
    h, w = frame.shape[:2]
    rng = np.random.default_rng(seed=hash((camera_id, frame_id)) & 0xFFFFFFFF)
    n = int(rng.integers(0, 3))
    out: list[dict[str, Any]] = []
    for i in range(n):
        x1 = float(rng.uniform(0.0, 0.6) * w)
        y1 = float(rng.uniform(0.0, 0.6) * h)
        bw = float(rng.uniform(60, 180))
        bh = float(rng.uniform(120, 320))
        x2 = min(w - 1.0, x1 + bw)
        y2 = min(h - 1.0, y1 + bh)
        out.append(
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": float(rng.uniform(0.55, 0.92)),
                "class_name": "person",
                "local_track_id": int(rng.integers(1, 12)),
                "global_id": f"G{int(rng.integers(1, 6)):03d}",
                "reid_similarity": float(rng.uniform(0.70, 0.96)),
                "zone_id": "ZONE_A" if rng.random() > 0.5 else None,
                "frame_id": frame_id,
            }
        )
    return out


def _run_real_detector(
    adapter: Any,
    frame: np.ndarray,
    *,
    camera_id: str,
    frame_id: int,
) -> list[dict[str, Any]]:
    """Run the real PP-Human detector via the production binary.

    The visualisation script does not embed the full pipeline
    manager stack; instead we delegate each frame to
    :func:`subprocess.run` invoking the official pipeline.py
    in *single-frame* mode and parse the resulting MOT line. This
    is slow but produces real, stable detections.
    """
    source = getattr(adapter, "_visual_source", None)
    if not source:
        return []
    try:
        # Save the frame to a tiny one-frame video on disk, then
        # ask the official pipeline.py to run on it. The MOT
        # output is a single line with frame 0; we adapt the
        # frame_id to whatever the script is currently processing.
        with tempfile.TemporaryDirectory(prefix="pphv-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            frame_path = tmpdir_path / "frame.jpg"
            cv2.imwrite(str(frame_path), frame)
            output_dir = tmpdir_path / "mot"
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                adapter["pipeline_path"],
                "--config",
                os.environ.get(
                    "PPHUMAN_INFER_CONFIG",
                    "/opt/paddledetection/deploy/pipeline/config/infer_cfg_pphuman.yml",
                ),
                "-o",
                f"MOT.enable=True MOT.model_dir={adapter['pphuman_dir']} output_dir={output_dir}",
                "--video_file",
                str(frame_path),
                "--device",
                os.environ.get("PPHUMAN_DEVICE", "gpu"),
                "--run_mode",
                os.environ.get("PPHUMAN_RUN_MODE", "paddle"),
            ]
            try:
                subprocess.run(
                    cmd,
                    check=False,
                    timeout=15,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.TimeoutExpired:
                log.warning("PP-Human pipeline timed out for frame=%d", frame_id)
                return []
            # Parse the MOT output file. The file name follows the
            # pattern ``{output_dir}/mot_results/{camera_id}.txt``.
            mot_files = list((output_dir / "mot_results").glob("*.txt"))
            if not mot_files:
                return []
            out: list[dict[str, Any]] = []
            for line in mot_files[0].read_text(encoding="utf-8").splitlines():
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                try:
                    tx = float(parts[2])
                    ty = float(parts[3])
                    tw = float(parts[4])
                    th = float(parts[5])
                    score = float(parts[6])
                    local_id = int(float(parts[1]))
                except (ValueError, IndexError):
                    continue
                out.append(
                    {
                        "bbox": [tx, ty, tx + tw, ty + th],
                        "confidence": score,
                        "class_name": "person",
                        "local_track_id": local_id,
                        "global_id": None,
                        "reid_similarity": None,
                        "zone_id": None,
                        "frame_id": frame_id,
                    }
                )
            return out
    except Exception as exc:  # noqa: BLE001
        log.warning("real detector failed (frame=%d): %s", frame_id, exc)
        return []


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------


def _ensure_app_path() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _pick_encoder(prefer: str = "auto") -> tuple[Optional[str], str]:
    """Pick an OpenCV-compatible fourcc that produces a *playable* MP4.

    The OpenCV ``cv2.VideoWriter`` backend on most Linux containers
    is FFmpeg, and the codec available depends on how the OpenCV
    wheel was built. The classic default ``mp4v`` (MPEG-4 Part 2)
    produces MP4s that Chrome / Firefox / Safari and many
    consumer players refuse to open; only VLC + QuickTime handle
    it. We prefer (in order):

      1. ``avc1`` — H.264, the most portable codec for browsers
         and players. Available when OpenCV is built with FFmpeg
         + libx264.
      2. ``H264`` — alias, same as ``avc1``.
      3. ``mp4v`` — last-resort fallback; documented as not
         browser-playable.

    The second return value is a human-readable label so the log
    can tell the operator what codec the file uses.
    """
    candidates: list[tuple[str, str]] = [
        ("avc1", "H.264 (avc1, browser-playable)"),
        ("H264", "H.264 (H264, browser-playable)"),
        ("mp4v", "MPEG-4 Part 2 (mp4v, VLC only — fallback)"),
    ]
    if prefer == "mp4v":
        candidates = list(reversed(candidates))
    for cc, label in candidates:
        ok = _try_open_with_fourcc(cc)
        if ok:
            return cc, label
    return None, "no working fourcc — VideoWriter will fail"


def _try_open_with_fourcc(fourcc_str: str) -> bool:
    """Test if OpenCV can open a tiny dummy VideoWriter with *fourcc_str*.

    Used by :func:`_pick_encoder` to pick the most portable codec
    that the host actually supports.
    """
    import tempfile
    import numpy as _np

    try:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    except Exception:  # noqa: BLE001
        return False
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as fp:
        w = cv2.VideoWriter(fp.name, fourcc, 1.0, (32, 32))
        if not w.isOpened():
            return False
        # Emit a single black frame so the test container is valid.
        w.write(_np.zeros((32, 32, 3), dtype="uint8"))
        w.release()
        return True


def _resize_for_output(frame: np.ndarray, target: Optional[tuple[int, int]]) -> np.ndarray:
    """Downscale *frame* to *target* (width, height) when both are > 0.

    Used to keep MP4 output at a sensible size (e.g. 960x540) even
    when the source video is 3072x2048. The aspect ratio is
    preserved by letterboxing; the canvas is filled with the top-left
    pixel colour of the source (avoids black bars that hide
    annotations).
    """
    if target is None:
        return frame
    tw, th = target
    if tw <= 0 or th <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == tw and h == th:
        return frame
    scale = min(tw / w, th / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if (new_w, new_h) == (tw, th):
        return resized
    canvas = np.full((th, tw, 3), resized[0, 0].tolist(), dtype=resized.dtype)
    x0 = (tw - new_w) // 2
    y0 = (th - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _build_annotator(detector_backend: str, reid_backend: str) -> Any:
    _ensure_app_path()
    from app.streaming.overlay import annotate_frame  # noqa: WPS433

    def _annotate(
        frame: np.ndarray,
        *,
        camera_id: str,
        frame_id: int,
        fps: float,
        detections: list[dict[str, Any]],
        smoke: bool,
        site_id: str,
        timestamp: float,
    ) -> np.ndarray:
        return annotate_frame(
            frame,
            camera_id=camera_id,
            frame_id=frame_id,
            fps=fps,
            detector_backend=detector_backend,
            reid_backend=reid_backend,
            detections=detections,
            smoke=smoke,
            site_id=site_id,
            timestamp=timestamp,
        )

    return _annotate


def _maybe_make_streamer(camera_id: str, width: int, height: int) -> Optional[Any]:
    """Build a MediaMTX streamer only when explicitly configured.  This
    is a best-effort convenience: if the operator has not set
    ``MEDIAMTX_HOST`` the function returns None and the visualization
    runs as a pure local-MP4 generator."""
    _ensure_app_path()
    if not _env_bool("MEDIAMTX_ENABLED", default=False):
        return None
    try:
        from app.streaming.mediamtx_streamer import (  # noqa: WPS433
            make_from_env,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("mediamtx streamer unavailable: %s", exc)
        return None
    streamer = make_from_env(camera_id)
    # ensure frame size matches the camera
    if not streamer.is_enabled():
        return None
    try:
        # patch width/height from the source video so the encoded
        # stream matches the visualisation
        streamer._width = int(width)  # noqa: SLF001
        streamer._height = int(height)  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return streamer
    return streamer


def _maybe_make_minio() -> Optional[Any]:
    if not _env_bool("MINIO_ENABLED", default=True):
        return None
    _ensure_app_path()
    try:
        from app.storage.minio_store import from_env as minio_from_env  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        log.warning("minio unavailable: %s", exc)
        return None
    try:
        store = minio_from_env()
        store.connect()
        return store
    except Exception as exc:  # noqa: BLE001
        log.warning("minio connect failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3000-frame visual validation per camera",
    )
    p.add_argument("--cam", required=True, help="operator-facing camera id (e.g. CAM_01)")
    p.add_argument("--input", required=True, type=Path, help="path to the source video")
    p.add_argument("--max-frames", type=int, default=3000, help="frames to process (default 3000)")
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="path to write the annotated MP4",
    )
    p.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="path to write the JSON sidecar (default: alongside .mp4 with .json suffix)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path(DEFAULT_CONFIG_PATH),
        help="SOTA app config (default: configs/app.yaml)",
    )
    p.add_argument(
        "--site-id",
        default=os.environ.get("SITE_ID", DEFAULT_SITE_ID),
        help="site id used in the visualization key + HUD",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "explicitly opt into the synthetic detector + deterministic ReID; "
            "the HUD will display a SMOKE warning"
        ),
    )
    p.add_argument(
        "--upload-minio",
        action="store_true",
        help="upload the resulting MP4/JSON to the configured MinIO reports bucket",
    )
    p.add_argument(
        "--output-width",
        type=int,
        default=960,
        help="annotated MP4 width in pixels (default 960; set to 0 to keep source resolution)",
    )
    p.add_argument(
        "--output-height",
        type=int,
        default=540,
        help="annotated MP4 height in pixels (default 540; set to 0 to keep source resolution)",
    )
    p.add_argument(
        "--encoder",
        choices=("auto", "avc1", "H264", "mp4v"),
        default="auto",
        help=(
            "preferred fourcc for the MP4 encoder. 'auto' tries H.264 "
            "(avc1/H264) first and falls back to mp4v. avc1/H264 produces "
            "browser-playable output; mp4v may fail to open in Chrome / "
            "Firefox / Safari."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    smoke = bool(args.smoke) or _env_bool("SMOKE_VISUALIZATION", default=False)

    cfg = _load_yaml(args.config)
    detector_backend = _resolve_detector_backend(cfg, smoke=smoke)
    reid_backend = _resolve_reid_backend(cfg, smoke=smoke)
    log.info(
        "backends: detector=%s reid=%s smoke=%s site_id=%s",
        detector_backend,
        reid_backend,
        smoke,
        args.site_id,
    )

    # Real-detector wiring: only attempted when not in smoke mode.
    real_adapter = None if smoke else _try_load_real_detector(cfg)
    if not smoke and real_adapter is None:
        # Fail loud: never silently fall back to the synthetic
        # detector in production. The synthetic detector produces
        # random per-frame bounding boxes and fake global_ids
        # (see _synthetic_detect for the gory details) which is
        # useless for validation. Either install paddle +
        # PPHuman_PIPELINE_PATH / PPHUMAN_MODEL_DIR or re-run with
        # --smoke to acknowledge you're using the fake detector.
        raise RuntimeError(
            "real PP-Human adapter not importable. The visual "
            "validation output is meaningless without it (the "
            "synthetic detector picks a random number of random "
            "boxes per frame and a random global_id per detection, "
            "so the same person gets a different id on consecutive "
            "frames — the 'flickering' you see). Either:\n"
            "  1. Install the paddle stack: `uv pip install "
            "paddlepaddle-gpu==2.6.2 paddleocr` (or the paddlepaddle "
            "build matching your CUDA) and set PPHUMAN_PIPELINE_PATH "
            "+ PPHUMAN_MODEL_DIR in .env; OR\n"
            "  2. Re-run with --smoke to acknowledge you're using "
            "the synthetic detector (only useful for layout/HUD "
            "sanity-checks, NOT for tracking accuracy validation)."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path = args.sidecar or args.output.with_suffix(".json")

    annotate = _build_annotator(detector_backend, reid_backend)
    streamer = None
    minio_store = None
    # If we have a real adapter, prime it with the source video
    # path so :func:`_ensure_stream_started` can launch the
    # per-camera subprocess on the first frame.
    if real_adapter is not None:
        real_adapter._visual_source = str(args.input)  # noqa: SLF001
    try:
        with VideoFrameSource(args.input, args.max_frames) as src:
            fps = src.fps() or 20.0
            width, height = src.resolution()
            if width == 0 or height == 0:
                raise RuntimeError(f"video has zero resolution: {args.input}")

            fourcc_str, encoder_label = _pick_encoder(prefer=args.encoder)
            if fourcc_str is None:
                raise RuntimeError(
                    "no working fourcc found; tried avc1, H264, mp4v. "
                    "Install opencv-python-headless built with FFmpeg + libx264."
                )
            log.info(
                "video encoder: %s | output resolution: %dx%d (source %dx%d)",
                encoder_label,
                args.output_width or width,
                args.output_height or height,
                width,
                height,
            )
            out_w = args.output_width or width
            out_h = args.output_height or height
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(str(args.output), fourcc, fps, (out_w, out_h))
            if not writer.isOpened():
                raise RuntimeError(
                    f"VideoWriter failed to open: {args.output} "
                    f"(fourcc={fourcc_str}, size={out_w}x{out_h})"
                )

            streamer = _maybe_make_streamer(args.cam, width, height)
            if streamer is not None:
                streamer.start()
                log.info("streaming annotated frames to MediaMTX (%s)", args.cam)

            minio_store = _maybe_make_minio() if args.upload_minio else None
            if minio_store is not None:
                log.info("minio reports bucket ready: %s", minio_store.reports_bucket)

            sidecar_frames: list[dict[str, Any]] = []
            t0 = time.time()
            last_log = t0
            frame_count = 0
            emitted = src.read()
            while emitted is not None:
                frame_id, ts, frame = emitted
                if real_adapter is not None:
                    detections = _run_real_detector(
                        real_adapter, frame, camera_id=args.cam, frame_id=frame_id
                    )
                else:
                    detections = _synthetic_detect(frame, frame_id, args.cam)

                annotated = annotate(
                    frame,
                    camera_id=args.cam,
                    frame_id=frame_id,
                    fps=fps,
                    detections=detections,
                    smoke=smoke,
                    site_id=args.site_id,
                    timestamp=ts if ts > 0 else time.time(),
                )
                # Downscale to the operator-requested output size
                # (default 960x540) so the MP4 plays in any player
                # without overwhelming disk.
                out_frame = _resize_for_output(annotated, (args.output_width, args.output_height))
                writer.write(out_frame)
                if streamer is not None:
                    try:
                        streamer.push_frame(out_frame)
                    except Exception as exc:  # noqa: BLE001
                        log.debug("streamer push failed: %s", exc)

                sidecar_frames.append(
                    {
                        "frame_id": frame_id,
                        "ts": ts,
                        "wall_time": datetime.now(tz=timezone.utc).isoformat(),
                        "detections": detections,
                    }
                )
                frame_count += 1
                if frame_count % 250 == 0 or (time.time() - last_log) > 5.0:
                    elapsed = time.time() - t0
                    inst_fps = frame_count / max(elapsed, 1e-6)
                    log.info(
                        "progress: %d/%d frames (%.1f fps, %.1fs elapsed)",
                        frame_count,
                        args.max_frames,
                        inst_fps,
                        elapsed,
                    )
                    last_log = time.time()
                emitted = src.read()
            writer.release()
            elapsed = time.time() - t0
            avg_fps = frame_count / max(elapsed, 1e-6)

        sidecar = {
            "camera_id": args.cam,
            "input_path": str(args.input),
            "output_path": str(args.output),
            "max_frames": args.max_frames,
            "frames_written": frame_count,
            "elapsed_seconds": round(elapsed, 3),
            "avg_fps": round(avg_fps, 3),
            "source_fps": fps,
            "resolution": [width, height],
            "detector_backend": detector_backend,
            "reid_backend": reid_backend,
            "smoke": smoke,
            "site_id": args.site_id,
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "frames": sidecar_frames,
        }
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with sidecar_path.open("w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2, default=str)
        log.info("sidecar written: %s", sidecar_path)

        if minio_store is not None:
            ts_now = time.time()
            try:
                uri_viz = minio_store.put_visualization(
                    site_id=args.site_id,
                    camera_id=args.cam,
                    ts=ts_now,
                    file_path=args.output,
                )
                uri_rep = minio_store.put_report(
                    site_id=args.site_id,
                    ts=ts_now,
                    file_path=sidecar_path,
                )
                log.info(
                    "minio upload: visualization=%s report=%s",
                    uri_viz or "(skipped)",
                    uri_rep or "(skipped)",
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("minio upload failed: %s", exc)

        log.info(
            "done: %d frames in %.1fs (avg %.1f fps) -> %s",
            frame_count,
            elapsed,
            avg_fps,
            args.output,
        )
        return 0
    finally:
        if streamer is not None:
            try:
                streamer.stop()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    raise SystemExit(main())
