"""Per-camera MediaMTX streamer backed by an ffmpeg subprocess (Phase 6).

Pattern adapted from ``Service/offline-people-counting``:

* Bounded ``queue.Queue(maxsize=2)`` of BGR frames.
* Daemon push thread drains the queue at the configured fps.
* Raw BGR bytes are written to ffmpeg's stdin.
* ffmpeg encodes with libx264 (zerolatency, ultrafast) and pushes
  RTSP/TCP to MediaMTX.
* Reconnect-with-exponential-backoff if ffmpeg dies.
* Public ``is_running()``, ``push_frame()``, ``stop()`` lifecycle.
* Public ``stream_urls()`` returns the consumer-facing HLS/WebRTC
  URLs the browser can subscribe to.

The streamer is **opt-in**: pass ``enabled=False`` and it is a
complete no-op (no ffmpeg subprocess is spawned, no thread is
started). This lets the visual-validation script run in CI without
a MediaMTX instance.
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from typing import Optional

import numpy as np

from .ffmpeg_writer import (
    build_ffmpeg_command,
    build_hls_url,
    build_publish_url,
    build_webrtc_url,
)

logger = logging.getLogger(__name__)


# Tunables. Lifted to module level so tests can override.
DEFAULT_QUEUE_MAXSIZE = 2
FRAME_STALE_AFTER_SECONDS = 5.0
RECONNECT_BACKOFF_BASE_SECONDS = 2.0
RECONNECT_BACKOFF_MAX_SECONDS = 60.0
FFMPEG_STOP_TIMEOUT_SECONDS = 5.0
PUSH_LOOP_TICK_SECONDS = 0.05


class MediaMTXStreamer:
    """Per-camera MediaMTX streamer backed by an ffmpeg subprocess."""

    def __init__(
        self,
        camera_id: str,
        width: int,
        height: int,
        *,
        fps: int = 10,
        bitrate_kbps: int = 1800,
        host: str = "",
        rtsp_port: int = 8554,
        hls_port: int = 8889,
        webrtc_port: int = 8890,
        stream_prefix: str = "sota-paddle-mtmc",
        publish_url_template: Optional[str] = None,
        ffmpeg_bin: str = "ffmpeg",
        max_reconnect_attempts: int = 5,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        enabled: bool = True,
    ) -> None:
        self._camera_id = camera_id
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._bitrate_kbps = int(bitrate_kbps)
        self._host = host
        self._hls_port = int(hls_port)
        self._webrtc_port = int(webrtc_port)
        self._stream_prefix = stream_prefix
        self._publish_url_template = publish_url_template
        self._ffmpeg_bin = ffmpeg_bin
        self._max_reconnect = int(max_reconnect_attempts)
        self._queue_maxsize = max(1, int(queue_maxsize))
        self._enabled = bool(enabled) and bool(host)

        # If the operator passed ``enabled=True`` but left the host
        # empty, surface the reason on construction so callers (and
        # the disabled-safe tests) can introspect it without
        # having to call ``start()``.
        if bool(enabled) and not bool(host):
            self._stop_reason: Optional[str] = "host_unset"

        # Publish URL is computed once and logged on start.
        self._output_url = build_publish_url(
            template=publish_url_template,
            host=host,
            rtsp_port=rtsp_port,
            prefix=stream_prefix,
            camera_id=camera_id,
        )

        # Runtime state.
        self._process: Optional[subprocess.Popen] = None
        self._running = False
        self._reconnecting = False
        self._reconnect_attempts = 0
        self._last_frame_time = 0.0
        self._frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=self._queue_maxsize)
        self._frame_interval = 1.0 / max(self._fps, 1)
        self._lock = threading.Lock()
        self._push_thread: Optional[threading.Thread] = None
        # ``_stop_reason`` is initialized above when host is empty;
        # otherwise it's set lazily by ``start()`` on failure paths.
        if not hasattr(self, "_stop_reason"):
            self._stop_reason: Optional[str] = None

    # ---- lifecycle ----
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.stop()

    def start(self) -> None:
        """Spawn ffmpeg + the push thread. No-op if disabled or already running."""
        with self._lock:
            if self._running:
                return
            if not self._enabled:
                logger.info(
                    "mediamtx streamer disabled (camera=%s host=%r)",
                    self._camera_id,
                    self._host,
                )
                return
            if not self._host:
                logger.warning(
                    "MEDIAMTX_HOST is empty; streamer disabled (camera=%s)",
                    self._camera_id,
                )
                self._stop_reason = "host_unset"
                return
            process = self._spawn_ffmpeg()
            if process is None:
                self._stop_reason = "ffmpeg_not_found"
                return
            self._process = process
            self._running = True
            self._push_thread = threading.Thread(
                target=self._push_loop,
                daemon=True,
                name=f"mediamtx-streamer-{self._camera_id}",
            )
            self._push_thread.start()
            logger.info(
                "mediamtx streamer started | camera=%s | url=%s | %dx%d | %dfps",
                self._camera_id,
                self._output_url,
                self._width,
                self._height,
                self._fps,
            )

    def stop(self) -> None:
        """Stop the push thread, close ffmpeg stdin, terminate the process."""
        with self._lock:
            if not self._running and self._process is None:
                return
            self._running = False
        if self._process:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
                self._process.wait(timeout=FFMPEG_STOP_TIMEOUT_SECONDS)
            except Exception as exc:  # noqa: BLE001
                logger.debug("ffmpeg stop error: %s", exc)
                try:
                    self._process.kill()
                except Exception:  # noqa: BLE001
                    pass
            finally:
                self._process = None
        if self._push_thread and self._push_thread.is_alive():
            self._push_thread.join(timeout=PUSH_LOOP_TICK_SECONDS * 4)
        self._push_thread = None
        logger.info("mediamtx streamer stopped | camera=%s", self._camera_id)

    def is_running(self) -> bool:
        return self._running

    def is_enabled(self) -> bool:
        return self._enabled

    def stop_reason(self) -> Optional[str]:
        return self._stop_reason

    # ---- producer side ----
    def push_frame(self, frame: np.ndarray) -> None:
        """Non-blocking enqueue. Drops the previous frame if the consumer is slow."""
        if not self._running:
            return
        try:
            self._frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frame_queue.put_nowait(frame)
            self._last_frame_time = time.monotonic()
        except queue.Full:
            pass

    def stream_urls(self) -> dict[str, str]:
        """Public HLS / WebRTC URLs a consumer (browser) can subscribe to."""
        return {
            "rtsp": self._output_url,
            "hls": build_hls_url(
                host=self._host,
                hls_port=self._hls_port,
                prefix=self._stream_prefix,
                camera_id=self._camera_id,
            ),
            "webrtc": build_webrtc_url(
                host=self._host,
                webrtc_port=self._webrtc_port,
                prefix=self._stream_prefix,
                camera_id=self._camera_id,
            ),
        }

    # ---- internals ----
    def _spawn_ffmpeg(self) -> Optional[subprocess.Popen]:
        command = build_ffmpeg_command(
            ffmpeg_bin=self._ffmpeg_bin,
            width=self._width,
            height=self._height,
            fps=self._fps,
            bitrate_kbps=self._bitrate_kbps,
            output_url=self._output_url,
        )
        logger.info("ffmpeg argv: %s", " ".join(str(c) for c in command if c is not None))
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.error(
                "ffmpeg not found at %s; install ffmpeg before enabling streaming",
                self._ffmpeg_bin,
            )
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("failed to spawn ffmpeg: %s", exc)
            return None
        self._drain_stderr_in_background(process)
        return process

    def _drain_stderr_in_background(self, process: subprocess.Popen) -> None:
        def _drain() -> None:
            try:
                for line in iter(process.stderr.readline, b""):
                    if line:
                        logger.warning("ffmpeg: %s", line.decode().rstrip())
            except Exception:  # noqa: BLE001
                return

        threading.Thread(
            target=_drain,
            daemon=True,
            name=f"ffmpeg-stderr-{self._camera_id}",
        ).start()

    def _push_loop(self) -> None:
        last_frame: Optional[np.ndarray] = None
        next_push_at = time.monotonic()
        import cv2

        while self._running:
            now = time.monotonic()
            sleep_for = next_push_at - now
            if sleep_for > 0:
                time.sleep(min(sleep_for, PUSH_LOOP_TICK_SECONDS))
                continue
            next_push_at = max(next_push_at + self._frame_interval, time.monotonic())
            try:
                last_frame = self._frame_queue.get_nowait()
            except queue.Empty:
                if (
                    last_frame is not None
                    and time.monotonic() - self._last_frame_time > FRAME_STALE_AFTER_SECONDS
                ):
                    continue
            if last_frame is None:
                continue
            if self._process is None or self._process.poll() is not None:
                logger.warning(
                    "ffmpeg stopped; reconnecting (camera=%s attempt=%d)",
                    self._camera_id,
                    self._reconnect_attempts,
                )
                if not self._try_reconnect():
                    break
                next_push_at = time.monotonic() + self._frame_interval
                continue
            try:
                if last_frame.shape[1] != self._width or last_frame.shape[0] != self._height:
                    resized = cv2.resize(last_frame, (self._width, self._height))
                else:
                    resized = last_frame
                if self._process.stdin:
                    self._process.stdin.write(resized.tobytes())
                    self._process.stdin.flush()
            except BrokenPipeError:
                logger.warning("ffmpeg stdin pipe closed (camera=%s)", self._camera_id)
                if self._process and self._process.stdin:
                    try:
                        self._process.stdin.close()
                    except Exception:  # noqa: BLE001
                        pass
                if not self._try_reconnect():
                    break
                next_push_at = time.monotonic() + self._frame_interval
            except Exception as exc:  # noqa: BLE001
                logger.warning("push frame error: %s", exc)

    def _try_reconnect(self) -> bool:
        if self._reconnecting:
            return False
        self._reconnecting = True
        try:
            self._reconnect_attempts += 1
            if self._reconnect_attempts > self._max_reconnect:
                logger.error(
                    "max reconnect reached; streaming stopped (camera=%s)",
                    self._camera_id,
                )
                self._running = False
                self._stop_reason = "max_reconnect"
                return False
            delay = min(
                RECONNECT_BACKOFF_BASE_SECONDS * (2**self._reconnect_attempts),
                RECONNECT_BACKOFF_MAX_SECONDS,
            )
            time.sleep(delay)
            if self._process:
                try:
                    self._process.kill()
                except Exception:  # noqa: BLE001
                    pass
            self._process = self._spawn_ffmpeg()
            if self._process is not None:
                self._reconnect_attempts = 0
            return self._process is not None
        finally:
            self._reconnecting = False


def make_from_env(camera_id: str) -> MediaMTXStreamer:
    """Build a streamer from the operator's env vars (see Phase 3/6 docs)."""
    enabled = os.environ.get("MEDIAMTX_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return MediaMTXStreamer(
        camera_id=camera_id,
        width=int(os.environ.get("MEDIAMTX_WIDTH", "960")),
        height=int(os.environ.get("MEDIAMTX_HEIGHT", "540")),
        fps=int(os.environ.get("MEDIAMTX_FPS", "10")),
        bitrate_kbps=int(os.environ.get("MEDIAMTX_BITRATE_KBPS", "1800")),
        host=os.environ.get("MEDIAMTX_HOST", "").strip(),
        rtsp_port=int(os.environ.get("MEDIAMTX_RTSP_PORT", "8554")),
        hls_port=int(os.environ.get("MEDIAMTX_HLS_PORT", "8889")),
        webrtc_port=int(os.environ.get("MEDIAMTX_WEBRTC_PORT", "8890")),
        stream_prefix=os.environ.get("MEDIAMTX_STREAM_PREFIX", "sota-paddle-mtmc").strip()
        or "sota-paddle-mtmc",
        publish_url_template=os.environ.get("MEDIAMTX_PUBLISH_URL_TEMPLATE") or None,
        ffmpeg_bin=os.environ.get("FFMPEG_BIN", "ffmpeg"),
        enabled=enabled,
    )
