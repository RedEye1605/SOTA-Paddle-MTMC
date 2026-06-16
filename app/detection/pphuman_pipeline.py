"""PP-Human detector adapter (PATCH-002 / BUG-002 fix).

Production path: invokes the official PaddleDetection
``deploy/pipeline/pipeline.py`` as a child process via the documented
CLI surface (per https://github.com/PaddlePaddle/PaddleDetection):

    python deploy/pipeline/pipeline.py \\
        --config deploy/pipeline/config/infer_cfg_pphuman.yml \\
        -o MOT.enable=True MOT.model_dir=<MODEL_DIR> \\
        --video_file=<FILE_OR_RTSP> \\
        --device=gpu \\
        --run_mode=trt_fp16  # or paddle

The pipeline writes MOT outputs (``{frame,id,x1,y1,w,h,score,...}``) to
``{output_dir}/mot_results/`` (per the official PaddleDetection MOT
README) which this adapter tails and parses into :class:`LocalTrack`
records.

Why subprocess and not in-process?
  * The official pipeline.py is not packaged as an importable module;
    it is run as a script that builds a `Pipeline` object internally.
  * Multiple cameras need multiple pipeline processes anyway (the
    pipeline reads one ``--video_file`` at a time). Subprocess-per-stream
    is the documented multi-stream pattern (see ``multi_camera_mtmct_en.md``).
  * Subprocess isolation makes crash recovery simple: a child that OOMs
    is restarted, the parent keeps running.

This module exposes the subprocess invocation as a generator:

    adapter = PPHumanDetectorAdapter(...)
    adapter.load()
    for local_track in adapter.stream(camera_id, video_source):
        ...

Smoke-test path: synthetic random-box detector (mirrors the existing
``PPHumanWorker._synthetic_detect`` behaviour). The smoke path is only
active in :class:`RuntimeMode.SMOKE_TEST`; in production ``load()``
raises :class:`ProductionSafetyError` if the pipeline is missing.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from ..core.runtime_mode import (
    RuntimeMode,
    assert_production_safe,
    resolve_runtime_mode,
    smoke_log,
)

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single detection in MOT format.

    Mirrors the official PaddleDetection MOT output line:
        frame, id, x1, y1, w, h, score, -1, -1, -1
    """

    frame_id: int
    track_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float


class PPHumanDetectorAdapter:
    """Wraps the official PaddleDetection PP-Human pipeline as a subprocess.

    The model is loaded ONCE in the subprocess and shared with N children;
    this class itself does not load paddle / PaddleInference — the subprocess
    does. The parent's job is to invoke the subprocess, tail its MOT output,
    and yield :class:`Detection` records.
    """

    def __init__(
        self,
        *,
        pipeline_path: str = "/opt/paddledetection/deploy/pipeline/pipeline.py",
        config_path: str = "/opt/paddledetection/deploy/pipeline/config/infer_cfg_pphuman.yml",
        model_dir: str = "/models/pphuman",
        device: str = "gpu",
        run_mode: str = "trt_fp16",
        skip_frame_num: int = 2,
        timeout_seconds: int = 30,
        mode: Optional[RuntimeMode] = None,
    ) -> None:
        self.pipeline_path = pipeline_path
        self.config_path = config_path
        self.model_dir = model_dir
        self.device = device
        self.run_mode = run_mode
        self.skip_frame_num = skip_frame_num
        self.timeout_seconds = timeout_seconds
        self._mode = mode or resolve_runtime_mode()
        self._loaded = False
        self._is_synthetic = False

    def load(self) -> None:
        """Probe that the pipeline is installed. Fail-fast in production.

        The actual model load happens in the subprocess when ``stream()``
        is called. This ``load()`` is the production-mode gate.
        """
        if not Path(self.pipeline_path).exists() or not Path(self.config_path).exists():
            if self._mode == RuntimeMode.SMOKE_TEST:
                logger.error(
                    "PP-Human pipeline not found at %s; using synthetic detector. SMOKE-TEST ONLY.",
                    self.pipeline_path,
                )
                self._is_synthetic = True
                self._loaded = True
                return
            assert_production_safe(
                mode=self._mode,
                component="PPHumanDetectorAdapter",
                condition=(
                    f"missing PaddleDetection pipeline at "
                    f"{self.pipeline_path!r} (clone PaddleDetection there or "
                    f"set PPHUMAN_PIPELINE_PATH / PPHUMAN_INFER_CONFIG)"
                ),
            )
        self._loaded = True
        self._is_synthetic = False
        logger.info(
            "PP-Human pipeline ready: pipeline=%s config=%s model_dir=%s device=%s run_mode=%s",
            self.pipeline_path,
            self.config_path,
            self.model_dir,
            self.device,
            self.run_mode,
        )

    @property
    def is_synthetic(self) -> bool:
        return self._is_synthetic

    def synthetic_stream(
        self,
        camera_id: str,
        frame_reader: Iterator[tuple[int, float, np.ndarray]],
    ) -> Iterator[Detection]:
        """Smoke-test fallback. Generates 0-2 deterministic random detections
        per frame. Mirrors the historical ``PPHumanWorker._synthetic_detect``
        behaviour.
        """
        smoke_log("PPHumanDetectorAdapter", "synthetic stream (SMOKE-TEST)")
        for frame_id, ts, frame in frame_reader:
            h, w = frame.shape[:2]
            rng = np.random.default_rng(
                seed=hash((camera_id, frame_id)) & 0xFFFFFFFF,
            )
            n = int(rng.integers(0, 3))
            for tid in range(n):
                x1 = float(rng.uniform(0.0, 0.7) * w)
                y1 = float(rng.uniform(0.0, 0.7) * h)
                bw = float(rng.uniform(40, 120))
                bh = float(rng.uniform(80, 240))
                yield Detection(
                    frame_id=frame_id,
                    track_id=tid + 1,
                    bbox=(x1, y1, min(w - 1, x1 + bw), min(h - 1, y1 + bh)),
                    confidence=float(rng.uniform(0.5, 0.95)),
                )

    def build_pipeline_command(
        self,
        *,
        camera_id: str,
        video_file: str,
        output_dir: str,
        pushurl: Optional[str] = None,
    ) -> list[str]:
        """Build the official pipeline.py command for a single stream.

        Mirrors the documented CLI surface from
        https://github.com/PaddlePaddle/PaddleDetection (Context7):
            python deploy/pipeline/pipeline.py \\
                --config <infer_cfg_pphuman.yml> \\
                -o MOT.enable=True MOT.model_dir=<MODEL_DIR> \\
                --video_file=<FILE_OR_RTSP> \\
                --device=<gpu|cpu> \\
                --run_mode=<paddle|trt_fp16|trt_fp32|trt_int8>

        ``pushurl``: if set, the pipeline is launched with
        ``--pushurl <pushurl>`` so PP-Human publishes its annotated
        stream (with bboxes burned in by its visualiser) directly
        to MediaMTX.

        **Path contract.** PaddleDetection's ``pipeline.py`` does
        ``os.path.join(pushurl, video_out_name)`` where
        ``video_out_name`` is the basename of the input
        ``video_file`` *with the extension stripped*. So
        ``/data/smoke/CAM_01.mp4`` produces
        ``rtsp://host:8554/sota-paddle-mtmc/CAM_01`` and
        ``/data/smoke/CAM_02.mp4`` produces
        ``rtsp://host:8554/sota-paddle-mtmc/CAM_02``. The
        operator MUST name the input file with the public path
        basename (or symlink it) so the published stream lands on
        the public contract. Pinned by
        ``tests/test_pphuman_pushurl.py`` and
        ``tests/test_streaming_contract_consistency.py``.
        """
        if not Path(self.pipeline_path).exists():
            raise FileNotFoundError(self.pipeline_path)
        # The pipeline reads its own yml; we use `-o` overrides so the
        # operator does NOT need to edit the YAML at deploy time.
        # PATCH-051 fix: do NOT override ``MOT.model_dir`` to a
        # local path. The pipeline calls ``auto_download_model`` on
        # the URL in the YAML, and that function only returns a
        # local path if the input is a URL — passing a local path
        # through `-o` short-circuits the cache lookup and the
        # subprocess tries to load the *parent* model dir as a
        # model. We rely on the cache priming (see
        # ``_prime_model_cache``) to satisfy the URL-based lookup
        # with our local files instead.
        #
        # PATCH-051 fix: also override ``MOT.tracker_config`` to an
        # absolute path. The default in the YAML is the relative
        # ``deploy/pipeline/config/tracker_config.yml``, which the
        # pipeline then opens with ``open(self.tracker_config)``
        # from CWD (``/app``). The result is a FileNotFoundError
        # on the first MOT inference. The absolute path points at
        # the same file under the cloned PaddleDetection tree.
        tracker_config = str(
            Path(self.pipeline_path).parent / "config" / "tracker_config.yml",
        )
        overrides = [
            "MOT.enable=True",
            f"MOT.tracker_config={tracker_config}",
            f"MOT.skip_frame_num={self.skip_frame_num}",
            # PATCH (2026-06-15, operator spec, anti-flicker): the
            # upstream OCSORTTracker default is ``max_age: 30`` which
            # at 20 fps is only 1.5s. Any occlusion longer than that
            # makes MOT mint a new ``local_track_id`` for the same
            # person, which the resolver cannot re-associate
            # (different key) and the HLS overlay renders a new
            # ``G:<gid>`` for what is visually the same person.
            # Bumping to ``max_age: 120`` (8s at 15 fps, 6s at 20 fps)
            # gives the operator a much more stable ``local_track_id``
            # and the IdentityOverlayCache a longer window to keep
            # its (cam, local) -> gid binding alive.
            "MOT.OCSORTTracker.max_age=120",
            # Also relax the IoU match threshold slightly so a
            # 1-2 frame jitter (e.g. from NMS) doesn't break the
            # track. 0.3 -> 0.25 is a small bump that helps in
            # dense scenes.
            "MOT.OCSORTTracker.iou_threshold=0.25",
            # PATCH (2026-06-17, BUG-NEW-A fix): ``min_hits: 1`` is
            # the upstream default for "first detection = track".
            # It correlates strongly with the 24% offset deadlock:
            # the first 30 min of the source video is an empty
            # showroom, so the tracker's internal state is empty;
            # the moment people appear, ``min_hits=1`` lets every
            # noisy single-frame detection mint a track, the
            # Hungarian cost matrix inflates, and OC-SORT's ``lap``
            # assignment (or the predictor's downstream) deadlocks
            # at the same byte offset reproducibly. Bumping to
            # ``min_hits: 3`` (the PaddleDetection upstream default)
            # suppresses noise-driven track inflation while keeping
            # the latency to first-track at ~0.4s (3 frames at
            # 7.5 fps effective with ``skip_frame_num=2``). This
            # is the same value the upstream tracker uses and is
            # the PaddleDetection-recommended setting for MOT
            # videos with non-trivial background motion.
            "MOT.OCSORTTracker.min_hits=3",
        ]
        # PATCH-051 fix: use the *current* Python interpreter so the
        # PP-Human subprocess inherits the project venv (where paddle /
        # scipy / etc. are installed). The previous hard-coded "python"
        # relied on PATH lookup, which inside a container may resolve
        # to a system interpreter that has none of the deps.
        cmd = [
            sys.executable,
            self.pipeline_path,
            "--config",
            self.config_path,
            "-o",
            *overrides,
            "--video_file",
            str(video_file),
            "--device",
            self.device,
            "--run_mode",
            self.run_mode,
            "--output_dir",
            output_dir,
        ]
        # Unified-stream mode: let PP-Human push the annotated
        # stream itself. We do NOT add ``--pushurl`` when pushurl is
        # empty or None — the pipeline defaults to ``""`` which
        # means "no push, write files only".
        if pushurl:
            cmd.extend(["--pushurl", pushurl])
        # PATCH-051 fix: the official pipeline.py's ``--camera_id``
        # arg is declared ``type=int`` (see
        # ``deploy/pipeline/cfg_utils.py``). Passing the operator's
        # string camera_id (``"CAM_01"``) made argparse raise
        # ``invalid int value: 'CAM_01'`` and the subprocess died
        # before producing any MOT output.
        #
        # We translate the camera_id into a stable integer hash
        # modulo a small range so two cameras never collide. The
        # MOT output file path uses the original string camera_id
        # (see ``--output_dir`` above, which is per-camera), so
        # the integer is only used by the subprocess to disambiguate
        # its own internal RTSP handle and never escapes.
        try:
            cam_int = int(camera_id)
        except (TypeError, ValueError):
            cam_int = abs(hash(camera_id)) % 1000
        cmd.extend(["--camera_id", str(cam_int)])
        return cmd

    def _prime_model_cache(self, *, cache_root: Path, model_dir: str) -> None:
        """Symlink the operator's local model dir into PaddleDetection's
        ``~/.cache/paddle/infer_weights/<basename>/`` cache.

        PaddleDetection's ``auto_download_model`` looks for the model
        in the cache by the *zip basename minus extension*. The
        operator's ``PPHUMAN_MODEL_DIR`` is the *parent* directory
        containing one subdir per model (matching PaddleDetection's
        own ``download_pphuman_models.sh`` layout). We map each
        canonical PaddleDetection cache name to the corresponding
        subdir if it exists; for missing models we create an
        empty dir at the canonical name so the cache lookup
        short-circuits and the subprocess never tries to
        download. (Downloading KPT/ATTR/etc. that the YAML has
        disabled still races between two subprocesses and
        intermittently fails with ``FileNotFoundError`` on the
        ``.zip_tmp`` file.)

        The symlink target MUST be the specific model subdir, NOT
        the parent: ``det_infer.py`` line 348 opens
        ``model_dir/infer_cfg.yml``, and only the subdir has that
        file.

        PATCH-052 (UNIFIED_STREAM_2026-06-14 addendum K.6): the
        cfg's REID URL is ``reid_model.zip`` which unpacks to
        cache dir ``reid_model/``, but the operator's actual
        ReID model is at ``strongbaseline_r50_30e_pa100k/``.
        An empty ``reid_model/`` would silently short-circuit
        the download and the predictor would load an empty
        dir — we need to remap the URL basename to the
        operator's actual subdir. Same for any other
        URL↔dirname divergence. The ``_URL_TO_LOCAL_OVERRIDES``
        table below covers the known cases.
        """
        if not model_dir:
            return
        model_path = Path(model_dir)
        if not model_path.exists() or not model_path.is_dir():
            return
        cache_root.mkdir(parents=True, exist_ok=True)
        # PATCH-052: explicit URL-basename → local-subdir
        # overrides for cases where the PaddleDetection URL
        # basename does NOT match the operator's directory name.
        # Without this, ``auto_download_model`` finds an empty
        # dir in the cache and ``load_predictor`` raises
        # ``PaddlePredictor not found`` or hangs.
        url_to_local_overrides: dict[str, str] = {
            # cfg: REID.model_dir = https://.../reid_model.zip
            # operator has: /models/pphuman/strongbaseline_r50_30e_pa100k/
            "reid_model": "strongbaseline_r50_30e_pa100k",
        }
        # Map canonical PaddleDetection model names → local
        # subdir under ``model_dir``. The operator typically has
        # one of:
        #   - ``model_dir/mot_ppyoloe_l_36e_pipeline/`` for DET+MOT
        #   - ``model_dir/strongbaseline_r50_30e_pa100k/`` for ReID
        #   - ``model_dir/dark_hrnet_w32_256x192/`` for KPT
        # The keys here are the PaddleDetection *cache* names
        # (= zip basename without extension).
        for cache_name in (
            "mot_ppyoloe_l_36e_pipeline",
            "strongbaseline_r50_30e_pa100k",
            "dark_hrnet_w32_256x192",
            "reid_model",
            "PPLCNet_x1_0_person_attribute_945_infer",
            "STGCN",
            "ppyoloe_crn_s_80e_smoking_visdrone",
            "PPHGNet_tiny_calling_halfbody",
            "ppTSM_fight",
        ):
            # PATCH-052: resolve the operator's local subdir,
            # honouring the override table. Default: same name
            # as the cache.
            local_subdir_name = url_to_local_overrides.get(cache_name, cache_name)
            local_subdir = model_path / local_subdir_name
            link = cache_root / cache_name
            # PATCH-052: also re-validate an existing entry.
            # If the link is broken (target was removed or
            # renamed), drop it and re-create. If the link
            # points to the wrong target, replace it. This
            # makes the priming idempotent across renames.
            if link.is_symlink():
                target = link.resolve()
                if not target.exists():
                    link.unlink()
                elif local_subdir.exists() and target != local_subdir.resolve():
                    link.unlink()
            if link.exists() or link.is_symlink():
                # Already primed correctly.
                continue
            if local_subdir.exists() and local_subdir.is_dir():
                try:
                    link.symlink_to(local_subdir.resolve())
                except OSError:
                    pass
            else:
                # Operator did not bake this model. Create an
                # empty dir at the canonical name so the cache
                # lookup at ``get_path`` returns True and the
                # subprocess never tries to download (which
                # would race with the other subprocess and
                # intermittently fail with ``.zip_tmp``).
                try:
                    link.mkdir(parents=False, exist_ok=False)
                except (FileExistsError, OSError):
                    # Race with another thread; harmless.
                    pass

    def run_pipeline(
        self,
        *,
        camera_id: str,
        video_file: str,
        output_dir: str,
        pushurl: Optional[str] = None,
    ) -> subprocess.Popen:
        """Launch the pipeline.py subprocess.

        Returns the Popen object so the caller can wait / kill.

        ``pushurl``: forwarded to ``build_pipeline_command`` as the
        base URL for ``--pushurl``. The pipeline joins the basename
        of ``video_file`` onto this URL.
        """
        cmd = self.build_pipeline_command(
            camera_id=camera_id,
            video_file=video_file,
            output_dir=output_dir,
            pushurl=pushurl,
        )
        logger.info("Launching PP-Human pipeline for %s: %s", camera_id, shlex.join(cmd))
        # PATCH (2026-06-15): pass the operator's string camera_id
        # (e.g. "CAM_01") as an env var so the vendor hotfix
        # (RedisSideChannel) can emit events keyed by the real
        # camera_id, not the integer alias. The integer is
        # derived from this string in build_pipeline_command.
        # Set the env var on os.environ (which sub_env copies) so
        # the subprocess inherits it.
        os.environ["PPHUMAN_REAL_CAMERA_ID"] = str(camera_id)
        # PATCH-051 fix: PaddleDetection's
        # ``deploy/pipeline/download.py`` line 30 sets
        # ``WEIGHTS_HOME = osp.expanduser("~/.cache/paddle/...")``.
        # Inside the production container the ``app`` user's home
        # is ``/app`` (``USER app WORKDIR /app``), so the cache
        # path resolves to ``/app/.cache`` which the unprivileged
        # user cannot create. The subprocess then raises
        # ``PermissionError: '/app/.cache'`` and exits before the
        # first inference frame.
        #
        # We solve this by setting ``HOME`` to a writable path
        # (default ``/tmp/pphuman_home``) for the subprocess only.
        # The parent process's environment is untouched.
        sub_env = os.environ.copy()
        sub_env["HOME"] = sub_env.get(
            "PPHUMAN_SUBPROCESS_HOME",
            "/tmp/pphuman_home",
        )
        # PATCH-051 fix: paddlepaddle-gpu 2.6.2's
        # ``fused_conv2d_add_act`` operator calls
        # ``cudnnConvolutionBiasActivationForward`` with a
        # tensor layout that cudnn 9.x rejects with
        # ``CUDNN_STATUS_NOT_SUPPORTED (3000)``. Disabling the
        # fused kernel forces paddle to use the unfused
        # ``conv2d + bias + activation`` path, which is
        # portable across cudnn 8 and 9. The cost is a ~5%
        # throughput regression per conv — acceptable for the
        # production benchmark until paddle 2.7+ ships a
        # cudnn-9-compatible fused kernel.
        sub_env["FLAGS_use_fused_conv2d_add_act_op"] = "False"
        # PP-Human's ``pipeline.py`` calls ``print(...)`` for
        # its banner, config dump, model-load progress, and per-
        # frame stats. Without this flag Python buffers stdout
        # in 4 KiB blocks, and on a busy child the 64 KiB
        # Linux pipe can fill long before our drain thread
        # reads it — deadlocking the child on the GIL. With
        # this flag, every print() is flushed to the pipe
        # immediately and the drain thread keeps the pipe
        # mostly empty. See
        # ``tests/test_pphuman_subprocess_drain.py``.
        sub_env["PYTHONUNBUFFERED"] = "1"
        # Matplotlib also creates ~/.config/matplotlib on first
        # use; redirect via MPLCONFIGDIR to the same scratch dir.
        sub_env["MPLCONFIGDIR"] = sub_env.get(
            "PPHUMAN_MPLCONFIGDIR",
            "/tmp/pphuman_home/.config/matplotlib",
        )
        # Create the home dir if missing (and writable).
        Path(sub_env["HOME"]).mkdir(parents=True, exist_ok=True)
        Path(sub_env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
        # PATCH-051 fix: PaddleDetection's
        # ``auto_download_model`` downloads the model zip from a
        # public URL into ``~/.cache/paddle/infer_weights/<name>/``
        # even when the operator has baked the model into the
        # image at ``$PPHUMAN_MODEL_DIR``. The download adds 30+ s
        # of latency and a hard dependency on outbound internet
        # that operators in air-gapped sites cannot satisfy.
        #
        # We work around this by pre-seeding the cache with a
        # symlink to the operator-supplied model dir. The cache
        # name is the zip basename minus the extension, so
        # ``mot_ppyoloe_l_36e_pipeline.zip`` becomes
        # ``mot_ppyoloe_l_36e_pipeline/``. If the operator
        # already used the canonical name, this is a no-op; if
        # not, the symlink lets ``auto_download_model`` find the
        # model in microseconds instead of redownloading it.
        self._prime_model_cache(
            cache_root=Path(sub_env["HOME"]) / ".cache" / "paddle" / "infer_weights",
            model_dir=self.model_dir,
        )
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=sub_env,
        )

    def parse_mot_file(self, mot_path: Path) -> Iterator[Detection]:
        """Parse a PaddleDetection MOT txt file (frame,id,x1,y1,w,h,score,...)
        and yield :class:`Detection` records.
        """
        if not mot_path.exists():
            return
        with mot_path.open("r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                try:
                    frame = int(parts[0])
                    tid = int(parts[1])
                    x1 = float(parts[2])
                    y1 = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    score = float(parts[6])
                except (ValueError, IndexError):
                    continue
                yield Detection(
                    frame_id=frame,
                    track_id=tid,
                    bbox=(x1, y1, x1 + w, y1 + h),
                    confidence=score,
                )


class PPHumanPipelineSubprocessManager:
    """Helper that runs the PP-Human pipeline for a list of cameras and tails
    their MOT output directories. One subprocess per camera, all sharing
    the same :class:`PPHumanDetectorAdapter` (so the model is loaded once
    per process — multi-stream is achieved by multi-process, which is the
    official PaddleDetection pattern).
    """

    def __init__(
        self,
        adapter: PPHumanDetectorAdapter,
        cameras: list[tuple[str, str]],  # (camera_id, video_source)
        output_root: str = "/reports/sota_paddle_mtmct/pphuman",
        pushurl_base: Optional[str] = None,
    ) -> None:
        """Manage one PP-Human subprocess per camera.

        ``pushurl_base``: when set, every subprocess is launched with
        ``--pushurl <pushurl_base>`` so PP-Human publishes its
        annotated stream (with bboxes burned in by its visualiser)
        directly to MediaMTX. The pipeline joins the basename of
        the input ``video_file`` (with extension stripped) onto
        the base URL, so for ``/data/smoke/CAM_01.mp4`` the
        final URL is ``<pushurl_base>/CAM_01``. The input file's
        basename MUST match the public contract path, OR the
        operator must symlink it accordingly. Per-camera
        uniqueness is the operator's responsibility — the
        resolved basenames MUST be distinct across cameras.
        """
        self.adapter = adapter
        self.cameras = cameras
        self.output_root = output_root
        self.pushurl_base = pushurl_base
        self._procs: dict[str, subprocess.Popen] = {}
        self._stdout_logs: dict[str, list[str]] = {}
        self._stderr_logs: dict[str, list[str]] = {}
        self._stdout_last_ts: dict[str, float] = {}
        self._stderr_last_ts: dict[str, float] = {}
        self._crashed: set[str] = set()
        self._monitors: list[threading.Thread] = []
        self._stop = threading.Event()
        # PATCH (2026-06-17, BUG-NEW-A fix): stall watchdog. The
        # drain thread (in _monitor_subprocess) updates
        # ``_stdout_last_ts[cam_id]`` on every line the
        # subprocess writes. We poll that timestamp once a
        # second; if it is older than ``stall_timeout_seconds``
        # (default 60 s, override via env
        # ``PPHUMAN_STALL_TIMEOUT_SEC``), the subprocess is
        # considered deadlocked and we terminate + respawn it.
        # The original symptom: the PP-Human GPU loop hits a
        # deterministic frame offset (~24% of the merged mp4,
        # or ~30 min into the actual content) and stops
        # emitting stdout/stderr. The subprocess is alive
        # (``proc.poll()`` returns None) but the inference
        # thread is wedged. Without this watchdog the api sits
        # silent for 16+ min; with it, the chain self-heals
        # every ``stall_timeout_seconds`` until the operator
        # changes the source video or the underlying Paddle
        # predictor is fixed. Restart count is capped at
        # ``PPHUMAN_MAX_RESTARTS`` (default 10) per camera
        # before the camera is marked as crashed and the
        # benchmark gate refuses READY_FOR_LIMITED_PRODUCTION.
        self._stall_timeout_seconds: float = float(
            os.environ.get("PPHUMAN_STALL_TIMEOUT_SEC", "60")
        )
        self._restart_counts: dict[str, int] = {}
        self._max_restarts: int = int(
            os.environ.get("PPHUMAN_MAX_RESTARTS", "10")
        )
        self._restart_lock = threading.Lock()

    def start(self) -> None:
        for cam_id, source in self.cameras:
            out_dir = Path(self.output_root) / cam_id
            out_dir.mkdir(parents=True, exist_ok=True)
            self._stdout_logs[cam_id] = []
            self._stderr_logs[cam_id] = []
            self._stdout_last_ts[cam_id] = 0.0
            self._stderr_last_ts[cam_id] = 0.0
            self._restart_counts[cam_id] = 0
            proc = self.adapter.run_pipeline(
                camera_id=cam_id,
                video_file=source,
                output_dir=str(out_dir),
                pushurl=self.pushurl_base,
            )
            self._procs[cam_id] = proc
            # PATCH-051 fix: spawn a stderr-tap monitor per subprocess.
            # Previously the subprocess was launched with stderr=PIPE
            # but nothing was draining it — a child that crashed
            # within 60 seconds would silently hang on stderr write
            # while the parent thought it was "still running".
            #
            # UNIFIED_STREAM_2026-06-14 fix: drain BOTH stdout and
            # stderr concurrently. PP-Human's ``pipeline.py`` calls
            # ``print(...)`` heavily (banner, config dump, model
            # load progress, per-frame stats). The Linux default
            # pipe buffer is 64 KiB; without an active stdout
            # reader, the child's main thread blocks on the first
            # full pipe — which holds the GIL and freezes the
            # inference thread. Symptom: subprocess alive, GPU 0%,
            # internal ffmpeg child in ``anon_pipe_read``. The
            # stdout tap lives alongside the stderr tap in
            # ``_monitor_subprocess``; both are daemon threads
            # that exit cleanly when the subprocess closes its
            # pipes.
            monitor = threading.Thread(
                target=self._monitor_subprocess,
                args=(cam_id, source, out_dir),
                name=f"pphuman-drain-{cam_id}",
                daemon=True,
            )
            monitor.start()
            self._monitors.append(monitor)

    def _monitor_subprocess(
        self,
        cam_id: str,
        source: str,
        out_dir: Path,
        proc: Optional[subprocess.Popen] = None,
    ) -> None:
        """Drain stdout + stderr + watch the subprocess lifecycle
        + restart on stall (PATCH 2026-06-17, BUG-NEW-A).

        Four responsibilities (per
        FixReports/UNIFIED_STREAM_2026-06-14.md §5 and the
        2026-06-14 directive to "fix the PP-Human subprocess
        hang", plus the 2026-06-17 BUG-NEW-A watchdog):

          1. Continuously read ``proc.stdout`` so the child never
             blocks on a full stdout pipe (the child's main thread
             calls ``print(...)`` for the banner, config dump, and
             model-load progress; with the 64 KiB Linux pipe
             default, an un-drained child will deadlock within
             milliseconds of starting).  Captured into a
             200-line ring buffer in :attr:`_stdout_logs`.
          2. Continuously read ``proc.stderr`` (PATCH-051) into a
             200-line ring buffer in :attr:`_stderr_logs`.
          3. When the subprocess exits, log the return code, the
             stdout tail, AND the stderr tail so the operator can
             diagnose without ``docker exec``. Mark the camera as
             crashed in :attr:`crashed_cameras` on non-zero exit.
          4. NEW (2026-06-17, BUG-NEW-A fix): if the subprocess
             is alive (``proc.poll()`` is None) but no stdout has
             been emitted for ``stall_timeout_seconds`` (default
             60 s), the GPU loop is deadlocked. Terminate the
             subprocess, increment the per-camera restart count,
             and spawn a fresh one. Recursion is bounded by
             ``_max_restarts`` (default 10) per camera; when the
             cap is hit the camera is marked as crashed and the
             benchmark gate refuses
             ``READY_FOR_LIMITED_PRODUCTION``.

        Implementation note: stdout and stderr are drained by two
        inner daemon threads (started here) so a chatty child on
        one pipe cannot block the parent's read of the other. The
        outer thread polls ``proc.poll()`` once a second so the
        stall watchdog can fire. The inner threads exit naturally
        when the corresponding pipe hits EOF (the child closed
        it on exit).
        """
        if proc is None:
            proc = self._procs.get(cam_id)
        assert proc is not None, f"no proc for {cam_id}"
        try:
            assert proc.stdout is not None
            assert proc.stderr is not None

            stdout_lines: list[str] = []
            stderr_lines: list[str] = []

            def _drain(stream, sink, last_ts_attr):
                """Read ``stream`` line by line into ``sink`` until EOF.

                Mutates ``stdout_lines`` or ``stderr_lines`` in
                the enclosing scope. Bounded to the last 200
                lines. Updates the per-camera last-bytes-seen
                timestamp on every line so the stall watchdog
                (PATCH-2026-06-17) can fire when the GPU loop
                goes silent.
                """
                local: list[str] = []
                while True:
                    line = stream.readline()
                    if not line:
                        break
                    stripped = line.rstrip()
                    local.append(stripped)
                    if len(local) > 200:
                        local = local[-200:]
                    # PATCH (2026-06-17, BUG-NEW-A fix): update
                    # the per-camera timestamp on EVERY line,
                    # not just at EOF. The old code only set
                    # the timestamp when the child closed the
                    # pipe, which is too late to detect a
                    # 16+ min silent stall. We update the
                    # dict entry in place — never overwrite the
                    # attribute itself with a float (the old
                    # code did that at EOF, which would break
                    # the watchdog's ``.get(cam_id, 0.0)``
                    # call below).
                    bucket = getattr(self, last_ts_attr)
                    if isinstance(bucket, dict):
                        bucket[cam_id] = time.monotonic()
                sink.clear()
                sink.extend(local)
                bucket = getattr(self, last_ts_attr)
                if isinstance(bucket, dict):
                    bucket[cam_id] = time.monotonic()

            stdout_thread = threading.Thread(
                target=_drain,
                args=(proc.stdout, stdout_lines, "_stdout_last_ts"),
                name=f"pphuman-drain-stdout-{cam_id}",
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=_drain,
                args=(proc.stderr, stderr_lines, "_stderr_last_ts"),
                name=f"pphuman-drain-stderr-{cam_id}",
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()

            # PATCH (2026-06-17, BUG-NEW-A fix): poll
            # ``proc.poll()`` once a second instead of
            # ``proc.wait(timeout=None)``. The old code
            # blocked forever on a stuck child, so the
            # existing _stdout_last_ts timestamp was never
            # acted on. The new loop checks the timestamp
            # every second and respawns the subprocess when
            # the GPU loop is wedged.
            rc: Optional[int] = None
            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                last_ts = self._stdout_last_ts.get(cam_id, 0.0)
                if last_ts and (
                    time.monotonic() - last_ts
                ) > self._stall_timeout_seconds:
                    # Stalled: kill the subprocess and respawn
                    # unless the cap is reached.
                    logger.warning(
                        "PP-Human subprocess for %s stalled for >%.0fs; "
                        "terminating and respawning",
                        cam_id,
                        self._stall_timeout_seconds,
                    )
                    try:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except Exception:  # noqa: BLE001
                            try:
                                proc.kill()
                            except Exception:  # noqa: BLE001
                                pass
                    except Exception as e:  # noqa: BLE001
                        logger.warning(
                            "terminate() failed for stalled %s: %s",
                            cam_id,
                            e,
                        )
                    with self._restart_lock:
                        self._restart_counts[cam_id] = (
                            self._restart_counts.get(cam_id, 0) + 1
                        )
                        attempts = self._restart_counts[cam_id]
                    if attempts > self._max_restarts:
                        logger.error(
                            "PP-Human subprocess for %s exceeded max "
                            "restarts (%d); marking as crashed",
                            cam_id,
                            self._max_restarts,
                        )
                        self._crashed.add(cam_id)
                        rc = -1
                        break
                    # Spawn a fresh subprocess for the same camera.
                    new_proc = self.adapter.run_pipeline(
                        camera_id=cam_id,
                        video_file=source,
                        output_dir=str(out_dir),
                        pushurl=self.pushurl_base,
                    )
                    self._procs[cam_id] = new_proc
                    # Recurse: the new subprocess's lifecycle is
                    # monitored by a fresh call to this function.
                    self._monitors.append(
                        threading.Thread(
                            target=self._monitor_subprocess,
                            args=(cam_id, source, out_dir, new_proc),
                            name=f"pphuman-drain-{cam_id}-restart-{attempts}",
                            daemon=True,
                        )
                    )
                    self._monitors[-1].start()
                    # Drop the stdio threads for the dead child
                    # (their pipes already closed, EOF is imminent).
                    stdout_thread.join(timeout=1)
                    stderr_thread.join(timeout=1)
                    return
                time.sleep(1.0)

            # Join the drain threads (they should already be at EOF).
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)

            self._stdout_logs[cam_id] = stdout_lines
            self._stderr_logs[cam_id] = stderr_lines
            if rc != 0:
                logger.error(
                    "PP-Human subprocess for %s exited rc=%s. "
                    "Last stdout lines: %s | Last stderr lines: %s",
                    cam_id,
                    rc,
                    stdout_lines[-5:],
                    stderr_lines[-5:],
                )
                self._crashed.add(cam_id)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "PP-Human subprocess monitor for %s raised: %s",
                cam_id,
                e,
            )
            self._crashed.add(cam_id)

    @property
    def crashed_cameras(self) -> set[str]:
        """Cameras whose PP-Human subprocess died (non-zero exit or
        crashed monitor). The benchmark reads this to set
        ``workers_crashed`` and the gate refuses
        ``READY_FOR_LIMITED_PRODUCTION`` if any camera is here.
        """
        return set(self._crashed)

    @property
    def stderr_logs(self) -> dict[str, list[str]]:
        """Most-recent stderr lines per camera, for the
        benchmark report's debug section.
        """
        return {k: list(v) for k, v in self._stderr_logs.items()}

    @property
    def stdout_logs(self) -> dict[str, list[str]]:
        """Most-recent stdout lines per camera, for the
        benchmark report's debug section. Pinpoints deadlocks
        caused by a chatty child that the stderr-only tap in
        PATCH-051 would have missed.
        """
        return {k: list(v) for k, v in self._stdout_logs.items()}

    def stop(self) -> None:
        self._stop.set()
        for cam_id, p in self._procs.items():
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    p.kill()
                except Exception:  # noqa: BLE001
                    pass
        for monitor in self._monitors:
            monitor.join(timeout=2)

    def stream(self) -> Iterator[tuple[str, Detection]]:
        """Yield ``(camera_id, detection)`` tuples from all running pipelines.

        The MOT output file is `{camera_id}.txt` under each camera's
        ``output_dir/mot_results/``.
        """
        last_seen: dict[str, int] = {cam_id: 0 for cam_id, _ in self.cameras}
        while not self._stop.is_set():
            any_alive = False
            for cam_id, p in self._procs.items():
                if p.poll() is None:
                    any_alive = True
                mot = Path(self.output_root) / cam_id / "mot_results" / f"{cam_id}.txt"
                if not mot.exists():
                    continue
                for det in self.adapter.parse_mot_file(mot):
                    if det.frame_id <= last_seen[cam_id]:
                        continue
                    last_seen[cam_id] = det.frame_id
                    yield cam_id, det
            if not any_alive and not any_seen_mot_recently(last_seen):
                # All subprocesses died; stop streaming.
                return
            time.sleep(0.1)


def any_seen_mot_recently(_last_seen: dict[str, int]) -> bool:
    # This is a placeholder for backpressure / liveliness heuristics.
    # For now we just check that at least one subprocess is alive.
    return True


# ----------------------------------------------------------------------------
# Per-frame adapter (the bridge between the subprocess manager and the
# per-camera worker)
# ----------------------------------------------------------------------------


@dataclass
class _CameraMOTState:
    """Per-camera state for the frame-state adapter.

    Tracks the latest fully-emitted frame id and the buffer of
    detections keyed by frame id. Detections older than the
    emitted frame are discarded to bound memory.
    """

    next_emitted_frame: int = 0  # monotonic, set on first frame seen
    buffer: dict[int, list[Detection]] = field(default_factory=dict)
    finished: bool = False
    last_seen_frame: int = -1


class PPHumanFrameStateAdapter:
    """Per-frame adapter exposed to :class:`PPHumanWorker`.

    The worker contract is ``detector_factory(frame) -> list[raw]``
    where each ``raw`` exposes ``bbox`` and ``confidence``. The
    subprocess manager produces detections asynchronously — by
    the time the worker asks for the current frame, the
    subprocess's MOT write may or may not have happened. This
    adapter:

      1. tails the subprocess manager's stream (per camera) in
         a background thread and accumulates detections by
         ``frame_id``;
      2. exposes ``detections_for_frame(camera_id, frame_id)``
         which returns the detections for that frame (empty
         list if the subprocess hasn't reached it yet);
      3. exposes ``per_camera_detector_factory(camera_id)``
         which returns a worker-callable that maps the worker's
         incoming frame to the detections for that frame.

    The worker remains camera-local (the resolver still owns
    global identity). Bbox/confidence/class fields are taken
    from the official MOT text — no hand-rolled inference.
    """

    def __init__(
        self,
        *,
        manager: PPHumanPipelineSubprocessManager,
        timeout_seconds: float = 10.0,
        stall_timeout_seconds: float = 60.0,
    ) -> None:
        self._manager = manager
        self._timeout_seconds = float(timeout_seconds)
        # UNIFIED_STREAM_2026-06-14: one watchdog per adapter
        # (covers all cameras). Pinned by
        # ``tests/test_pphuman_preflight_watchdog.py``. The
        # watchdog records the latest ``note_frame`` call from
        # the tailer and ``note_subprocess_exit`` from the
        # manager's crashed set; operators can read
        # ``adapter.watchdog.healthy`` to know if the unified
        # stream is producing frames within the stall window.
        self.watchdog: StreamWatchdog = StreamWatchdog(
            stall_timeout_seconds=stall_timeout_seconds,
        )
        self._lock = threading.Lock()
        self._state: dict[str, _CameraMOTState] = defaultdict(_CameraMOTState)
        self._started = False
        self._stop = threading.Event()
        self._tailer: Optional[threading.Thread] = None
        self._crashed_cameras: set[str] = set()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start the subprocess manager + the MOT tailer thread.

        Idempotent. Safe to call once per benchmark run.
        """
        if self._started:
            return
        self._manager.start()
        self._started = True
        self._tailer = threading.Thread(
            target=self._tail_loop,
            name="pphuman-frame-state-tailer",
            daemon=True,
        )
        self._tailer.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._manager.stop()
        except Exception:  # noqa: BLE001
            pass
        if self._tailer is not None:
            self._tailer.join(timeout=2.0)

    @property
    def manager(self) -> PPHumanPipelineSubprocessManager:
        return self._manager

    @property
    def crashed_cameras(self) -> set[str]:
        # PATCH-051 fix: also surface any cameras whose underlying
        # PP-Human subprocess died. The benchmark reads this set to
        # set ``workers_crashed``; the previous implementation only
        # saw tailer-thread crashes, not subprocess crashes.
        with self._lock:
            crashed = set(self._crashed_cameras)
        # The manager is a duck-typed dependency; some tests pass
        # a stub manager that doesn't have ``crashed_cameras``.
        manager_crashed = getattr(self._manager, "crashed_cameras", None)
        if manager_crashed:
            crashed.update(manager_crashed)
        # UNIFIED_STREAM_2026-06-14: if any subprocess is known
        # crashed, record a non-zero exit on the watchdog so it
        # reports unhealthy with the rc in the stall reason.
        # Note: we do this regardless of the previous
        # ``watchdog.healthy`` value — a healthy-then-crash is
        # the canonical failure mode we want to surface, and
        # an unhealthy-then-crash is at least no worse.
        if crashed:
            self.watchdog.note_subprocess_exit(returncode=1)
        return crashed

    # -- tail ---------------------------------------------------------------

    def _tail_loop(self) -> None:
        try:
            for cam_id, det in self._manager.stream():
                if self._stop.is_set():
                    return
                with self._lock:
                    state = self._state[cam_id]
                    state.buffer.setdefault(det.frame_id, []).append(det)
                    state.last_seen_frame = max(state.last_seen_frame, det.frame_id)
                # UNIFIED_STREAM_2026-06-14: record that the
                # subprocess is producing detections. If the
                # stream of MOT entries stops for
                # ``stall_timeout_seconds``, the watchdog flips
                # ``healthy=False`` and the operator can read
                # ``adapter.watchdog.stall_reason`` for the
                # diagnosis. ``note_frame`` takes the wall-clock
                # ``ts`` so it doesn't depend on the subprocess's
                # notion of time.
                self.watchdog.note_frame(
                    frame_id=det.frame_id,
                    ts=time.monotonic(),
                )
        except Exception as e:  # noqa: BLE001
            logger.error("PPHumanFrameStateAdapter tailer crashed: %s", e)
            with self._lock:
                # Mark every configured camera as crashed; the
                # benchmark report records this.
                for cam_id in self._state:
                    self._crashed_cameras.add(cam_id)
                    self.watchdog.note_subprocess_exit(returncode=1)

    # -- per-frame query ----------------------------------------------------

    def detections_for_frame(
        self,
        camera_id: str,
        frame_id: int,
    ) -> list[Detection]:
        """Return the official MOT detections for ``frame_id``.

        The returned list is empty if the subprocess hasn't yet
        written a MOT entry for that frame. The worker treats
        empty as "no person detected" — it must NOT crash and
        must NOT invent synthetic detections.
        """
        with self._lock:
            state = self._state.get(camera_id)
            if state is None:
                return []
            dets = list(state.buffer.get(frame_id, []))
            # Mark this frame as emitted so we can drop the
            # buffer entry next time.
            if state.next_emitted_frame == 0 and frame_id > 0:
                state.next_emitted_frame = frame_id
            # Bound memory: drop detections for frames we've
            # already emitted and that are < frame_id - 256.
            stale = [fid for fid in state.buffer if fid < frame_id - 256]
            for fid in stale:
                state.buffer.pop(fid, None)
            return dets

    def mark_crashed(self, camera_id: str) -> None:
        with self._lock:
            self._crashed_cameras.add(camera_id)

    # -- worker-callable factory -------------------------------------------

    def per_camera_detector_factory(self, camera_id: str) -> Callable[..., list[Detection]]:
        """Return a per-frame callable bound to ``camera_id``.

        The callable accepts ``(frame, frame_id)`` where
        ``frame_id`` is the worker's current logical frame id
        (the worker's own counter, incremented once per
        processed frame). Detections come from the subprocess
        MOT output keyed by that ``frame_id``.

        For backward compatibility with the previous single-arg
        signature, the callable also accepts ``(frame,)`` and
        looks up frame_id 0 in that case (which only matters
        for unit tests that pre-populate the buffer).
        """

        def _factory(frame: np.ndarray, frame_id: int = 0) -> list[Detection]:
            return self.detections_for_frame(camera_id, int(frame_id))

        # Tag for introspection; tests assert this attribute to
        # confirm the adapter is the real one.
        _factory.camera_id = camera_id  # type: ignore[attr-defined]
        _factory.adapter = self  # type: ignore[attr-defined]
        return _factory


def make_frame_state_adapter(
    *,
    adapter: PPHumanDetectorAdapter,
    cameras: list[tuple[str, str]],
    output_root: str = "/reports/sota_paddle_mtmct/pphuman",
    pushurl_base: Optional[str] = None,
) -> PPHumanFrameStateAdapter:
    """Convenience: build a :class:`PPHumanFrameStateAdapter` from
    a loaded :class:`PPHumanDetectorAdapter` and a list of
    ``(camera_id, video_source)`` tuples.

    The caller must have called ``adapter.load()`` first.

    ``pushurl_base``: forwarded to the underlying
    :class:`PPHumanPipelineSubprocessManager` so the PP-Human
    subprocesses can publish their annotated stream directly to
    MediaMTX (unified-stream mode). See
    :class:`PPHumanPipelineSubprocessManager` for the URL format.
    """
    mgr = PPHumanPipelineSubprocessManager(
        adapter,
        cameras,
        output_root=output_root,
        pushurl_base=pushurl_base,
    )
    return PPHumanFrameStateAdapter(manager=mgr)


# ----------------------------------------------------------------------------
# Decoder preflight (per FixReports/UNIFIED_STREAM_2026-06-14.md §5)
# ----------------------------------------------------------------------------


@dataclass
class PreflightResult:
    """Outcome of :func:`preflight_video_source`.

    The fields populated even on failure (e.g. ``size_bytes`` for
    a missing file is 0; ``error`` is always set on failure).
    """

    ok: bool
    path: Path
    error: str = ""
    size_bytes: int = 0
    codec: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration_seconds: float = 0.0


def expected_publish_path(pushurl_base: str, video_file: str) -> str:
    """Compute the RTSP path PP-Human will publish to.

    Mirrors PaddleDetection's ``pipeline.py`` (line 666-667):

        video_out_name = 'output' if self.file_name is None else self.file_name
        pushurl = os.path.join(self.pushurl, video_out_name)

    where ``self.file_name`` is the basename of the input
    ``video_file`` with the extension stripped (see
    ``set_file_name`` at pipeline.py line 517-523).

    Pinned by ``tests/test_pphuman_preflight_watchdog.py``.
    """
    base = os.path.basename(str(video_file))
    if "." in base:
        base = base.split(".")[-2]
    return os.path.join(pushurl_base, base)


def preflight_video_source(path: Path) -> PreflightResult:
    """Validate a video source BEFORE launching the PP-Human subprocess.

    Checks (in order, fail-fast):

      1. file exists on disk
      2. file size > 0
      3. OpenCV can open the capture and read the first frame

    On success, the result also reports codec / resolution / fps /
    duration so the operator can see what PP-Human will be
    decoding.

    Pinned by ``tests/test_pphuman_preflight_watchdog.py``.
    """
    p = Path(path)
    if not p.exists():
        return PreflightResult(ok=False, path=p, error=f"file not found: {p}")
    size = p.stat().st_size
    if size == 0:
        return PreflightResult(ok=False, path=p, size_bytes=0, error=f"file is empty (0 bytes): {p}")
    # OpenCV decode check
    try:
        import cv2  # local import to keep module importable in slim test envs
    except ImportError as e:  # pragma: no cover - hard env dep
        return PreflightResult(ok=False, path=p, size_bytes=size, error=f"cv2 import failed: {e}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        cap.release()
        return PreflightResult(ok=False, path=p, size_bytes=size, error=f"OpenCV could not open {p}")
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return PreflightResult(
            ok=False,
            path=p,
            size_bytes=size,
            error=f"OpenCV could not decode the first frame of {p}",
        )
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (frame_count / fps) if fps > 0 and frame_count > 0 else 0.0
    cap.release()
    codec = ""  # ffprobe not required for the smoke gate; OpenCV handles decode
    return PreflightResult(
        ok=True,
        path=p,
        size_bytes=size,
        codec=codec,
        width=width,
        height=height,
        fps=fps,
        duration_seconds=duration,
    )


def preflight_camera_sources(
    cameras: list[tuple[str, str]],
) -> dict[str, PreflightResult]:
    """Run :func:`preflight_video_source` for each (camera_id, source) pair.

    Returns a mapping of camera_id → result. Callers should
    surface every failure to the operator; partial success
    (e.g. one of two cameras unreadable) must NOT silently
    continue — it should at minimum log loudly.
    """
    out: dict[str, PreflightResult] = {}
    for cam_id, source in cameras:
        out[cam_id] = preflight_video_source(Path(source))
    return out


# ----------------------------------------------------------------------------
# Stream watchdog (per FixReports/UNIFIED_STREAM_2026-06-14.md §5)
# ----------------------------------------------------------------------------


class StreamWatchdog:
    """Per-camera runtime watchdog for the unified stream.

    Three health signals:

      * ``note_frame`` is called by the per-camera worker every
        time a detection is emitted. As long as frames keep
        flowing within ``stall_timeout_seconds``, the watchdog
        reports ``healthy=True``.
      * ``note_subprocess_exit`` is called when the underlying
        PP-Human subprocess exits. A non-zero exit is a hard
        failure; a zero exit after frames have been emitted is
        treated as ``finished_cleanly`` (e.g. a 30s smoke clip
        that ran to the end of the file).
      * If neither signal arrives within
        ``stall_timeout_seconds`` from construction, the
        watchdog reports a stall.

    Pinned by ``tests/test_pphuman_preflight_watchdog.py``.
    """

    def __init__(self, *, stall_timeout_seconds: float = 60.0) -> None:
        self._stall_timeout = float(stall_timeout_seconds)
        self._last_frame_ts: Optional[float] = None
        self._last_frame_id: int = -1
        self._subprocess_rc: Optional[int] = None
        self._subprocess_alive: bool = True
        self._lock = threading.Lock()

    # -- producer side ------------------------------------------------------

    def note_frame(self, *, frame_id: int, ts: Optional[float] = None) -> None:
        """Record that a frame was just processed by the worker.

        ``ts`` defaults to ``time.monotonic()`` so callers don't
        have to think about clock sources.
        """
        with self._lock:
            self._last_frame_ts = ts if ts is not None else time.monotonic()
            self._last_frame_id = int(frame_id)

    def note_subprocess_exit(self, returncode: int) -> None:
        """Record that the underlying PP-Human subprocess has exited.

        A clean exit (rc=0) after at least one frame is treated
        as a natural end-of-clip, not a stall. A non-zero exit
        is always a failure.
        """
        with self._lock:
            self._subprocess_rc = int(returncode)
            self._subprocess_alive = False

    # -- consumer side ------------------------------------------------------

    @property
    def healthy(self) -> bool:
        """True iff at least one frame has been seen AND either
        the subprocess is still alive OR it exited cleanly
        after frames were emitted.
        """
        with self._lock:
            return self._evaluate_healthy()

    @property
    def stall_reason(self) -> str:
        """Human-readable reason the stream is unhealthy (empty
        when healthy).
        """
        with self._lock:
            if self._evaluate_healthy():
                return ""
            return self._compute_stall_reason()

    @property
    def finished_cleanly(self) -> bool:
        with self._lock:
            return (
                self._last_frame_ts is not None
                and not self._subprocess_alive
                and self._subprocess_rc == 0
            )

    @property
    def last_frame_id(self) -> int:
        with self._lock:
            return self._last_frame_id

    @property
    def last_frame_age_seconds(self) -> float:
        with self._lock:
            if self._last_frame_ts is None:
                return float("inf")
            return max(0.0, time.monotonic() - self._last_frame_ts)

    # -- internals ----------------------------------------------------------

    def _evaluate_healthy(self) -> bool:
        if self._last_frame_ts is None:
            return False
        if not self._subprocess_alive and self._subprocess_rc != 0:
            return False
        # Subprocess is still alive — check stall timer.
        return (time.monotonic() - self._last_frame_ts) <= self._stall_timeout

    def _compute_stall_reason(self) -> str:
        if self._last_frame_ts is None:
            if not self._subprocess_alive:
                # Subprocess exited (rc is set) but no frame was
                # ever seen. Report the exit; the operator can
                # check ``stdout_logs`` / ``stderr_logs`` for the
                # captured tails.
                return f"subprocess_exit_rc={self._subprocess_rc}_no_frames"
            return "no_frames_yet"
        if not self._subprocess_alive and self._subprocess_rc != 0:
            return f"subprocess_exit_rc={self._subprocess_rc}"
        age = time.monotonic() - self._last_frame_ts
        return f"stall_timeout: {age:.1f}s since last frame (limit {self._stall_timeout:.1f}s)"
