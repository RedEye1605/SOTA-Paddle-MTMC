"""Application entry point.

Wires the multi-camera runner, tracklet collector, ReID worker, identity
resolver, telemetry worker, and FastAPI server.

Run modes (from `--mode` or `SOTA_RUNTIME_MODE` env var):
  - `production` (default; alias `multi_rtsp`): real models, no fallbacks
  - `smoke_test` (alias `single_cam_smoke`): synthetic detector + histogram ReID
  - `benchmark`: production + extra FPS / GPU logging
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional


from .api.server import build_app
from .cli.args import parse_args
from .cli.config import (
    env_int,
    env_str,
    get_active_cameras,
    load_all_configs,
    resolve_rtsp_url,
)
from .cli.logging_setup import setup_logging
from .detection.pphuman_pipeline import PPHumanDetectorAdapter
from .identity.camera_topology import CameraTopology
from .identity.resolver import GlobalIdentityResolver, ResolverConfig
from .identity.scoring import ScoreWeights
from .reid.transreid_adapter import TransReIDAdapter
from .reid.base import ReIDConfig
from .core.runtime_mode import RuntimeMode, resolve_runtime_mode
from .seed import seed_legacy_topology
from .storage.minio_store import from_env as minio_from_env
from .storage.postgres import from_env as pg_from_env
from .storage.qdrant_store import from_env as qdrant_from_env
from .storage.redis_state import from_env as redis_from_env
from .telemetry.metrics import REGISTRY
from .telemetry.mqtt_client import from_env as mqtt_from_env
from .utils.gpu import gpu_memory_used_mb
from .workers.multi_camera_runner import CameraSource, MultiCameraRunner
from .workers.detection_event_consumer import DetectionEventConsumer
from .workers.evidence_rekey_worker import EvidenceRekeyWorker, RekeyConfig
from .workers.identity_overlay_cache import IdentityOverlayCache
from .workers.pel_sweeper import PELSweeper, from_env as pel_sweeper_from_env
from .workers.reid_worker import ReIDWorker
from .workers.telemetry_worker import TelemetryWorker
from .workers.tracklet_collector import TrackletCollector
# PATCH (2026-06-16, SAHI integration): SAHI worker is in-process;
# SahiDetector runs the same PaddleDetection model the pipeline uses.
from .detection.sahi_detector import SahiConfig, SahiDetector
from .detection.sahi_worker import SAHIWorker, SAHIWorkerConfig
# PATCH (2026-06-16, SAHI integration): SAHITrackletBridge
# consumes stream:detections_sahi and emits auxiliary tracklets
# to TrackletCollector.on_sahi_detection. Marked source="sahi"
# + provisional=True.
from .detection.sahi_tracklet_bridge import (
    SAHIBridgeConfig,
    SAHITrackletBridge,
)
from .utils.frame_buffer import RTSPFrameBuffer

logger = logging.getLogger(__name__)


def select_reid_adapter(reid_cfg: dict, mode: RuntimeMode):
    # PATCH (2026-06-15, operator spec, transreid-only): the
    # operator dropped pphuman_strongbaseline / vanilla-transreid
    # / clipreid. The api now picks TransReID MSMT17 (the same
    # model the sidecar runs, ``vit_transreid_msmt.pth``). In
    # production the api image has no torch and the adapter
    # ``load()`` fails, so the api reid_worker is effectively a
    # passthrough — it downloads BGR frames, crops bboxes, and
    # hands off to the sidecar via the ``stream:tracklets`` ->
    # sidecar -> ``stream:embeddings`` chain. The adapter is
    # configured (so the api can still emit to ``stream:embeddings``
    # with the right ``model_name``) but ``extract()`` returns
    # zeros and the real work happens in the sidecar.
    active = reid_cfg.get("active_model", "transreid_msmt")
    if active in ("transreid", "transreid_msmt"):
        # PATCH-011 + PATCH-2026-06-15: profile + ignore_classifier_head
        # from the active reid config; default to MSMT17 profile
        # (the sidecar's profile) for shared vocabulary.
        profile = reid_cfg.get("profile", "msmt17")
        ignore_head = bool(reid_cfg.get("ignore_classifier_head", True))
        require_ckpt = bool(reid_cfg.get("require_checkpoint_in_production", True))
        return TransReIDAdapter(
            ReIDConfig(
                name="transreid_msmt",
                embedding_dim=int(reid_cfg.get("embedding_dim", 3840)),
                qdrant_collection="person_reid_transreid_msmt",
            ),
            profile=profile,
            ignore_classifier_head=ignore_head,
            require_checkpoint_in_production=require_ckpt,
            mode=mode,
        )
    raise ValueError(
        f"Unknown active ReID model: {active!r}. Only "
        f"'transreid_msmt' is supported (PP-Human / "
        f"vanilla-TransReID / CLIP-ReID have been dropped "
        f"per operator spec)."
    )


def build_app_context(
    args,
    configs: dict,
    mode: RuntimeMode,
) -> dict:
    """Build the storage / worker context. Factored out for tests."""
    storage_cfg = configs["app"].get("storage", {})
    identity_cfg = configs["app"].get("identity", {})
    reid_cfg = configs["app"].get("reid", {})

    # ----- Storage -----
    pg = pg_from_env() if storage_cfg.get("postgres_enabled", True) else None
    if pg is not None:
        pg.connect()
        if not pg.healthcheck():
            raise RuntimeError("PostgreSQL healthcheck failed; aborting")
        # Reconcile DB with the YAML source of truth. Idempotent —
        # a stored fingerprint makes warm restarts a no-op.
        # SEED_FORCE=1 to force a re-seed (e.g. after editing YAML).
        try:
            repo_root = Path(__file__).resolve().parent.parent
            counts = seed_legacy_topology(pg, repo_root=repo_root)
            logger.info("startup seed complete: %s", counts)
        except Exception as e:  # noqa: BLE001
            # Seed failures must NEVER crash the API. Log loudly,
            # continue — the rest of the app can still serve a
            # degraded mode.
            logger.warning("startup seed failed (continuing): %s", e)
    qdrant = qdrant_from_env() if storage_cfg.get("qdrant_enabled", True) else None
    if qdrant is not None:
        qdrant.connect()
        qdrant.init_collections()
    redis_s = redis_from_env() if storage_cfg.get("redis_enabled", True) else None
    if redis_s is not None:
        redis_s.connect()
    minio = minio_from_env() if storage_cfg.get("minio_enabled", True) else None
    if minio is not None:
        minio.connect()

    # ----- Camera topology -----
    topology = CameraTopology()
    if pg is not None:
        topology.load_from_rows(pg.fetch_camera_links())

    # ----- ReID adapter (one model, shared) -----
    # PATCH-NNN fix: load + warmup in a background thread so a
    # slow TRT rebuild doesn't block the runner from
    # starting. The runner + cameras + MediaMTX streamers
    # need to be up BEFORE the ReID model finishes loading.
    reid_adapter = select_reid_adapter(reid_cfg, mode)

    def _load_reid() -> None:
        try:
            reid_adapter.load()
            reid_adapter.warmup()
            logger.info("ReID adapter ready: %s", reid_adapter.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ReID adapter load failed: %s", exc)

    reid_load_thread = threading.Thread(
        target=_load_reid, daemon=True, name="reid-loader"
    )
    reid_load_thread.start()

    # ----- Resolver -----
    # PATCH (2026-06-17, BUG-1): read the reid_override_threshold
    # from config so the operator can disable it (set to null/None)
    # without rebuilding. Default 0.95 — a high-confidence cosine
    # short-circuit that prevents 1 person = N stored embeddings
    # when time_diff pushes the 5-factor final_score below threshold.
    _reid_override_raw = identity_cfg.get("reid_override_threshold", 0.95)
    _reid_override = None if _reid_override_raw in (None, "null") else float(_reid_override_raw)
    resolver_cfg = ResolverConfig(
        auto_match_threshold=float(identity_cfg.get("auto_match_threshold", 0.82)),
        candidate_threshold=float(identity_cfg.get("candidate_threshold", 0.72)),
        ambiguous_margin=float(identity_cfg.get("ambiguous_margin", 0.04)),
        prefer_new_id_when_ambiguous=bool(identity_cfg.get("prefer_new_id_when_ambiguous", True)),
        use_camera_topology=bool(identity_cfg.get("use_camera_topology", True)),
        use_zone_transitions=bool(identity_cfg.get("use_zone_transitions", True)),
        persistence_window_seconds=int(identity_cfg.get("persistence_window_seconds", 86_400)),
        enable_stage3_24h_fallback=bool(identity_cfg.get("enable_stage3_24h_fallback", True)),
        stage3_auto_match_threshold=float(identity_cfg.get("stage3_auto_match_threshold", 0.92)),
        reid_override_threshold=_reid_override,
        weights=ScoreWeights(
            reid_weight=float(identity_cfg.get("weights", {}).get("reid_weight", 0.55)),
            temporal_weight=float(identity_cfg.get("weights", {}).get("temporal_weight", 0.20)),
            camera_weight=float(identity_cfg.get("weights", {}).get("camera_weight", 0.15)),
            quality_weight=float(identity_cfg.get("weights", {}).get("quality_weight", 0.05)),
            zone_weight=float(identity_cfg.get("weights", {}).get("zone_weight", 0.05)),
        ),
    )
    resolver = GlobalIdentityResolver(
        pg=pg,
        qdrant=qdrant,
        redis=redis_s,
        topology=topology,
        config=resolver_cfg,
        model_name=reid_adapter.name,
    )

    # ----- Tracklet collector -----
    zone_rows = pg.fetch_zones() if pg is not None else []
    site_id = configs["cameras"].get("site_id", "default_site")
    tracklet_cfg = reid_cfg
    collector = TrackletCollector(
        pg=pg,
        redis=redis_s,
        minio=minio,
        site_id=site_id,
        zone_rows=zone_rows,
        min_track_age_frames=int(tracklet_cfg.get("min_track_age_frames", 10)),
        min_crops_per_tracklet=int(tracklet_cfg.get("min_crops_per_tracklet", 5)),
        max_crops_per_tracklet=int(tracklet_cfg.get("max_crops_per_tracklet", 15)),
        min_person_height_px=float(tracklet_cfg.get("min_person_height_px", 60.0)),
    )

    # ----- ReID worker (PATCH-004: real crop download) -----
    reid_worker = ReIDWorker(
        adapter=reid_adapter,
        pg=pg,
        qdrant=qdrant,
        redis=redis_s,
        minio=minio,
        mode=mode,
    )

    # PATCH (2026-06-15): DetectionEventConsumer — pulls structured
    # detection events from stream:detections (emitted by the PP-Human
    # vendor hotfix) and feeds TrackletCollector.on_detection(). This
    # wires the persistent-ID chain end-to-end. When ENABLE_PERSISTENT_ID
    # is false, the consumer is created but not started.
    persistent_id_enabled = env_str("ENABLE_PERSISTENT_ID", "true").lower() == "true"
    detection_event_consumer = DetectionEventConsumer(
        redis=redis_s,
        collector=collector,
    )
    identity_overlay_cache = IdentityOverlayCache(redis=redis_s)

    # ----- Evidence re-key worker (PATCH-029) -----
    evidence_cfg = configs["app"].get("evidence", {})
    rekey_worker = EvidenceRekeyWorker(
        minio=minio,
        pg=pg,
        redis=redis_s,
        config=RekeyConfig(
            enabled=bool(evidence_cfg.get("rekey_after_global_id", True)),
            keep_pending_copy=bool(evidence_cfg.get("keep_pending_copy", False)),
            retry_max=int(evidence_cfg.get("rekey_retry_max", 3)),
        ),
    )

    # ----- Detector adapter (one model, shared) -----
    detection_cfg = configs["app"].get("detection_tracking", {})
    detector = PPHumanDetectorAdapter(
        pipeline_path=env_str(
            "PPHUMAN_PIPELINE_PATH",
            detection_cfg.get(
                "pphuman_pipeline_path", "/opt/paddledetection/deploy/pipeline/pipeline.py"
            ),
        ),
        config_path=env_str(
            "PPHUMAN_INFER_CONFIG",
            detection_cfg.get(
                "pphuman_infer_config",
                "/opt/paddledetection/deploy/pipeline/config/infer_cfg_pphuman.yml",
            ),
        ),
        model_dir=env_str(
            "PPHUMAN_MODEL_DIR", detection_cfg.get("pphuman_model_dir", "/models/pphuman")
        ),
        device=env_str("PPHUMAN_DEVICE", "gpu"),
        run_mode=env_str(
            "PPHUMAN_RUN_MODE", configs["app"].get("runtime", {}).get("run_mode", "trt_fp16")
        ),
        skip_frame_num=int(
            detection_cfg.get(
                "skip_frame_num", configs["app"].get("streams", {}).get("skip_frame_num", 2)
            )
        ),
        mode=mode,
    )
    try:
        detector.load()
    except Exception as e:  # noqa: BLE001
        # In smoke mode the adapter is permissive; in production it
        # raises ProductionSafetyError on its own.
        if mode != RuntimeMode.SMOKE_TEST:
            raise
        logger.warning("detector.load failed in smoke mode: %s", e)

    # ----- MQTT -----
    mqtt = mqtt_from_env()
    if mqtt is not None:
        mqtt.connect()

    # ----- Telemetry worker -----
    telemetry = TelemetryWorker(pg=pg, redis=redis_s, mqtt=mqtt, site_id=site_id)

    # ----- PEL sweeper (reclaim stuck messages from dead consumers) -----
    pel_sweeper: Optional[PELSweeper] = None
    if redis_s is not None:
        pel_sweeper = pel_sweeper_from_env(redis_s)
        pel_sweeper.start()

    # PATCH (2026-06-16, SAHI integration): construct one
    # SAHIWorker per RTSP URL. SAHIWorker is a no-op when
    # SAHI_ENABLED=false (the default). One SahiDetector is
    # shared across all workers (the model load is expensive).
    sahi_workers: list = []
    sahi_rtsp_urls_env = env_str(
        "SAHI_RTSP_URLS",
        "rtsp://mediamtx:8554/cam1_merged/,rtsp://mediamtx:8554/cam2_merged/",
    )
    sahi_camera_ids_env = env_str("SAHI_CAMERA_IDS", "CAM_01,CAM_02")
    sahi_rtsp_urls = [u.strip() for u in sahi_rtsp_urls_env.split(",") if u.strip()]
    sahi_camera_ids = [c.strip() for c in sahi_camera_ids_env.split(",") if c.strip()]
    if sahi_rtsp_urls and sahi_camera_ids:
        sahi_detector_cfg = SahiConfig.from_env()
        # PATCH (2026-06-16, root-cause fix): SahiDetector needs the
        # DET model's .pdmodel + .pdiparams, NOT the PP-Human
        # pipeline YAML (which is a Pipeline-config, not a
        # paddle.inference.Config). The DET model is the same
        # pedestrian detector used by the PP-Human MOT pipeline.
        # Operator override via SAHI_MODEL_FILE / SAHI_PARAMS_FILE.
        _sahi_model_root = env_str(
            "SAHI_MODEL_DIR",
            env_str("PPHUMAN_MODEL_DIR", "/models/pphuman")
            + "/mot_ppyoloe_l_36e_pipeline",
        )
        sahi_detector = SahiDetector(
            config=sahi_detector_cfg,
            model_file=env_str(
                "SAHI_MODEL_FILE", f"{_sahi_model_root}/model.pdmodel"
            ),
            params_file=env_str(
                "SAHI_PARAMS_FILE", f"{_sahi_model_root}/model.pdiparams"
            ),
        )
        for cam_id, rtsp_url in zip(sahi_camera_ids, sahi_rtsp_urls):
            sahi_cfg = SAHIWorkerConfig.from_env(
                camera_id=cam_id, rtsp_url=rtsp_url
            )
            sahi_buffer = RTSPFrameBuffer(url=rtsp_url, camera_id=cam_id)
            worker = SAHIWorker(
                config=sahi_cfg,
                buffer=sahi_buffer,
                detector=sahi_detector,
                redis=redis_s,
            )
            sahi_workers.append(worker)
        logger.info("SAHI integration prepared %d workers", len(sahi_workers))

    # PATCH (2026-06-16, SAHI integration): SAHITrackletBridge
    # consumes stream:detections_sahi and emits auxiliary tracklets
    # to TrackletCollector.on_sahi_detection. Marked source="sahi"
    # + provisional=True. Built unconditionally so the rest of the
    # lifecycle wiring is symmetric; ``start()`` is a cheap no-op
    # when there are no SAHI workers.
    sahi_bridge: Optional[SAHITrackletBridge] = None
    if sahi_workers:
        sahi_bridge = SAHITrackletBridge(
            redis=redis_s,
            collector=collector,
            config=SAHIBridgeConfig.from_env(),
        )

    # PATCH (2026-06-15): start the persistent-ID threads
    # (DetectionEventConsumer + IdentityOverlayCache) BEFORE the
    # return so the threads are actually spawned.
    if persistent_id_enabled:
        try:
            detection_event_consumer.start()
            identity_overlay_cache.start()
            # PATCH (2026-06-15): start the tracklet collector's
            # background finalize loop so the persistent-ID chain
            # progresses even when runner.stream() blocks in
            # production mode.
            collector.start()
            # PATCH (2026-06-16, SAHI integration): start the
            # SAHI workers (each is a no-op if SAHI_ENABLED=false).
            for w in sahi_workers:
                w.start()
            # PATCH (2026-06-16, SAHI integration): start the
            # SAHITrackletBridge so SAHI-only detections become
            # provisional tracklets in the persistent-ID chain.
            if sahi_bridge is not None:
                sahi_bridge.start()
            print(
                "[main] persistent-ID threads started (detection_consumer, overlay_cache, auto_finalize, sahi_workers=%d, sahi_bridge=%s)" % (
                    len(sahi_workers),
                    "on" if sahi_bridge is not None else "off",
                ),
                flush=True,
            )
            logger.info(
                "persistent-ID threads started (detection_consumer, overlay_cache, auto_finalize)"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[main] persistent-ID thread start failed: {e}", flush=True)
            logger.warning("persistent-ID thread start failed: %s", e)

    return {
        "pg": pg,
        "qdrant": qdrant,
        "redis": redis_s,
        "minio": minio,
        "topology": topology,
        "reid_adapter": reid_adapter,
        "detector": detector,
        "resolver": resolver,
        "collector": collector,
        "reid_worker": reid_worker,
        "rekey_worker": rekey_worker,
        "mqtt": mqtt,
        "telemetry": telemetry,
        "pel_sweeper": pel_sweeper,
        "site_id": site_id,
        "mode": mode,
        # PATCH (2026-06-15): persistent-ID components.
        "detection_event_consumer": detection_event_consumer,
        "identity_overlay_cache": identity_overlay_cache,
        # PATCH (2026-06-16, SAHI integration): list of SAHIWorker
        # instances. Each is a no-op when SAHI_ENABLED=false.
        "sahi_workers": sahi_workers,
        # PATCH (2026-06-16, SAHI integration): SAHITrackletBridge
        # instance. None when no SAHI workers are configured.
        "sahi_bridge": sahi_bridge,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging()
    configs = load_all_configs(
        args.config,
        args.cameras_config,
        args.zones_config,
        args.links_config,
    )
    # Resolve runtime mode (production / smoke_test / benchmark).
    arg_mode = args.mode or configs["app"].get("app", {}).get("mode", "")
    if arg_mode:
        mode = RuntimeMode.from_string(arg_mode)
    else:
        mode = resolve_runtime_mode()
    logger.info("SOTA-Paddle-MTMC starting in mode=%s", mode.value)

    ctx = build_app_context(args, configs, mode=mode)

    # ----- Multi-camera runner -----
    active_cams = get_active_cameras(configs["cameras"])
    max_initial = int(configs["app"].get("streams", {}).get("max_initial_cameras", 2))
    if mode == RuntimeMode.SMOKE_TEST:
        cam = next(
            (c for c in active_cams if c["camera_id"] == (args.camera_id or "CAM_01")),
            active_cams[0],
        )
        cam["rtsp_url"] = args.video_file or resolve_rtsp_url(cam)
        sources = [
            CameraSource(
                camera_id=cam["camera_id"],
                source=cam["rtsp_url"],
                width=cam["width"],
                height=cam["height"],
                fps_target=cam["fps_target"],
            )
        ]
    else:
        sources = []
        for cam in active_cams[:max_initial]:
            cam["rtsp_url"] = resolve_rtsp_url(cam)
            sources.append(
                CameraSource(
                    camera_id=cam["camera_id"],
                    source=cam["rtsp_url"],
                    width=cam["width"],
                    height=cam["height"],
                    fps_target=cam["fps_target"],
                )
            )

    smoke_max_seconds = args.smoke_max_seconds or env_int("SMOKE_MAX_SECONDS", 0)

    # PATCH-007 fix: detector is shared (or the runner refuses).
    # When running in production with a real detector, also build a
    # PPHumanFrameStateAdapter so each per-camera worker gets a
    # ``frame_state`` (a per-frame callable backed by the subprocess
    # manager's MOT output). Without this the worker falls back to
    # the synthetic branch and the production-mode safety check
    # refuses to start.
    frame_state_adapter = None
    if not ctx["detector"].is_synthetic:
        from app.detection.pphuman_pipeline import (
            make_frame_state_adapter,
            preflight_camera_sources,
        )
        from pathlib import Path as _Path
        cameras_for_manager = [
            (s.camera_id, s.source) for s in sources
        ]
        output_root = "/app/reports/sota_paddle_mtmct/pphuman"
        _Path(output_root).mkdir(parents=True, exist_ok=True)
        # Decoder preflight: verify every input video is decodable
        # BEFORE we spawn the PP-Human subprocesses. The 2.2 GB
        # merged CCTV videos stall PP-Human's video decoder; a
        # preflight catches this at startup so we never report
        # the unified stream as "healthy but empty".
        # See FixReports/UNIFIED_STREAM_2026-06-14.md §5.
        for cam_id, source in cameras_for_manager:
            res = preflight_camera_sources([(cam_id, source)])[cam_id]
            if res.ok:
                logger.info(
                    "preflight OK | camera=%s | path=%s | %dx%d | %.2f fps | "
                    "%.1fs | %d bytes",
                    cam_id, res.path, res.width, res.height, res.fps,
                    res.duration_seconds, res.size_bytes,
                )
            else:
                logger.error(
                    "preflight FAILED | camera=%s | path=%s | error=%s",
                    cam_id, res.path, res.error,
                )
        # Unified-stream mode: let PP-Human publish its annotated
        # stream directly to MediaMTX via ``--pushurl``. The
        # operator's ffmpeg streamer is then redundant and should
        # be disabled (MEDIAMTX_ENABLED=false). If both are enabled
        # by accident, log a loud warning and prefer the direct
        # path (the ffmpeg streamer will fail to bind to the
        # already-taken RTSP path).
        direct_push = env_str("MEDIAMTX_PPHUMAN_DIRECT_PUSH", "true").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        mediamtx_enabled = env_str("MEDIAMTX_ENABLED", "true").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if direct_push and mediamtx_enabled:
            logger.warning(
                "MEDIAMTX_PPHUMAN_DIRECT_PUSH=true AND "
                "MEDIAMTX_ENABLED=true — both will try to publish to "
                "MediaMTX. The ffmpeg streamer will fail (RTSP path "
                "in use). Set MEDIAMTX_ENABLED=false to silence this."
            )
        # Resolve the pushurl base: explicit env wins; otherwise
        # auto-derive from MEDIAMTX_HOST / RTSP_PORT / STREAM_PREFIX.
        # The trailing slash is important — PP-Human does
        # ``os.path.join(pushurl, filename)`` which only inserts a
        # separator if the left side ends in one.
        pushurl_base = env_str("MEDIAMTX_PUSHURL_BASE", "").strip()
        if direct_push and not pushurl_base:
            host = env_str("MEDIAMTX_HOST", "").strip()
            port = env_str("MEDIAMTX_RTSP_PORT", "8554").strip() or "8554"
            prefix = (
                env_str("MEDIAMTX_STREAM_PREFIX", "sota-paddle-mtmc").strip()
                or "sota-paddle-mtmc"
            )
            if host:
                pushurl_base = f"rtsp://{host}:{port}/{prefix}/"
                logger.info(
                    "PP-Human pushurl auto-derived from env: %s", pushurl_base
                )
        if direct_push and pushurl_base:
            logger.info(
                "Unified stream mode: PP-Human will publish annotated "
                "stream directly to MediaMTX at %s<basename>", pushurl_base
            )
        elif direct_push and not pushurl_base:
            logger.warning(
                "MEDIAMTX_PPHUMAN_DIRECT_PUSH=true but no "
                "MEDIAMTX_PUSHURL_BASE / MEDIAMTX_HOST could be "
                "resolved. Falling back to file-only output."
            )
            pushurl_base = None
        if not direct_push:
            pushurl_base = None
        # ``make_frame_state_adapter`` constructs its own
        # :class:`PPHumanPipelineSubprocessManager` internally and
        # wires it to a :class:`PPHumanFrameStateAdapter`. The
        # ``pipeline_manager`` here would be a duplicate; the
        # ``frame_state_adapter`` is the only thing the runner
        # needs.
        frame_state_adapter = make_frame_state_adapter(
            adapter=ctx["detector"],
            cameras=cameras_for_manager,
            output_root=output_root,
            pushurl_base=pushurl_base,
        )
        # NB: must call .start() to spawn the PP-Human
        # subprocess for each camera and start the MOT-output
        # tailer thread. Without this the per-camera workers
        # see no detections and the cameras never emit any
        # frames to the runner.
        frame_state_adapter.start()
    runner = MultiCameraRunner(
        sources,
        skip_frame_num=int(configs["app"].get("streams", {}).get("skip_frame_num", 2)),
        smoke_test_mode=(mode == RuntimeMode.SMOKE_TEST),
        detector=ctx["detector"] if not ctx["detector"].is_synthetic else None,
        mode=mode,
        frame_state_adapter=frame_state_adapter,
        # PATCH (2026-06-17, identity_overlay wiring fix): the
        # IdentityOverlayCache is constructed and started above but
        # until now it was never passed to the runner. The runner
        # falls back to local_track_id only and the HLS overlay
        # never shows ``G:<gid>``. Wire it here so the overlay
        # reflects the resolver's ``(camera, local) -> gid``
        # binding.
        identity_overlay_cache=ctx.get("identity_overlay_cache"),
    )
    runner.start()

    # ----- Background workers -----
    stop = threading.Event()
    threads = []
    for name, target in [
        ("reid", ctx["reid_worker"].run),
        ("resolver", ctx["resolver"].run),  # PATCH-006
        ("rekey", ctx["rekey_worker"].run),  # PATCH-029
        ("telemetry", ctx["telemetry"].run),
    ]:
        t = threading.Thread(target=target, args=(stop,), daemon=True, name=f"worker-{name}")
        t.start()
        threads.append(t)

    # ----- FastAPI -----
    import uvicorn

    api_cfg = configs["app"].get("telemetry", {})
    api_host = api_cfg.get("api_host", "0.0.0.0")
    api_port = int(api_cfg.get("api_port", 8000))
    # PATCH-014: FastAPI auth wired here.
    # PATCH (2026-06-17): pass the MinioStore so the /health endpoint
    # can run a cached reachability probe (30s TTL, 2s wall-clock
    # bound) against the operator's external MinIO cluster. When
    # ``storage.minio_enabled`` is False, ``ctx["minio"]`` is None
    # and the probe short-circuits to "disabled".
    api_app = build_app(pg=ctx["pg"], minio=ctx.get("minio"), mode=mode)
    api_thread = threading.Thread(
        target=lambda: uvicorn.run(api_app, host=api_host, port=api_port, log_level="warning"),
        daemon=True,
        name="api",
    )
    api_thread.start()

    # ----- Signal handlers -----
    def _stop(signum, _frame):
        logger.info("Received signal %d, shutting down", signum)
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _stop)
        except Exception:  # noqa: BLE001
            pass

    # ----- Main loop: drain frame results -> tracklet collector -----
    last_gauge_log = 0.0
    last_tracklet_finalize = 0.0
    try:
        for result in runner.stream(
            max_seconds=smoke_max_seconds if mode == RuntimeMode.SMOKE_TEST else None,
        ):
            ctx["collector"].on_frame(result)
            if (time.time() - last_tracklet_finalize) > 5.0:
                closed = ctx["collector"].finalize_stale()
                if closed:
                    ctx["collector"].emit_closed_tracklets(closed)
                last_tracklet_finalize = time.time()
            if (time.time() - last_gauge_log) > 10.0:
                mb = gpu_memory_used_mb()
                if mb is not None:
                    REGISTRY.gpu_memory_used.set(mb * 1024 * 1024)
                last_gauge_log = time.time()
            if stop.is_set():
                break
    finally:
        stop.set()
        runner.stop()
        # PATCH (2026-06-17, PEL sweeper): stop the background
        # PEL sweeper. thread.join(timeout=2) inside stop() so
        # SIGTERM doesn't leak a daemon thread.
        pel_sweeper = ctx.get("pel_sweeper")
        if pel_sweeper is not None:
            try:
                pel_sweeper.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("PELSweeper stop failed: %s", e)
        # PATCH (2026-06-16, SAHI integration): stop the SAHI
        # workers gracefully. Each stop() joins its thread within
        # 2s. No-op when SAHI_ENABLED=false.
        for w in ctx.get("sahi_workers", []):
            try:
                w.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("SAHIWorker stop failed: %s", e)
        # PATCH (2026-06-16, SAHI integration): stop the
        # SAHITrackletBridge. None when no SAHI workers were built.
        sahi_bridge = ctx.get("sahi_bridge")
        if sahi_bridge is not None:
            try:
                sahi_bridge.stop()
            except Exception as e:  # noqa: BLE001
                logger.warning("SAHITrackletBridge stop failed: %s", e)
        # Tear down infra clients explicitly so SIGTERM doesn't leak
        # the connection pools.  Each close() is best-effort so a
        # failure on one store does not block the others.
        for name in ("pg", "qdrant", "redis", "minio", "mqtt"):
            obj = ctx.get(name)
            if obj is None:
                continue
            close = getattr(obj, "close", None) or getattr(obj, "disconnect", None)
            if close is None:
                continue
            try:
                close()
            except Exception as e:  # noqa: BLE001
                logger.warning("teardown %s.close() failed: %s", name, e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
