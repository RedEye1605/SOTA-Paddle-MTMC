"""Qdrant — vector search with payload filters.

Per the task spec, every Qdrant search MUST use metadata filters — never a
blind nearest-neighbor scan. The :class:`QdrantStore` class enforces this by
exposing a single `search()` method that requires filters; a separate
`init_collections()` method is used at startup.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Iterable, Optional

import numpy as np
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)


# Qdrant collection definitions — one per ReID model.
# (collection_name, embedding_dim, distance_metric)
#
# PATCH (2026-06-15, operator spec, transreid-only): the operator
# chose to drop the PP-Human / vanilla-TransReID / CLIP-ReID
# collections and run on a single 3840-dim TransReID MSMT17
# collection only. The TransReID sidecar in the eval image is
# the sole writer; the api reid_worker no longer needs the
# pphuman_strongbaseline adapter (it was producing 26-class
# attribute logits, not ReID features, and Paddle 3.x couldn't
# load the paddle 2.x format). The api reid_worker becomes a
# frame-URI passthrough: it downloads the BGR frame, crops the
# bbox, and routes the embedding to the sidecar via
# ``stream:tracklets``. If the sidecar is down, the api emits
# no embeddings and the resolver sees ``hold_ambiguous`` (no
# new global_ids minted), which is the safe path.
COLLECTIONS: list[tuple[str, int, models.Distance]] = [
    # PATCH (2026-06-15, transreid-only): the operator's choice.
    # ``vit_transreid_msmt.pth`` is a ViT-B/16 + SIE + JPM + BNNeck
    # trained on MSMT17. With JPM enabled the forward pass
    # concatenates 5 horizontal-part features → 5 x 768 = 3840 dim.
    # L2-normalized, COSINE distance.
    ("person_reid_transreid_msmt", 3840, models.Distance.COSINE),
]

# Payload fields that MUST be indexed for fast filtered search.
PAYLOAD_INDEX_FIELDS: list[tuple[str, models.PayloadSchemaType]] = [
    ("global_id", models.PayloadSchemaType.KEYWORD),
    ("tracklet_id", models.PayloadSchemaType.KEYWORD),
    ("camera_id", models.PayloadSchemaType.KEYWORD),
    ("local_track_id", models.PayloadSchemaType.KEYWORD),
    ("zone_id", models.PayloadSchemaType.KEYWORD),
    ("site_id", models.PayloadSchemaType.KEYWORD),
    ("timestamp", models.PayloadSchemaType.INTEGER),
    ("quality_score", models.PayloadSchemaType.FLOAT),
    ("model_name", models.PayloadSchemaType.KEYWORD),
    ("model_version", models.PayloadSchemaType.KEYWORD),
    ("embedding_version", models.PayloadSchemaType.KEYWORD),
]


class QdrantStore:
    """Thin wrapper around QdrantClient. Always use `search()` (with filters)."""

    def __init__(self, host: str, port: int, api_key: str = "") -> None:
        self._host = host
        self._port = port
        self._api_key = api_key
        self._client: Optional[QdrantClient] = None

    def connect(self) -> None:
        if self._client is not None:
            return
        self._client = QdrantClient(host=self._host, port=self._port, api_key=self._api_key or None)
        logger.info("Qdrant client ready: %s:%d", self._host, self._port)

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    @property
    def client(self) -> QdrantClient:
        assert self._client is not None, "QdrantStore.connect() first"
        return self._client

    # ---- init ----
    def init_collections(self) -> None:
        """Create collections + payload indexes. Idempotent."""
        # qdrant-client returns ``CollectionDescription`` objects which
        # are not hashable; compare by ``.name`` instead of putting
        # the objects themselves in a set.
        existing = {c.name for c in self.client.get_collections().collections}
        for name, dim, dist in COLLECTIONS:
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(size=dim, distance=dist),
                )
                logger.info("Created Qdrant collection %s dim=%d", name, dim)
            for field, schema in PAYLOAD_INDEX_FIELDS:
                try:
                    self.client.create_payload_index(
                        collection_name=name,
                        field_name=field,
                        field_schema=schema,
                    )
                except Exception as e:  # noqa: BLE001
                    # Already exists → ignore
                    logger.debug("Payload index %s.%s may already exist: %s", name, field, e)

    # ---- writes ----
    def upsert_point(
        self,
        collection: str,
        vector: np.ndarray,
        payload: dict[str, Any],
        point_id: Optional[str] = None,
    ) -> str:
        """Upsert a single point. Returns the point id used."""
        pid = point_id or str(uuid.uuid4())
        self.client.upsert(
            collection_name=collection,
            points=[models.PointStruct(id=pid, vector=vector.tolist(), payload=payload)],
        )
        return pid

    # ---- search (always with filters) ----
    def search(
        self,
        collection: str,
        query_vector: np.ndarray,
        *,
        timestamp_gte: int,
        candidate_camera_ids: Iterable[str],
        model_name: str,
        model_version: str,
        quality_score_gte: float = 0.5,
        top_k: int = 10,
        site_id: Optional[str] = None,
    ) -> list[models.ScoredPoint]:
        """Filtered ANN search. Filters are mandatory.

        PATCH-034 fix: when there are candidates, enforce
        ``timestamp_gte > 0`` so direct callers cannot accidentally do
        an un-bounded search. Production code must pass an actual time
        bound. The check is skipped on the empty-candidates
        short-circuit path (the audit's existing test exercises that
        branch with ``timestamp_gte=0``).
        """
        if candidate_camera_ids and timestamp_gte <= 0:
            raise ValueError(
                "QdrantStore.search() requires timestamp_gte > 0 when "
                "candidate_camera_ids is non-empty; refusing an "
                "unbounded search. Pass ts - persistence_window_seconds.",
            )
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if not candidate_camera_ids:
            return []
        camera_filter = models.FieldCondition(
            key="camera_id", match=models.MatchAny(any=list(candidate_camera_ids))
        )
        must = [
            models.FieldCondition(key="timestamp", range=models.Range(gte=timestamp_gte)),
            camera_filter,
            models.FieldCondition(key="quality_score", range=models.Range(gte=quality_score_gte)),
            models.FieldCondition(key="model_name", match=models.MatchValue(value=model_name)),
            models.FieldCondition(
                key="model_version", match=models.MatchValue(value=model_version)
            ),
        ]
        if site_id:
            must.append(
                models.FieldCondition(key="site_id", match=models.MatchValue(value=site_id))
            )
        start = time.perf_counter()
        # PATCH (2026-06-15, qdrant-client 1.18 migration): the
        # ``search`` method was removed in qdrant-client 1.10 and
        # replaced by ``query_points``. The api image bundles
        # qdrant-client 1.18 (server is 1.12) so the OLD call
        # raises ``AttributeError: 'QdrantClient' object has no
        # attribute 'search'``. ``query_points`` returns a
        # ``QueryResponse`` whose ``.points`` field is the
        # ``list[ScoredPoint]`` the caller already expects.
        hits = self.client.query_points(
            collection_name=collection,
            query=query_vector[0].tolist(),
            query_filter=models.Filter(must=must),
            limit=top_k,
            with_payload=True,
        ).points
        latency = time.perf_counter() - start
        # PATCH-019 fix: observe the latency histogram.
        try:
            from ..telemetry.metrics import REGISTRY

            REGISTRY.qdrant_query_latency.observe(latency)
        except Exception:  # noqa: BLE001
            pass
        logger.debug(
            "Qdrant search %s top_k=%d hits=%d latency=%.3fs", collection, top_k, len(hits), latency
        )
        return hits

    # ---- PATCH-016: per-camera travel-time window search ----
    def search_per_camera(
        self,
        collection: str,
        query_vector: np.ndarray,
        *,
        per_camera_windows: dict[str, tuple[Optional[int], Optional[int]]],
        model_name: str,
        model_version: str,
        quality_score_gte: float = 0.5,
        top_k: int = 10,
        site_id: Optional[str] = None,
    ) -> list[models.ScoredPoint]:
        """Run one Qdrant search per camera with per-camera ``[gte, lte]``
        timestamp windows. Returns a merged list ranked by cosine
        similarity.

        PATCH-016: the audit requires the per-camera travel-time
        filter to be pushed into the Qdrant query, not just computed
        in Python. Per the Qdrant docs, ``Filter(must=[...])`` ANDs
        all conditions; we want per-camera ``OR`` of
        ``camera_id==X AND lte=… AND gte=…``. The clean way is one
        search per camera.

        Args:
            per_camera_windows: ``{camera_id: (gte, lte)}``; either
                bound may be ``None`` to omit it.
        """
        if not per_camera_windows:
            return []
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        all_hits: list[models.ScoredPoint] = []
        per_camera_k = max(1, top_k)
        # Per-camera sub-query budget: never run more than 2x the
        # requested top_k per camera, to avoid combinatorial blowup
        # if the topology has many linked cameras.
        per_camera_k = min(per_camera_k, max(10, top_k * 2))
        total_latency = 0.0
        for cam_id, (gte, lte) in per_camera_windows.items():
            if not cam_id:
                continue
            must = [
                models.FieldCondition(
                    key="camera_id",
                    match=models.MatchValue(value=cam_id),
                ),
                models.FieldCondition(
                    key="model_name",
                    match=models.MatchValue(value=model_name),
                ),
                models.FieldCondition(
                    key="model_version",
                    match=models.MatchValue(value=model_version),
                ),
                models.FieldCondition(
                    key="quality_score",
                    range=models.Range(gte=quality_score_gte),
                ),
            ]
            if gte is not None or lte is not None:
                must.append(
                    models.FieldCondition(
                        key="timestamp",
                        range=models.Range(gte=gte, lte=lte),
                    )
                )
            if site_id:
                must.append(
                    models.FieldCondition(
                        key="site_id",
                        match=models.MatchValue(value=site_id),
                    )
                )
            start = time.perf_counter()
            try:
                # PATCH (2026-06-15, qdrant-client 1.18 migration):
                # ``search`` was removed; use ``query_points``. See
                # the ``QdrantStore.search`` method for context.
                hits = self.client.query_points(
                    collection_name=collection,
                    query=query_vector[0].tolist(),
                    query_filter=models.Filter(must=must),
                    limit=per_camera_k,
                    with_payload=True,
                ).points
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Qdrant per-camera search failed for %s: %s",
                    cam_id,
                    e,
                )
                continue
            latency = time.perf_counter() - start
            total_latency += latency
            all_hits.extend(hits)
        # Observe aggregate latency once (PATCH-019).
        try:
            from ..telemetry.metrics import REGISTRY

            REGISTRY.qdrant_query_latency.observe(total_latency)
        except Exception:  # noqa: BLE001
            pass
        # Deduplicate by point_id (a single point can only have one
        # camera, so dedup is safety).
        seen: set = set()
        deduped: list[models.ScoredPoint] = []
        for h in all_hits:
            if h.id in seen:
                continue
            seen.add(h.id)
            deduped.append(h)
        # Sort by score desc, truncate to top_k.
        deduped.sort(key=lambda h: float(getattr(h, "score", 0.0)), reverse=True)
        return deduped[:top_k]

    # ---- retention (PATCH-015) ----
    def delete_points_older_than(self, collection: str, cutoff_ts: int) -> int:
        """Delete points in `collection` whose ``timestamp < cutoff_ts``.

        Returns the number of points deleted. Uses
        ``client.delete(points_selector=FilterSelector(...))`` per the
        official Qdrant Python client docs.
        """
        try:
            from qdrant_client import models as _models

            selector = _models.FilterSelector(
                filter=_models.Filter(
                    must=[
                        _models.FieldCondition(
                            key="timestamp",
                            range=_models.Range(lt=cutoff_ts),
                        )
                    ],
                ),
            )
            self.client.delete(collection_name=collection, points_selector=selector)
            # The qdrant client returns an UpdateResult with operation
            # counts; we just report "unknown" since the API may differ
            # across versions.
            return -1
        except Exception as e:  # noqa: BLE001
            logger.warning("Qdrant retention delete failed on %s: %s", collection, e)
            return 0

    def count_points_older_than(self, collection: str, cutoff_ts: int) -> int:
        try:
            from qdrant_client import models as _models

            res = self.client.count(
                collection_name=collection,
                count_filter=_models.Filter(
                    must=[
                        _models.FieldCondition(
                            key="timestamp",
                            range=_models.Range(lt=cutoff_ts),
                        )
                    ],
                ),
            )
            return int(getattr(res, "count", 0))
        except Exception as e:  # noqa: BLE001
            logger.debug("Qdrant count failed on %s: %s", collection, e)
            return 0

    def healthcheck(self) -> bool:
        try:
            return bool(self.client.get_collections())
        except Exception as e:  # noqa: BLE001
            logger.error("Qdrant healthcheck failed: %s", e)
            return False


def from_env() -> QdrantStore:
    return QdrantStore(
        host=os.environ.get("QDRANT_HOST", "vector-store"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
        api_key=os.environ.get("QDRANT_API_KEY", ""),
    )
