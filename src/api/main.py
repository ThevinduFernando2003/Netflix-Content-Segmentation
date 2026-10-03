"""FastAPI service. Assignment goes through the persisted pipeline only."""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from api.deps import Artifacts, load_artifacts
from api.schemas import (
    AssignRequest,
    AssignResponse,
    ClusterProbability,
    TitleClusterResponse,
    TitleRef,
)
from evaluation.colors import CLUSTER_COLORS, cluster_color
from logging_conf import configure_logging

logger = logging.getLogger(__name__)

REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests",
    ["method", "path", "status"],
)
LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["path"],
)


def _rate_limit() -> int:
    return int(os.environ.get("RATE_LIMIT_PER_MINUTE", "120"))


class _WindowLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = {}

    def allow(self, key: str) -> bool:
        limit = _rate_limit()
        if limit <= 0:
            return True
        now = time.monotonic()
        recent = [stamp for stamp in self._hits.get(key, []) if now - stamp < 60.0]
        if len(recent) >= limit:
            self._hits[key] = recent
            return False
        recent.append(now)
        self._hits[key] = recent
        return True


LIMITER = _WindowLimiter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    app.state.ready = False
    app.state.artifacts = None
    try:
        artifacts = load_artifacts()
        app.state.artifacts = artifacts
        app.state.ready = True
        logger.info(
            "Loaded model %s k=%s rows=%s",
            artifacts.model_version,
            artifacts.metadata.get("chosen_k"),
            artifacts.metadata.get("row_count"),
        )
    except Exception:
        logger.exception("Artifacts are not loaded. /ready will return 503.")
    yield


app = FastAPI(
    title="Netflix content segmentation",
    version="1.0.0",
    lifespan=lifespan,
    description=(
        "Assigns a title to a segment fitted in feature space. "
        "t-SNE and UMAP coordinates served by /projection do not preserve distance."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        item.strip() for item in os.environ.get("CORS_ORIGINS", "*").split(",") if item.strip()
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    request.state.request_id = request_id
    client = request.client.host if request.client else "unknown"
    if not LIMITER.allow(client):
        response = JSONResponse(
            status_code=429, content={"detail": "Rate limit exceeded", "request_id": request_id}
        )
        response.headers["X-Request-ID"] = request_id
        return response
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error request_id=%s", request_id)
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
        )
    elapsed = time.perf_counter() - started
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    LATENCY.labels(path).observe(elapsed)
    logger.info(
        "request_id=%s method=%s path=%s status=%s took_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed * 1000,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("request_id=%s failed", request_id)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


def _artifacts(request: Request) -> Artifacts:
    artifacts = request.app.state.artifacts
    if not request.app.state.ready or artifacts is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded")
    return artifacts


def _profile(artifacts: Artifacts, cluster_id: int) -> dict[str, Any]:
    profile = artifacts.profiles_by_id.get(cluster_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Unknown cluster id {cluster_id}")
    return profile


def _why(assignment: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    owned = set(assignment.get("genres") or [])
    distinctive = [row["genre"] for row in profile.get("genre_lift", [])]
    reasons = [f"Shares distinctive genre: {genre}" for genre in distinctive if genre in owned][:6]
    text = str(assignment.get("text", "")).lower()
    for row in profile.get("top_terms", []):
        term = str(row["term"]).lower()
        if term and term in text:
            reasons.append(f"Overlaps c-TF-IDF term: {term}")
        if len(reasons) >= 8:
            break
    if not reasons:
        reasons.append(
            "No distinctive genre or title-term overlap; assigned by overall feature distance."
        )
    return reasons


def _probabilities(
    artifacts: Artifacts, values: list[float] | None
) -> list[ClusterProbability] | None:
    if values is None:
        return None
    named: list[ClusterProbability] = []
    for cluster_id, probability in enumerate(values):
        profile = artifacts.profiles_by_id.get(cluster_id, {})
        named.append(
            ClusterProbability(
                cluster_id=cluster_id,
                cluster_name=str(profile.get("name", f"Cluster {cluster_id}")),
                probability=float(probability),
            )
        )
    return named


def _request_frame(payload: AssignRequest) -> pd.DataFrame:
    if payload.duration_minutes is not None:
        duration = f"{payload.duration_minutes} min"
    elif payload.duration_seasons is not None:
        duration = (
            f"{payload.duration_seasons} Season"
            if payload.duration_seasons == 1
            else f"{payload.duration_seasons} Seasons"
        )
    else:
        duration = pd.NA
    return pd.DataFrame(
        [
            {
                "show_id": "NEW",
                "type": payload.type,
                "title": payload.title,
                "director": pd.NA,
                "cast": "",
                "country": payload.country if payload.country else pd.NA,
                "date_added": pd.NA,
                "release_year": payload.release_year,
                "rating": payload.rating if payload.rating else pd.NA,
                "duration": duration,
                "listed_in": ", ".join(payload.genres),
                "description": payload.description,
            }
        ]
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict[str, str]:
    if not request.app.state.ready:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded")
    return {"status": "ready"}


@app.get("/meta")
def meta(request: Request) -> dict[str, Any]:
    artifacts = _artifacts(request)
    body = dict(artifacts.metadata)
    body["cluster_colors"] = {
        str(profile["cluster_id"]): cluster_color(int(profile["cluster_id"]))
        for profile in artifacts.profiles
    }
    body["palette"] = CLUSTER_COLORS
    body["projection_caveat"] = (
        "t-SNE and UMAP do not preserve global distances or cluster sizes. "
        "Colour the projection by cluster; do not read gaps or areas as structure. "
        "Clustering was done in feature space."
    )
    return body


@app.get("/figures/{name}")
def figure(name: str, request: Request) -> FileResponse:
    """Serve a precomputed report figure baked into the model directory."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.png", name):
        raise HTTPException(status_code=404, detail="Unknown figure")
    artifacts = _artifacts(request)
    path = artifacts.directory / "figures" / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Figure {name} is not in this build")
    return FileResponse(path, media_type="image/png")


@app.get("/clusters")
def clusters(request: Request) -> dict[str, Any]:
    artifacts = _artifacts(request)
    summary = []
    for profile in artifacts.profiles:
        summary.append(
            {
                "cluster_id": profile["cluster_id"],
                "name": profile["name"],
                "size": profile["size"],
                "percentage": profile["percentage"],
                "interpretable": profile["interpretable"],
                "genre_lift": profile["genre_lift"][:8],
                "top_terms": profile["top_terms"][:15],
                "representative_titles": profile["representative_titles"],
                "color": cluster_color(int(profile["cluster_id"])),
            }
        )
    return {"clusters": summary}


@app.get("/clusters/{cluster_id}")
def cluster_detail(cluster_id: int, request: Request) -> dict[str, Any]:
    artifacts = _artifacts(request)
    profile = dict(_profile(artifacts, cluster_id))
    profile["color"] = cluster_color(cluster_id)
    return profile


@app.get("/clusters/{cluster_id}/titles")
def cluster_titles(
    cluster_id: int,
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    artifacts = _artifacts(request)
    _profile(artifacts, cluster_id)
    members = artifacts.model.members(cluster_id)
    page = members[offset : offset + limit]
    return {
        "cluster_id": cluster_id,
        "total": len(members),
        "limit": limit,
        "offset": offset,
        "titles": page,
    }


@app.get("/titles/{show_id}/cluster", response_model=TitleClusterResponse)
def title_cluster(show_id: str, request: Request) -> TitleClusterResponse:
    artifacts = _artifacts(request)
    record = artifacts.model.title_record(show_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown show_id {show_id}")
    profile = _profile(artifacts, int(record["cluster_id"]))
    return TitleClusterResponse(
        show_id=record["show_id"],
        title=record["title"],
        cluster_id=int(record["cluster_id"]),
        cluster_name=str(profile["name"]),
        distance_to_centroid=float(record["distance_to_centroid"]),
        silhouette=record["silhouette"],
        model_version=artifacts.model_version,
    )


@app.post("/assign", response_model=AssignResponse)
def assign(payload: AssignRequest, request: Request) -> AssignResponse:
    artifacts = _artifacts(request)
    started = time.perf_counter()
    # Snapshot state so tests can prove this handler does not refit.
    centers_before = np.array(artifacts.model.clusterer_.cluster_centers_, copy=True)
    labels_before = np.array(artifacts.model.labels_, copy=True)
    assigned = artifacts.model.assign_frame(_request_frame(payload))[0]
    if not np.allclose(centers_before, artifacts.model.clusterer_.cluster_centers_):
        raise HTTPException(status_code=500, detail="Assignment mutated cluster centers")
    if not np.array_equal(labels_before, artifacts.model.labels_):
        raise HTTPException(status_code=500, detail="Assignment mutated training labels")
    profile = _profile(artifacts, int(assigned["cluster_id"]))
    vector = artifacts.model.transform_matrix(_request_frame(payload))[0][0]
    neighbours = artifacts.model.nearest_in_cluster(vector, int(assigned["cluster_id"]), n=5)
    return AssignResponse(
        cluster_id=int(assigned["cluster_id"]),
        cluster_name=str(profile["name"]),
        distance_to_centroid=float(assigned["distance_to_centroid"]),
        membership_probabilities=_probabilities(artifacts, assigned["membership_probabilities"]),
        is_outlier=bool(assigned["is_outlier"]),
        nearest_titles=[
            TitleRef(show_id=row["show_id"], title=row["title"], distance=float(row["distance"]))
            for row in neighbours
        ],
        why=_why(assigned, profile),
        unknown_genres=list(assigned["unknown_genres"]),
        model_version=artifacts.model_version,
        took_ms=(time.perf_counter() - started) * 1000,
    )


@app.get("/projection")
def projection(
    request: Request, sample: int = Query(default=1500, ge=10, le=20000)
) -> dict[str, Any]:
    artifacts = _artifacts(request)
    frame = artifacts.projection
    if frame is None or frame.empty:
        raise HTTPException(status_code=503, detail="Projection artifact is missing")
    total = int(len(frame))
    if sample < total:
        rng = np.random.default_rng(42)
        chosen = np.sort(rng.choice(total, size=sample, replace=False))
        frame = frame.iloc[chosen]
    records = frame.to_dict(orient="records")
    return {
        "sample_size": len(records),
        "total": total,
        "caveat": (
            "Projection distances are not interpretable. t-SNE and UMAP do not preserve "
            "global distances or cluster area. Clusters were fit in feature space."
        ),
        "points": records,
    }


@app.get("/similar/{show_id}")
def similar(
    show_id: str, request: Request, n: int = Query(default=5, ge=1, le=50)
) -> dict[str, Any]:
    artifacts = _artifacts(request)
    record = artifacts.model.title_record(show_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Unknown show_id {show_id}")
    position = artifacts.model.id_to_pos_[str(show_id)]
    vector = artifacts.model.matrix_[position].astype(np.float64)
    neighbours = artifacts.model.nearest_in_cluster(
        vector,
        int(record["cluster_id"]),
        n=n,
        exclude_show_id=str(show_id),
    )
    return {
        "show_id": show_id,
        "cluster_id": int(record["cluster_id"]),
        "neighbours": [
            {"show_id": row["show_id"], "title": row["title"], "distance": float(row["distance"])}
            for row in neighbours
        ],
    }


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
