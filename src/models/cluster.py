"""Persisted segmentation estimator.

Assignment always calls ``transform`` on the fitted steps. Nothing in ``predict``
or ``assign_frame`` refits the vectoriser, the scalers, or the clusterer.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from config import seed_from
from features.blocks import RepresentationBuilder
from features.engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class SegmentationModel(BaseEstimator):
    """Feature engineering, weighted representation, spherical k-means, and a GMM.

    Rows are L2-normalised before k-means, so Euclidean distance to a centroid is
    monotonically related to cosine distance. That is spherical k-means.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        representation: str = "r1",
        k: int = 6,
        weights: dict[str, float] | None = None,
        seed: int | None = None,
        drop_type: bool = False,
        drop_rating: bool = False,
        drop_primary_genre: bool = False,
        fit_gmm: bool = True,
        compute_silhouette: bool = True,
    ) -> None:
        self.config = config
        self.representation = representation
        self.k = k
        self.weights = weights
        self.seed = seed
        self.drop_type = drop_type
        self.drop_rating = drop_rating
        self.drop_primary_genre = drop_primary_genre
        self.fit_gmm = fit_gmm
        self.compute_silhouette = compute_silhouette

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> SegmentationModel:
        del y
        cfg = self.config or {}
        seed = seed_from(cfg, self.seed)
        spec = cfg.get("representations", {}).get(self.representation)
        if spec is None:
            raise KeyError(
                f"Unknown representation '{self.representation}'. "
                f"Known: {list(cfg.get('representations', {}))}."
            )
        blocks = list(spec["blocks"])
        svd = int(cfg.get("features", {}).get("svd_components", 150))
        self.engineer_ = FeatureEngineer(config=cfg)
        engineered = self.engineer_.fit_transform(X)
        self.representer_ = RepresentationBuilder(
            config=cfg,
            blocks=blocks,
            weights=self.weights,
            svd_components=svd,
            seed=seed,
            drop_type=self.drop_type,
            drop_rating=self.drop_rating,
            drop_primary_genre=self.drop_primary_genre,
            spherical=True,
        )
        gmm_cfg = cfg.get("gmm", {})
        with threadpool_limits(limits=1):
            matrix = np.asarray(self.representer_.fit_transform(engineered), dtype=np.float64)
            self.clusterer_ = KMeans(
                n_clusters=int(self.k),
                random_state=seed,
                n_init=10,
                init="k-means++",
                algorithm="lloyd",
            )
            self.clusterer_.fit(matrix)
            self.gmm_ = None
            if self.fit_gmm:
                self.gmm_ = GaussianMixture(
                    n_components=int(self.k),
                    covariance_type=str(gmm_cfg.get("covariance_type", "diag")),
                    random_state=seed,
                    n_init=int(gmm_cfg.get("n_init", 1)),
                    reg_covar=float(gmm_cfg.get("reg_covar", 1e-6)),
                )
                self.gmm_.fit(matrix)
        self.matrix_ = matrix.astype(np.float32)
        self.labels_ = np.asarray(self.clusterer_.labels_, dtype=int)
        self._store_catalogue(engineered)
        self._store_geometry(matrix)
        logger.info(
            "Fitted %s k=%s on %s rows (%s features).",
            self.representation,
            self.k,
            len(self.labels_),
            matrix.shape[1],
        )
        return self

    def transform_matrix(self, X: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        engineered = self.engineer_.transform(X)
        matrix = np.asarray(self.representer_.transform(engineered), dtype=np.float64)
        return matrix, engineered

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        matrix, _ = self.transform_matrix(X)
        with threadpool_limits(limits=1):
            return np.asarray(self.clusterer_.predict(matrix), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        if self.gmm_ is None:
            return None
        matrix, _ = self.transform_matrix(X)
        with threadpool_limits(limits=1):
            return np.asarray(self.gmm_.predict_proba(matrix), dtype=float)

    def assign_frame(self, X: pd.DataFrame) -> list[dict[str, Any]]:
        """Assign rows through the fitted pipeline. Does not refit."""
        matrix, engineered = self.transform_matrix(X)
        with threadpool_limits(limits=1):
            labels = np.asarray(self.clusterer_.predict(matrix), dtype=int)
            probabilities = (
                None if self.gmm_ is None else np.asarray(self.gmm_.predict_proba(matrix))
            )
        centers = self.clusterer_.cluster_centers_
        distances = np.linalg.norm(matrix - centers[labels], axis=1)
        genre_model = self.representer_.block_models_.get("genre")
        unknown = getattr(genre_model, "last_unknown_", None)
        if unknown is None or len(unknown) != len(X):
            unknown = [[] for _ in range(len(X))]
        assigned: list[dict[str, Any]] = []
        for row_index in range(len(X)):
            cluster_id = int(labels[row_index])
            distance = float(distances[row_index])
            threshold = float(self.p95_.get(cluster_id, float("inf")))
            proba_row = None if probabilities is None else probabilities[row_index].tolist()
            assigned.append(
                {
                    "cluster_id": cluster_id,
                    "distance_to_centroid": distance,
                    "is_outlier": bool(distance > threshold),
                    "membership_probabilities": proba_row,
                    "unknown_genres": list(unknown[row_index]),
                    "genres": list(engineered["genre_list"].iloc[row_index]),
                    "text": str(engineered["text"].iloc[row_index]),
                    "show_id": str(engineered["show_id"].iloc[row_index]),
                    "title": str(engineered["title"].iloc[row_index]),
                }
            )
        return assigned

    def members(self, cluster_id: int) -> list[dict[str, Any]]:
        mask = np.flatnonzero(self.labels_ == int(cluster_id))
        order = mask[np.argsort(self.distances_[mask])]
        rows: list[dict[str, Any]] = []
        for position in order:
            rows.append(self._row_payload(int(position)))
        return rows

    def title_record(self, show_id: str) -> dict[str, Any] | None:
        position = self.id_to_pos_.get(str(show_id))
        if position is None:
            return None
        return self._row_payload(position)

    def nearest_in_cluster(
        self,
        vector: np.ndarray,
        cluster_id: int,
        n: int = 5,
        exclude_show_id: str | None = None,
    ) -> list[dict[str, Any]]:
        mask = np.flatnonzero(self.labels_ == int(cluster_id))
        if len(mask) == 0:
            return []
        subset = self.matrix_[mask].astype(np.float64)
        distances = np.linalg.norm(subset - np.asarray(vector, dtype=np.float64), axis=1)
        order = np.argsort(distances)
        found: list[dict[str, Any]] = []
        for rank in order:
            position = int(mask[int(rank)])
            show_id = self.show_ids_[position]
            if exclude_show_id is not None and show_id == exclude_show_id:
                continue
            payload = self._row_payload(position)
            payload["distance"] = float(distances[int(rank)])
            found.append(payload)
            if len(found) >= n:
                break
        return found

    def _store_catalogue(self, engineered: pd.DataFrame) -> None:
        self.show_ids_ = engineered["show_id"].astype(str).tolist()
        self.titles_ = engineered["title"].astype(str).tolist()
        self.types_ = engineered["type"].astype(str).fillna("").tolist()
        self.ratings_ = engineered["rating"].astype("string").fillna("").astype(str).tolist()
        self.countries_ = (
            engineered["primary_country"].astype("string").fillna("").astype(str).tolist()
        )
        self.years_ = engineered["release_year"].tolist()
        self.genres_ = [list(values) for values in engineered["genre_list"].tolist()]
        self.id_to_pos_ = {show_id: index for index, show_id in enumerate(self.show_ids_)}

    def _store_geometry(self, matrix: np.ndarray) -> None:
        labels = self.labels_
        centers = self.clusterer_.cluster_centers_
        distances = np.linalg.norm(matrix - centers[labels], axis=1)
        self.distances_ = distances.astype(np.float64)
        self.p95_ = {}
        for cluster_id in range(int(self.k)):
            member_distances = distances[labels == cluster_id]
            if len(member_distances) == 0:
                self.p95_[cluster_id] = float("inf")
            else:
                self.p95_[cluster_id] = float(np.percentile(member_distances, 95))
        self.silhouettes_ = np.full(len(labels), np.nan, dtype=float)
        if self.compute_silhouette and len(np.unique(labels)) > 1 and len(labels) > int(self.k):
            from sklearn.metrics import silhouette_samples

            self.silhouettes_ = np.asarray(silhouette_samples(matrix, labels), dtype=float)

    def _row_payload(self, position: int) -> dict[str, Any]:
        year = self.years_[position]
        year_value = (
            None if year is None or (isinstance(year, float) and np.isnan(year)) else int(year)
        )
        silhouette = self.silhouettes_[position]
        silhouette_value = None if np.isnan(silhouette) else float(silhouette)
        return {
            "show_id": self.show_ids_[position],
            "title": self.titles_[position],
            "type": self.types_[position],
            "rating": self.ratings_[position],
            "country": self.countries_[position],
            "release_year": year_value,
            "genres": self.genres_[position],
            "cluster_id": int(self.labels_[position]),
            "distance_to_centroid": float(self.distances_[position]),
            "silhouette": silhouette_value,
        }


def embed_frame(
    frame: pd.DataFrame,
    config: dict[str, Any],
    representation: str,
    weights: dict[str, float] | None = None,
    seed: int | None = None,
    drop_type: bool = False,
    drop_rating: bool = False,
    drop_primary_genre: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, RepresentationBuilder]:
    """Fit the feature engineer and the weighted representation. Does not cluster."""
    spec = config.get("representations", {}).get(representation)
    if spec is None:
        raise KeyError(f"Unknown representation '{representation}'.")
    resolved_seed = seed_from(config, seed)
    engineer = FeatureEngineer(config=config)
    engineered = engineer.fit_transform(frame)
    representer = RepresentationBuilder(
        config=config,
        blocks=list(spec["blocks"]),
        weights=weights,
        svd_components=int(config.get("features", {}).get("svd_components", 150)),
        seed=resolved_seed,
        drop_type=drop_type,
        drop_rating=drop_rating,
        drop_primary_genre=drop_primary_genre,
        spherical=True,
    )
    with threadpool_limits(limits=1):
        matrix = np.asarray(representer.fit_transform(engineered), dtype=np.float64)
    return matrix, engineered, representer


def fit_kmeans(matrix: np.ndarray, k: int, seed: int, minibatch: bool = False) -> Any:
    """Fit spherical-space k-means. ``matrix`` must already be L2-normalised."""
    with threadpool_limits(limits=1):
        if minibatch:
            model = MiniBatchKMeans(
                n_clusters=k,
                random_state=seed,
                n_init=10,
                init="k-means++",
                batch_size=min(1024, len(matrix)),
            )
        else:
            model = KMeans(
                n_clusters=k,
                random_state=seed,
                n_init=10,
                init="k-means++",
                algorithm="lloyd",
            )
        model.fit(matrix)
    return model
