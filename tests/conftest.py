"""Shared fixtures. Thread counts are pinned before NumPy work so labels stay deterministic."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
from pathlib import Path

import pandas as pd
import pytest

from config import load_config
from data.load import load_titles
from features.build_pipeline import build_segmentation_model
from features.engineer import FeatureEngineer
from interpret.profile import build_cluster_profiles
from models.persistence import save_model

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "mini_netflix.csv"


@pytest.fixture(scope="session")
def mini_config() -> dict:
    config = load_config(ROOT / "config.yaml")
    config["features"]["tfidf"]["min_df"] = 1
    config["features"]["tfidf"]["max_features"] = 500
    config["features"]["svd_components"] = 8
    config["features"]["categorical_min_frequency"] = 1
    config["random_seed"] = 42
    return config


@pytest.fixture(scope="session")
def raw_mini() -> pd.DataFrame:
    return load_titles(FIXTURE)


@pytest.fixture(scope="session")
def engineered_mini(raw_mini: pd.DataFrame, mini_config: dict) -> pd.DataFrame:
    return FeatureEngineer(mini_config).fit_transform(raw_mini)


@pytest.fixture(scope="session")
def fitted_mini(raw_mini: pd.DataFrame, mini_config: dict):
    model = build_segmentation_model(
        mini_config,
        representation="r1",
        k=3,
        seed=42,
        fit_gmm=True,
        compute_silhouette=True,
    )
    model.fit(raw_mini)
    return model


@pytest.fixture(scope="session")
def api_client(fitted_mini, engineered_mini, tmp_path_factory: pytest.TempPathFactory):
    directory = tmp_path_factory.mktemp("artifacts")
    save_model(fitted_mini, directory / "pipeline.joblib")
    rows = [fitted_mini._row_payload(index) for index in range(len(fitted_mini.labels_))]
    profiles = build_cluster_profiles(engineered_mini, rows, fitted_mini.labels_)
    (directory / "cluster_profiles.json").write_text(json.dumps(profiles), encoding="utf-8")
    metadata = {
        "model_version": "test-r1-k3",
        "chosen_k": 3,
        "k_reason": "Fixture groups are separable at k=3.",
        "algorithm": "spherical_kmeans",
        "representation": {"name": "r1", "blocks": ["genre", "categorical", "numeric"]},
        "validation_scores": {
            "silhouette": 0.5,
            "calinski_harabasz": 1.0,
            "davies_bouldin": 1.0,
            "bootstrap_ari": 0.9,
        },
        "external_validation": {
            "type": {"ari": 0.1, "nmi": 0.1},
            "rating": {"ari": 0.2, "nmi": 0.2},
            "primary_genre": {"ari": 0.3, "nmi": 0.3},
        },
        "row_count": int(len(fitted_mini.labels_)),
        "cluster_sizes": {},
        "cluster_labels": [
            {"cluster_id": profile["cluster_id"], "name": profile["name"], "size": profile["size"]}
            for profile in profiles
        ],
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    projection = pd.DataFrame(
        {
            "show_id": fitted_mini.show_ids_,
            "title": fitted_mini.titles_,
            "cluster": fitted_mini.labels_.astype(int),
            "type": fitted_mini.types_,
            "rating": fitted_mini.ratings_,
            "decade": ["2010s"] * len(fitted_mini.labels_),
            "pca_x": fitted_mini.distances_,
            "pca_y": fitted_mini.silhouettes_,
            "tsne_x": fitted_mini.distances_,
            "tsne_y": fitted_mini.silhouettes_,
            "umap_x": fitted_mini.distances_,
            "umap_y": fitted_mini.silhouettes_,
        }
    )
    projection.to_parquet(directory / "projection_2d.parquet", index=False)
    os.environ["ARTIFACT_DIR"] = str(directory)
    os.environ["RATE_LIMIT_PER_MINUTE"] = "10000"
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as client:
        yield client
