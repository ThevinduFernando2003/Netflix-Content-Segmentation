"""Fit the product model and persist one pipeline plus projections.

Cluster labels are produced in feature space. PCA, t-SNE and UMAP are stored
only so the UI can colour them. Nothing is clustered in projection coordinates.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from _common import git_sha, prepare, reports_dir, write_json  # noqa: E402
from config import project_path, seed_from  # noqa: E402
from data.load import file_sha256, load_titles  # noqa: E402
from features.build_pipeline import build_segmentation_model  # noqa: E402
from interpret.profile import build_cluster_profiles  # noqa: E402
from models.persistence import save_model  # noqa: E402
from models.select_k import safe_internal_scores  # noqa: E402


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="CSV path. Defaults to config paths.raw_csv.")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--representation", default=None)
    parser.add_argument(
        "--ignore-decision",
        action="store_true",
        help="Do not read reports/k_selection.json. Used by CI fixture builds.",
    )
    args = parser.parse_args()
    config = prepare()
    reports = reports_dir(config)
    decision_path = reports / "k_selection.json"
    decision: dict[str, Any] = {}
    if decision_path.is_file() and not args.ignore_decision and args.k is None:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    representation = str(
        args.representation or decision.get("representation") or config["product"]["representation"]
    )
    k = int(args.k or decision.get("chosen_k") or config["product"]["k"])
    reason = str(decision.get("reason", config["product"].get("k_reason", "")))
    seed = seed_from(config)
    raw_path = Path(args.data) if args.data else project_path(config, "raw_csv")
    frame = load_titles(raw_path)
    model = build_segmentation_model(
        config,
        representation=representation,
        k=k,
        seed=seed,
        fit_gmm=True,
        compute_silhouette=True,
    )
    model.fit(frame)
    # A second transform must reproduce the training labels. This is the production invariant.
    reproduced = model.predict(frame)
    if not np.array_equal(reproduced, model.labels_):
        raise RuntimeError("Transforming the training rows did not reproduce fitted labels.")
    models_dir = project_path(config, "models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, models_dir / "pipeline.joblib")
    engineered = model.engineer_.transform(frame)
    rows = [model._row_payload(index) for index in range(len(model.labels_))]
    profiles = build_cluster_profiles(
        engineered,
        rows,
        model.labels_,
        name_overrides=config.get("cluster_name_overrides") or {},
    )
    (models_dir / "cluster_profiles.json").write_text(
        json.dumps(profiles, indent=2),
        encoding="utf-8",
    )
    scores = safe_internal_scores(
        model.matrix_.astype(np.float64), model.labels_, inertia=float(model.clusterer_.inertia_)
    )
    external = {}
    if decision:
        external = decision.get("external", {})
        scores["bootstrap_ari"] = next(
            (float(row["mean_ari"]) for row in decision.get("stability", []) if int(row["k"]) == k),
            None,
        )
        scores["bootstrap_ari_std"] = next(
            (float(row["std_ari"]) for row in decision.get("stability", []) if int(row["k"]) == k),
            None,
        )
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(ROOT),
        "data_sha256": file_sha256(raw_path),
        "row_count": int(len(model.labels_)),
        "representation": {
            "name": representation,
            "blocks": config["representations"][representation]["blocks"],
            "weights": config["block_weights"],
            "svd_components": config["features"]["svd_components"],
            "duration_strategy": config["duration"]["strategy"],
            "spherical_kmeans": True,
        },
        "chosen_k": k,
        "k_reason": reason,
        "algorithm": "spherical_kmeans",
        "hyperparameters": {
            "n_init": 10,
            "init": "k-means++",
            "algorithm": "lloyd",
            "random_state": seed,
            "gmm_covariance": config["gmm"]["covariance_type"],
        },
        "validation_scores": scores,
        "external_validation": external,
        "cluster_sizes": {str(profile["cluster_id"]): profile["size"] for profile in profiles},
        "cluster_labels": [
            {
                "cluster_id": profile["cluster_id"],
                "name": profile["name"],
                "size": profile["size"],
                "percentage": profile["percentage"],
                "interpretable": profile["interpretable"],
            }
            for profile in profiles
        ],
        "model_version": f"{representation}-k{k}-{seed}",
    }
    write_json(models_dir / "metadata.json", metadata)
    projection = _project(model, config)
    projection.to_parquet(models_dir / "projection_2d.parquet", index=False)
    print(
        f"Saved model k={k} representation={representation} rows={len(model.labels_)}", flush=True
    )


def _project(model: Any, config: dict[str, Any]) -> pd.DataFrame:
    matrix = model.matrix_.astype(np.float64)
    seed = seed_from(config)
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=seed)
    pca_xy = pca.fit_transform(matrix)
    from sklearn.manifold import TSNE

    tsne_cfg = config.get("projection", {})
    perplexity = min(float(tsne_cfg.get("tsne_perplexity", 30)), max(5.0, (len(matrix) - 1) / 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=int(tsne_cfg.get("tsne_max_iter", 750)),
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    tsne_xy = tsne.fit_transform(matrix)
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(tsne_cfg.get("umap_n_neighbors", 15)),
            min_dist=float(tsne_cfg.get("umap_min_dist", 0.1)),
            random_state=seed,
            metric="euclidean",
        )
        umap_xy = reducer.fit_transform(matrix)
    except Exception as exc:
        # The runtime image does not need UMAP. Training does. Fail loudly if it is missing.
        raise RuntimeError(
            "UMAP is required at train time so the UI can read precomputed coordinates. "
            "Install requirements-dev.txt. The runtime image does not import umap."
        ) from exc
    decades = []
    for year in model.years_:
        if year is None or (isinstance(year, float) and np.isnan(year)):
            decades.append("unknown")
        else:
            decades.append(str(int(year) // 10 * 10) + "s")
    return pd.DataFrame(
        {
            "show_id": model.show_ids_,
            "title": model.titles_,
            "cluster": model.labels_.astype(int),
            "type": model.types_,
            "rating": model.ratings_,
            "decade": decades,
            "pca_x": pca_xy[:, 0],
            "pca_y": pca_xy[:, 1],
            "tsne_x": tsne_xy[:, 0],
            "tsne_y": tsne_xy[:, 1],
            "umap_x": umap_xy[:, 0],
            "umap_y": umap_xy[:, 1],
        }
    )


if __name__ == "__main__":
    main()
