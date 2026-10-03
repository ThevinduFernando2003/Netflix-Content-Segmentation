"""Bootstrap stability of cluster labels.

The pairwise adjusted Rand index is computed on the overlapping rows of each
pair of resamples. ARI is invariant to label permutation, so the cluster ids
do not need to be aligned first.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from features.build_pipeline import build_segmentation_model

logger = logging.getLogger(__name__)


def adjusted_rand_on_overlap(
    index_a: np.ndarray,
    labels_a: np.ndarray,
    index_b: np.ndarray,
    labels_b: np.ndarray,
) -> float:
    """ARI on the intersection of two bootstrap index sets."""
    map_a = {int(index): int(label) for index, label in zip(index_a, labels_a, strict=True)}
    map_b = {int(index): int(label) for index, label in zip(index_b, labels_b, strict=True)}
    common = sorted(set(map_a) & set(map_b))
    if len(common) < 2:
        return float("nan")
    left = [map_a[index] for index in common]
    right = [map_b[index] for index in common]
    return float(adjusted_rand_score(left, right))


def _fit_bootstrap(task: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """One resample. Top-level so joblib can pickle it."""
    model = build_segmentation_model(
        task["config"],
        representation=task["representation"],
        k=task["k"],
        weights=task["weights"],
        seed=task["seed"],
        drop_type=task["drop_type"],
        drop_rating=task["drop_rating"],
        drop_primary_genre=task["drop_primary_genre"],
        fit_gmm=False,
        compute_silhouette=False,
    )
    model.fit(task["frame"].iloc[task["chosen"]])
    subset = task["frame"].iloc[task["chosen"]]
    first_position: dict[str, int] = {}
    for position, show_id in zip(
        task["chosen"], subset["show_id"].astype(str).str.strip(), strict=True
    ):
        first_position.setdefault(show_id, int(position))
    kept = [first_position[show_id] for show_id in model.show_ids_]
    return np.asarray(kept, dtype=int), np.asarray(model.labels_, dtype=int)


def bootstrap_stability(
    frame: pd.DataFrame,
    config: dict[str, Any],
    k: int,
    representation: str,
    n_bootstrap: int = 20,
    fraction: float = 0.8,
    seed: int = 42,
    weights: dict[str, float] | None = None,
    drop_type: bool = False,
    drop_rating: bool = False,
    drop_primary_genre: bool = False,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Refit the whole pipeline on each resample and summarise pairwise ARI."""
    rng = np.random.default_rng(seed)
    n_rows = len(frame)
    sample_size = max(int(round(fraction * n_rows)), k + 2)
    sample_size = min(sample_size, n_rows)
    tasks = []
    for _ in range(n_bootstrap):
        child_seed = int(rng.integers(0, 2**31 - 1))
        chosen = np.sort(rng.choice(n_rows, size=sample_size, replace=False))
        tasks.append(
            {
                "frame": frame,
                "chosen": chosen,
                "config": config,
                "representation": representation,
                "k": int(k),
                "weights": weights,
                "seed": child_seed,
                "drop_type": drop_type,
                "drop_rating": drop_rating,
                "drop_primary_genre": drop_primary_genre,
            }
        )
    if n_jobs == 1:
        runs = [_fit_bootstrap(task) for task in tasks]
    else:
        from joblib import Parallel, delayed

        runs = Parallel(n_jobs=n_jobs)(delayed(_fit_bootstrap)(task) for task in tasks)
    logger.info(
        "Stability k=%s finished %s resamples representation=%s",
        k,
        n_bootstrap,
        representation,
    )
    pair_scores: list[float] = []
    for left in range(n_bootstrap):
        for right in range(left + 1, n_bootstrap):
            score = adjusted_rand_on_overlap(
                runs[left][0], runs[left][1], runs[right][0], runs[right][1]
            )
            if np.isfinite(score):
                pair_scores.append(float(score))
    values = np.asarray(pair_scores, dtype=float)
    return {
        "k": int(k),
        "n_bootstrap": int(n_bootstrap),
        "n_pairs": int(len(pair_scores)),
        "fraction": float(fraction),
        "mean_ari": float(np.mean(values)) if len(values) else float("nan"),
        "std_ari": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "pair_aris": pair_scores,
    }


def seed_stability(
    matrix: np.ndarray,
    k: int,
    seeds: list[int],
) -> dict[str, Any]:
    """ARI between k-means fits that differ only by ``random_state`` on a fixed matrix."""
    from models.cluster import fit_kmeans

    labelings = [np.asarray(fit_kmeans(matrix, k, seed).labels_, dtype=int) for seed in seeds]
    pair_scores: list[float] = []
    for left in range(len(labelings)):
        for right in range(left + 1, len(labelings)):
            pair_scores.append(float(adjusted_rand_score(labelings[left], labelings[right])))
    values = np.asarray(pair_scores, dtype=float)
    reference = [float(adjusted_rand_score(labelings[0], other)) for other in labelings[1:]]
    return {
        "seeds": [int(seed) for seed in seeds],
        "mean_pairwise_ari": float(np.mean(values)) if len(values) else float("nan"),
        "std_pairwise_ari": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "mean_ari_vs_first_seed": float(np.mean(reference)) if reference else float("nan"),
        "pair_aris": pair_scores,
    }
