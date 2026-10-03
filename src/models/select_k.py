"""Internal scores used to compare values of k. The elbow is recorded, not trusted alone."""

from __future__ import annotations

from typing import Any

import numpy as np
from kneed import KneeLocator
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from models.cluster import fit_kmeans


def safe_internal_scores(
    matrix: np.ndarray, labels: np.ndarray, inertia: float | None = None
) -> dict[str, Any]:
    """Scores on non-noise points. Returns NaN when a score is undefined."""
    labels = np.asarray(labels)
    noise_fraction = float(np.mean(labels < 0)) if len(labels) else float("nan")
    keep = labels >= 0
    kept_labels = labels[keep]
    n_clusters = int(len(set(kept_labels.tolist()))) if len(kept_labels) else 0
    result: dict[str, Any] = {
        "silhouette": float("nan"),
        "calinski_harabasz": float("nan"),
        "davies_bouldin": float("nan"),
        "inertia": float(inertia) if inertia is not None else float("nan"),
        "noise_fraction": noise_fraction,
        "n_clusters_found": n_clusters,
        "n_points_scored": int(keep.sum()),
    }
    if n_clusters < 2 or int(keep.sum()) <= n_clusters:
        return result
    scored = matrix[keep]
    result["silhouette"] = float(silhouette_score(scored, kept_labels))
    result["calinski_harabasz"] = float(calinski_harabasz_score(scored, kept_labels))
    result["davies_bouldin"] = float(davies_bouldin_score(scored, kept_labels))
    return result


def k_grid_scores(matrix: np.ndarray, k_values: list[int], seed: int) -> list[dict[str, Any]]:
    """Fit spherical k-means for each k on an already-built representation."""
    rows: list[dict[str, Any]] = []
    for k in k_values:
        if k < 2 or k >= len(matrix):
            continue
        model = fit_kmeans(matrix, k, seed)
        scores = safe_internal_scores(matrix, model.labels_, inertia=float(model.inertia_))
        scores["k"] = int(k)
        rows.append(scores)
    return rows


def gmm_information_criteria(
    matrix: np.ndarray,
    k_values: list[int],
    seed: int,
    covariance_type: str = "diag",
) -> list[dict[str, Any]]:
    """BIC and AIC for a diagonal Gaussian mixture. Full covariance is ill-conditioned here."""
    rows: list[dict[str, Any]] = []
    for k in k_values:
        if k < 2 or k >= len(matrix):
            continue
        with threadpool_limits(limits=1):
            model = GaussianMixture(
                n_components=k,
                covariance_type=covariance_type,
                random_state=seed,
                n_init=1,
                reg_covar=1e-6,
            )
            model.fit(matrix)
        rows.append(
            {
                "k": int(k),
                "bic": float(model.bic(matrix)),
                "aic": float(model.aic(matrix)),
                "covariance_type": covariance_type,
            }
        )
    return rows


def locate_elbow(k_values: list[int], inertias: list[float]) -> int | None:
    """Programmatic knee. This is a heuristic, not evidence that the clusters exist."""
    if len(k_values) < 3:
        return None
    locator = KneeLocator(k_values, inertias, curve="convex", direction="decreasing")
    knee = locator.knee
    if knee is None:
        return None
    return int(knee)


def gap_statistic(
    matrix: np.ndarray,
    k_values: list[int],
    seed: int,
    n_references: int = 5,
) -> list[dict[str, Any]]:
    """Tibshirani gap statistic against uniform data in the observed bounding box."""
    rng = np.random.default_rng(seed)
    lows = matrix.min(axis=0)
    highs = matrix.max(axis=0)
    span = np.where(highs > lows, highs - lows, 1.0)
    rows: list[dict[str, Any]] = []
    for k in k_values:
        if k < 2 or k >= len(matrix):
            continue
        observed = float(fit_kmeans(matrix, k, seed).inertia_)
        reference_logs: list[float] = []
        for ref_id in range(n_references):
            reference = lows + rng.random(matrix.shape) * span
            # Reference draws are not on the sphere. Renormalise so the null matches the geometry.
            norms = np.linalg.norm(reference, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            reference = reference / norms
            inertia = float(fit_kmeans(reference, k, seed + ref_id + 1).inertia_)
            reference_logs.append(np.log(max(inertia, 1e-12)))
        ref_mean = float(np.mean(reference_logs))
        gap = ref_mean - float(np.log(max(observed, 1e-12)))
        rows.append(
            {
                "k": int(k),
                "gap": float(gap),
                "observed_inertia": observed,
                "reference_log_inertia_mean": ref_mean,
                "n_references": int(n_references),
            }
        )
    return rows


def choose_k(
    scores: list[dict[str, Any]],
    stability: list[dict[str, Any]],
    *,
    interpretable_min: int = 5,
    interpretable_max: int = 10,
    stability_floor: float = 0.80,
) -> tuple[int, str]:
    """Pick a nameable k that is still stable.

    The maximum ARI on a sweep that includes k=2 is usually a coarse cut. On
    this feature set that cut rediscovers ``type``. Using it as the reference
    would reject every finer segmentation. The reference band is therefore
    ``interpretable_min``..``interpretable_max``. Inside that band, take the
    finest k whose mean bootstrap ARI is at least ``stability_floor``. If none
    clear the floor, take the highest mean ARI in the band.
    """
    by_k = {int(row["k"]): row for row in scores}
    band = [
        row
        for row in stability
        if interpretable_min <= int(row["k"]) <= interpretable_max and int(row["k"]) in by_k
    ]
    if not band:
        band = [row for row in stability if int(row["k"]) in by_k]
    clearing = [row for row in band if float(row["mean_ari"]) >= stability_floor]
    if clearing:
        chosen_row = max(clearing, key=lambda row: (int(row["k"]), float(row["mean_ari"])))
        selection = (
            f"the finest k in {interpretable_min}..{interpretable_max} whose mean "
            f"bootstrap ARI is at least {stability_floor:.2f}"
        )
    else:
        chosen_row = max(
            band,
            key=lambda row: (float(row["mean_ari"]), float(by_k[int(row["k"])]["silhouette"])),
        )
        selection = (
            f"the highest mean bootstrap ARI inside {interpretable_min}..{interpretable_max}, "
            f"because none reached {stability_floor:.2f}"
        )
    chosen = int(chosen_row["k"])
    best_band = max(band, key=lambda row: float(row["mean_ari"]))
    best_all = max(stability, key=lambda row: float(row["mean_ari"]))
    disagreed: list[str] = []
    if int(best_all["k"]) != chosen:
        disagreed.append(
            f"maximum bootstrap ARI at k={int(best_all['k'])} "
            f"({float(best_all['mean_ari']):.4f}), a coarser cut excluded from the ceiling"
        )
    if int(best_band["k"]) != chosen:
        disagreed.append(
            f"highest ARI inside {interpretable_min}..{interpretable_max} at "
            f"k={int(best_band['k'])} ({float(best_band['mean_ari']):.4f})"
        )
    knee = locate_elbow(
        [int(row["k"]) for row in scores],
        [float(row["inertia"]) for row in scores],
    )
    if knee is not None and knee != chosen:
        disagreed.append(f"inertia knee at k={knee}")
    best_sil = max(scores, key=lambda row: row["silhouette"])
    if int(best_sil["k"]) != chosen:
        disagreed.append(
            f"max silhouette at k={int(best_sil['k'])} ({float(best_sil['silhouette']):.4f})"
        )
    best_ch = max(scores, key=lambda row: row["calinski_harabasz"])
    if int(best_ch["k"]) != chosen:
        disagreed.append(f"max Calinski-Harabasz at k={int(best_ch['k'])}")
    best_db = min(scores, key=lambda row: row["davies_bouldin"])
    if int(best_db["k"]) != chosen:
        disagreed.append(f"best (lowest) Davies-Bouldin at k={int(best_db['k'])}")
    reason = (
        f"Chosen k={chosen} as {selection}. "
        f"Bootstrap ARI is {float(chosen_row['mean_ari']):.4f} "
        f"± {float(chosen_row['std_ari']):.4f}. "
        "Stability outranks a flattering silhouette: a compact cluster that moves "
        "under resampling is noise. "
        f"k below {interpretable_min} is not allowed to set the ceiling, because an ARI "
        "near 1.0 there is the stability of a coarse partition rather than a content segmentation."
    )
    if disagreed:
        reason += " Criteria that disagreed: " + "; ".join(disagreed) + "."
    else:
        reason += " Elbow, silhouette, Calinski-Harabasz and Davies-Bouldin agreed with this k."
    return chosen, reason
