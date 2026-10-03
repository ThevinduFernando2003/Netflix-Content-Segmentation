"""k sweep edges and the programmatic knee."""

from __future__ import annotations

import numpy as np

from models.cluster import embed_frame
from models.select_k import (
    choose_k,
    gap_statistic,
    gmm_information_criteria,
    k_grid_scores,
    locate_elbow,
    safe_internal_scores,
)


def test_sweep_returns_one_score_per_k(raw_mini, mini_config: dict) -> None:
    matrix, _, _ = embed_frame(raw_mini, mini_config, "r1", seed=42)
    scores = k_grid_scores(matrix, [2, 3, 4], 42)
    assert [row["k"] for row in scores] == [2, 3, 4]
    for row in scores:
        assert -1.0 <= row["silhouette"] <= 1.0


def test_knee_on_a_synthetic_elbow() -> None:
    ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    inertias = [900.0, 500.0, 200.0, 190.0, 180.0, 170.0, 160.0, 150.0, 140.0]
    assert locate_elbow(ks, inertias) == 4


def test_k_two_and_k_near_n_do_not_crash() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(6, 3))
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    scores = k_grid_scores(matrix, [2, 5], 42)
    assert [row["k"] for row in scores] == [2, 5]
    gmm = gmm_information_criteria(matrix, [2], 42)
    assert gmm[0]["k"] == 2
    assert np.isfinite(gmm[0]["bic"])
    gap = gap_statistic(matrix, [2], 42, n_references=2)
    assert gap[0]["k"] == 2
    safe = safe_internal_scores(matrix, np.array([0, 0, 0, -1, -1, -1]))
    assert safe["noise_fraction"] == 0.5


def test_choose_k_ignores_a_perfect_coarse_split() -> None:
    scores = [
        {
            "k": k,
            "silhouette": 0.2 if k == 2 else 0.12,
            "inertia": float(1000 - 20 * k),
            "calinski_harabasz": float(500 - 10 * k),
            "davies_bouldin": 2.5 - 0.01 * k,
        }
        for k in range(2, 11)
    ]
    stability = [
        {"k": 2, "mean_ari": 0.9996, "std_ari": 0.0005},
        {"k": 3, "mean_ari": 0.9968, "std_ari": 0.0019},
        {"k": 4, "mean_ari": 0.9901, "std_ari": 0.0076},
        {"k": 5, "mean_ari": 0.9075, "std_ari": 0.1346},
        {"k": 6, "mean_ari": 0.8203, "std_ari": 0.1211},
        {"k": 7, "mean_ari": 0.7630, "std_ari": 0.1158},
        {"k": 8, "mean_ari": 0.7274, "std_ari": 0.1061},
        {"k": 9, "mean_ari": 0.7248, "std_ari": 0.0980},
        {"k": 10, "mean_ari": 0.7268, "std_ari": 0.0848},
    ]
    chosen, reason = choose_k(scores, stability)
    assert chosen == 6
    assert "finest" in reason
    assert "k=2" in reason


def test_choose_k_falls_back_when_nothing_clears_the_floor() -> None:
    scores = [
        {
            "k": k,
            "silhouette": 0.1,
            "inertia": float(100 - k),
            "calinski_harabasz": float(50 - k),
            "davies_bouldin": 2.0,
        }
        for k in (5, 6)
    ]
    stability = [
        {"k": 5, "mean_ari": 0.61, "std_ari": 0.05},
        {"k": 6, "mean_ari": 0.55, "std_ari": 0.05},
    ]
    chosen, reason = choose_k(scores, stability, stability_floor=0.80)
    assert chosen == 5
    assert "none reached" in reason
