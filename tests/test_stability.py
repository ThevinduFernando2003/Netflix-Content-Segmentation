"""Bootstrap ARI wrapper and reproducibility."""

from __future__ import annotations

import numpy as np

from models.stability import adjusted_rand_on_overlap, bootstrap_stability


def test_identical_labels_have_ari_one() -> None:
    index = np.arange(9)
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert adjusted_rand_on_overlap(index, labels, index, labels) == 1.0


def test_random_labels_have_ari_near_zero() -> None:
    rng = np.random.default_rng(0)
    index = np.arange(300)
    left = rng.integers(0, 5, size=300)
    right = rng.integers(0, 5, size=300)
    score = adjusted_rand_on_overlap(index, left, index, right)
    assert abs(score) < 0.15


def test_bootstrap_pair_count_and_seed(raw_mini, mini_config: dict) -> None:
    kwargs = {
        "frame": raw_mini,
        "config": mini_config,
        "k": 3,
        "representation": "r1",
        "n_bootstrap": 4,
        "fraction": 0.8,
        "seed": 42,
        "n_jobs": 1,
    }
    first = bootstrap_stability(**kwargs)
    second = bootstrap_stability(**kwargs)
    assert first["n_pairs"] == 6
    assert first["mean_ari"] == second["mean_ari"]
    assert 0.0 <= first["mean_ari"] <= 1.0
