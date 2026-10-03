"""Figure helpers and external association tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage

from evaluation.external import association_scores, contingency_table
from evaluation.plots import (
    plot_dendrogram,
    plot_elbow,
    plot_explained_variance,
    plot_heatmap,
    plot_k_distance,
    plot_lines,
    plot_numeric_heatmap,
    plot_per_cluster_silhouette,
    plot_stability,
)
from interpret.label import name_from_profile


def test_external_scores_and_row_normalised_table() -> None:
    labels = np.array([0, 0, 1, 1])
    attribute = pd.Series(["Movie", "Movie", "TV Show", "TV Show"])
    scores = association_scores(labels, attribute)
    assert scores["ari"] == 1.0
    table = contingency_table(labels, attribute)
    normalised = np.asarray(table["normalised"])
    assert np.allclose(normalised.sum(axis=1), 1.0)


def test_plots_write_files(tmp_path) -> None:
    plot_elbow(tmp_path / "elbow.png", [2, 3, 4], [9.0, 4.0, 3.0], 3)
    labels = np.array([0, 0, 0, 1, 1, 1])
    silhouettes = np.array([0.2, 0.3, 0.1, 0.4, 0.2, 0.1])
    plot_per_cluster_silhouette(tmp_path / "sil.png", labels, silhouettes)
    matrix = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    plot_dendrogram(tmp_path / "dend.png", linkage(matrix, method="ward"), "ward")
    assert (tmp_path / "elbow.png").is_file()
    assert (tmp_path / "sil.png").is_file()
    assert (tmp_path / "dend.png").is_file()
    plot_lines(tmp_path / "lines.png", [2, 3], {"silhouette": [0.1, 0.2]}, "k", "score", "scores")
    plot_stability(tmp_path / "stability.png", [2, 3], [0.8, 0.7], [0.01, 0.02], [0.1, 0.2])
    plot_heatmap(tmp_path / "heat.png", np.eye(2), ["a", "b"], ["c", "d"], "heat", "x", "y", "v")
    plot_explained_variance(tmp_path / "var.png", np.array([0.4, 0.2, 0.1]))
    plot_k_distance(tmp_path / "kdist.png", np.array([0.1, 0.2, 0.8]), 0.5)
    plot_numeric_heatmap(
        tmp_path / "num.png",
        [
            {
                "cluster_id": 0,
                "name": "Example",
                "numeric_profile": [
                    {"feature": "release_year", "standardised_difference": 0.4},
                ],
            }
        ],
    )
    weak_name, evidence, interpretable = name_from_profile(
        {
            "genre_lift": [{"genre": "Dramas", "lift": 1.1, "cluster_rate": 0.4}],
            "top_terms": [],
            "composition": {
                "type": {"Movie": 10},
                "top_countries": [],
                "release_year": {"median": 2018},
            },
            "size": 10,
        }
    )
    assert weak_name == "Uninterpretable mix"
    assert interpretable is False
    assert evidence
