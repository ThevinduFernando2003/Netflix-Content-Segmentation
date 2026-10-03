"""End-to-end fit, recovery of the three fixture groups, and determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from data.clean import clean_dataframe
from features.build_pipeline import build_segmentation_model


def test_training_rows_reproduce_labels(fitted_mini, raw_mini: pd.DataFrame) -> None:
    again = fitted_mini.predict(raw_mini)
    assert np.array_equal(again, fitted_mini.labels_)


def test_single_row_and_unseen_category(fitted_mini, raw_mini: pd.DataFrame) -> None:
    row = raw_mini.iloc[[10]].copy()
    row.loc[row.index[0], "country"] = "Narnia"
    row.loc[row.index[0], "rating"] = "MADE-UP"
    labels = fitted_mini.predict(row)
    assert labels.shape == (1,)
    assigned = fitted_mini.assign_frame(row)
    assert assigned[0]["cluster_id"] == int(labels[0])


def test_three_groups_are_recovered(fitted_mini, raw_mini, mini_config: dict) -> None:
    cleaned = clean_dataframe(raw_mini, mini_config)
    assert len(cleaned) == len(fitted_mini.labels_)
    score = adjusted_rand_score(cleaned["expected_group"], fitted_mini.labels_)
    assert score > 0.6


def test_two_seeded_fits_are_identical(raw_mini: pd.DataFrame, mini_config: dict) -> None:
    first = build_segmentation_model(
        mini_config, representation="r1", k=3, seed=42, fit_gmm=False, compute_silhouette=False
    )
    second = build_segmentation_model(
        mini_config, representation="r1", k=3, seed=42, fit_gmm=False, compute_silhouette=False
    )
    first.fit(raw_mini)
    second.fit(raw_mini)
    assert np.array_equal(first.labels_, second.labels_)


def test_assign_does_not_move_centers(fitted_mini, raw_mini: pd.DataFrame) -> None:
    centers = np.array(fitted_mini.clusterer_.cluster_centers_, copy=True)
    labels = np.array(fitted_mini.labels_, copy=True)
    row = raw_mini.iloc[[0]].copy()
    row.loc[row.index[0], "description"] = (
        "A brand new story about something unseen in the catalogue."
    )
    row.loc[row.index[0], "title"] = "Brand New Title"
    row.loc[row.index[0], "show_id"] = "NEW"
    fitted_mini.assign_frame(row)
    assert np.allclose(centers, fitted_mini.clusterer_.cluster_centers_)
    assert np.array_equal(labels, fitted_mini.labels_)
    assert np.array_equal(fitted_mini.predict(raw_mini), labels)
