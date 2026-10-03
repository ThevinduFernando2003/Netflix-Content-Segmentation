"""Lift, c-TF-IDF, prototypes, and names."""

from __future__ import annotations

import numpy as np

from interpret.ctfidf import top_terms_per_document
from interpret.profile import build_cluster_profiles, genre_lift_table, representative_and_boundary


def test_lift_against_a_hand_worked_example() -> None:
    genres = [["Drama"], ["Drama"], ["Drama"], ["Comedy"], ["Comedy"]]
    labels = np.array([0, 0, 1, 1, 1])
    table = genre_lift_table(genres, labels, cluster_id=0, top_n=5, min_support=1)
    drama = next(row for row in table if row["genre"] == "Drama")
    # Cluster 0 is 2/2 drama. Global drama rate is 3/5. Lift = 1 / 0.6.
    assert drama["cluster_rate"] == 1.0
    assert drama["global_rate"] == 0.6
    assert drama["lift"] == 1 / 0.6


def test_ctfidf_finds_the_constructed_term() -> None:
    documents = [
        "dragon dragon dragon dragon castle",
        "lawsuit lawsuit lawsuit lawsuit contract",
    ]
    ranked = top_terms_per_document(documents, top_n=3)
    assert ranked[0][0][0] == "dragon"
    assert ranked[1][0][0] == "lawsuit"


def test_representatives_are_nearest_to_the_centroid() -> None:
    rows = [
        {
            "show_id": "a",
            "title": "Far",
            "distance_to_centroid": 0.9,
            "silhouette": 0.2,
            "type": "Movie",
            "rating": "PG",
            "release_year": 2010,
            "genres": [],
        },
        {
            "show_id": "b",
            "title": "Near",
            "distance_to_centroid": 0.1,
            "silhouette": 0.4,
            "type": "Movie",
            "rating": "PG",
            "release_year": 2010,
            "genres": [],
        },
        {
            "show_id": "c",
            "title": "Edge",
            "distance_to_centroid": 0.4,
            "silhouette": -0.2,
            "type": "Movie",
            "rating": "PG",
            "release_year": 2010,
            "genres": [],
        },
    ]
    representatives, boundary = representative_and_boundary(rows, n=1)
    assert representatives[0]["show_id"] == "b"
    assert boundary[0]["show_id"] == "c"


def test_every_cluster_has_a_name(fitted_mini, engineered_mini) -> None:
    rows = [fitted_mini._row_payload(index) for index in range(len(fitted_mini.labels_))]
    profiles = build_cluster_profiles(engineered_mini, rows, fitted_mini.labels_)
    assert len(profiles) == 3
    for profile in profiles:
        assert profile["name"].strip()
        assert profile["representative_titles"]
        assert profile["boundary_titles"]
        assert profile["genre_lift"]
