"""Cluster profiles: lift, c-TF-IDF, numeric contrasts, prototypes and boundary titles."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from interpret.ctfidf import top_terms_per_document
from interpret.label import name_from_profile

NUMERIC_PROFILE_COLUMNS = [
    "release_year",
    "year_added",
    "years_to_platform",
    "n_genres",
    "n_cast",
    "n_countries",
    "n_directors",
    "description_word_count",
    "duration_normalised",
    "director_missing",
    "country_missing",
    "cast_missing",
]


def genre_lift_table(
    genre_lists: list[list[str]],
    labels: np.ndarray,
    cluster_id: int,
    top_n: int = 8,
    min_support: int = 3,
) -> list[dict[str, Any]]:
    """Rank genres by lift (cluster rate / global rate), not by raw frequency."""
    labels = np.asarray(labels)
    global_counts: Counter[str] = Counter()
    cluster_counts: Counter[str] = Counter()
    n_global = len(genre_lists)
    n_cluster = int(np.sum(labels == cluster_id))
    if n_global == 0 or n_cluster == 0:
        return []
    for genres, label in zip(genre_lists, labels, strict=True):
        unique = set(genres)
        global_counts.update(unique)
        if int(label) == int(cluster_id):
            cluster_counts.update(unique)
    rows: list[dict[str, Any]] = []
    for genre, count in cluster_counts.items():
        if count < min_support:
            continue
        global_rate = global_counts[genre] / n_global
        if global_rate <= 0:
            continue
        cluster_rate = count / n_cluster
        rows.append(
            {
                "genre": genre,
                "count": int(count),
                "cluster_rate": float(cluster_rate),
                "global_rate": float(global_rate),
                "lift": float(cluster_rate / global_rate),
            }
        )
    rows.sort(key=lambda item: (-item["lift"], -item["count"], item["genre"]))
    return rows[:top_n]


def numeric_profile(
    frame: pd.DataFrame,
    labels: np.ndarray,
    cluster_id: int,
    columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    columns = columns or [column for column in NUMERIC_PROFILE_COLUMNS if column in frame.columns]
    labels = np.asarray(labels)
    member = labels == cluster_id
    rows: list[dict[str, Any]] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        global_values = values.to_numpy(dtype=float)
        cluster_values = global_values[member]
        global_finite = global_values[np.isfinite(global_values)]
        cluster_finite = cluster_values[np.isfinite(cluster_values)]
        if len(global_finite) == 0 or len(cluster_finite) == 0:
            continue
        global_mean = float(np.mean(global_finite))
        global_std = float(np.std(global_finite, ddof=0))
        cluster_mean = float(np.mean(cluster_finite))
        difference = 0.0 if global_std < 1e-12 else (cluster_mean - global_mean) / global_std
        rows.append(
            {
                "feature": column,
                "mean": cluster_mean,
                "median": float(np.median(cluster_finite)),
                "global_mean": global_mean,
                "global_std": global_std,
                "standardised_difference": float(difference),
            }
        )
    return rows


def composition(frame: pd.DataFrame, labels: np.ndarray, cluster_id: int) -> dict[str, Any]:
    member = frame.loc[np.asarray(labels) == cluster_id]
    type_counts = member["type"].fillna("Missing").astype(str).value_counts().to_dict()
    rating_counts = member["rating"].fillna("Missing").astype(str).value_counts().head(8).to_dict()
    country_counts = (
        member["primary_country"].fillna("Missing").astype(str).value_counts().head(5).to_dict()
    )
    years = pd.to_numeric(member["release_year"], errors="coerce").dropna()
    return {
        "type": {str(key): int(value) for key, value in type_counts.items()},
        "ratings": {str(key): int(value) for key, value in rating_counts.items()},
        "top_countries": [
            {"country": str(key), "count": int(value)} for key, value in country_counts.items()
        ],
        "release_year": {
            "median": float(years.median()) if len(years) else None,
            "mean": float(years.mean()) if len(years) else None,
            "min": int(years.min()) if len(years) else None,
            "max": int(years.max()) if len(years) else None,
        },
    }


def representative_and_boundary(
    catalogue_rows: list[dict[str, Any]],
    n: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Closest rows to the centroid, and the lowest-silhouette rows in the cluster."""
    by_distance = sorted(catalogue_rows, key=lambda row: row["distance_to_centroid"])
    representatives = [_public_title(row) for row in by_distance[:n]]
    with_silhouette = [row for row in catalogue_rows if row.get("silhouette") is not None]
    by_silhouette = sorted(with_silhouette, key=lambda row: row["silhouette"])
    boundary = [_public_title(row) for row in by_silhouette[:n]]
    return representatives, boundary


def build_cluster_profiles(
    frame: pd.DataFrame,
    model_rows: list[dict[str, Any]],
    labels: np.ndarray,
    name_overrides: dict[str, str] | None = None,
    top_genres: int = 8,
    top_terms: int = 15,
    n_examples: int = 5,
) -> list[dict[str, Any]]:
    """Profile every cluster. Names come from evidence, with optional overrides."""
    labels = np.asarray(labels, dtype=int)
    overrides = {int(key): value for key, value in (name_overrides or {}).items()}
    documents: list[str] = []
    cluster_ids = sorted(int(value) for value in np.unique(labels) if int(value) >= 0)
    grouped: dict[int, list[dict[str, Any]]] = {cluster_id: [] for cluster_id in cluster_ids}
    for row in model_rows:
        grouped[int(row["cluster_id"])].append(row)
    for cluster_id in cluster_ids:
        text = " ".join(str(row["title"]) for row in grouped[cluster_id])
        documents.append(text if text.strip() else "empty")
    term_lists = top_terms_per_document(documents, top_n=top_terms)
    n_rows = len(labels)
    profiles: list[dict[str, Any]] = []
    genre_lists = [list(values) for values in frame["genre_list"].tolist()]
    for position, cluster_id in enumerate(cluster_ids):
        members = grouped[cluster_id]
        size = len(members)
        lift = genre_lift_table(genre_lists, labels, cluster_id, top_n=top_genres)
        numbers = numeric_profile(frame, labels, cluster_id)
        mix = composition(frame, labels, cluster_id)
        representatives, boundary = representative_and_boundary(members, n=n_examples)
        terms = [{"term": term, "score": score} for term, score in term_lists[position]]
        draft = {
            "cluster_id": cluster_id,
            "size": int(size),
            "percentage": float(100.0 * size / n_rows) if n_rows else 0.0,
            "small_cluster": bool(n_rows and (size / n_rows) < 0.02),
            "genre_lift": lift,
            "top_terms": terms,
            "numeric_profile": numbers,
            "composition": mix,
            "representative_titles": representatives,
            "boundary_titles": boundary,
        }
        name, evidence, interpretable = name_from_profile(draft)
        if cluster_id in overrides and str(overrides[cluster_id]).strip():
            name = str(overrides[cluster_id]).strip()
            evidence = evidence + " Human override applied from config cluster_name_overrides."
        draft["name"] = name
        draft["name_evidence"] = evidence
        draft["interpretable"] = interpretable
        profiles.append(draft)
    return profiles


def _public_title(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "show_id": row["show_id"],
        "title": row["title"],
        "type": row.get("type", ""),
        "rating": row.get("rating", ""),
        "release_year": row.get("release_year"),
        "distance_to_centroid": row.get("distance_to_centroid"),
        "silhouette": row.get("silhouette"),
        "genres": row.get("genres", []),
    }
