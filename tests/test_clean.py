"""One assertion per catalogue defect called out in the data contract."""

from __future__ import annotations

import pandas as pd
import pytest

from data.clean import clean_dataframe, listwise_deletion_count
from data.schema import NetflixSchemaError, validate_titles


def test_rating_duration_contamination_is_moved(raw_mini: pd.DataFrame, mini_config: dict) -> None:
    cleaned = clean_dataframe(raw_mini, mini_config)
    row = cleaned.loc[cleaned["show_id"] == "s1"].iloc[0]
    assert pd.isna(row["rating"])
    assert row["duration_minutes"] == 74
    assert pd.isna(row["duration_seasons"])


def test_missing_director_cast_country_rating_are_kept(
    raw_mini: pd.DataFrame, mini_config: dict
) -> None:
    cleaned = clean_dataframe(raw_mini, mini_config)
    row = cleaned.loc[cleaned["show_id"] == "s2"].iloc[0]
    assert row["director_missing"] == 1
    assert row["cast_missing"] == 1
    assert row["country_missing"] == 1
    assert pd.isna(row["rating"])
    assert len(cleaned) == 120


def test_movie_minutes_and_series_seasons(raw_mini: pd.DataFrame, mini_config: dict) -> None:
    cleaned = clean_dataframe(raw_mini, mini_config)
    series = cleaned.loc[cleaned["show_id"] == "s3"].iloc[0]
    movie = cleaned.loc[cleaned["show_id"] == "s4"].iloc[0]
    assert series["type"] == "TV Show"
    assert series["duration_seasons"] == 2
    assert pd.isna(series["duration_minutes"])
    assert movie["type"] == "Movie"
    assert movie["duration_minutes"] == 95
    assert pd.isna(movie["duration_seasons"])


def test_multi_country_and_five_genres(raw_mini: pd.DataFrame, mini_config: dict) -> None:
    cleaned = clean_dataframe(raw_mini, mini_config)
    multi = cleaned.loc[cleaned["show_id"] == "s41"].iloc[0]
    wide = cleaned.loc[cleaned["show_id"] == "s42"].iloc[0]
    assert multi["country_list"] == ["United States", "India", "France"]
    assert multi["primary_country"] == "United States"
    assert multi["title"] == "Hello, World"
    assert len(wide["genre_list"]) == 5


def test_accented_title_and_explicit_date(raw_mini: pd.DataFrame, mini_config: dict) -> None:
    cleaned = clean_dataframe(raw_mini, mini_config)
    row = cleaned.loc[cleaned["show_id"] == "s81"].iloc[0]
    assert row["title"] == "Amélie Abroad"
    assert int(row["year_added"]) == 2021


def test_duplicate_show_id_dropped(raw_mini: pd.DataFrame, mini_config: dict) -> None:
    assert raw_mini["show_id"].duplicated().sum() == 1
    cleaned = clean_dataframe(raw_mini, mini_config)
    assert cleaned["show_id"].duplicated().sum() == 0
    assert len(cleaned) == 120


def test_missing_text_is_empty_not_unknown(raw_mini: pd.DataFrame, mini_config: dict) -> None:
    frame = raw_mini.copy()
    frame.loc[frame["show_id"] == "s4", "description"] = "not given"
    cleaned = clean_dataframe(frame, mini_config)
    text = cleaned.loc[cleaned["show_id"] == "s4", "text"].iloc[0]
    assert "unknown" not in text.lower()
    assert "not given" not in text.lower()


def test_listwise_deletion_counts_missing_director(
    raw_mini: pd.DataFrame, mini_config: dict
) -> None:
    counts = listwise_deletion_count(raw_mini, mini_config)
    assert counts["listwise_partial_fields"] >= 1
    assert counts["missing_director"] >= 1


def test_schema_rejects_missing_column() -> None:
    with pytest.raises(NetflixSchemaError, match="missing required columns"):
        validate_titles(pd.DataFrame({"show_id": ["s1"]}))


def test_duration_percentile_does_not_impute_other_unit(engineered_mini: pd.DataFrame) -> None:
    movies = engineered_mini.loc[engineered_mini["type"] == "Movie", "duration_normalised"]
    series = engineered_mini.loc[engineered_mini["type"] == "TV Show", "duration_normalised"]
    assert movies.notna().all()
    assert series.notna().all()
    assert ((movies > 0) & (movies <= 1)).all()
    assert ((series > 0) & (series <= 1)).all()
