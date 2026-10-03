"""Deterministic cleaning for mixed Netflix catalogue fields."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RATING_DURATION_RE = re.compile(r"^\d+\s*min$", re.IGNORECASE)
MINUTES_RE = re.compile(r"(\d+)\s*min", re.IGNORECASE)
SEASONS_RE = re.compile(r"(\d+)\s*seasons?", re.IGNORECASE)

DEFAULT_MISSING_TOKENS = ("not given", "unknown", "n/a", "na", "none", "null")
DEFAULT_DATE_FORMATS = (
    "%B %d, %Y",
    "%b %d, %Y",
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m/%d/%y",
)


def missing_tokens(config: dict[str, Any] | None) -> set[str]:
    raw = DEFAULT_MISSING_TOKENS
    if config is not None:
        raw = tuple(config.get("cleaning", {}).get("missing_tokens", DEFAULT_MISSING_TOKENS))
    return {token.strip().lower() for token in raw}


def date_formats(config: dict[str, Any] | None) -> list[str]:
    if config is None:
        return list(DEFAULT_DATE_FORMATS)
    return list(config.get("cleaning", {}).get("date_formats", DEFAULT_DATE_FORMATS))


def is_missing_token(value: Any, tokens: set[str]) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, (pd.Timestamp, datetime)):
        return bool(pd.isna(value))
    text = str(value).strip().lower()
    return text == "" or text in tokens or text == "nan" or text == "<na>"


def split_multi(value: Any, tokens: set[str] | None = None) -> list[str]:
    """Split a comma-separated catalogue field. Empties and missing tokens are dropped."""
    token_set = tokens if tokens is not None else set(DEFAULT_MISSING_TOKENS)
    if isinstance(value, list):
        parts = value
    elif is_missing_token(value, token_set):
        return []
    else:
        parts = str(value).split(",")
    cleaned: list[str] = []
    for part in parts:
        item = str(part).strip()
        if item and item.lower() not in token_set and item.lower() != "nan":
            cleaned.append(item)
    return cleaned


def _parse_one_date(value: Any, formats: list[str], tokens: set[str]) -> pd.Timestamp:
    if is_missing_token(value, tokens):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime)):
        stamp = pd.Timestamp(value)
        return stamp if not pd.isna(stamp) else pd.NaT
    text = str(value).strip()
    for fmt in formats:
        try:
            return pd.Timestamp(datetime.strptime(text, fmt))
        except ValueError:
            continue
    return pd.NaT


def parse_date_added(
    series: pd.Series,
    formats: list[str] | None = None,
    tokens: set[str] | None = None,
) -> tuple[pd.Series, int]:
    """Parse ``date_added`` with explicit formats only.

    Returns the parsed series and the number of non-missing values that failed.
    """
    fmt = list(formats) if formats is not None else list(DEFAULT_DATE_FORMATS)
    token_set = tokens if tokens is not None else set(DEFAULT_MISSING_TOKENS)
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        return parsed, int(parsed.isna().sum() - int(series.isna().sum()))

    parsed_values: list[pd.Timestamp] = []
    failures = 0
    for value in series.tolist():
        if is_missing_token(value, token_set):
            parsed_values.append(pd.NaT)
            continue
        stamp = _parse_one_date(value, fmt, token_set)
        if pd.isna(stamp):
            failures += 1
            parsed_values.append(pd.NaT)
        else:
            parsed_values.append(stamp)
    return pd.Series(parsed_values, index=series.index, dtype="datetime64[ns]"), failures


def clean_dataframe(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Clean a titles frame without dropping rows for missing director, cast, or country.

    Duration is parsed into ``duration_minutes`` and ``duration_seasons``. The
    scale-free percentile used for clustering is fitted later, so movies are not
    given an imputed season count.
    """
    tokens = missing_tokens(config)
    formats = date_formats(config)
    out = df.copy()

    out["show_id"] = out["show_id"].astype(str).str.strip()
    before = len(out)
    out = out.drop_duplicates(subset=["show_id"], keep="first")
    dropped = before - len(out)
    if dropped:
        logger.info("Dropped %s duplicate show_id rows.", dropped)

    for column in ("director", "cast", "country", "rating", "duration", "listed_in", "type"):
        if column not in out.columns:
            out[column] = pd.NA
        out[column] = out[column].apply(
            lambda value: pd.NA if is_missing_token(value, tokens) else value
        )

    if "description" not in out.columns:
        out["description"] = ""
    out["description"] = out["description"].apply(
        lambda value: "" if is_missing_token(value, tokens) else str(value).strip()
    )
    out["title"] = out["title"].apply(lambda value: "" if pd.isna(value) else str(value).strip())

    rating_text = out["rating"].astype("string")
    contaminated = rating_text.str.match(RATING_DURATION_RE, na=False)
    if int(contaminated.sum()):
        logger.info(
            "Moved %s duration-like rating values into duration and set rating to missing.",
            int(contaminated.sum()),
        )
        duration_missing = out["duration"].isna()
        out.loc[contaminated & duration_missing, "duration"] = out.loc[
            contaminated & duration_missing, "rating"
        ]
        out.loc[contaminated, "rating"] = pd.NA

    duration_text = out["duration"].astype("string")
    minutes = duration_text.str.extract(MINUTES_RE, expand=False)
    seasons = duration_text.str.extract(SEASONS_RE, expand=False)
    out["duration_minutes"] = pd.to_numeric(minutes, errors="coerce")
    out["duration_seasons"] = pd.to_numeric(seasons, errors="coerce")
    unparsed = (
        out["duration"].notna() & out["duration_minutes"].isna() & out["duration_seasons"].isna()
    )
    if int(unparsed.sum()):
        logger.info("Duration parse failures: %s", int(unparsed.sum()))

    # A movie must not carry a season count, and a series must not carry minutes.
    movie = out["type"].astype("string").str.strip().str.lower().eq("movie")
    series = out["type"].astype("string").str.strip().str.lower().isin(["tv show", "tv"])
    out.loc[movie, "duration_seasons"] = np.nan
    out.loc[series, "duration_minutes"] = np.nan
    out.loc[movie, "type"] = "Movie"
    out.loc[series, "type"] = "TV Show"

    out["director_list"] = out["director"].apply(lambda value: split_multi(value, tokens))
    out["cast_list"] = out["cast"].apply(lambda value: split_multi(value, tokens))
    out["country_list"] = out["country"].apply(lambda value: split_multi(value, tokens))
    out["genre_list"] = out["listed_in"].apply(lambda value: split_multi(value, tokens))

    out["primary_country"] = out["country_list"].apply(
        lambda values: values[0] if values else pd.NA
    )
    out["primary_genre"] = out["genre_list"].apply(lambda values: values[0] if values else pd.NA)
    out["director_missing"] = (
        out["director_list"].apply(lambda values: 0 if values else 1).astype(int)
    )
    out["cast_missing"] = out["cast_list"].apply(lambda values: 0 if values else 1).astype(int)
    out["country_missing"] = (
        out["country_list"].apply(lambda values: 0 if values else 1).astype(int)
    )
    out["n_directors"] = out["director_list"].apply(len).astype(int)
    out["n_cast"] = out["cast_list"].apply(len).astype(int)
    out["n_countries"] = out["country_list"].apply(len).astype(int)
    out["n_genres"] = out["genre_list"].apply(len).astype(int)
    out["description_word_count"] = (
        out["description"].fillna("").astype(str).str.split().apply(len).astype(int)
    )

    parsed_dates, date_failures = parse_date_added(df.loc[out.index, "date_added"], formats, tokens)
    out["date_added_parsed"] = parsed_dates
    out["date_parse_failure"] = parsed_dates.isna() & df.loc[out.index, "date_added"].apply(
        lambda value: not is_missing_token(value, tokens)
    )
    out.attrs["date_parse_failures"] = int(date_failures)
    out["year_added"] = out["date_added_parsed"].dt.year.astype("float")
    out["release_year"] = pd.to_numeric(out["release_year"], errors="coerce")
    out["years_to_platform"] = out["year_added"] - out["release_year"]
    out["text"] = (
        out["title"].fillna("").astype(str).str.strip()
        + " "
        + out["description"].fillna("").astype(str).str.strip()
    ).str.strip()
    return out.reset_index(drop=True)


def listwise_deletion_count(
    df: pd.DataFrame, config: dict[str, Any] | None = None
) -> dict[str, int]:
    """Count rows that would be discarded if we dropped any row with a missing field.

    ``cast`` and ``description`` are reported separately: when the source file has
    no such column, counting them as missing would discard the entire catalogue.
    """
    tokens = missing_tokens(config)
    work = df.copy()
    for column in ("cast", "description"):
        if column not in work.columns:
            work[column] = pd.NA
    fields = [
        "director",
        "cast",
        "country",
        "date_added",
        "rating",
        "duration",
        "listed_in",
        "description",
    ]
    masks = {
        field: work[field].apply(lambda value: is_missing_token(value, tokens)) for field in fields
    }
    partial_fields = ["director", "country", "date_added", "rating", "duration", "listed_in"]
    partial = np.logical_or.reduce([masks[field].to_numpy() for field in partial_fields])
    including_absent = np.logical_or.reduce([masks[field].to_numpy() for field in fields])
    return {
        "rows": int(len(work)),
        "listwise_partial_fields": int(partial.sum()),
        "listwise_including_cast_and_description": int(including_absent.sum()),
        **{f"missing_{field}": int(masks[field].sum()) for field in fields},
    }
