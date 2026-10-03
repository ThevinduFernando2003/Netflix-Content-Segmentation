"""Row-level feature engineering fitted on the training catalogue.

``duration_normalised`` is a within-type percentile computed from the training
empirical distribution. Movies never receive an imputed season count, and series
never receive an imputed runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from data.clean import clean_dataframe


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Clean raw title rows and attach the fitted duration percentile."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureEngineer:
        del y
        cleaned = clean_dataframe(X, self.config)
        self.movie_minutes_ = _sorted_reference(cleaned, "Movie", "duration_minutes")
        self.show_seasons_ = _sorted_reference(cleaned, "TV Show", "duration_seasons")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cleaned = clean_dataframe(X, self.config)
        cleaned["duration_normalised"] = _duration_percentiles(
            cleaned, self.movie_minutes_, self.show_seasons_
        )
        return cleaned


def _sorted_reference(frame: pd.DataFrame, title_type: str, column: str) -> np.ndarray:
    values = frame.loc[frame["type"] == title_type, column].dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return np.sort(values)


def _duration_percentiles(
    frame: pd.DataFrame,
    movie_minutes: np.ndarray,
    show_seasons: np.ndarray,
) -> np.ndarray:
    """Empirical CDF percentile in ``(0, 1]`` against the training reference."""
    result = np.full(len(frame), np.nan, dtype=float)
    if len(movie_minutes):
        movie = frame["type"].eq("Movie").to_numpy()
        values = frame["duration_minutes"].to_numpy(dtype=float)
        ok = movie & np.isfinite(values)
        if ok.any():
            result[ok] = np.searchsorted(movie_minutes, values[ok], side="right") / len(
                movie_minutes
            )
    if len(show_seasons):
        series = frame["type"].eq("TV Show").to_numpy()
        values = frame["duration_seasons"].to_numpy(dtype=float)
        ok = series & np.isfinite(values)
        if ok.any():
            result[ok] = np.searchsorted(show_seasons, values[ok], side="right") / len(show_seasons)
    return result
