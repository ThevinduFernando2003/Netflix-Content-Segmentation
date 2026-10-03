"""Feature blocks and the weighted representation.

Blocks are scaled to unit total variance and then multiplied by ``sqrt(weight)``
so each block's contribution to total variance equals its configured weight.
Rows are L2-normalised afterwards (spherical k-means). The weight check in the
tests inspects the blocks before that row normalisation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer, Normalizer, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

NUMERIC_DEFAULTS = [
    "release_year",
    "year_added",
    "years_to_platform",
    "n_genres",
    "n_cast",
    "n_countries",
    "n_directors",
    "duration_normalised",
    "director_missing",
    "country_missing",
    "cast_missing",
]


def _as_ngram(value: Any) -> tuple[int, int]:
    if isinstance(value, list | tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    return (1, 2)


class TextBlock(BaseEstimator, TransformerMixin):
    """TF-IDF (L2 rows) followed by TruncatedSVD and a second L2 normalisation.

    PCA is intentionally not used: it would densify the sparse TF-IDF matrix.
    """

    def __init__(
        self,
        tfidf_params: dict[str, Any] | None = None,
        svd_components: int = 150,
        seed: int = 42,
    ) -> None:
        self.tfidf_params = tfidf_params
        self.svd_components = svd_components
        self.seed = seed

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> TextBlock:
        del y
        params = dict(self.tfidf_params or {})
        self.vectorizer_ = TfidfVectorizer(
            ngram_range=_as_ngram(params.get("ngram_range", (1, 2))),
            min_df=int(params.get("min_df", 5)),
            max_features=int(params.get("max_features", 20_000)),
            sublinear_tf=bool(params.get("sublinear_tf", True)),
            stop_words=params.get("stop_words", "english"),
            strip_accents=params.get("strip_accents", "unicode"),
            dtype=np.float32,
            norm="l2",
        )
        text = _text_column(X)
        try:
            matrix = self.vectorizer_.fit_transform(text)
        except ValueError:
            logger.warning("TF-IDF min_df is too high for this corpus; retrying with min_df=1.")
            self.vectorizer_.set_params(min_df=1)
            matrix = self.vectorizer_.fit_transform(text)
        n_features = int(matrix.shape[1])
        n_samples = int(matrix.shape[0])
        max_components = min(n_features - 1, n_samples - 1)
        self.lsa_normalizer_ = Normalizer(copy=False)
        if max_components < 1:
            logger.warning(
                "TF-IDF rank is too small for TruncatedSVD (n_features=%s); keeping L2 TF-IDF.",
                n_features,
            )
            self.svd_ = None
            self.n_components_ = n_features
            dense = matrix.toarray().astype(np.float64)
            self.lsa_normalizer_.fit(dense)
            self.explained_variance_ratio_ = np.ones(n_features, dtype=float)
            return self
        self.n_components_ = int(min(int(self.svd_components), max_components))
        self.svd_ = TruncatedSVD(
            n_components=self.n_components_,
            random_state=int(self.seed),
            algorithm="randomized",
        )
        reduced = self.svd_.fit_transform(matrix)
        self.lsa_normalizer_.fit(reduced)
        self.explained_variance_ratio_ = np.asarray(
            self.svd_.explained_variance_ratio_, dtype=float
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        matrix = self.transform_tfidf(X)
        if self.svd_ is None:
            reduced = matrix.toarray().astype(np.float64)
        else:
            reduced = self.svd_.transform(matrix)
        return np.asarray(self.lsa_normalizer_.transform(reduced), dtype=np.float64)

    def transform_tfidf(self, X: pd.DataFrame) -> Any:
        return self.vectorizer_.transform(_text_column(X))


class GenreBlock(BaseEstimator, TransformerMixin):
    """Multi-hot genres. Unseen labels are ignored and recorded on ``last_unknown_``."""

    def __init__(self, drop_primary: bool = False) -> None:
        self.drop_primary = drop_primary

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> GenreBlock:
        del y
        self.mlb_ = MultiLabelBinarizer(sparse_output=False)
        self.mlb_.fit(self._raw_lists(X))
        self.classes_ = {str(label) for label in self.mlb_.classes_}
        self.last_unknown_: list[list[str]] = []
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        lists, unknown = self._filter_lists(X)
        self.last_unknown_ = unknown
        if len(self.mlb_.classes_) == 0:
            return np.zeros((len(X), 1), dtype=np.float64)
        return np.asarray(self.mlb_.transform(lists), dtype=np.float64)

    def _raw_lists(self, X: pd.DataFrame) -> list[list[str]]:
        lists, _ = self._filter_lists(X, drop_unknown=False)
        return lists

    def _filter_lists(
        self, X: pd.DataFrame, drop_unknown: bool = True
    ) -> tuple[list[list[str]], list[list[str]]]:
        known: set[str] = set(getattr(self, "classes_", set()))
        lists: list[list[str]] = []
        unknown_rows: list[list[str]] = []
        for genres in X["genre_list"].tolist():
            values = [str(item) for item in genres] if isinstance(genres, list) else []
            if self.drop_primary and values:
                values = values[1:]
            if drop_unknown and known:
                unknown = [item for item in values if item not in known]
                values = [item for item in values if item in known]
            else:
                unknown = []
            lists.append(values)
            unknown_rows.append(unknown)
        return lists, unknown_rows


class CategoricalBlock(BaseEstimator, TransformerMixin):
    """One-hot categoricals.

    Distinct categories are an equal Euclidean distance apart. That geometry is
    arbitrary, so this block is down-weighted by the representation builder.
    Missing values become an explicit ``__MISSING__`` level rather than a
    free-text token.
    """

    def __init__(
        self,
        min_frequency: int = 20,
        drop_type: bool = False,
        drop_rating: bool = False,
    ) -> None:
        self.min_frequency = min_frequency
        self.drop_type = drop_type
        self.drop_rating = drop_rating

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CategoricalBlock:
        del y
        self.columns_ = _categorical_columns(self.drop_type, self.drop_rating)
        frame = _categorical_frame(X, self.columns_)
        min_frequency = int(self.min_frequency)
        if min_frequency >= len(frame):
            logger.warning(
                "categorical min_frequency=%s is >= n_samples=%s; using 1.",
                min_frequency,
                len(frame),
            )
            min_frequency = 1
        self.min_frequency_ = min_frequency
        self.encoder_ = OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=self.min_frequency_ if self.min_frequency_ > 1 else None,
            sparse_output=False,
            dtype=np.float64,
        )
        self.encoder_.fit(frame)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = _categorical_frame(X, self.columns_)
        transformed = self.encoder_.transform(frame)
        return np.asarray(transformed, dtype=np.float64)


class NumericBlock(BaseEstimator, TransformerMixin):
    """Median imputation with missing indicators, then standardisation."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> NumericBlock:
        del y
        self.columns_ = [
            column for column in (self.columns or NUMERIC_DEFAULTS) if column in X.columns
        ]
        frame = X[self.columns_].apply(pd.to_numeric, errors="coerce")
        self.imputer_ = SimpleImputer(strategy="median", add_indicator=True)
        imputed = self.imputer_.fit_transform(frame)
        self.scaler_ = StandardScaler()
        self.scaler_.fit(imputed)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = X[self.columns_].apply(pd.to_numeric, errors="coerce")
        imputed = self.imputer_.transform(frame)
        scaled = self.scaler_.transform(imputed)
        return np.asarray(scaled, dtype=np.float64)


class RepresentationBuilder(BaseEstimator, TransformerMixin):
    """Concatenate weighted blocks and optionally L2-normalise each row."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        blocks: list[str] | None = None,
        weights: dict[str, float] | None = None,
        svd_components: int | None = None,
        seed: int | None = None,
        drop_type: bool = False,
        drop_rating: bool = False,
        drop_primary_genre: bool = False,
        spherical: bool = True,
    ) -> None:
        self.config = config
        self.blocks = blocks
        self.weights = weights
        self.svd_components = svd_components
        self.seed = seed
        self.drop_type = drop_type
        self.drop_rating = drop_rating
        self.drop_primary_genre = drop_primary_genre
        self.spherical = spherical

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> RepresentationBuilder:
        del y
        cfg = self.config or {}
        features = cfg.get("features", {})
        self.block_names_ = list(self.blocks or ["genre", "numeric"])
        seed = int(self.seed if self.seed is not None else cfg.get("random_seed", 42))
        svd_components = int(
            self.svd_components
            if self.svd_components is not None
            else features.get("svd_components", 150)
        )
        weight_map = dict(cfg.get("block_weights", {}))
        if self.weights:
            weight_map.update(self.weights)
        self.weight_map_ = {name: float(weight_map.get(name, 1.0)) for name in self.block_names_}
        self.block_models_: dict[str, BaseEstimator] = {}
        raw: dict[str, np.ndarray] = {}
        if "text" in self.block_names_:
            text = TextBlock(
                tfidf_params=features.get("tfidf", {}),
                svd_components=svd_components,
                seed=seed,
            )
            text.fit(X)
            self.block_models_["text"] = text
            raw["text"] = text.transform(X)
        if "genre" in self.block_names_:
            genre = GenreBlock(drop_primary=bool(self.drop_primary_genre))
            genre.fit(X)
            self.block_models_["genre"] = genre
            raw["genre"] = genre.transform(X)
        if "categorical" in self.block_names_:
            categorical = CategoricalBlock(
                min_frequency=int(features.get("categorical_min_frequency", 20)),
                drop_type=bool(self.drop_type),
                drop_rating=bool(self.drop_rating),
            )
            categorical.fit(X)
            self.block_models_["categorical"] = categorical
            raw["categorical"] = categorical.transform(X)
        if "numeric" in self.block_names_:
            numeric = NumericBlock(columns=list(features.get("numeric_columns", NUMERIC_DEFAULTS)))
            numeric.fit(X)
            self.block_models_["numeric"] = numeric
            raw["numeric"] = numeric.transform(X)
        self.scales_ = {}
        for name, matrix in raw.items():
            total = float(np.var(matrix, axis=0, ddof=0).sum())
            self.scales_[name] = float(np.sqrt(total)) if total > 1e-12 else 1.0
        self.n_features_out_ = int(sum(matrix.shape[1] for matrix in raw.values()))
        return self

    def transform_blocks(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Weighted blocks before concatenation and row L2-normalisation."""
        weighted: dict[str, np.ndarray] = {}
        for name in self.block_names_:
            model = self.block_models_[name]
            matrix = np.asarray(model.transform(X), dtype=np.float64)
            scale = self.scales_[name]
            weight = self.weight_map_[name]
            weighted[name] = matrix / scale * np.sqrt(weight)
        return weighted

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts = [self.transform_blocks(X)[name] for name in self.block_names_]
        if not parts:
            raise RuntimeError("Representation has no feature blocks.")
        stacked = np.hstack(parts)
        stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
        if self.spherical:
            return _l2_rows(stacked)
        return stacked

    def explained_variance_ratio(self) -> np.ndarray | None:
        text = self.block_models_.get("text")
        if text is None or not hasattr(text, "explained_variance_ratio_"):
            return None
        return np.asarray(text.explained_variance_ratio_, dtype=float)


def block_variance_contributions(blocks: dict[str, np.ndarray]) -> dict[str, float]:
    """Sum of column variances for each already-weighted block."""
    return {name: float(np.var(matrix, axis=0, ddof=0).sum()) for name, matrix in blocks.items()}


def _l2_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _text_column(frame: pd.DataFrame) -> list[str]:
    if "text" in frame.columns:
        return frame["text"].fillna("").astype(str).tolist()
    title = frame["title"].fillna("").astype(str) if "title" in frame.columns else ""
    description = (
        frame["description"].fillna("").astype(str) if "description" in frame.columns else ""
    )
    if isinstance(title, str):
        return [""] * len(frame)
    return (title + " " + description).str.strip().tolist()


def _categorical_columns(drop_type: bool, drop_rating: bool) -> list[str]:
    columns = ["type", "primary_country", "rating"]
    if drop_type:
        columns.remove("type")
    if drop_rating:
        columns.remove("rating")
    return columns


def _categorical_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data: dict[str, pd.Series] = {}
    for column in columns:
        series = frame[column] if column in frame.columns else pd.Series(pd.NA, index=frame.index)
        text = series.astype("string").fillna("__MISSING__")
        text = text.replace({"": "__MISSING__", "<NA>": "__MISSING__", "nan": "__MISSING__"})
        data[column] = text.astype(str)
    return pd.DataFrame(data, index=frame.index)
