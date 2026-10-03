"""Feature-block shapes, scaling, and the variance-weight invariant."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.blocks import (
    CategoricalBlock,
    GenreBlock,
    NumericBlock,
    RepresentationBuilder,
    TextBlock,
    block_variance_contributions,
)
from features.gower import gower_matrix


def test_genre_block_single_and_unseen(engineered_mini: pd.DataFrame) -> None:
    block = GenreBlock().fit(engineered_mini)
    single = engineered_mini.iloc[[0]].copy()
    single["genre_list"] = pd.Series(
        [["Children & Family Movies"]], index=single.index, dtype=object
    )
    encoded = block.transform(single)
    assert encoded.shape == (1, len(block.mlb_.classes_))
    assert encoded.dtype == np.float64
    alien = engineered_mini.iloc[[0]].copy()
    alien["genre_list"] = pd.Series(
        [["Children & Family Movies", "Underwater Basket Weaving"]],
        index=alien.index,
        dtype=object,
    )
    transformed = block.transform(alien)
    assert transformed.shape[1] == len(block.mlb_.classes_)
    assert block.last_unknown_ == [["Underwater Basket Weaving"]]


def test_text_rows_are_l2_normalised(engineered_mini: pd.DataFrame) -> None:
    block = TextBlock(tfidf_params={"min_df": 1, "max_features": 200}, svd_components=8, seed=42)
    block.fit(engineered_mini)
    tfidf = block.transform_tfidf(engineered_mini)
    norms = np.linalg.norm(tfidf.toarray(), axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)
    reduced = block.transform(engineered_mini)
    reduced_norms = np.linalg.norm(reduced, axis=1)
    assert np.allclose(reduced_norms, 1.0, atol=1e-5)


def test_numeric_block_is_standardised_and_has_indicators(engineered_mini: pd.DataFrame) -> None:
    block = NumericBlock().fit(engineered_mini)
    transformed = block.transform(engineered_mini)
    assert transformed.shape[0] == len(engineered_mini)
    assert transformed.shape[1] > len(block.columns_)
    for index, variance in enumerate(block.scaler_.var_):
        if variance > 1e-8:
            assert transformed[:, index].mean() == pytest.approx(0.0, abs=1e-6)
            assert transformed[:, index].std() == pytest.approx(1.0, abs=1e-6)


def test_categorical_unseen_country_is_ignored(engineered_mini: pd.DataFrame) -> None:
    block = CategoricalBlock(min_frequency=1).fit(engineered_mini)
    frame = engineered_mini.iloc[[0]].copy()
    frame.loc[frame.index[0], "primary_country"] = "Narnia"
    encoded = block.transform(frame)
    assert encoded.shape[1] == block.transform(engineered_mini.iloc[[0]]).shape[1]
    assert np.isfinite(encoded).all()


def test_block_weights_match_configured_variance(
    engineered_mini: pd.DataFrame, mini_config: dict
) -> None:
    weights = {"text": 1.0, "genre": 1.0, "categorical": 0.5, "numeric": 0.5}
    builder = RepresentationBuilder(
        config=mini_config,
        blocks=["text", "genre", "categorical", "numeric"],
        weights=weights,
        svd_components=8,
        seed=42,
        spherical=True,
    )
    builder.fit(engineered_mini)
    contributions = block_variance_contributions(builder.transform_blocks(engineered_mini))
    total = sum(contributions.values())
    weight_total = sum(weights.values())
    for name, weight in weights.items():
        assert contributions[name] / total == pytest.approx(
            weight / weight_total, rel=1e-5, abs=1e-6
        )


def test_gower_is_symmetric_float32() -> None:
    numeric = np.array([[0.0, 1.0], [1.0, 1.0], [0.5, 0.0]], dtype=np.float32)
    categorical = np.array([["a"], ["b"], ["a"]])
    distances = gower_matrix(numeric, categorical, block_size=2)
    assert distances.dtype == np.float32
    assert distances.shape == (3, 3)
    assert np.allclose(distances, distances.T)
    assert distances[0, 0] == 0
    assert distances[0, 1] > distances[0, 2]
