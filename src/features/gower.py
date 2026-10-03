"""Gower distance for a mixed-type comparison, computed in float32 blocks.

The full catalogue pairwise matrix is about 618 MB at float64. This routine
never materialises that matrix: callers pass a subsample and a row block size.
"""

from __future__ import annotations

import numpy as np


def gower_matrix(
    numeric: np.ndarray,
    categorical: np.ndarray,
    block_size: int = 400,
) -> np.ndarray:
    """Pairwise Gower distance. Numeric columns are range-normalised; categoricals mismatch."""
    numeric = np.asarray(numeric, dtype=np.float32)
    categorical = np.asarray(categorical)
    n_rows = numeric.shape[0]
    if categorical.shape[0] != n_rows:
        raise ValueError("Numeric and categorical matrices must have the same number of rows.")
    ranges = np.ptp(numeric, axis=0).astype(np.float32)
    ranges[ranges == 0] = np.float32(1.0)
    n_features = int(numeric.shape[1] + categorical.shape[1])
    if n_features == 0:
        raise ValueError("Gower distance needs at least one feature.")
    distances = np.zeros((n_rows, n_rows), dtype=np.float32)
    for start in range(0, n_rows, block_size):
        stop = min(n_rows, start + block_size)
        num_diff = np.abs(numeric[start:stop, None, :] - numeric[None, :, :]) / ranges
        num_sum = num_diff.sum(axis=2)
        cat_sum = (categorical[start:stop, None, :] != categorical[None, :, :]).sum(axis=2)
        distances[start:stop] = ((num_sum + cat_sum) / np.float32(n_features)).astype(np.float32)
    np.fill_diagonal(distances, np.float32(0.0))
    return distances
