"""External validation against attributes withheld from the representation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def association_scores(labels: np.ndarray, attribute: pd.Series) -> dict[str, float]:
    """ARI and NMI between cluster ids and a held-out attribute."""
    labels = np.asarray(labels)
    values = attribute.astype("string").fillna("__MISSING__").astype(str).to_numpy()
    keep = np.array([value != "" for value in values])
    if keep.sum() < 2 or len(set(labels[keep].tolist())) < 2 or len(set(values[keep].tolist())) < 2:
        return {"ari": float("nan"), "nmi": float("nan"), "n": int(keep.sum())}
    return {
        "ari": float(adjusted_rand_score(values[keep], labels[keep])),
        "nmi": float(normalized_mutual_info_score(values[keep], labels[keep])),
        "n": int(keep.sum()),
    }


def contingency_table(
    labels: np.ndarray,
    attribute: pd.Series,
    normalise: str = "row",
) -> dict[str, Any]:
    """Cluster x attribute counts, optionally row-normalised (each cluster sums to 1)."""
    frame = pd.DataFrame(
        {
            "cluster": np.asarray(labels, dtype=int),
            "attribute": attribute.astype("string").fillna("__MISSING__").astype(str),
        }
    )
    table = pd.crosstab(frame["cluster"], frame["attribute"])
    if normalise == "row":
        denom = table.sum(axis=1).replace(0, np.nan)
        normalised = table.div(denom, axis=0).fillna(0.0)
    else:
        normalised = table.astype(float)
    return {
        "clusters": [int(value) for value in table.index.tolist()],
        "attributes": [str(value) for value in table.columns.tolist()],
        "counts": table.to_numpy().astype(int).tolist(),
        "normalised": normalised.to_numpy().astype(float).tolist(),
    }
