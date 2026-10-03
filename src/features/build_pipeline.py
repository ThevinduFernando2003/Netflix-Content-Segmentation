"""Factory for the single persisted segmentation estimator."""

from __future__ import annotations

from typing import Any

from models.cluster import SegmentationModel


def build_segmentation_model(
    config: dict[str, Any],
    representation: str | None = None,
    k: int | None = None,
    weights: dict[str, float] | None = None,
    seed: int | None = None,
    drop_type: bool = False,
    drop_rating: bool = False,
    drop_primary_genre: bool = False,
    fit_gmm: bool = True,
    compute_silhouette: bool = True,
) -> SegmentationModel:
    """Build an unfitted model. Training and ``/assign`` share this object."""
    product = config.get("product", {})
    return SegmentationModel(
        config=config,
        representation=representation or product.get("representation", "r1"),
        k=int(k if k is not None else product.get("k", 6)),
        weights=weights,
        seed=seed,
        drop_type=drop_type,
        drop_rating=drop_rating,
        drop_primary_genre=drop_primary_genre,
        fit_gmm=fit_gmm,
        compute_silhouette=compute_silhouette,
    )
