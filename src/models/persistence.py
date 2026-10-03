"""Save and load the single fitted segmentation object."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from models.cluster import SegmentationModel


def save_model(model: SegmentationModel, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
    return target


def load_model(path: Path | str) -> SegmentationModel:
    model = joblib.load(Path(path))
    if not isinstance(model, SegmentationModel):
        raise TypeError(
            f"Artifact at {path} is {type(model).__name__}, expected SegmentationModel. "
            "Train with scripts/train.py so assignment uses the fitted pipeline."
        )
    return model


def assign_saved(path: Path | str, frame: Any) -> list[dict[str, Any]]:
    """Load a model from disk and assign. Used by the subprocess persistence test."""
    model = load_model(path)
    return model.assign_frame(frame)
