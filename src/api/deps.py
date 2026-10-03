"""Load fitted artifacts once. The API never refits them."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config import load_config, project_path
from models.cluster import SegmentationModel
from models.persistence import load_model


@dataclass
class Artifacts:
    model: SegmentationModel
    profiles: list[dict[str, Any]]
    profiles_by_id: dict[int, dict[str, Any]]
    metadata: dict[str, Any]
    projection: pd.DataFrame | None
    directory: Path

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "unknown"))


def artifact_dir() -> Path:
    override = os.environ.get("ARTIFACT_DIR")
    if override:
        return Path(override)
    config = load_config()
    return project_path(config, "models_dir")


def load_artifacts(directory: Path | None = None) -> Artifacts:
    root = directory or artifact_dir()
    model = load_model(root / "pipeline.joblib")
    profiles = json.loads((root / "cluster_profiles.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    projection_path = root / "projection_2d.parquet"
    projection = pd.read_parquet(projection_path) if projection_path.is_file() else None
    by_id = {int(profile["cluster_id"]): profile for profile in profiles}
    return Artifacts(
        model=model,
        profiles=profiles,
        profiles_by_id=by_id,
        metadata=metadata,
        projection=projection,
        directory=root,
    )
