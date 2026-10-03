"""Project configuration and the global random seed."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Threaded into every estimator that accepts a seed. Do not change per call site.
RANDOM_SEED = 42

CORE_COLUMNS = [
    "show_id",
    "type",
    "title",
    "director",
    "country",
    "date_added",
    "release_year",
    "rating",
    "duration",
    "listed_in",
]
OPTIONAL_COLUMNS = ["cast", "description"]


def project_root() -> Path:
    """Locate the repository root whether we run from source or a container."""
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "config.yaml").is_file():
            return candidate
    return Path.cwd()


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    root = project_root()
    config_path = Path(path) if path is not None else root / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config at {config_path} must be a mapping.")
    loaded["_config_path"] = str(config_path)
    loaded["_project_root"] = str(root)
    return loaded


def project_path(config: dict[str, Any], key: str) -> Path:
    root = Path(config.get("_project_root", project_root()))
    raw = config["paths"][key]
    path = Path(str(raw))
    if not path.is_absolute():
        path = root / path
    return path


def seed_from(config: dict[str, Any] | None, override: int | None = None) -> int:
    if override is not None:
        return int(override)
    if config is None:
        return RANDOM_SEED
    return int(config.get("random_seed", RANDOM_SEED))
