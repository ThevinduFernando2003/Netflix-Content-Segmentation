"""Load the raw catalogue and normalise optional columns."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from config import OPTIONAL_COLUMNS
from data.schema import validate_titles

logger = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_titles(path: Path | str) -> pd.DataFrame:
    """Load a CSV extract, validate the core schema, and fill absent optional columns.

    Missing ``cast`` or ``description`` become empty strings. They are never filled
    with the token ``Unknown``, which would become a real text dimension.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Catalogue CSV not found at {csv_path}. "
            "Place the extract at data/raw/netflix_titles.csv or pass an explicit path."
        )
    frame = pd.read_csv(csv_path)
    validate_titles(frame)
    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            logger.warning(
                "Column %s is absent from %s; filling with empty strings.",
                column,
                csv_path.name,
            )
            frame[column] = ""
        else:
            frame[column] = frame[column].fillna("")
    return frame
