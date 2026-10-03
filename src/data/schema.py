"""Schema checks for the Netflix catalogue extract."""

from __future__ import annotations

import pandas as pd

from config import CORE_COLUMNS, OPTIONAL_COLUMNS


class NetflixSchemaError(ValueError):
    """Raised when a frame cannot be used as a Netflix titles extract."""


def validate_titles(df: pd.DataFrame) -> None:
    """Raise an actionable error if required columns are missing or the frame is empty.

    ``cast`` and ``description`` are optional because the catalogue extract shipped
    with this project does not contain them. Callers add empty strings before
    feature engineering. Every other canonical column is required.
    """
    if not isinstance(df, pd.DataFrame):
        raise NetflixSchemaError(
            f"Expected a pandas DataFrame of Netflix titles, got {type(df).__name__}."
        )
    missing = [column for column in CORE_COLUMNS if column not in df.columns]
    if missing:
        raise NetflixSchemaError(
            "Netflix titles extract is missing required columns: "
            f"{missing}. Expected at least {CORE_COLUMNS}. "
            f"Found columns: {list(df.columns)}. "
            "Optional columns (filled with empty strings when absent): "
            f"{OPTIONAL_COLUMNS}."
        )
    if len(df) == 0:
        raise NetflixSchemaError(
            "Netflix titles extract has no rows. Provide a non-empty CSV with "
            f"columns {CORE_COLUMNS}."
        )
    if df["show_id"].isna().all():
        raise NetflixSchemaError("Column show_id is entirely null.")
