"""The saved pipeline assigns the same title in a fresh interpreter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from models.persistence import save_model

ROOT = Path(__file__).resolve().parents[1]


def test_subprocess_assign_matches(fitted_mini, tmp_path: Path) -> None:
    path = save_model(fitted_mini, tmp_path / "pipeline.joblib")
    in_process = int(fitted_mini.predict(_frame())[0])
    src = repr(str(ROOT / "src"))
    model_path = repr(str(path))
    script = f"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
sys.path.insert(0, {src})
import pandas as pd
from models.persistence import load_model
model = load_model({model_path})
frame = pd.DataFrame([{{
    "show_id": "NEW",
    "type": "Movie",
    "title": "Paper Kites",
    "director": None,
    "cast": "",
    "country": "United States",
    "date_added": None,
    "release_year": 2019,
    "rating": "PG",
    "duration": "98 min",
    "listed_in": "Children & Family Movies, Comedies",
    "description": "Children build kites and learn about friendship at school.",
}}])
print(int(model.predict(frame)[0]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    assert int(completed.stdout.strip()) == in_process


def _frame():
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "show_id": "NEW",
                "type": "Movie",
                "title": "Paper Kites",
                "director": None,
                "cast": "",
                "country": "United States",
                "date_added": None,
                "release_year": 2019,
                "rating": "PG",
                "duration": "98 min",
                "listed_in": "Children & Family Movies, Comedies",
                "description": "Children build kites and learn about friendship at school.",
            }
        ]
    )
