"""Start the API and the UI. Used by `make run` on Windows and Unix."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.setdefault("OMP_NUM_THREADS", "1")
    api = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            os.environ.get("PORT", "8000"),
        ],
        cwd=ROOT,
        env=env,
    )
    ui = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/streamlit_app.py",
            "--server.port",
            "8501",
            "--server.address",
            "0.0.0.0",
        ],
        cwd=ROOT,
        env=env,
    )
    try:
        api.wait()
    finally:
        ui.terminate()
        api.terminate()


if __name__ == "__main__":
    main()
