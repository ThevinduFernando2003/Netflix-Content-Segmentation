"""Shared helpers for the command-line scripts."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from config import load_config, project_path  # noqa: E402


def git_sha(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _cell(value: Any) -> str:
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.4f}"
    return str(value).replace("|", "/")


def reports_dir(config: dict[str, Any]) -> Path:
    path = project_path(config, "reports_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(config: dict[str, Any]) -> Path:
    path = project_path(config, "figures_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare() -> dict[str, Any]:
    from logging_conf import configure_logging

    configure_logging()
    return load_config()
