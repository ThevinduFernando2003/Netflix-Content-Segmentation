"""Logging setup. Library code logs; it does not print."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        root.addHandler(handler)
    logging.getLogger("sklearn").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("umap").setLevel(logging.WARNING)
