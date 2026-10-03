"""Fixed cluster colours. The API and the figures share this list."""

from __future__ import annotations

CLUSTER_COLORS = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
    "#1f78b4",
    "#b2df8a",
    "#fb9a99",
    "#cab2d6",
    "#fdbf6f",
    "#b15928",
    "#6a3d9a",
    "#33a02c",
    "#ff7f00",
    "#a6cee3",
    "#e31a1c",
    "#8dd3c7",
]


def cluster_color(cluster_id: int) -> str:
    return CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]
