"""Figure writers. Cluster colours are assigned by cluster id and reused everywhere."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from evaluation.colors import cluster_color


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_lines(
    path: Path,
    x: list[int],
    series: dict[str, list[float]],
    xlabel: str,
    ylabel: str,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, values in series.items():
        ax.plot(x, values, marker="o", label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_elbow(path: Path, k_values: list[int], inertias: list[float], knee: int | None) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k_values, inertias, marker="o", color="#1f78b4")
    if knee is not None:
        ax.axvline(knee, color="#e31a1c", linestyle="--", label=f"KneeLocator k={knee}")
        ax.legend()
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow curve (heuristic, not evidence)")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_stability(
    path: Path,
    k_values: list[int],
    means: list[float],
    stds: list[float],
    silhouettes: list[float],
) -> Path:
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.errorbar(k_values, means, yerr=stds, marker="o", color="#1b9e77", label="Bootstrap ARI")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Mean pairwise ARI")
    ax1.set_ylim(-0.05, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(k_values, silhouettes, marker="s", color="#d95f02", label="Silhouette")
    ax2.set_ylabel("Silhouette")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title("Stability versus silhouette")
    ax1.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_per_cluster_silhouette(path: Path, labels: np.ndarray, silhouettes: np.ndarray) -> Path:
    """Horizontal silhouette bars, one colour per cluster, worst clusters visible."""
    labels = np.asarray(labels)
    silhouettes = np.asarray(silhouettes)
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    for cluster_id in sorted(int(value) for value in np.unique(labels)):
        values = np.sort(silhouettes[labels == cluster_id])
        size = len(values)
        y_upper = y_lower + size
        color = cluster_color(cluster_id)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, values, facecolor=color, alpha=0.85)
        ax.text(-0.05, y_lower + 0.5 * size, str(cluster_id))
        y_lower = y_upper + 10
    ax.axvline(float(np.nanmean(silhouettes)), color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Titles grouped by cluster")
    ax.set_title("Per-cluster silhouette")
    ax.set_yticks([])
    return _save(fig, path)


def plot_dendrogram(path: Path, linkage_matrix: np.ndarray, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(linkage_matrix, truncate_mode="lastp", p=30, ax=ax, color_threshold=0)
    ax.set_title(title)
    ax.set_xlabel("Cluster (truncated to the last 30 merges)")
    ax.set_ylabel("Distance")
    return _save(fig, path)


def plot_heatmap(
    path: Path,
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(max(8, len(x_labels) * 0.45), max(4, len(y_labels) * 0.45)))
    image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(x_labels)), labels=x_labels, rotation=90)
    ax.set_yticks(range(len(y_labels)), labels=y_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=cbar_label)
    return _save(fig, path)


def plot_explained_variance(path: Path, ratio: np.ndarray) -> Path:
    cumulative = np.cumsum(ratio)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, color="#7570b3")
    ax.set_xlabel("TruncatedSVD components")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title("LSA explained-variance curve")
    ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_k_distance(path: Path, distances: np.ndarray, eps: float | None) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(len(distances)), distances, color="#666666")
    if eps is not None:
        ax.axhline(eps, color="#e31a1c", linestyle="--", label=f"eps={eps:.4f}")
        ax.legend()
    ax.set_xlabel("Titles sorted by k-distance")
    ax.set_ylabel("Distance to k-th neighbour")
    ax.set_title("k-distance plot used to choose DBSCAN eps")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_numeric_heatmap(path: Path, profiles: list[dict[str, Any]]) -> Path:
    features: list[str] = []
    for profile in profiles:
        for row in profile["numeric_profile"]:
            if row["feature"] not in features:
                features.append(row["feature"])
    matrix = np.zeros((len(profiles), len(features)))
    y_labels = []
    for row_index, profile in enumerate(profiles):
        y_labels.append(f"{profile['cluster_id']}: {profile['name']}")
        lookup = {
            item["feature"]: item["standardised_difference"] for item in profile["numeric_profile"]
        }
        for col_index, feature in enumerate(features):
            matrix[row_index, col_index] = lookup.get(feature, 0.0)
    return plot_heatmap(
        path,
        matrix,
        features,
        y_labels,
        "Standardised difference from the catalogue mean",
        "Feature",
        "Cluster",
        "z",
    )
