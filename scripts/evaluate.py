"""Algorithm comparison, cluster profiles, and the evaluation write-up.

Numbers are computed here. Re-run the script to refresh the reports.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
from kneed import KneeLocator  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from sklearn.cluster import DBSCAN, BisectingKMeans  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from _common import figures_dir, markdown_table, prepare, reports_dir, write_json  # noqa: E402
from config import project_path, seed_from  # noqa: E402
from data.load import load_titles  # noqa: E402
from evaluation.plots import plot_dendrogram, plot_k_distance, plot_numeric_heatmap  # noqa: E402
from interpret.profile import build_cluster_profiles  # noqa: E402
from models.cluster import fit_kmeans  # noqa: E402
from models.persistence import load_model  # noqa: E402
from models.select_k import safe_internal_scores  # noqa: E402


def main() -> None:
    config = prepare()
    models_dir = project_path(config, "models_dir")
    model = load_model(models_dir / "pipeline.joblib")
    frame = load_titles(project_path(config, "raw_csv"))
    engineered = model.engineer_.transform(frame)
    matrix = model.matrix_.astype(np.float64)
    labels = model.labels_
    k = int(model.k)
    seed = seed_from(config)
    figures = figures_dir(config)
    reports = reports_dir(config)

    rows = [model._row_payload(index) for index in range(len(labels))]
    profiles = build_cluster_profiles(
        engineered,
        rows,
        labels,
        name_overrides=config.get("cluster_name_overrides") or {},
    )
    (models_dir / "cluster_profiles.json").write_text(
        json.dumps(profiles, indent=2), encoding="utf-8"
    )
    plot_numeric_heatmap(figures / "numeric_profile_heatmap.png", profiles)
    (reports / "cluster_profiles.md").write_text(_profiles_markdown(profiles), encoding="utf-8")

    algorithms = _compare_algorithms(matrix, k, seed, config, figures)
    (reports / "algorithm_comparison.md").write_text(
        _algorithms_markdown(algorithms, k), encoding="utf-8"
    )
    metadata_path = models_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["cluster_labels"] = [
        {
            "cluster_id": profile["cluster_id"],
            "name": profile["name"],
            "size": profile["size"],
            "percentage": profile["percentage"],
            "interpretable": profile["interpretable"],
        }
        for profile in profiles
    ]
    metadata["algorithm_comparison"] = [
        {key: value for key, value in row.items() if key != "note"} | {"note": row.get("note", "")}
        for row in algorithms
    ]
    _attach_k_sweep(metadata, reports)
    copied = _copy_figures(figures, models_dir / "figures")
    metadata["figures"] = copied
    write_json(metadata_path, metadata)
    (reports / "evaluation.md").write_text(
        _evaluation_markdown(metadata, profiles, algorithms), encoding="utf-8"
    )
    print(f"Wrote profiles for {len(profiles)} clusters", flush=True)


def _attach_k_sweep(metadata: dict[str, Any], reports: Path) -> None:
    """Copy the k sweep into model metadata so the API can plot it without the reports tree."""
    path = reports / "k_selection.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    stability = {int(row["k"]): row for row in payload.get("stability", [])}
    gmm = {int(row["k"]): row for row in payload.get("gmm", [])}
    sweep = []
    for score in payload.get("scores", []):
        k = int(score["k"])
        sweep.append(
            {
                "k": k,
                "silhouette": score["silhouette"],
                "calinski_harabasz": score["calinski_harabasz"],
                "davies_bouldin": score["davies_bouldin"],
                "inertia": score["inertia"],
                "bootstrap_ari": stability.get(k, {}).get("mean_ari"),
                "bootstrap_ari_std": stability.get(k, {}).get("std_ari"),
                "bic": gmm.get(k, {}).get("bic"),
                "aic": gmm.get(k, {}).get("aic"),
            }
        )
    metadata["k_sweep"] = sweep
    metadata["ari_vs_type"] = payload.get("ari_vs_type", [])


def _copy_figures(source: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    names = [
        "elbow_product.png",
        "internal_indices.png",
        "calinski_harabasz.png",
        "gmm_bic_aic.png",
        "stability_ari.png",
        "silhouette_clusters.png",
        "contingency_type.png",
        "contingency_rating.png",
        "contingency_primary_genre.png",
        "dendrogram_ward.png",
        "dendrogram_cosine.png",
        "dbscan_k_distance.png",
        "numeric_profile_heatmap.png",
        "explained_variance_svd.png",
    ]
    copied = []
    for name in names:
        src = source / name
        if src.is_file():
            shutil.copyfile(src, destination / name)
            copied.append(name)
    return copied


def _compare_algorithms(
    matrix: np.ndarray,
    k: int,
    seed: int,
    config: dict[str, Any],
    figures: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    kmeans = fit_kmeans(matrix, k, seed, minibatch=False)
    rows.append(
        _row(
            "Spherical KMeans",
            kmeans.labels_,
            matrix,
            time.perf_counter() - started,
            float(kmeans.inertia_),
            "Primary. L2-normalised rows, Euclidean k-means.",
        )
    )

    started = time.perf_counter()
    mini = fit_kmeans(matrix, k, seed, minibatch=True)
    rows.append(
        _row(
            "MiniBatchKMeans",
            mini.labels_,
            matrix,
            time.perf_counter() - started,
            float(mini.inertia_),
            "Same k. At this row count the speed gain is real and the quality loss is the point of the comparison.",
        )
    )

    started = time.perf_counter()
    with threadpool_limits(limits=1):
        ward = linkage(matrix, method="ward")
    ward_labels = fcluster(ward, t=k, criterion="maxclust") - 1
    plot_dendrogram(
        figures / "dendrogram_ward.png",
        ward,
        "Ward dendrogram on the product representation (top 30 merges)",
    )
    rows.append(
        _row(
            "Agglomerative Ward",
            ward_labels,
            matrix,
            time.perf_counter() - started,
            None,
            "Euclidean Ward on the dense representation. Ward is not used with cosine.",
        )
    )

    started = time.perf_counter()
    cosine_labels, cosine_note, cosine_link, cosine_matrix = _cosine_agglomerative(matrix, k)
    if cosine_link is not None:
        plot_dendrogram(
            figures / "dendrogram_cosine.png",
            cosine_link,
            "Average-linkage cosine dendrogram (top 30 merges)",
        )
    rows.append(
        _row(
            "Agglomerative average cosine",
            cosine_labels,
            cosine_matrix,
            time.perf_counter() - started,
            None,
            cosine_note,
        )
    )

    started = time.perf_counter()
    with threadpool_limits(limits=1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=str(config.get("gmm", {}).get("covariance_type", "diag")),
            random_state=seed,
            n_init=1,
            reg_covar=1e-6,
        )
        gmm.fit(matrix)
    gmm_labels = gmm.predict(matrix)
    gmm_row = _row(
        "GaussianMixture diag",
        gmm_labels,
        matrix,
        time.perf_counter() - started,
        None,
        "Soft assignments. Diagonal covariance because a full covariance is ill-conditioned at this width.",
    )
    gmm_row["bic"] = float(gmm.bic(matrix))
    gmm_row["aic"] = float(gmm.aic(matrix))
    rows.append(gmm_row)

    db_row = _dbscan(matrix, config, figures)
    rows.append(db_row)

    started = time.perf_counter()
    with threadpool_limits(limits=1):
        bisect = BisectingKMeans(n_clusters=k, random_state=seed, n_init=10)
        bisect.fit(matrix)
    rows.append(
        _row(
            "BisectingKMeans",
            bisect.labels_,
            matrix,
            time.perf_counter() - started,
            float(bisect.inertia_),
            "Optional hierarchical-divisive check on the same spherical representation.",
        )
    )
    return rows


def _cosine_agglomerative(
    matrix: np.ndarray, k: int
) -> tuple[np.ndarray, str, np.ndarray | None, np.ndarray]:
    """Average linkage on cosine distance. Ward is never paired with cosine."""
    n_rows = len(matrix)
    note = "Full catalogue, average linkage, cosine distance."
    sample = matrix
    if n_rows > 4000:
        # The condensed cosine matrix is hundreds of MB. Keep the dendrogram on a fixed subsample.
        rng = np.random.default_rng(42)
        chosen = np.sort(rng.choice(n_rows, size=4000, replace=False))
        sample = matrix[chosen]
        note = (
            "Average linkage with cosine distance on a fixed 4,000-row subsample "
            "(condensed float64 distances). Ward is not used with cosine. "
            "Scores for this row are on that subsample. "
            "The product labels are still the full-data spherical k-means."
        )
    condensed = pdist(sample, metric="cosine")
    condensed = np.nan_to_num(condensed, nan=1.0, posinf=1.0)
    link = linkage(condensed, method="average")
    labels = fcluster(link, t=k, criterion="maxclust") - 1
    return labels, note, link, sample


def _dbscan(matrix: np.ndarray, config: dict[str, Any], figures: Path) -> dict[str, Any]:
    min_samples = int(config.get("dbscan", {}).get("min_samples", 15))
    neighbours = NearestNeighbors(n_neighbors=min_samples).fit(matrix)
    distances, _ = neighbours.kneighbors(matrix)
    k_distance = np.sort(distances[:, -1])
    locator = KneeLocator(
        list(range(len(k_distance))),
        k_distance.tolist(),
        curve="convex",
        direction="increasing",
    )
    eps = (
        float(locator.knee_y) if locator.knee_y is not None else float(np.quantile(k_distance, 0.9))
    )
    eps_source = (
        "KneeLocator on the k-distance curve"
        if locator.knee_y is not None
        else "90th percentile of k-distance (no knee found)"
    )
    started = time.perf_counter()
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(matrix)
    plot_k_distance(figures / "dbscan_k_distance.png", k_distance, eps)
    row = _row(
        "DBSCAN",
        labels,
        matrix,
        time.perf_counter() - started,
        None,
        (
            f"eps={eps:.4f} from {eps_source}, min_samples={min_samples}. "
            "HDBSCAN was not used: it pulls in a compiled build and is a poor fit for the runtime image. "
            "DBSCAN is the density-based result, including the noise fraction."
        ),
    )
    row["eps"] = eps
    return row


def _row(
    name: str,
    labels: np.ndarray,
    matrix: np.ndarray,
    seconds: float,
    inertia: float | None,
    note: str,
) -> dict[str, Any]:
    scores = safe_internal_scores(matrix, np.asarray(labels), inertia=inertia)
    scores["algorithm"] = name
    scores["seconds"] = float(seconds)
    scores["note"] = note
    return scores


def _algorithms_markdown(rows: list[dict[str, Any]], k: int) -> str:
    lines = [
        "# Algorithm comparison",
        "",
        "Generated by `scripts/evaluate.py` on the product representation. "
        f"Partitioning methods use k={k}. DBSCAN chooses its own cluster count and may label noise as -1.",
        "",
        markdown_table(
            [
                "algorithm",
                "seconds",
                "silhouette",
                "CH",
                "DB",
                "inertia",
                "noise_fraction",
                "n_clusters",
            ],
            [
                [
                    row["algorithm"],
                    row["seconds"],
                    row["silhouette"],
                    row["calinski_harabasz"],
                    row["davies_bouldin"],
                    row["inertia"],
                    row["noise_fraction"],
                    row["n_clusters_found"],
                ]
                for row in rows
            ],
        ),
        "",
    ]
    for row in rows:
        lines.append(f"- **{row['algorithm']}.** {row['note']}")
    db = next(row for row in rows if row["algorithm"] == "DBSCAN")
    noise = float(db["noise_fraction"])
    found = int(db["n_clusters_found"])
    if found <= 1:
        density_text = (
            f"DBSCAN found **{found}** cluster and marked **{noise:.4f}** "
            f"({noise * 100:.1f}% of titles) as noise, with eps={db.get('eps', float('nan')):.4f}. "
            "The k-distance knee is high enough that almost every title falls in one component. "
            "There is no density gap that recovers the six segments. "
            "On this spherical representation, density-based clustering collapses. "
            "That is a negative result, not a parameter that was tuned until it looked like k-means."
        )
    elif noise >= 0.5:
        density_text = (
            f"Noise fraction: **{noise:.4f}** ({noise * 100:.1f}% of titles). "
            f"Clusters found: **{found}**. eps={db.get('eps', float('nan')):.4f}. "
            "Most points were labelled noise. Distances on this representation do not form "
            "a usable neighbourhood scale. The failure is reported rather than tuned away."
        )
    else:
        density_text = (
            f"Noise fraction: **{noise:.4f}** ({noise * 100:.1f}% of titles). "
            f"Clusters found: **{found}**. eps={db.get('eps', float('nan')):.4f}."
        )
    lines.extend(
        [
            "",
            "## Density-based result",
            "",
            density_text,
            "",
            "The k-distance curve is `reports/figures/dbscan_k_distance.png`.",
            "",
            "Dendrograms: `reports/figures/dendrogram_ward.png` and `dendrogram_cosine.png`. "
            "They show nested structure. The product model is still a flat spherical k-means at the chosen k.",
            "",
        ]
    )
    return "\n".join(lines)


def _profiles_markdown(profiles: list[dict[str, Any]]) -> str:
    total = sum(profile["size"] for profile in profiles)
    lines = [
        "# Cluster profiles",
        "",
        "Generated by `scripts/evaluate.py`. Genres are ranked by lift (cluster rate / catalogue rate), "
        "not by raw frequency. Terms are class-based TF-IDF on the titles in each cluster "
        "(the extract has no synopsis). Representative titles are the five rows closest to the "
        "spherical k-means centroid. Boundary titles are the five lowest silhouettes in the cluster.",
        "",
        "## Catalogue narrative",
        "",
        _narrative(profiles, total),
        "",
        "## Sizes",
        "",
        markdown_table(
            ["id", "name", "size", "percent", "interpretable"],
            [
                [
                    profile["cluster_id"],
                    profile["name"],
                    profile["size"],
                    profile["percentage"],
                    profile["interpretable"],
                ]
                for profile in profiles
            ],
        ),
        "",
        "Clusters under 2% of the catalogue are flagged in their sections.",
        "",
        "Projection caveat: PCA, t-SNE and UMAP coordinates in `models/projection_2d.parquet` "
        "are a view of the feature space. t-SNE and UMAP do not preserve global distances or "
        "cluster sizes. Apparent gaps between blobs are not interpretable. "
        "Titles were clustered in the weighted feature space, then the projection was coloured. "
        "Nothing was clustered in t-SNE or UMAP coordinates.",
        "",
    ]
    for profile in profiles:
        lines.extend(_one_profile(profile, total))
    lines.append("Numeric heatmap: `reports/figures/numeric_profile_heatmap.png`.")
    lines.append("")
    return "\n".join(lines)


def _narrative(profiles: list[dict[str, Any]], total: int) -> str:
    named = [profile for profile in profiles if profile["interpretable"]]
    unnamed = [profile for profile in profiles if not profile["interpretable"]]
    bits = [
        f"{profile['name']} ({profile['size']} titles, {profile['percentage']:.1f}%)"
        for profile in named
    ]
    text = (
        f"The catalogue of {total} titles separates into {len(profiles)} segments in the product "
        "feature space. "
        + (
            "; ".join(bits) + ". "
            if bits
            else "No segment cleared the lift threshold used for naming. "
        )
    )
    if unnamed:
        ids = ", ".join(str(profile["cluster_id"]) for profile in unnamed)
        text += (
            f"Cluster(s) {ids} did not show a genre lift high enough to support a theme. "
            "They are left as an uninterpretable mix rather than given a decorative name."
        )
    else:
        text += "Every segment had a genre lift strong enough to name from the evidence below. "
    mixed = []
    for profile in profiles:
        rates = [
            float(row["cluster_rate"])
            for row in profile.get("genre_lift") or []
            if float(row["lift"]) >= 1.8
        ]
        if rates and max(rates) < 0.5:
            leader = (profile.get("genre_lift") or [{}])[0]
            mixed.append(
                f"{profile['name']} (cluster {profile['cluster_id']}, "
                f"top lift {leader.get('genre', 'n/a')} covers "
                f"{float(leader.get('cluster_rate') or 0):.0%} of the cluster)"
            )
    if mixed:
        text += (
            "These names describe the highest-lift genres, not a pure theme: "
            + "; ".join(mixed)
            + ". "
            "Read those sections as mixed bags with a real centre, not as clean shelves."
        )
    return text


def _one_profile(profile: dict[str, Any], total: int) -> list[str]:
    small = ""
    if profile["small_cluster"]:
        small = (
            " This cluster is under 2% of the catalogue. Treat it as a niche if the lift is specific, "
            "or as a noise pocket if the lift is a grab-bag of unrelated genres."
        )
    lines = [
        f"## {profile['cluster_id']}. {profile['name']}",
        "",
        f"Size **{profile['size']}** / {total} ({profile['percentage']:.2f}%).{small}",
        "",
        profile["name_evidence"],
        "",
        "### Distinctive genres by lift",
        "",
        markdown_table(
            ["genre", "count", "cluster_rate", "global_rate", "lift"],
            [
                [row["genre"], row["count"], row["cluster_rate"], row["global_rate"], row["lift"]]
                for row in profile["genre_lift"]
            ]
            or [["(none)", 0, 0.0, 0.0, 0.0]],
        ),
        "",
        "### c-TF-IDF terms from titles",
        "",
        ", ".join(f"{row['term']} ({row['score']:.3f})" for row in profile["top_terms"][:15])
        or "(none)",
        "",
        "### Numeric profile",
        "",
        markdown_table(
            ["feature", "mean", "median", "global_mean", "z"],
            [
                [
                    row["feature"],
                    row["mean"],
                    row["median"],
                    row["global_mean"],
                    row["standardised_difference"],
                ]
                for row in profile["numeric_profile"]
            ],
        ),
        "",
        "### Composition",
        "",
        f"Type: {profile['composition']['type']}.",
        "",
        f"Top countries: {profile['composition']['top_countries']}.",
        "",
        f"Ratings: {profile['composition']['ratings']}.",
        "",
        f"Release year: {profile['composition']['release_year']}.",
        "",
        "### Representative titles",
        "",
        markdown_table(
            ["show_id", "title", "distance"],
            [
                [row["show_id"], row["title"], row["distance_to_centroid"]]
                for row in profile["representative_titles"]
            ],
        ),
        "",
        "### Boundary titles",
        "",
        markdown_table(
            ["show_id", "title", "silhouette"],
            [
                [row["show_id"], row["title"], row["silhouette"]]
                for row in profile["boundary_titles"]
            ],
        ),
        "",
    ]
    return lines


def _evaluation_markdown(
    metadata: dict[str, Any],
    profiles: list[dict[str, Any]],
    algorithms: list[dict[str, Any]],
) -> str:
    scores = metadata["validation_scores"]
    external = metadata.get("external_validation") or {}
    lines = [
        "# Evaluation",
        "",
        "Generated by `scripts/train.py` and `scripts/evaluate.py`.",
        "",
        f"Model version `{metadata['model_version']}` on {metadata['row_count']} titles. "
        f"Representation `{metadata['representation']['name']}`, k={metadata['chosen_k']}.",
        "",
        metadata.get("k_reason", ""),
        "",
        "## Internal scores of the product model",
        "",
        markdown_table(
            ["silhouette", "CH", "DB", "inertia", "bootstrap_ARI"],
            [
                [
                    scores.get("silhouette"),
                    scores.get("calinski_harabasz"),
                    scores.get("davies_bouldin"),
                    scores.get("inertia"),
                    scores.get("bootstrap_ari"),
                ]
            ],
        ),
        "",
        "## External scores",
        "",
    ]
    if external:
        lines.append(
            markdown_table(
                ["attribute", "ARI", "NMI"],
                [
                    [name, external[name]["ari"], external[name]["nmi"]]
                    for name in ("type", "rating", "primary_genre")
                    if name in external
                ],
            )
        )
    else:
        lines.append("External scores were not in the k-selection artifact.")
    lines.extend(
        [
            "",
            "## Segments",
            "",
            markdown_table(
                ["id", "name", "size", "percent"],
                [
                    [profile["cluster_id"], profile["name"], profile["size"], profile["percentage"]]
                    for profile in profiles
                ],
            ),
            "",
            "## Algorithm noise fractions",
            "",
            markdown_table(
                ["algorithm", "noise_fraction", "n_clusters"],
                [
                    [row["algorithm"], row["noise_fraction"], row["n_clusters_found"]]
                    for row in algorithms
                ],
            ),
            "",
            "t-SNE and UMAP distances are not interpreted. See the caveat in `reports/cluster_profiles.md`.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
