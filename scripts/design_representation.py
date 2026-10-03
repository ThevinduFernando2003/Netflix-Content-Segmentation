"""Compare representations and write reports/representation_design.md.

All scores are computed in this script. Re-run it to refresh the report.
"""

from __future__ import annotations

import os
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
from sklearn.cluster import AgglomerativeClustering  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402

from _common import figures_dir, markdown_table, prepare, reports_dir, write_json  # noqa: E402
from config import project_path, seed_from  # noqa: E402
from data.clean import clean_dataframe  # noqa: E402
from data.load import load_titles  # noqa: E402
from evaluation.external import association_scores  # noqa: E402
from evaluation.plots import plot_elbow, plot_explained_variance  # noqa: E402
from features.build_pipeline import build_segmentation_model  # noqa: E402
from features.engineer import FeatureEngineer  # noqa: E402
from features.gower import gower_matrix  # noqa: E402
from models.cluster import embed_frame  # noqa: E402
from models.select_k import k_grid_scores, locate_elbow  # noqa: E402
from models.stability import bootstrap_stability  # noqa: E402


def _candidate_k(scores: list[dict[str, Any]]) -> tuple[int, int | None]:
    ks = [int(row["k"]) for row in scores]
    inertias = [float(row["inertia"]) for row in scores]
    knee = locate_elbow(ks, inertias)
    if knee is not None:
        return knee, knee
    best = max(scores, key=lambda row: row["silhouette"])
    return int(best["k"]), None


def _external(
    frame: Any, config: dict[str, Any], representation: str, k: int, seed: int
) -> dict[str, Any]:
    model = build_segmentation_model(
        config,
        representation=representation,
        k=k,
        seed=seed,
        drop_type=True,
        drop_rating=True,
        drop_primary_genre=True,
        fit_gmm=False,
        compute_silhouette=False,
    )
    model.fit(frame)
    engineered = clean_dataframe(frame, config)
    # align to the fitted row order (clean drops duplicate show_ids; the source has none)
    if len(engineered) != len(model.labels_):
        engineered = model.engineer_.transform(frame)
    return {
        "type": association_scores(model.labels_, engineered["type"]),
        "rating": association_scores(model.labels_, engineered["rating"].astype("string")),
        "primary_genre": association_scores(
            model.labels_, engineered["primary_genre"].astype("string")
        ),
    }


def _score_at(scores: list[dict[str, Any]], k: int) -> dict[str, Any]:
    for row in scores:
        if int(row["k"]) == int(k):
            return row
    raise KeyError(k)


def main() -> None:
    config = prepare()
    seed = seed_from(config)
    frame = load_titles(project_path(config, "raw_csv"))
    k_cfg = config["k_selection"]
    k_values = list(range(int(k_cfg["k_min"]), int(k_cfg["k_max"]) + 1))
    n_boot = int(k_cfg.get("comparison_bootstrap", 20))
    n_jobs = int(k_cfg.get("n_jobs", 1))
    figures = figures_dir(config)
    results: dict[str, Any] = {}

    for name in ("r1", "r2", "r3"):
        started = time.perf_counter()
        matrix, _engineered, representer = embed_frame(frame, config, name, seed=seed)
        scores = k_grid_scores(matrix, k_values, seed)
        candidate, knee = _candidate_k(scores)
        stability = bootstrap_stability(
            frame,
            config,
            k=candidate,
            representation=name,
            n_bootstrap=n_boot,
            fraction=float(k_cfg["bootstrap_fraction"]),
            seed=seed,
            n_jobs=n_jobs,
        )
        external = _external(frame, config, name, candidate, seed)
        ratio = representer.explained_variance_ratio()
        results[name] = {
            "matrix_shape": list(matrix.shape),
            "scores": scores,
            "candidate_k": candidate,
            "knee": knee,
            "stability": {key: value for key, value in stability.items() if key != "pair_aris"},
            "external": external,
            "seconds": time.perf_counter() - started,
            "cumulative_explained_variance": None if ratio is None else float(np.sum(ratio)),
            "svd_components": None if ratio is None else int(len(ratio)),
        }
        plot_elbow(
            figures / f"elbow_{name}.png",
            [int(row["k"]) for row in scores],
            [float(row["inertia"]) for row in scores],
            knee,
        )
        if ratio is not None and name == "r2":
            plot_explained_variance(figures / "explained_variance_svd.png", ratio)
            results[name]["explained_variance_ratio_head"] = [float(value) for value in ratio[:10]]
        print(
            f"{name} shape={matrix.shape} knee={knee} candidate={candidate} "
            f"ARI={stability['mean_ari']:.3f} in {results[name]['seconds']:.0f}s",
            flush=True,
        )

    # Shared-k snapshot so the three representations are not compared only at different knees.
    shared_k = int(np.median([results[name]["candidate_k"] for name in results]))
    for name in results:
        results[name]["at_shared_k"] = _score_at(results[name]["scores"], shared_k)
        results[name]["at_candidate"] = _score_at(
            results[name]["scores"], results[name]["candidate_k"]
        )

    sensitivity = []
    sens_boot = int(k_cfg.get("sensitivity_bootstrap", 10))
    for weights in config["weight_sensitivity"]:
        weight_map = {key: float(value) for key, value in weights.items()}
        matrix, _, _ = embed_frame(frame, config, "r2", weights=weight_map, seed=seed)
        k_for_weights = int(results["r2"]["candidate_k"])
        row = _score_at(k_grid_scores(matrix, [k_for_weights], seed), k_for_weights)
        stability = bootstrap_stability(
            frame,
            config,
            k=k_for_weights,
            representation="r2",
            n_bootstrap=sens_boot,
            fraction=float(k_cfg["bootstrap_fraction"]),
            seed=seed,
            weights=weight_map,
            n_jobs=n_jobs,
        )
        sensitivity.append(
            {
                "weights": weight_map,
                "k": k_for_weights,
                "silhouette": row["silhouette"],
                "calinski_harabasz": row["calinski_harabasz"],
                "davies_bouldin": row["davies_bouldin"],
                "bootstrap_ari": stability["mean_ari"],
                "bootstrap_ari_std": stability["std_ari"],
                "n_bootstrap": sens_boot,
            }
        )
        print(f"weights {weight_map} ARI={stability['mean_ari']:.3f}", flush=True)

    try:
        gower = _gower_comparison(frame, config, shared_k, seed)
    except Exception as exc:  # comparison only; do not discard the representation scores
        gower = {
            "subsample": 0,
            "k": shared_k,
            "silhouette": float("nan"),
            "full_catalogue_float64_gb": (len(frame) ** 2) * 8 / 1e9,
            "note": f"Gower comparison failed: {exc}",
        }
        print(f"Gower failed: {exc}", flush=True)
    try:
        prototypes = _kprototypes_comparison(frame, config, shared_k, seed)
    except Exception as exc:
        prototypes = {"available": False, "reason": str(exc)}
        print(f"KPrototypes failed: {exc}", flush=True)
    choice, reason = _choose(results)
    payload = {
        "choice": choice,
        "reason": reason,
        "shared_k": shared_k,
        "results": results,
        "weight_sensitivity": sensitivity,
        "gower": gower,
        "kprototypes": prototypes,
    }
    write_json(reports_dir(config) / "representation_scores.json", payload)
    report = _render(config, payload)
    target = reports_dir(config) / "representation_design.md"
    target.write_text(report, encoding="utf-8")
    print(f"Wrote {target} choice={choice}", flush=True)


def _choose(results: dict[str, Any]) -> tuple[str, str]:
    ranking = sorted(results, key=lambda name: results[name]["stability"]["mean_ari"], reverse=True)
    best = ranking[0]
    best_ari = float(results[best]["stability"]["mean_ari"])
    r1_ari = float(results["r1"]["stability"]["mean_ari"])
    # Titles are the only text available. Prefer the simpler genre representation
    # unless text improves stability by a clear margin.
    if best != "r1" and (best_ari - r1_ari) < 0.05:
        chosen = "r1"
        reason = (
            f"{best} has the highest bootstrap ARI ({best_ari:.3f}) but leads R1 "
            f"({r1_ari:.3f}) by under 0.05. The catalogue has no synopsis, so the text "
            "block is titles only. R1 is the product: it is simpler, interpretable, and "
            "statistically tied on the metric that decides whether clusters exist."
        )
    elif best == "r1":
        chosen = "r1"
        reason = (
            f"R1 has the highest bootstrap ARI ({r1_ari:.3f}) at its candidate k. "
            "Genre, type, country, rating and the duration percentile recover stable "
            "segments without a title TF-IDF block that the missing synopses cannot support."
        )
    else:
        chosen = best
        reason = (
            f"{best} leads on bootstrap ARI ({best_ari:.3f} versus R1 {r1_ari:.3f}, "
            "a gap of at least 0.05). The SVD title block earns its complexity on the "
            "stability criterion, so it is the product representation."
        )
    return chosen, reason


def _gower_comparison(frame: Any, config: dict[str, Any], k: int, seed: int) -> dict[str, Any]:
    cleaned = FeatureEngineer(config).fit_transform(frame)
    rng = np.random.default_rng(seed)
    subsample = int(config.get("gower", {}).get("subsample", 3000))
    subsample = min(subsample, len(cleaned))
    chosen = np.sort(rng.choice(len(cleaned), size=subsample, replace=False))
    sample = cleaned.iloc[chosen].reset_index(drop=True)
    numeric_cols = [
        "release_year",
        "year_added",
        "years_to_platform",
        "n_genres",
        "n_countries",
        "duration_normalised",
        "director_missing",
    ]
    numeric = sample[numeric_cols].apply(pd_to_numeric_fill, axis=0).to_numpy(dtype=np.float32)
    # Column-wise median fill without importing the whole imputer path twice.
    for column in range(numeric.shape[1]):
        values = numeric[:, column]
        median = np.nanmedian(values)
        values[~np.isfinite(values)] = median
        numeric[:, column] = values
    categorical = np.column_stack(
        [
            sample["type"].astype("string").fillna("__MISSING__").astype(str).to_numpy(),
            sample["primary_country"].astype("string").fillna("__MISSING__").astype(str).to_numpy(),
            sample["rating"].astype("string").fillna("__MISSING__").astype(str).to_numpy(),
            sample["primary_genre"].astype("string").fillna("__MISSING__").astype(str).to_numpy(),
        ]
    )
    distances = gower_matrix(
        numeric, categorical, block_size=int(config.get("gower", {}).get("block_size", 400))
    )
    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    labels = model.fit_predict(distances)
    return {
        "subsample": subsample,
        "k": k,
        "note": (
            "Gower distances were computed in float32 blocks on a subsample. "
            "This comparison is not the product model and cannot assign the held-out rows."
        ),
        "silhouette": float(silhouette_score(distances, labels, metric="precomputed")),
        "calinski_harabasz": float("nan"),
        "davies_bouldin": float("nan"),
        "full_catalogue_float64_gb": (len(frame) ** 2) * 8 / 1e9,
    }


def pd_to_numeric_fill(column: Any) -> Any:
    import pandas as pd

    return pd.to_numeric(column, errors="coerce")


def _kprototypes_comparison(
    frame: Any, config: dict[str, Any], k: int, seed: int
) -> dict[str, Any]:
    try:
        from kmodes.kprototypes import KPrototypes
    except ImportError as exc:
        return {"available": False, "reason": str(exc)}
    import pandas as pd

    cleaned = FeatureEngineer(config).fit_transform(frame)
    rng = np.random.default_rng(seed)
    subsample = min(2000, len(cleaned))
    chosen = np.sort(rng.choice(len(cleaned), size=subsample, replace=False))
    sample = cleaned.iloc[chosen]
    numeric_cols = ["release_year", "n_genres", "duration_normalised", "director_missing"]
    numbers = sample[numeric_cols].apply(lambda column: pd.to_numeric(column, errors="coerce"))
    numbers = numbers.fillna(numbers.median(numeric_only=True))
    categories = (
        sample[["type", "primary_country", "rating", "primary_genre"]]
        .astype("string")
        .fillna("__MISSING__")
    )
    matrix = np.empty((len(sample), len(numeric_cols) + categories.shape[1]), dtype=object)
    matrix[:, : len(numeric_cols)] = numbers.to_numpy(dtype=float)
    matrix[:, len(numeric_cols) :] = categories.to_numpy()
    categorical_idx = list(range(len(numeric_cols), matrix.shape[1]))
    model = KPrototypes(n_clusters=k, init="Huang", random_state=seed, n_init=2, max_iter=20)
    labels = model.fit_predict(matrix, categorical=categorical_idx)
    # Internal indices need a numeric space. Use the scaled numeric columns only as a rough view,
    # and also report ARI against primary genre as a descriptive association, not a fitness score.
    genre_codes = pd.factorize(categories["primary_genre"])[0]
    return {
        "available": True,
        "subsample": subsample,
        "k": k,
        "cost": float(model.cost_),
        "ari_vs_primary_genre": float(adjusted_rand_score(genre_codes, labels)),
        "note": (
            "KPrototypes on a 2,000-row subsample of type, country, rating, primary genre "
            "and a few numerics. It is a mixed-type alternative to one-hot Euclidean distance, "
            "not the product model."
        ),
    }


def _render(config: dict[str, Any], payload: dict[str, Any]) -> str:
    choice = payload["choice"]
    spec = config["representations"][choice]
    lines = [
        "# Representation design",
        "",
        "Generated by `scripts/design_representation.py`. Every score below comes from that run.",
        "",
        "## Product representation",
        "",
        f"**{choice} — {spec['name']}.**",
        "",
        payload["reason"],
        "",
        "The deciding metric is bootstrap adjusted Rand index, not silhouette. "
        "A compact cluster with unstable labels is noise. Silhouette on this catalogue "
        "is expected to sit roughly between 0.05 and 0.25; that is normal for overlapping "
        "genres and is not treated as a failure.",
        "",
        "## What the file actually contains",
        "",
        "There is no `description` and no `cast`. The text block is the title, L2-normalised "
        "TF-IDF (1–2 grams, `min_df=5`), reduced with TruncatedSVD (not PCA) and L2-normalised "
        "again so Euclidean k-means is spherical k-means. See `reports/data_profile.md`.",
        "",
        "## Duration",
        "",
        "The numeric block uses `duration_normalised`: the training-set percentile of minutes "
        "among movies, or of seasons among series. Median-imputing both `duration_minutes` and "
        "`duration_seasons` would hand every movie the median season count and every series the "
        "median runtime, and k-means would split on that artefact. Clustering the two types "
        "separately was the alternative; it was rejected for the product model because the "
        "catalogue is one segmentation problem and `type` is an observed attribute, not a leak.",
        "",
        "## Block dominance and one-hot geometry",
        "",
        "Blocks are scaled to unit total variance, multiplied by `sqrt(weight)`, concatenated, "
        "then L2-normalised per row. Default weights are "
        f"`{config['block_weights']}`. A raw `hstack` of 20,000 TF-IDF columns with eight "
        "numeric columns is not used.",
        "",
        "One-hot categories sit at a Euclidean distance of sqrt(2) from each other, so India is "
        "exactly as far from Japan as from the United States. That geometry is arbitrary. "
        "The categorical block is therefore down-weighted, and Gower distance plus KPrototypes "
        "are reported as mixed-type alternatives. They are comparisons, not the serving model: "
        "Gower was computed on a subsample because the full float64 matrix would be about "
        f"{payload['gower'].get('full_catalogue_float64_gb', float('nan')):.2f} GB.",
        "",
        "## Internal scores at each representation's candidate k",
        "",
        "Candidate k is the `KneeLocator` knee of inertia when it exists, otherwise the "
        "silhouette maximum. The knee is a heuristic. The full k=2..20 stability curve for "
        "the product representation is in `reports/k_selection.md`.",
        "",
        markdown_table(
            [
                "rep",
                "shape",
                "candidate_k",
                "knee",
                "silhouette",
                "CH",
                "DB",
                "bootstrap_ARI",
                "ARI_std",
                "cum_SVD_var",
            ],
            [
                [
                    name,
                    "x".join(str(part) for part in payload["results"][name]["matrix_shape"]),
                    payload["results"][name]["candidate_k"],
                    payload["results"][name]["knee"],
                    payload["results"][name]["at_candidate"]["silhouette"],
                    payload["results"][name]["at_candidate"]["calinski_harabasz"],
                    payload["results"][name]["at_candidate"]["davies_bouldin"],
                    payload["results"][name]["stability"]["mean_ari"],
                    payload["results"][name]["stability"]["std_ari"],
                    payload["results"][name]["cumulative_explained_variance"],
                ]
                for name in ("r1", "r2", "r3")
            ],
        ),
        "",
        f"Bootstrap: {config['k_selection']['comparison_bootstrap']} resamples, "
        f"{config['k_selection']['bootstrap_fraction']:.0%} of rows without replacement, "
        "whole pipeline refit, mean pairwise ARI on the overlap.",
        "",
        f"## Same k = {payload['shared_k']} for a like-for-like internal comparison",
        "",
        markdown_table(
            ["rep", "silhouette", "CH", "DB", "inertia"],
            [
                [
                    name,
                    payload["results"][name]["at_shared_k"]["silhouette"],
                    payload["results"][name]["at_shared_k"]["calinski_harabasz"],
                    payload["results"][name]["at_shared_k"]["davies_bouldin"],
                    payload["results"][name]["at_shared_k"]["inertia"],
                ]
                for name in ("r1", "r2", "r3")
            ],
        ),
        "",
        "## External association at the candidate k",
        "",
        "This is a separate fit that never sees `type`, `rating`, or the primary genre. "
        "Secondary genres remain in R1 and R2, so agreement with primary genre is partly "
        "co-occurrence, not an independent discovery. Agreement with `type` and `rating` "
        "is the cleaner check.",
        "",
        markdown_table(
            [
                "rep",
                "ARI_type",
                "NMI_type",
                "ARI_rating",
                "NMI_rating",
                "ARI_primary_genre",
                "NMI_primary_genre",
            ],
            [
                [
                    name,
                    payload["results"][name]["external"]["type"]["ari"],
                    payload["results"][name]["external"]["type"]["nmi"],
                    payload["results"][name]["external"]["rating"]["ari"],
                    payload["results"][name]["external"]["rating"]["nmi"],
                    payload["results"][name]["external"]["primary_genre"]["ari"],
                    payload["results"][name]["external"]["primary_genre"]["nmi"],
                ]
                for name in ("r1", "r2", "r3")
            ],
        ),
        "",
        "## SVD explained variance",
        "",
    ]
    r2 = payload["results"]["r2"]
    if r2["cumulative_explained_variance"] is None:
        lines.append("R2 did not fit an SVD.")
    else:
        lines.append(
            f"TruncatedSVD kept **{r2['svd_components']}** components and explained "
            f"**{r2['cumulative_explained_variance']:.3f}** of TF-IDF variance. "
            "150 is the configured width; the component count is capped by the rank of the "
            "title matrix when the vocabulary is smaller. The curve is "
            "`reports/figures/explained_variance_svd.png`. Components are L2-normalised "
            "after the SVD, which is the usual LSA step so Euclidean distance tracks cosine."
        )
    lines.extend(
        [
            "",
            "## Weight sensitivity on R2",
            "",
            "Each block is scaled to unit total variance before weighting, so these ratios "
            "are the intended variance contributions before the final row L2 normalisation.",
            "",
            markdown_table(
                ["weights", "k", "silhouette", "CH", "DB", "bootstrap_ARI", "n_bootstrap"],
                [
                    [
                        row["weights"],
                        row["k"],
                        row["silhouette"],
                        row["calinski_harabasz"],
                        row["davies_bouldin"],
                        row["bootstrap_ari"],
                        row["n_bootstrap"],
                    ]
                    for row in payload["weight_sensitivity"]
                ],
            ),
            "",
            "## Gower subsample",
            "",
            payload["gower"]["note"],
            "",
            markdown_table(
                ["subsample", "k", "silhouette_precomputed"],
                [
                    [
                        payload["gower"]["subsample"],
                        payload["gower"]["k"],
                        payload["gower"]["silhouette"],
                    ]
                ],
            ),
            "",
            "Calinski-Harabasz and Davies-Bouldin are not reported for Gower: both assume "
            "Euclidean geometry, and scoring them on a precomputed distance matrix would be misleading.",
            "",
            "## KPrototypes subsample",
            "",
        ]
    )
    prototypes = payload["kprototypes"]
    if not prototypes.get("available", False):
        lines.append(f"KPrototypes was not run: {prototypes.get('reason', 'unavailable')}.")
    else:
        lines.append(prototypes["note"])
        lines.extend(
            [
                "",
                markdown_table(
                    ["subsample", "k", "cost", "ARI_vs_primary_genre"],
                    [
                        [
                            prototypes["subsample"],
                            prototypes["k"],
                            prototypes["cost"],
                            prototypes["ari_vs_primary_genre"],
                        ]
                    ],
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Silhouette by k",
            "",
        ]
    )
    for name in ("r1", "r2", "r3"):
        lines.append(f"### {name}")
        lines.append("")
        lines.append(
            markdown_table(
                ["k", "silhouette", "CH", "DB", "inertia"],
                [
                    [
                        row["k"],
                        row["silhouette"],
                        row["calinski_harabasz"],
                        row["davies_bouldin"],
                        row["inertia"],
                    ]
                    for row in payload["results"][name]["scores"]
                ],
            )
        )
        lines.append("")
    lines.append("Elbow plots: `reports/figures/elbow_r1.png`, `elbow_r2.png`, `elbow_r3.png`.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
