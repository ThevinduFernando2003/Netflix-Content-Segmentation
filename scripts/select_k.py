"""Sweep k, bootstrap stability, and external validation.

Writes reports/k_selection.md and reports/stability.md. Re-run to refresh every number.
"""

from __future__ import annotations

import argparse
import os
import sys
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
from sklearn.metrics import adjusted_rand_score, silhouette_samples  # noqa: E402

from _common import figures_dir, markdown_table, prepare, reports_dir, write_json  # noqa: E402
from config import project_path, seed_from  # noqa: E402
from data.clean import clean_dataframe  # noqa: E402
from data.load import load_titles  # noqa: E402
from evaluation.external import association_scores, contingency_table  # noqa: E402
from evaluation.plots import (  # noqa: E402
    plot_elbow,
    plot_heatmap,
    plot_lines,
    plot_per_cluster_silhouette,
    plot_stability,
)
from features.build_pipeline import build_segmentation_model  # noqa: E402
from models.cluster import embed_frame, fit_kmeans  # noqa: E402
from models.select_k import (  # noqa: E402
    choose_k,
    gap_statistic,
    gmm_information_criteria,
    k_grid_scores,
    locate_elbow,
)
from models.stability import bootstrap_stability, seed_stability  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", default=None)
    parser.add_argument(
        "--from-sweep",
        action="store_true",
        help="Reuse reports/k_selection.json scores and recompute only the decision and chosen-k checks.",
    )
    args = parser.parse_args()
    config = prepare()
    reports = reports_dir(config)
    choice_path = reports / "representation_scores.json"
    representation = args.representation or config["product"]["representation"]
    if args.representation is None and choice_path.is_file():
        import json

        recommendation = json.loads(choice_path.read_text(encoding="utf-8"))
        representation = str(recommendation["choice"])
    seed = seed_from(config)
    frame = load_titles(project_path(config, "raw_csv"))
    k_cfg = config["k_selection"]
    k_values = list(range(int(k_cfg["k_min"]), int(k_cfg["k_max"]) + 1))
    matrix, engineered, _representer = embed_frame(frame, config, representation, seed=seed)
    sweep_path = reports / "k_selection.json"
    if args.from_sweep:
        if not sweep_path.is_file():
            raise FileNotFoundError(f"No sweep to reuse at {sweep_path}.")
        import json

        previous = json.loads(sweep_path.read_text(encoding="utf-8"))
        if str(previous.get("representation")) != representation:
            raise ValueError(
                f"Sweep representation is {previous.get('representation')}, not {representation}."
            )
        scores = list(previous["scores"])
        gmm_rows = list(previous["gmm"])
        gap_rows = list(previous["gap"])
        stability_rows = list(previous["stability"])
        print("reusing sweep scores from k_selection.json", flush=True)
    else:
        scores = k_grid_scores(matrix, k_values, seed)
        gmm_rows = gmm_information_criteria(
            matrix,
            k_values,
            seed,
            covariance_type=str(config.get("gmm", {}).get("covariance_type", "diag")),
        )
        gap_rows = gap_statistic(
            matrix,
            k_values,
            seed,
            n_references=int(k_cfg.get("gap_references", 5)),
        )
        stability_rows = []
        for k in k_values:
            print(f"bootstrap k={k}", flush=True)
            row = bootstrap_stability(
                frame,
                config,
                k=k,
                representation=representation,
                n_bootstrap=int(k_cfg["n_bootstrap"]),
                fraction=float(k_cfg["bootstrap_fraction"]),
                seed=seed,
                n_jobs=int(k_cfg.get("n_jobs", 1)),
            )
            stability_rows.append({key: value for key, value in row.items() if key != "pair_aris"})
    chosen, reason = choose_k(
        scores,
        stability_rows,
        interpretable_min=int(k_cfg.get("interpretable_min", 5)),
        interpretable_max=int(k_cfg.get("interpretable_max", 10)),
        stability_floor=float(k_cfg.get("stability_floor", 0.80)),
    )
    model = fit_kmeans(matrix, chosen, seed)
    silhouettes = silhouette_samples(matrix, model.labels_)
    type_labels = engineered["type"].astype(str).to_numpy()
    coarse_ks = sorted(
        {
            int(k)
            for k in (2, 3, 4, chosen)
            if 2 <= int(k) < len(matrix) and int(k) in {int(row["k"]) for row in scores}
        }
    )
    ari_vs_type = []
    for k_check in coarse_ks:
        labels_k = model.labels_ if k_check == chosen else fit_kmeans(matrix, k_check, seed).labels_
        ari_vs_type.append(
            {
                "k": int(k_check),
                "ari_vs_type": float(adjusted_rand_score(labels_k, type_labels)),
            }
        )
    seed_rows = seed_stability(matrix, chosen, [int(value) for value in k_cfg["seed_sweep"]])
    holdout = build_segmentation_model(
        config,
        representation=representation,
        k=chosen,
        seed=seed,
        drop_type=True,
        drop_rating=True,
        drop_primary_genre=True,
        fit_gmm=False,
        compute_silhouette=False,
    )
    holdout.fit(frame)
    held = clean_dataframe(frame, config)
    if len(held) != len(holdout.labels_):
        held = holdout.engineer_.transform(frame)
    external = {
        "type": association_scores(holdout.labels_, held["type"]),
        "rating": association_scores(holdout.labels_, held["rating"].astype("string")),
        "primary_genre": association_scores(
            holdout.labels_, held["primary_genre"].astype("string")
        ),
    }
    tables = {
        "type": contingency_table(holdout.labels_, held["type"]),
        "rating": contingency_table(holdout.labels_, held["rating"].astype("string")),
        "primary_genre": _top_genre_contingency(holdout.labels_, held["primary_genre"]),
    }
    figures = figures_dir(config)
    ks = [int(row["k"]) for row in scores]
    plot_elbow(
        figures / "elbow_product.png",
        ks,
        [float(row["inertia"]) for row in scores],
        locate_elbow(ks, [float(row["inertia"]) for row in scores]),
    )
    plot_lines(
        figures / "internal_indices.png",
        ks,
        {
            "silhouette": [float(row["silhouette"]) for row in scores],
            "davies_bouldin": [float(row["davies_bouldin"]) for row in scores],
        },
        "k",
        "score",
        "Silhouette and Davies-Bouldin",
    )
    plot_lines(
        figures / "calinski_harabasz.png",
        ks,
        {"calinski_harabasz": [float(row["calinski_harabasz"]) for row in scores]},
        "k",
        "Calinski-Harabasz",
        "Calinski-Harabasz",
    )
    plot_lines(
        figures / "gmm_bic_aic.png",
        [int(row["k"]) for row in gmm_rows],
        {
            "BIC": [float(row["bic"]) for row in gmm_rows],
            "AIC": [float(row["aic"]) for row in gmm_rows],
        },
        "k",
        "information criterion",
        "Diagonal Gaussian mixture BIC and AIC",
    )
    plot_stability(
        figures / "stability_ari.png",
        [int(row["k"]) for row in stability_rows],
        [float(row["mean_ari"]) for row in stability_rows],
        [float(row["std_ari"]) for row in stability_rows],
        [
            float(next(item["silhouette"] for item in scores if int(item["k"]) == int(row["k"])))
            for row in stability_rows
        ],
    )
    plot_per_cluster_silhouette(figures / "silhouette_clusters.png", model.labels_, silhouettes)
    for attribute, table in tables.items():
        # Keep heatmaps readable: ratings and types are small; genres may be wide.
        matrix_norm = np.asarray(table["normalised"], dtype=float)
        attributes = table["attributes"]
        if len(attributes) > 12:
            totals = matrix_norm.sum(axis=0)
            keep = np.argsort(totals)[::-1][:12]
            matrix_norm = matrix_norm[:, keep]
            attributes = [attributes[int(index)] for index in keep]
        plot_heatmap(
            figures / f"contingency_{attribute}.png",
            matrix_norm,
            attributes,
            [str(cluster) for cluster in table["clusters"]],
            f"Cluster x held-out {attribute} (row-normalised)",
            attribute,
            "cluster",
            "share of cluster",
        )
    payload = {
        "representation": representation,
        "chosen_k": chosen,
        "reason": reason,
        "scores": scores,
        "stability": stability_rows,
        "gmm": gmm_rows,
        "gap": gap_rows,
        "seed_stability": {key: value for key, value in seed_rows.items() if key != "pair_aris"},
        "external": external,
        "ari_vs_type": ari_vs_type,
        "mean_silhouette_chosen": float(np.mean(silhouettes)),
        "per_cluster_silhouette": {
            str(cluster): float(np.mean(silhouettes[model.labels_ == cluster]))
            for cluster in sorted(set(model.labels_.tolist()))
        },
    }
    write_json(reports / "k_selection.json", payload)
    (reports / "k_selection.md").write_text(_render(payload), encoding="utf-8")
    (reports / "stability.md").write_text(_render_stability(payload), encoding="utf-8")
    print(f"chosen k={chosen} representation={representation}", flush=True)


def _top_genre_contingency(labels: np.ndarray, genres: Any) -> dict[str, Any]:

    series = genres.astype("string").fillna("__MISSING__")
    top = series.value_counts().head(12).index
    collapsed = series.where(series.isin(top), other="other")
    return contingency_table(labels, collapsed)


def _render(payload: dict[str, Any]) -> str:
    knee = locate_elbow(
        [int(row["k"]) for row in payload["scores"]],
        [float(row["inertia"]) for row in payload["scores"]],
    )
    best_bic = min(payload["gmm"], key=lambda row: row["bic"])
    lines = [
        "# Choosing k",
        "",
        "Generated by `scripts/select_k.py`. Re-run it to refresh every number.",
        "",
        f"## Decision: k = {payload['chosen_k']} on `{payload['representation']}`",
        "",
        payload["reason"],
        "",
        "The elbow is a heuristic, not evidence. It is plotted and located with `KneeLocator`, "
        f"which returned k={knee}. Gaussian mixture BIC (diagonal covariance) is lowest at "
        f"k={int(best_bic['k'])} (BIC={float(best_bic['bic']):.1f}). "
        "Those model-selection curves are allowed to disagree with the stability choice; "
        "the report says so instead of averaging them into a false consensus.",
        "",
        "k=2, k=3 and k=4 are scored, then excluded from the ceiling. "
        "The table below is the adjusted Rand index between a spherical k-means fit "
        "on the product representation (which is allowed to see `type`) and the `type` column. "
        "A value near 1 means that k rediscovered movies versus series.",
        "",
        markdown_table(
            ["k", "ARI_vs_type"],
            [[row["k"], row["ari_vs_type"]] for row in payload.get("ari_vs_type", [])],
        ),
        "",
        f"Mean silhouette at the chosen k: **{payload['mean_silhouette_chosen']:.4f}**. "
        "Modest silhouette is expected once genres overlap. The per-cluster plot "
        "(`reports/figures/silhouette_clusters.png`) shows which segments are weak.",
        "",
        "## Bootstrap stability",
        "",
        "For each k, 20 resamples of 80% of the rows (without replacement) refit the whole "
        "pipeline. The table is the mean and sample standard deviation of the pairwise "
        "adjusted Rand index on overlapping rows. A k with mediocre silhouette and ARI near "
        "0.8 is preferred to a k with a prettier silhouette and ARI near 0.3.",
        "",
        markdown_table(
            [
                "k",
                "silhouette",
                "CH",
                "DB",
                "inertia",
                "bootstrap_ARI",
                "ARI_std",
                "BIC",
                "AIC",
                "gap",
            ],
            _joined_rows(payload),
        ),
        "",
        "Figures: `reports/figures/elbow_product.png`, `internal_indices.png`, "
        "`calinski_harabasz.png`, `gmm_bic_aic.png`, `stability_ari.png`, `silhouette_clusters.png`.",
        "",
        "## Per-cluster mean silhouette at the chosen k",
        "",
        markdown_table(
            ["cluster", "mean_silhouette"],
            [[cluster, value] for cluster, value in payload["per_cluster_silhouette"].items()],
        ),
        "",
        "## Random seeds at the chosen k",
        "",
        "Same representation, same k, different `random_state` values. "
        f"Mean pairwise ARI = **{payload['seed_stability']['mean_pairwise_ari']:.4f}** "
        f"(std {payload['seed_stability']['std_pairwise_ari']:.4f}).",
        "",
        "## External validation",
        "",
        "One dedicated fit withholds `type`, `rating`, and the primary genre. "
        "Secondary genres are still visible to R1/R2, so primary-genre agreement is partly "
        "co-occurrence. Type and rating are the cleaner held-out attributes. "
        "Heatmaps are row-normalised (each cluster sums to 1): "
        "`reports/figures/contingency_type.png`, `contingency_rating.png`, `contingency_primary_genre.png`.",
        "",
        markdown_table(
            ["attribute", "ARI", "NMI", "n"],
            [
                [
                    name,
                    payload["external"][name]["ari"],
                    payload["external"][name]["nmi"],
                    payload["external"][name]["n"],
                ]
                for name in ("type", "rating", "primary_genre")
            ],
        ),
        "",
        "## Gap statistic",
        "",
        "Gap compares log inertia with uniform draws in the observed bounding box, "
        "renormalised onto the sphere so the null matches spherical k-means. "
        "It is a supporting signal, not the rule that chose k.",
        "",
        markdown_table(
            ["k", "gap", "observed_inertia"],
            [[row["k"], row["gap"], row["observed_inertia"]] for row in payload["gap"]],
        ),
        "",
    ]
    return "\n".join(lines)


def _joined_rows(payload: dict[str, Any]) -> list[list[Any]]:
    gmm = {int(row["k"]): row for row in payload["gmm"]}
    gap = {int(row["k"]): row for row in payload["gap"]}
    stability = {int(row["k"]): row for row in payload["stability"]}
    rows = []
    for score in payload["scores"]:
        k = int(score["k"])
        rows.append(
            [
                k,
                score["silhouette"],
                score["calinski_harabasz"],
                score["davies_bouldin"],
                score["inertia"],
                stability[k]["mean_ari"],
                stability[k]["std_ari"],
                gmm[k]["bic"],
                gmm[k]["aic"],
                gap[k]["gap"],
            ]
        )
    return rows


def _render_stability(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stability",
            "",
            "Generated by `scripts/select_k.py`.",
            "",
            f"Representation `{payload['representation']}`, chosen k={payload['chosen_k']}.",
            "",
            payload["reason"],
            "",
            markdown_table(
                ["k", "mean_ARI", "std_ARI", "n_bootstrap", "n_pairs"],
                [
                    [row["k"], row["mean_ari"], row["std_ari"], row["n_bootstrap"], row["n_pairs"]]
                    for row in payload["stability"]
                ],
            ),
            "",
            "The curve is `reports/figures/stability_ari.png`.",
            "",
            "## Seed sweep",
            "",
            markdown_table(
                ["mean_pairwise_ARI", "std", "mean_ARI_vs_first_seed"],
                [
                    [
                        payload["seed_stability"]["mean_pairwise_ari"],
                        payload["seed_stability"]["std_pairwise_ari"],
                        payload["seed_stability"]["mean_ari_vs_first_seed"],
                    ]
                ],
            ),
            "",
        ]
    )


if __name__ == "__main__":
    main()
