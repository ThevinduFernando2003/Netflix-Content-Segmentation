# Demo script

Narration for a short walkthrough. Numbers are the ones in `reports/k_selection.md`, `reports/representation_design.md`, and `reports/cluster_profiles.md`. Do not read a number that is not in those files.

## Problem

We have about 8,790 Netflix titles and no labelled segments. The job is to group the catalogue into segments a person can name, and to assign a new title without refitting.

## Mixed data

The hard part is the representation, not the choice of k-means. A 20,000-column TF-IDF block would drown a handful of numeric columns, and this file does not even have a synopsis. Duration is minutes for films and seasons for series. Median-imputing both would invent a season count for every film. The model uses one within-type duration percentile instead. Blocks are scaled to the same total variance and then weighted. Rows are L2-normalised, so Euclidean k-means is spherical k-means and tracks cosine distance.

## How k was chosen

Sweep k from 2 to 20. k=2 has bootstrap ARI 0.9996 and agrees with movie versus series (ARI 0.9953 against `type`). That is not a content segmentation, so it does not set the ceiling. k=6 is the finest value between 5 and 10 whose mean bootstrap ARI stays at least 0.80 (0.8203 ± 0.1211). The elbow says 9. BIC never bottoms out inside the sweep. Say that disagreement out loud.

## Named segments

Six segments: international drama films (the largest), international series, US family and action films, US kids' series and docuseries, US documentaries and stand-up, US independent dramas. Point at clusters 0 and 4 and say they are mixed: the name is the highest-lift genre, and it does not cover most of the cluster.

## Live assignment

Submit the family-film example. It should land in the US family and action segment, with a membership bar from the Gaussian mixture. Then submit the underwater tax-law example. The outlier flag should be on: k-means always returns a cluster, and the flag says this title is farther from the centroid than 95% of the titles that formed it. Unknown genres come back in the response.

## External validation

A second fit never sees type, rating, or the primary genre. It still agrees with type (ARI 0.2632, NMI 0.4271) more than with rating (ARI 0.0665, NMI 0.1165). Show the heatmap. Say this is association, not accuracy.

## Architecture

One saved pipeline. The API loads it at startup. The UI never imports it. Projections were computed once. t-SNE and UMAP gaps are not distances.
