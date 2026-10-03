"""Segment explorer. Talks to the API over HTTP and does not import the model."""

from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
CAVEAT = (
    "Projection distances are not interpretable. t-SNE and UMAP do not preserve global "
    "distances or cluster sizes, so gaps and blob areas are not structure. "
    "Titles were clustered in feature space; this map only colours that result."
)

EXAMPLES = {
    "Family film": {
        "title": "The Paper Kite Club",
        "description": "Two children build kites, chase a lost dog, and learn what friendship costs when the summer ends.",
        "type": "Movie",
        "genres": "Children & Family Movies, Comedies",
        "release_year": 2019,
        "country": "United States",
        "rating": "PG",
        "duration_minutes": 98,
        "duration_seasons": 0,
    },
    "Crime drama": {
        "title": "Night Docket",
        "description": "A detective and a prosecutor unravel a murder that reaches into a city's political machine and its oldest families.",
        "type": "Movie",
        "genres": "Dramas, Thrillers, Crime TV Shows",
        "release_year": 2018,
        "country": "United States",
        "rating": "TV-MA",
        "duration_minutes": 124,
        "duration_seasons": 0,
    },
    "Deliberate outlier": {
        "title": "Estonian Tax Law, Underwater",
        "description": "A silent experimental lecture on bracket creep, filmed inside a drained reservoir, with no characters, no story, and a glossary of statutes.",
        "type": "Movie",
        "genres": "Not A Real Genre, Underwater Basket Weaving",
        "release_year": 1900,
        "country": "Atlantis",
        "rating": "UNRATED-XYZ",
        "duration_minutes": 4,
        "duration_seasons": 0,
    },
}


def _client() -> httpx.Client:
    return httpx.Client(base_url=API_BASE_URL, timeout=60.0)


def _get(path: str, params: dict | None = None) -> dict | list:
    try:
        with _client() as client:
            response = client.get(path, params=params)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        st.error(f"The API at {API_BASE_URL} did not respond successfully. {exc}")
        st.stop()


def _post(path: str, payload: dict) -> dict:
    try:
        with _client() as client:
            response = client.post(path, json=payload)
            if response.status_code == 422:
                st.error(response.json())
                st.stop()
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        st.error(f"The API at {API_BASE_URL} did not respond successfully. {exc}")
        st.stop()


@st.cache_data(ttl=300)
def load_meta() -> dict:
    return _get("/meta")


@st.cache_data(ttl=300)
def load_clusters() -> list[dict]:
    return _get("/clusters")["clusters"]


@st.cache_data(ttl=300)
def load_projection(sample: int) -> dict:
    return _get("/projection", params={"sample": sample})


@st.cache_data(ttl=300)
def load_figure(name: str) -> bytes | None:
    try:
        with _client() as client:
            image = client.get(f"/figures/{name}")
        if image.status_code != 200:
            return None
        return image.content
    except httpx.HTTPError:
        return None


def color_map(meta: dict, clusters: list[dict]) -> dict[str, str]:
    mapping = {str(cluster["cluster_id"]): cluster["color"] for cluster in clusters}
    mapping.update({str(key): value for key, value in (meta.get("cluster_colors") or {}).items()})
    return mapping


def cluster_label(cluster: dict) -> str:
    return f"{cluster['cluster_id']}: {cluster['name']}"


st.set_page_config(page_title="Netflix content segments", layout="wide")
st.title("Netflix content segments")
st.caption(f"API: {API_BASE_URL}")

meta = load_meta()
clusters = load_clusters()
colors = color_map(meta, clusters)
name_by_id = {int(cluster["cluster_id"]): cluster["name"] for cluster in clusters}

tab_map, tab_explore, tab_assign, tab_compare, tab_k, tab_about = st.tabs(
    [
        "Segment map",
        "Segment explorer",
        "Assign a new title",
        "Compare segments",
        "How many segments?",
        "About",
    ]
)

with tab_map:
    st.info(CAVEAT)
    left, right, sample_col = st.columns(3)
    method = left.selectbox("Projection", ["UMAP", "t-SNE", "PCA"])
    colour = right.selectbox("Colour by", ["cluster", "type", "rating", "decade"])
    sample = sample_col.slider("Sample size", min_value=200, max_value=4000, value=1500, step=100)
    payload = load_projection(sample)
    st.caption(payload["caveat"])
    frame = pd.DataFrame(payload["points"])
    x_col = {"UMAP": "umap_x", "t-SNE": "tsne_x", "PCA": "pca_x"}[method]
    y_col = {"UMAP": "umap_y", "t-SNE": "tsne_y", "PCA": "pca_y"}[method]
    frame["segment"] = frame["cluster"].map(
        lambda value: f"{int(value)}: {name_by_id.get(int(value), value)}"
    )
    colour_col = "segment" if colour == "cluster" else colour
    color_arg = colors if colour == "cluster" else None
    if colour == "cluster":
        figure = px.scatter(
            frame,
            x=x_col,
            y=y_col,
            color="segment",
            color_discrete_map={
                f"{int(key)}: {name_by_id.get(int(key), key)}": value
                for key, value in colors.items()
            },
            hover_data=["title", "type", "rating", "decade", "show_id"],
            labels={x_col: f"{method} 1", y_col: f"{method} 2", "segment": "Segment"},
        )
    else:
        figure = px.scatter(
            frame,
            x=x_col,
            y=y_col,
            color=colour_col,
            hover_data=["title", "segment", "show_id"],
            labels={x_col: f"{method} 1", y_col: f"{method} 2"},
        )
    figure.update_layout(legend_title_text=colour, height=640)
    st.plotly_chart(figure, use_container_width=True)
    del color_arg

with tab_explore:
    choice = st.selectbox("Segment", clusters, format_func=cluster_label)
    detail = _get(f"/clusters/{choice['cluster_id']}")
    st.subheader(detail["name"])
    st.write(
        f"{detail['size']} titles ({detail['percentage']:.1f}% of the catalogue). "
        f"{'Interpretable from genre lift.' if detail['interpretable'] else 'Not interpretable enough to name confidently.'}"
    )
    st.write(detail["name_evidence"])
    lift = detail["genre_lift"]
    if lift:
        lift_fig = px.bar(
            lift,
            x="lift",
            y="genre",
            orientation="h",
            title="Genre lift (cluster rate / catalogue rate)",
            labels={"lift": "Lift", "genre": "Genre"},
        )
        lift_fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380)
        st.plotly_chart(lift_fig, use_container_width=True)
    st.markdown("**c-TF-IDF terms:** " + ", ".join(row["term"] for row in detail["top_terms"][:15]))
    numbers = detail["numeric_profile"]
    if numbers:
        num_fig = px.bar(
            numbers,
            x="standardised_difference",
            y="feature",
            orientation="h",
            title="Standardised difference from the catalogue mean",
            labels={"standardised_difference": "z", "feature": "Feature"},
            color_discrete_sequence=[colors.get(str(detail["cluster_id"]), "#1b9e77")],
        )
        num_fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
        st.plotly_chart(num_fig, use_container_width=True)
    type_counts = detail["composition"]["type"]
    type_fig = px.bar(
        x=list(type_counts.keys()),
        y=list(type_counts.values()),
        labels={"x": "Type", "y": "Titles"},
        title="Type",
        color_discrete_sequence=[colors.get(str(detail["cluster_id"]), "#1b9e77")],
    )
    st.plotly_chart(type_fig, use_container_width=True)
    left, right = st.columns(2)
    left.markdown("**Closest to the centroid**")
    for row in detail["representative_titles"]:
        left.write(
            f"{row['title']} ({row['release_year']}) — distance {row['distance_to_centroid']:.3f}"
        )
    right.markdown("**Boundary titles (lowest silhouette)**")
    for row in detail["boundary_titles"]:
        silhouette = row["silhouette"]
        right.write(
            f"{row['title']} — silhouette {silhouette if silhouette is not None else 'n/a'}"
        )

with tab_assign:
    st.write(
        "The model is not refit. A point farther than the cluster's 95th percentile training distance is flagged as an outlier."
    )
    preset = st.selectbox("Example", ["(blank)"] + list(EXAMPLES))
    seed_values = EXAMPLES.get(preset, {})
    with st.form("assign"):
        title = st.text_input("Title", value=seed_values.get("title", ""))
        description = st.text_area(
            "Description", value=seed_values.get("description", ""), height=120
        )
        kind = st.selectbox(
            "Type",
            ["Movie", "TV Show"],
            index=0 if seed_values.get("type", "Movie") == "Movie" else 1,
        )
        genres = st.text_input("Genres (comma separated)", value=seed_values.get("genres", ""))
        year = st.number_input(
            "Release year",
            min_value=1900,
            max_value=2100,
            value=int(seed_values.get("release_year", 2018)),
        )
        country = st.text_input("Country", value=seed_values.get("country", ""))
        rating = st.text_input("Rating", value=seed_values.get("rating", ""))
        minutes = st.number_input(
            "Duration minutes (0 if a series)",
            min_value=0,
            max_value=500,
            value=int(seed_values.get("duration_minutes", 100)),
        )
        seasons = st.number_input(
            "Duration seasons (0 if a movie)",
            min_value=0,
            max_value=50,
            value=int(seed_values.get("duration_seasons", 0)),
        )
        submitted = st.form_submit_button("Assign")
    if submitted:
        body = {
            "title": title,
            "description": description,
            "type": kind,
            "genres": [item.strip() for item in genres.split(",") if item.strip()],
            "release_year": int(year),
            "country": country or None,
            "rating": rating or None,
            "duration_minutes": int(minutes) or None,
            "duration_seasons": int(seasons) or None,
        }
        with st.spinner("Assigning"):
            result = _post("/assign", body)
        if result["is_outlier"]:
            st.error(
                f"Outlier. Assigned to {result['cluster_id']}: {result['cluster_name']} "
                f"but the distance {result['distance_to_centroid']:.3f} is beyond that segment's 95th percentile. "
                "K-means still names a nearest segment; that does not mean the title fits it."
            )
        else:
            st.success(
                f"Segment {result['cluster_id']}: {result['cluster_name']} "
                f"(distance {result['distance_to_centroid']:.3f})"
            )
        st.write("Why")
        for reason in result["why"]:
            st.write(f"- {reason}")
        if result["unknown_genres"]:
            st.warning("Ignored unknown genres: " + ", ".join(result["unknown_genres"]))
        if result["membership_probabilities"]:
            prob_fig = px.bar(
                result["membership_probabilities"],
                x="probability",
                y="cluster_name",
                orientation="h",
                title="Gaussian mixture membership",
                labels={"probability": "Probability", "cluster_name": "Segment"},
            )
            st.plotly_chart(prob_fig, use_container_width=True)
        st.write("Nearest titles in the assigned segment")
        for row in result["nearest_titles"]:
            st.write(f"{row['title']} ({row['show_id']}) — distance {row['distance']:.3f}")
        st.caption(f"Model {result['model_version']} in {result['took_ms']:.0f} ms")

with tab_compare:
    picked = st.multiselect(
        "Segments", clusters, default=clusters[:2], format_func=cluster_label, max_selections=3
    )
    if len(picked) < 2:
        st.info("Pick two or three segments.")
    else:
        details = [_get(f"/clusters/{cluster['cluster_id']}") for cluster in picked]
        features = []
        for detail in details:
            for row in detail["numeric_profile"]:
                if row["feature"] not in features:
                    features.append(row["feature"])
        figure = go.Figure()
        for detail in details:
            lookup = {
                row["feature"]: row["standardised_difference"] for row in detail["numeric_profile"]
            }
            figure.add_trace(
                go.Bar(
                    name=f"{detail['cluster_id']}: {detail['name']}",
                    x=features,
                    y=[lookup.get(feature, 0.0) for feature in features],
                    marker_color=colors.get(str(detail["cluster_id"]), "#333333"),
                )
            )
        figure.update_layout(
            barmode="group",
            title="Standardised numeric profiles",
            xaxis_title="Feature",
            yaxis_title="z versus the catalogue",
            height=480,
        )
        st.plotly_chart(figure, use_container_width=True)
        columns = st.columns(len(details))
        for column, detail in zip(columns, details, strict=True):
            column.markdown(f"**{detail['cluster_id']}: {detail['name']}**")
            column.write(f"{detail['size']} titles ({detail['percentage']:.1f}%)")
            column.write(
                ", ".join(row["genre"] for row in detail["genre_lift"][:5]) or "No lift genres"
            )
            column.write(", ".join(row["term"] for row in detail["top_terms"][:8]))

with tab_k:
    st.subheader("Why this many segments")
    st.write(meta.get("k_reason", "The k-selection report has not been attached to this model."))
    st.write(
        f"Product representation `{meta.get('representation', {}).get('name', '')}`, "
        f"k={meta.get('chosen_k')}, algorithm `{meta.get('algorithm')}`."
    )
    scores = meta.get("validation_scores") or {}
    external = meta.get("external_validation") or {}
    st.markdown(
        f"Silhouette **{scores.get('silhouette', 'n/a')}**, "
        f"Calinski-Harabasz **{scores.get('calinski_harabasz', 'n/a')}**, "
        f"Davies-Bouldin **{scores.get('davies_bouldin', 'n/a')}**, "
        f"bootstrap ARI **{scores.get('bootstrap_ari', 'n/a')}**."
    )
    if external:
        st.markdown(
            "Held-out association (the fit that produced these numbers never saw the attribute): "
            + ", ".join(
                f"{name} ARI={external[name].get('ari'):.3f} / NMI={external[name].get('nmi'):.3f}"
                for name in ("type", "rating", "primary_genre")
                if name in external and isinstance(external[name], dict)
            )
        )
    st.write(
        "The elbow is a heuristic. Stability (bootstrap ARI) decided k. "
        "k=2 is almost perfectly stable because it rediscovers movie versus series, "
        "so it was not allowed to set the ceiling. Silhouette on overlapping genres "
        "is modest on purpose; it was not tuned upward."
    )
    sweep = meta.get("k_sweep") or []
    if sweep:
        sweep_frame = pd.DataFrame(sweep)
        stability_fig = go.Figure()
        stability_fig.add_trace(
            go.Scatter(
                x=sweep_frame["k"],
                y=sweep_frame["bootstrap_ari"],
                error_y={"type": "data", "array": sweep_frame["bootstrap_ari_std"]},
                name="Bootstrap ARI",
                mode="lines+markers",
            )
        )
        stability_fig.add_trace(
            go.Scatter(
                x=sweep_frame["k"],
                y=sweep_frame["silhouette"],
                name="Silhouette",
                mode="lines+markers",
                yaxis="y2",
            )
        )
        stability_fig.update_layout(
            title="Bootstrap stability and silhouette",
            xaxis_title="k",
            yaxis_title="Bootstrap ARI",
            yaxis2={"title": "Silhouette", "overlaying": "y", "side": "right"},
            height=420,
            legend={"orientation": "h"},
        )
        st.plotly_chart(stability_fig, use_container_width=True)
        index_fig = px.line(
            sweep_frame,
            x="k",
            y=["calinski_harabasz", "davies_bouldin"],
            markers=True,
            title="Calinski-Harabasz and Davies-Bouldin",
            labels={"value": "Score", "k": "k", "variable": "Index"},
        )
        index_fig.update_layout(height=420)
        st.plotly_chart(index_fig, use_container_width=True)
        elbow_fig = px.line(
            sweep_frame,
            x="k",
            y="inertia",
            markers=True,
            title="Inertia (the elbow is a heuristic)",
            labels={"inertia": "Inertia", "k": "k"},
        )
        elbow_fig.update_layout(height=380)
        st.plotly_chart(elbow_fig, use_container_width=True)
    else:
        st.info(
            "This model build does not include the k sweep. Run scripts/select_k.py and scripts/evaluate.py."
        )
    for figure_name, caption in (
        ("elbow_product.png", "Inertia and the programmatic knee."),
        ("stability_ari.png", "Bootstrap ARI against silhouette."),
        ("silhouette_clusters.png", "Per-cluster silhouette at the chosen k."),
        ("gmm_bic_aic.png", "Diagonal Gaussian mixture BIC and AIC."),
        ("dendrogram_ward.png", "Ward dendrogram, truncated to the top merges."),
        ("dendrogram_cosine.png", "Average-linkage cosine dendrogram."),
        ("contingency_type.png", "Cluster by held-out type, row-normalised."),
        ("contingency_rating.png", "Cluster by held-out rating, row-normalised."),
        ("contingency_primary_genre.png", "Cluster by held-out primary genre, row-normalised."),
    ):
        figure_bytes = load_figure(figure_name)
        if figure_bytes:
            st.image(figure_bytes, caption=caption)
        else:
            st.caption(f"{caption} Not in this build ({API_BASE_URL}/figures/{figure_name}).")
    size_fig = px.bar(
        clusters,
        x="name",
        y="size",
        color="name",
        color_discrete_map={
            cluster["name"]: colors.get(str(cluster["cluster_id"]), "#333333")
            for cluster in clusters
        },
        title="Segment sizes",
        labels={"name": "Segment", "size": "Titles"},
    )
    size_fig.update_layout(showlegend=False, height=420)
    st.plotly_chart(size_fig, use_container_width=True)

with tab_about:
    st.subheader("How it works")
    st.write(
        "Each title is a weighted mix of multi-hot genres, one-hot type / country / rating, "
        "and standardised numeric features including a within-type duration percentile. "
        "Blocks are scaled to unit total variance before weighting so text cannot drown the rest "
        "by column count. Rows are L2-normalised and clustered with spherical k-means "
        "(Euclidean k-means on the sphere, which tracks cosine distance). "
        "A diagonal Gaussian mixture supplies membership probabilities. "
        "A new title is transformed by the saved pipeline. It is never used to refit."
    )
    st.subheader("Limitations")
    st.markdown("""
- There is no ground truth. Agreement with held-out type and rating is association, not accuracy.
- The result depends on the metric, the weights, and k. Other reasonable settings move the borders.
- High-dimensional distances concentrate. SVD and spherical k-means reduce that problem; they do not remove it.
- One-hot countries are an arbitrary geometry: every pair of countries is the same distance apart. The categorical block is down-weighted for that reason.
- t-SNE and UMAP do not preserve global distance or cluster size.
- This extract has no synopsis and no cast, so the text block is the title only and was not selected as the product representation when it failed to improve stability.
- English stop words do not suit every title.
- The catalogue is one snapshot. Assignment of a new title does not update the segments.
        """)
    st.write(meta.get("projection_caveat", CAVEAT))
