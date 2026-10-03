"""HTTP contract, including the outlier flag and a frozen pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def test_health_and_ready(api_client) -> None:
    health = api_client.get("/health")
    ready = api_client.get("/ready")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert ready.status_code == 200


def test_meta_and_clusters(api_client) -> None:
    meta = api_client.get("/meta")
    clusters = api_client.get("/clusters")
    assert meta.status_code == 200
    body = meta.json()
    assert body["chosen_k"] == 3
    assert "cluster_colors" in body
    assert "projection_caveat" in body
    assert clusters.status_code == 200
    assert len(clusters.json()["clusters"]) == 3


def test_cluster_detail_404_and_pagination(api_client) -> None:
    missing = api_client.get("/clusters/99")
    assert missing.status_code == 404
    page = api_client.get("/clusters/0/titles", params={"limit": 2, "offset": 0})
    assert page.status_code == 200
    payload = page.json()
    assert payload["limit"] == 2
    assert len(payload["titles"]) <= 2
    empty = api_client.get("/clusters/0/titles", params={"limit": 10, "offset": 100000})
    assert empty.status_code == 200
    assert empty.json()["titles"] == []
    bounded = api_client.get("/clusters/0/titles", params={"limit": 500})
    assert bounded.status_code == 422


def test_assign_unknown_genre_and_existing_title(api_client) -> None:
    listing = api_client.get("/clusters/0/titles", params={"limit": 1}).json()
    show_id = listing["titles"][0]["show_id"]
    before = api_client.get(f"/titles/{show_id}/cluster")
    assert before.status_code == 200
    centers = np.array(api_client.app.state.artifacts.model.clusterer_.cluster_centers_, copy=True)
    labels = np.array(api_client.app.state.artifacts.model.labels_, copy=True)
    response = api_client.post(
        "/assign",
        json={
            "title": "Paper Kites",
            "description": "Children build kites and learn about friendship at school with a dragon.",
            "type": "Movie",
            "genres": ["Children & Family Movies", "Not A Real Genre"],
            "release_year": 2019,
            "country": "United States",
            "rating": "PG",
            "duration_minutes": 98,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "Not A Real Genre" in body["unknown_genres"]
    assert body["why"]
    assert body["nearest_titles"]
    assert body["membership_probabilities"]
    after = api_client.get(f"/titles/{show_id}/cluster")
    assert after.json()["cluster_id"] == before.json()["cluster_id"]
    assert np.allclose(centers, api_client.app.state.artifacts.model.clusterer_.cluster_centers_)
    assert np.array_equal(labels, api_client.app.state.artifacts.model.labels_)
    missing = api_client.get("/titles/does-not-exist/cluster")
    assert missing.status_code == 404


def test_outlier_flag_and_validation_errors(api_client) -> None:
    absurd = api_client.post(
        "/assign",
        json={
            "title": "Estonian Tax Law, Underwater",
            "description": "A silent experimental lecture on bracket creep filmed inside a drained reservoir with statutes and no story.",
            "type": "Movie",
            "genres": ["Not A Real Genre", "Underwater Basket Weaving"],
            "release_year": 1900,
            "country": "Atlantis",
            "rating": "UNRATED-XYZ",
            "duration_minutes": 4,
        },
    )
    assert absurd.status_code == 200
    assert absurd.json()["is_outlier"] is True
    invalid = api_client.post(
        "/assign",
        json={
            "title": "",
            "description": "short",
            "type": "Special",
            "release_year": 1800,
        },
    )
    assert invalid.status_code == 422
    detail = invalid.json()["detail"]
    assert any(item["loc"][-1] == "title" for item in detail)


def test_adversarial_inputs_do_not_500(api_client) -> None:
    long_description = "word " * 2000
    too_long = api_client.post(
        "/assign",
        json={
            "title": "Too Long",
            "description": long_description,
            "type": "Movie",
            "release_year": 2020,
        },
    )
    assert too_long.status_code == 422
    noisy = api_client.post(
        "/assign",
        json={
            "title": "Hello 🎬'; DROP TABLE titles; --",
            "description": "A detective story about a murder, a crime, and a prosecutor in a rainy city.",
            "type": "Movie",
            "genres": ["Nope"] * 15,
            "release_year": 2020,
            "country": "../etc/passwd",
            "rating": "R",
            "duration_minutes": 100,
        },
    )
    assert noisy.status_code == 200
    traversal = api_client.get("/titles/..%2F..%2Fetc%2Fpasswd/cluster")
    assert traversal.status_code in {404, 400}


def test_projection_sample_similar_and_metrics(api_client) -> None:
    projection = api_client.get("/projection", params={"sample": 10})
    assert projection.status_code == 200
    body = projection.json()
    assert body["sample_size"] == 10
    assert (
        "not interpretable" in body["caveat"].lower() or "do not preserve" in body["caveat"].lower()
    )
    show_id = body["points"][0]["show_id"]
    similar = api_client.get(f"/similar/{show_id}", params={"n": 3})
    assert similar.status_code == 200
    assert len(similar.json()["neighbours"]) <= 3
    metrics = api_client.get("/metrics")
    assert metrics.status_code == 200
    assert "http_requests_total" in metrics.text


def test_figure_route_rejects_traversal_and_serves_png(api_client) -> None:
    missing = api_client.get("/figures/elbow_product.png")
    assert missing.status_code == 404
    traversal = api_client.get("/figures/..%2Fmetadata.json")
    assert traversal.status_code == 404
    figures = Path(os.environ["ARTIFACT_DIR"]) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (figures / "elbow_product.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    found = api_client.get("/figures/elbow_product.png")
    assert found.status_code == 200
    assert found.headers["content-type"].startswith("image/png")
