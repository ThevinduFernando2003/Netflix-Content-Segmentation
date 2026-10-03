"""HTTP schemas. Field limits are the API contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClusterProbability(BaseModel):
    cluster_id: int
    cluster_name: str
    probability: float


class TitleRef(BaseModel):
    show_id: str
    title: str
    distance: float


class AssignRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    description: str = Field(min_length=10, max_length=5000)
    type: Literal["Movie", "TV Show"]
    genres: list[str] = Field(default_factory=list, max_length=20)
    release_year: int = Field(ge=1900, le=2100)
    country: str | None = None
    rating: str | None = None
    duration_minutes: int | None = Field(default=None, ge=1, le=500)
    duration_seasons: int | None = Field(default=None, ge=1, le=50)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "The Paper Kite Club",
                    "description": "Two children build kites and learn what friendship costs when the summer ends.",
                    "type": "Movie",
                    "genres": ["Children & Family Movies", "Comedies"],
                    "release_year": 2019,
                    "country": "United States",
                    "rating": "PG",
                    "duration_minutes": 98,
                }
            ]
        }
    }


class AssignResponse(BaseModel):
    cluster_id: int
    cluster_name: str
    distance_to_centroid: float
    membership_probabilities: list[ClusterProbability] | None
    is_outlier: bool
    nearest_titles: list[TitleRef]
    why: list[str]
    unknown_genres: list[str]
    model_version: str
    took_ms: float


class TitleClusterResponse(BaseModel):
    show_id: str
    title: str
    cluster_id: int
    cluster_name: str
    distance_to_centroid: float
    silhouette: float | None
    model_version: str
