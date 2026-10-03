"""Human names derived from lift, composition, and c-TF-IDF evidence.

A cluster whose top genres barely beat the catalogue base rate is left unnamed
rather than given a decorative label.
"""

from __future__ import annotations

from typing import Any


def name_from_profile(profile: dict[str, Any]) -> tuple[str, str, bool]:
    """Return ``(name, evidence, interpretable)``."""
    lift_rows = profile.get("genre_lift") or []
    terms = [item["term"] for item in profile.get("top_terms", [])[:6]]
    composition = profile.get("composition") or {}
    type_counts = composition.get("type") or {}
    countries = composition.get("top_countries") or []
    year = (composition.get("release_year") or {}).get("median")
    total_type = sum(type_counts.values()) or 1
    movie_share = type_counts.get("Movie", 0) / total_type
    series_share = type_counts.get("TV Show", 0) / total_type
    kind = "Films" if movie_share >= 0.75 else "Series" if series_share >= 0.75 else "Titles"
    top = lift_rows[:3]
    strong = [row for row in top if row["lift"] >= 1.8 and row["cluster_rate"] >= 0.25]
    if not strong:
        evidence = (
            "No genre reaches lift >= 1.8 with cluster rate >= 0.25. "
            f"Highest lifts: {_fmt_lifts(top)}. "
            f"Title terms: {', '.join(terms) if terms else 'none'}. "
            "Left unnamed rather than forcing a theme."
        )
        return "Uninterpretable mix", evidence, False

    genre_bits = [_short_genre(row["genre"]) for row in strong[:2]]
    country = ""
    if countries:
        leader = countries[0]
        cluster_size = profile.get("size") or 1
        if leader["count"] / cluster_size >= 0.45 and leader["country"] not in {
            "",
            "Missing",
            "__MISSING__",
        }:
            country = leader["country"]
    era = ""
    if year is not None and year < 2000:
        era = "Classic"
    elif year is not None and year >= 2018:
        era = "Recent"
    pieces = [piece for piece in [era, country, " & ".join(genre_bits), kind] if piece]
    name = " ".join(pieces)
    # Keep names in the requested 2–5 word band when the evidence is specific.
    words = name.split()
    if len(words) > 5:
        name = " ".join(words[:5])
    if len(words) < 2:
        name = f"{name} Segment".strip()
    evidence = (
        f"Named from genre lift ({_fmt_lifts(strong)}), "
        f"type mix movies={movie_share:.0%} / series={series_share:.0%}, "
        f"leading country {countries[0]['country'] if countries else 'n/a'} "
        f"({countries[0]['count'] if countries else 0} titles), "
        f"median release year {year}. "
        f"c-TF-IDF terms: {', '.join(terms) if terms else 'none'}."
    )
    return name, evidence, True


def _short_genre(genre: str) -> str:
    replacements = {
        "Children & Family Movies": "Family",
        "Kids' TV": "Kids",
        "Stand-Up Comedy": "Stand-Up",
        "Stand-Up Comedy & Talk Shows": "Stand-Up & Talk",
        "International Movies": "International",
        "International TV Shows": "International",
        "TV Dramas": "TV Drama",
        "TV Comedies": "TV Comedy",
        "Crime TV Shows": "Crime TV",
        "Docuseries": "Docuseries",
        "Documentaries": "Documentaries",
        "Action & Adventure": "Action",
        "TV Action & Adventure": "TV Action",
        "Romantic Movies": "Romance",
        "Romantic TV Shows": "Romantic TV",
        "Sci-Fi & Fantasy": "Sci-Fi",
        "TV Sci-Fi & Fantasy": "Sci-Fi TV",
        "Horror Movies": "Horror",
        "Music & Musicals": "Music",
        "Anime Series": "Anime",
        "Anime Features": "Anime",
        "Independent Movies": "Independent",
        "British TV Shows": "British TV",
        "Korean TV Shows": "Korean TV",
        "Spanish-Language TV Shows": "Spanish-Language TV",
        "Reality TV": "Reality",
        "Children & Family": "Family",
    }
    return replacements.get(genre, genre)


def _fmt_lifts(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    return "; ".join(
        f"{row['genre']} lift={row['lift']:.2f} rate={row['cluster_rate']:.0%}" for row in rows
    )
