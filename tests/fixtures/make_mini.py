"""Write tests/fixtures/mini_netflix.csv. Run once and commit the CSV."""

from __future__ import annotations

import csv
from pathlib import Path

OUT = Path(__file__).with_name("mini_netflix.csv")

GROUPS = [
    (
        "kids",
        "Children & Family Movies, Comedies",
        "PG",
        "United States",
        "A bright animated friendship story about school, a dragon, and a summer adventure.",
        "Movie",
        "95 min",
        2018,
    ),
    (
        "crime",
        "Dramas, Thrillers",
        "R",
        "United States",
        "A detective investigates a murder, a crime syndicate, and a corrupt prosecutor.",
        "Movie",
        "120 min",
        2019,
    ),
    (
        "docs",
        "Documentaries, International Movies",
        "TV-14",
        "India",
        "A documentary about history, culture, and society across an international community.",
        "Movie",
        "84 min",
        2015,
    ),
]


def main() -> None:
    rows: list[dict[str, str]] = []
    counter = 1
    for group, genres, rating, country, description, kind, duration, year in GROUPS:
        for index in range(40):
            title = f"{group.title()} Tale {index}"
            director = f"{group.title()} Director"
            cast = f"Actor {group} {index}, Co-star {group}"
            date_added = "9/25/2021"
            show_id = f"s{counter}"
            row_type = kind
            row_duration = duration
            row_rating = rating
            row_country = country
            row_genres = genres
            row_director = director
            row_cast = cast
            row_title = title
            if group == "kids" and index == 0:
                row_rating = "74 min"
                row_duration = ""
            elif group == "kids" and index == 1:
                row_director = "not given"
                row_cast = ""
                row_country = "not given"
                row_rating = ""
                date_added = ""
            elif group == "kids" and index == 2:
                row_type = "TV Show"
                row_duration = "2 Seasons"
                row_genres = "Kids' TV, TV Comedies"
            elif group == "crime" and index == 0:
                row_country = "United States, India, France"
                row_title = "Hello, World"
            elif group == "crime" and index == 1:
                row_genres = "Dramas, Thrillers, Horror Movies, Independent Movies, Crime TV Shows"
            elif group == "docs" and index == 0:
                row_title = "Amélie Abroad"
                date_added = "September 25, 2021"
            rows.append(
                {
                    "show_id": show_id,
                    "type": row_type,
                    "title": row_title,
                    "director": row_director,
                    "cast": row_cast,
                    "country": row_country,
                    "date_added": date_added,
                    "release_year": str(year),
                    "rating": row_rating,
                    "duration": row_duration,
                    "listed_in": row_genres,
                    "description": description,
                    "expected_group": group,
                }
            )
            counter += 1
    rows.append(dict(rows[0]))
    rows[-1]["show_id"] = rows[0]["show_id"]
    rows[-1]["title"] = rows[0]["title"] + " duplicate"
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
