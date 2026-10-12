"""Capture screenshots and a walkthrough video of the local UI and API.

Defaults to the API on http://127.0.0.1:8011 and Streamlit on http://127.0.0.1:8511.
Override with API_BASE_URL and UI_BASE_URL.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
VIDEO = ROOT / "docs" / "demo_video"
UI = os.environ.get("UI_BASE_URL", "http://127.0.0.1:8511").rstrip("/")
API = os.environ.get("API_BASE_URL", "http://127.0.0.1:8011").rstrip("/")


def _shot(page: Page, name: str) -> None:
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / name), full_page=True)
    print(f"saved {name}", flush=True)


def _tab(page: Page, name: str) -> None:
    page.get_by_role("tab", name=name).click()
    page.wait_for_timeout(1200)


def _choose_example(page: Page, label: str, title_snippet: str) -> None:
    page.locator('[data-testid="stSelectbox"]:visible').first.click()
    page.get_by_role("option", name=label).click()
    page.wait_for_function(
        """(snippet) => {
            const field = document.querySelector('input[aria-label="Title"]');
            return field && field.value.includes(snippet);
        }""",
        arg=title_snippet,
        timeout=20_000,
    )


def _assign(page: Page, example: str, title_snippet: str) -> None:
    _tab(page, "Assign a new title")
    page.get_by_text("The model is not refit").first.wait_for(timeout=30_000)
    _choose_example(page, example, title_snippet)
    page.get_by_role("button", name="Assign").click()
    page.locator('[data-testid="stAlert"]:visible').wait_for(timeout=90_000)
    page.wait_for_timeout(2000)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    VIDEO.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(VIDEO),
            record_video_size={"width": 1440, "height": 900},
        )
        page = context.new_page()
        page.goto(UI, wait_until="domcontentloaded", timeout=60_000)
        page.get_by_text("Netflix content segments").first.wait_for(timeout=60_000)
        page.get_by_text("Projection distances are not interpretable").first.wait_for(timeout=60_000)
        page.locator(".js-plotly-plot").first.wait_for(timeout=60_000)
        page.wait_for_timeout(2500)
        _shot(page, "01_segment_map.png")

        _tab(page, "Segment explorer")
        page.get_by_text("Closest to the centroid").wait_for(timeout=60_000)
        page.wait_for_timeout(2000)
        _shot(page, "02_segment_explorer.png")

        _assign(page, "Family film", "Paper Kite")
        _shot(page, "03_assign_family.png")

        _assign(page, "Deliberate outlier", "Estonian")
        _shot(page, "04_assign_outlier.png")

        _tab(page, "Compare segments")
        page.get_by_text("Standardised numeric profiles").wait_for(timeout=60_000)
        page.wait_for_timeout(2000)
        _shot(page, "05_compare_segments.png")

        _tab(page, "How many segments?")
        page.get_by_text("Why this many segments").wait_for(timeout=60_000)
        page.wait_for_timeout(3000)
        _shot(page, "06_k_selection.png")

        _tab(page, "About")
        page.get_by_text("Limitations").wait_for(timeout=30_000)
        _shot(page, "07_about.png")

        page.goto(f"{API}/docs", wait_until="domcontentloaded", timeout=60_000)
        page.get_by_text("/assign").first.wait_for(timeout=30_000)
        page.wait_for_timeout(1500)
        _shot(page, "08_api_docs.png")

        for filename, shot_name in (
            ("dendrogram_ward.png", "09_dendrogram.png"),
            ("silhouette_clusters.png", "10_silhouette.png"),
            ("stability_ari.png", "11_stability.png"),
            ("contingency_type.png", "12_contingency_type.png"),
            ("contingency_rating.png", "13_contingency_rating.png"),
        ):
            page.goto(f"{API}/figures/{filename}", wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_timeout(500)
            page.screenshot(path=str(OUT / shot_name), full_page=True)
            print(f"saved {shot_name}", flush=True)

        video = page.video
        if video is not None:
            target = OUT / "demo_walkthrough.webm"
            video.save_as(str(target))
            print(f"saved {target}", flush=True)
        context.close()
        browser.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"capture failed: {exc}", file=sys.stderr)
        raise
