"""
Fetches xG data from Understat using Playwright headless browser.
Understat now requires full JS rendering — httpx/requests get empty data.

Flow:
  1. Launch headless Chromium via Playwright
  2. Navigate to Understat league page
  3. Wait for JS to render and inject data variables
  4. Extract datesData from the page context
  5. Parse and normalise xG records

Rate-limited: 5-second wait between page loads.
"""
from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger(__name__)

UNDERSTAT_BASE = "https://understat.com"

UNDERSTAT_LEAGUE_SLUGS = {
    "EPL": "EPL",
    "La Liga": "La_liga",
    "Bundesliga": "Bundesliga",
    "Serie A": "Serie_A",
    "Ligue 1": "Ligue_1",
}

REQUEST_DELAY = 5.0


def fetch_league_xg(league_name: str, season_year: int) -> list[dict] | None:
    """
    Fetch all match xG data for a league/season from Understat.
    Uses Playwright headless browser to render the JS-heavy page.
    Returns list of match dicts with home_xg, away_xg, or None on failure.
    """
    slug = UNDERSTAT_LEAGUE_SLUGS.get(league_name)
    if not slug:
        logger.error(f"Unknown league: {league_name}")
        return None

    url = f"{UNDERSTAT_BASE}/league/{slug}/{season_year}"
    logger.info(f"Fetching Understat xG: {url}")

    time.sleep(REQUEST_DELAY)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error(
            "Playwright not installed. Run: pip install playwright && playwright install chromium"
        )
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait for Understat to inject data into the DOM
            page.wait_for_timeout(3000)

            # Extract datesData from the page's JS context
            data = page.evaluate("""
                () => {
                    // Try multiple known variable names
                    if (typeof datesData !== 'undefined') return datesData;
                    if (typeof matchesData !== 'undefined') return matchesData;

                    // Fallback: search script tags for JSON.parse patterns
                    const scripts = document.querySelectorAll('script');
                    for (const s of scripts) {
                        const text = s.textContent;
                        const match = text.match(/var\\s+datesData\\s*=\\s*JSON\\.parse\\('(.+?)'\\)/);
                        if (match) {
                            return JSON.parse(match[1].replace(/\\\\x/g, (m) => m));
                        }
                    }
                    return null;
                }
            """)

            browser.close()

            if not data:
                logger.error("Could not extract datesData from Understat page")
                return None

            return _normalise_xg_records(data)

    except Exception as e:
        logger.error(f"Playwright Understat fetch failed: {e}")
        return None


def _normalise_xg_records(data: list[dict]) -> list[dict]:
    """Convert raw Understat match dicts to normalised records."""
    records = []
    for match in data:
        if match.get("isResult") is not True:
            continue  # Skip future fixtures
        try:
            records.append({
                "understat_id": int(match["id"]),
                "match_date": match["datetime"][:10],
                "home_team": match["h"]["title"],
                "away_team": match["a"]["title"],
                "home_xg": float(match["xG"]["h"]),
                "away_xg": float(match["xG"]["a"]),
                "home_goals": int(match["goals"]["h"]),
                "away_goals": int(match["goals"]["a"]),
            })
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Skipping Understat record: {e}")
    logger.info(f"Parsed {len(records)} xG records from Understat")
    return records


def upsert_xg_data(records: list[dict], db_client) -> int:
    """Upsert xG records into match_xg table."""
    if not records:
        return 0
    result = (
        db_client.table("match_xg")
        .upsert(records, on_conflict="understat_id")
        .execute()
    )
    return len(result.data) if result.data else 0
