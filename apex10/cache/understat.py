"""
Fetches xG data from Understat.
Understat has no public API — we scrape the JSON embedded in page HTML.
Data is fetched async to avoid blocking on 5 seasons × N teams.
"""
from __future__ import annotations

import json
import logging
import re

import httpx

logger = logging.getLogger(__name__)

UNDERSTAT_BASE = "https://understat.com"

UNDERSTAT_LEAGUE_SLUGS = {
    "EPL": "EPL",
    "La Liga": "La_liga",
    "Bundesliga": "Bundesliga",
    "Serie A": "Serie_A",
    "Ligue 1": "Ligue_1",
}


def fetch_league_xg(league_name: str, season_year: int) -> list[dict] | None:
    """
    Fetch all match xG data for a league/season from Understat.
    Returns list of match dicts with home_xg, away_xg, or None on failure.
    """
    slug = UNDERSTAT_LEAGUE_SLUGS.get(league_name)
    if not slug:
        logger.error(f"Unknown league: {league_name}")
        return None

    url = f"{UNDERSTAT_BASE}/league/{slug}/{season_year}"
    logger.info(f"Fetching Understat xG: {url}")

    try:
        with httpx.Client(timeout=30.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error(f"Understat fetch failed: {e}")
        return None

    return _parse_xg_from_html(response.text)


def _parse_xg_from_html(html: str) -> list[dict] | None:
    """
    Extract the datesData JSON blob embedded in Understat page HTML.
    Understat embeds data as: var datesData = JSON.parse('...')
    """
    pattern = r"var datesData\s*=\s*JSON\.parse\('(.+?)'\)"
    match = re.search(pattern, html)
    if not match:
        logger.error("Could not find datesData in Understat HTML")
        return None

    try:
        # Understat escapes unicode — decode it
        raw_json = match.group(1).encode().decode("unicode_escape")
        data = json.loads(raw_json)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Failed to parse Understat JSON: {e}")
        return None

    return _normalise_xg_records(data)


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
