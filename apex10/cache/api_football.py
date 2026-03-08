"""
Fetches match results from API-Football (v3.football.api-sports.io).
Returns clean, validated match records ready for Supabase insertion.
"""
from __future__ import annotations

import logging

import httpx

from apex10.config import LEAGUES, get_api_config

logger = logging.getLogger(__name__)

# Seasons to backfill — 5 years per spec
BACKFILL_SEASONS = [2019, 2020, 2021, 2022, 2023]


def _headers() -> dict:
    return {
        "x-apisports-key": get_api_config().API_FOOTBALL_KEY,
        "x-apisports-host": "v3.football.api-sports.io",
    }


def fetch_fixtures(league_id: int, season: int) -> list[dict]:
    """
    Fetch all finished fixtures for a given league + season.
    Returns a list of raw fixture dicts from the API.
    Raises httpx.HTTPStatusError on non-2xx responses.
    """
    cfg = get_api_config()
    url = f"{cfg.API_FOOTBALL_BASE}/fixtures"
    params = {
        "league": league_id,
        "season": season,
        "status": "FT",  # Full Time only
    }

    logger.info(f"Fetching fixtures: league={league_id}, season={season}")

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers=_headers(), params=params)
        response.raise_for_status()

    data = response.json()
    fixtures = data.get("response", [])
    logger.info(f"Received {len(fixtures)} fixtures")
    return fixtures


def parse_fixture(raw: dict) -> dict | None:
    """
    Parse a raw API-Football fixture dict into a clean match record.
    Returns None if essential fields are missing (graceful skip).
    """
    try:
        fixture = raw["fixture"]
        teams = raw["teams"]
        goals = raw["goals"]
        league = raw["league"]

        home_goals = goals.get("home")
        away_goals = goals.get("away")

        # Skip if score is missing (abandoned match etc.)
        if home_goals is None or away_goals is None:
            logger.warning(f"Skipping fixture {fixture['id']} — missing score")
            return None

        return {
            "api_match_id": fixture["id"],
            "league": league["name"],
            "season": league["season"],
            "match_date": fixture["date"][:10],  # ISO date only
            "home_team": teams["home"]["name"],
            "away_team": teams["away"]["name"],
            "home_goals": int(home_goals),
            "away_goals": int(away_goals),
            "status": "finished",
            "raw_json": raw,
        }
    except (KeyError, TypeError) as e:
        logger.warning(f"Failed to parse fixture: {e}")
        return None


def upsert_matches(matches: list[dict], db_client) -> int:
    """
    Upsert parsed match records into the matches table.
    Returns count of records upserted.
    Uses api_match_id as conflict key — safe to re-run.
    """
    if not matches:
        return 0

    result = (
        db_client.table("matches")
        .upsert(matches, on_conflict="api_match_id")
        .execute()
    )
    count = len(result.data) if result.data else 0
    logger.info(f"Upserted {count} matches")
    return count


def backfill_league(league_name: str, db_client) -> dict:
    """
    Backfill all BACKFILL_SEASONS for a given league.
    Returns summary: {season: count_upserted}
    """
    league_id = LEAGUES.LEAGUE_IDS[league_name]
    summary = {}

    for season in BACKFILL_SEASONS:
        raw_fixtures = fetch_fixtures(league_id, season)
        parsed = [p for f in raw_fixtures if (p := parse_fixture(f)) is not None]
        count = upsert_matches(parsed, db_client)
        summary[season] = count
        logger.info(f"{league_name} {season}: {count} matches upserted")

    return summary
