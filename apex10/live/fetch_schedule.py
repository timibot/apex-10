"""
Standalone worker to fetch upcoming weekend fixtures for all major leagues.
Runs every Friday morning via GitHub Actions.
Caches fixture data into the Supabase `weekly_schedule` table.

Primary source: The Odds API (reliable, always current)
TheSportsDB is no longer used — its free API returns stale/incorrect data.
"""
import logging
from datetime import date, timedelta

import httpx

from apex10.config import APEX_ENV, get_api_config
from apex10.db import get_client
from apex10.enrichment.odds_api import ODDS_SPORT_KEYS, ODDS_TEAM_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Leagues to fetch — must match keys in ODDS_SPORT_KEYS
TARGET_LEAGUES = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]

# How many days ahead to look (covers a full weekend round)
WINDOW_DAYS = 6


def _normalise(name: str) -> str:
    return ODDS_TEAM_MAP.get(name, name)


def _fetch_league_from_odds_api(
    client: httpx.Client,
    league_name: str,
    api_key: str,
    today: date,
    window_end: date,
) -> list[dict]:
    """
    Fetch this week's fixtures for a single league from The Odds API.
    Returns a list of fixture dicts ready for the weekly_schedule table.
    """
    sport_key = ODDS_SPORT_KEYS.get(league_name)
    if not sport_key:
        logger.warning(f"No Odds API sport key for {league_name} — skipping")
        return []

    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "uk",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }

    try:
        r = client.get(url, params=params)
        r.raise_for_status()
        remaining = r.headers.get("x-requests-remaining", "?")
        events = r.json()
    except Exception as e:
        logger.error(f"Odds API failed for {league_name}: {e}")
        return []

    fixtures = []
    for event in events:
        commence = event.get("commence_time", "")
        match_date = commence[:10]       # "YYYY-MM-DD"
        match_time = commence[11:16]     # "HH:MM"

        # Only keep games within this week's window
        if not (today.isoformat() <= match_date <= window_end.isoformat()):
            continue

        home = _normalise(event.get("home_team", ""))
        away = _normalise(event.get("away_team", ""))

        # Stable integer fixture_id from the Odds API event UUID
        # Prefix 9 keeps it out of TheSportsDB ID range (6-7 digits)
        fixture_id = int(
            "9" + str(abs(hash(event.get("id", f"{home}{away}{match_date}"))) % 10 ** 8)
        )

        fixtures.append({
            "fixture_id": fixture_id,
            "league": league_name,
            "match_date": match_date,
            "match_time": match_time + " UTC",
            "home_team": home,
            "away_team": away,
            "round": 0,
            "status": "Upcoming",
        })

    logger.info(
        f"{league_name}: {len(fixtures)} fixtures this week "
        f"({remaining} Odds API credits remaining)"
    )
    return fixtures


def run():
    logger.info(f"═══ APEX-10 Schedule Fetcher ({APEX_ENV}) ═══")

    today = date.today()
    window_end = today + timedelta(days=WINDOW_DAYS)
    logger.info(f"Fetching fixtures for window: {today} → {window_end}")

    cfg = get_api_config()
    all_fixtures = []

    with httpx.Client(timeout=20.0) as client:
        for league_name in TARGET_LEAGUES:
            try:
                fixtures = _fetch_league_from_odds_api(
                    client, league_name, cfg.ODDS_API_KEY, today, window_end
                )
                all_fixtures.extend(fixtures)
            except Exception as e:
                logger.error(f"Failed fetching {league_name}: {e}")

    total = len(all_fixtures)
    if total == 0:
        logger.warning(
            "No fixtures found across all leagues. "
            "Possible causes: international break, Odds API key issue, or all games "
            "are outside the 6-day window."
        )
        return

    logger.info(f"Total fixtures to cache: {total}")

    # Write to Supabase
    db = get_client()
    try:
        # Clear previous week's rows (keep LOCKED rows for settlement tracking)
        db.table("weekly_schedule").delete().neq("status", "LOCKED").execute()
        db.table("weekly_schedule").upsert(all_fixtures, on_conflict="fixture_id").execute()
        logger.info(f"Successfully cached {total} fixtures to 'weekly_schedule'.")
    except Exception as e:
        logger.error(f"Failed to write to Supabase: {e}")


if __name__ == "__main__":
    run()
