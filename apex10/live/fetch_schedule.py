"""
Standalone worker to fetch the ~50 major league matches for the upcoming weekend.
Runs once per week (Friday morning via GitHub Actions).
Caches the fixtures into the Supabase `weekly_schedule` table.

NOTE: UEFA fixtures are NOT cached here — they are fetched directly by
inference.py for fatigue/sandwich scoring only.
"""
import logging
from datetime import date, timedelta

import httpx

from apex10.config import APEX_ENV, get_api_config
from apex10.db import get_client
from apex10.enrichment.odds_api import ODDS_SPORT_KEYS, ODDS_TEAM_MAP
from apex10.live.inference import _current_season_str, _fetch_league_fixtures

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _normalise_odds_name(name: str) -> str:
    """Map Odds API team name to our normalised convention."""
    return ODDS_TEAM_MAP.get(name, name)


def _fetch_fixtures_from_odds_api(
    client: httpx.Client, league_name: str, api_key: str
) -> list[dict]:
    """
    Fallback fixture source: pull this week's events directly from The Odds API.
    Used when TheSportsDB returns 0 fixtures for a league (stale data / missing round).
    Returns fixtures in the same dict format as _fetch_league_fixtures().
    """
    sport_key = ODDS_SPORT_KEYS.get(league_name)
    if not sport_key:
        logger.warning(f"No Odds API sport key for {league_name} — cannot fall back")
        return []

    today = date.today()
    window_end = today + timedelta(days=6)

    try:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        params = {
            "apiKey": api_key,
            "regions": "uk",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }
        r = client.get(url, params=params)
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        logger.error(f"Odds API fallback failed for {league_name}: {e}")
        return []

    fixtures = []
    for event in events:
        commence = event.get("commence_time", "")
        match_date = commence[:10]          # "YYYY-MM-DD"
        match_time = commence[11:16] if len(commence) > 15 else ""  # "HH:MM"

        # Only keep games within the current week's window
        if not (today.isoformat() <= match_date <= window_end.isoformat()):
            continue

        home = _normalise_odds_name(event.get("home_team", ""))
        away = _normalise_odds_name(event.get("away_team", ""))

        # Generate a stable integer fixture_id from the Odds API event ID
        # Prefix 9 avoids collision with TheSportsDB IDs (which are 6-7 digits)
        fixture_id = int("9" + str(abs(hash(event.get("id", f"{home}{away}{match_date}"))) % 10 ** 8))

        fixtures.append({
            "id": fixture_id,
            "league": league_name,
            "match_date": match_date,
            "time": match_time + " UTC",
            "home_team": home,
            "away_team": away,
            "round": "Unknown",
        })

    logger.info(
        f"Odds API fallback: {league_name} — {len(fixtures)} fixtures "
        f"(window {today} → {window_end})"
    )
    return fixtures


def run():
    logger.info(f"═══ APEX-10 Schedule Fetcher ({APEX_ENV}) ═══")

    # Major leagues for the core model
    thesportsdb_leagues = {
        "Premier League": 4328,
        "La Liga": 4335,
        "Bundesliga": 4331,
        "Serie A": 4332,
        "Ligue 1": 4334,
    }
    season = _current_season_str()   # e.g. "2025-2026"
    cfg = get_api_config()

    all_fixtures = []

    with httpx.Client(timeout=15.0) as client:
        for league_name, lid in thesportsdb_leagues.items():
            try:
                fixtures = _fetch_league_fixtures(client, league_name, lid, season)

                if not fixtures:
                    # TheSportsDB has stale / missing data for this round — use Odds API
                    logger.warning(
                        f"{league_name}: TheSportsDB returned 0 fixtures "
                        f"— falling back to Odds API"
                    )
                    fixtures = _fetch_fixtures_from_odds_api(
                        client, league_name, cfg.ODDS_API_KEY
                    )

                all_fixtures.extend(fixtures)
                logger.info(f"{league_name}: {len(fixtures)} fixtures loaded")

            except Exception as e:
                logger.error(f"Failed fetching {league_name}: {e}")

    if not all_fixtures:
        logger.warning("No fixtures found to cache for any league.")
        return

    # Write to Supabase
    db = get_client()

    payload = [
        {
            "fixture_id": f["id"],
            "league": f["league"],
            "match_date": f["match_date"],
            "match_time": f["time"],
            "home_team": f["home_team"],
            "away_team": f["away_team"],
            "round": f["round"],
            "status": "Upcoming",
        }
        for f in all_fixtures
    ]

    try:
        # Clear previous week's rows, keep any LOCKED rows
        db.table("weekly_schedule").delete().neq("status", "LOCKED").execute()
        db.table("weekly_schedule").upsert(payload, on_conflict="fixture_id").execute()
        logger.info(
            f"Successfully cached {len(payload)} fixtures to 'weekly_schedule'."
        )
    except Exception as e:
        logger.error(f"Failed to write to Supabase: {e}")


if __name__ == "__main__":
    run()
