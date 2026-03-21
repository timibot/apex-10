"""
Standalone worker to fetch the 50 major league matches for the upcoming weekend.
Runs once per week. Caches the fixtures into the Supabase `weekly_schedule` table.
"""
import logging
from datetime import datetime, timezone

import httpx

from apex10.config import LeagueConfig, APEX_ENV
from apex10.db import get_client
from apex10.live.inference import _fetch_league_fixtures, fetch_european_schedule

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run():
    logger.info(f"═══ APEX-10 Schedule Fetcher ({APEX_ENV}) ═══")
    
    # We only fetch major leagues for the core model
    thesportsdb_leagues = {
        "Premier League": 4328,
        "La Liga": 4335,
        "Bundesliga": 4331,
        "Serie A": 4332,
        "Ligue 1": 4334,
    }
    season = "2025-2026"
    
    all_fixtures = []
    
    with httpx.Client(timeout=15.0) as client:
        # 1. Fetch Major Leagues
        for league_name, lid in thesportsdb_leagues.items():
            
            try:
                fixtures = _fetch_league_fixtures(client, league_name, lid, season)
                all_fixtures.extend(fixtures)
            except Exception as e:
                logger.error(f"Failed fetching {league_name}: {e}")
                
        # 2. Fetch European Schedules
        try:
            euro_fixtures = fetch_european_schedule()
            for ef in euro_fixtures:
                all_fixtures.append({
                    "id": 0, # UEFA matches don't use SportsDB IDs in our pipeline
                    "league": "UEFA",
                    "match_date": ef["date"],
                    "home_team": ef["home"],
                    "away_team": ef["away"],
                    "round": 0,
                    "time": "00:00"
                })
        except Exception as e:
            logger.error(f"Failed fetching European schedule: {e}")

    if not all_fixtures:
        logger.warning("No fixtures found to cache.")
        return

    # Cache into Supabase
    db = get_client()
    
    payload = []
    for f in all_fixtures:
        payload.append({
            "fixture_id": f["id"],
            "league": f["league"],
            "match_date": f["match_date"],
            "match_time": f["time"],
            "home_team": f["home_team"],
            "away_team": f["away_team"],
            "round": f["round"],
            "status": "Upcoming"
        })
        
    try:
        # Clear old rows (e.g. from previous weeks) to keep the table lightweight
        db.table("weekly_schedule").delete().neq("status", "LOCKED").execute()
        
        # Upsert the new week's schedule
        db.table("weekly_schedule").upsert(payload, on_conflict="fixture_id").execute()
        logger.info(f"Successfully cached {len(payload)} fixtures to Supabase 'weekly_schedule' table.")
    except Exception as e:
        logger.error(f"Failed to write to Supabase: {e}")

if __name__ == "__main__":
    run()
