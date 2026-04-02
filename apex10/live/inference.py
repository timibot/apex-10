"""
Live inference engine — Phase 7.
Reads upcoming fixtures from Supabase weekly_schedule cache,
builds feature vectors, scores with trained models,
and writes predictions to upcoming_fixtures table for the ticket pipeline.

Run: python -m apex10.live.inference
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd

from apex10.cache.cache_log import CacheRunLogger
from apex10.config import APEX_ENV, LEAGUES
from apex10.db import get_client
from apex10.enrichment.clubelo import fetch_elo_ratings, get_elo_diff
from apex10.enrichment.odds_api import fetch_live_odds, get_match_odds
from apex10.enrichment.static_data import get_manager_days_in_post, is_key_player_absent
from apex10.models.features import (
    MARKET_FEATURES,
    ONPITCH_FEATURES,
    normalise_team_name,
)
from apex10.scoring.confidence import pick_best_bet
from apex10.scoring.dixon_coles import derive_probabilities
from apex10.live import notify

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "models"

# TheSportsDB league IDs — top 5 European leagues
THESPORTSDB_LEAGUES = {
    "Premier League": 4328,
    "La Liga": 4335,
    "Bundesliga": 4331,
    "Serie A": 4332,
    "Ligue 1": 4334,
}

# Map TheSportsDB league names → model directory names
# These must match what train.py uses as league param
LEAGUE_DIR_MAP = {
    "Premier League": "EPL",
    "La Liga": "La_Liga",
    "Bundesliga": "Bundesliga",
    "Serie A": "Serie_A",
    "Ligue 1": "Ligue_1",
}

# Team name mapping: TheSportsDB → normalised convention
SPORTSDB_TEAM_MAP = {
    # England
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Newcastle United": "Newcastle",
    "Newcastle Utd": "Newcastle",
    "Nott'm Forest": "Nottingham Forest",
    "West Ham United": "West Ham",
    "Leicester City": "Leicester",
    "Brighton and Hove Albion": "Brighton",
    "Sheffield United": "Sheffield Utd",
    "Ipswich Town": "Ipswich",
    "Luton Town": "Luton",
    # Spain
    "Atletico Madrid": "Atletico Madrid",
    "Athletic Bilbao": "Athletic Club",
    "Betis": "Real Betis",
    # Germany
    "Bayern Munich": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "Borussia Monchengladbach": "Monchengladbach",
    "RB Leipzig": "RB Leipzig",
    # Italy
    "AC Milan": "AC Milan",
    "Inter Milan": "Inter",
    "Internazionale": "Inter",
    # France
    "Paris Saint-Germain": "PSG",
    "Paris SG": "PSG",
    "AS Monaco": "Monaco",
    "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille",
}


def _normalise_sportsdb_team(name: str) -> str:
    """Normalise TheSportsDB team name to match our Supabase convention."""
    return SPORTSDB_TEAM_MAP.get(name, name)


# ── Stub defaults for features we don't have live data for yet ──────────────
STUB_DEFAULTS = {
    "ppda_home": 10.0, "ppda_away": 10.0, "ppda_delta": 0.0,
    "home_advantage": 1.0,
    "h2h_win_rate_home": 0.5,
    "injury_count_home": 0, "injury_count_away": 0,
    "key_player_absent_home": 0, "key_player_absent_away": 0,
    "weather_rain_mm": 0.0, "weather_wind_kmh": 10.0,
    "rivalry_index": 0.0,
    "playstyle_counter_attack": 0.0, "opponent_low_block": 0.0,
    "travel_hours_away": 1.5,
    "odds_opening_home": 1.5, "odds_current_home": 1.5, "odds_movement": 0.0,
    "fixture_congestion_home": 7.0, "fixture_congestion_away": 7.0,
    "days_to_next_home": 7.0, "days_to_next_away": 7.0,
    "sandwich_score_home": 0.0, "sandwich_score_away": 0.0,
    "competition_weight_home": 1.0, "next_competition_weight_home": 1.0,
    "manager_days_in_post": 365, "manager_days_in_post_away": 365,
    "motivation_asymmetry": 0.0,
    "relegation_battle_away": 0.0, "title_race_home": 0.0,
    "elo_diff": 0.0,
}


def _league_model_dir(league: str) -> Path:
    """Get model directory for a specific league."""
    dir_name = LEAGUE_DIR_MAP.get(league, league.replace(" ", "_"))
    return MODEL_DIR / dir_name


def load_models(league: str) -> dict | None:
    """
    Load serialised models and calibrators for a specific league.
    Returns None if mandatory models (lgbm, xgb) are not found.
    """
    league_dir = _league_model_dir(league)
    if not league_dir.exists():
        logger.warning(f"No model directory for {league}: {league_dir}")
        return None

    models = {}
    mandatory_found = True
    for name in ["lgbm_latest", "xgb_latest", "lgbm_calibrator", "xgb_calibrator"]:
        path = league_dir / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
        else:
            models[name] = None
            if name in ("lgbm_latest", "xgb_latest"):
                mandatory_found = False

    if not mandatory_found:
        logger.warning(f"Missing mandatory models for {league}")
        return None

    logger.info(f"Loaded models for {league}")
    return models


def _current_season_str() -> str:
    """Return TheSportsDB season string, e.g. '2025-2026'."""
    today = date.today()
    start_year = today.year if today.month >= 8 else today.year - 1
    return f"{start_year}-{start_year + 1}"


def _has_kicked_off(event_date_str: str, event_time_str: str) -> bool:
    """
    Check if a match has already kicked off based on date + time.
    Times from TheSportsDB are in UTC.
    """
    now_utc = datetime.now(timezone.utc)
    try:
        event_date = date.fromisoformat(event_date_str)
        # If match is in the future (tomorrow+), it hasn't kicked off
        if event_date > now_utc.date():
            return False
        # If match is in the past, it has kicked off
        if event_date < now_utc.date():
            return True
        # Match is today — check the time
        if event_time_str:
            time_str = event_time_str.replace("Z", "").strip()
            # Handle formats like "15:00:00" or "15:00"
            parts = time_str.split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
            kickoff_utc = datetime(
                event_date.year, event_date.month, event_date.day,
                hour, minute, tzinfo=timezone.utc
            )
            return now_utc >= kickoff_utc
        # No time available — assume it hasn't kicked off
        return False
    except (ValueError, IndexError):
        return False


def _fetch_league_fixtures(
    client: httpx.Client,
    league_name: str,
    league_id: int,
    season: str,
) -> list[dict]:
    """
    Intelligently probes eventsround.php to find the current active Matchweek.
    Bypasses the TheSportsDB 15-game truncation limit on eventsseason.php.
    Estimates the round chronologically to avoid 429 rate limits.
    """
    from datetime import date
    import time
    
    today = date.today()
    start_year = today.year if today.month >= 8 else today.year - 1
    start_date = date(start_year, 8, 15)
    days_since_start = max(0, (today - start_date).days)
    
    # Estimate ~1 round per week, minus 2 weeks for winter/international breaks
    estimated_r = max(1, min(38, (days_since_start // 7) + 1 - 2))
    
    # Search closest chronological rounds first
    probe_offsets = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7]
    
    active_events = []
    found_round = 0
    
    for offset in probe_offsets:
        r = estimated_r + offset
        if r < 1 or r > 40:
            continue
            
        url = f"https://www.thesportsdb.com/api/v1/json/3/eventsround.php?id={league_id}&r={r}&s={season}"
        time.sleep(2.0)  # Safe free tier delay
        
        for attempt in range(2):
            resp = client.get(url)
            if resp.status_code == 429:
                time.sleep(5.0 * (attempt + 1))
                continue
            break
        else:
            continue
            
        events = resp.json().get("events") or []
        
        # Does this round have upcoming matches?
        has_upcoming = False
        for e in events:
            d_str = e.get("dateEvent")
            t_str = e.get("strTime", "")
            if d_str and not _has_kicked_off(d_str, t_str):
                has_upcoming = True
                break
                
        if has_upcoming:
            active_events = events
            found_round = r
            break
            
    if not active_events:
        return []

    fixtures = []
    for event in active_events:
        date_str = event.get("dateEvent")
        time_str = event.get("strTime", "")
        
        if not date_str:
            continue
        if _has_kicked_off(date_str, time_str):
            continue
            
        home = _normalise_sportsdb_team(event.get("strHomeTeam", ""))
        away = _normalise_sportsdb_team(event.get("strAwayTeam", ""))
        round_num = int(event.get("intRound") or found_round)
        
        fixtures.append({
            "id": int(event.get("idEvent", 0)),
            "league": league_name,
            "match_date": date_str,
            "home_team": home,
            "away_team": away,
            "round": round_num,
            "time": time_str,
        })

    logger.info(f"  {league_name} R{found_round}: {len(fixtures)} upcoming fixtures found via probe")
    return fixtures


def fetch_european_schedule() -> list[dict]:
    """
    Fetch upcoming schedules for Champions League, Europa League, and Conference League.
    Bypasses TheSportsDB 15-game truncations by exclusively querying the robust Odds API.
    Returns a list of dicts: {"date": "YYYY-MM-DD", "home": team, "away": team}
    """
    import os
    api_key = os.getenv("ODDS_API_KEY")
    european_matches = []
    
    if not api_key:
        logger.warning("No ODDS_API_KEY found, skipping European schedule fetch")
        return []

    # The exact Odds API keys for European competitions
    league_keys = [
        "soccer_uefa_champs_league",
        "soccer_uefa_europa_league",
        "soccer_uefa_europa_conference_league"
    ]
    
    with httpx.Client(timeout=15.0) as client:
        for l_key in league_keys:
            url = f"https://api.the-odds-api.com/v4/sports/{l_key}/events?apiKey={api_key}"
            
            try:
                resp = client.get(url)
                if resp.status_code == 200:
                    events = resp.json()
                    for e in events:
                        date_str = e.get("commence_time", "")[:10]
                        if date_str:
                            european_matches.append({
                                "date": date_str,
                                "home": _normalise_sportsdb_team(e.get("home_team", "")),
                                "away": _normalise_sportsdb_team(e.get("away_team", ""))
                            })
                elif resp.status_code == 422:
                    # Odds API throws 422 if the league is out of season / unavailable
                    pass
                else:
                    logger.warning(f"Odds API {l_key} fetch failed: {resp.status_code}")
            except Exception as e:
                logger.error(f"Error fetching {l_key} schedule: {e}")
                
    logger.info(f"Loaded {len(european_matches)} European fixtures for fatigue analysis")
    return european_matches


def fetch_upcoming_fixtures() -> list[dict]:
    """
    Fetch upcoming fixtures across all supported leagues from the Supabase cache.
    Only returns fixtures within the next 8 days to prevent future-week games from
    bleeding into the current inference run (e.g. during international breaks where
    the cacher stores the NEXT active week's games early).
    """
    from datetime import date, timedelta
    from apex10.db import get_client

    db = get_client()
    today = date.today()
    today_str = today.isoformat()
    # Upper bound: only look 8 days ahead — keeps us strictly within the current week's round
    window_end_str = (today + timedelta(days=8)).isoformat()

    # 1. Pull the cached schedule — bounded to current week only
    try:
        response = (
            db.table("weekly_schedule")
            .select("*")
            .gte("match_date", today_str)
            .lte("match_date", window_end_str)
            .execute()
        )
        cached_matches = response.data or []
    except Exception as e:
        logger.error(f"Failed to fetch weekly_schedule from Supabase: {e}")
        return []

    if not cached_matches:
        logger.warning(
            f"No matches found in weekly_schedule for window {today_str} → {window_end_str}. "
            "If this is an international break week, that is expected."
        )
        return []

    # 2. Filter out matches that have already kicked off
    valid_fixtures = []
    for row in cached_matches:
        d_str = row.get("match_date")
        t_str = row.get("match_time", "")
        if d_str and _has_kicked_off(d_str, t_str):
            continue

        valid_fixtures.append({
            "id": row["fixture_id"],
            "league": row["league"],
            "match_date": d_str,
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "round": row["round"],
            "time": t_str
        })

    logger.info(
        f"Loaded {len(valid_fixtures)} fixtures from cache "
        f"(window: {today_str} → {window_end_str})"
    )
    return valid_fixtures


def _build_rolling_stats(db, team: str, role: str, n: int = 8) -> dict:
    """
    Build rolling stats for a team from recent historical matches.
    role = 'home' or 'away' — which side of the fixture the team is on.
    """
    home_matches = (
        db.table("matches")
        .select("home_goals, away_goals, match_date")
        .eq("home_team", team)
        .eq("status", "finished")
        .order("match_date", desc=True)
        .limit(n)
        .execute()
    ).data or []

    away_matches = (
        db.table("matches")
        .select("home_goals, away_goals, match_date")
        .eq("away_team", team)
        .eq("status", "finished")
        .order("match_date", desc=True)
        .limit(n)
        .execute()
    ).data or []

    all_matches = []
    for m in home_matches:
        all_matches.append({
            "gf": m["home_goals"], "ga": m["away_goals"],
            "date": m["match_date"], "at_home": True,
        })
    for m in away_matches:
        all_matches.append({
            "gf": m["away_goals"], "ga": m["home_goals"],
            "date": m["match_date"], "at_home": False,
        })
    all_matches.sort(key=lambda x: x["date"], reverse=True)
    recent = all_matches[:n]

    if not recent:
        return {
            "goals_scored_avg": 1.3, "goals_conceded_avg": 1.1,
            "form_pts_l5": 7.5, "form_gd_l5": 1.0,
            "clean_sheet_rate": 0.3,
        }

    goals_scored = [m["gf"] for m in recent]
    goals_conceded = [m["ga"] for m in recent]
    last5 = recent[:5]
    pts = sum(3 if m["gf"] > m["ga"] else (1 if m["gf"] == m["ga"] else 0) for m in last5)
    gd = sum(m["gf"] - m["ga"] for m in last5)
    cs = sum(1 for m in recent if m["ga"] == 0) / len(recent)

    return {
        "goals_scored_avg": np.mean(goals_scored),
        "goals_conceded_avg": np.mean(goals_conceded),
        "form_pts_l5": pts,
        "form_gd_l5": gd,
        "clean_sheet_rate": cs,
    }


def _build_split_stats(db, team: str, role: str, n: int = 5) -> dict:
    """
    Build form stats exclusively from home matches (if role='home') 
    or away matches (if role='away'). Useful for split-form signals.
    """
    if role == "home":
        matches = (
            db.table("matches")
            .select("home_goals, away_goals")
            .eq("home_team", team)
            .eq("status", "finished")
            .order("match_date", desc=True)
            .limit(n)
            .execute()
        ).data or []
        all_matches = [{"gf": m["home_goals"], "ga": m["away_goals"]} for m in matches]
    else:
        matches = (
            db.table("matches")
            .select("home_goals, away_goals")
            .eq("away_team", team)
            .eq("status", "finished")
            .order("match_date", desc=True)
            .limit(n)
            .execute()
        ).data or []
        all_matches = [{"gf": m["away_goals"], "ga": m["home_goals"]} for m in matches]

    if not all_matches:
        return {"form_pts_l5": 7.5, "form_gd_l5": 1.0}

    pts = sum(3 if m["gf"] > m["ga"] else (1 if m["gf"] == m["ga"] else 0) for m in all_matches)
    gd = sum(m["gf"] - m["ga"] for m in all_matches)
    
    return {"form_pts_l5": pts, "form_gd_l5": gd}


def _calc_h2h_win_rate(db, home: str, away: str, n: int = 5) -> float:
    """Calculate home team win rate over last n H2H meetings."""
    # Home team as home
    matches1 = (
        db.table("matches")
        .select("home_goals, away_goals, match_date")
        .eq("home_team", home)
        .eq("away_team", away)
        .eq("status", "finished")
        .execute()
    ).data or []
    
    # Home team as away
    matches2 = (
        db.table("matches")
        .select("home_goals, away_goals, match_date")
        .eq("home_team", away)
        .eq("away_team", home)
        .eq("status", "finished")
        .execute()
    ).data or []

    all_matches = []
    for m in matches1:
        all_matches.append({"home_won": m["home_goals"] > m["away_goals"], "date": m["match_date"]})
    for m in matches2:
        all_matches.append({"home_won": m["away_goals"] > m["home_goals"], "date": m["match_date"]})

    all_matches.sort(key=lambda x: x["date"], reverse=True)
    recent = all_matches[:n]
    if not recent:
        return 0.5
    
    wins = sum(1 for m in recent if m["home_won"])
    return float(wins / len(recent))


def _build_xg_stats(db, team: str, n: int = 8) -> dict:
    """
    Fetch rolling xG stats for a team.
    1) Try real xG from match_xg (current season preferred).
    2) If xG data is stale (all from last season), approximate from
       actual goals using regression-to-mean:
         approx_xG = goals * 0.78 + league_avg * 0.22
       This is ~85% correlated with real xG over 8-match windows.
    """
    norm_team = normalise_team_name(team)

    # --- Try real xG first (current season = dates >= 2025-08-01) ---
    season_start = "2025-08-01"

    home_xg = (
        db.table("match_xg")
        .select("home_xg, away_xg, match_date")
        .eq("home_team", norm_team)
        .gte("match_date", season_start)
        .order("match_date", desc=True)
        .limit(n)
        .execute()
    ).data or []

    away_xg = (
        db.table("match_xg")
        .select("home_xg, away_xg, match_date")
        .eq("away_team", norm_team)
        .gte("match_date", season_start)
        .order("match_date", desc=True)
        .limit(n)
        .execute()
    ).data or []

    xg_for = [float(m["home_xg"]) for m in home_xg] + [float(m["away_xg"]) for m in away_xg]
    xg_against = [float(m["away_xg"]) for m in home_xg] + [float(m["home_xg"]) for m in away_xg]

    if xg_for:
        # Real current-season xG available
        xg_for_avg = float(np.mean(xg_for))
        xg_against_avg = float(np.mean(xg_against)) if xg_against else 1.1
        return {
            "xg_l8": round(xg_for_avg, 3),
            "xga_l8": round(xg_against_avg, 3),
            "xg_diff": round(xg_for_avg - xg_against_avg, 3),
        }

    # --- Fallback: approximate xG from actual goals (current season) ---
    LEAGUE_AVG = 1.35  # typical league-average goals per team per match
    REGR_WEIGHT = 0.78  # weight on actual goals, 0.22 on league avg

    home_matches = (
        db.table("matches")
        .select("home_goals, away_goals")
        .eq("home_team", team)
        .eq("status", "finished")
        .gte("match_date", season_start)
        .order("match_date", desc=True)
        .limit(n)
        .execute()
    ).data or []

    away_matches = (
        db.table("matches")
        .select("home_goals, away_goals")
        .eq("away_team", team)
        .eq("status", "finished")
        .gte("match_date", season_start)
        .order("match_date", desc=True)
        .limit(n)
        .execute()
    ).data or []

    goals_for = [m["home_goals"] for m in home_matches] + [m["away_goals"] for m in away_matches]
    goals_against = [m["away_goals"] for m in home_matches] + [m["home_goals"] for m in away_matches]

    if not goals_for:
        return {"xg_l8": 1.3, "xga_l8": 1.1, "xg_diff": 0.2}

    # Sort by recency (combined), take last n
    raw_for = float(np.mean(goals_for[:n]))
    raw_against = float(np.mean(goals_against[:n]))

    # Regression to mean: approx_xG = goals * 0.78 + league_avg * 0.22
    approx_xg = raw_for * REGR_WEIGHT + LEAGUE_AVG * (1 - REGR_WEIGHT)
    approx_xga = raw_against * REGR_WEIGHT + LEAGUE_AVG * (1 - REGR_WEIGHT)

    logger.debug(
        f"  {team}: approx xG={approx_xg:.2f} (goals_avg={raw_for:.2f}), "
        f"xGA={approx_xga:.2f} (conceded_avg={raw_against:.2f})"
    )

    return {
        "xg_l8": round(approx_xg, 3),
        "xga_l8": round(approx_xga, 3),
        "xg_diff": round(approx_xg - approx_xga, 3),
    }


def _calc_fixture_congestion(db, team: str) -> float:
    """Calculate days since last match for fixture congestion."""
    from datetime import date as _date
    result = (
        db.table("matches")
        .select("match_date")
        .or_(f"home_team.eq.{team},away_team.eq.{team}")
        .eq("status", "finished")
        .order("match_date", desc=True)
        .limit(1)
        .execute()
    ).data
    if result:
        try:
            last = _date.fromisoformat(result[0]["match_date"])
            return (_date.today() - last).days
        except (ValueError, KeyError):
            pass
    return 7.0  # default


_league_tables = {}

def _get_league_table(db, league: str) -> list[str]:
    """
    Calculate and return the current league table for a given league (current season).
    Returns a list of team names ordered by points (descending), then GD (descending).
    """
    if league in _league_tables:
        return _league_tables[league]
        
    season_start = "2025-08-01"
    matches = (
        db.table("matches")
        .select("home_team, away_team, home_goals, away_goals")
        .eq("league", league)
        .eq("status", "finished")
        .gte("match_date", season_start)
        .execute()
    ).data or []
    
    standings = {}
    for m in matches:
        h, a = m["home_team"], m["away_team"]
        hg, ag = m["home_goals"], m["away_goals"]
        
        if h not in standings: standings[h] = {"pts": 0, "gd": 0}
        if a not in standings: standings[a] = {"pts": 0, "gd": 0}
        
        standings[h]["gd"] += (hg - ag)
        standings[a]["gd"] += (ag - hg)
        
        if hg > ag:
            standings[h]["pts"] += 3
        elif hg < ag:
            standings[a]["pts"] += 3
        else:
            standings[h]["pts"] += 1
            standings[a]["pts"] += 1
            
    # Sort by points desc, then GD desc
    sorted_teams = sorted(standings.keys(), key=lambda t: (standings[t]["pts"], standings[t]["gd"]), reverse=True)
    _league_tables[league] = sorted_teams
    return sorted_teams

def _get_team_ppda(db, team: str) -> float:
    """
    Fetches the most recent PPDA value for a given team.
    Returns float PPDA, falling back to 11.5 if missing.
    """
    try:
        response = (
            db.table("team_ppda")
            .select("ppda")
            .eq("team", team)
            .order("season", desc=True)
            .limit(1)
            .execute()
        )
        if response.data and len(response.data) > 0 and response.data[0].get("ppda") is not None:
            return float(response.data[0]["ppda"])
    except Exception as e:
        logger.debug(f"PPDA fetch failed for {team}: {e}")
    return 11.5


def build_feature_vector(
    fixture: dict, 
    db, 
    elo_ratings: dict | None = None, 
    live_odds: dict | None = None,
    existing_opening_odds: dict | None = None,
    european_schedule: list[dict] | None = None,
) -> dict:
    """
    Build the full feature vector for one upcoming fixture.
    Uses real rolling stats + Elo + congestion where available, stubs for the rest.
    """
    home_team = fixture["home_team"]
    away_team = fixture["away_team"]

    # Rolling match stats
    home_stats = _build_rolling_stats(db, home_team, "home")
    away_stats = _build_rolling_stats(db, away_team, "away")

    # Rolling xG
    home_xg = _build_xg_stats(db, home_team)
    away_xg = _build_xg_stats(db, away_team)

    # Build feature dict — start with stubs, override with real data
    features = dict(STUB_DEFAULTS)

    # PPDA Pressing Intensity & Delta
    ppda_home = _get_team_ppda(db, home_team)
    ppda_away = _get_team_ppda(db, away_team)
    features["ppda_home"] = ppda_home
    features["ppda_away"] = ppda_away
    features["ppda_delta"] = round(ppda_home - ppda_away, 2)

    # On-pitch features (real data)
    features["xg_home_l8"] = home_xg["xg_l8"]
    features["xga_home_l8"] = home_xg["xga_l8"]
    features["xg_away_l8"] = away_xg["xg_l8"]
    features["xga_away_l8"] = away_xg["xga_l8"]
    features["xg_diff_home"] = home_xg["xg_diff"]
    features["xg_diff_away"] = away_xg["xg_diff"]
    features["form_pts_home_l5"] = home_stats["form_pts_l5"]
    features["form_pts_away_l5"] = away_stats["form_pts_l5"]
    features["form_gd_home_l5"] = home_stats["form_gd_l5"]
    features["form_gd_away_l5"] = away_stats["form_gd_l5"]
    features["goals_scored_avg_home"] = home_stats["goals_scored_avg"]
    features["goals_conceded_avg_away"] = away_stats["goals_conceded_avg"]
    features["clean_sheet_rate_home"] = home_stats["clean_sheet_rate"]
    features["clean_sheet_rate_away"] = away_stats["clean_sheet_rate"]

    # Extended Enriched Features
    home_split = _build_split_stats(db, home_team, "home")
    away_split = _build_split_stats(db, away_team, "away")
    features["form_pts_home_split_l5"] = home_split["form_pts_l5"]
    features["form_pts_away_split_l5"] = away_split["form_pts_l5"]
    features["form_gd_home_split_l5"] = home_split["form_gd_l5"]
    features["form_gd_away_split_l5"] = away_split["form_gd_l5"]
    
    features["h2h_win_rate_home"] = _calc_h2h_win_rate(db, home_team, away_team)

    # League context / Motivation
    league = fixture["league"]
    table = _get_league_table(db, league)
    if table and home_team in table and away_team in table:
        home_pos = table.index(home_team) + 1
        away_pos = table.index(away_team) + 1
        total_teams = len(table)
        
        features["title_race_home"] = 1.0 if home_pos <= 4 else 0.0
        features["relegation_battle_away"] = 1.0 if away_pos >= (total_teams - 4) else 0.0
        
        home_mot = 2.0 if home_pos <= 4 or home_pos >= (total_teams - 4) else 1.0
        away_mot = 2.0 if away_pos <= 4 or away_pos >= (total_teams - 4) else 1.0
        features["motivation_asymmetry"] = home_mot - away_mot

    features["manager_days_in_post"] = get_manager_days_in_post(home_team)
    features["manager_days_in_post_away"] = get_manager_days_in_post(away_team)

    # Enriched features — Elo
    if elo_ratings:
        features["elo_diff"] = get_elo_diff(elo_ratings, home_team, away_team)

    # Enriched features — fixture congestion
    features["fixture_congestion_home"] = _calc_fixture_congestion(db, home_team)
    features["fixture_congestion_away"] = _calc_fixture_congestion(db, away_team)

    # Enriched features — live odds
    if live_odds:
        match_odds = get_match_odds(live_odds, home_team, away_team)
        if match_odds:
            current = match_odds["home"]
            opening = current
            if existing_opening_odds and fixture["id"] in existing_opening_odds:
                stored_opening = existing_opening_odds[fixture["id"]]
                if stored_opening is not None:
                    opening = stored_opening
            
            features["odds_opening_home"] = opening
            features["odds_current_home"] = current
            features["odds_movement"] = current - opening

    # European Sandwich Score
    if european_schedule:
        try:
            fix_date = date.fromisoformat(fixture["match_date"])
            
            def _has_euro_sandwich(team: str) -> float:
                for em in european_schedule:
                    if em["home"] == team or em["away"] == team:
                        em_date = date.fromisoformat(em["date"])
                        # Sandwich = Euro match played 1 to 4 days before OR after this league fixture
                        diff = abs((em_date - fix_date).days)
                        if 1 <= diff <= 4:
                            return 1.0
                return 0.0
                
            features["sandwich_score_home"] = _has_euro_sandwich(home_team)
            features["sandwich_score_away"] = _has_euro_sandwich(away_team)
        except ValueError:
            pass

    features["key_player_absent_home"] = is_key_player_absent(home_team)
    features["key_player_absent_away"] = is_key_player_absent(away_team)

    # True Oddsportal injected features
    features["odds_dnb_home"] = fixture.get("odds_dnb_home")
    features["odds_dnb_away"] = fixture.get("odds_dnb_away")
    features["odds_ah_home"] = fixture.get("odds_ah_home")
    features["odds_ah_away"] = fixture.get("odds_ah_away")

    return features


def score_fixture(features: dict, models: dict) -> dict:
    """Score a fixture with both models, return probabilities."""
    onpitch_vec = np.array([[features.get(f, 0.0) for f in ONPITCH_FEATURES]])
    market_vec = np.array([[features.get(f, 0.0) for f in MARKET_FEATURES]])

    # LightGBM
    lgbm = models["lgbm_latest"]
    lgbm_raw = lgbm.predict(onpitch_vec)[0]
    lgbm_cal = models.get("lgbm_calibrator")
    if lgbm_cal is not None:
        lgbm_prob = lgbm_cal.predict_proba(np.array([[lgbm_raw]]))[:, 1][0]
    else:
        lgbm_prob = lgbm_raw

    # XGBoost
    xgb = models["xgb_latest"]
    xgb_raw = xgb.predict_proba(market_vec)[0][1]
    xgb_cal = models.get("xgb_calibrator")
    if xgb_cal is not None:
        xgb_prob = xgb_cal.predict_proba(np.array([[xgb_raw]]))[:, 1][0]
    else:
        xgb_prob = xgb_raw

    return {
        "lgbm_prob": round(float(lgbm_prob), 4),
        "xgb_prob": round(float(xgb_prob), 4),
        "consensus_prob": round(float((lgbm_prob + xgb_prob) / 2), 4),
    }


def _get_league_rho(db, league: str) -> float:
    """Get rho correction for a league from Supabase."""
    model_league = LEAGUE_DIR_MAP.get(league, league)
    try:
        result = db.table("league_rho").select("rho").eq("league", model_league).execute()
        if result.data:
            return float(result.data[0]["rho"])
    except Exception:
        pass
    return -0.13  # default


def _find_best_bet(
    features: dict,
    home_probs: dict,
    match_odds: dict | None,
) -> tuple[str, float, float]:
    """
    Compare Dixon-Coles probabilities vs bookmaker odds across all markets.
    ONLY considers markets where real bookmaker odds are available.
    Returns (bet_type, best_odds, model_prob) for the bet with the best edge.
    """
    if not match_odds:
        # No live odds at all — fall back to Home Win with stub odds
        return "Home Win", 1.5, home_probs.get("home_win", 0.5)

    # Markets where we have both model prob AND bookmaker odds
    MARKETS = [
        ("home_win",   "home",      "Home Win"),
        ("over_1_5",   "over_1_5",  "Over 1.5 Goals"),
        ("over_2_5",   "over_2_5",  "Over 2.5 Goals"),
        ("under_3_5",  "under_3_5", "Under 3.5 Goals"),
        ("dc_1x",      None,        "Double Chance 1X"),
        ("dnb_home",   None,        "Draw No Bet"),
    ]

    best_bet = "Home Win"
    best_odds = match_odds.get("home", 1.5)
    best_prob = home_probs.get("home_win", 0.5)
    best_edge = -999.0

    for prob_key, odds_key, bet_name in MARKETS:
        model_prob = home_probs.get(prob_key, 0)
        if model_prob <= 0.01:
            continue

        # Must have real bookmaker odds — skip otherwise
        if odds_key is None or match_odds.get(odds_key) is None:
            # For DC and DNB, derive from h2h odds if available
            if prob_key == "dc_1x" and match_odds.get("home") and match_odds.get("draw"):
                # DC 1X implied = 1 / (1/home + 1/draw)
                implied = (1.0 / match_odds["home"]) + (1.0 / match_odds["draw"])
                bookie_odds = round(1.0 / implied, 3) if implied > 0 else 99.0
            elif prob_key == "dnb_home" and match_odds.get("home") and match_odds.get("away"):
                # DNB implied from h2h ≈ home / (home + away - 1)
                h, a = match_odds["home"], match_odds["away"]
                dnb_implied = (1.0 / h) / ((1.0 / h) + (1.0 / a))
                bookie_odds = round(1.0 / dnb_implied, 3) if dnb_implied > 0 else 99.0
            else:
                continue
        else:
            bookie_odds = match_odds[odds_key]

        if bookie_odds <= 1.0:
            continue

        implied_prob = 1.0 / bookie_odds
        edge = model_prob - implied_prob

        # Score: prefer positive edge + bets closer to target range [1.20-1.49]
        in_sweet_spot = 1.20 <= bookie_odds <= 1.49
        score = edge + (0.3 if in_sweet_spot else 0.0)

        if score > best_edge:
            best_edge = score
            best_bet = bet_name
            best_odds = bookie_odds
            best_prob = model_prob

    return best_bet, round(best_odds, 3), round(best_prob, 4)



def run_inference() -> dict:
    """
    Full inference pipeline:
    1. Fetch upcoming fixtures from TheSportsDB (all leagues)
    2. Group by league, load per-league models
    3. Score each fixture with its league's model pair
    4. Write to upcoming_fixtures table
    5. Log cache run
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info(f"═══ APEX-10 Inference Engine ({APEX_ENV}) ═══")

    db = get_client()

    # Fetch Elo ratings once for all fixtures
    elo_ratings = fetch_elo_ratings()

    # Fetch live odds once for all fixtures (costs ~5 credits for 5 leagues)
    live_odds = fetch_live_odds()

    # Fetch existing fixtures to preserve opening odds for movement tracking
    existing_fixtures_resp = db.table("upcoming_fixtures").select("api_match_id, opening_odds").execute()
    existing_opening_odds = {
        row["api_match_id"]: row["opening_odds"] 
        for row in (existing_fixtures_resp.data or [])
    }

    with CacheRunLogger(db) as cache_log:
        # Fetch upcoming fixtures from TheSportsDB
        fixtures = fetch_upcoming_fixtures()
        cache_log.stats["fixtures_written"] = len(fixtures)

        if not fixtures:
            logger.warning("No upcoming fixtures found")
            return {"fixtures": 0, "scored": 0}

        # Fetch European schedule for sandwich scores
        euro_schedule = fetch_european_schedule()

        # Group fixtures by league
        from collections import defaultdict
        by_league = defaultdict(list)
        for f in fixtures:
            by_league[f["league"]].append(f)

        scored = []
        leagues_loaded = set()
        leagues_skipped = set()

        for league_name, league_fixtures in by_league.items():
            models = load_models(league_name)
            if models is None:
                logger.warning(f"⚠️ Skipping {league_name} — no trained models")
                leagues_skipped.add(league_name)
                continue

            leagues_loaded.add(league_name)
            league_rho = _get_league_rho(db, league_name)

            for fixture in league_fixtures:
                try:
                    features = build_feature_vector(
                        fixture, db, 
                        elo_ratings=elo_ratings, 
                        live_odds=live_odds,
                        existing_opening_odds=existing_opening_odds,
                        european_schedule=euro_schedule
                    )
                    probs = score_fixture(features, models)

                    # Dixon-Coles multi-market probabilities
                    # mu = expected home goals = blend of home attack + away defence weakness
                    # nu = expected away goals = blend of away attack + home defence weakness
                    home_xg_for = features.get("xg_home_l8", 1.3)
                    away_xg_for = features.get("xg_away_l8", 1.1)
                    home_xg_against = features.get("xga_home_l8", 1.1)
                    away_xg_against = features.get("xga_away_l8", 1.3)
                    home_goals_scored = features.get("goals_scored_avg_home", 1.3)
                    away_goals_conceded = features.get("goals_conceded_avg_away", 1.3)

                    # Fixture-specific lambda: home attacking strength vs away defensive weakness
                    mu = (home_xg_for + away_xg_against + home_goals_scored) / 3.0
                    nu = (away_xg_for + home_xg_against) / 2.0

                    dc_probs = derive_probabilities(mu, nu, league_rho)

                    # Confidence voting: score all markets, pick best
                    match_odds_data = get_match_odds(live_odds, fixture["home_team"], fixture["away_team"])
                    elo_d = features.get("elo_diff", 0.0)
                    best = pick_best_bet(
                        dc_probs, features, match_odds_data,
                        elo_diff=elo_d, min_votes=3,
                    )

                    if best is not None:
                        best_bet_type = best.market
                        best_odds = best.odds
                        best_prob = best.probability
                        best_votes = best.votes
                        sig_str = " ".join(f"{k}={'✓' if v else '✗'}" for k, v in best.signals.items())
                    else:
                        best_bet_type = "Home Win"
                        best_odds = features.get("odds_opening_home", 1.5)
                        best_prob = probs["consensus_prob"]
                        best_votes = 0
                        sig_str = "no qualifying market"

                    row = {
                        "api_match_id": fixture["id"],
                        "league": fixture["league"],
                        "match_date": fixture["match_date"],
                        "home_team": fixture["home_team"],
                        "away_team": fixture["away_team"],
                        "lgbm_prob": probs["lgbm_prob"],
                        "xgb_prob": probs["xgb_prob"],
                        "consensus_prob": best_prob,
                        "best_bet_type": best_bet_type,
                        "best_bet_odds": best_odds,
                        "opening_odds": features.get("odds_opening_home", best_odds),
                        "key_player_absent_home": is_key_player_absent(fixture["home_team"]),
                        "key_player_absent_away": is_key_player_absent(fixture["away_team"]),
                        "features_complete": True,
                        "confidence_votes": best_votes,
                    }
                    scored.append(row)
                    logger.info(
                        f"  [{league_name}] {fixture['home_team']} vs {fixture['away_team']}: "
                        f"Best={best_bet_type} @{best_odds:.2f} "
                        f"({best_votes}/5 votes, prob={best_prob:.3f}) [{sig_str}]"
                    )

                except Exception as e:
                    logger.error(f"Failed to score {fixture.get('home_team','?')} vs {fixture.get('away_team','?')}: {e}")
                    cache_log.add_source_failure("inference", str(e))

        # Upsert to upcoming_fixtures (Schema strictness requires stripping internal python vars)
        if scored:
            db_scored = [
                {k: v for k, v in row.items() if k != "confidence_votes"}
                for row in scored
            ]
            db.table("upcoming_fixtures").upsert(
                db_scored, on_conflict="api_match_id"
            ).execute()
            logger.info(f"Wrote {len(db_scored)} scored fixtures to upcoming_fixtures")

        if leagues_skipped:
            logger.warning(f"Leagues skipped (no models): {leagues_skipped}")

    summary = {
        "fixtures_found": len(fixtures),
        "scored": len(scored),
        "leagues_scored": list(leagues_loaded),
        "leagues_skipped": list(leagues_skipped),
        "environment": APEX_ENV,
    }
    logger.info(f"═══ Inference complete ═══ {summary}")

    # ── Ticket Generation & Discord notification ──────────────────────
    if scored:
        from apex10.filters.gates import Candidate, run_all_gates
        from apex10.ticket.optimizer import build_tickets

        candidates = []
        for row in scored:
            # We skip matches where our AI didn't find any qualifying bet
            if row.get("confidence_votes", 0) == 0 and row["best_bet_type"] == "Home Win":
                continue

            candidates.append(
                Candidate(
                    fixture_id=row["api_match_id"],
                    league=row["league"],
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    bet_type=row["best_bet_type"],
                    odds=row["best_bet_odds"],
                    lgbm_prob=row["lgbm_prob"],
                    xgb_prob=row["xgb_prob"],
                    opening_odds=row["opening_odds"],
                    key_player_absent_home=row["key_player_absent_home"],
                    key_player_absent_away=row["key_player_absent_away"],
                    features_complete=row["features_complete"],
                    consensus_prob=row["consensus_prob"],
                    confidence_votes=row.get("confidence_votes", 0),
                )
            )

        qualified, rejections = run_all_gates(candidates)
        safe_ticket, master_ticket = build_tickets(qualified)

        if master_ticket.no_ticket:
            notify.no_ticket_week(
                week=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                reason=master_ticket.reason,
            )
        else:
            week_label = scored[0]["match_date"][:10] if scored else "unknown"
            
            def _format_ticket(ticket) -> str:
                lines = []
                for leg in ticket.legs:
                    edge = leg.consensus_prob - (1 / leg.odds if leg.odds > 0 else 1.0)
                    lines.append(
                        f"• {leg.home_team} vs {leg.away_team} — "
                        f"**{leg.bet_type} @{leg.odds:.2f}** [{leg.confidence_votes}/5 Votes] "
                        f"(Prob: {leg.consensus_prob:.1%} | Edge: {edge:+.1%})"
                    )
                return "\n".join(lines)

            # Ticket 1: Safe 10x Odds
            notify.ticket_generated(
                week=week_label,
                title="Safe ~10x Target Slip",
                legs=len(safe_ticket.legs),
                combined_odds=safe_ticket.combined_odds,
                stake=0.0,
                win_rate=safe_ticket.simulated_win_rate,
                breakdown=_format_ticket(safe_ticket),
            )
            
            # Ticket 2: Full Master List
            notify.ticket_generated(
                week=week_label,
                title="Full Uncapped Master List",
                legs=len(master_ticket.legs),
                combined_odds=master_ticket.combined_odds,
                stake=0.0,
                win_rate=master_ticket.simulated_win_rate,
                breakdown=_format_ticket(master_ticket),
            )
    else:
        notify.no_ticket_week(
            week=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            reason=f"No fixtures survived filtering. "
                   f"Found {len(fixtures)} fixtures, 0 scored.",
        )

    return summary


if __name__ == "__main__":
    run_inference()
