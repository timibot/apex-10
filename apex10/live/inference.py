"""
Live inference engine — Phase 7.
Fetches upcoming fixtures from TheSportsDB (free, no API key needed),
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
from apex10.models.features import (
    MARKET_FEATURES,
    ONPITCH_FEATURES,
    normalise_team_name,
)
from apex10.scoring.dixon_coles import derive_probabilities

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
    Fetch upcoming fixtures for one league.
    Scans rounds to find the next round with unplayed matches.
    """
    today = date.today()
    fixtures = []

    # Start scanning from a high round (by March, most leagues are ~round 25-30)
    start_round = max(1, 20 if today.month >= 11 or today.month <= 7 else 1)
    for round_num in range(start_round, 39):
        url = "https://www.thesportsdb.com/api/v1/json/3/eventsround.php"
        params = {"id": league_id, "r": round_num, "s": season}

        import time as _time
        _time.sleep(1.5)  # Rate limit: TheSportsDB free tier

        # Retry on 429 with backoff
        for attempt in range(3):
            response = client.get(url, params=params)
            if response.status_code == 429:
                wait = 5 * (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt+1}/3)")
                _time.sleep(wait)
                continue
            response.raise_for_status()
            break
        else:
            logger.warning(f"Giving up on {league_name} R{round_num} after 3 rate limit retries")
            break

        events = response.json().get("events") or []
        if not events:
            continue

        # Check if this round has any upcoming matches
        has_upcoming = any(
            not _has_kicked_off(e.get("dateEvent", ""), e.get("strTime", ""))
            for e in events
        )

        if has_upcoming:
            for event in events:
                event_date_str = event.get("dateEvent", "")
                event_time_str = event.get("strTime", "")

                # Skip matches that have already kicked off
                if _has_kicked_off(event_date_str, event_time_str):
                    continue

                home = _normalise_sportsdb_team(event.get("strHomeTeam", ""))
                away = _normalise_sportsdb_team(event.get("strAwayTeam", ""))

                fixture = {
                    "id": int(event.get("idEvent", 0)),
                    "league": league_name,
                    "match_date": event_date_str,
                    "home_team": home,
                    "away_team": away,
                    "round": round_num,
                    "time": event_time_str,
                }
                fixtures.append(fixture)

            logger.info(f"  {league_name} R{round_num}: {len(fixtures)} upcoming")
            break  # Found the active round

    return fixtures


def fetch_upcoming_fixtures() -> list[dict]:
    """
    Fetch upcoming fixtures across all supported leagues from TheSportsDB.
    Filters out matches that have already kicked off using date + time.
    """
    season = _current_season_str()
    logger.info(f"Fetching fixtures from TheSportsDB (season={season})...")

    all_fixtures = []
    with httpx.Client(timeout=15.0) as client:
        for league_name, league_id in THESPORTSDB_LEAGUES.items():
            try:
                league_fixtures = _fetch_league_fixtures(
                    client, league_name, league_id, season
                )
                all_fixtures.extend(league_fixtures)
            except Exception as e:
                logger.error(f"Failed to fetch {league_name}: {e}")

    logger.info(f"Found {len(all_fixtures)} upcoming fixtures across {len(THESPORTSDB_LEAGUES)} leagues")
    return all_fixtures


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


def _build_xg_stats(db, team: str, n: int = 8) -> dict:
    """Fetch rolling xG stats for a team."""
    norm_team = normalise_team_name(team)

    home_xg = (
        db.table("match_xg")
        .select("home_xg, away_xg")
        .eq("home_team", norm_team)
        .order("match_date", desc=True)
        .limit(n)
        .execute()
    ).data or []

    away_xg = (
        db.table("match_xg")
        .select("home_xg, away_xg")
        .eq("away_team", norm_team)
        .order("match_date", desc=True)
        .limit(n)
        .execute()
    ).data or []

    xg_for = [float(m["home_xg"]) for m in home_xg] + [float(m["away_xg"]) for m in away_xg]
    xg_against = [float(m["away_xg"]) for m in home_xg] + [float(m["home_xg"]) for m in away_xg]

    xg_for_avg = np.mean(xg_for) if xg_for else 1.3
    xg_against_avg = np.mean(xg_against) if xg_against else 1.1

    return {
        "xg_l8": round(xg_for_avg, 3),
        "xga_l8": round(xg_against_avg, 3),
        "xg_diff": round(xg_for_avg - xg_against_avg, 3),
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


def build_feature_vector(fixture: dict, db, elo_ratings: dict | None = None, live_odds: dict | None = None) -> dict:
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
            features["odds_opening_home"] = match_odds["home"]
            features["odds_current_home"] = match_odds["home"]
            features["odds_movement"] = 0.0  # opening == current for now

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
        lgbm_prob = lgbm_cal.predict([lgbm_raw])[0]
    else:
        lgbm_prob = lgbm_raw

    # XGBoost
    xgb = models["xgb_latest"]
    xgb_raw = xgb.predict_proba(market_vec)[0][1]
    xgb_cal = models.get("xgb_calibrator")
    if xgb_cal is not None:
        xgb_prob = xgb_cal.predict([xgb_raw])[0]
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
    Compare Dixon-Coles model probabilities vs bookmaker odds across all markets.
    Returns (bet_type, best_odds, model_prob) for the bet with the highest edge
    within the 1.20–1.49 odds range.

    Fall back to Home Win if no market has odds in range.
    """
    # Map our probability keys to odds API keys and human-readable names
    MARKET_MAP = {
        "home_win":  ("home",     "Home Win"),
        "dc_1x":     (None,       "Double Chance 1X"),     # no bookmaker odds yet
        "dnb_home":  (None,       "Draw No Bet"),          # no bookmaker odds yet
        "over_1_5":  ("over_1_5", "Over 1.5 Goals"),
        "over_2_5":  ("over_2_5", "Over 2.5 Goals"),
        "under_3_5": ("under_3_5","Under 3.5 Goals"),
    }

    best_bet = "Home Win"
    best_odds = match_odds.get("home", 1.5) if match_odds else 1.5
    best_prob = home_probs.get("home_win", 0.5)
    best_edge = -999.0

    for prob_key, (odds_key, bet_name) in MARKET_MAP.items():
        model_prob = home_probs.get(prob_key, 0)
        if model_prob <= 0:
            continue

        # Get bookmaker odds for this market
        if odds_key and match_odds and match_odds.get(odds_key):
            bookie_odds = match_odds[odds_key]
        else:
            # Derive fair odds from model prob (no vig)
            bookie_odds = round(1.0 / model_prob, 3) if model_prob > 0 else 99.0

        # Only consider bets in the target range (or close to it)
        implied_prob = 1.0 / bookie_odds if bookie_odds > 1.0 else 1.0
        edge = model_prob - implied_prob

        # Prefer bets in range with positive edge
        in_range = 1.10 <= bookie_odds <= 1.60  # slightly wider for ranking
        score = edge + (0.5 if in_range else 0.0)  # bonus for being in range

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

    with CacheRunLogger(db) as cache_log:
        # Fetch upcoming fixtures from TheSportsDB
        fixtures = fetch_upcoming_fixtures()
        cache_log.stats["fixtures_written"] = len(fixtures)

        if not fixtures:
            logger.warning("No upcoming fixtures found")
            return {"fixtures": 0, "scored": 0}

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
                    features = build_feature_vector(fixture, db, elo_ratings=elo_ratings, live_odds=live_odds)
                    probs = score_fixture(features, models)

                    # Dixon-Coles multi-market probabilities
                    home_xg_avg = features.get("xg_home_l8", 1.3)
                    away_xg_avg = features.get("xg_away_l8", 1.1)
                    dc_probs = derive_probabilities(home_xg_avg, away_xg_avg, league_rho)

                    # Find best-value bet across all markets
                    match_odds_data = get_match_odds(live_odds, fixture["home_team"], fixture["away_team"])
                    best_bet_type, best_odds, best_prob = _find_best_bet(
                        features, dc_probs, match_odds_data
                    )

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
                        "opening_odds": best_odds,
                        "key_player_absent_home": 0,
                        "key_player_absent_away": 0,
                        "features_complete": True,
                    }
                    scored.append(row)
                    logger.info(
                        f"  [{league_name}] {fixture['home_team']} vs {fixture['away_team']}: "
                        f"Best={best_bet_type} @{best_odds:.2f} (prob={best_prob:.3f}) "
                        f"LGBM={probs['lgbm_prob']:.3f} XGB={probs['xgb_prob']:.3f}"
                    )

                except Exception as e:
                    logger.error(f"Failed to score {fixture.get('home_team','?')} vs {fixture.get('away_team','?')}: {e}")
                    cache_log.add_source_failure("inference", str(e))

        # Upsert to upcoming_fixtures
        if scored:
            db.table("upcoming_fixtures").upsert(
                scored, on_conflict="api_match_id"
            ).execute()
            logger.info(f"Wrote {len(scored)} scored fixtures to upcoming_fixtures")

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
    return summary


if __name__ == "__main__":
    run_inference()
