"""
Live inference engine — Phase 7.
Fetches upcoming fixtures, builds feature vectors, scores with trained models,
and writes predictions to upcoming_fixtures table for the ticket pipeline.

Run: python -m apex10.live.inference
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd

from apex10.cache.cache_log import CacheRunLogger
from apex10.config import APEX_ENV, LEAGUES, get_api_config
from apex10.db import get_client
from apex10.models.features import (
    MARKET_FEATURES,
    ONPITCH_FEATURES,
    normalise_team_name,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "models"

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


def load_models() -> dict:
    """Load serialised models and calibrators from disk."""
    models = {}
    for name in ["lgbm_latest", "xgb_latest", "lgbm_calibrator", "xgb_calibrator"]:
        path = MODEL_DIR / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
            logger.info(f"Loaded {name}")
        else:
            logger.warning(f"Model not found: {path}")
            models[name] = None
    return models


def fetch_upcoming_fixtures(league_id: int, days_ahead: int = 10) -> list[dict]:
    """Fetch upcoming (not started) fixtures from API-Football."""
    cfg = get_api_config()
    url = f"{cfg.API_FOOTBALL_BASE}/fixtures"

    today = date.today()
    end_date = today + timedelta(days=days_ahead)

    params = {
        "league": league_id,
        "season": today.year if today.month >= 8 else today.year - 1,
        "from": str(today),
        "to": str(end_date),
        "status": "NS",  # Not Started
    }

    headers = {
        "x-apisports-key": cfg.API_FOOTBALL_KEY,
        "x-apisports-host": "v3.football.api-sports.io",
    }

    logger.info(f"Fetching upcoming fixtures: {today} to {end_date}")

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers=headers, params=params)
        response.raise_for_status()

    fixtures = response.json().get("response", [])
    logger.info(f"Found {len(fixtures)} upcoming fixtures")
    return fixtures


def _build_rolling_stats(db, team: str, role: str, n: int = 8) -> dict:
    """
    Build rolling stats for a team from recent historical matches.
    role = 'home' or 'away' — which side of the fixture the team is on.
    """
    # Get recent finished matches for this team (either home or away)
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

    # Combine and sort
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
    home_xg = (
        db.table("match_xg")
        .select("home_xg, away_xg")
        .eq("home_team", normalise_team_name(team))
        .order("match_date", desc=True)
        .limit(n)
        .execute()
    ).data or []

    away_xg = (
        db.table("match_xg")
        .select("home_xg, away_xg")
        .eq("away_team", normalise_team_name(team))
        .order("match_date", desc=True)
        .limit(n)
        .execute()
    ).data or []

    xg_for = [float(m["home_xg"]) for m in home_xg] + [float(m["away_xg"]) for m in away_xg]
    xg_against = [float(m["away_xg"]) for m in home_xg] + [float(m["home_xg"]) for m in away_xg]

    # Sort by recency would need dates — just average all
    xg_for_avg = np.mean(xg_for) if xg_for else 1.3
    xg_against_avg = np.mean(xg_against) if xg_against else 1.1

    return {
        "xg_l8": round(xg_for_avg, 3),
        "xga_l8": round(xg_against_avg, 3),
        "xg_diff": round(xg_for_avg - xg_against_avg, 3),
    }


def build_feature_vector(fixture: dict, db) -> dict:
    """
    Build the full 46-feature vector for one upcoming fixture.
    Uses real rolling stats where available, stubs for the rest.
    """
    teams = fixture["teams"]
    home_team = teams["home"]["name"]
    away_team = teams["away"]["name"]

    # Normalise for xG lookup
    home_norm = normalise_team_name(home_team)
    away_norm = normalise_team_name(away_team)

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

    return features


def score_fixture(
    features: dict,
    models: dict,
) -> dict:
    """Score a fixture with both models, return probabilities."""
    onpitch_vec = np.array([[features.get(f, 0.0) for f in ONPITCH_FEATURES]])
    market_vec = np.array([[features.get(f, 0.0) for f in MARKET_FEATURES]])

    # LightGBM
    lgbm = models["lgbm_latest"]
    lgbm_raw = lgbm.predict(onpitch_vec)[0]
    # Apply calibrator if available and not None
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


def run_inference(days_ahead: int = 10) -> dict:
    """
    Full inference pipeline:
    1. Load models
    2. Fetch upcoming fixtures
    3. Score each fixture
    4. Write to upcoming_fixtures table
    5. Log cache run
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info(f"═══ APEX-10 Inference Engine ({APEX_ENV}) ═══")

    db = get_client()
    models = load_models()

    if models["lgbm_latest"] is None or models["xgb_latest"] is None:
        raise RuntimeError("Models not found — run training first: python -m apex10.models.train")

    league_id = LEAGUES.LEAGUE_IDS["EPL"]

    with CacheRunLogger(db) as cache_log:
        # Fetch upcoming fixtures
        raw_fixtures = fetch_upcoming_fixtures(league_id, days_ahead)
        cache_log.stats["fixtures_written"] = len(raw_fixtures)

        if not raw_fixtures:
            logger.warning("No upcoming fixtures found")
            return {"fixtures": 0, "scored": 0}

        scored = []
        for raw in raw_fixtures:
            try:
                fixture_info = raw["fixture"]
                teams = raw["teams"]
                league = raw["league"]

                # Build features from historical data
                features = build_feature_vector(raw, db)

                # Score with both models
                probs = score_fixture(features, models)

                # Determine best bet (home win for now — Phase 2: expand)
                implied_prob = 1.0 / features.get("odds_opening_home", 1.5)
                best_odds = features.get("odds_opening_home", 1.5)

                row = {
                    "api_match_id": fixture_info["id"],
                    "league": league["name"],
                    "match_date": fixture_info["date"][:10],
                    "home_team": teams["home"]["name"],
                    "away_team": teams["away"]["name"],
                    "lgbm_prob": probs["lgbm_prob"],
                    "xgb_prob": probs["xgb_prob"],
                    "consensus_prob": probs["consensus_prob"],
                    "best_bet_type": "Home Win",
                    "best_bet_odds": best_odds,
                    "opening_odds": best_odds,
                    "key_player_absent_home": 0,
                    "key_player_absent_away": 0,
                    "features_complete": True,
                }
                scored.append(row)
                logger.info(
                    f"  {teams['home']['name']} vs {teams['away']['name']}: "
                    f"LGBM={probs['lgbm_prob']:.3f} XGB={probs['xgb_prob']:.3f} "
                    f"consensus={probs['consensus_prob']:.3f}"
                )

            except Exception as e:
                logger.error(f"Failed to score fixture: {e}")
                cache_log.add_source_failure("inference", str(e))

        # Upsert to upcoming_fixtures
        if scored:
            db.table("upcoming_fixtures").upsert(
                scored, on_conflict="api_match_id"
            ).execute()
            logger.info(f"Wrote {len(scored)} scored fixtures to upcoming_fixtures")

    summary = {
        "fixtures_found": len(raw_fixtures),
        "scored": len(scored),
        "environment": APEX_ENV,
    }
    logger.info(f"═══ Inference complete ═══ {summary}")
    return summary


if __name__ == "__main__":
    run_inference()
