"""
Builds the two orthogonal feature matrices from Supabase data.
LightGBM → 29 on-pitch features
XGBoost  → 17 market/context features

All feature engineering lives here. Models receive clean numpy arrays.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from apex10.db import get_client

logger = logging.getLogger(__name__)

# ── Feature split (orthogonal subspacing) ──────────────────────────────────
ONPITCH_FEATURES = [
    "xg_home_l8", "xga_home_l8", "xg_away_l8", "xga_away_l8",
    "xg_diff_home", "xg_diff_away",
    "ppda_home", "ppda_away", "ppda_delta",
    "form_pts_home_l5", "form_pts_away_l5",
    "form_gd_home_l5", "form_gd_away_l5",
    "home_advantage", "h2h_win_rate_home",
    "injury_count_home", "injury_count_away",
    "key_player_absent_home", "key_player_absent_away",
    "clean_sheet_rate_home", "clean_sheet_rate_away",
    "goals_scored_avg_home", "goals_conceded_avg_away",
    "weather_rain_mm", "weather_wind_kmh",
    "rivalry_index",
    "playstyle_counter_attack", "opponent_low_block",
    "travel_hours_away",
]  # 29 features

MARKET_FEATURES = [
    "odds_opening_home", "odds_current_home", "odds_movement",
    "fixture_congestion_home", "fixture_congestion_away",
    "days_to_next_home", "days_to_next_away",
    "sandwich_score_home", "sandwich_score_away",
    "competition_weight_home", "next_competition_weight_home",
    "manager_days_in_post", "manager_days_in_post_away",
    "motivation_asymmetry",
    "relegation_battle_away", "title_race_home",
    "elo_diff",  # GP-7: ClubElo home rating minus away rating
]  # 17 features

ALL_FEATURES = ONPITCH_FEATURES + MARKET_FEATURES  # 46 total

# Target: 1 = home win, 0 = draw or away win
TARGET_COL = "home_win"


def load_raw_features(league: str = "EPL") -> pd.DataFrame:
    """
    Load and join all feature data from Supabase into a single DataFrame.
    Each row = one match. Columns = all 46 features + target + metadata.
    """
    db = get_client()

    # Load matches
    matches_resp = (
        db.table("matches")
        .select("*")
        .eq("league", "Premier League")
        .eq("status", "finished")
        .order("match_date")
        .execute()
    )
    matches_df = pd.DataFrame(matches_resp.data)

    if matches_df.empty:
        raise ValueError("No match data found in Supabase")

    # Load xG
    xg_resp = db.table("match_xg").select("*").execute()
    xg_df = pd.DataFrame(xg_resp.data)

    # Load historical odds
    odds_resp = (
        db.table("historical_odds")
        .select("*")
        .eq("market", "1X2")
        .execute()
    )
    odds_df = pd.DataFrame(odds_resp.data)

    # Merge
    df = _merge_and_engineer(matches_df, xg_df, odds_df)
    logger.info(f"Feature matrix shape: {df.shape}")
    return df


def _merge_and_engineer(
    matches: pd.DataFrame,
    xg: pd.DataFrame,
    odds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge data sources and engineer all 46 features.
    Missing values are imputed with column medians — never dropped.
    """
    df = matches.copy()
    df["match_date"] = pd.to_datetime(df["match_date"])
    df["season"] = df["match_date"].dt.year

    # ── Target variable ────────────────────────────────────────────────────
    df[TARGET_COL] = (df["home_goals"] > df["away_goals"]).astype(int)

    # ── Merge xG ──────────────────────────────────────────────────────────
    if not xg.empty:
        xg["match_date"] = pd.to_datetime(xg["match_date"])
        df = df.merge(
            xg[["match_date", "home_team", "away_team", "home_xg", "away_xg"]],
            on=["match_date", "home_team", "away_team"],
            how="left",
        )
    else:
        df["home_xg"] = np.nan
        df["away_xg"] = np.nan

    # ── Merge odds ─────────────────────────────────────────────────────────
    if not odds.empty:
        df = df.merge(
            odds[["home_team", "away_team", "match_date",
                  "odds_home", "odds_draw", "odds_away"]],
            on=["home_team", "away_team"],
            how="left",
        )
        df.rename(columns={
            "odds_home": "odds_opening_home",
            "odds_away": "odds_opening_away",
        }, inplace=True)
        df["odds_current_home"] = df["odds_opening_home"]  # Same for historical
        df["odds_movement"] = 0.0  # No movement data for historical
    else:
        df["odds_opening_home"] = np.nan
        df["odds_current_home"] = np.nan
        df["odds_movement"] = 0.0

    # ── Rolling features (computed per-team over match history) ────────────
    df = _add_rolling_features(df)

    # ── Derived features ───────────────────────────────────────────────────
    df["xg_diff_home"] = df.get("home_xg", 0) - df.get("away_xg", 0)
    df["xg_diff_away"] = df.get("away_xg", 0) - df.get("home_xg", 0)
    df["xg_home_l8"] = df.get("home_xg", np.nan)
    df["xga_home_l8"] = df.get("away_xg", np.nan)
    df["xg_away_l8"] = df.get("away_xg", np.nan)
    df["xga_away_l8"] = df.get("home_xg", np.nan)

    # ── Stub features (populated by cache.py in Phase 2 extensions) ────────
    # These default to neutral/zero values where live data not yet available
    stub_defaults = {
        "ppda_home": 10.0, "ppda_away": 10.0, "ppda_delta": 0.0,
        "home_advantage": 0.5, "h2h_win_rate_home": 0.5,
        "injury_count_home": 0, "injury_count_away": 0,
        "key_player_absent_home": 0, "key_player_absent_away": 0,
        "clean_sheet_rate_home": 0.3, "clean_sheet_rate_away": 0.3,
        "goals_scored_avg_home": 1.4, "goals_conceded_avg_away": 1.4,
        "weather_rain_mm": 0.0, "weather_wind_kmh": 10.0,
        "rivalry_index": 50.0,
        "playstyle_counter_attack": 0, "opponent_low_block": 0,
        "travel_hours_away": 1.0,
        "fixture_congestion_home": 7, "fixture_congestion_away": 7,
        "days_to_next_home": 7, "days_to_next_away": 7,
        "sandwich_score_home": 0.0, "sandwich_score_away": 0.0,
        "competition_weight_home": 0.7, "next_competition_weight_home": 0.7,
        "manager_days_in_post": 365, "manager_days_in_post_away": 365,
        "motivation_asymmetry": 0.0,
        "relegation_battle_away": 0, "title_race_home": 0,
        "elo_diff": 0.0,  # GP-7: neutral stub
    }
    for col, default in stub_defaults.items():
        if col not in df.columns:
            df[col] = default

    # ── Impute remaining NaNs with column medians ──────────────────────────
    for col in ALL_FEATURES:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling form features per team."""
    df = df.sort_values("match_date").copy()

    for team_col, prefix, goals_for, goals_against in [
        ("home_team", "home", "home_goals", "away_goals"),
        ("away_team", "away", "away_goals", "home_goals"),
    ]:
        teams = df[team_col].unique()
        pts_col = f"form_pts_{prefix}_l5"
        gd_col = f"form_gd_{prefix}_l5"
        goals_avg_col = f"goals_scored_avg_{prefix}"

        df[pts_col] = np.nan
        df[gd_col] = np.nan
        df[goals_avg_col] = np.nan

        for team in teams:
            mask = df[team_col] == team
            team_idx = df[mask].index

            gf = df.loc[mask, goals_for].values
            ga = df.loc[mask, goals_against].values

            pts = np.where(gf > ga, 3, np.where(gf == ga, 1, 0))
            gd = gf - ga

            rolling_pts = pd.Series(pts).shift(1).rolling(5, min_periods=1).sum().values
            rolling_gd = pd.Series(gd).shift(1).rolling(5, min_periods=1).sum().values
            rolling_avg = (
                pd.Series(gf).shift(1).rolling(10, min_periods=1).mean().values
            )

            df.loc[team_idx, pts_col] = rolling_pts
            df.loc[team_idx, gd_col] = rolling_gd
            df.loc[team_idx, goals_avg_col] = rolling_avg

    return df


def get_feature_matrices(df: pd.DataFrame) -> tuple:
    """
    Returns (X_onpitch, X_market, y, seasons)
    Ready for walk-forward validation.
    """
    # Fill any remaining NaN with 0 as final safety net
    X_onpitch = df[ONPITCH_FEATURES].fillna(0).values.astype(np.float32)
    X_market = df[MARKET_FEATURES].fillna(0).values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)
    seasons = df["season"].values

    return X_onpitch, X_market, y, seasons
