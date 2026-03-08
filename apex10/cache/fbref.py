"""
FBref PPDA fetcher via soccerdata.
soccerdata wraps FBref with built-in 3-second rate limit enforcement,
caching, and retry logic — no custom scraping needed.
Fallback: pandas.read_html() directly if soccerdata fails.
"""
from __future__ import annotations

import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)

# FBref mandatory delay — enforced by soccerdata, documented here
FBREF_RATE_LIMIT_SECONDS = 3

FBREF_LEAGUE_CODES = {
    "EPL": "ENG-Premier League",
    "La Liga": "ESP-La Liga",
    "Bundesliga": "GER-Bundesliga",
    "Serie A": "ITA-Serie A",
    "Ligue 1": "FRA-Ligue 1",
}


def fetch_ppda_soccerdata(
    league_name: str, season: int
) -> pd.DataFrame | None:
    """
    Fetch PPDA per team per season via soccerdata (primary method).
    Returns DataFrame with columns: team, ppda, season.
    Returns None on failure — caller falls back to direct method.
    """
    try:
        import soccerdata as sd

        league_code = FBREF_LEAGUE_CODES.get(league_name)
        if not league_code:
            logger.error(
                f"Unknown league for soccerdata: {league_name}"
            )
            return None

        fbref = sd.FBref(leagues=league_code, seasons=season)
        df = fbref.read_team_season_stats(stat_type="defense")

        if df is None or df.empty:
            return None

        # soccerdata returns multi-index — flatten
        df = df.reset_index()

        # PPDA column name varies — find it
        ppda_col = next(
            (c for c in df.columns if "ppda" in str(c).lower()),
            None,
        )
        if ppda_col is None:
            logger.warning(
                "PPDA column not found in FBref defense stats"
            )
            return None

        result = pd.DataFrame(
            {
                "team": df["team"]
                if "team" in df.columns
                else df.iloc[:, 0],
                "ppda": df[ppda_col].astype(float),
                "season": season,
                "league": league_name,
            }
        )

        logger.info(
            f"FBref PPDA (soccerdata): {len(result)} teams, "
            f"{league_name} {season}"
        )
        return result

    except ImportError:
        logger.warning(
            "soccerdata not installed — falling back to direct method"
        )
        return None
    except Exception as e:
        logger.warning(f"soccerdata FBref failed: {e} — falling back")
        return None


def fetch_ppda_direct(
    league_name: str, season: int
) -> pd.DataFrame | None:
    """
    Fallback: fetch PPDA directly from FBref via pandas.read_html().
    Respects mandatory 3-second delay between requests.
    """
    season_str = f"{season}-{str(season + 1)[-2:]}"
    league_slug = {
        "EPL": "9",
        "La Liga": "12",
        "Bundesliga": "20",
        "Serie A": "11",
        "Ligue 1": "13",
    }.get(league_name)

    if not league_slug:
        return None

    url = (
        f"https://fbref.com/en/comps/{league_slug}/{season_str}/"
        f"{season_str}-{league_name.replace(' ', '-')}-Stats"
    )

    try:
        time.sleep(FBREF_RATE_LIMIT_SECONDS)
        tables = pd.read_html(
            url, attrs={"id": "stats_squads_defense_for"}
        )
        if not tables:
            return None

        df = tables[0]
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(str(c) for c in col).strip()
                for col in df.columns
            ]

        ppda_col = next(
            (c for c in df.columns if "ppda" in c.lower()),
            None,
        )
        if ppda_col is None:
            logger.warning(
                f"PPDA column not found in FBref HTML: {url}"
            )
            return None

        squad_col = next(
            (c for c in df.columns if "squad" in c.lower()),
            df.columns[0],
        )

        result = pd.DataFrame(
            {
                "team": df[squad_col],
                "ppda": pd.to_numeric(df[ppda_col], errors="coerce"),
                "season": season,
                "league": league_name,
            }
        ).dropna(subset=["ppda"])

        logger.info(
            f"FBref PPDA (direct): {len(result)} teams, "
            f"{league_name} {season}"
        )
        return result

    except Exception as e:
        logger.error(f"FBref direct fetch failed: {e}")
        return None


def fetch_ppda(
    league_name: str, season: int
) -> pd.DataFrame | None:
    """
    Fetch PPDA with automatic fallback chain:
    1. soccerdata (primary — built-in rate limiting + retry)
    2. pandas.read_html() directly (fallback)
    3. None — caller uses stub value and logs to sources_failed
    """
    df = fetch_ppda_soccerdata(league_name, season)
    if df is not None and not df.empty:
        return df

    logger.warning(
        f"soccerdata failed for {league_name} {season} — "
        f"trying direct"
    )
    df = fetch_ppda_direct(league_name, season)
    if df is not None and not df.empty:
        return df

    logger.error(
        f"All PPDA fetch methods failed for "
        f"{league_name} {season}"
    )
    return None


def upsert_ppda(records: list[dict], db) -> int:
    """Upsert PPDA records into team_ppda table."""
    if not records:
        return 0
    result = (
        db.table("team_ppda")
        .upsert(records, on_conflict="team,season,league")
        .execute()
    )
    return len(result.data) if result.data else 0
