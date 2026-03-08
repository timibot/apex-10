"""
Fetches historical Elo ratings from ClubElo (clubelo.com).
ClubElo provides a free CSV API: http://api.clubelo.com/{team_name}
Returns full rating history per team — we store the rating closest to each match date.
"""
from __future__ import annotations

import io
import logging
from datetime import date

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

CLUBELO_BASE = "http://api.clubelo.com"

# ClubElo uses its own team name slugs (no spaces, specific casing)
EPL_TEAM_SLUGS = [
    "ManCity", "Liverpool", "Arsenal", "Chelsea", "Tottenham",
    "ManUnited", "Newcastle", "Brighton", "AstonVilla", "WestHam",
    "Brentford", "Fulham", "CrystalPalace", "Wolves", "Everton",
    "Nottingham", "Leicester", "Leeds", "Southampton", "Burnley",
    "Bournemouth", "Sheffield United", "Luton",
]


def fetch_team_elo(team_slug: str) -> pd.DataFrame | None:
    """
    Fetch full Elo rating history for a team.
    Returns DataFrame with columns: From, To, Club, Country, Level, Elo
    """
    url = f"{CLUBELO_BASE}/{team_slug}"
    logger.debug(f"Fetching ClubElo: {url}")

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning(f"ClubElo fetch failed for {team_slug}: {e}")
        return None

    try:
        df = pd.read_csv(io.StringIO(response.text))
        df["From"] = pd.to_datetime(df["From"])
        df["To"] = pd.to_datetime(df["To"])
        return df
    except Exception as e:
        logger.warning(f"ClubElo parse failed for {team_slug}: {e}")
        return None


def get_elo_on_date(team_slug: str, target_date: date, history_df: pd.DataFrame) -> float | None:
    """
    Return the Elo rating for a team on a specific date.
    Finds the row where target_date falls between From and To.
    """
    target = pd.Timestamp(target_date)
    mask = (history_df["From"] <= target) & (history_df["To"] >= target)
    matching = history_df[mask]

    if matching.empty:
        logger.debug(f"No Elo rating found for {team_slug} on {target_date}")
        return None

    return float(matching.iloc[-1]["Elo"])


def upsert_elo_ratings(records: list[dict], db_client) -> int:
    """Upsert Elo records into club_elo table."""
    if not records:
        return 0
    result = (
        db_client.table("club_elo")
        .upsert(records, on_conflict="team_slug,rating_date")
        .execute()
    )
    return len(result.data) if result.data else 0
