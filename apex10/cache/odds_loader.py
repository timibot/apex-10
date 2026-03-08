"""
Loads historical odds from football-data.co.uk CSV files.
URL pattern: https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv
e.g. https://www.football-data.co.uk/mmz4281/2324/E0.csv  (EPL 2023/24)
"""
from __future__ import annotations

import io
import logging

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

# football-data.co.uk league codes
LEAGUE_CODES = {
    "EPL": "E0",
    "La Liga": "SP1",
    "Bundesliga": "D1",
    "Serie A": "I1",
    "Ligue 1": "F1",
}

# Season string format: 2019/20 → "1920", 2023/24 → "2324"
SEASONS = {
    2019: "1920",
    2020: "2021",
    2021: "2122",
    2022: "2223",
    2023: "2324",
}

# Pinnacle columns in the CSV (best sharp odds available)
PINNACLE_COLS = {
    "home": "PSH",   # Pinnacle start home
    "draw": "PSD",
    "away": "PSA",
}

BASE_URL = "https://www.football-data.co.uk/mmz4281"


def build_csv_url(league_name: str, season_year: int) -> str:
    code = LEAGUE_CODES[league_name]
    season_str = SEASONS[season_year]
    return f"{BASE_URL}/{season_str}/{code}.csv"


def fetch_odds_csv(league_name: str, season_year: int) -> pd.DataFrame | None:
    """
    Download and parse odds CSV for a league/season.
    Returns DataFrame or None if fetch fails.
    """
    url = build_csv_url(league_name, season_year)
    logger.info(f"Fetching odds CSV: {url}")

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None

    try:
        df = pd.read_csv(io.StringIO(response.text), encoding="latin-1")
    except Exception as e:
        logger.error(f"Failed to parse CSV from {url}: {e}")
        return None

    logger.info(f"Loaded {len(df)} rows from {url}")
    return df


def parse_odds_rows(df: pd.DataFrame, league_name: str, season_year: int) -> list[dict]:
    """
    Extract relevant odds columns and return list of dicts for DB insertion.
    Gracefully skips rows with missing Pinnacle odds.
    """
    records = []

    for _, row in df.iterrows():
        # Skip rows missing essential Pinnacle odds
        if any(pd.isna(row.get(col)) for col in PINNACLE_COLS.values()):
            continue

        try:
            record = {
                "league": league_name,
                "season": season_year,
                "home_team": str(row.get("HomeTeam", "")).strip(),
                "away_team": str(row.get("AwayTeam", "")).strip(),
                "match_date": str(row.get("Date", "")).strip(),
                "odds_home": float(row[PINNACLE_COLS["home"]]),
                "odds_draw": float(row[PINNACLE_COLS["draw"]]),
                "odds_away": float(row[PINNACLE_COLS["away"]]),
                "bookmaker": "Pinnacle",
                "market": "1X2",
            }
            records.append(record)
        except (ValueError, KeyError) as e:
            logger.warning(f"Skipping row: {e}")
            continue

    return records


def upsert_historical_odds(records: list[dict], db_client) -> int:
    """
    Upsert odds records into historical_odds table.
    Note: No unique key on this table — check for duplicates before calling
    in production. Safe for initial backfill.
    """
    if not records:
        return 0

    result = db_client.table("historical_odds").insert(records).execute()
    count = len(result.data) if result.data else 0
    logger.info(f"Inserted {count} odds records")
    return count
