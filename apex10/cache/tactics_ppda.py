import logging
import soccerdata as sd
import pandas as pd
from typing import List, Dict

logger = logging.getLogger(__name__)

# Entity Resolution: Map Understat/soccerdata team names to APEX-10 generic DB names
TEAM_NAME_MAP = {
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle United": "Newcastle",
    "Sheffield United": "Sheffield Utd",
    "Leeds United": "Leeds",
    "West Ham United": "West Ham",
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Spurs": "Tottenham",
    "Leicester City": "Leicester",
    "Brighton and Hove Albion": "Brighton",
    "Ipswich Town": "Ipswich",
    "Luton Town": "Luton",
}

def fetch_rolling_ppda(league: str, season: int) -> List[Dict]:
    """
    Fetches match logs from Understat, calculates the L5 rolling PPDA per team,
    and returns a normalized list of dictionaries for DB insertion.
    """
    # Map APEX-10 generic league names to soccerdata expected formats
    sd_league_map = {
        "EPL": "ENG-Premier League",
        "La Liga": "ESP-La Liga",
        "Bundesliga": "GER-Bundesliga",
        "Serie A": "ITA-Serie A",
        "Ligue 1": "FRA-Ligue 1"
    }
    
    sd_league = sd_league_map.get(league)
    if not sd_league:
        logger.error(f"Unknown league mapping for {league}")
        return []

    # Map season to expected soccerdata format: 2025 -> "2425", 2024 -> "2324"
    season_str = f"{str(season-1)[-2:]}{str(season)[-2:]}"

    try:
        logger.info(f"Fetching L5 PPDA for {league} ({season_str}) via Understat...")
        understat = sd.Understat(leagues=sd_league, seasons=season_str)
        
        # Pull team match stats.
        match_stats = understat.read_team_match_stats()
        
        # Reset index to make filtering and grouping easier
        df = match_stats.reset_index()

        # df has 'date', 'home_team', 'away_team', 'home_ppda', 'away_ppda'
        # We need a continuous timeline of each team's PPDA, whether they were home or away
        home_side = df[['date', 'home_team', 'home_ppda']].rename(
            columns={'home_team': 'team', 'home_ppda': 'ppda'}
        )
        away_side = df[['date', 'away_team', 'away_ppda']].rename(
            columns={'away_team': 'team', 'away_ppda': 'ppda'}
        )
        
        # Merge them into one master event log!
        all_stats = pd.concat([home_side, away_side])
        
        # Ensure date sorting for accurate rolling windows
        all_stats['date'] = pd.to_datetime(all_stats['date'])
        all_stats = all_stats.sort_values(by=['team', 'date'])
        
        # Calculate the mean of the L5 games
        l5_ppda = all_stats.groupby('team')['ppda'].apply(lambda x: x.tail(5).mean()).reset_index()

        # Format records for APEX-10 Database
        records = []
        for _, row in l5_ppda.iterrows():
            raw_team_name = row['team']
            mapped_name = TEAM_NAME_MAP.get(raw_team_name, raw_team_name)
            
            # Avoid inserting NaN if a team has no data
            if pd.isna(row['ppda']):
                continue
                
            records.append({
                "team": mapped_name,
                "season": season,
                "league": league,
                "ppda": round(float(row['ppda']), 2)
            })

        logger.info(f"Successfully processed L5 PPDA for {len(records)} {league} teams.")
        return records

    except Exception as e:
        logger.error(f"Understat PPDA Fetch Failed for {league}: {e}")
        return []

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

def backfill_all_ppda(season: int = 2025):
    """Fetch and upsert L5 PPDA for all unsupported leagues."""
    from apex10.db import get_client
    db = get_client()
    leagues = ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
    
    total = 0
    for l in leagues:
        records = fetch_rolling_ppda(l, season)
        if records:
            inserted = upsert_ppda(records, db)
            logger.info(f"[{l}] Upserted {inserted} PPDA records.")
            total += inserted
        else:
            logger.warning(f"[{l}] Failed to fetch PPDA data.")
    return total

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    backfill_all_ppda()
