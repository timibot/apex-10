"""
Backfill current-season match results from TheSportsDB → Supabase `matches` table.

TheSportsDB is free, no API key needed. Returns finished matches with scores.
Run: python -m apex10.cache.backfill_results
"""
from __future__ import annotations

import logging
import time
from datetime import date

import httpx

from apex10.db import get_client

logger = logging.getLogger(__name__)

# TheSportsDB league IDs
LEAGUES = {
    "Premier League": 4328,
    "La Liga": 4335,
    "Bundesliga": 4331,
    "Serie A": 4332,
    "Ligue 1": 4334,
}

# TheSportsDB team names → our normalised convention
TEAM_MAP = {
    # England
    "Man United": "Manchester United", "Man City": "Manchester City",
    "Spurs": "Tottenham", "Tottenham Hotspur": "Tottenham",
    "Wolverhampton Wanderers": "Wolves", "Wolverhampton": "Wolves",
    "Newcastle United": "Newcastle", "Newcastle Utd": "Newcastle",
    "Nott'm Forest": "Nottingham Forest",
    "West Ham United": "West Ham", "Leicester City": "Leicester",
    "Brighton and Hove Albion": "Brighton",
    "Sheffield United": "Sheffield Utd",
    "Ipswich Town": "Ipswich", "Luton Town": "Luton",
    "AFC Bournemouth": "Bournemouth",
    # Spain
    "Athletic Bilbao": "Athletic Club", "Betis": "Real Betis",
    "CA Osasuna": "Osasuna",
    # Germany
    "Borussia Dortmund": "Dortmund",
    "Borussia Monchengladbach": "Borussia Mönchengladbach",
    "Borussia M'gladbach": "Borussia Mönchengladbach",
    "FC Augsburg": "FC Augsburg",
    "1. FC Heidenheim 1846": "FC Heidenheim",
    "FC St. Pauli": "St Pauli",
    "FSV Mainz 05": "Mainz", "1. FSV Mainz 05": "Mainz",
    "1. FC Köln": "FC Köln", "1. FC Koln": "FC Köln",
    "Hamburger SV": "Hamburg",
    "VfB Stuttgart": "Stuttgart",
    "VfL Wolfsburg": "Wolfsburg",
    "SC Freiburg": "Freiburg",
    "TSG Hoffenheim": "Hoffenheim",
    "SV Werder Bremen": "Werder Bremen",
    # Italy
    "Inter Milan": "Inter", "Internazionale": "Inter",
    "SSC Napoli": "Napoli", "ACF Fiorentina": "Fiorentina",
    "AS Roma": "Roma", "SS Lazio": "Lazio",
    "Atalanta BC": "Atalanta",
    # France
    "Paris Saint-Germain": "PSG", "Paris SG": "PSG",
    "AS Monaco": "Monaco",
    "Olympique Lyonnais": "Lyon", "Olympique de Marseille": "Marseille",
    "RC Lens": "Lens", "RC Strasbourg Alsace": "Strasbourg",
    "LOSC Lille": "Lille", "Stade Rennais": "Rennes",
    "Stade Brestois 29": "Brest", "FC Lorient": "Lorient",
    "OGC Nice": "Nice", "FC Nantes": "Nantes",
    "Le Havre AC": "Le Havre", "FC Metz": "Metz",
    "AJ Auxerre": "Auxerre", "Angers SCO": "Angers",
}


def _norm(name: str) -> str:
    return TEAM_MAP.get(name, name)


def _current_season() -> str:
    today = date.today()
    start_year = today.year if today.month >= 8 else today.year - 1
    return f"{start_year}-{start_year + 1}"


def backfill_league_results(
    league_name: str,
    league_id: int,
    season: str,
    client: httpx.Client,
    db,
) -> int:
    """
    Fetch all finished matches for a league + season from TheSportsDB.
    Upserts into the `matches` table. Returns count of upserted rows.
    """
    records = []
    max_rounds = 38 if league_name in ("Premier League", "La Liga") else 34

    for round_num in range(1, max_rounds + 1):
        url = (
            f"https://www.thesportsdb.com/api/v1/json/3/eventsround.php"
            f"?id={league_id}&r={round_num}&s={season}"
        )
        try:
            resp = client.get(url)
            if resp.status_code == 429:
                logger.warning(f"  R{round_num}: rate limited, waiting 10s...")
                time.sleep(10)
                resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"  R{round_num}: fetch failed ({e})")
            time.sleep(5)
            continue

        events = data.get("events") or []
        round_count = 0

        for ev in events:
            status = ev.get("strStatus", "")
            home_score = ev.get("intHomeScore")
            away_score = ev.get("intAwayScore")

            # Only store finished matches with valid scores
            if status != "Match Finished" or home_score is None or away_score is None:
                continue

            home = _norm(ev.get("strHomeTeam", ""))
            away = _norm(ev.get("strAwayTeam", ""))

            records.append({
                "api_match_id": int(ev.get("idEvent", 0)),
                "league": league_name,
                "season": int(season.split("-")[0]),
                "match_date": ev.get("dateEvent", ""),
                "home_team": home,
                "away_team": away,
                "home_goals": int(home_score),
                "away_goals": int(away_score),
                "status": "finished",
            })
            round_count += 1

        if round_count > 0:
            logger.debug(f"  R{round_num}: {round_count} finished matches")

        # Rate limiting — TheSportsDB free tier
        time.sleep(1.5)

    if not records:
        logger.warning(f"  No finished matches found for {league_name} {season}")
        return 0

    # Upsert in batches of 50
    total = 0
    for i in range(0, len(records), 50):
        batch = records[i:i + 50]
        try:
            db.table("matches").upsert(
                batch, on_conflict="api_match_id"
            ).execute()
            total += len(batch)
        except Exception as e:
            logger.error(f"  Upsert failed for batch {i}: {e}")

    logger.info(f"  {league_name}: {total} matches upserted for {season}")
    return total


def run_backfill(leagues: list[str] | None = None) -> dict:
    """
    Backfill current-season results for specified leagues.
    Returns summary dict.
    """
    season = _current_season()
    target = leagues or list(LEAGUES.keys())
    db = get_client()

    logger.info(f"═══ Backfilling {season} results from TheSportsDB ═══")

    summary = {"season": season, "leagues": {}}

    with httpx.Client(timeout=15.0) as client:
        for league_name in target:
            league_id = LEAGUES.get(league_name)
            if not league_id:
                logger.warning(f"Unknown league: {league_name}")
                continue

            logger.info(f"Fetching {league_name} ({season})...")
            count = backfill_league_results(
                league_name, league_id, season, client, db
            )
            summary["leagues"][league_name] = count

    total = sum(summary["leagues"].values())
    logger.info(f"═══ Backfill complete: {total} matches across {len(target)} leagues ═══")
    summary["total"] = total
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    result = run_backfill()
    print(f"\nDone: {result['total']} matches backfilled for {result['season']}")
    for league, count in result["leagues"].items():
        print(f"  {league}: {count}")
