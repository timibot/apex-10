"""
ClubElo integration — free API, no key needed.
Fetches current Elo ratings for all top clubs.
API: http://api.clubelo.com/YYYY-MM-DD → CSV
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import date

import httpx

logger = logging.getLogger(__name__)

CLUBELO_BASE = "http://api.clubelo.com"

# ClubElo name → our normalised name
ELO_TEAM_MAP = {
    # England
    "Arsenal": "Arsenal", "Man City": "Manchester City",
    "Liverpool": "Liverpool", "Chelsea": "Chelsea",
    "Man United": "Manchester United", "Newcastle": "Newcastle",
    "Aston Villa": "Aston Villa", "Tottenham": "Tottenham",
    "Brighton": "Brighton", "West Ham": "West Ham",
    "Bournemouth": "Bournemouth", "Brentford": "Brentford",
    "Crystal Palace": "Crystal Palace", "Fulham": "Fulham",
    "Everton": "Everton", "Wolves": "Wolves",
    "Forest": "Nottingham Forest", "Burnley": "Burnley",
    "Sunderland": "Sunderland", "Leeds": "Leeds United",
    # Spain
    "Barcelona": "Barcelona", "Real Madrid": "Real Madrid",
    "Atletico": "Atletico Madrid", "Bilbao": "Athletic Club",
    "Sociedad": "Real Sociedad", "Betis": "Real Betis",
    "Villarreal": "Villarreal", "Celta": "Celta Vigo",
    "Osasuna": "Osasuna", "Getafe": "Getafe",
    "Sevilla": "Sevilla", "Rayo Vallecano": "Rayo Vallecano",
    "Valencia": "Valencia", "Girona": "Girona",
    "Espanyol": "Espanyol", "Mallorca": "Mallorca",
    "Levante": "Levante",
    # Germany
    "Bayern": "Bayern Munich", "Dortmund": "Dortmund",
    "Leverkusen": "Bayer Leverkusen", "RB Leipzig": "RB Leipzig",
    "Stuttgart": "Stuttgart", "Frankfurt": "Eintracht Frankfurt",
    "Freiburg": "Freiburg", "Hoffenheim": "Hoffenheim",
    "Werder": "Werder Bremen", "Mainz": "Mainz",
    "Union Berlin": "Union Berlin", "Wolfsburg": "Wolfsburg",
    "M'gladbach": "Monchengladbach", "Augsburg": "Augsburg",
    "Koeln": "FC Koln",
    # Italy
    "Inter": "Inter", "Milan": "AC Milan",
    "Juventus": "Juventus", "Napoli": "Napoli",
    "Roma": "Roma", "Atalanta": "Atalanta",
    "Lazio": "Lazio", "Fiorentina": "Fiorentina",
    "Bologna": "Bologna", "Genoa": "Genoa",
    "Sassuolo": "Sassuolo", "Como": "Como",
    "Cagliari": "Cagliari", "Verona": "Hellas Verona",
    "Cremonese": "Cremonese",
    # France
    "Paris SG": "PSG", "Marseille": "Marseille",
    "Lyon": "Lyon", "Monaco": "Monaco",
    "Lille": "Lille", "Lens": "Lens",
    "Rennes": "Rennes", "Strasbourg": "Strasbourg",
    "Toulouse": "Toulouse", "Brest": "Brest",
    "Lorient": "Lorient", "Nice": "Nice",
}


def fetch_elo_ratings(for_date: date | None = None) -> dict[str, float]:
    """
    Fetch current Elo ratings from ClubElo.
    Returns dict mapping normalised team name → Elo rating.
    """
    d = for_date or date.today()
    url = f"{CLUBELO_BASE}/{d.isoformat()}"

    logger.info(f"Fetching ClubElo ratings for {d}")
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error(f"ClubElo fetch failed: {e}")
        return {}

    # Parse CSV: Rank,Club,Country,Level,Elo,From,To
    ratings = {}
    reader = csv.DictReader(io.StringIO(response.text))
    for row in reader:
        club = row.get("Club", "")
        try:
            elo = float(row.get("Elo", 0))
        except (ValueError, TypeError):
            continue

        # Map to our normalised name
        normalised = ELO_TEAM_MAP.get(club, club)
        ratings[normalised] = elo

    logger.info(f"Loaded {len(ratings)} Elo ratings")
    return ratings


def get_elo_diff(ratings: dict[str, float], home: str, away: str) -> float:
    """
    Calculate Elo difference (home - away).
    Returns 0.0 if either team is not found.
    """
    home_elo = ratings.get(home, 0)
    away_elo = ratings.get(away, 0)
    if home_elo == 0 or away_elo == 0:
        return 0.0
    return home_elo - away_elo
