"""
The Odds API integration — live pre-match odds for upcoming fixtures.
API: https://api.the-odds-api.com/v4/sports/{sport}/odds
Free tier: 500 credits/month
"""
from __future__ import annotations

import logging
from datetime import date

import httpx

from apex10.config import get_api_config

logger = logging.getLogger(__name__)

# Our league names → The Odds API sport keys
ODDS_SPORT_KEYS = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Bundesliga": "soccer_germany_bundesliga",
    "Serie A": "soccer_italy_serie_a",
    "Ligue 1": "soccer_france_ligue_one",
}

# The Odds API team names → our normalised names
ODDS_TEAM_MAP = {
    # EPL
    "Arsenal": "Arsenal", "Manchester City": "Manchester City",
    "Liverpool": "Liverpool", "Chelsea": "Chelsea",
    "Manchester United": "Manchester United", "Newcastle United": "Newcastle",
    "Aston Villa": "Aston Villa", "Tottenham Hotspur": "Tottenham",
    "Brighton and Hove Albion": "Brighton", "West Ham United": "West Ham",
    "AFC Bournemouth": "Bournemouth", "Brentford": "Brentford",
    "Crystal Palace": "Crystal Palace", "Fulham": "Fulham",
    "Everton": "Everton", "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nottingham Forest", "Burnley": "Burnley",
    "Leeds United": "Leeds United", "Sunderland": "Sunderland",
    # La Liga
    "Barcelona": "Barcelona", "Real Madrid": "Real Madrid",
    "Atletico Madrid": "Atletico Madrid", "Athletic Bilbao": "Athletic Club",
    "Real Sociedad": "Real Sociedad", "Real Betis": "Real Betis",
    "Villarreal": "Villarreal", "Celta Vigo": "Celta Vigo",
    "Osasuna": "Osasuna", "CA Osasuna": "Osasuna",
    "Getafe": "Getafe", "Sevilla FC": "Sevilla", "Sevilla": "Sevilla",
    "Rayo Vallecano": "Rayo Vallecano", "Valencia": "Valencia",
    "Girona": "Girona", "Espanyol": "Espanyol", "Mallorca": "Mallorca",
    "Levante": "Levante", "Elche CF": "Elche", "Oviedo": "Oviedo",
    "Deportivo Alavés": "Alaves", "Alavés": "Alaves",
    # Bundesliga
    "Bayern Munich": "Bayern Munich", "Borussia Dortmund": "Dortmund",
    "Bayer Leverkusen": "Bayer Leverkusen", "RB Leipzig": "RB Leipzig",
    "VfB Stuttgart": "Stuttgart", "Eintracht Frankfurt": "Eintracht Frankfurt",
    "SC Freiburg": "Freiburg", "TSG Hoffenheim": "Hoffenheim",
    "SV Werder Bremen": "Werder Bremen", "Werder Bremen": "Werder Bremen",
    "1. FSV Mainz 05": "Mainz", "FSV Mainz 05": "Mainz",
    "Union Berlin": "Union Berlin",
    "VfL Wolfsburg": "Wolfsburg", "Wolfsburg": "Wolfsburg",
    "Borussia Monchengladbach": "Borussia Mönchengladbach",
    "Monchengladbach": "Borussia Mönchengladbach",
    "FC Augsburg": "FC Augsburg", "Augsburg": "FC Augsburg",
    "1. FC Koln": "FC Köln", "1. FC Köln": "FC Köln",
    "Hamburger SV": "Hamburg",
    "1. FC Heidenheim": "FC Heidenheim", "1. FC Heidenheim 1846": "FC Heidenheim",
    "FC St. Pauli": "St Pauli",
    # Serie A
    "Inter Milan": "Inter", "AC Milan": "AC Milan",
    "Juventus": "Juventus", "SSC Napoli": "Napoli",
    "AS Roma": "Roma", "Atalanta BC": "Atalanta",
    "SS Lazio": "Lazio", "ACF Fiorentina": "Fiorentina",
    "Bologna": "Bologna", "Genoa": "Genoa",
    "Sassuolo": "Sassuolo", "Como": "Como",
    "Cagliari": "Cagliari", "Hellas Verona": "Hellas Verona",
    "Cremonese": "Cremonese", "Lecce": "Lecce",
    "Udinese": "Udinese", "Parma": "Parma", "Pisa": "Pisa",
    # Ligue 1
    "Paris Saint Germain": "PSG", "Olympique Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon", "AS Monaco": "Monaco",
    "LOSC Lille": "Lille", "RC Lens": "Lens",
    "Stade Rennais": "Rennes", "RC Strasbourg Alsace": "Strasbourg",
    "Toulouse": "Toulouse", "Stade Brestois 29": "Brest",
    "FC Lorient": "Lorient", "OGC Nice": "Nice",
    "FC Nantes": "Nantes", "Le Havre AC": "Le Havre",
    "FC Metz": "Metz", "AJ Auxerre": "Auxerre", "Auxerre": "Auxerre",
    "Angers": "Angers", "Angers SCO": "Angers",
    "Paris FC": "Paris FC",
}


def _normalise(name: str) -> str:
    """Normalise odds API team name to our convention."""
    return ODDS_TEAM_MAP.get(name, name)


def fetch_live_odds(leagues: list[str] | None = None) -> dict[str, dict]:
    """
    Fetch live pre-match h2h odds for all specified leagues.
    Returns dict: { "HomeTeam vs AwayTeam": {"home": float, "draw": float, "away": float} }
    Each call costs 1 credit per sport key on The Odds API.
    """
    cfg = get_api_config()
    all_odds = {}

    target_leagues = leagues or list(ODDS_SPORT_KEYS.keys())

    for league in target_leagues:
        sport_key = ODDS_SPORT_KEYS.get(league)
        if not sport_key:
            continue

        url = f"{cfg.ODDS_API_BASE}/sports/{sport_key}/odds"
        params = {
            "apiKey": cfg.ODDS_API_KEY,
            "regions": "uk",
            "markets": "h2h,totals",
            "oddsFormat": "decimal",
        }

        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()

            data = response.json()

            # Check remaining credits
            remaining = response.headers.get("x-requests-remaining", "?")
            used = response.headers.get("x-requests-used", "?")
            logger.info(f"Odds API: {league} ({sport_key}) — {len(data)} events, credits: {remaining} remaining / {used} used")

            for event in data:
                home = _normalise(event.get("home_team", ""))
                away = _normalise(event.get("away_team", ""))
                match_key = f"{home} vs {away}"

                # Get best odds from all bookmakers
                best_home = 1.0
                best_draw = 1.0
                best_away = 1.0
                # Over/Under totals
                best_over_1_5 = 1.0
                best_over_2_5 = 1.0
                best_under_2_5 = 1.0
                best_under_3_5 = 1.0
                # BTTS
                best_btts_yes = 1.0
                best_btts_no = 1.0

                for bookmaker in event.get("bookmakers", []):
                    for market in bookmaker.get("markets", []):
                        mkey = market.get("key", "")
                        outcomes = {o["name"]: o.get("price", 0) for o in market.get("outcomes", [])}

                        if mkey == "h2h":
                            if home_price := outcomes.get(event.get("home_team", "")):
                                best_home = max(best_home, home_price)
                            if draw_price := outcomes.get("Draw"):
                                best_draw = max(best_draw, draw_price)
                            if away_price := outcomes.get(event.get("away_team", "")):
                                best_away = max(best_away, away_price)

                        elif mkey == "totals":
                            point = market.get("outcomes", [{}])[0].get("point", 0)
                            over_price = outcomes.get("Over", 0)
                            under_price = outcomes.get("Under", 0)
                            if point == 1.5:
                                if over_price > best_over_1_5:
                                    best_over_1_5 = over_price
                            elif point == 2.5:
                                if over_price > best_over_2_5:
                                    best_over_2_5 = over_price
                                if under_price > best_under_2_5:
                                    best_under_2_5 = under_price
                            elif point == 3.5:
                                if under_price > best_under_3_5:
                                    best_under_3_5 = under_price
                                    
                        elif mkey == "btts":
                            if yes_price := outcomes.get("Yes"):
                                best_btts_yes = max(best_btts_yes, yes_price)
                            if no_price := outcomes.get("No"):
                                best_btts_no = max(best_btts_no, no_price)

                if best_home > 1.0:  # Only store if we actually got odds
                    all_odds[match_key] = {
                        "home": best_home,
                        "draw": best_draw,
                        "away": best_away,
                        "over_1_5": best_over_1_5 if best_over_1_5 > 1.0 else None,
                        "over_2_5": best_over_2_5 if best_over_2_5 > 1.0 else None,
                        "under_2_5": best_under_2_5 if best_under_2_5 > 1.0 else None,
                        "under_3_5": best_under_3_5 if best_under_3_5 > 1.0 else None,
                        "btts_yes": best_btts_yes if best_btts_yes > 1.0 else None,
                        "btts_no": best_btts_no if best_btts_no > 1.0 else None,
                    }

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error(f"Odds API: Invalid or Exhausted API key (401)")
                from apex10.live import notify
                notify.api_exhausted("Odds")
            elif e.response.status_code == 422:
                logger.warning(f"Odds API: {league} not in season")
            elif e.response.status_code == 429:
                logger.error(f"Odds API: Rate limited or exhausted (429)")
                from apex10.live import notify
                notify.api_exhausted("Odds")
            else:
                logger.error(f"Odds API error for {league}: {e}")
        except Exception as e:
            logger.error(f"Odds API fetch failed for {league}: {e}")

    logger.info(f"Fetched odds for {len(all_odds)} matches across {len(target_leagues)} leagues")
    return all_odds


def get_match_odds(all_odds: dict, home: str, away: str) -> dict:
    """
    Look up odds for a specific match.
    Returns {"home": float, "draw": float, "away": float} or None.
    """
    key = f"{home} vs {away}"
    return all_odds.get(key)
