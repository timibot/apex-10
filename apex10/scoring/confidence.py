"""
Confidence voting scorer — picks the best bet per fixture.

For each of 10+ markets, counts how many of 5 independent signals agree
the outcome is likely. Picks the market with the most votes, highest
probability, and highest odds as tiebreaker.

Signals:
  1. Dixon-Coles probability  ≥ 0.70
  2. Form alignment           favoured side has better recent form
  3. xG dominance             xG supports the market direction
  4. Elo advantage             Elo diff supports the pick
  5. Bookmaker agreement       implied prob from odds ≥ 0.60

Tiebreak: votes → probability → odds (descending).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Odds range for accumulator legs
ODDS_MIN = 1.10
ODDS_MAX = 1.60


@dataclass
class MarketPick:
    """One scored market for a fixture."""
    market: str
    probability: float
    odds: float
    votes: int
    signals: dict = field(default_factory=dict)

    @property
    def sort_key(self):
        """Sort by: votes desc, probability desc, odds desc."""
        return (self.votes, self.probability, self.odds)


# ── Market definitions ────────────────────────────────────────────────────
# Each market: (dc_prob_key, bet_label, direction)
#   direction: "home" = needs home team strong, "away" = away, "neutral" = goals-based
MARKETS = [
    ("home_win",   "Home Win",          "home"),
    ("away_win",   "Away Win",          "away"),
    ("dc_1x",      "Double Chance 1X",  "home"),
    ("dc_x2",      "Double Chance X2",  "away"),
    ("dnb_home",   "Draw No Bet",       "home"),
    ("dnb_away",   "DNB Away",          "away"),
    ("over_1_5",   "Over 1.5 Goals",    "neutral"),
    ("over_2_5",   "Over 2.5 Goals",    "neutral"),
    ("under_3_5",  "Under 3.5 Goals",   "neutral"),
    ("btts_yes",   "BTTS Yes",          "neutral"),
    ("btts_no",    "BTTS No",           "neutral"),
]


def _get_market_odds(
    market_key: str,
    match_odds: dict | None,
) -> float | None:
    """Get bookmaker odds for a market. Returns None if unavailable."""
    if not match_odds:
        return None

    # Direct odds from Odds API
    DIRECT_MAP = {
        "home_win": "home",
        "away_win": "away",
        "over_1_5": "over_1_5",
        "over_2_5": "over_2_5",
        "under_3_5": "under_3_5",
    }
    if market_key in DIRECT_MAP and match_odds.get(DIRECT_MAP[market_key]):
        return match_odds[DIRECT_MAP[market_key]]

    # Derived odds from h2h
    h = match_odds.get("home", 0)
    d = match_odds.get("draw", 0)
    a = match_odds.get("away", 0)

    if market_key == "dc_1x" and h > 1 and d > 1:
        implied = (1.0 / h) + (1.0 / d)
        return round(1.0 / implied, 3) if implied > 0 else None

    if market_key == "dc_x2" and a > 1 and d > 1:
        implied = (1.0 / a) + (1.0 / d)
        return round(1.0 / implied, 3) if implied > 0 else None

    if market_key == "dnb_home" and h > 1 and a > 1:
        p_home = 1.0 / h
        p_away = 1.0 / a
        dnb = p_home / (p_home + p_away)
        return round(1.0 / dnb, 3) if dnb > 0 else None

    if market_key == "dnb_away" and h > 1 and a > 1:
        p_home = 1.0 / h
        p_away = 1.0 / a
        dnb = p_away / (p_home + p_away)
        return round(1.0 / dnb, 3) if dnb > 0 else None

    # BTTS — derive from totals if available, else skip
    if market_key in ("btts_yes", "btts_no"):
        return None  # No direct BTTS odds from API yet

    return None


def _vote_signals(
    market_key: str,
    direction: str,
    dc_prob: float,
    odds: float,
    features: dict,
    elo_diff: float,
) -> dict[str, bool]:
    """
    Count how many of 5 independent signals agree this market is likely.
    Returns dict of signal_name → True/False.
    """
    signals = {}

    # Signal 1: Dixon-Coles probability ≥ 0.70
    signals["poisson"] = dc_prob >= 0.70

    # Signal 2: Form alignment
    home_form = features.get("form_pts_home_l5", 0)
    away_form = features.get("form_pts_away_l5", 0)
    if direction == "home":
        signals["form"] = home_form > away_form
    elif direction == "away":
        signals["form"] = away_form > home_form
    else:
        # Neutral (goals markets) — both teams have some scoring form
        home_gd = features.get("form_gd_home_l5", 0)
        away_gd = features.get("form_gd_away_l5", 0)
        if market_key in ("over_1_5", "over_2_5", "btts_yes"):
            # High scoring form supports overs/btts
            signals["form"] = (home_gd + away_gd) > 0
        else:
            # Low scoring or defensive form
            home_goals_avg = features.get("goals_scored_avg_home", 1.3)
            away_goals_conceded = features.get("goals_conceded_avg_away", 1.3)
            signals["form"] = (home_goals_avg + away_goals_conceded) < 3.0

    # Signal 3: xG dominance
    home_xg = features.get("xg_home_l8", 1.3)
    away_xg = features.get("xg_away_l8", 1.1)
    home_xga = features.get("xga_home_l8", 1.1)
    away_xga = features.get("xga_away_l8", 1.3)

    if direction == "home":
        signals["xg"] = home_xg > away_xg  # Home team creates more
    elif direction == "away":
        signals["xg"] = away_xg > home_xg  # Away team creates more
    else:
        total_xg = home_xg + away_xg
        if market_key in ("over_1_5", "over_2_5", "btts_yes"):
            signals["xg"] = total_xg > 2.3  # High combined xG
        elif market_key == "under_3_5":
            signals["xg"] = total_xg < 3.0  # Low combined xG
        else:
            signals["xg"] = total_xg < 2.5  # Defensive

    # Signal 4: Elo advantage
    if direction == "home":
        signals["elo"] = elo_diff > 50  # Home team rated higher
    elif direction == "away":
        signals["elo"] = elo_diff < -50  # Away team rated higher
    else:
        # Neutral — Elo doesn't strongly predict goals markets, mild check
        signals["elo"] = abs(elo_diff) < 300  # Not a massive mismatch

    # Signal 5: Bookmaker agreement (implied prob ≥ 0.60)
    if odds > 1.0:
        implied_prob = 1.0 / odds
        signals["bookmaker"] = implied_prob >= 0.60
    else:
        signals["bookmaker"] = False

    return signals


def score_all_markets(
    dc_probs: dict,
    features: dict,
    match_odds: dict | None,
    elo_diff: float = 0.0,
) -> list[MarketPick]:
    """
    Score all markets for a fixture and return sorted list of picks.

    Returns list sorted by (votes desc, probability desc, odds desc).
    Only includes markets with valid odds.
    """
    picks = []

    for prob_key, label, direction in MARKETS:
        dc_prob = dc_probs.get(prob_key, 0)
        if dc_prob < 0.01:
            continue

        odds = _get_market_odds(prob_key, match_odds)
        if odds is None or odds <= 1.0:
            # No bookmaker odds — derive fair odds from model
            odds = round(1.0 / dc_prob, 3) if dc_prob > 0 else None
            if odds is None:
                continue

        signals = _vote_signals(prob_key, direction, dc_prob, odds, features, elo_diff)
        votes = sum(1 for v in signals.values() if v)

        picks.append(MarketPick(
            market=label,
            probability=round(dc_prob, 4),
            odds=round(odds, 3),
            votes=votes,
            signals=signals,
        ))

    # Sort: votes desc → probability desc → odds desc
    picks.sort(key=lambda p: p.sort_key, reverse=True)
    return picks


def pick_best_bet(
    dc_probs: dict,
    features: dict,
    match_odds: dict | None,
    elo_diff: float = 0.0,
    min_votes: int = 3,
    odds_range: tuple[float, float] = (1.10, 1.60),
) -> MarketPick | None:
    """
    Pick the single best bet for a fixture.

    1. Score all markets
    2. Filter by min_votes and odds range
    3. Return top pick (highest votes → highest prob → highest odds)
    4. Returns None if no market qualifies
    """
    all_picks = score_all_markets(dc_probs, features, match_odds, elo_diff)

    for pick in all_picks:
        if pick.votes >= min_votes and odds_range[0] <= pick.odds <= odds_range[1]:
            return pick

    # Fallback: return best pick regardless of odds range if it has enough votes
    for pick in all_picks:
        if pick.votes >= min_votes:
            return pick

    return None
