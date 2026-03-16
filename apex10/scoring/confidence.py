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
ODDS_MAX = 1.50


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


# Per-market Poisson probability thresholds
# Tuned to each market's natural probability range
POISSON_THRESHOLDS = {
    "home_win":   0.55,  # ~40-60% typical
    "away_win":   0.45,  # ~20-40% typical
    "dc_1x":      0.65,  # ~55-75% typical
    "dc_x2":      0.55,  # ~45-65% typical
    "dnb_home":   0.55,  # ~45-60% typical
    "dnb_away":   0.45,  # ~25-45% typical
    "over_1_5":   0.65,  # ~70-85% typical
    "over_2_5":   0.50,  # ~40-60% typical
    "under_3_5":  0.65,  # ~60-80% typical
    "btts_yes":   0.50,  # ~45-60% typical
    "btts_no":    0.50,  # ~40-55% typical
}


def _get_market_odds(
    market_key: str,
    match_odds: dict | None,
    dc_probs: dict | None = None,
) -> float | None:
    """
    Get bookmaker odds for a market.

    Priority:
    1. Direct odds from Odds API
    2. Derived from h2h odds (DC, DNB)
    3. Derived via vig-ratio from related market
       (Over 1.5, Under 3.5 from Over 2.5)
    """
    if not match_odds:
        return None

    # 1. Direct odds from Odds API
    DIRECT_MAP = {
        "home_win": "home",
        "away_win": "away",
        "over_1_5": "over_1_5",
        "over_2_5": "over_2_5",
        "under_2_5": "under_2_5",
        "under_3_5": "under_3_5",
    }
    if market_key in DIRECT_MAP and match_odds.get(DIRECT_MAP[market_key]):
        return match_odds[DIRECT_MAP[market_key]]

    # 2. Derived from h2h odds
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

    # 3. Vig-ratio derivation from CLOSEST available market
    #    Bookmaker vig = (bookmaker implied prob) / (model prob)
    #    Applied to target market: target_odds = 1 / (model_target_prob * vig)
    #
    #    Over 1.5  ← derive from Over 2.5  (both "over" markets)
    #    Under 3.5 ← derive from Under 2.5 (both "under" markets, closest line)
    #    Under 2.5 ← derive from Over 2.5  (complementary market)
    if dc_probs and market_key in ("over_1_5", "under_3_5", "under_2_5"):
        # Choose the closest reference market
        if market_key == "under_3_5":
            # Under 3.5 is closest to Under 2.5
            ref_odds = match_odds.get("under_2_5")
            ref_key = "under_2_5"
            # If no Under 2.5 available, try deriving from Over 2.5 complement
            if not ref_odds or ref_odds <= 1.0:
                ref_o25 = match_odds.get("over_2_5")
                if ref_o25 and ref_o25 > 1.0:
                    # under_2_5 implied = 1 - over_2_5 implied
                    u25_implied = 1.0 - (1.0 / ref_o25)
                    ref_odds = 1.0 / u25_implied if u25_implied > 0.05 else None
                    ref_key = "under_2_5"
        elif market_key == "over_1_5":
            # Over 1.5 is closest to Over 2.5
            ref_odds = match_odds.get("over_2_5")
            ref_key = "over_2_5"
        else:  # under_2_5
            # Complementary: derive from Over 2.5
            ref_odds = match_odds.get("over_2_5")
            ref_key = "over_2_5"

        model_ref = dc_probs.get(ref_key, 0)

        if ref_odds and ref_odds > 1.0 and model_ref > 0.05:
            bookie_implied = 1.0 / ref_odds
            raw_vig = bookie_implied / model_ref

            # Dampen vig when crossing goal lines (e.g. Over 2.5 → Over 1.5)
            # Bookmaker margins differ across lines, so raw vig is too aggressive
            # sqrt dampening: vig=1.15 → 1.07, vig=0.85 → 0.92
            if raw_vig > 1.0:
                vig_ratio = 1.0 + (raw_vig - 1.0) ** 0.5 * 0.5
            else:
                vig_ratio = 1.0 - (1.0 - raw_vig) ** 0.5 * 0.5

            target_model_prob = dc_probs.get(market_key, 0)
            if target_model_prob > 0.05:
                adjusted_prob = target_model_prob * vig_ratio
                adjusted_prob = min(max(adjusted_prob, 0.05), 0.98)
                derived_odds = round(1.0 / adjusted_prob, 2)
                if derived_odds > 1.0:
                    logger.debug(
                        f"Derived {market_key} odds: {derived_odds:.2f} "
                        f"(from {ref_key}, raw_vig={raw_vig:.3f}, "
                        f"dampened={vig_ratio:.3f})"
                    )
                    return derived_odds

    # BTTS — no reliable derivation yet
    if market_key in ("btts_yes", "btts_no"):
        return None

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

    # Signal 1: Dixon-Coles probability ≥ market-specific threshold
    threshold = POISSON_THRESHOLDS.get(market_key, 0.60)
    signals["poisson"] = dc_prob >= threshold

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

        odds = _get_market_odds(prob_key, match_odds, dc_probs)
        if odds is None or odds <= 1.0:
            # No real or derivable bookmaker odds — skip
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
    odds_range: tuple[float, float] = (1.10, 1.50),
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

    return None
