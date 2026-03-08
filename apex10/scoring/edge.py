"""
Edge calculation: model probability vs implied probability.
Both LightGBM and XGBoost must independently show >= 4% edge.
"""
from __future__ import annotations

import logging

from apex10.config import ODDS

logger = logging.getLogger(__name__)


def calculate_edge(model_prob: float, market_odds: float) -> float:
    """
    Edge = model_probability - implied_probability.
    market_odds is already vig-stripped (Power Method applied upstream).
    """
    if market_odds <= 1.0:
        return -1.0  # Invalid odds → always negative edge

    implied_prob = 1.0 / market_odds
    return round(model_prob - implied_prob, 4)


def dual_edge_check(
    lgbm_prob: float,
    xgb_prob: float,
    market_odds: float,
    min_edge: float = ODDS.MIN_EDGE,
) -> dict:
    """
    Gate 2: Both models must independently show >= min_edge.
    Returns result dict with pass/fail and individual edges.
    """
    lgbm_edge = calculate_edge(lgbm_prob, market_odds)
    xgb_edge = calculate_edge(xgb_prob, market_odds)

    passed = lgbm_edge >= min_edge and xgb_edge >= min_edge

    return {
        "passed": passed,
        "lgbm_edge": lgbm_edge,
        "xgb_edge": xgb_edge,
        "min_edge": min_edge,
        "market_odds": market_odds,
    }


def divergence_check(
    lgbm_prob: float,
    xgb_prob: float,
    max_divergence: float = ODDS.MAX_DIVERGENCE,
) -> dict:
    """
    Part of Gate 2: models must not diverge by more than max_divergence.
    Divergence > 10pp = ambiguous signal → drop.
    """
    divergence = abs(lgbm_prob - xgb_prob)
    passed = divergence <= max_divergence

    return {
        "passed": passed,
        "divergence": round(divergence, 4),
        "lgbm_prob": lgbm_prob,
        "xgb_prob": xgb_prob,
        "max_divergence": max_divergence,
    }


def consensus_probability(lgbm_prob: float, xgb_prob: float) -> float:
    """
    Simple average of both model probabilities.
    Used as the confidence score for ticket ranking.
    """
    return round((lgbm_prob + xgb_prob) / 2.0, 4)
