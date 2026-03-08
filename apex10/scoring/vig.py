"""
Power Method vig removal.
Solves for exponent k where sum(implied_prob_i ^ k) = 1.0.
Correctly models favourite-longshot bias — bookmakers apply less margin
to heavy favourites, so linear removal understates true favourite probability.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


def _power_sum(k: float, raw_probs: np.ndarray) -> float:
    """Objective function: sum(p_i ^ k) - 1.0. Root = correct k."""
    return float(np.sum(raw_probs**k) - 1.0)


def remove_vig_power(odds: list[float]) -> list[float]:
    """
    Remove vig from a set of odds using the Power Method.

    Args:
        odds: Raw bookmaker odds for all outcomes (e.g. [1.30, 4.50, 9.00])

    Returns:
        True (vig-free) probabilities that sum to 1.0.

    Raises:
        ValueError: If odds list is empty or contains invalid values.
    """
    if not odds or any(o <= 1.0 for o in odds):
        raise ValueError(f"Invalid odds: {odds}. All odds must be > 1.0")

    raw_probs = np.array([1.0 / o for o in odds])
    overround = float(raw_probs.sum())

    if overround <= 1.0:
        logger.warning("Overround <= 1.0 — returning raw implied probs")
        return list(raw_probs)

    try:
        # k > 1 when overround > 1 (standard bookmaker margin)
        k = brentq(_power_sum, 0.5, 3.0, args=(raw_probs,), xtol=1e-8)
    except ValueError:
        # Fallback to linear if solver fails
        logger.warning("Power Method solver failed — falling back to linear")
        return list(raw_probs / overround)

    true_probs = raw_probs**k
    # Normalise to correct any floating point drift
    true_probs = true_probs / true_probs.sum()

    logger.debug(f"Vig removed: overround={overround:.4f}, k={k:.4f}")
    return list(true_probs)


def remove_vig_linear(odds: list[float]) -> list[float]:
    """
    Linear vig removal (divide by overround).
    Kept for unit test comparison — confirms Power Method diverges on favourites.
    """
    raw_probs = np.array([1.0 / o for o in odds])
    return list(raw_probs / raw_probs.sum())


def get_true_home_prob(home_odds: float, draw_odds: float, away_odds: float) -> float:
    """Convenience: return vig-free home win probability from 1X2 odds."""
    probs = remove_vig_power([home_odds, draw_odds, away_odds])
    return probs[0]
