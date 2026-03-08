"""
Dixon-Coles bivariate Poisson goal model.
Derives probabilities for all bet types from xG rates + rho correction.
All probabilities used downstream for edge calculation.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.stats import poisson

from apex10.cache.rho import tau

logger = logging.getLogger(__name__)

# Maximum scoreline to sum over — beyond this Poisson mass is negligible
MAX_GOALS = 10


def _build_score_matrix(
    mu: float,
    nu: float,
    rho: float,
    max_goals: int = MAX_GOALS,
) -> np.ndarray:
    """
    Build (max_goals x max_goals) joint probability matrix P(home=i, away=j).
    Applies Dixon-Coles tau correction to low-scoring cells only.
    All values guaranteed non-negative.
    """
    matrix = np.zeros((max_goals, max_goals))

    for i in range(max_goals):
        for j in range(max_goals):
            p_i = poisson.pmf(i, mu)
            p_j = poisson.pmf(j, nu)
            t = tau(i, j, mu, nu, rho)
            matrix[i, j] = max(0.0, p_i * p_j * t)

    # Normalise — tau correction can slightly shift the total
    total = matrix.sum()
    if total > 0:
        matrix /= total

    return matrix


def derive_probabilities(
    home_xg: float,
    away_xg: float,
    rho: float,
) -> dict:
    """
    Derive probabilities for all APEX-10 bet types from xG + rho.

    Returns dict with probability for each bet type.
    All values in [0, 1].
    """
    mu = max(home_xg, 0.1)  # Guard against zero xG
    nu = max(away_xg, 0.1)

    m = _build_score_matrix(mu, nu, rho)

    # ── Core outcomes ──────────────────────────────────────────────────────
    home_win = float(np.sum(np.tril(m, -1)))  # i > j
    draw = float(np.sum(np.diag(m)))  # i == j
    away_win = float(np.sum(np.triu(m, 1)))  # j > i

    # ── Draw No Bet (DNB) — home ──────────────────────────────────────────
    # Win or stake returned on draw → P(win) / P(win or draw)
    dnb_home_prob = (
        home_win / (home_win + away_win) if (home_win + away_win) > 0 else 0.5
    )

    # ── Asian Handicap -0.5 (same as 1X2 home win) ────────────────────────
    ah_minus_half = home_win

    # ── Asian Handicap -1.0 (win by 2+ = win, win by 1 = push) ───────────
    ah_minus_one_win = float(
        np.sum([m[i, j] for i in range(MAX_GOALS)
                for j in range(MAX_GOALS) if i - j >= 2])
    )
    ah_minus_one_push = float(
        np.sum([m[i, j] for i in range(MAX_GOALS)
                for j in range(MAX_GOALS) if i - j == 1])
    )
    # EV-equivalent probability
    ah_minus_one = ah_minus_one_win + (0.5 * ah_minus_one_push)

    # ── Asian Handicap -1.5 (win by 2+) ───────────────────────────────────
    ah_minus_one_half = float(
        np.sum([m[i, j] for i in range(MAX_GOALS)
                for j in range(MAX_GOALS) if i - j >= 2])
    )

    # ── Over/Under goals ─────────────────────────────────────────────────
    over_1_5 = float(
        np.sum([m[i, j] for i in range(MAX_GOALS)
                for j in range(MAX_GOALS) if i + j >= 2])
    )
    over_2_5 = float(
        np.sum([m[i, j] for i in range(MAX_GOALS)
                for j in range(MAX_GOALS) if i + j >= 3])
    )
    under_3_5 = float(
        np.sum([m[i, j] for i in range(MAX_GOALS)
                for j in range(MAX_GOALS) if i + j <= 3])
    )

    # ── BTTS No (at least one team keeps a clean sheet) ───────────────────
    btts_yes = float(
        np.sum([m[i, j] for i in range(1, MAX_GOALS)
                for j in range(1, MAX_GOALS)])
    )
    btts_no = 1.0 - btts_yes

    # ── Double Chance ─────────────────────────────────────────────────────
    dc_1x = home_win + draw  # Home win or draw
    dc_x2 = away_win + draw  # Away win or draw

    # ── Team to Score 1+ ─────────────────────────────────────────────────
    home_scores = 1.0 - float(np.sum(m[:, 0]))  # P(home goals > 0)

    return {
        "home_win": round(home_win, 4),
        "draw": round(draw, 4),
        "away_win": round(away_win, 4),
        "dnb_home": round(dnb_home_prob, 4),
        "ah_minus_0_5": round(ah_minus_half, 4),
        "ah_minus_1_0": round(ah_minus_one, 4),
        "ah_minus_1_5": round(ah_minus_one_half, 4),
        "over_1_5": round(over_1_5, 4),
        "over_2_5": round(over_2_5, 4),
        "under_3_5": round(under_3_5, 4),
        "btts_no": round(btts_no, 4),
        "dc_1x": round(dc_1x, 4),
        "dc_x2": round(dc_x2, 4),
        "home_scores": round(home_scores, 4),
    }


def prob_to_odds(prob: float) -> float | None:
    """Convert probability to decimal odds. Returns None if prob <= 0."""
    if prob <= 0:
        return None
    return round(1.0 / prob, 3)
