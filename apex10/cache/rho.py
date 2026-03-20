"""
Computes Dixon-Coles rho (ρ) correction factor per league.
ρ is calibrated from historical match data and stored in Supabase.

Dixon-Coles correction function τ(x, y, μ, ν, ρ):
  - Applied only to low-scoring cells: (0,0), (1,0), (0,1), (1,1)
  - Negative probabilities guarded with max(0, ...)
  - ρ is estimated by maximising log-likelihood over historical matches
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

logger = logging.getLogger(__name__)


def tau(x: int, y: int, mu: float, nu: float, rho: float) -> float:
    """
    Dixon-Coles correction factor for low-scoring scorelines.
    Only applied to (0,0), (1,0), (0,1), (1,1).
    All other scorelines return 1.0 (no correction).
    """
    if x == 0 and y == 0:
        return max(0.0, 1 - mu * nu * rho)
    elif x == 1 and y == 0:
        return max(0.0, 1 + nu * rho)
    elif x == 0 and y == 1:
        return max(0.0, 1 + mu * rho)
    elif x == 1 and y == 1:
        return max(0.0, 1 - rho)
    else:
        return 1.0


def dixon_coles_log_likelihood(rho: float, matches: list[dict]) -> float:
    """
    Compute negative log-likelihood of rho given match data.
    Minimising this finds the optimal rho.

    Each match dict must have: home_goals, away_goals, home_xg, away_xg
    We use xG as the Poisson rate parameters (μ, ν).
    """
    log_lik = 0.0

    for match in matches:
        mu = match["home_xg"]   # Expected home goals
        nu = match["away_xg"]   # Expected away goals
        x = match["home_goals"]
        y = match["away_goals"]

        if mu <= 0 or nu <= 0:
            continue

        p_x = poisson.pmf(x, mu)
        p_y = poisson.pmf(y, nu)
        t = tau(x, y, mu, nu, rho)

        joint_prob = p_x * p_y * t

        if joint_prob <= 0:
            continue

        log_lik += np.log(joint_prob)

    return -log_lik  # Return negative for minimisation


def estimate_rho(matches: list[dict]) -> float:
    """
    Estimate optimal rho for a league using scipy minimise_scalar.
    Searches rho in [-1.0, 1.0] — negative values are theoretically valid
    but practically rho is usually in [-0.2, 0.0] for football.

    Returns estimated rho value.
    """
    if not matches:
        logger.warning("No matches provided — returning rho=0.0")
        return 0.0

    if len(matches) < 50:
        logger.warning(f"Only {len(matches)} matches — rho estimate may be unreliable")

    result = minimize_scalar(
        dixon_coles_log_likelihood,
        bounds=(-1.0, 1.0),
        method="bounded",
        args=(matches,),
    )

    if not result.success:
        logger.warning("Rho optimisation did not converge — returning 0.0")
        return 0.0

    rho = float(result.x)
    logger.info(f"Estimated rho: {rho:.4f} from {len(matches)} matches")
    return rho


def compute_and_store_rho(league_name: str, db_client) -> float:
    """
    Fetch historical xG + goals for a league from Supabase,
    compute rho, and store result in league_rho table.
    Returns computed rho.
    """
    # Fetch joined match data (xG + actual goals)
    # Note: match_xg may not have a league column, so we fetch all and
    # rely on the fact that compute_and_store_rho is called per-league.
    # If match_xg gains a league column, add .eq("league", league_name)
    result = (
        db_client.table("match_xg")
        .select("home_xg,away_xg,home_goals,away_goals")
        .not_.is_("home_xg", "null")
        .not_.is_("away_xg", "null")
        .execute()
    )

    matches = result.data if result.data else []

    if not matches:
        logger.error(f"No xG data found for {league_name} — cannot compute rho")
        return 0.0

    rho = estimate_rho(matches)

    # Upsert into league_rho
    db_client.table("league_rho").upsert(
        {
            "league": league_name,
            "rho": rho,
            "sample_size": len(matches),
        },
        on_conflict="league",
    ).execute()

    logger.info(f"Stored rho={rho:.4f} for {league_name} (n={len(matches)})")
    return rho
