"""
Greedy ticket optimizer + Monte Carlo win probability.

Optimizer:
  Sort qualified legs by consensus_prob DESC, then odds DESC within same tier.
  Accumulate until product >= 10.0. Stop immediately.

Monte Carlo:
  Simulate 10,000 ticket outcomes by drawing each leg result from its
  probability distribution. Reports simulated win rate + 90% CI.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from apex10.config import ODDS
from apex10.filters.gates import TIER_RANK, Candidate, ConfidenceTier

logger = logging.getLogger(__name__)

MONTE_CARLO_SIMULATIONS = 10_000
CONFIDENCE_INTERVAL = 0.90
CONFIDENCE_ROUNDING = 3


@dataclass
class TicketLeg:
    fixture_id: int
    league: str
    home_team: str
    away_team: str
    bet_type: str
    odds: float
    consensus_prob: float
    lgbm_prob: float
    xgb_prob: float
    lgbm_edge: float
    xgb_edge: float
    confidence_votes: int = 0
    tier: str = "A"


@dataclass
class Ticket:
    legs: list[TicketLeg]
    combined_odds: float
    simulated_win_rate: float
    ci_low: float
    ci_high: float
    no_ticket: bool = False
    reason: str = ""

    def __str__(self) -> str:
        if self.no_ticket:
            return f"NO TICKET THIS WEEK — {self.reason}"
        lines = [
            f"APEX-10 TICKET ({len(self.legs)} legs)",
            f"Combined odds: {self.combined_odds:.2f}x",
            f"Simulated win rate: {self.simulated_win_rate * 100:.1f}% "
            f"(90% CI: {self.ci_low * 100:.1f}%–{self.ci_high * 100:.1f}%)",
            "─" * 40,
        ]
        for i, leg in enumerate(self.legs, 1):
            lines.append(
                f"{i}. [{leg.tier}] {leg.home_team} vs {leg.away_team} "
                f"[{leg.bet_type}] @ {leg.odds:.2f} "
                f"(conf: {leg.confidence_votes / 5.0 * 100:.0f}%)"
            )
        return "\n".join(lines)


def _sort_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """
    Sort by:
      1. Tier rank ASC (S=0 first)
      2. Model Probability DESC
      3. Odds DESC
    """
    return sorted(
        candidates,
        key=lambda c: (TIER_RANK.get(c.tier, 3), -c.consensus_prob, -c.odds),
    )


def build_ticket(qualified: list[Candidate]) -> Ticket:
    """
    Greedy accumulation:
      1. Sort candidates by confidence DESC, odds DESC within tier
      2. Multiply odds until product >= 10.0
      3. Stop immediately — no extra legs

    Returns Ticket with no_ticket=True if not enough qualified legs.
    """
    if not qualified:
        logger.warning("No qualified candidates — no ticket this week")
        return Ticket(
            legs=[],
            combined_odds=1.0,
            simulated_win_rate=0.0,
            ci_low=0.0,
            ci_high=0.0,
            no_ticket=True,
            reason="No candidates passed all 6 gates",
        )

    sorted_candidates = _sort_candidates(qualified)
    selected_legs: list[TicketLeg] = []
    running_product = 1.0

    for c in sorted_candidates:
        leg = TicketLeg(
            fixture_id=c.fixture_id,
            league=c.league,
            home_team=c.home_team,
            away_team=c.away_team,
            bet_type=c.bet_type,
            odds=c.odds,
            consensus_prob=c.consensus_prob,
            lgbm_prob=c.lgbm_prob,
            xgb_prob=c.xgb_prob,
            lgbm_edge=c.lgbm_edge,
            xgb_edge=c.xgb_edge,
            confidence_votes=c.confidence_votes,
            tier=c.tier.value if isinstance(c.tier, ConfidenceTier) else str(c.tier),
        )
        selected_legs.append(leg)
        running_product = round(running_product * c.odds, 4)

    if running_product < ODDS.TARGET_PRODUCT:
        logger.info(
            f"Note: Generated ticket product ({running_product:.2f}x) "
            f"is below target ({ODDS.TARGET_PRODUCT}x). Proceeding dynamically."
        )

    win_rate, ci_low, ci_high = monte_carlo_win_probability(selected_legs)

    ticket = Ticket(
        legs=selected_legs,
        combined_odds=round(running_product, 3),
        simulated_win_rate=round(win_rate, CONFIDENCE_ROUNDING),
        ci_low=round(ci_low, CONFIDENCE_ROUNDING),
        ci_high=round(ci_high, CONFIDENCE_ROUNDING),
    )

    logger.info(str(ticket))
    return ticket


def monte_carlo_win_probability(
    legs: list[TicketLeg],
    n_simulations: int = MONTE_CARLO_SIMULATIONS,
    ci: float = CONFIDENCE_INTERVAL,
) -> tuple[float, float, float]:
    """
    Simulate n_simulations ticket outcomes.
    Each leg outcome drawn from Bernoulli(consensus_prob).
    Ticket wins only if all legs win.

    Returns (win_rate, ci_low, ci_high).
    """
    rng = np.random.default_rng()
    probs = np.array([leg.consensus_prob for leg in legs])

    # Shape: (n_simulations, n_legs) — each cell is 1 (win) or 0 (loss)
    outcomes = rng.random((n_simulations, len(legs))) < probs

    # Ticket wins only if all legs win
    ticket_wins = outcomes.all(axis=1)
    win_rate = float(ticket_wins.mean())

    # Bootstrap CI
    alpha = (1 - ci) / 2
    bootstrap_samples = np.array([
        np.random.choice(ticket_wins, size=n_simulations, replace=True).mean()
        for _ in range(1000)
    ])
    ci_low = float(np.quantile(bootstrap_samples, alpha))
    ci_high = float(np.quantile(bootstrap_samples, 1 - alpha))

    logger.info(
        f"Monte Carlo: win_rate={win_rate:.3f}, "
        f"90% CI=[{ci_low:.3f}, {ci_high:.3f}] "
        f"({n_simulations:,} simulations)"
    )
    return win_rate, ci_low, ci_high


def quarter_kelly_stake(
    bank: float,
    win_prob: float,
    combined_odds: float,
    kelly_fraction: float = 0.25,
) -> float:
    """
    Quarter-Kelly staking formula.
    Kelly fraction = (edge / odds) × kelly_fraction
    Edge = (win_prob × combined_odds) - 1

    Returns stake amount. Always >= 0.
    """
    b = combined_odds - 1.0
    edge = (win_prob * combined_odds) - 1.0

    if edge <= 0 or b <= 0:
        return 0.0

    full_kelly = edge / b
    stake = round(bank * full_kelly * kelly_fraction, 2)
    return max(0.0, stake)
