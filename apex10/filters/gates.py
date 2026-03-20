"""
The 6-gate filter sequence. Applied in strict order.
A leg is dropped the moment it fails any gate — no re-evaluation.

Gate order:
  1. Odds range         → 1.10 <= odds <= 1.60 (paper) or 1.20-1.49 (prod)
  2. Confidence vote    → confidence_votes >= 3 (paper) or >= 4 (prod)
  3. Lineup risk        → no key player absent flag on favourite
  4. NaN check          → no missing critical features
  5. Line movement      → drift <= 0.08
  6. Correlation        → max 3 legs from same league
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from apex10.config import ODDS

logger = logging.getLogger(__name__)

# Confidence vote thresholds
MIN_VOTES_PAPER = 3     # Paper trading: 3/5 signals must agree
MIN_VOTES_PRODUCTION = 4  # Production: 4/5 signals must agree


class ConfidenceTier(str, Enum):
    S = "S"  # 5/5 votes — max conviction
    A = "A"  # 4/5 votes — strong conviction
    B = "B"  # 3/5 votes — acceptable for paper
    X = "X"  # <3 votes — auto drop


TIER_RANK = {
    ConfidenceTier.S: 0,
    ConfidenceTier.A: 1,
    ConfidenceTier.B: 2,
    ConfidenceTier.X: 3,
}


@dataclass
class Candidate:
    """A fixture candidate being evaluated through the gates."""

    fixture_id: int
    league: str
    home_team: str
    away_team: str
    bet_type: str
    odds: float
    lgbm_prob: float
    xgb_prob: float
    opening_odds: float
    key_player_absent_home: int = 0
    key_player_absent_away: int = 0
    features_complete: bool = True
    consensus_prob: float = 0.0
    confidence_votes: int = 0
    tier: ConfidenceTier = ConfidenceTier.A
    line_movement_flag: bool = False

    # Legacy fields kept for compatibility
    lgbm_edge: float = 0.0
    xgb_edge: float = 0.0

    @property
    def drift(self) -> float:
        """Positive drift = odds lengthened (sharp money fading)."""
        return round(self.odds - self.opening_odds, 4)

    @property
    def is_home_favourite(self) -> bool:
        return self.lgbm_prob >= 0.5


@dataclass
class GateResult:
    passed: bool
    gate: int
    gate_name: str
    reason: str = ""


def gate_1_odds_range(c: Candidate) -> GateResult:
    """Gate 1: Odds must be in acceptable accumulator range."""
    min_odds = ODDS.MIN_ODDS
    max_odds = ODDS.MAX_ODDS
    passed = min_odds <= c.odds <= max_odds
    return GateResult(
        passed=passed,
        gate=1,
        gate_name="odds_range",
        reason="" if passed else (
            f"Odds {c.odds} outside [{min_odds}, {max_odds}]"
        ),
    )


def gate_2_confidence(c: Candidate) -> GateResult:
    """Gate 2: Confidence votes must meet minimum threshold."""
    min_votes = MIN_VOTES_PAPER
    passed = c.confidence_votes >= min_votes
    return GateResult(
        passed=passed,
        gate=2,
        gate_name="confidence",
        reason="" if passed else (
            f"Only {c.confidence_votes}/5 signals agree (need {min_votes}+)"
        ),
    )


def gate_3_lineup_risk(c: Candidate) -> GateResult:
    """Gate 3: Key player absent on the favoured side triggers drop."""
    if c.is_home_favourite and c.key_player_absent_home:
        return GateResult(
            passed=False,
            gate=3,
            gate_name="lineup_risk",
            reason=f"Key player absent for home favourite ({c.home_team})",
        )
    if not c.is_home_favourite and c.key_player_absent_away:
        return GateResult(
            passed=False,
            gate=3,
            gate_name="lineup_risk",
            reason=f"Key player absent for away favourite ({c.away_team})",
        )
    return GateResult(passed=True, gate=3, gate_name="lineup_risk")


def gate_4_nan_check(c: Candidate) -> GateResult:
    """Gate 4: Critical features must not be missing."""
    if not c.features_complete:
        return GateResult(
            passed=False,
            gate=4,
            gate_name="nan_check",
            reason="One or more critical features are NaN",
        )
    for val, name in [
        (c.odds, "odds"),
        (c.lgbm_prob, "lgbm_prob"),
        (c.xgb_prob, "xgb_prob"),
    ]:
        if val is None or val != val:  # noqa: PLR0124
            return GateResult(
                passed=False,
                gate=4,
                gate_name="nan_check",
                reason=f"Field '{name}' is NaN or None",
            )
    return GateResult(passed=True, gate=4, gate_name="nan_check")


def gate_5_line_movement(c: Candidate) -> GateResult:
    """Gate 5: Line movement check."""
    if c.drift <= ODDS.MAX_DRIFT:
        return GateResult(passed=True, gate=5, gate_name="line_movement")
    if c.drift > 0.12:
        return GateResult(
            passed=False,
            gate=5,
            gate_name="line_movement",
            reason=f"Severe drift {c.drift:.3f} > 0.12 — auto drop",
        )
    else:
        c.line_movement_flag = True
        return GateResult(
            passed=True,
            gate=5,
            gate_name="line_movement",
            reason=f"Moderate drift {c.drift:.3f} — B-tier flag",
        )


def gate_6_correlation(
    c: Candidate, approved_leagues: dict[str, int]
) -> GateResult:
    """Gate 6: Max 3 legs from the same league on any ticket."""
    current_count = approved_leagues.get(c.league, 0)
    if current_count >= ODDS.MAX_LEGS_PER_LEAGUE:
        return GateResult(
            passed=False,
            gate=6,
            gate_name="correlation",
            reason=(
                f"League {c.league} already has {current_count} legs "
                f"(max {ODDS.MAX_LEGS_PER_LEAGUE})"
            ),
        )
    return GateResult(passed=True, gate=6, gate_name="correlation")


def assign_tier(votes: int, has_flag: bool = False) -> ConfidenceTier:
    """Assign confidence tier based on vote count."""
    if votes >= 5:
        return ConfidenceTier.S
    if votes >= 4 and not has_flag:
        return ConfidenceTier.A
    if votes >= 3:
        return ConfidenceTier.B
    return ConfidenceTier.X


def run_all_gates(
    candidates: list[Candidate],
) -> tuple[list[Candidate], list[dict]]:
    """
    Run all 6 gates on a list of candidates.
    Returns (qualified_candidates, rejection_log).
    """
    qualified = []
    rejection_log = []
    league_counts: dict[str, int] = {}

    gates = [
        gate_1_odds_range,
        gate_2_confidence,
        gate_3_lineup_risk,
        gate_4_nan_check,
        gate_5_line_movement,
    ]

    # Pre-sort candidates to ensure highest confidence and odds get league slots first
    # Sort order: 1. confidence_votes (DESC), 2. odds (DESC), 3. fixture_id (ASC)
    sorted_candidates = sorted(
        candidates, 
        key=lambda c: (c.confidence_votes, c.odds, -c.fixture_id), 
        reverse=True
    )

    for c in sorted_candidates:
        rejected = False

        for gate_fn in gates:
            result = gate_fn(c)
            if not result.passed:
                rejection_log.append({
                    "fixture_id": c.fixture_id,
                    "home_team": c.home_team,
                    "away_team": c.away_team,
                    "gate": result.gate,
                    "gate_name": result.gate_name,
                    "reason": result.reason,
                })
                rejected = True
                break

        if rejected:
            continue

        # Gate 6: stateful — league cap
        # No swap logic needed because we pre-sorted the candidates.
        g6 = gate_6_correlation(c, league_counts)
        if not g6.passed:
            rejection_log.append({
                "fixture_id": c.fixture_id,
                "home_team": c.home_team,
                "away_team": c.away_team,
                "gate": 6,
                "gate_name": "correlation",
                "reason": g6.reason,
            })
            continue

        # Passed all gates
        league_counts[c.league] = league_counts.get(c.league, 0) + 1
        c.tier = assign_tier(
            c.confidence_votes,
            has_flag=getattr(c, "line_movement_flag", False),
        )

        if c.tier == ConfidenceTier.X:
            rejection_log.append({
                "fixture_id": c.fixture_id,
                "home_team": c.home_team,
                "away_team": c.away_team,
                "gate": 2,
                "gate_name": "tier_assignment",
                "reason": "Assigned X-tier — auto drop",
            })
            continue

        qualified.append(c)

    # --- Post-loop Swap Logic ---
    # User requested: replace <5/5 accepted games with 5/5 rejected games from Gate 6
    gate_6_rejects_5_5 = [
        c for c in sorted_candidates 
        if c.confidence_votes == 5 and c not in qualified
        and any(r["fixture_id"] == c.fixture_id and r["gate"] == 6 for r in rejection_log)
    ]
    
    if gate_6_rejects_5_5:
        # Find weakest accepted games (< 5 votes), weakest are at the end of the list
        weak_accepted = [c for c in qualified if c.confidence_votes < 5]
        
        while gate_6_rejects_5_5 and weak_accepted:
            new_leg = gate_6_rejects_5_5.pop(0)  # Strongest 5/5 reject (best odds)
            weak_leg = weak_accepted.pop(-1)     # Weakest accepted (lowest votes, lowest odds)
            
            logger.info(
                f"Gate 6 Override: Swapping IN 5/5 {new_leg.home_team} vs {new_leg.away_team} "
                f"for {weak_leg.confidence_votes}/5 {weak_leg.home_team} vs {weak_leg.away_team}"
            )
            
            qualified.remove(weak_leg)
            
            # Record why the weak leg was dropped
            rejection_log.append({
                "fixture_id": weak_leg.fixture_id,
                "home_team": weak_leg.home_team,
                "away_team": weak_leg.away_team,
                "gate": 6,
                "gate_name": "correlation_override",
                "reason": f"Swapped out to rescue 5/5 game {new_leg.home_team}",
            })
            
            # Remove the new_leg from the rejection log so it doesn't show as rejected
            for r in list(rejection_log):
                if r["fixture_id"] == new_leg.fixture_id and r["gate"] == 6:
                    rejection_log.remove(r)
                    break
                    
            # Set tier properly and add to qualified
            new_leg.tier = ConfidenceTier.S
            qualified.append(new_leg)

    logger.info(
        f"Gate results: {len(qualified)} qualified, "
        f"{len(rejection_log)} rejected from {len(candidates)} candidates"
    )
    return qualified, rejection_log
