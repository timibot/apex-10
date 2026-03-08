"""
The 6-gate filter sequence. Applied in strict order.
A leg is dropped the moment it fails any gate — no re-evaluation.

Gate order (spec-mandated, do not reorder):
  1. Odds range         → 1.20 <= odds <= 1.49
  2. Dual edge          → both models >= 4% edge + divergence <= 10%
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
from apex10.scoring.edge import divergence_check, dual_edge_check

logger = logging.getLogger(__name__)


class ConfidenceTier(str, Enum):
    S = "S"  # Divergence <=3%, both edge >=6%, no flags
    A = "A"  # Divergence <=5%, both edge >=4%, no critical flags
    B = "B"  # Divergence 5-10%, or flagged
    X = "X"  # Auto drop — never placed


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
    # Feature completeness flag set by NaN gate
    features_complete: bool = True
    # Populated after passing all gates
    consensus_prob: float = 0.0
    lgbm_edge: float = 0.0
    xgb_edge: float = 0.0
    # GP-4: Confidence tier
    tier: ConfidenceTier = ConfidenceTier.A
    # GP-5: Line movement flag for B-tier assignment
    line_movement_flag: bool = False

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
    """Gate 1: Odds must be in [1.20, 1.49]."""
    passed = ODDS.MIN_ODDS <= c.odds <= ODDS.MAX_ODDS
    return GateResult(
        passed=passed,
        gate=1,
        gate_name="odds_range",
        reason="" if passed else (
            f"Odds {c.odds} outside [{ODDS.MIN_ODDS}, {ODDS.MAX_ODDS}]"
        ),
    )


def gate_2_dual_edge(c: Candidate) -> GateResult:
    """Gate 2: Both models >= 4% edge AND divergence <= 10%."""
    edge_result = dual_edge_check(c.lgbm_prob, c.xgb_prob, c.odds)
    div_result = divergence_check(c.lgbm_prob, c.xgb_prob)

    if not edge_result["passed"]:
        return GateResult(
            passed=False,
            gate=2,
            gate_name="dual_edge",
            reason=(
                f"Edge too low: lgbm={edge_result['lgbm_edge']:.3f}, "
                f"xgb={edge_result['xgb_edge']:.3f}, min={ODDS.MIN_EDGE}"
            ),
        )

    if not div_result["passed"]:
        return GateResult(
            passed=False,
            gate=2,
            gate_name="dual_edge",
            reason=(
                f"Model divergence {div_result['divergence']:.3f} "
                f"> {ODDS.MAX_DIVERGENCE}"
            ),
        )

    return GateResult(passed=True, gate=2, gate_name="dual_edge")


def gate_3_lineup_risk(c: Candidate) -> GateResult:
    """
    Gate 3: Key player absent on the favoured side triggers drop.
    If home team is favourite and key home player absent → drop.
    If away team is favourite and key away player absent → drop.
    """
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
    # Also verify odds and probabilities are finite
    for val, name in [
        (c.odds, "odds"),
        (c.lgbm_prob, "lgbm_prob"),
        (c.xgb_prob, "xgb_prob"),
    ]:
        if val is None or val != val:  # NaN check  # noqa: PLR0124
            return GateResult(
                passed=False,
                gate=4,
                gate_name="nan_check",
                reason=f"Field '{name}' is NaN or None",
            )
    return GateResult(passed=True, gate=4, gate_name="nan_check")


def gate_5_line_movement(c: Candidate) -> GateResult:
    """
    Gate 5: Line movement with one-sharp B-tier path.
    drift > 0.12 = severe (both-sharp) → drop.
    drift 0.08–0.12 = moderate (one-sharp) → B-tier flag, pass.
    drift <= 0.08 = healthy → pass.
    """
    if c.drift <= ODDS.MAX_DRIFT:
        return GateResult(passed=True, gate=5, gate_name="line_movement")

    # Drift > 0.08 — magnitude determines severity
    if c.drift > 0.12:
        # Severe drift — treat as both-sharp → drop
        return GateResult(
            passed=False,
            gate=5,
            gate_name="line_movement",
            reason=(
                f"Severe drift {c.drift:.3f} > 0.12 — "
                f"both-sharp signal, auto drop"
            ),
        )
    else:
        # Moderate drift — one-sharp → B-tier flag, still passes
        c.line_movement_flag = True
        return GateResult(
            passed=True,
            gate=5,
            gate_name="line_movement",
            reason=(
                f"Moderate drift {c.drift:.3f} — "
                f"one-sharp flag, B-tier assigned"
            ),
        )


def gate_6_correlation(
    c: Candidate, approved_leagues: dict[str, int]
) -> GateResult:
    """
    Gate 6: Max 3 legs from the same league on any ticket.
    approved_leagues = {league_name: count_already_approved}
    """
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


def assign_tier(
    lgbm_prob: float,
    xgb_prob: float,
    lgbm_edge: float,
    xgb_edge: float,
    has_flag: bool = False,
) -> ConfidenceTier:
    """
    Assign confidence tier based on model divergence and edge.
    Called after a candidate passes Gates 1-6.
    """
    divergence = abs(lgbm_prob - xgb_prob)
    both_have_edge = lgbm_edge >= 0.04 and xgb_edge >= 0.04
    both_high_edge = lgbm_edge >= 0.06 and xgb_edge >= 0.06

    # Auto drop
    if divergence > 0.10 or not both_have_edge:
        return ConfidenceTier.X

    # S-tier: tight agreement, both strong edge, no flags
    if divergence <= 0.03 and both_high_edge and not has_flag:
        return ConfidenceTier.S

    # A-tier: acceptable agreement, both meet minimum edge
    if divergence <= 0.05 and both_have_edge and not has_flag:
        return ConfidenceTier.A

    # B-tier: everything else that passed gates
    return ConfidenceTier.B


def run_all_gates(
    candidates: list[Candidate],
) -> tuple[list[Candidate], list[dict]]:
    """
    Run all 6 gates on a list of candidates.
    Returns (qualified_candidates, rejection_log).
    Gate 6 is applied statefully — league counts accumulate as legs pass.
    """
    qualified = []
    rejection_log = []
    league_counts: dict[str, int] = {}

    gates = [
        gate_1_odds_range,
        gate_2_dual_edge,
        gate_3_lineup_risk,
        gate_4_nan_check,
        gate_5_line_movement,
    ]

    for c in candidates:
        rejected = False

        # Gates 1–5: stateless
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

        # Gate 6: stateful — uses running league count
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

        # Passed all gates — annotate edges and consensus
        league_counts[c.league] = league_counts.get(c.league, 0) + 1
        c.consensus_prob = round((c.lgbm_prob + c.xgb_prob) / 2.0, 4)
        c.lgbm_edge = round(c.lgbm_prob - (1.0 / c.odds), 4)
        c.xgb_edge = round(c.xgb_prob - (1.0 / c.odds), 4)

        # GP-4: Assign confidence tier
        c.tier = assign_tier(
            c.lgbm_prob,
            c.xgb_prob,
            c.lgbm_edge,
            c.xgb_edge,
            has_flag=getattr(c, "line_movement_flag", False),
        )

        # GP-4: X-tier = auto drop even if passed gates 1–6
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

    logger.info(
        f"Gate results: {len(qualified)} qualified, "
        f"{len(rejection_log)} rejected from {len(candidates)} candidates"
    )
    return qualified, rejection_log
