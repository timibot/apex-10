"""
Dual lineup rotation sub-model.
Predicts P(full_strength_XI) for each team in each fixture.
Used in Gate 3 to decide whether to drop or flag a leg.

Features (per team):
  - days_to_next: days until next fixture
  - competition_weight_current: importance of this match (0-1)
  - competition_weight_next: importance of next match (0-1)
  - importance_delta: next - current
  - squad_depth_ratio: available players / full squad size
  - manager_rotation_rate: historical rotation frequency
  - fixture_congestion: days since last match

Labels: 1 = full-strength XI fielded, 0 = rotated
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

LINEUP_FEATURES = [
    "days_to_next",
    "competition_weight_current",
    "competition_weight_next",
    "importance_delta",
    "squad_depth_ratio",
    "manager_rotation_rate",
    "fixture_congestion",
]

FULL_XI_THRESHOLD = 0.72
QUALITY_DEGRADATION = 0.70
LINEUP_DIVERGENCE_THRESHOLD = 0.10


def build_lineup_features(candidate_row: dict) -> np.ndarray:
    """Build feature vector for lineup prediction from a fixture row."""
    comp_w = float(
        candidate_row.get("competition_weight_home", 0.7)
    )
    next_w = float(
        candidate_row.get("next_competition_weight_home", 0.7)
    )
    return np.array(
        [
            [
                float(
                    candidate_row.get("days_to_next_home", 7)
                ),
                comp_w,
                next_w,
                next_w - comp_w,
                float(
                    candidate_row.get("squad_depth_ratio", 0.8)
                ),
                float(
                    candidate_row.get(
                        "manager_rotation_rate", 0.3
                    )
                ),
                float(
                    candidate_row.get(
                        "fixture_congestion_home", 7
                    )
                ),
            ]
        ],
        dtype=np.float32,
    )


def predict_full_xi_probability(
    lgbm_model,
    xgb_model,
    features: np.ndarray,
) -> dict:
    """
    Predict P(full_strength_XI) from both models.
    Returns consensus, divergence, and flag status.
    """
    lgbm_prob = float(lgbm_model.predict(features)[0])
    xgb_prob = float(xgb_model.predict_proba(features)[0, 1])
    consensus = (lgbm_prob + xgb_prob) / 2.0
    divergence = abs(lgbm_prob - xgb_prob)

    return {
        "consensus_prob": round(consensus, 4),
        "lgbm_prob": round(lgbm_prob, 4),
        "xgb_prob": round(xgb_prob, 4),
        "divergence": round(divergence, 4),
        "lineup_flag": divergence > LINEUP_DIVERGENCE_THRESHOLD,
        "full_xi_likely": consensus >= FULL_XI_THRESHOLD,
    }


def recalculate_odds_degraded(original_odds: float) -> float:
    """
    Estimate odds if squad is degraded to 70% quality.
    Degraded squad → lower win probability → higher odds.
    """
    if original_odds <= 1.0:
        return original_odds
    original_prob = 1.0 / original_odds
    degraded_prob = original_prob * QUALITY_DEGRADATION
    if degraded_prob <= 0:
        return 99.0
    return round(1.0 / degraded_prob, 3)


def run_lineup_gate(
    lgbm_model,
    xgb_model,
    candidate_row: dict,
    current_odds: float,
) -> dict:
    """
    Full Gate 3 lineup logic.
    Returns: {passed, reason, lineup_flag, recalculated_odds}
    """
    features = build_lineup_features(candidate_row)
    lineup = predict_full_xi_probability(
        lgbm_model, xgb_model, features
    )

    result = {
        "passed": True,
        "reason": "",
        "lineup_flag": lineup["lineup_flag"],
        "full_xi_prob": lineup["consensus_prob"],
        "recalculated_odds": current_odds,
    }

    # If P(full_XI) < threshold, recalculate with degraded quality
    if not lineup["full_xi_likely"]:
        degraded_odds = recalculate_odds_degraded(current_odds)
        result["recalculated_odds"] = degraded_odds

        from apex10.config import ODDS

        if degraded_odds > ODDS.MAX_ODDS:
            result["passed"] = False
            result["reason"] = (
                f"P(full_XI)={lineup['consensus_prob']:.2f} "
                f"< {FULL_XI_THRESHOLD}. "
                f"Degraded odds {degraded_odds:.2f} "
                f"> {ODDS.MAX_ODDS} — drop."
            )
            return result

    if lineup["lineup_flag"]:
        result["reason"] = (
            f"Lineup models diverge "
            f"{lineup['divergence']:.2f} > "
            f"{LINEUP_DIVERGENCE_THRESHOLD} — B-tier flag"
        )

    return result
