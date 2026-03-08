"""Tests for all 6 filter gates."""
from apex10.filters.gates import (
    Candidate,
    gate_1_odds_range,
    gate_2_dual_edge,
    gate_3_lineup_risk,
    gate_4_nan_check,
    gate_5_line_movement,
    gate_6_correlation,
    run_all_gates,
)


def make_candidate(**kwargs) -> Candidate:
    defaults = {
        "fixture_id": 1,
        "league": "EPL",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "bet_type": "home_win",
        "odds": 1.35,
        "lgbm_prob": 0.84,   # clearly clears 0.741 + 0.04
        "xgb_prob": 0.82,    # clearly clears 0.741 + 0.04
        "opening_odds": 1.33,
        "key_player_absent_home": 0,
        "key_player_absent_away": 0,
        "features_complete": True,
    }
    defaults.update(kwargs)
    return Candidate(**defaults)


class TestGate1OddsRange:
    def test_passes_in_range(self):
        assert gate_1_odds_range(make_candidate(odds=1.35)).passed is True

    def test_fails_below_min(self):
        assert gate_1_odds_range(make_candidate(odds=1.15)).passed is False

    def test_fails_above_max(self):
        assert gate_1_odds_range(make_candidate(odds=1.55)).passed is False

    def test_passes_at_exact_min(self):
        assert gate_1_odds_range(make_candidate(odds=1.20)).passed is True

    def test_passes_at_exact_max(self):
        assert gate_1_odds_range(make_candidate(odds=1.49)).passed is True

    def test_fails_at_1_50(self):
        assert gate_1_odds_range(make_candidate(odds=1.50)).passed is False

    def test_gate_number_is_1(self):
        result = gate_1_odds_range(make_candidate(odds=1.35))
        assert result.gate == 1


class TestGate2DualEdge:
    def test_passes_both_models_have_edge(self):
        # odds=1.30 → implied=0.769. Both models at 0.84/0.82 → edge ~0.07. Clear.
        c = make_candidate(odds=1.30, lgbm_prob=0.84, xgb_prob=0.82)
        assert gate_2_dual_edge(c).passed is True

    def test_fails_if_lgbm_below_min_edge(self):
        c = make_candidate(odds=1.35, lgbm_prob=0.76, xgb_prob=0.80)
        assert gate_2_dual_edge(c).passed is False

    def test_fails_if_xgb_below_min_edge(self):
        c = make_candidate(odds=1.35, lgbm_prob=0.82, xgb_prob=0.76)
        assert gate_2_dual_edge(c).passed is False

    def test_fails_on_high_divergence(self):
        c = make_candidate(odds=1.25, lgbm_prob=0.88, xgb_prob=0.76)
        assert gate_2_dual_edge(c).passed is False

    def test_gate_number_is_2(self):
        c = make_candidate(odds=1.30, lgbm_prob=0.84, xgb_prob=0.82)
        result = gate_2_dual_edge(c)
        assert result.gate == 2


class TestGate3LineupRisk:
    def test_passes_no_absences(self):
        assert gate_3_lineup_risk(make_candidate()).passed is True

    def test_fails_home_favourite_key_player_absent(self):
        c = make_candidate(lgbm_prob=0.78, key_player_absent_home=1)
        assert gate_3_lineup_risk(c).passed is False

    def test_passes_away_key_player_absent_when_home_is_favourite(self):
        """Away key player absent is irrelevant if home is the favourite."""
        c = make_candidate(lgbm_prob=0.78, key_player_absent_away=1)
        assert gate_3_lineup_risk(c).passed is True

    def test_fails_away_favourite_key_player_absent(self):
        c = make_candidate(
            lgbm_prob=0.35, xgb_prob=0.32, key_player_absent_away=1
        )
        assert gate_3_lineup_risk(c).passed is False


class TestGate4NanCheck:
    def test_passes_complete_features(self):
        assert gate_4_nan_check(make_candidate()).passed is True

    def test_fails_incomplete_features(self):
        c = make_candidate(features_complete=False)
        assert gate_4_nan_check(c).passed is False

    def test_fails_nan_odds(self):
        c = make_candidate(odds=float("nan"))
        assert gate_4_nan_check(c).passed is False


class TestGate5LineMovement:
    def test_passes_no_drift(self):
        c = make_candidate(odds=1.33, opening_odds=1.33)
        assert gate_5_line_movement(c).passed is True

    def test_passes_shortening(self):
        """Odds shortening (negative drift) is fine — money coming in."""
        c = make_candidate(odds=1.25, opening_odds=1.33)
        assert gate_5_line_movement(c).passed is True

    def test_moderate_drift_passes_with_flag(self):
        """Drift 0.09 = one-sharp → passes with B-tier flag."""
        c = make_candidate(odds=1.42, opening_odds=1.33)
        result = gate_5_line_movement(c)
        assert result.passed is True
        assert c.line_movement_flag is True

    def test_passes_drift_exactly_at_threshold(self):
        c = make_candidate(odds=1.41, opening_odds=1.33)  # drift = 0.08
        assert gate_5_line_movement(c).passed is True

    def test_severe_drift_above_012_fails(self):
        """Drift > 0.12 = both-sharp → auto drop."""
        c = make_candidate(odds=1.47, opening_odds=1.33)  # drift = 0.14
        assert gate_5_line_movement(c).passed is False


class TestGate6Correlation:
    def test_passes_first_leg_from_league(self):
        c = make_candidate(league="EPL")
        result = gate_6_correlation(c, {"EPL": 0})
        assert result.passed is True

    def test_passes_third_leg_from_league(self):
        c = make_candidate(league="EPL")
        result = gate_6_correlation(c, {"EPL": 2})
        assert result.passed is True

    def test_fails_fourth_leg_from_same_league(self):
        c = make_candidate(league="EPL")
        result = gate_6_correlation(c, {"EPL": 3})
        assert result.passed is False

    def test_passes_different_league(self):
        c = make_candidate(league="La Liga")
        result = gate_6_correlation(c, {"EPL": 3})
        assert result.passed is True


class TestRunAllGates:
    def _good_candidate(self, fixture_id=1, league="EPL") -> Candidate:
        # implied = 1/1.28 ≈ 0.781. Both need >= 0.781 + 0.04 = 0.821
        return make_candidate(
            fixture_id=fixture_id,
            league=league,
            odds=1.28,
            lgbm_prob=0.87,
            xgb_prob=0.85,
            opening_odds=1.27,
        )

    def test_all_good_candidates_pass(self):
        candidates = [self._good_candidate(i) for i in range(3)]
        qualified, rejected = run_all_gates(candidates)
        assert len(qualified) == 3
        assert len(rejected) == 0

    def test_bad_odds_candidate_rejected(self):
        candidates = [
            self._good_candidate(1),
            make_candidate(fixture_id=2, odds=1.60),  # Gate 1 fail
        ]
        qualified, rejected = run_all_gates(candidates)
        assert len(qualified) == 1
        assert len(rejected) == 1
        assert rejected[0]["gate"] == 1

    def test_gate_6_enforced_across_candidates(self):
        """4th leg from same league must be rejected."""
        candidates = [
            self._good_candidate(i, league="EPL") for i in range(4)
        ]
        qualified, rejected = run_all_gates(candidates)
        assert len(qualified) == 3
        assert len(rejected) == 1
        assert rejected[0]["gate"] == 6

    def test_rejection_log_contains_reason(self):
        candidates = [make_candidate(odds=1.60)]
        _, rejected = run_all_gates(candidates)
        assert "reason" in rejected[0]
        assert len(rejected[0]["reason"]) > 0
