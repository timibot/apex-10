"""Tests for Dixon-Coles score matrix and bet type derivation."""
import numpy as np
import pytest

from apex10.scoring.dixon_coles import (
    MAX_GOALS,
    _build_score_matrix,
    derive_probabilities,
    prob_to_odds,
)


class TestScoreMatrix:
    def test_matrix_sums_to_one(self):
        m = _build_score_matrix(1.5, 1.0, -0.1)
        assert m.sum() == pytest.approx(1.0, abs=1e-6)

    def test_all_values_non_negative(self):
        m = _build_score_matrix(1.5, 1.0, -0.2)
        assert (m >= 0).all()

    def test_zero_zero_inflated_vs_standard_poisson(self):
        """
        Dixon-Coles with negative rho should give higher 0-0 probability
        than standard Poisson (rho=0).
        """
        m_corrected = _build_score_matrix(1.5, 1.0, -0.1)
        m_standard = _build_score_matrix(1.5, 1.0, 0.0)
        assert m_corrected[0, 0] > m_standard[0, 0]

    def test_matrix_shape(self):
        m = _build_score_matrix(1.5, 1.0, -0.1)
        assert m.shape == (10, 10)

    def test_handles_extreme_xg(self):
        """Should not crash on extreme xG values."""
        m = _build_score_matrix(0.1, 0.1, -0.1)
        assert m.sum() == pytest.approx(1.0, abs=1e-5)


class TestDeriveProbabilities:
    def _probs(self):
        return derive_probabilities(1.5, 1.0, -0.1)

    def test_returns_all_bet_types(self):
        p = self._probs()
        expected_keys = [
            "home_win", "draw", "away_win", "dnb_home",
            "ah_minus_0_5", "ah_minus_1_0", "ah_minus_1_5",
            "over_1_5", "over_2_5", "under_3_5",
            "btts_no", "dc_1x", "dc_x2", "home_scores",
        ]
        for key in expected_keys:
            assert key in p, f"Missing key: {key}"

    def test_home_draw_away_sum_to_one(self):
        p = self._probs()
        total = p["home_win"] + p["draw"] + p["away_win"]
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_all_probs_between_zero_and_one(self):
        p = self._probs()
        for key, val in p.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_over_1_5_greater_than_over_2_5(self):
        p = self._probs()
        assert p["over_1_5"] > p["over_2_5"]

    def test_btts_no_plus_btts_yes_approx_one(self):
        p = self._probs()
        m = _build_score_matrix(1.5, 1.0, -0.1)
        btts_yes = float(
            np.sum([m[i, j] for i in range(1, MAX_GOALS)
                    for j in range(1, MAX_GOALS)])
        )
        assert p["btts_no"] + btts_yes == pytest.approx(1.0, abs=1e-3)

    def test_dc_1x_greater_than_home_win(self):
        p = self._probs()
        assert p["dc_1x"] > p["home_win"]

    def test_dominant_home_team_high_home_win_prob(self):
        p = derive_probabilities(3.0, 0.5, -0.1)
        assert p["home_win"] > 0.70


class TestProbToOdds:
    def test_conversion(self):
        assert prob_to_odds(0.5) == pytest.approx(2.0, abs=0.01)

    def test_zero_prob_returns_none(self):
        assert prob_to_odds(0.0) is None

    def test_high_prob_gives_short_odds(self):
        assert prob_to_odds(0.80) < 1.30
