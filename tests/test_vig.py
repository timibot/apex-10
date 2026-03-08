"""Tests for Power Method vig removal."""
import pytest

from apex10.scoring.vig import get_true_home_prob, remove_vig_linear, remove_vig_power


class TestRemoveVigPower:
    def test_probabilities_sum_to_one(self):
        probs = remove_vig_power([1.30, 4.50, 9.00])
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_favourite_prob_higher_than_linear(self):
        """
        Power Method gives higher probability to heavy favourites than linear.
        This is the core assertion from the spec — a 1.30 favourite should
        shift from ~72% (linear) to ~74% (power method).
        """
        odds = [1.30, 4.50, 9.00]
        power_probs = remove_vig_power(odds)
        linear_probs = remove_vig_linear(odds)
        # Favourite (index 0) should have higher prob with Power Method
        assert power_probs[0] > linear_probs[0]

    def test_favourite_shift_magnitude(self):
        """1.30 favourite should shift by roughly 2 percentage points."""
        odds = [1.30, 4.50, 9.00]
        power_probs = remove_vig_power(odds)
        linear_probs = remove_vig_linear(odds)
        diff = power_probs[0] - linear_probs[0]
        assert 0.01 < diff < 0.05  # 1–5 percentage point shift

    def test_all_probabilities_positive(self):
        probs = remove_vig_power([1.20, 3.50, 6.00])
        assert all(p > 0 for p in probs)

    def test_raises_on_invalid_odds(self):
        with pytest.raises(ValueError):
            remove_vig_power([0.90, 3.00, 6.00])

    def test_raises_on_empty_list(self):
        with pytest.raises((ValueError, Exception)):
            remove_vig_power([])

    def test_two_way_market(self):
        """Asian handicap / DNB — two outcomes."""
        probs = remove_vig_power([1.85, 2.00])
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)
        assert all(p > 0 for p in probs)

    def test_no_overround_in_output(self):
        """Output must have no vig — sum must be exactly 1."""
        probs = remove_vig_power([1.40, 3.20, 7.00])
        assert sum(probs) == pytest.approx(1.0, abs=1e-5)


class TestGetTrueHomeProb:
    def test_returns_float(self):
        prob = get_true_home_prob(1.30, 4.50, 9.00)
        assert isinstance(prob, float)

    def test_heavy_favourite_above_70_pct(self):
        prob = get_true_home_prob(1.25, 5.00, 10.00)
        assert prob > 0.70
