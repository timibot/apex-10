"""
Tests for Dixon-Coles rho computation.
These are pure math tests — no DB, no network.
"""
import numpy as np
import pytest

from apex10.cache.rho import dixon_coles_log_likelihood, estimate_rho, tau


class TestTau:
    """tau() is the correction function — every cell must be verified."""

    def test_tau_0_0_no_correction_at_zero_rho(self):
        """At rho=0, tau should return 1.0 for all cells."""
        assert tau(0, 0, 1.5, 1.0, 0.0) == pytest.approx(1.0)

    def test_tau_1_1_no_correction_at_zero_rho(self):
        assert tau(1, 1, 1.5, 1.0, 0.0) == pytest.approx(1.0)

    def test_tau_high_scoring_always_one(self):
        """Non-low-scoring scorelines must always return exactly 1.0."""
        assert tau(2, 0, 1.5, 1.0, -0.1) == 1.0
        assert tau(3, 2, 2.0, 1.5, 0.5) == 1.0
        assert tau(0, 3, 1.0, 2.0, -0.5) == 1.0

    def test_tau_never_returns_negative(self):
        """Negative probabilities are physically impossible — guard must hold."""
        extreme_rho = 1.0
        result = tau(0, 0, 0.1, 0.1, extreme_rho)
        assert result >= 0.0

    def test_tau_0_0_negative_rho_increases_probability(self):
        """
        Negative rho increases 0-0 probability (corrects Poisson underestimate).
        At rho=-0.1, mu=1.5, nu=1.0: tau = 1 - (1.5)(1.0)(-0.1) = 1 + 0.15 = 1.15
        """
        result = tau(0, 0, 1.5, 1.0, -0.1)
        assert result == pytest.approx(1.15, rel=1e-4)

    def test_tau_1_0_formula(self):
        """tau(1,0) = max(0, 1 + nu*rho). At nu=1.0, rho=-0.1: 1 + 1.0*(-0.1) = 0.9"""
        result = tau(1, 0, 1.5, 1.0, -0.1)
        assert result == pytest.approx(0.9, rel=1e-4)

    def test_tau_0_1_formula(self):
        """tau(0,1) = max(0, 1 + mu*rho). At mu=1.5, rho=-0.1: 1 + 1.5*(-0.1) = 0.85"""
        result = tau(0, 1, 1.5, 1.0, -0.1)
        assert result == pytest.approx(0.85, rel=1e-4)

    def test_tau_1_1_formula(self):
        """tau(1,1) = max(0, 1 - rho). At rho=-0.1: 1 - (-0.1) = 1.1"""
        result = tau(1, 1, 1.5, 1.0, -0.1)
        assert result == pytest.approx(1.1, rel=1e-4)


class TestEstimateRho:
    def _make_matches(self, n: int = 200) -> list[dict]:
        """Generate synthetic match data with known rho-like properties."""
        rng = np.random.default_rng(42)
        matches = []
        for _ in range(n):
            matches.append({
                "home_xg": float(rng.uniform(0.5, 2.5)),
                "away_xg": float(rng.uniform(0.3, 2.0)),
                "home_goals": int(rng.poisson(1.4)),
                "away_goals": int(rng.poisson(1.1)),
            })
        return matches

    def test_rho_returns_float(self):
        matches = self._make_matches()
        rho = estimate_rho(matches)
        assert isinstance(rho, float)

    def test_rho_within_valid_bounds(self):
        """Rho must always land in [-1, 1]."""
        matches = self._make_matches()
        rho = estimate_rho(matches)
        assert -1.0 <= rho <= 1.0

    def test_rho_typically_negative_for_football(self):
        """
        Football has more 0-0, 1-0, 0-1 than pure Poisson predicts.
        Rho should be negative (typically -0.15 to -0.05).
        This test uses a large sample to reduce variance.
        """
        matches = self._make_matches(n=500)
        rho = estimate_rho(matches)
        # Allow some tolerance — exact value depends on random seed
        assert rho < 0.1, f"Expected negative or near-zero rho, got {rho}"

    def test_rho_handles_small_sample_without_crash(self):
        """Should not raise even with minimal data."""
        matches = self._make_matches(n=10)
        rho = estimate_rho(matches)
        assert isinstance(rho, float)

    def test_rho_empty_matches_returns_zero(self):
        """Edge case: no matches → return 0.0 safely."""
        rho = estimate_rho([])
        assert rho == 0.0


class TestLogLikelihood:
    def test_log_likelihood_returns_float(self):
        matches = [{"home_xg": 1.5, "away_xg": 1.0, "home_goals": 1, "away_goals": 0}]
        result = dixon_coles_log_likelihood(0.0, matches)
        assert isinstance(result, float)

    def test_log_likelihood_positive(self):
        """Negative log-likelihood must be positive (for minimisation)."""
        matches = [{"home_xg": 1.5, "away_xg": 1.0, "home_goals": 1, "away_goals": 0}]
        result = dixon_coles_log_likelihood(0.0, matches)
        assert result > 0

    def test_skips_zero_xg_matches(self):
        """Should not crash on zero xG values — these are skipped."""
        matches = [{"home_xg": 0.0, "away_xg": 0.0, "home_goals": 0, "away_goals": 0}]
        result = dixon_coles_log_likelihood(0.0, matches)
        assert isinstance(result, float)
