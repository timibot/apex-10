"""Tests for lineup sub-model logic."""
import numpy as np
import pytest

from apex10.models.lineup_model import (
    LINEUP_FEATURES,
    QUALITY_DEGRADATION,
    build_lineup_features,
    recalculate_odds_degraded,
)


class TestRecalculateOddsDegraded:
    def test_odds_increase_with_degradation(self):
        """Weaker squad → lower win prob → higher odds."""
        degraded = recalculate_odds_degraded(1.30)
        assert degraded > 1.30

    def test_heavily_favoured_team_still_increases(self):
        degraded = recalculate_odds_degraded(1.20)
        assert degraded > 1.20

    def test_formula_correctness(self):
        degraded = recalculate_odds_degraded(1.30)
        expected = 1.0 / ((1.0 / 1.30) * QUALITY_DEGRADATION)
        assert degraded == pytest.approx(expected, rel=0.01)

    def test_handles_zero_odds(self):
        result = recalculate_odds_degraded(0.0)
        assert result == 0.0

    def test_handles_exactly_1_odds(self):
        result = recalculate_odds_degraded(1.0)
        assert result == 1.0


class TestBuildLineupFeatures:
    def test_returns_correct_shape(self):
        row = {
            "days_to_next_home": 7,
            "competition_weight_home": 0.7,
            "next_competition_weight_home": 0.9,
            "squad_depth_ratio": 0.85,
            "manager_rotation_rate": 0.25,
            "fixture_congestion_home": 4,
        }
        features = build_lineup_features(row)
        assert features.shape == (1, len(LINEUP_FEATURES))

    def test_uses_defaults_for_missing_fields(self):
        features = build_lineup_features({})
        assert features.shape == (1, len(LINEUP_FEATURES))
        assert not np.isnan(features).any()

    def test_all_values_finite(self):
        features = build_lineup_features({})
        assert np.isfinite(features).all()


class TestLineupFeatureList:
    def test_correct_feature_count(self):
        assert len(LINEUP_FEATURES) == 7

    def test_all_features_are_strings(self):
        assert all(isinstance(f, str) for f in LINEUP_FEATURES)
