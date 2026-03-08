"""Tests for confidence tier assignment and tier-aware optimizer sort."""

from apex10.filters.gates import (
    TIER_RANK,
    ConfidenceTier,
    assign_tier,
)


class TestAssignTier:
    def test_s_tier_tight_agreement_high_edge(self):
        tier = assign_tier(
            lgbm_prob=0.85, xgb_prob=0.83,
            lgbm_edge=0.07, xgb_edge=0.06,
            has_flag=False,
        )
        assert tier == ConfidenceTier.S

    def test_a_tier_acceptable_agreement(self):
        tier = assign_tier(
            lgbm_prob=0.85, xgb_prob=0.81,
            lgbm_edge=0.05, xgb_edge=0.05,
            has_flag=False,
        )
        assert tier == ConfidenceTier.A

    def test_b_tier_when_flagged(self):
        """S or A conditions but flag → forced to B."""
        tier = assign_tier(
            lgbm_prob=0.85, xgb_prob=0.83,
            lgbm_edge=0.07, xgb_edge=0.06,
            has_flag=True,
        )
        assert tier == ConfidenceTier.B

    def test_b_tier_moderate_divergence(self):
        tier = assign_tier(
            lgbm_prob=0.85, xgb_prob=0.78,
            lgbm_edge=0.05, xgb_edge=0.04,
            has_flag=False,
        )
        assert tier == ConfidenceTier.B

    def test_x_tier_high_divergence(self):
        tier = assign_tier(
            lgbm_prob=0.85, xgb_prob=0.72,
            lgbm_edge=0.06, xgb_edge=0.04,
            has_flag=False,
        )
        assert tier == ConfidenceTier.X

    def test_x_tier_insufficient_edge(self):
        tier = assign_tier(
            lgbm_prob=0.80, xgb_prob=0.79,
            lgbm_edge=0.02, xgb_edge=0.02,
            has_flag=False,
        )
        assert tier == ConfidenceTier.X

    def test_tier_rank_order(self):
        """S < A < B < X."""
        assert (
            TIER_RANK[ConfidenceTier.S]
            < TIER_RANK[ConfidenceTier.A]
        )
        assert (
            TIER_RANK[ConfidenceTier.A]
            < TIER_RANK[ConfidenceTier.B]
        )
        assert (
            TIER_RANK[ConfidenceTier.B]
            < TIER_RANK[ConfidenceTier.X]
        )

    def test_s_tier_not_assigned_at_0_04_divergence(self):
        """Divergence 0.04 > 0.03 — cannot be S, should be A."""
        tier = assign_tier(
            lgbm_prob=0.85, xgb_prob=0.81,
            lgbm_edge=0.07, xgb_edge=0.07,
            has_flag=False,
        )
        assert tier == ConfidenceTier.A


class TestTierAwareSort:
    def test_s_tier_sorted_before_a(self):
        from apex10.ticket.optimizer import _sort_candidates
        from tests.test_gates import make_candidate

        s_leg = make_candidate(fixture_id=1, odds=1.25)
        s_leg.consensus_prob = 0.84
        s_leg.tier = ConfidenceTier.S

        a_leg = make_candidate(fixture_id=2, odds=1.38)
        a_leg.consensus_prob = 0.82
        a_leg.tier = ConfidenceTier.A

        sorted_legs = _sort_candidates([a_leg, s_leg])
        assert sorted_legs[0].tier == ConfidenceTier.S

    def test_within_same_tier_higher_odds_first(self):
        from apex10.ticket.optimizer import _sort_candidates
        from tests.test_gates import make_candidate

        low_odds = make_candidate(fixture_id=1, odds=1.22)
        low_odds.tier = ConfidenceTier.A
        low_odds.consensus_prob = 0.84

        high_odds = make_candidate(fixture_id=2, odds=1.38)
        high_odds.tier = ConfidenceTier.A
        high_odds.consensus_prob = 0.84

        sorted_legs = _sort_candidates([low_odds, high_odds])
        assert sorted_legs[0].odds >= sorted_legs[1].odds
