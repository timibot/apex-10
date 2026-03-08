"""Tests for ticket optimizer and Monte Carlo."""
import pytest

from apex10.filters.gates import Candidate
from apex10.ticket.optimizer import (
    TicketLeg,
    _sort_candidates,
    build_ticket,
    monte_carlo_win_probability,
    quarter_kelly_stake,
)


def make_candidate(**kwargs) -> Candidate:
    """Import-free candidate factory for optimizer tests."""
    defaults = {
        "fixture_id": 1,
        "league": "EPL",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "bet_type": "home_win",
        "odds": 1.35,
        "lgbm_prob": 0.84,
        "xgb_prob": 0.82,
        "opening_odds": 1.33,
        "key_player_absent_home": 0,
        "key_player_absent_away": 0,
        "features_complete": True,
    }
    defaults.update(kwargs)
    return Candidate(**defaults)


def make_qualified_candidate(
    fixture_id=1, odds=1.35, consensus_prob=0.80, league="EPL"
) -> Candidate:
    c = make_candidate(
        fixture_id=fixture_id,
        league=league,
        odds=odds,
        lgbm_prob=consensus_prob + 0.01,
        xgb_prob=consensus_prob - 0.01,
        opening_odds=odds - 0.01,
    )
    c.consensus_prob = consensus_prob
    c.lgbm_edge = 0.05
    c.xgb_edge = 0.05
    return c


class TestSortCandidates:
    def test_same_tier_sorted_by_odds_desc(self):
        """Within same tier, higher odds first."""
        c1 = make_qualified_candidate(1, odds=1.30, consensus_prob=0.75)
        c2 = make_qualified_candidate(2, odds=1.38, consensus_prob=0.85)
        sorted_c = _sort_candidates([c1, c2])
        assert sorted_c[0].odds >= sorted_c[1].odds

    def test_same_confidence_sorted_by_odds_desc(self):
        c1 = make_qualified_candidate(1, odds=1.25, consensus_prob=0.80)
        c2 = make_qualified_candidate(2, odds=1.38, consensus_prob=0.80)
        sorted_c = _sort_candidates([c1, c2])
        assert sorted_c[0].odds >= sorted_c[1].odds


class TestBuildTicket:
    def test_builds_ticket_when_enough_legs(self):
        candidates = [
            make_qualified_candidate(i, odds=1.35) for i in range(10)
        ]
        ticket = build_ticket(candidates)
        assert not ticket.no_ticket

    def test_combined_odds_above_target(self):
        candidates = [
            make_qualified_candidate(i, odds=1.35) for i in range(10)
        ]
        ticket = build_ticket(candidates)
        assert ticket.combined_odds >= 10.0

    def test_no_ticket_when_no_candidates(self):
        ticket = build_ticket([])
        assert ticket.no_ticket is True

    def test_stops_at_target(self):
        """Should not add more legs than necessary."""
        candidates = [
            make_qualified_candidate(i, odds=1.40) for i in range(15)
        ]
        ticket = build_ticket(candidates)
        assert ticket.combined_odds < 20.0

    def test_no_ticket_if_product_cannot_reach_target(self):
        candidates = [
            make_qualified_candidate(i, odds=1.25) for i in range(2)
        ]
        ticket = build_ticket(candidates)
        assert ticket.no_ticket is True

    def test_ticket_contains_legs(self):
        candidates = [
            make_qualified_candidate(i, odds=1.35) for i in range(10)
        ]
        ticket = build_ticket(candidates)
        assert len(ticket.legs) > 0

    def test_ticket_has_monte_carlo_stats(self):
        candidates = [
            make_qualified_candidate(i, odds=1.35) for i in range(10)
        ]
        ticket = build_ticket(candidates)
        assert 0 < ticket.simulated_win_rate < 1
        assert ticket.ci_low < ticket.simulated_win_rate < ticket.ci_high


class TestMonteCarlo:
    def _legs(self, n=6, prob=0.85, odds=1.35):
        return [
            TicketLeg(
                fixture_id=i,
                league="EPL",
                home_team=f"Home{i}",
                away_team=f"Away{i}",
                bet_type="home_win",
                odds=odds,
                consensus_prob=prob,
                lgbm_prob=prob,
                xgb_prob=prob,
                lgbm_edge=0.05,
                xgb_edge=0.05,
            )
            for i in range(n)
        ]

    def test_returns_three_values(self):
        result = monte_carlo_win_probability(self._legs())
        assert len(result) == 3

    def test_win_rate_in_valid_range(self):
        win_rate, _, _ = monte_carlo_win_probability(self._legs())
        assert 0.0 <= win_rate <= 1.0

    def test_ci_ordered_correctly(self):
        _, ci_low, ci_high = monte_carlo_win_probability(self._legs())
        assert ci_low <= ci_high

    def test_high_prob_legs_give_reasonable_win_rate(self):
        """6 legs at 85% each → ticket win ≈ 0.85^6 ≈ 0.38."""
        win_rate, _, _ = monte_carlo_win_probability(
            self._legs(n=6, prob=0.85)
        )
        assert 0.25 < win_rate < 0.55

    def test_win_rate_decreases_with_more_legs(self):
        rate_5, _, _ = monte_carlo_win_probability(
            self._legs(n=5, prob=0.85)
        )
        rate_8, _, _ = monte_carlo_win_probability(
            self._legs(n=8, prob=0.85)
        )
        assert rate_5 > rate_8


class TestQuarterKelly:
    def test_returns_positive_stake_with_edge(self):
        # edge = 0.12*12 - 1 = 0.44 > 0 → positive stake
        stake = quarter_kelly_stake(
            bank=100_000, win_prob=0.12, combined_odds=12.0
        )
        assert stake > 0

    def test_returns_zero_on_negative_edge(self):
        stake = quarter_kelly_stake(
            bank=100_000, win_prob=0.01, combined_odds=10.0
        )
        assert stake == 0.0

    def test_stake_scales_with_bank(self):
        s1 = quarter_kelly_stake(
            bank=50_000, win_prob=0.12, combined_odds=12.0
        )
        s2 = quarter_kelly_stake(
            bank=100_000, win_prob=0.12, combined_odds=12.0
        )
        assert pytest.approx(s2, rel=0.01) == s1 * 2

    def test_quarter_kelly_smaller_than_full_kelly(self):
        full = quarter_kelly_stake(
            bank=100_000, win_prob=0.12,
            combined_odds=12.0, kelly_fraction=1.0,
        )
        quarter = quarter_kelly_stake(
            bank=100_000, win_prob=0.12,
            combined_odds=12.0, kelly_fraction=0.25,
        )
        assert quarter < full
