"""
Tests for score.py — leg evaluation, Brier computation, bank updates.
All DB and network calls mocked.
"""
from unittest.mock import MagicMock

import pytest

from apex10.score.score import (
    compute_brier_score,
    evaluate_leg,
    evaluate_ticket,
    update_bank,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


def result(home=2, away=0):
    return {"home_goals": home, "away_goals": away}


def leg(bet_type="home_win", prob=0.80, fixture_id=1):
    return {
        "bet_type": bet_type,
        "consensus_prob": prob,
        "fixture_id": fixture_id,
    }


# ── evaluate_leg ──────────────────────────────────────────────────────────


class TestEvaluateLeg:
    # ── home_win ──────────────────────────────────────────────────────────
    def test_home_win_correct(self):
        assert evaluate_leg(leg("home_win"), result(2, 0)) == "win"

    def test_home_win_loss(self):
        assert evaluate_leg(leg("home_win"), result(0, 1)) == "loss"

    def test_home_win_draw_is_loss(self):
        assert evaluate_leg(leg("home_win"), result(1, 1)) == "loss"

    # ── dnb_home ──────────────────────────────────────────────────────────
    def test_dnb_home_win(self):
        assert evaluate_leg(leg("dnb_home"), result(2, 1)) == "win"

    def test_dnb_home_draw_is_void(self):
        assert evaluate_leg(leg("dnb_home"), result(1, 1)) == "void"

    def test_dnb_home_loss(self):
        assert evaluate_leg(leg("dnb_home"), result(0, 1)) == "loss"

    # ── ah_minus_1_0 ──────────────────────────────────────────────────────
    def test_ah_minus_1_win_by_2(self):
        assert evaluate_leg(leg("ah_minus_1_0"), result(2, 0)) == "win"

    def test_ah_minus_1_push_win_by_1(self):
        assert evaluate_leg(leg("ah_minus_1_0"), result(2, 1)) == "void"

    def test_ah_minus_1_loss_draw(self):
        assert evaluate_leg(leg("ah_minus_1_0"), result(1, 1)) == "loss"

    def test_ah_minus_1_loss_away_win(self):
        assert evaluate_leg(leg("ah_minus_1_0"), result(0, 1)) == "loss"

    # ── ah_minus_1_5 ──────────────────────────────────────────────────────
    def test_ah_minus_1_5_win(self):
        assert evaluate_leg(leg("ah_minus_1_5"), result(3, 1)) == "win"

    def test_ah_minus_1_5_loss_win_by_1(self):
        assert evaluate_leg(leg("ah_minus_1_5"), result(2, 1)) == "loss"

    # ── over / under ──────────────────────────────────────────────────────
    def test_over_1_5_win(self):
        assert evaluate_leg(leg("over_1_5"), result(1, 1)) == "win"

    def test_over_1_5_loss(self):
        assert evaluate_leg(leg("over_1_5"), result(1, 0)) == "loss"

    def test_over_2_5_win(self):
        assert evaluate_leg(leg("over_2_5"), result(2, 1)) == "win"

    def test_over_2_5_loss(self):
        assert evaluate_leg(leg("over_2_5"), result(1, 1)) == "loss"

    def test_under_3_5_win(self):
        assert evaluate_leg(leg("under_3_5"), result(2, 1)) == "win"

    def test_under_3_5_loss(self):
        assert evaluate_leg(leg("under_3_5"), result(2, 2)) == "loss"

    def test_under_3_5_exactly_3_goals(self):
        assert evaluate_leg(leg("under_3_5"), result(2, 1)) == "win"

    def test_under_3_5_exactly_4_goals(self):
        assert evaluate_leg(leg("under_3_5"), result(2, 2)) == "loss"

    # ── btts_no ───────────────────────────────────────────────────────────
    def test_btts_no_win_home_clean_sheet(self):
        assert evaluate_leg(leg("btts_no"), result(2, 0)) == "win"

    def test_btts_no_win_away_clean_sheet(self):
        assert evaluate_leg(leg("btts_no"), result(0, 1)) == "win"

    def test_btts_no_loss_both_score(self):
        assert evaluate_leg(leg("btts_no"), result(1, 1)) == "loss"

    # ── double chance ─────────────────────────────────────────────────────
    def test_dc_1x_win_on_home_win(self):
        assert evaluate_leg(leg("dc_1x"), result(2, 0)) == "win"

    def test_dc_1x_win_on_draw(self):
        assert evaluate_leg(leg("dc_1x"), result(1, 1)) == "win"

    def test_dc_1x_loss_on_away_win(self):
        assert evaluate_leg(leg("dc_1x"), result(0, 1)) == "loss"

    def test_dc_x2_win_on_away_win(self):
        assert evaluate_leg(leg("dc_x2"), result(0, 2)) == "win"

    def test_dc_x2_win_on_draw(self):
        assert evaluate_leg(leg("dc_x2"), result(1, 1)) == "win"

    def test_dc_x2_loss_on_home_win(self):
        assert evaluate_leg(leg("dc_x2"), result(2, 0)) == "loss"

    # ── home_scores ───────────────────────────────────────────────────────
    def test_home_scores_win(self):
        assert evaluate_leg(leg("home_scores"), result(1, 0)) == "win"

    def test_home_scores_loss(self):
        assert evaluate_leg(leg("home_scores"), result(0, 1)) == "loss"

    # ── edge cases ────────────────────────────────────────────────────────
    def test_missing_goals_returns_void(self):
        assert (
            evaluate_leg(
                leg("home_win"),
                {"home_goals": None, "away_goals": None},
            )
            == "void"
        )

    def test_unknown_bet_type_returns_void(self):
        assert evaluate_leg(leg("mystery_bet"), result(1, 0)) == "void"


# ── evaluate_ticket ───────────────────────────────────────────────────────


class TestEvaluateTicket:
    def test_all_legs_win(self):
        legs = [leg("home_win", fixture_id=i) for i in range(3)]
        results = {i: result(2, 0) for i in range(3)}
        ev = evaluate_ticket(legs, results)
        assert ev["ticket_result"] == "win"
        assert all(r == "win" for r in ev["leg_results"])

    def test_one_leg_loss_makes_ticket_loss(self):
        legs = [leg("home_win", fixture_id=i) for i in range(3)]
        results = {0: result(2, 0), 1: result(2, 0), 2: result(0, 1)}
        ev = evaluate_ticket(legs, results)
        assert ev["ticket_result"] == "loss"

    def test_missing_result_marked_no_data(self):
        legs = [leg("home_win", fixture_id=99)]
        ev = evaluate_ticket(legs, {})
        assert ev["leg_results"][0] == "no_data"

    def test_all_void_legs_ticket_no_data(self):
        legs = [leg("dnb_home", fixture_id=i) for i in range(2)]
        results = {i: result(1, 1) for i in range(2)}
        ev = evaluate_ticket(legs, results)
        assert ev["ticket_result"] == "no_data"

    def test_void_legs_excluded_from_win_check(self):
        """Ticket with one void and one win should still be win."""
        legs = [leg("dnb_home", fixture_id=0), leg("home_win", fixture_id=1)]
        results = {0: result(1, 1), 1: result(2, 0)}
        ev = evaluate_ticket(legs, results)
        assert ev["ticket_result"] == "win"
        assert ev["legs_void"] == 1

    def test_counts_evaluated_legs(self):
        legs = [leg("home_win", fixture_id=i) for i in range(4)]
        results = {i: result(2, 0) for i in range(4)}
        ev = evaluate_ticket(legs, results)
        assert ev["legs_evaluated"] == 4


# ── compute_brier_score ───────────────────────────────────────────────────


class TestComputeBrierScore:
    def test_perfect_prediction_low_brier(self):
        legs_data = [leg("home_win", prob=0.99)]
        leg_results = ["win"]
        score = compute_brier_score(legs_data, leg_results)
        assert score < 0.01

    def test_worst_prediction_high_brier(self):
        legs_data = [leg("home_win", prob=0.99)]
        leg_results = ["loss"]
        score = compute_brier_score(legs_data, leg_results)
        assert score > 0.9

    def test_void_legs_excluded(self):
        legs_data = [leg("home_win", prob=0.90), leg("dnb_home", prob=0.80)]
        leg_results = ["win", "void"]
        score = compute_brier_score(legs_data, leg_results)
        expected = (0.90 - 1.0) ** 2
        assert score == pytest.approx(expected, abs=1e-4)

    def test_returns_none_with_no_decisive_legs(self):
        legs_data = [leg("dnb_home")]
        leg_results = ["void"]
        assert compute_brier_score(legs_data, leg_results) is None

    def test_mean_over_multiple_legs(self):
        legs_data = [leg("home_win", prob=0.80), leg("home_win", prob=0.70)]
        leg_results = ["win", "win"]
        score = compute_brier_score(legs_data, leg_results)
        expected = ((0.80 - 1.0) ** 2 + (0.70 - 1.0) ** 2) / 2
        assert score == pytest.approx(expected, abs=1e-4)

    def test_brier_score_between_zero_and_one(self):
        legs_data = [leg("home_win", prob=0.75)]
        for lr in ["win", "loss"]:
            score = compute_brier_score(legs_data, [lr])
            assert 0.0 <= score <= 1.0


# ── update_bank ───────────────────────────────────────────────────────────


class TestUpdateBank:
    def _mock_db(self, current_bank=10000.0):
        db = MagicMock()
        select_chain = db.table.return_value.select.return_value
        select_chain.eq.return_value.execute.return_value.data = [
            {
                "current_bank": current_bank,
                "initial_bank": 10000.0,
                "extracted_profit": 0.0,
                "total_wins": 0,
                "total_losses": 0,
            }
        ]
        update_chain = db.table.return_value.update.return_value
        update_chain.eq.return_value.execute.return_value = MagicMock()
        return db

    def test_win_increases_bank(self):
        db = self._mock_db(current_bank=10000.0)
        result = update_bank(db, "win", stake=100.0, combined_odds=12.0)
        assert result["new_bank"] > 10000.0

    def test_loss_decreases_bank(self):
        db = self._mock_db(current_bank=10000.0)
        result = update_bank(db, "loss", stake=100.0, combined_odds=12.0)
        assert result["new_bank"] == pytest.approx(9900.0)

    def test_void_no_change(self):
        db = self._mock_db(current_bank=10000.0)
        result = update_bank(db, "void", stake=100.0, combined_odds=12.0)
        assert result["new_bank"] == pytest.approx(10000.0)

    def test_no_data_no_change(self):
        db = self._mock_db(current_bank=10000.0)
        result = update_bank(db, "no_data", stake=100.0, combined_odds=12.0)
        assert result["new_bank"] == pytest.approx(10000.0)

    def test_raises_if_bank_not_initialised(self):
        db = MagicMock()
        select_chain = db.table.return_value.select.return_value
        select_chain.eq.return_value.execute.return_value.data = []
        with pytest.raises(ValueError, match="bank_state not initialised"):
            update_bank(db, "loss", stake=100.0, combined_odds=10.0)

