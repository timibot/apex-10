"""
Tests for paper trading exit gate logic.
Mocks DB — tests the three conditions independently and in combination.
"""
from unittest.mock import MagicMock

from apex10.config import MODEL, STAKING
from apex10.score.score import check_paper_trading_exit


def _mock_db(
    ticket_count=20,
    brier_scores=None,
    current_bank=10500.0,
    initial_bank=10000.0,
):
    """Build a mock DB that returns controllable values for each condition."""
    if brier_scores is None:
        brier_scores = [0.18] * 10

    db = MagicMock()

    def table_side_effect(table_name):
        mock_table = MagicMock()

        if table_name == "tickets":
            select_chain = mock_table.select.return_value
            select_chain.not_.eq.return_value.execute.return_value.count = (
                ticket_count
            )

        elif table_name == "brier_log":
            select_chain = mock_table.select.return_value
            order_chain = (
                select_chain.not_.is_.return_value.order.return_value
            )
            order_chain.limit.return_value.execute.return_value.data = [
                {"brier_score": s} for s in brier_scores
            ]

        elif table_name == "bank_state":
            select_chain = mock_table.select.return_value
            select_chain.eq.return_value.execute.return_value.data = [
                {
                    "current_bank": current_bank,
                    "initial_bank": initial_bank,
                }
            ]

        return mock_table

    db.table.side_effect = table_side_effect
    return db


class TestPaperTradingExitConditions:
    def test_all_conditions_met_returns_ready(self):
        db = _mock_db(
            ticket_count=20,
            brier_scores=[0.18] * 10,
            current_bank=10500.0,
            initial_bank=10000.0,
        )
        result = check_paper_trading_exit(db)
        assert result["ready_for_live"] is True

    def test_condition_1_fails_below_20_tickets(self):
        db = _mock_db(ticket_count=15)
        result = check_paper_trading_exit(db)
        assert result["conditions"]["c1_ticket_count"]["passed"] is False
        assert result["ready_for_live"] is False

    def test_condition_1_passes_at_exactly_20(self):
        db = _mock_db(ticket_count=20)
        result = check_paper_trading_exit(db)
        assert result["conditions"]["c1_ticket_count"]["passed"] is True

    def test_condition_2_fails_high_brier_variance(self):
        # variance of alternating 0.05/0.40 = 0.034 > 0.02 threshold
        brier_scores = [
            0.05, 0.40, 0.05, 0.40, 0.05,
            0.40, 0.05, 0.40, 0.05, 0.40,
        ]
        db = _mock_db(ticket_count=20, brier_scores=brier_scores)
        result = check_paper_trading_exit(db)
        assert result["conditions"]["c2_brier_variance"]["passed"] is False
        assert result["ready_for_live"] is False

    def test_condition_2_fails_fewer_than_10_scores(self):
        db = _mock_db(ticket_count=20, brier_scores=[0.18] * 8)
        result = check_paper_trading_exit(db)
        assert result["conditions"]["c2_brier_variance"]["passed"] is False

    def test_condition_2_passes_stable_scores(self):
        db = _mock_db(ticket_count=20, brier_scores=[0.19] * 10)
        result = check_paper_trading_exit(db)
        assert result["conditions"]["c2_brier_variance"]["passed"] is True

    def test_condition_3_fails_roi_below_floor(self):
        db = _mock_db(
            ticket_count=20,
            current_bank=8000.0,
            initial_bank=10000.0,
        )
        result = check_paper_trading_exit(db)
        assert result["conditions"]["c3_simulated_roi"]["passed"] is False
        assert result["ready_for_live"] is False

    def test_condition_3_passes_positive_roi(self):
        db = _mock_db(
            ticket_count=20,
            current_bank=11000.0,
            initial_bank=10000.0,
        )
        result = check_paper_trading_exit(db)
        assert result["conditions"]["c3_simulated_roi"]["passed"] is True

    def test_condition_3_passes_at_floor(self):
        db = _mock_db(
            ticket_count=20,
            current_bank=9500.0,
            initial_bank=10000.0,
        )
        result = check_paper_trading_exit(db)
        assert result["conditions"]["c3_simulated_roi"]["passed"] is True

    def test_all_three_must_pass_simultaneously(self):
        """Two conditions passing with one failing = not ready."""
        db = _mock_db(
            ticket_count=25,
            brier_scores=[0.18] * 10,
            current_bank=8000.0,
            initial_bank=10000.0,
        )
        result = check_paper_trading_exit(db)
        assert result["ready_for_live"] is False
        assert result["conditions"]["c1_ticket_count"]["passed"] is True
        assert result["conditions"]["c2_brier_variance"]["passed"] is True
        assert result["conditions"]["c3_simulated_roi"]["passed"] is False

    def test_result_contains_all_condition_keys(self):
        db = _mock_db()
        result = check_paper_trading_exit(db)
        assert "ready_for_live" in result
        assert "conditions" in result
        for key in [
            "c1_ticket_count",
            "c2_brier_variance",
            "c3_simulated_roi",
        ]:
            assert key in result["conditions"]

    def test_each_condition_has_value_threshold_passed(self):
        db = _mock_db()
        result = check_paper_trading_exit(db)
        for cond in result["conditions"].values():
            assert "value" in cond
            assert "threshold" in cond
            assert "passed" in cond

    def test_thresholds_match_config(self):
        db = _mock_db()
        result = check_paper_trading_exit(db)
        c = result["conditions"]
        assert c["c1_ticket_count"]["threshold"] == MODEL.MIN_PAPER_TICKETS
        assert (
            c["c2_brier_variance"]["threshold"] == MODEL.BRIER_VARIANCE_GATE
        )
        assert (
            c["c3_simulated_roi"]["threshold"] == STAKING.SIMULATED_ROI_FLOOR
        )
