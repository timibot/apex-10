"""Tests for 70/30 reinvestment rule."""
from unittest.mock import MagicMock

import pytest

from apex10.score.score import (
    EXTRACT_FRACTION,
    REINVEST_FRACTION,
    update_bank,
)


def _mock_db(
    current_bank=10000.0, extracted=0.0, wins=0, losses=0
):
    db = MagicMock()

    def table_side(name):
        t = MagicMock()
        if name == "bank_state":
            t.select.return_value.eq.return_value\
                .execute.return_value.data = [
                {
                    "current_bank": current_bank,
                    "initial_bank": 10000.0,
                    "extracted_profit": extracted,
                    "total_wins": wins,
                    "total_losses": losses,
                }
            ]
            t.update.return_value.eq.return_value\
                .execute.return_value = MagicMock()
        return t

    db.table.side_effect = table_side
    return db


class TestReinvestmentRule:
    def test_win_reinvests_70_pct(self):
        db = _mock_db(current_bank=10000.0)
        result = update_bank(
            db, "win", stake=100.0, combined_odds=12.0
        )
        gross_profit = 100.0 * (12.0 - 1.0)  # = 1100
        expected_reinvest = round(
            gross_profit * REINVEST_FRACTION, 2
        )
        expected_new_bank = round(
            10000.0 + expected_reinvest, 2
        )
        assert result["new_bank"] == pytest.approx(
            expected_new_bank, rel=0.001
        )

    def test_win_extracts_30_pct(self):
        db = _mock_db(current_bank=10000.0)
        result = update_bank(
            db, "win", stake=100.0, combined_odds=12.0
        )
        gross_profit = 100.0 * (12.0 - 1.0)
        expected_extract = round(
            gross_profit * EXTRACT_FRACTION, 2
        )
        assert result["extracted_this_week"] == pytest.approx(
            expected_extract, rel=0.001
        )

    def test_reinvest_plus_extract_equals_gross_profit(self):
        db = _mock_db(current_bank=10000.0)
        result = update_bank(
            db, "win", stake=100.0, combined_odds=12.0
        )
        gross_profit = 100.0 * (12.0 - 1.0)
        total = result["extracted_this_week"] + (
            result["new_bank"] - 10000.0
        )
        assert total == pytest.approx(gross_profit, rel=0.001)

    def test_loss_deducts_full_stake(self):
        db = _mock_db(current_bank=10000.0)
        result = update_bank(
            db, "loss", stake=100.0, combined_odds=12.0
        )
        assert result["new_bank"] == pytest.approx(9900.0)
        assert result["extracted_this_week"] == 0.0

    def test_void_no_change(self):
        db = _mock_db(current_bank=10000.0)
        result = update_bank(
            db, "void", stake=100.0, combined_odds=12.0
        )
        assert result["new_bank"] == pytest.approx(10000.0)

    def test_fractions_sum_to_one(self):
        assert REINVEST_FRACTION + EXTRACT_FRACTION == pytest.approx(
            1.0
        )

    def test_total_extracted_accumulates(self):
        db = _mock_db(current_bank=10000.0, extracted=500.0)
        result = update_bank(
            db, "win", stake=100.0, combined_odds=12.0
        )
        assert result["total_extracted"] > 500.0


class TestBankRaisesWithoutInit:
    def test_raises_on_empty_bank_state(self):
        db = MagicMock()
        db.table.return_value.select.return_value.eq.return_value\
            .execute.return_value.data = []
        with pytest.raises(
            ValueError, match="bank_state not initialised"
        ):
            update_bank(
                db, "loss", stake=100.0, combined_odds=10.0
            )
