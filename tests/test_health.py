"""Tests for health monitor — stake multiplier, Brier breach/recovery."""
from unittest.mock import MagicMock, patch

import pytest

from apex10.config import MODEL
from apex10.live.health import (
    STAKE_MULTIPLIER_NORMAL,
    get_adjusted_stake,
    get_rolling_brier,
    get_stake_multiplier,
    run_health_check,
)


def _mock_db(rolling_brier=None, multiplier=1.0):
    db = MagicMock()

    def table_side(name):
        t = MagicMock()
        if name == "brier_log" and rolling_brier is not None:
            scores = [rolling_brier] * MODEL.ROLLING_BRIER_WINDOW
            chain = t.select.return_value.not_.is_.return_value
            chain.order.return_value.limit.return_value\
                .execute.return_value.data = [
                {"brier_score": s} for s in scores
            ]
        elif name == "brier_log":
            chain = t.select.return_value.not_.is_.return_value
            chain.order.return_value.limit.return_value\
                .execute.return_value.data = []
        elif name == "bank_state":
            t.select.return_value.eq.return_value\
                .execute.return_value.data = [
                {"stake_multiplier": multiplier}
            ]
            t.update.return_value.eq.return_value\
                .execute.return_value = MagicMock()
        return t

    db.table.side_effect = table_side
    return db


class TestGetStakeMultiplier:
    def test_returns_1_when_normal(self):
        db = _mock_db(multiplier=1.0)
        assert get_stake_multiplier(db) == 1.0

    def test_returns_half_when_halved(self):
        db = _mock_db(multiplier=0.5)
        assert get_stake_multiplier(db) == 0.5

    def test_defaults_to_1_when_no_data(self):
        db = MagicMock()

        def table_side(name):
            t = MagicMock()
            if name == "bank_state":
                t.select.return_value.eq.return_value\
                    .execute.return_value.data = []
            return t

        db.table.side_effect = table_side
        assert get_stake_multiplier(db) == STAKE_MULTIPLIER_NORMAL


class TestGetRollingBrier:
    def test_returns_none_below_window(self):
        db = MagicMock()

        def table_side(name):
            t = MagicMock()
            if name == "brier_log":
                chain = t.select.return_value.not_.is_.return_value
                chain.order.return_value.limit.return_value\
                    .execute.return_value.data = [
                    {"brier_score": 0.18}  # Only 1 — below window
                ]
            return t

        db.table.side_effect = table_side
        assert get_rolling_brier(db) is None

    def test_returns_average_at_full_window(self):
        db = MagicMock()
        scores = [0.18] * MODEL.ROLLING_BRIER_WINDOW

        def table_side(name):
            t = MagicMock()
            if name == "brier_log":
                chain = t.select.return_value.not_.is_.return_value
                chain.order.return_value.limit.return_value\
                    .execute.return_value.data = [
                    {"brier_score": s} for s in scores
                ]
            return t

        db.table.side_effect = table_side
        result = get_rolling_brier(db)
        assert result == pytest.approx(0.18, abs=1e-4)


class TestRunHealthCheck:
    def test_no_action_healthy_brier(self):
        db = _mock_db(rolling_brier=0.17, multiplier=1.0)
        with patch("apex10.live.health.brier_breach") as mock_alert:
            result = run_health_check(db)
        assert result["action_taken"] == "none"
        mock_alert.assert_not_called()

    def test_halves_stakes_on_breach(self):
        db = _mock_db(
            rolling_brier=MODEL.BRIER_LIVE_ALERT + 0.01,
            multiplier=1.0,
        )
        with patch("apex10.live.health.brier_breach", return_value=True):
            result = run_health_check(db)
        assert result["action_taken"] == "stakes_halved"

    def test_no_duplicate_halving(self):
        """If stakes already halved, no second alert should fire."""
        db = _mock_db(
            rolling_brier=MODEL.BRIER_LIVE_ALERT + 0.01,
            multiplier=0.5,
        )
        with patch("apex10.live.health.brier_breach") as mock_alert:
            result = run_health_check(db)
        mock_alert.assert_not_called()
        assert result.get("note") == "Already halved"

    def test_restores_stakes_on_recovery(self):
        """Brier recovers below gate while stakes are halved → restore."""
        db = _mock_db(
            rolling_brier=MODEL.BRIER_GATE - 0.01,
            multiplier=0.5,
        )
        with patch(
            "apex10.live.health.brier_recovered", return_value=True
        ):
            result = run_health_check(db)
        assert result["action_taken"] == "stakes_restored"

    def test_no_restore_if_not_halved(self):
        """Brier healthy and stakes already normal → no action."""
        db = _mock_db(
            rolling_brier=MODEL.BRIER_GATE - 0.01,
            multiplier=1.0,
        )
        result = run_health_check(db)
        assert result["action_taken"] == "none"

    def test_returns_none_action_when_no_brier_data(self):
        db = _mock_db(rolling_brier=None, multiplier=1.0)
        result = run_health_check(db)
        assert result["rolling_brier"] is None
        assert "note" in result


class TestGetAdjustedStake:
    def test_full_stake_when_multiplier_normal(self):
        db = _mock_db(multiplier=1.0)
        adjusted = get_adjusted_stake(1000.0, db)
        assert adjusted == pytest.approx(1000.0)

    def test_halved_stake_when_multiplier_half(self):
        db = _mock_db(multiplier=0.5)
        adjusted = get_adjusted_stake(1000.0, db)
        assert adjusted == pytest.approx(500.0)

    def test_zero_stake_stays_zero(self):
        db = _mock_db(multiplier=0.5)
        assert get_adjusted_stake(0.0, db) == 0.0
