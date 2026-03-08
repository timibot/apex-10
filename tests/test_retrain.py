"""Tests for annual retrain orchestrator."""
from unittest.mock import MagicMock, patch

import pytest

from apex10.live.retrain import get_active_model_brier, run_annual_retrain


class TestGetActiveModelBrier:
    def test_returns_brier_when_model_exists(self):
        db = MagicMock()
        chain = db.table.return_value.select.return_value.eq.return_value
        chain.order.return_value.limit.return_value\
            .execute.return_value.data = [
            {"brier_score": 0.18, "version": 1}
        ]
        assert get_active_model_brier(db) == pytest.approx(0.18)

    def test_returns_none_when_no_active_model(self):
        db = MagicMock()
        chain = db.table.return_value.select.return_value.eq.return_value
        chain.order.return_value.limit.return_value\
            .execute.return_value.data = []
        assert get_active_model_brier(db) is None


class TestRunAnnualRetrain:
    def _mock_training_success(self, lgbm=0.17, xgb=0.18):
        return {
            "passed": True,
            "version": 2,
            "lgbm_brier": lgbm,
            "xgb_brier": xgb,
            "shap_lgbm_pruning": [],
            "shap_xgb_pruning": [],
            "subspacing_valid": True,
        }

    def _mock_training_failure(self):
        return {
            "passed": False,
            "version": 0,
            "lgbm_brier": 0.25,
            "xgb_brier": 0.26,
        }

    def test_deploys_when_new_model_better(self):
        with (
            patch(
                "apex10.live.retrain.run_training",
                return_value=self._mock_training_success(
                    lgbm=0.17, xgb=0.18
                ),
            ),
            patch("apex10.live.retrain.get_client") as mock_db_fn,
            patch(
                "apex10.live.retrain.get_active_model_brier",
                return_value=0.19,
            ),
            patch(
                "apex10.live.retrain.retrain_complete",
                return_value=True,
            ),
        ):
            mock_db_fn.return_value = MagicMock()
            result = run_annual_retrain(n_trials=1)
        assert result["deployed"] is True

    def test_keeps_current_when_new_model_worse(self):
        with (
            patch(
                "apex10.live.retrain.run_training",
                return_value=self._mock_training_success(
                    lgbm=0.20, xgb=0.21
                ),
            ),
            patch("apex10.live.retrain.get_client") as mock_db_fn,
            patch(
                "apex10.live.retrain.get_active_model_brier",
                return_value=0.17,
            ),
            patch(
                "apex10.live.retrain.retrain_complete",
                return_value=True,
            ),
        ):
            mock_db = MagicMock()
            mock_db_fn.return_value = mock_db
            result = run_annual_retrain(n_trials=1)
        assert result["deployed"] is False

    def test_deploys_when_no_active_model(self):
        with (
            patch(
                "apex10.live.retrain.run_training",
                return_value=self._mock_training_success(),
            ),
            patch("apex10.live.retrain.get_client") as mock_db_fn,
            patch(
                "apex10.live.retrain.get_active_model_brier",
                return_value=None,
            ),
            patch(
                "apex10.live.retrain.retrain_complete",
                return_value=True,
            ),
        ):
            mock_db_fn.return_value = MagicMock()
            result = run_annual_retrain(n_trials=1)
        assert result["deployed"] is True

    def test_does_not_deploy_when_training_fails_gate(self):
        with (
            patch(
                "apex10.live.retrain.run_training",
                return_value=self._mock_training_failure(),
            ),
            patch("apex10.live.retrain.get_client") as mock_db_fn,
            patch(
                "apex10.live.retrain.get_active_model_brier",
                return_value=0.18,
            ),
            patch(
                "apex10.live.retrain.retrain_complete",
                return_value=True,
            ),
        ):
            mock_db_fn.return_value = MagicMock()
            result = run_annual_retrain(n_trials=1)
        assert result["deployed"] is False
        assert "Brier gate" in result["reason"]

    def test_result_contains_required_keys(self):
        with (
            patch(
                "apex10.live.retrain.run_training",
                return_value=self._mock_training_success(),
            ),
            patch("apex10.live.retrain.get_client") as mock_db_fn,
            patch(
                "apex10.live.retrain.get_active_model_brier",
                return_value=0.19,
            ),
            patch(
                "apex10.live.retrain.retrain_complete",
                return_value=True,
            ),
        ):
            mock_db_fn.return_value = MagicMock()
            result = run_annual_retrain(n_trials=1)
        for key in [
            "deployed",
            "reason",
            "new_version",
            "new_lgbm_brier",
            "new_xgb_brier",
            "previous_brier",
            "retrain_date",
        ]:
            assert key in result
