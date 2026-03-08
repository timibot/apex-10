"""
Tests for apex10/config.py
Every constant is verified. Any regression in config values breaks a build.
"""
import os
from unittest.mock import patch

import pytest


class TestOddsConfig:
    def test_min_odds_value(self):
        from apex10.config import ODDS
        assert ODDS.MIN_ODDS == 1.20

    def test_max_odds_value(self):
        from apex10.config import ODDS
        assert ODDS.MAX_ODDS == 1.49

    def test_target_product(self):
        from apex10.config import ODDS
        assert ODDS.TARGET_PRODUCT == 10.0

    def test_min_edge(self):
        from apex10.config import ODDS
        assert ODDS.MIN_EDGE == 0.04

    def test_max_divergence(self):
        from apex10.config import ODDS
        assert ODDS.MAX_DIVERGENCE == 0.10

    def test_max_drift(self):
        from apex10.config import ODDS
        assert ODDS.MAX_DRIFT == 0.08

    def test_max_legs_per_league(self):
        from apex10.config import ODDS
        assert ODDS.MAX_LEGS_PER_LEAGUE == 3

    def test_odds_config_is_frozen(self):
        """Config must be immutable — no accidental mutation in runtime."""
        from apex10.config import ODDS
        with pytest.raises((AttributeError, TypeError)):
            ODDS.MIN_ODDS = 1.10  # type: ignore


class TestModelConfig:
    def test_brier_gate(self):
        from apex10.config import MODEL
        assert MODEL.BRIER_GATE == 0.20

    def test_brier_live_alert(self):
        from apex10.config import MODEL
        assert MODEL.BRIER_LIVE_ALERT == 0.24

    def test_brier_live_alert_above_gate(self):
        """Live alert threshold must always be above deployment gate."""
        from apex10.config import MODEL
        assert MODEL.BRIER_LIVE_ALERT > MODEL.BRIER_GATE

    def test_train_years(self):
        from apex10.config import MODEL
        assert MODEL.TRAIN_YEARS == 5

    def test_min_paper_tickets(self):
        from apex10.config import MODEL
        assert MODEL.MIN_PAPER_TICKETS == 20

    def test_rolling_brier_window(self):
        from apex10.config import MODEL
        assert MODEL.ROLLING_BRIER_WINDOW == 15


class TestStakingConfig:
    def test_kelly_fraction(self):
        from apex10.config import STAKING
        assert STAKING.KELLY_FRACTION == 0.25

    def test_kelly_is_quarter(self):
        """Sanity: must be quarter-Kelly, not full Kelly."""
        from apex10.config import STAKING
        assert STAKING.KELLY_FRACTION < 0.5

    def test_roi_floor(self):
        from apex10.config import STAKING
        assert STAKING.SIMULATED_ROI_FLOOR == -0.05


class TestLeagueConfig:
    def test_epl_is_active_in_phase_1(self):
        from apex10.config import LEAGUES
        assert "EPL" in LEAGUES.ACTIVE_LEAGUES

    def test_epl_league_id(self):
        from apex10.config import LEAGUES
        assert LEAGUES.LEAGUE_IDS["EPL"] == 39

    def test_all_leagues_contains_active(self):
        from apex10.config import LEAGUES
        for league in LEAGUES.ACTIVE_LEAGUES:
            assert league in LEAGUES.ALL_LEAGUES

    def test_all_five_leagues_defined(self):
        from apex10.config import LEAGUES
        assert len(LEAGUES.ALL_LEAGUES) == 5


class TestRequireEnv:
    def test_require_env_raises_on_missing(self):
        from apex10.config import _require_env
        with patch.dict(os.environ, {}, clear=True):
            # Remove the key if it exists
            os.environ.pop("NONEXISTENT_KEY_XYZ", None)
            with pytest.raises(EnvironmentError, match="NONEXISTENT_KEY_XYZ"):
                _require_env("NONEXISTENT_KEY_XYZ")

    def test_require_env_returns_value(self):
        from apex10.config import _require_env
        with patch.dict(os.environ, {"TEST_KEY": "test_value"}):
            assert _require_env("TEST_KEY") == "test_value"
