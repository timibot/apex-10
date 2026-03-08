"""Tests for walk-forward split logic — pure math, no DB."""
import numpy as np
import pytest

from apex10.models.walk_forward import check_brier_gate, get_season_splits


class TestGetSeasonSplits:
    def _seasons(self):
        # 5 seasons × 380 matches each
        return np.array(
            [2021] * 380 + [2022] * 380 + [2023] * 380 + [2024] * 380 + [2025] * 380
        )

    def test_returns_all_keys(self):
        splits = get_season_splits(self._seasons())
        assert all(k in splits for k in ["train", "val", "test"])

    def test_train_is_first_three_seasons(self):
        splits = get_season_splits(self._seasons())
        seasons = self._seasons()
        train_seasons = set(seasons[splits["train"]])
        assert train_seasons == {2021, 2022, 2023}

    def test_val_is_fourth_season(self):
        splits = get_season_splits(self._seasons())
        seasons = self._seasons()
        val_seasons = set(seasons[splits["val"]])
        assert val_seasons == {2024}

    def test_test_is_fifth_season(self):
        splits = get_season_splits(self._seasons())
        seasons = self._seasons()
        test_seasons = set(seasons[splits["test"]])
        assert test_seasons == {2025}

    def test_no_overlap_between_splits(self):
        splits = get_season_splits(self._seasons())
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_all_indices_covered(self):
        seasons = self._seasons()
        splits = get_season_splits(seasons)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == len(seasons)

    def test_raises_on_too_few_seasons(self):
        with pytest.raises(ValueError):
            get_season_splits(np.array([2021] * 100 + [2022] * 100))


class TestBrierGate:
    def test_passes_below_gate(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.75, 0.25, 0.70, 0.30, 0.80])
        result = check_brier_gate(y_true, y_prob, gate=0.20)
        assert result["passed"] == True  # noqa: E712 — may be numpy bool_

    def test_fails_above_gate(self):
        # Random probabilities produce high Brier
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.1, 0.9, 0.1])  # Completely wrong
        result = check_brier_gate(y_true, y_prob, gate=0.20)
        assert result["passed"] == False  # noqa: E712 — may be numpy bool_

    def test_result_contains_required_keys(self):
        y_true = np.array([1, 0, 1])
        y_prob = np.array([0.7, 0.3, 0.8])
        result = check_brier_gate(y_true, y_prob)
        assert all(k in result for k in ["brier_score", "gate", "passed", "gap"])

    def test_gap_is_gate_minus_score(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.8, 0.2, 0.7, 0.3])
        result = check_brier_gate(y_true, y_prob, gate=0.20)
        assert result["gap"] == pytest.approx(
            result["gate"] - result["brier_score"], abs=1e-4
        )
