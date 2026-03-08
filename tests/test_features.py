"""Tests for feature builder — uses synthetic data, no DB calls."""
import numpy as np
import pandas as pd

from apex10.models.features import (
    ALL_FEATURES,
    MARKET_FEATURES,
    ONPITCH_FEATURES,
    get_feature_matrices,
)


class TestFeatureSplit:
    def test_onpitch_feature_count(self):
        assert len(ONPITCH_FEATURES) == 29

    def test_market_feature_count(self):
        assert len(MARKET_FEATURES) == 16

    def test_total_feature_count(self):
        assert len(ALL_FEATURES) == 45

    def test_no_overlap_between_feature_sets(self):
        overlap = set(ONPITCH_FEATURES) & set(MARKET_FEATURES)
        assert len(overlap) == 0, f"Feature overlap detected: {overlap}"

    def test_all_features_are_unique(self):
        assert len(ALL_FEATURES) == len(set(ALL_FEATURES))


class TestGetFeatureMatrices:
    def _make_df(self, n: int = 100) -> pd.DataFrame:
        """Synthetic DataFrame with all required columns."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(index=range(n))
        df["home_win"] = rng.integers(0, 2, n)
        df["season"] = np.repeat([2021, 2022, 2023, 2024, 2025], n // 5)[:n]
        for col in ALL_FEATURES:
            df[col] = rng.uniform(0, 1, n)
        return df

    def test_returns_four_arrays(self):
        df = self._make_df()
        result = get_feature_matrices(df)
        assert len(result) == 4

    def test_onpitch_shape(self):
        df = self._make_df(100)
        X_onpitch, _, _, _ = get_feature_matrices(df)
        assert X_onpitch.shape == (100, 29)

    def test_market_shape(self):
        df = self._make_df(100)
        _, X_market, _, _ = get_feature_matrices(df)
        assert X_market.shape == (100, 16)

    def test_target_is_binary(self):
        df = self._make_df()
        _, _, y, _ = get_feature_matrices(df)
        assert set(np.unique(y)).issubset({0, 1})

    def test_no_nans_in_output(self):
        df = self._make_df()
        X_op, X_mk, y, _ = get_feature_matrices(df)
        assert not np.isnan(X_op).any()
        assert not np.isnan(X_mk).any()
        assert not np.isnan(y).any()
