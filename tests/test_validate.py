"""Tests for cache validation assertions."""

import pytest

from apex10.cache.validate import (
    MIN_FIXTURES_PER_SEASON,
    ValidationError,
    assert_fixture_count,
    assert_odds_valid,
    assert_ppda_not_all_stubs,
    assert_rho_in_valid_range,
)


class TestAssertFixtureCount:
    def test_passes_above_minimum(self):
        assert_fixture_count(380, "EPL", 2023)

    def test_fails_below_minimum(self):
        with pytest.raises(ValidationError, match="FIXTURE_COUNT_LOW"):
            assert_fixture_count(50, "EPL", 2023)

    def test_fails_at_zero(self):
        with pytest.raises(ValidationError):
            assert_fixture_count(0, "EPL", 2023)

    def test_passes_at_exact_minimum(self):
        assert_fixture_count(MIN_FIXTURES_PER_SEASON, "EPL", 2023)


class TestAssertOddsValid:
    def test_passes_valid_odds(self):
        records = [{"odds_home": 1.85}, {"odds_home": 2.10}]
        assert_odds_valid(records)

    def test_fails_odds_below_one(self):
        records = [{"odds_home": 0.90}]
        with pytest.raises(ValidationError, match="INVALID_ODDS"):
            assert_odds_valid(records)

    def test_fails_null_odds(self):
        records = [{"odds_home": None}]
        with pytest.raises(ValidationError, match="INVALID_ODDS"):
            assert_odds_valid(records)

    def test_fails_odds_exactly_one(self):
        records = [{"odds_home": 1.0}]
        with pytest.raises(ValidationError):
            assert_odds_valid(records)

    def test_passes_empty_records(self):
        assert_odds_valid([])


class TestAssertPpdaNotAllStubs:
    def test_fails_all_stubs(self):
        records = [{"ppda": 10.0}, {"ppda": 10.0}, {"ppda": 10.0}]
        with pytest.raises(ValidationError, match="PPDA_ALL_STUBS"):
            assert_ppda_not_all_stubs(records)

    def test_passes_real_data(self):
        records = [{"ppda": 8.5}, {"ppda": 11.2}, {"ppda": 9.8}]
        assert_ppda_not_all_stubs(records)

    def test_passes_mixed_data(self):
        records = [{"ppda": 10.0}, {"ppda": 8.5}]
        assert_ppda_not_all_stubs(records)

    def test_passes_empty_records(self):
        assert_ppda_not_all_stubs([])


class TestAssertRhoInValidRange:
    def test_passes_typical_football_rho(self):
        assert_rho_in_valid_range(-0.1, "EPL")

    def test_passes_zero_rho(self):
        assert_rho_in_valid_range(0.0, "EPL")

    def test_fails_rho_too_positive(self):
        with pytest.raises(ValidationError, match="RHO_OUT_OF_RANGE"):
            assert_rho_in_valid_range(0.8, "EPL")

    def test_fails_rho_too_negative(self):
        with pytest.raises(ValidationError, match="RHO_OUT_OF_RANGE"):
            assert_rho_in_valid_range(-0.8, "EPL")

    def test_passes_at_boundary(self):
        assert_rho_in_valid_range(0.5, "EPL")
        assert_rho_in_valid_range(-0.5, "EPL")
