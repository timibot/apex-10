"""Tests for cache_log freshness check."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from apex10.cache.cache_log import MAX_DATA_AGE_DAYS, check_data_freshness


def _mock_db(last_run_iso: str | None):
    db = MagicMock()

    def table_side(name):
        t = MagicMock()
        if name == "cache_log":
            if last_run_iso:
                chain = t.select.return_value.eq.return_value
                chain.order.return_value.limit.return_value\
                    .execute.return_value.data = [
                    {"run_timestamp": last_run_iso, "success": True}
                ]
            else:
                chain = t.select.return_value.eq.return_value
                chain.order.return_value.limit.return_value\
                    .execute.return_value.data = []
        return t

    db.table.side_effect = table_side
    return db


class TestCheckDataFreshness:
    def test_fresh_data_returns_true(self):
        recent = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).isoformat()
        db = _mock_db(recent)
        result = check_data_freshness(db)
        assert result["fresh"] is True

    def test_stale_data_returns_false(self):
        old = (
            datetime.now(timezone.utc) - timedelta(days=10)
        ).isoformat()
        db = _mock_db(old)
        result = check_data_freshness(db)
        assert result["fresh"] is False

    def test_no_cache_log_returns_false(self):
        db = _mock_db(None)
        result = check_data_freshness(db)
        assert result["fresh"] is False

    def test_stale_result_has_reason(self):
        old = (
            datetime.now(timezone.utc) - timedelta(days=10)
        ).isoformat()
        db = _mock_db(old)
        result = check_data_freshness(db)
        assert result["reason"] is not None
        assert len(result["reason"]) > 0

    def test_fresh_result_has_no_reason(self):
        recent = (
            datetime.now(timezone.utc) - timedelta(hours=6)
        ).isoformat()
        db = _mock_db(recent)
        result = check_data_freshness(db)
        assert result["reason"] is None

    def test_result_contains_age_days(self):
        recent = (
            datetime.now(timezone.utc) - timedelta(days=2)
        ).isoformat()
        db = _mock_db(recent)
        result = check_data_freshness(db)
        assert result["age_days"] is not None
        assert 1.5 < result["age_days"] < 2.5

    def test_boundary_just_within_max_age(self):
        boundary = (
            datetime.now(timezone.utc)
            - timedelta(days=MAX_DATA_AGE_DAYS - 0.1)
        ).isoformat()
        db = _mock_db(boundary)
        result = check_data_freshness(db)
        assert result["fresh"] is True
