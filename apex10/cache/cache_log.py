"""
Cache run logger — writes diagnostics after every cache run.
ticket.py reads this to verify data freshness before generating a ticket.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ticket.py halts if cache data is older than this
MAX_DATA_AGE_DAYS = 7


class CacheRunLogger:
    """Context manager that times a cache run and writes the log."""

    def __init__(self, db):
        self.db = db
        self._start = None
        self.stats = {
            "fixtures_written": 0,
            "xg_written": 0,
            "odds_written": 0,
            "ppda_written": 0,
            "api_requests_used": {},
            "sources_failed": [],
        }

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        runtime = round(time.monotonic() - self._start, 1)
        success = exc_type is None

        try:
            self.db.table("cache_log").insert({
                "run_timestamp": datetime.now(timezone.utc).isoformat(),
                "fixtures_written": self.stats["fixtures_written"],
                "xg_written": self.stats["xg_written"],
                "odds_written": self.stats["odds_written"],
                "ppda_written": self.stats["ppda_written"],
                "api_requests_used": self.stats["api_requests_used"],
                "sources_failed": self.stats["sources_failed"],
                "runtime_seconds": runtime,
                "success": success,
            }).execute()
            logger.info(
                f"Cache log written: success={success}, "
                f"runtime={runtime}s, "
                f"fixtures={self.stats['fixtures_written']}"
            )
        except Exception as e:
            logger.error(f"Failed to write cache log: {e}")

        return False  # Never suppress exceptions

    def add_source_failure(self, source: str, error: str) -> None:
        self.stats["sources_failed"].append(
            {"source": source, "error": str(error)}
        )

    def add_api_usage(self, api_name: str, requests_used: int) -> None:
        self.stats["api_requests_used"][api_name] = requests_used


def check_data_freshness(db) -> dict:
    """
    Read the most recent successful cache_log entry.
    Returns freshness status dict.
    Called by ticket.py on startup — halts if data is stale.
    """
    result = (
        db.table("cache_log")
        .select("*")
        .eq("success", True)
        .order("run_timestamp", desc=True)
        .limit(1)
        .execute()
    )

    if not result.data:
        return {
            "fresh": False,
            "reason": "No successful cache run found in cache_log",
            "last_run": None,
            "age_days": None,
        }

    last_run_str = result.data[0]["run_timestamp"]
    last_run = datetime.fromisoformat(
        last_run_str.replace("Z", "+00:00")
    )
    now = datetime.now(timezone.utc)
    age = now - last_run
    age_days = age.total_seconds() / 86400

    fresh = age_days <= MAX_DATA_AGE_DAYS

    status = {
        "fresh": fresh,
        "last_run": last_run_str,
        "age_days": round(age_days, 1),
        "max_age_days": MAX_DATA_AGE_DAYS,
        "reason": None
        if fresh
        else (
            f"Data is {age_days:.1f} days old — "
            f"exceeds {MAX_DATA_AGE_DAYS} day limit. "
            f"Run cache.py before generating a ticket."
        ),
    }

    if not fresh:
        logger.warning(f"⚠️ Stale data: {status['reason']}")
    else:
        logger.info(
            f"✅ Data fresh: last cache run {age_days:.1f} days ago"
        )

    return status
