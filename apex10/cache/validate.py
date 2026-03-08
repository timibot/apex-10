"""
Post-write validation assertions for cache.py.
Every table write calls the relevant validator before advancing.
Any failure raises a named exception — GitHub Actions catches it,
Discord alert fires, ticket.py never runs on bad data.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# Minimum records expected per season per league
MIN_FIXTURES_PER_SEASON = 200
MIN_XG_RECORDS = 200
MIN_ODDS_RECORDS = 100


class ValidationError(Exception):
    """Raised when a post-write assertion fails. Always named."""


def assert_fixture_count(count: int, league: str, season: int) -> None:
    """Assert enough fixtures were written for a league/season."""
    if count < MIN_FIXTURES_PER_SEASON:
        raise ValidationError(
            f"FIXTURE_COUNT_LOW: {league} {season} — "
            f"got {count}, expected >= {MIN_FIXTURES_PER_SEASON}. "
            f"API may have returned partial data."
        )
    logger.info(f"✅ Fixture count OK: {league} {season} = {count}")


def assert_no_null_xg(db, league: str) -> None:
    """Assert zero NULL xG records exist after Understat load."""
    result = (
        db.table("match_xg")
        .select("id", count="exact")
        .is_("home_xg", "null")
        .execute()
    )
    null_count = result.count or 0
    if null_count > 0:
        raise ValidationError(
            f"NULL_XG_DETECTED: {null_count} records have NULL home_xg "
            f"in match_xg. Understat extraction may have partially failed."
        )
    logger.info(f"✅ xG null check OK: {league}")


def assert_odds_valid(records: list[dict]) -> None:
    """Assert all odds values are > 1.0 and not None."""
    invalid = [
        r
        for r in records
        if r.get("odds_home") is None or r.get("odds_home", 0) <= 1.0
    ]
    if invalid:
        raise ValidationError(
            f"INVALID_ODDS_DETECTED: {len(invalid)} records with "
            f"odds <= 1.0 or NULL. Sample: {invalid[:3]}"
        )
    logger.info(f"✅ Odds validity OK: {len(records)} records")


def assert_min_odds_records(
    count: int, league: str, season: int
) -> None:
    """Assert enough odds records were loaded for a season."""
    if count < MIN_ODDS_RECORDS:
        raise ValidationError(
            f"ODDS_COUNT_LOW: {league} {season} — "
            f"got {count} odds records, expected >= {MIN_ODDS_RECORDS}."
        )
    logger.info(f"✅ Odds count OK: {league} {season} = {count}")


def assert_ppda_not_all_stubs(
    records: list[dict], stub_value: float = 10.0
) -> None:
    """
    Assert PPDA records contain real data — not all defaulting to stub.
    If every record has ppda == stub_value, FBref fetch silently failed.
    """
    if not records:
        return
    all_stub = all(
        r.get("ppda", stub_value) == stub_value for r in records
    )
    if all_stub:
        raise ValidationError(
            f"PPDA_ALL_STUBS: All {len(records)} PPDA records equal "
            f"stub value {stub_value} — FBref fetch may have failed."
        )
    logger.info(f"✅ PPDA real data confirmed: {len(records)} records")


def assert_rho_in_valid_range(rho: float, league: str) -> None:
    """Assert computed rho is in a sensible range for football."""
    if not (-0.5 <= rho <= 0.5):
        raise ValidationError(
            f"RHO_OUT_OF_RANGE: {league} rho={rho:.4f} is outside "
            f"[-0.5, 0.5]. Dixon-Coles estimation may have failed."
        )
    logger.info(f"✅ Rho range OK: {league} rho={rho:.4f}")


def assert_upcoming_fixtures_fresh(
    db, min_count: int = 5
) -> None:
    """
    Assert upcoming_fixtures table has at least min_count rows
    for the current week. Called at end of cache run.
    """
    today = date.today()
    week_end = today + timedelta(days=7)

    result = (
        db.table("upcoming_fixtures")
        .select("id", count="exact")
        .gte("match_date", str(today))
        .lt("match_date", str(week_end))
        .execute()
    )
    count = result.count or 0
    if count < min_count:
        raise ValidationError(
            f"NO_UPCOMING_FIXTURES: Only {count} upcoming fixtures "
            f"for week {today} → {week_end}. Expected >= {min_count}."
        )
    logger.info(f"✅ Upcoming fixtures OK: {count} this week")
