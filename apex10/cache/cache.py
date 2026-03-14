"""
Cache orchestrator — called by GitHub Actions every Thursday.
Runs all data fetchers in correct dependency order.
Fail-fast per section: logs errors but continues other sections.
"""
from __future__ import annotations

import logging

from apex10.cache.api_football import backfill_league
from apex10.cache.odds_loader import fetch_odds_csv, parse_odds_rows, upsert_historical_odds
from apex10.cache.rho import compute_and_store_rho
from apex10.cache.understat import fetch_league_xg, upsert_xg_data
from apex10.db import get_client

logger = logging.getLogger(__name__)

BACKFILL_SEASONS = [2022, 2023, 2024]  # API-Football free plan: 2022-2024

ALL_LEAGUES = ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]


def run_full_backfill(league_name: str = "EPL") -> dict:
    """
    Run complete data backfill for a league.
    Order: matches → odds → xG → rho
    Returns summary dict for logging/alerting.
    """
    db = get_client()
    summary = {"league": league_name, "errors": []}

    # 1. Match results
    try:
        match_summary = backfill_league(league_name, db)
        summary["matches"] = match_summary
        logger.info(f"Matches backfill complete: {match_summary}")
    except Exception as e:
        logger.error(f"Match backfill failed: {e}")
        summary["errors"].append(f"matches: {e}")

    # 2. Historical odds
    odds_total = 0
    for season in BACKFILL_SEASONS:
        try:
            df = fetch_odds_csv(league_name, season)
            if df is not None:
                records = parse_odds_rows(df, league_name, season)
                count = upsert_historical_odds(records, db)
                odds_total += count
        except Exception as e:
            logger.error(f"Odds backfill failed {season}: {e}")
            summary["errors"].append(f"odds_{season}: {e}")
    summary["odds"] = odds_total

    # 3. Understat xG
    xg_total = 0
    for season in BACKFILL_SEASONS:
        try:
            records = fetch_league_xg(league_name, season)
            if records:
                count = upsert_xg_data(records, db)
                xg_total += count
        except Exception as e:
            logger.error(f"xG backfill failed {season}: {e}")
            summary["errors"].append(f"xg_{season}: {e}")
    summary["xg"] = xg_total

    # 4. Rho computation — depends on xG being loaded
    try:
        rho = compute_and_store_rho(league_name, db)
        summary["rho"] = rho
    except Exception as e:
        logger.error(f"Rho computation failed: {e}")
        summary["errors"].append(f"rho: {e}")

    return summary


def run_all_leagues() -> dict:
    """Run backfill for every league in ALL_LEAGUES."""
    results = {}
    for league in ALL_LEAGUES:
        logger.info(f"\n{'='*60}\n  Backfilling {league}\n{'='*60}")
        try:
            results[league] = run_full_backfill(league)
        except Exception as e:
            logger.error(f"Failed to backfill {league}: {e}")
            results[league] = {"errors": [str(e)]}
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        league = sys.argv[1]
        result = run_full_backfill(league)
        print(result)
    else:
        # Run all leagues
        results = run_all_leagues()
        for league, summary in results.items():
            errors = summary.get("errors", [])
            print(f"\n{league}: matches={summary.get('matches','-')} "
                  f"odds={summary.get('odds','-')} xg={summary.get('xg','-')} "
                  f"errors={len(errors)}")

