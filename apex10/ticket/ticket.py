"""
Weekly ticket generator.
Called by GitHub Actions every Thursday after cache.py completes.
Reads live data from Supabase, runs gates, builds ticket, logs result.
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta

from apex10.cache.cache_log import check_data_freshness
from apex10.config import STAKING
from apex10.db import get_client
from apex10.filters.gates import Candidate, run_all_gates
from apex10.live.health import get_adjusted_stake
from apex10.live.notify import no_ticket_week as notify_no_ticket
from apex10.live.notify import system_error, ticket_generated
from apex10.ticket.optimizer import build_ticket, quarter_kelly_stake

logger = logging.getLogger(__name__)


def get_current_bank(db) -> float:
    """Read live bank balance from Supabase bank_state table."""
    result = db.table("bank_state").select("current_bank").eq("id", 1).execute()
    if not result.data:
        raise ValueError("bank_state table is empty — initialise with starting bank")
    return float(result.data[0]["current_bank"])


def get_week_candidates(db, week_start: date) -> list[Candidate]:
    """
    Load this week's fixture candidates from Supabase.
    In live mode, cache.py has pre-populated upcoming fixtures with
    model probabilities, odds, and feature flags.
    """
    week_end = week_start + timedelta(days=7)

    result = (
        db.table("upcoming_fixtures")
        .select("*")
        .gte("match_date", str(week_start))
        .lt("match_date", str(week_end))
        .execute()
    )

    candidates = []
    for row in result.data or []:
        try:
            c = Candidate(
                fixture_id=row["api_match_id"],
                league=row["league"],
                home_team=row["home_team"],
                away_team=row["away_team"],
                bet_type=row["best_bet_type"],
                odds=float(row["best_bet_odds"]),
                lgbm_prob=float(row["lgbm_prob"]),
                xgb_prob=float(row["xgb_prob"]),
                opening_odds=float(row["opening_odds"]),
                key_player_absent_home=int(
                    row.get("key_player_absent_home", 0)
                ),
                key_player_absent_away=int(
                    row.get("key_player_absent_away", 0)
                ),
                features_complete=bool(row.get("features_complete", True)),
                # Votes encoded in xgb_prob as votes/5.0 by inference
                confidence_votes=round(float(row.get("xgb_prob", 0)) * 5),
            )
            candidates.append(c)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Skipping malformed candidate row: {e}")

    logger.info(f"Loaded {len(candidates)} candidates for week {week_start}")
    return candidates


def run_weekly_ticket(week_start: date | None = None) -> dict:
    """
    Full weekly ticket pipeline.
    Returns summary dict for Discord notification and Supabase logging.
    """
    if week_start is None:
        # Default to the Monday of the current week
        today = date.today()
        week_start = today - timedelta(days=today.weekday())

    db = get_client()
    logger.info(f"Running ticket for week: {week_start}")

    # ── 0. Freshness check — halt if cache data is stale ──────────────────
    freshness = check_data_freshness(db)
    if not freshness["fresh"]:
        msg = freshness["reason"]
        logger.error(f"❌ Stale data — ticket generation halted: {msg}")
        system_error("ticket.py", msg)
        return {"no_ticket": True, "reason": msg, "stale_data": True}

    # ── 1. Load candidates ────────────────────────────────────────────────
    candidates = get_week_candidates(db, week_start)

    if not candidates:
        logger.warning("No candidates found — logging no-ticket week")
        _log_no_ticket(db, week_start, "No upcoming fixtures found in DB")
        return {"no_ticket": True, "reason": "No candidates"}

    # ── 2. Run gates ──────────────────────────────────────────────────────
    qualified, rejection_log = run_all_gates(candidates)

    # ── 3. Build ticket ───────────────────────────────────────────────────
    ticket = build_ticket(qualified)

    if ticket.no_ticket:
        _log_no_ticket(db, week_start, ticket.reason)
        return {"no_ticket": True, "reason": ticket.reason}

    # ── 4. Quarter-Kelly stake with health multiplier ─────────────────────
    bank = get_current_bank(db)
    base_stake = quarter_kelly_stake(
        bank=bank,
        win_prob=ticket.simulated_win_rate,
        combined_odds=ticket.combined_odds,
        kelly_fraction=STAKING.KELLY_FRACTION,
    )
    # Apply stake multiplier (halved if Brier breach active)
    stake = get_adjusted_stake(base_stake, db)

    # ── 5. Log to Supabase ────────────────────────────────────────────────
    legs_json = [
        {
            "fixture_id": leg.fixture_id,
            "league": leg.league,
            "home_team": leg.home_team,
            "away_team": leg.away_team,
            "bet_type": leg.bet_type,
            "odds": leg.odds,
            "consensus_prob": leg.consensus_prob,
            "lgbm_edge": leg.lgbm_edge,
            "xgb_edge": leg.xgb_edge,
        }
        for leg in ticket.legs
    ]

    db.table("tickets").upsert(
        {
            "week_start": str(week_start),
            "legs": legs_json,
            "combined_odds": ticket.combined_odds,
            "stake": stake,
            "status": "pending",
            "simulated_win_rate": ticket.simulated_win_rate,
            "monte_carlo_ci_low": ticket.ci_low,
            "monte_carlo_ci_high": ticket.ci_high,
        },
        on_conflict="week_start",
    ).execute()

    # ── 6. Discord notification ───────────────────────────────────────────
    ticket_generated(
        week=str(week_start),
        legs=len(ticket.legs),
        combined_odds=ticket.combined_odds,
        stake=stake,
        win_rate=ticket.simulated_win_rate,
    )

    summary = {
        "no_ticket": False,
        "week_start": str(week_start),
        "legs": len(ticket.legs),
        "combined_odds": ticket.combined_odds,
        "stake": stake,
        "bank": bank,
        "simulated_win_rate": ticket.simulated_win_rate,
        "ci": [ticket.ci_low, ticket.ci_high],
        "rejected": len(rejection_log),
    }

    logger.info(f"Ticket logged: {json.dumps(summary, indent=2)}")
    return summary


def _log_no_ticket(db, week_start: date, reason: str) -> None:
    db.table("tickets").upsert(
        {
            "week_start": str(week_start),
            "legs": [],
            "combined_odds": 0.0,
            "status": "no_ticket",
            "simulated_win_rate": 0.0,
            "monte_carlo_ci_low": 0.0,
            "monte_carlo_ci_high": 0.0,
        },
        on_conflict="week_start",
    ).execute()
    notify_no_ticket(str(week_start), reason)
    logger.info(f"No-ticket week logged: {week_start} — {reason}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_weekly_ticket()
