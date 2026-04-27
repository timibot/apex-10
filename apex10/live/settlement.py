"""
settlement.py
-------------
Runs every Thursday at 08:00 UTC to grade last weekend's predictions.

For each row in `upcoming_fixtures` where:
  - match_date < today  (game has passed)
  - actual_outcome IS NULL  (not yet graded)

It fetches final scores from TheSportsDB eventsday.php (grouped by date),
matches by team name, resolves the bet outcome, computes Brier contribution,
and writes results back to the table.

Also emits a running performance report to Discord.
"""

import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from apex10.live import notify

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Market outcome resolvers
# ---------------------------------------------------------------------------

MARKET_RESOLVERS = {
    "Home Win":          lambda r: 1.0 if r["home_goals"] > r["away_goals"] else 0.0,
    "Away Win":          lambda r: 1.0 if r["away_goals"] > r["home_goals"] else 0.0,
    "Double Chance 1X":  lambda r: 1.0 if r["home_goals"] >= r["away_goals"] else 0.0,
    "Double Chance X2":  lambda r: 1.0 if r["away_goals"] >= r["home_goals"] else 0.0,
    "Draw No Bet":       lambda r: 1.0 if r["home_goals"] > r["away_goals"] else (0.5 if r["home_goals"] == r["away_goals"] else 0.0),
    "DNB Away":          lambda r: 1.0 if r["away_goals"] > r["home_goals"] else (0.5 if r["home_goals"] == r["away_goals"] else 0.0),
    "Over 1.5 Goals":    lambda r: 1.0 if (r["home_goals"] + r["away_goals"]) >= 2 else 0.0,
    "Over 2.5 Goals":    lambda r: 1.0 if (r["home_goals"] + r["away_goals"]) >= 3 else 0.0,
    "Under 3.5 Goals":   lambda r: 1.0 if (r["home_goals"] + r["away_goals"]) <= 3 else 0.0,
    "BTTS Yes":          lambda r: 1.0 if r["home_goals"] >= 1 and r["away_goals"] >= 1 else 0.0,
    "BTTS No":           lambda r: 1.0 if r["home_goals"] == 0 or r["away_goals"] == 0 else 0.0,
}


def resolve_outcome(market: str, result: Dict[str, Any]) -> Optional[float]:
    if result.get("status") != "FINISHED":
        return None
    resolver = MARKET_RESOLVERS.get(market)
    if resolver is None:
        logger.warning(f"Unknown market '{market}' — skipping.")
        return None
    return resolver(result)


def brier_contribution(predicted_prob: float, actual_outcome: float) -> float:
    return round((predicted_prob - actual_outcome) ** 2, 6)


# ---------------------------------------------------------------------------
# TheSportsDB: fetch all soccer events on a given date
# Uses eventsday.php — works on free key, grouped by date to minimise calls
# ---------------------------------------------------------------------------

_results_cache: Dict[str, List[Dict]] = {}  # date_str → list of event dicts


def _normalise_name(name: str) -> str:
    """Lower-case, strip punctuation for fuzzy team name matching."""
    import re
    return re.sub(r"[^a-z0-9 ]", "", name.lower().strip())


def fetch_day_results(date_str: str) -> List[Dict[str, Any]]:
    """
    Fetch all completed soccer events on `date_str` (YYYY-MM-DD) from TheSportsDB.
    Returns list of dicts with: home_team, away_team, home_goals, away_goals.
    Results are cached so the same date is only fetched once per run.
    """
    if date_str in _results_cache:
        return _results_cache[date_str]

    url = f"https://www.thesportsdb.com/api/v1/json/3/eventsday.php?d={date_str}&s=Soccer"

    for attempt in range(3):
        try:
            time.sleep(1.5)
            with httpx.Client(timeout=12.0) as client:
                resp = client.get(url)

            if resp.status_code == 429:
                wait = 6.0 * (attempt + 1)
                logger.warning(f"TheSportsDB rate-limited on {date_str}, waiting {wait}s")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()
            events = data.get("events") or []

            completed = []
            for ev in events:
                status = ev.get("strStatus") or ""
                home_score = ev.get("intHomeScore")
                away_score = ev.get("intAwayScore")

                if status == "Match Finished" and home_score is not None and away_score is not None:
                    completed.append({
                        "home_team":  ev.get("strHomeTeam", ""),
                        "away_team":  ev.get("strAwayTeam", ""),
                        "home_goals": int(home_score),
                        "away_goals": int(away_score),
                        "status":     "FINISHED",
                    })
                elif status in ("Postponed", "Cancelled", "Abandoned"):
                    completed.append({
                        "home_team": ev.get("strHomeTeam", ""),
                        "away_team": ev.get("strAwayTeam", ""),
                        "status":    status.upper(),
                    })

            _results_cache[date_str] = completed
            logger.info(f"  {date_str}: fetched {len(completed)} completed events from TheSportsDB")
            return completed

        except Exception as e:
            logger.warning(f"eventsday fetch error (attempt {attempt+1}) for {date_str}: {e}")
            time.sleep(3.0)

    _results_cache[date_str] = []
    return []


def match_result(fix: Dict, day_results: List[Dict]) -> Optional[Dict]:
    """
    Find a result from `day_results` matching the fixture's home/away teams.
    Uses normalised name comparison (strips accents etc via simple lowercasing).
    """
    home_norm = _normalise_name(fix["home_team"])
    away_norm = _normalise_name(fix["away_team"])

    for result in day_results:
        r_home = _normalise_name(result["home_team"])
        r_away = _normalise_name(result["away_team"])

        # Exact match first
        if r_home == home_norm and r_away == away_norm:
            return result

        # Partial match (one team name contained in the other)
        if (home_norm in r_home or r_home in home_norm) and \
           (away_norm in r_away or r_away in away_norm):
            return result

    return None


# ---------------------------------------------------------------------------
# Main settlement loop
# ---------------------------------------------------------------------------

def settle_pending(db_client: Any) -> Dict[str, Any]:
    """
    Find all upcoming_fixtures from past match dates that haven't been graded,
    fetch results from TheSportsDB (grouped by date), and write outcomes back.
    """
    today_str = date.today().isoformat()

    response = (
        db_client.table("upcoming_fixtures")
        .select("id, api_match_id, league, home_team, away_team, match_date, best_bet_type, consensus_prob, actual_outcome")
        .lt("match_date", today_str)
        .execute()
    )
    all_past: List[Dict[str, Any]] = response.data or []

    # Filter ungraded in Python — avoids supabase-py .is_() NULL quirks
    pending = [row for row in all_past if row.get("actual_outcome") is None]

    if not pending:
        logger.info(f"All {len(all_past)} past fixtures already settled.")
        return {"settled": 0, "skipped": 0, "errors": 0, "total": 0}

    logger.info(f"Found {len(pending)} ungraded fixtures (of {len(all_past)} past) to settle...")

    # Group pending fixtures by match_date for batch fetching
    by_date: Dict[str, List[Dict]] = defaultdict(list)
    for fix in pending:
        by_date[fix["match_date"][:10]].append(fix)

    settled = skipped = errors = 0

    for d_str in sorted(by_date.keys()):
        day_fixtures = by_date[d_str]
        logger.info(f"\n--- {d_str} ({len(day_fixtures)} fixtures) ---")

        # One API call covers all fixtures on this date
        day_results = fetch_day_results(d_str)

        for fix in day_fixtures:
            result = match_result(fix, day_results)

            if result is None:
                logger.info(f"  ❓ {fix['home_team']} vs {fix['away_team']} — no match in TheSportsDB for {d_str}")
                skipped += 1
                continue

            if result["status"] != "FINISHED":
                logger.info(f"  ⏭️  {fix['home_team']} vs {fix['away_team']} — {result['status']}")
                skipped += 1
                continue

            outcome = resolve_outcome(fix["best_bet_type"], result)

            if outcome is None:
                logger.warning(f"  ⚠️  Unknown market '{fix['best_bet_type']}' — skipping")
                skipped += 1
                continue

            brier = brier_contribution(float(fix["consensus_prob"]), outcome)
            hg = result["home_goals"]
            ag = result["away_goals"]
            label = "WIN ✅" if outcome == 1.0 else ("PUSH" if outcome == 0.5 else "LOSS ❌")

            logger.info(f"  {label} {fix['home_team']} {hg}-{ag} {fix['away_team']} "
                        f"[{fix['best_bet_type']}] prob={fix['consensus_prob']:.2f} brier={brier:.4f}")

            try:
                db_client.table("upcoming_fixtures").update({
                    "actual_outcome":     outcome,
                    "brier_contribution": brier,
                    "settled_at":         datetime.now(timezone.utc).isoformat(),
                }).eq("id", fix["id"]).execute()
                settled += 1
            except Exception as e:
                logger.error(f"  DB update failed for id {fix['id']}: {e}")
                errors += 1

    logger.info(f"\nSettlement done: {settled} settled, {skipped} not found/not finished, {errors} errors")
    return {"settled": settled, "skipped": skipped, "errors": errors, "total": len(pending)}


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------

def emit_calibration_report(db_client: Any, run_id: str) -> Dict:
    response = (
        db_client.table("upcoming_fixtures")
        .select("league, best_bet_type, best_bet_odds, consensus_prob, actual_outcome, brier_contribution")
        .execute()
    )
    all_rows: List[Dict[str, Any]] = response.data or []

    # Only settled non-void rows
    rows = [r for r in all_rows if r.get("actual_outcome") is not None and float(r["actual_outcome"]) != 0.5]

    if not rows:
        logger.warning("No settled rows yet — calibration skipped.")
        return {}

    total = len(rows)
    wins = sum(1 for r in rows if float(r["actual_outcome"]) == 1.0)
    briers = [float(r["brier_contribution"]) for r in rows if r.get("brier_contribution") is not None]
    avg_brier = round(sum(briers) / len(briers), 6) if briers else 0.0

    roi_sum = sum((float(r["best_bet_odds"]) * float(r["actual_outcome"])) - 1.0 for r in rows)
    roi_pct = round((roi_sum / total) * 100, 2)

    by_league: Dict[str, dict] = {}
    for r in rows:
        lg = r["league"]
        if lg not in by_league:
            by_league[lg] = {"total": 0, "wins": 0, "roi": 0.0}
        by_league[lg]["total"] += 1
        outcome = float(r["actual_outcome"])
        by_league[lg]["wins"] += 1 if outcome == 1.0 else 0
        by_league[lg]["roi"] += (float(r["best_bet_odds"]) * outcome) - 1.0

    by_market: Dict[str, dict] = {}
    for r in rows:
        mk = r["best_bet_type"]
        if mk not in by_market:
            by_market[mk] = {"total": 0, "wins": 0}
        by_market[mk]["total"] += 1
        if float(r["actual_outcome"]) == 1.0:
            by_market[mk]["wins"] += 1

    report = {
        "run_id":        run_id,
        "generated":     datetime.now(timezone.utc).isoformat(),
        "total_settled": total,
        "win_rate":      round(wins / total, 4),
        "avg_brier":     avg_brier,
        "roi_pct":       roi_pct,
        "by_league": {
            lg: {
                "n":        v["total"],
                "win_rate": round(v["wins"] / v["total"], 3),
                "roi_pct":  round((v["roi"] / v["total"]) * 100, 2),
            }
            for lg, v in by_league.items()
        },
        "by_market": {
            mk: {
                "n":        v["total"],
                "win_rate": round(v["wins"] / v["total"], 3),
            }
            for mk, v in by_market.items()
        },
    }

    os.makedirs("apex10/diagnostics", exist_ok=True)
    with open(f"apex10/diagnostics/calibration_{run_id}.json", "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"\nCalibration: {total} graded | Win rate: {report['win_rate']:.1%} | ROI: {roi_pct:+.1f}% | Brier: {avg_brier}")
    for lg, v in sorted(report["by_league"].items(), key=lambda x: -x[1]["win_rate"]):
        logger.info(f"    {lg}: {v['n']} bets | {v['win_rate']:.1%} win | ROI {v['roi_pct']:+.1f}%")
    logger.info("")
    for mk, v in sorted(report["by_market"].items(), key=lambda x: -x[1]["win_rate"]):
        logger.info(f"    {mk}: {v['n']} bets | {v['win_rate']:.1%} win")

    return report


# ---------------------------------------------------------------------------
# Discord performance summary
# ---------------------------------------------------------------------------

def _discord_report(report: Dict, settlement: Dict) -> None:
    settled_now = settlement.get("settled", 0)

    if not report:
        notify._send(
            f"ℹ️ **APEX-10 Settlement** — {settled_now} new results graded. "
            f"Skipped: {settlement.get('skipped', 0)} (not in TheSportsDB yet). "
            f"No historical data for full report yet."
        )
        return

    total = report.get("total_settled", 0)
    win_rate = report.get("win_rate", 0)
    roi = report.get("roi_pct", 0)
    brier = report.get("avg_brier", 0)

    lines = [
        "📊 **APEX-10 Weekly Performance Report**",
        "",
        f"🎯 Win Rate: **{win_rate:.1%}** ({int(win_rate * total)}/{total} bets graded)",
        f"💰 Paper ROI: **{roi:+.1f}%**",
        f"📐 Brier Score: **{brier}** _(0 = perfect)_",
        "",
        "**By League (best → worst):**",
    ]
    for lg, v in sorted(report.get("by_league", {}).items(), key=lambda x: -x[1]["win_rate"]):
        lines.append(f"  {lg}: {v['win_rate']:.1%} win | ROI {v['roi_pct']:+.1f}% ({v['n']} bets)")

    lines += ["", "**By Market (worst → best):**"]
    for mk, v in sorted(report.get("by_market", {}).items(), key=lambda x: x[1]["win_rate"]):
        lines.append(f"  {mk}: {v['win_rate']:.1%} ({v['n']} bets)")

    lines += ["", f"_This run: {settled_now} new results graded_"]

    notify._send("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(db_client: Any) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    logger.info(f"\n{'='*60}")
    logger.info(f"  APEX-10 Settlement — {run_id}")
    logger.info(f"{'='*60}\n")

    settlement_result = settle_pending(db_client)
    report = emit_calibration_report(db_client, run_id)
    _discord_report(report, settlement_result)


if __name__ == "__main__":
    import traceback
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    try:
        from apex10.db import get_client
        run(get_client())
    except Exception:
        traceback.print_exc()
        raise
