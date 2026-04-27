"""
settlement.py
-------------
Runs every Thursday at 08:00 UTC to grade last weekend's predictions.

For each row in `upcoming_fixtures` where:
  - match_date < today  (game has passed)
  - actual_outcome IS NULL  (not yet graded)

It fetches the final score from TheSportsDB using the stored api_match_id
(which IS a TheSportsDB event ID), resolves the bet outcome, computes the
Brier contribution, and writes results back to the table.

Also emits a running performance report to Discord.
"""

import json
import os
import time
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional

import httpx

from apex10.live import notify

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
        print(f"⚠️  Unknown market '{market}' — skipping.")
        return None
    return resolver(result)


def brier_contribution(predicted_prob: float, actual_outcome: float) -> float:
    return round((predicted_prob - actual_outcome) ** 2, 6)


# ---------------------------------------------------------------------------
# TheSportsDB result fetch  (api_match_id IS a TheSportsDB event ID)
# ---------------------------------------------------------------------------

def fetch_result(event_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch the final score for a TheSportsDB event.
    TheSportsDB free tier — no API key required.
    Returns dict with status FINISHED/NOT_FINISHED/NOT_FOUND, and goals if finished.
    """
    url = f"https://www.thesportsdb.com/api/v1/json/3/lookupevent.php?id={event_id}"

    for attempt in range(3):
        try:
            time.sleep(1.5)          # respect free-tier rate limit
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url)

            if resp.status_code == 429:
                wait = 5.0 * (attempt + 1)
                print(f"  ⏳ Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            events = data.get("events")
            if not events or events[0] is None:
                return {"status": "NOT_FOUND"}

            event = events[0]

            # TheSportsDB marks finished matches with strStatus == "Match Finished"
            # and populates intHomeScore / intAwayScore
            status = event.get("strStatus", "")
            home_score = event.get("intHomeScore")
            away_score = event.get("intAwayScore")

            if status == "Match Finished" and home_score is not None and away_score is not None:
                return {
                    "status": "FINISHED",
                    "home_goals": int(home_score),
                    "away_goals": int(away_score),
                }

            # Postponed / cancelled
            if status in ("Postponed", "Cancelled", "Abandoned"):
                return {"status": status.upper()}

            # Not yet played or live
            return {"status": "NOT_FINISHED"}

        except Exception as e:
            print(f"  ⚠️  Error fetching event {event_id} (attempt {attempt+1}): {e}")
            time.sleep(3.0)

    return None


# ---------------------------------------------------------------------------
# Main settlement loop
# ---------------------------------------------------------------------------

def settle_pending(db_client: Any) -> Dict[str, Any]:
    """
    Find all upcoming_fixtures from past match dates that haven't been graded,
    fetch results from TheSportsDB, and write outcomes back.
    """
    today_str = date.today().isoformat()

    response = (
        db_client.table("upcoming_fixtures")
        .select("id, api_match_id, league, home_team, away_team, match_date, best_bet_type, consensus_prob, actual_outcome")
        .lt("match_date", today_str)
        .execute()
    )
    all_past: List[Dict[str, Any]] = response.data or []

    # Filter ungraded rows in Python — avoids supabase-py .is_() NULL filter quirks
    pending = [row for row in all_past if row.get("actual_outcome") is None]

    if not pending:
        print(f"✅  All {len(all_past)} past fixtures already settled.")
        return {"settled": 0, "skipped": 0, "errors": 0, "total": 0}

    print(f"📋  Found {len(pending)} ungraded fixtures (of {len(all_past)} past) to settle...")
    settled = skipped = errors = 0

    for fix in pending:
        event_id = fix["api_match_id"]
        result = fetch_result(event_id)

        if result is None:
            print(f"  ❓ {fix['home_team']} vs {fix['away_team']} — fetch failed entirely")
            errors += 1
            continue

        if result["status"] == "NOT_FOUND":
            # Event ID not in TheSportsDB — permanent skip, mark as void
            print(f"  🚫 {fix['home_team']} vs {fix['away_team']} (ID {event_id}) — not found in TheSportsDB, voiding")
            try:
                db_client.table("upcoming_fixtures").update({
                    "actual_outcome":     0.5,    # void/push
                    "brier_contribution": 0.0,
                    "settled_at":         datetime.now(timezone.utc).isoformat(),
                }).eq("id", fix["id"]).execute()
                settled += 1
            except Exception as e:
                print(f"    DB update failed: {e}")
                errors += 1
            continue

        if result["status"] != "FINISHED":
            print(f"  ⏭️  {fix['home_team']} vs {fix['away_team']} — {result['status']}, skipping")
            skipped += 1
            continue

        outcome = resolve_outcome(fix["best_bet_type"], result)

        if outcome is None:
            print(f"  ⚠️  {fix['home_team']} vs {fix['away_team']} — unknown market '{fix['best_bet_type']}'")
            skipped += 1
            continue

        brier = brier_contribution(float(fix["consensus_prob"]), outcome)
        hg = result["home_goals"]
        ag = result["away_goals"]
        label = "WIN ✅" if outcome == 1.0 else ("PUSH ↔️" if outcome == 0.5 else "LOSS ❌")

        print(f"  {label} {fix['home_team']} {hg}-{ag} {fix['away_team']} "
              f"[{fix['best_bet_type']}] prob={fix['consensus_prob']:.2f} brier={brier:.4f}")

        try:
            db_client.table("upcoming_fixtures").update({
                "actual_outcome":     outcome,
                "brier_contribution": brier,
                "settled_at":         datetime.now(timezone.utc).isoformat(),
            }).eq("id", fix["id"]).execute()
            settled += 1
        except Exception as e:
            print(f"    ⚠️  DB update failed: {e}")
            errors += 1

    print(f"\n📊  Settlement done: {settled} settled, {skipped} not finished yet, {errors} errors")
    return {"settled": settled, "skipped": skipped, "errors": errors, "total": len(pending)}


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------

def emit_calibration_report(db_client: Any, run_id: str) -> Dict:
    response = (
        db_client.table("upcoming_fixtures")
        .select("league, best_bet_type, best_bet_odds, consensus_prob, actual_outcome, brier_contribution")
        .not_.is_("actual_outcome", "null")
        .execute()
    )
    rows: List[Dict[str, Any]] = response.data or []

    if not rows:
        print("⚠️  No settled rows yet — calibration skipped.")
        return {}

    # Filter out voids (0.5 from NOT_FOUND) for main stats
    real_rows = [r for r in rows if float(r["actual_outcome"]) != 0.5]
    total = len(real_rows)
    if total == 0:
        print("⚠️  All settled rows are voids.")
        return {}

    wins = sum(1 for r in real_rows if float(r["actual_outcome"]) == 1.0)
    briers = [float(r["brier_contribution"]) for r in real_rows if r["brier_contribution"] is not None]
    avg_brier = round(sum(briers) / len(briers), 6) if briers else 0.0

    roi_sum = sum((float(r["best_bet_odds"]) * float(r["actual_outcome"])) - 1.0 for r in real_rows)
    roi_pct = round((roi_sum / total) * 100, 2)

    # By league
    by_league: Dict[str, dict] = {}
    for r in real_rows:
        lg = r["league"]
        if lg not in by_league:
            by_league[lg] = {"total": 0, "wins": 0, "roi": 0.0}
        by_league[lg]["total"] += 1
        outcome = float(r["actual_outcome"])
        odds = float(r["best_bet_odds"])
        by_league[lg]["wins"] += 1 if outcome == 1.0 else 0
        by_league[lg]["roi"] += (odds * outcome) - 1.0

    # By market
    by_market: Dict[str, dict] = {}
    for r in real_rows:
        mk = r["best_bet_type"]
        if mk not in by_market:
            by_market[mk] = {"total": 0, "wins": 0}
        by_market[mk]["total"] += 1
        if float(r["actual_outcome"]) == 1.0:
            by_market[mk]["wins"] += 1

    report = {
        "run_id":    run_id,
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_settled": total,
        "win_rate":  round(wins / total, 4),
        "avg_brier": avg_brier,
        "roi_pct":   roi_pct,
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

    print(f"\n📊  Calibration: {total} graded | Win rate: {report['win_rate']:.1%} | ROI: {roi_pct:+.1f}% | Brier: {avg_brier}")
    for lg, v in sorted(report["by_league"].items(), key=lambda x: -x[1]["win_rate"]):
        print(f"    {lg}: {v['n']} bets | {v['win_rate']:.1%} win | ROI {v['roi_pct']:+.1f}%")
    print()
    for mk, v in sorted(report["by_market"].items(), key=lambda x: -x[1]["win_rate"]):
        print(f"    {mk}: {v['n']} bets | {v['win_rate']:.1%} win")

    return report


# ---------------------------------------------------------------------------
# Discord performance summary
# ---------------------------------------------------------------------------

def _discord_report(report: Dict, settlement: Dict) -> None:
    if not report:
        newly = settlement.get("settled", 0)
        notify._send(f"ℹ️ **APEX-10 Settlement** — {newly} results graded this run. No historical data yet for full report.")
        return

    settled_now = settlement.get("settled", 0)
    total = report.get("total_settled", 0)
    win_rate = report.get("win_rate", 0)
    roi = report.get("roi_pct", 0)
    brier = report.get("avg_brier", 0)

    lines = [
        "📊 **APEX-10 Weekly Performance Report**",
        "",
        f"🎯 Win Rate: **{win_rate:.1%}** ({int(win_rate * total)}/{total} bets graded)",
        f"💰 Paper ROI: **{roi:+.1f}%**",
        f"📐 Brier Score: **{brier}** _(0 = perfect, lower is better)_",
        "",
        "**By League:**",
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
    print(f"\n{'='*60}")
    print(f"  APEX-10 Settlement — {run_id}")
    print(f"{'='*60}\n")

    settlement_result = settle_pending(db_client)
    report = emit_calibration_report(db_client, run_id)
    _discord_report(report, settlement_result)


if __name__ == "__main__":
    import logging
    import traceback
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    try:
        from apex10.db import get_client
        _db = get_client()
        run(_db)
    except Exception as _exc:
        print("\n" + "=" * 60)
        print("SETTLEMENT CRASHED — full traceback:")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        raise  # still exit non-zero so GitHub Actions marks it failed
