"""
settlement.py
-------------
Runs on Tuesdays at 08:00 UTC to grade last weekend's predictions.

For each row in `upcoming_fixtures` where:
  - match_date < today  (game has passed)
  - actual_outcome IS NULL  (not yet graded)

It fetches the final score from API-Football, resolves the bet outcome,
computes the Brier contribution, and writes results back to the table.

Also emits a running performance report to Discord.
"""

import json
import os
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
    """
    Returns 1.0 (win), 0.0 (loss), 0.5 (push/void), or None
    if the market is unknown or the match hasn't finished.
    """
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
# API-Football result fetch
# ---------------------------------------------------------------------------

def fetch_result(fixture_id: int, api_key: str) -> Optional[Dict[str, Any]]:
    """Fetch the final score for a fixture from API-Football."""
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": api_key}
    params = {"id": fixture_id}

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        if not data.get("response"):
            return {"status": "NOT_FOUND"}

        fixture = data["response"][0]
        status = fixture["fixture"]["status"]["short"]  # "FT", "PST", "CANC", etc.

        if status not in ("FT", "AET", "PEN"):
            return {"status": "NOT_FINISHED"}

        return {
            "status": "FINISHED",
            "home_goals": fixture["goals"]["home"],
            "away_goals": fixture["goals"]["away"],
        }
    except Exception as e:
        print(f"⚠️  API error fetching fixture {fixture_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main settlement loop
# ---------------------------------------------------------------------------

def settle_pending(db_client: Any, api_key: str) -> Dict[str, Any]:
    """
    Find all upcoming_fixtures from past match dates that haven't been graded,
    fetch results, resolve outcomes, and write back to the database.
    """
    today_str = date.today().isoformat()

    response = (
        db_client.table("upcoming_fixtures")
        .select("id, api_match_id, league, home_team, away_team, match_date, best_bet_type, consensus_prob")
        .lt("match_date", today_str)       # Game has passed
        .is_("actual_outcome", "null")     # Not yet graded
        .execute()
    )
    pending: List[Dict[str, Any]] = response.data or []

    if not pending:
        print("✅  No pending fixtures to settle.")
        return {"settled": 0, "skipped": 0, "errors": 0, "total": 0}

    print(f"📋  Found {len(pending)} ungraded fixtures to settle...")
    settled = skipped = errors = 0

    for fix in pending:
        fixture_id = fix["api_match_id"]
        result = fetch_result(fixture_id, api_key)

        if result is None:
            errors += 1
            continue

        outcome = resolve_outcome(fix["best_bet_type"], result)

        if outcome is None:
            skipped += 1
            continue

        brier = brier_contribution(float(fix["consensus_prob"]), outcome)
        outcome_label = "WIN" if outcome == 1.0 else ("PUSH" if outcome == 0.5 else "LOSS")
        print(f"  {'✅' if outcome == 1.0 else '❌'} {fix['home_team']} vs {fix['away_team']} "
              f"[{fix['best_bet_type']}] → {outcome_label} | brier={brier}")

        try:
            db_client.table("upcoming_fixtures").update({
                "actual_outcome":     outcome,
                "brier_contribution": brier,
                "settled_at":         datetime.now(timezone.utc).isoformat(),
            }).eq("id", fix["id"]).execute()
            settled += 1
        except Exception as e:
            print(f"⚠️  DB update failed for id {fix['id']}: {e}")
            errors += 1

    print(f"\n📊  Settlement: {settled} settled, {skipped} skipped (not finished), {errors} errors")
    return {"settled": settled, "skipped": skipped, "errors": errors, "total": len(pending)}


# ---------------------------------------------------------------------------
# Calibration report
# ---------------------------------------------------------------------------

def emit_calibration_report(db_client: Any, run_id: str) -> Dict:
    """
    Reads all settled rows from upcoming_fixtures and computes:
    - Win rate per league
    - Win rate per market/bet type
    - Overall ROI (paper trade)
    - Brier score
    """
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

    total = len(rows)
    wins = sum(1 for r in rows if float(r["actual_outcome"]) == 1.0)
    briers = [float(r["brier_contribution"]) for r in rows if r["brier_contribution"] is not None]
    avg_brier = round(sum(briers) / len(briers), 6) if briers else 0.0

    # ROI calculation: sum(odds * outcome - 1) / n
    roi_sum = sum((float(r["best_bet_odds"]) * float(r["actual_outcome"])) - 1.0 for r in rows)
    roi_pct = round((roi_sum / total) * 100, 2)

    # By league
    by_league: Dict[str, dict] = {}
    for r in rows:
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
    for r in rows:
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
    filepath = f"apex10/diagnostics/calibration_{run_id}.json"
    with open(filepath, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n📊  Calibration: {total} settled | Win rate: {report['win_rate']:.1%} | ROI: {roi_pct:+.1f}% | Brier: {avg_brier}")
    for lg, v in report["by_league"].items():
        print(f"    {lg}: {v['n']} bets | {v['win_rate']:.1%} win | ROI {v['roi_pct']:+.1f}%")
    return report


# ---------------------------------------------------------------------------
# Discord performance summary
# ---------------------------------------------------------------------------

def _discord_report(report: Dict, settlement: Dict) -> None:
    if not report:
        notify._send("ℹ️ **APEX-10 Settlement** — No settled results yet.")
        return

    settled = settlement.get("settled", 0)
    total = report.get("total_settled", 0)
    win_rate = report.get("win_rate", 0)
    roi = report.get("roi_pct", 0)
    brier = report.get("avg_brier", 0)

    lines = [
        f"📊 **APEX-10 Weekly Performance Report**",
        f"",
        f"🎯 Win Rate: **{win_rate:.1%}** ({int(win_rate * total)}/{total} bets)",
        f"💰 Paper ROI: **{roi:+.1f}%**",
        f"📐 Brier Score: **{brier}** _(lower = better, perfect = 0)_",
        f"",
        f"**By League:**",
    ]
    for lg, v in report.get("by_league", {}).items():
        lines.append(f"  {lg}: {v['win_rate']:.1%} win | ROI {v['roi_pct']:+.1f}% ({v['n']} bets)")
    lines.append("")
    lines.append(f"**By Market:**")
    for mk, v in sorted(report.get("by_market", {}).items(), key=lambda x: -x[1]["win_rate"]):
        lines.append(f"  {mk}: {v['win_rate']:.1%} ({v['n']} bets)")
    lines.append("")
    lines.append(f"_This week: {settled} new results graded_")

    notify._send("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(db_client: Any, api_key: str) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    print(f"\n{'='*60}")
    print(f"  APEX-10 Settlement — {run_id}")
    print(f"{'='*60}\n")

    settlement_result = settle_pending(db_client, api_key)
    report = emit_calibration_report(db_client, run_id)
    _discord_report(report, settlement_result)


if __name__ == "__main__":
    from apex10.db import get_client
    from apex10.config import get_api_config

    _db = get_client()
    _cfg = get_api_config()
    run(_db, _cfg.API_FOOTBALL_KEY)
