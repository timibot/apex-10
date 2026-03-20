"""
settlement.py
-------------
Runs on Monday mornings (post-weekend fixtures) to:
  1. Fetch final match results from the API
  2. Resolve each ungraded paper_trade_leg to actual_outcome
  3. Compute brier_contribution per leg
  4. Update the DB row in-place
  5. Emit a weekly calibration summary to diagnostics/

Cron: 0 7 * * 1  (Monday 07:00 UTC, before cache.py runs at 08:00)
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from apex10.live import notify

# ---------------------------------------------------------------------------
# Outcome resolution
# ---------------------------------------------------------------------------

MARKET_RESOLVERS = {
    "Home Win":           lambda r: 1.0 if r["home_goals"] > r["away_goals"] else 0.0,
    "Away Win":           lambda r: 1.0 if r["away_goals"] > r["home_goals"] else 0.0,
    "Double Chance 1X":   lambda r: 1.0 if r["home_goals"] >= r["away_goals"] else 0.0,
    "Double Chance X2":   lambda r: 1.0 if r["away_goals"] >= r["home_goals"] else 0.0,
    "Draw No Bet":        lambda r: 1.0 if r["home_goals"] > r["away_goals"] else (0.5 if r["home_goals"] == r["away_goals"] else 0.0),
    "DNB Away":           lambda r: 1.0 if r["away_goals"] > r["home_goals"] else (0.5 if r["home_goals"] == r["away_goals"] else 0.0),
    "Over 1.5 Goals":     lambda r: 1.0 if (r["home_goals"] + r["away_goals"]) >= 2 else 0.0,
    "Over 2.5 Goals":     lambda r: 1.0 if (r["home_goals"] + r["away_goals"]) >= 3 else 0.0,
    "Under 3.5 Goals":    lambda r: 1.0 if (r["home_goals"] + r["away_goals"]) <= 3 else 0.0,
    "BTTS Yes":           lambda r: 1.0 if r["home_goals"] >= 1 and r["away_goals"] >= 1 else 0.0,
    "BTTS No":            lambda r: 1.0 if r["home_goals"] == 0 or r["away_goals"] == 0 else 0.0,
}

def resolve_outcome(market: str, result: Dict[str, Any]) -> Optional[float]:
    """
    Returns 1.0 (win), 0.0 (loss), 0.5 (push / match void), or None if
    the market is unrecognised or the result is unavailable.
    """
    if result.get("status") != "FINISHED":
        return None  # Postponed / abandoned — leave NULL, retry next cycle

    resolver = MARKET_RESOLVERS.get(market)
    if resolver is None:
        print(f"⚠️  Unknown market '{market}' — cannot resolve.")
        return None

    return resolver(result)

def brier_contribution(predicted_prob: float, actual_outcome: float) -> float:
    """
    Single-observation Brier score component: (p - o)^2
    Lower is better. Range [0, 1].
    """
    return round((predicted_prob - actual_outcome) ** 2, 6)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_result(fixture_id: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Fetches the final score for a fixture from the configured results API.
    Replace the URL/params with your actual provider (API-Football, Opta, etc.)
    """
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {"x-apisports-key": api_key}
    params = {"id": fixture_id}

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        fixture = data["response"][0]
        status = fixture["fixture"]["status"]["short"]  # "FT", "PST", "CANC" etc.

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

def settle_pending_legs(db_client: Any, api_key: str) -> Dict[str, Any]:
    """
    Queries all ungraded legs (actual_outcome IS NULL), resolves each one,
    and updates the DB. Returns a settlement summary dict.
    """
    response = (
        db_client.table("paper_trade_legs")
        .select("*")
        .is_("actual_outcome", "null")
        .execute()
    )
    pending: List[Dict[str, Any]] = response.data or []

    if not pending:
        print("✅  No pending legs to settle.")
        return {"settled": 0, "skipped": 0, "errors": 0}

    settled = skipped = errors = 0

    for leg in pending:
        fixture_id = leg["fixture_id"]
        result = fetch_result(fixture_id, api_key)

        if result is None:
            errors += 1
            continue

        outcome = resolve_outcome(leg["market"], result)

        if outcome is None:
            # Match not finished or unknown market — retry next run
            skipped += 1
            continue

        # consensus_prob is what we hold the model accountable to
        consensus = leg["consensus_prob"]
        brier = brier_contribution(consensus, outcome)

        try:
            db_client.table("paper_trade_legs").update({
                "actual_outcome": outcome,
                "brier_contribution": brier,
                "settled_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", leg["id"]).execute()
            settled += 1
        except Exception as e:
            print(f"⚠️  DB update failed for leg {leg['id']}: {e}")
            errors += 1

    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_pending": len(pending),
        "settled": settled,
        "skipped": skipped,
        "errors": errors,
    }
    print(f"📋  Settlement complete: {summary}")
    return summary

# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------

def emit_calibration_report(db_client: Any, run_id: str) -> None:
    """
    Reads all settled legs and emits a calibration summary JSON.
    Groups by model (lgbm / xgb / consensus) and computes:
      - Mean Brier score
      - Reliability bins (predicted prob bucket → actual win rate)
    This is the data you need to plot the reliability diagram.
    """
    response = (
        db_client.table("paper_trade_legs")
        .select("p_lgbm, p_xgb, consensus_prob, actual_outcome, league, market")
        .not_.is_("actual_outcome", "null")
        .execute()
    )
    legs: List[Dict[str, Any]] = response.data or []

    if not legs:
        print("⚠️  No settled legs yet — calibration report skipped.")
        return

    # Per-model Brier scores
    def mean_brier(probs, outcomes):
        return round(sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(outcomes), 6)

    outcomes   = [float(l["actual_outcome"]) for l in legs]
    lgbm_probs = [float(l["p_lgbm"])         for l in legs]
    xgb_probs  = [float(l["p_xgb"])          for l in legs]
    cons_probs = [float(l["consensus_prob"]) for l in legs]

    # Reliability bins — 10 buckets [0.0–0.1), [0.1–0.2), …, [0.9–1.0]
    def reliability_bins(probs, outcomes, n_bins=10):
        bins = {i: {"count": 0, "sum_probs": 0.0, "sum_outcomes": 0.0}
                for i in range(n_bins)}
        for p, o in zip(probs, outcomes):
            bucket = min(int(p * n_bins), n_bins - 1)
            bins[bucket]["count"] += 1
            bins[bucket]["sum_probs"] += p
            bins[bucket]["sum_outcomes"] += o

        result = {}
        for i, b in bins.items():
            if b["count"] == 0:
                continue
            lower = round(i / n_bins, 1)
            upper = round((i + 1) / n_bins, 1)
            label = f"{lower:.1f}-{upper:.1f}"
            result[label] = {
                "n":              b["count"],
                "mean_predicted": round(b["sum_probs"]    / b["count"], 4),
                "actual_win_rate": round(b["sum_outcomes"] / b["count"], 4),
            }
        return result

    # Per-league breakdown
    league_brier: Dict[str, list] = {}
    for leg in legs:
        lg = leg["league"]
        if lg not in league_brier:
            league_brier[lg] = []
        league_brier[lg].append((float(leg["consensus_prob"]), float(leg["actual_outcome"])))

    league_summary = {
        lg: {
            "n": len(pairs),
            "brier": round(sum((p - o) ** 2 for p, o in pairs) / len(pairs), 6),
        }
        for lg, pairs in league_brier.items()
    }

    report = {
        "run_id":    run_id,
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_legs":    len(legs),
        "brier_scores": {
            "lgbm":      mean_brier(lgbm_probs, outcomes),
            "xgb":       mean_brier(xgb_probs,  outcomes),
            "consensus": mean_brier(cons_probs,  outcomes),
        },
        "reliability_bins": {
            "lgbm":      reliability_bins(lgbm_probs, outcomes),
            "xgb":       reliability_bins(xgb_probs,  outcomes),
            "consensus": reliability_bins(cons_probs,  outcomes),
        },
        "by_league": league_summary,
    }

    os.makedirs("apex10/diagnostics", exist_ok=True)
    filepath = f"apex10/diagnostics/calibration_{run_id}.json"
    with open(filepath, "w") as f:
        json.dump(report, f, indent=4)

    print(f"📊  Calibration report written to {filepath}")
    print(f"    Brier → LightGBM: {report['brier_scores']['lgbm']} | "
          f"XGBoost: {report['brier_scores']['xgb']} | "
          f"Consensus: {report['brier_scores']['consensus']}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(db_client: Any, api_key: str) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    print(f"\n{'='*60}")
    print(f"  APEX-10 Settlement — {run_id}")
    print(f"{'='*60}\n")

    settlement_result = settle_pending_legs(db_client, api_key)
    emit_calibration_report(db_client, run_id)

    # ── Discord notification ──────────────────────────────────────────
    settled = settlement_result.get("settled", 0)
    skipped = settlement_result.get("skipped", 0)
    errors = settlement_result.get("errors", 0)
    total = settlement_result.get("total_pending", 0)

    if total > 0:
        # Compute overall Brier from settled legs
        try:
            resp = (
                db_client.table("paper_trade_legs")
                .select("brier_contribution")
                .not_.is_("brier_contribution", "null")
                .execute()
            )
            briers = [float(r["brier_contribution"]) for r in (resp.data or [])]
            avg_brier = sum(briers) / len(briers) if briers else 0.0
        except Exception:
            avg_brier = 0.0

        result_str = f"{settled} settled, {skipped} skipped, {errors} errors"
        notify.result_logged(
            week=run_id,
            result=result_str,
            new_bank=0.0,  # paper trading
            brier=round(avg_brier, 4),
        )
    else:
        notify._send("ℹ️ **APEX-10 Settlement** — No pending legs to settle this week.")

if __name__ == "__main__":
    from apex10.db import get_client
    from apex10.config import get_api_config

    _db = get_client()
    _api_key = get_api_config().API_FOOTBALL_KEY

    run(_db, _api_key)
