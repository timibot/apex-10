"""
Monday result logger.
Runs every Monday morning before cache.py.
Fetches last week's results from API-Football, matches against
logged ticket predictions in Supabase, marks each leg and the
full ticket win/loss, updates Brier score tracking and bank state.
Zero manual work.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta

import httpx

from apex10.config import MODEL, STAKING, get_api_config
from apex10.db import get_client
from apex10.live.health import run_health_check
from apex10.live.notify import result_logged as notify_result

logger = logging.getLogger(__name__)


# ── Result fetching ───────────────────────────────────────────────────────


def fetch_results(
    match_date_from: date, match_date_to: date
) -> dict[int, dict]:
    """
    Fetch finished match results from API-Football for a date range.
    Returns {api_match_id: result_dict}.
    """
    cfg = get_api_config()
    url = f"{cfg.API_FOOTBALL_BASE}/fixtures"
    headers = {
        "x-apisports-key": cfg.API_FOOTBALL_KEY,
        "x-apisports-host": "v3.football.api-sports.io",
    }
    params = {
        "status": "FT",
        "from": str(match_date_from),
        "to": str(match_date_to),
    }

    logger.info(f"Fetching results: {match_date_from} → {match_date_to}")

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers=headers, params=params)
        response.raise_for_status()

    fixtures = response.json().get("response", [])
    logger.info(f"Received {len(fixtures)} finished fixtures")

    results = {}
    for f in fixtures:
        try:
            match_id = f["fixture"]["id"]
            results[match_id] = {
                "home_goals": f["goals"]["home"],
                "away_goals": f["goals"]["away"],
                "home_team": f["teams"]["home"]["name"],
                "away_team": f["teams"]["away"]["name"],
            }
        except (KeyError, TypeError):
            continue

    return results


# ── Leg result evaluation ─────────────────────────────────────────────────


def evaluate_leg(leg: dict, result: dict) -> str:
    """
    Determine win/loss/void for a single ticket leg.

    Args:
        leg:    Leg dict from tickets.legs JSON field.
        result: Result dict {home_goals, away_goals}.

    Returns:
        "win" | "loss" | "void"
    """
    bet_type = leg.get("bet_type", "")
    home_g = result.get("home_goals")
    away_g = result.get("away_goals")

    if home_g is None or away_g is None:
        return "void"

    home_g, away_g = int(home_g), int(away_g)
    total_goals = home_g + away_g

    if bet_type == "home_win":
        return "win" if home_g > away_g else "loss"

    elif bet_type == "away_win":
        return "win" if away_g > home_g else "loss"

    elif bet_type == "dnb_home":
        if home_g > away_g:
            return "win"
        elif home_g == away_g:
            return "void"  # Stake returned
        return "loss"

    elif bet_type == "ah_minus_0_5":
        return "win" if home_g > away_g else "loss"

    elif bet_type == "ah_minus_1_0":
        diff = home_g - away_g
        if diff >= 2:
            return "win"
        elif diff == 1:
            return "void"  # Push — stake returned
        return "loss"

    elif bet_type == "ah_minus_1_5":
        return "win" if home_g - away_g >= 2 else "loss"

    elif bet_type == "over_1_5":
        return "win" if total_goals >= 2 else "loss"

    elif bet_type == "over_2_5":
        return "win" if total_goals >= 3 else "loss"

    elif bet_type == "under_3_5":
        return "win" if total_goals <= 3 else "loss"

    elif bet_type == "btts_no":
        both_scored = home_g > 0 and away_g > 0
        return "loss" if both_scored else "win"

    elif bet_type == "dc_1x":
        return "win" if home_g >= away_g else "loss"

    elif bet_type == "dc_x2":
        return "win" if away_g >= home_g else "loss"

    elif bet_type == "home_scores":
        return "win" if home_g > 0 else "loss"

    logger.warning(f"Unknown bet type: {bet_type} — marking void")
    return "void"


def evaluate_ticket(legs: list[dict], results: dict[int, dict]) -> dict:
    """
    Evaluate all legs in a ticket against match results.
    Ticket wins only if all non-void legs win.
    """
    leg_results = []

    for leg in legs:
        match_id = leg.get("fixture_id")
        if match_id not in results:
            leg_results.append("no_data")
            continue
        leg_results.append(evaluate_leg(leg, results[match_id]))

    # Exclude void and no_data legs from ticket result
    decisive = [r for r in leg_results if r in ("win", "loss")]

    if not decisive:
        ticket_result = "no_data"
    elif all(r == "win" for r in decisive):
        ticket_result = "win"
    else:
        ticket_result = "loss"

    return {
        "leg_results": leg_results,
        "ticket_result": ticket_result,
        "legs_evaluated": len(decisive),
        "legs_void": leg_results.count("void"),
    }


# ── Brier score computation ───────────────────────────────────────────────


def compute_brier_score(
    legs: list[dict], leg_results: list[str]
) -> float | None:
    """
    Compute Brier score for a ticket: mean((prob - outcome)^2).
    Void and no_data legs excluded. Returns None if no decisive legs.
    """
    scores = []
    for leg, lr in zip(legs, leg_results):
        if lr not in ("win", "loss"):
            continue
        prob = float(leg.get("consensus_prob", 0.5))
        outcome = 1.0 if lr == "win" else 0.0
        scores.append((prob - outcome) ** 2)

    if not scores:
        return None

    return round(sum(scores) / len(scores), 4)


def compute_rolling_brier(
    db, window: int = MODEL.ROLLING_BRIER_WINDOW
) -> float | None:
    """
    Compute rolling Brier score over the last `window` logged tickets.
    Returns None if fewer than window tickets exist.
    """
    result = (
        db.table("brier_log")
        .select("brier_score")
        .not_.is_("brier_score", "null")
        .order("logged_at", desc=True)
        .limit(window)
        .execute()
    )

    scores = [row["brier_score"] for row in (result.data or [])]

    if len(scores) < window:
        logger.info(
            f"Only {len(scores)} Brier scores logged "
            f"— rolling not yet available"
        )
        return None

    return round(sum(scores) / len(scores), 4)


# ── Bank state ────────────────────────────────────────────────────────────

REINVEST_FRACTION = 0.70  # 70% back into bank
EXTRACT_FRACTION = 0.30  # 30% realised profit


def update_bank(
    db, result: str, stake: float, combined_odds: float = 1.0
) -> dict:
    """
    Update bank_state after a ticket result.
    Win: 70% of net profit reinvested, 30% extracted.
    Loss: stake deducted in full.
    Returns dict with new_bank, extracted_profit, net_profit.
    """
    bank_row = db.table("bank_state").select("*").eq("id", 1).execute()
    if not bank_row.data:
        raise ValueError("bank_state not initialised")

    row = bank_row.data[0]
    current_bank = float(row["current_bank"])
    extracted_so_far = float(row.get("extracted_profit") or 0.0)
    total_wins = int(row.get("total_wins") or 0)
    total_losses = int(row.get("total_losses") or 0)

    net_profit = 0.0
    extracted_this_week = 0.0

    if result == "win":
        gross_profit = stake * (combined_odds - 1.0)
        reinvested = round(gross_profit * REINVEST_FRACTION, 2)
        extracted_this_week = round(gross_profit * EXTRACT_FRACTION, 2)
        new_bank = round(current_bank + reinvested, 2)
        net_profit = gross_profit
        total_wins += 1
        logger.info(
            f"WIN: gross_profit=₦{gross_profit:.2f}, "
            f"reinvested=₦{reinvested:.2f} (70%), "
            f"extracted=₦{extracted_this_week:.2f} (30%)"
        )
    elif result == "loss":
        new_bank = round(current_bank - stake, 2)
        total_losses += 1
        logger.info(f"LOSS: stake deducted ₦{stake:.2f}")
    else:
        new_bank = current_bank  # void / no_data — no change
        logger.info(f"No bank change: result={result}")

    db.table("bank_state").update({
        "current_bank": new_bank,
        "extracted_profit": round(
            extracted_so_far + extracted_this_week, 2
        ),
        "total_wins": total_wins,
        "total_losses": total_losses,
    }).eq("id", 1).execute()

    return {
        "new_bank": new_bank,
        "net_profit": round(net_profit, 2),
        "extracted_this_week": extracted_this_week,
        "total_extracted": round(
            extracted_so_far + extracted_this_week, 2
        ),
    }


# ── Paper trading exit gate ───────────────────────────────────────────────


def check_paper_trading_exit(db) -> dict:
    """
    Check all three paper trading exit conditions simultaneously.
    All three must be met before going live.

    Condition 1: ticket_count >= 20
    Condition 2: brier_variance(last_10) < 0.02
    Condition 3: simulated_roi >= -0.05
    """
    # ── Condition 1: ticket count ─────────────────────────────────────────
    ticket_resp = (
        db.table("tickets")
        .select("id", count="exact")
        .not_.eq("status", "no_ticket")
        .execute()
    )
    ticket_count = ticket_resp.count or 0
    cond_1 = ticket_count >= MODEL.MIN_PAPER_TICKETS

    # ── Condition 2: Brier variance over last 10 ──────────────────────────
    brier_resp = (
        db.table("brier_log")
        .select("brier_score")
        .not_.is_("brier_score", "null")
        .order("logged_at", desc=True)
        .limit(10)
        .execute()
    )
    brier_scores = [row["brier_score"] for row in (brier_resp.data or [])]

    if len(brier_scores) >= 10:
        import statistics

        brier_variance = round(statistics.variance(brier_scores), 5)
        cond_2 = brier_variance < MODEL.BRIER_VARIANCE_GATE
    else:
        brier_variance = None
        cond_2 = False

    # ── Condition 3: simulated ROI ────────────────────────────────────────
    bank_resp = db.table("bank_state").select("*").eq("id", 1).execute()
    if bank_resp.data:
        current_bank = float(bank_resp.data[0]["current_bank"])
        initial_bank = float(bank_resp.data[0]["initial_bank"])
        simulated_roi = round(
            (current_bank - initial_bank) / initial_bank, 4
        )
        cond_3 = simulated_roi >= STAKING.SIMULATED_ROI_FLOOR
    else:
        simulated_roi = None
        cond_3 = False

    all_passed = cond_1 and cond_2 and cond_3

    result = {
        "ready_for_live": all_passed,
        "conditions": {
            "c1_ticket_count": {
                "value": ticket_count,
                "threshold": MODEL.MIN_PAPER_TICKETS,
                "passed": cond_1,
            },
            "c2_brier_variance": {
                "value": brier_variance,
                "threshold": MODEL.BRIER_VARIANCE_GATE,
                "passed": cond_2,
            },
            "c3_simulated_roi": {
                "value": simulated_roi,
                "threshold": STAKING.SIMULATED_ROI_FLOOR,
                "passed": cond_3,
            },
        },
    }

    if all_passed:
        logger.info(
            "✅ All paper trading exit conditions met — READY FOR LIVE"
        )
    else:
        failed = [
            k for k, v in result["conditions"].items() if not v["passed"]
        ]
        logger.info(
            f"Paper trading continuing — conditions not yet met: {failed}"
        )

    return result


# ── Orchestrator ──────────────────────────────────────────────────────────


def run_weekly_scoring(target_week: date | None = None) -> dict:
    """
    Full Monday scoring pipeline.
    1. Find last week's pending ticket
    2. Fetch match results from API-Football
    3. Evaluate all legs
    4. Compute and log Brier score
    5. Update bank state
    6. Check paper trading exit conditions
    """
    db = get_client()

    if target_week is None:
        today = date.today()
        # Last week's Monday
        target_week = today - timedelta(days=today.weekday() + 7)

    logger.info(f"Scoring week: {target_week}")

    # ── 1. Load pending ticket ────────────────────────────────────────────
    ticket_resp = (
        db.table("tickets")
        .select("*")
        .eq("week_start", str(target_week))
        .eq("status", "pending")
        .execute()
    )

    if not ticket_resp.data:
        logger.info(
            f"No pending ticket found for {target_week} — nothing to score"
        )
        return {"scored": False, "reason": "No pending ticket"}

    ticket_row = ticket_resp.data[0]
    ticket_id = ticket_row["id"]
    legs = ticket_row["legs"]
    stake = float(ticket_row.get("stake") or 0.0)

    if not legs:
        logger.info("No-ticket week — marking scored with no action")
        db.table("tickets").update(
            {"status": "no_action"}
        ).eq("id", ticket_id).execute()
        return {"scored": True, "result": "no_action"}

    # ── 2. Fetch results for match week ───────────────────────────────────
    week_end = target_week + timedelta(days=7)
    results = fetch_results(target_week, week_end)

    # ── 3. Evaluate legs ──────────────────────────────────────────────────
    evaluation = evaluate_ticket(legs, results)
    ticket_result = evaluation["ticket_result"]

    # ── 4. Compute Brier score ────────────────────────────────────────────
    brier = compute_brier_score(legs, evaluation["leg_results"])

    # ── 5. Update ticket in DB ────────────────────────────────────────────
    db.table("tickets").update({
        "status": ticket_result,
        "result_logged_at": date.today().isoformat(),
        "brier_score": brier,
    }).eq("id", ticket_id).execute()

    # ── 6. Log per-leg results ────────────────────────────────────────────
    for leg, leg_result in zip(legs, evaluation["leg_results"]):
        db.table("ticket_legs").insert({
            "ticket_id": ticket_id,
            "api_match_id": leg.get("fixture_id"),
            "bet_type": leg.get("bet_type"),
            "odds": leg.get("odds"),
            "consensus_prob": leg.get("consensus_prob"),
            "result": leg_result,
        }).execute()

    # ── 7. Log Brier score ────────────────────────────────────────────────
    if brier is not None:
        rolling = compute_rolling_brier(db)
        db.table("brier_log").insert({
            "ticket_id": ticket_id,
            "brier_score": brier,
            "rolling_brier_15": rolling,
        }).execute()

    # ── 8. Update bank ────────────────────────────────────────────────────
    combined_odds = float(ticket_row.get("combined_odds") or 1.0)
    bank_update = update_bank(db, ticket_result, stake, combined_odds)
    new_bank = bank_update["new_bank"]

    # ── 9. Run health check ───────────────────────────────────────────────
    health = run_health_check(db)

    # ── 10. Check paper trading exit conditions ───────────────────────────
    exit_check = check_paper_trading_exit(db)

    # ── 11. Discord notification ───────────────────────────────────────────
    if brier is not None:
        notify_result(
            week=str(target_week),
            result=ticket_result,
            new_bank=new_bank,
            brier=brier,
        )

    summary = {
        "scored": True,
        "week": str(target_week),
        "ticket_id": ticket_id,
        "ticket_result": ticket_result,
        "legs_evaluated": evaluation["legs_evaluated"],
        "legs_void": evaluation["legs_void"],
        "brier_score": brier,
        "new_bank": new_bank,
        "stake_multiplier": health["stake_multiplier"],
        "health_action": health["action_taken"],
        "paper_trading_exit": exit_check,
    }

    logger.info(f"Scoring complete: {summary}")
    return summary


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO)
    run_weekly_scoring()
