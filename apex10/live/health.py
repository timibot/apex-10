"""
Rolling Brier health monitor.
Runs after every result is logged.
Manages the stake multiplier — halves stakes on breach, restores on recovery.
Reads/writes stake_multiplier to bank_state.
"""
from __future__ import annotations

import logging

from apex10.config import MODEL
from apex10.db import get_client
from apex10.live.notify import brier_breach, brier_recovered

logger = logging.getLogger(__name__)

STAKE_MULTIPLIER_NORMAL = 1.0
STAKE_MULTIPLIER_HALVED = 0.5


def get_stake_multiplier(db) -> float:
    """
    Read current stake multiplier from bank_state.
    Defaults to 1.0 if column does not exist yet.
    """
    result = (
        db.table("bank_state")
        .select("stake_multiplier")
        .eq("id", 1)
        .execute()
    )
    if not result.data:
        return STAKE_MULTIPLIER_NORMAL
    return float(
        result.data[0].get("stake_multiplier") or STAKE_MULTIPLIER_NORMAL
    )


def set_stake_multiplier(db, multiplier: float) -> None:
    db.table("bank_state").update(
        {"stake_multiplier": multiplier}
    ).eq("id", 1).execute()
    logger.info(f"Stake multiplier set to {multiplier}")


def get_rolling_brier(
    db, window: int = MODEL.ROLLING_BRIER_WINDOW
) -> float | None:
    """
    Fetch rolling Brier score over last `window` tickets from brier_log.
    Returns None if insufficient data.
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
        return None

    return round(sum(scores) / len(scores), 4)


def run_health_check(db=None) -> dict:
    """
    Evaluate current system health. Triggered after every result log.

    Actions:
      - Rolling Brier > 0.24 → halve stakes + Discord alert
      - Rolling Brier recovers to <= 0.20 → restore stakes + Discord alert
      - Neither threshold crossed → no action
    """
    if db is None:
        db = get_client()

    rolling_brier = get_rolling_brier(db)
    current_multiplier = get_stake_multiplier(db)

    status = {
        "rolling_brier": rolling_brier,
        "stake_multiplier": current_multiplier,
        "action_taken": None,
        "alert_sent": False,
    }

    if rolling_brier is None:
        logger.info(
            "Insufficient Brier history for health check — no action"
        )
        status["note"] = "Insufficient history"
        return status

    # ── Breach: Brier too high → halve stakes ────────────────────────────
    if rolling_brier > MODEL.BRIER_LIVE_ALERT:
        if current_multiplier != STAKE_MULTIPLIER_HALVED:
            set_stake_multiplier(db, STAKE_MULTIPLIER_HALVED)
            alert_sent = brier_breach(
                rolling_brier, MODEL.BRIER_LIVE_ALERT
            )
            status["action_taken"] = "stakes_halved"
            status["alert_sent"] = alert_sent
            logger.warning(
                f"⚠️ Brier breach: {rolling_brier:.4f} > "
                f"{MODEL.BRIER_LIVE_ALERT} — stakes halved"
            )
        else:
            logger.warning(
                f"Brier still breached: {rolling_brier:.4f} "
                f"— stakes already halved"
            )
            status["note"] = "Already halved"

    # ── Recovery: Brier back within gate → restore stakes ────────────────
    elif (
        rolling_brier <= MODEL.BRIER_GATE
        and current_multiplier == STAKE_MULTIPLIER_HALVED
    ):
        set_stake_multiplier(db, STAKE_MULTIPLIER_NORMAL)
        alert_sent = brier_recovered(rolling_brier)
        status["action_taken"] = "stakes_restored"
        status["alert_sent"] = alert_sent
        logger.info(
            f"✅ Brier recovered: {rolling_brier:.4f} — stakes restored"
        )

    else:
        logger.info(
            f"Brier healthy: {rolling_brier:.4f} — no action needed"
        )
        status["action_taken"] = "none"

    return status


def get_adjusted_stake(base_stake: float, db=None) -> float:
    """
    Apply stake multiplier to base Quarter-Kelly stake.
    Called by ticket.py before logging the stake.
    """
    if db is None:
        db = get_client()
    multiplier = get_stake_multiplier(db)
    adjusted = round(base_stake * multiplier, 2)
    if multiplier != STAKE_MULTIPLIER_NORMAL:
        logger.info(
            f"Stake adjusted: {base_stake:.2f} × {multiplier} = "
            f"{adjusted:.2f}"
        )
    return adjusted
