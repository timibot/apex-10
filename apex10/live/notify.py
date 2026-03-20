"""
APEX-10 Discord notification centre.
Single module for all outbound alerts — nothing else calls Discord directly.
All methods are fire-and-forget: they log on failure but never raise.
"""
from __future__ import annotations

import logging
from enum import Enum

import httpx

from apex10.config import get_api_config

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "ℹ️"
    SUCCESS = "✅"
    WARNING = "⚠️"
    CRITICAL = "🚨"


def _send(message: str) -> bool:
    """
    Send a message to the configured Discord webhook.
    Returns True on success, False on any failure. Never raises.
    """
    try:
        cfg = get_api_config()
        if not cfg.DISCORD_WEBHOOK:
            logger.debug(
                "No Discord webhook configured — skipping notification"
            )
            return False
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(cfg.DISCORD_WEBHOOK, json={"content": message})
            resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Discord notification failed: {e}")
        return False


def _fmt(level: AlertLevel, title: str, body: str) -> str:
    return f"{level.value} **APEX-10 — {title}**\n{body}"


# ── Public alert methods ───────────────────────────────────────────────────


def ticket_generated(
    week: str,
    legs: int,
    combined_odds: float,
    stake: float,
    win_rate: float,
    breakdown: str = "",
) -> bool:
    body = (
        f"Week: {week}\n"
        f"Legs: {legs} | Combined odds: {combined_odds:.2f}x\n"
        f"Stake: ₦{stake:,.2f} | Simulated win rate: {win_rate:.1%}"
    )
    if breakdown:
        body += f"\n\n**Ticket Breakdown:**\n{breakdown}"
        
    return _send(_fmt(
        AlertLevel.SUCCESS,
        "Ticket Generated",
        body,
    ))


def no_ticket_week(week: str, reason: str) -> bool:
    return _send(_fmt(
        AlertLevel.INFO,
        "No Ticket This Week",
        f"Week: {week}\nReason: {reason}",
    ))


def result_logged(
    week: str, result: str, new_bank: float, brier: float
) -> bool:
    level = AlertLevel.SUCCESS if result == "win" else AlertLevel.INFO
    return _send(_fmt(
        level,
        f"Result Logged — {result.upper()}",
        f"Week: {week}\n"
        f"Brier score: {brier:.4f}\n"
        f"Bank: ₦{new_bank:,.2f}",
    ))


def brier_breach(rolling_brier: float, threshold: float) -> bool:
    return _send(_fmt(
        AlertLevel.CRITICAL,
        "Brier Breach — Stakes Halved",
        f"Rolling 15-ticket Brier: **{rolling_brier:.4f}** "
        f"exceeds threshold {threshold}\n"
        f"Stakes halved automatically.\n"
        f"Action required: review SHAP drift, schedule retrain.",
    ))


def brier_recovered(rolling_brier: float) -> bool:
    return _send(_fmt(
        AlertLevel.SUCCESS,
        "Brier Recovered — Full Stakes Restored",
        f"Rolling Brier back to {rolling_brier:.4f} "
        f"— within acceptable range.",
    ))


def retrain_complete(
    version: int,
    lgbm_brier: float,
    xgb_brier: float,
    deployed: bool,
) -> bool:
    status = (
        "Deployed ✅"
        if deployed
        else "Not deployed — previous model retained"
    )
    return _send(_fmt(
        AlertLevel.INFO,
        "Annual Retrain Complete",
        f"New model version: v{version}\n"
        f"LightGBM Brier: {lgbm_brier:.4f} | "
        f"XGBoost Brier: {xgb_brier:.4f}\n"
        f"Status: {status}",
    ))


def paper_trading_complete(ticket_count: int, roi: float) -> bool:
    return _send(_fmt(
        AlertLevel.SUCCESS,
        "Paper Trading Exit Conditions Met — Ready for Live",
        f"Qualified tickets: {ticket_count}\n"
        f"Simulated ROI: {roi:.1%}\n"
        f"All three exit conditions passed. Deploy to live.",
    ))


def system_error(component: str, error: str) -> bool:
    return _send(_fmt(
        AlertLevel.CRITICAL,
        f"System Error — {component}",
        f"Error: {error}\nCheck GitHub Actions logs immediately.",
    ))
