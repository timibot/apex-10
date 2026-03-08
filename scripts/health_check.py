"""
Manual health dashboard — run anytime to see full system status.
python scripts/health_check.py
"""
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from dotenv import load_dotenv

load_dotenv()

from apex10.config import MODEL  # noqa: E402
from apex10.db import get_client  # noqa: E402
from apex10.live.health import get_rolling_brier  # noqa: E402
from apex10.score.score import check_paper_trading_exit  # noqa: E402


def print_dashboard():
    db = get_client()

    print("\n" + "═" * 50)
    print("  APEX-10 · System Health Dashboard")
    print("═" * 50)

    # ── Bank state ────────────────────────────────────────────────────────
    bank = db.table("bank_state").select("*").eq("id", 1).execute()
    if bank.data:
        b = bank.data[0]
        current = float(b["current_bank"])
        initial = float(b["initial_bank"])
        roi = (current - initial) / initial
        multiplier = float(b.get("stake_multiplier") or 1.0)
        print(f"\n  Bank:          ₦{current:>12,.2f}")
        print(f"  Initial:       ₦{initial:>12,.2f}")
        print(f"  ROI:           {roi:>+.1%}")
        status = "⚠️ HALVED" if multiplier < 1.0 else "✅ NORMAL"
        print(f"  Stake mult:    {multiplier}x {status}")

    # ── Brier health ──────────────────────────────────────────────────────
    rolling = get_rolling_brier(db)
    print(f"\n  Rolling Brier: {rolling if rolling else 'Insufficient data'}")
    if rolling:
        if rolling > MODEL.BRIER_LIVE_ALERT:
            print(f"  Brier status:  🚨 BREACH (>{MODEL.BRIER_LIVE_ALERT})")
        elif rolling > MODEL.BRIER_GATE:
            print(f"  Brier status:  ⚠️  WARNING (>{MODEL.BRIER_GATE})")
        else:
            print(f"  Brier status:  ✅ HEALTHY (<{MODEL.BRIER_GATE})")

    # ── Ticket history ────────────────────────────────────────────────────
    tickets = (
        db.table("tickets")
        .select("status")
        .not_.eq("status", "no_ticket")
        .execute()
    )
    if tickets.data:
        statuses = [t["status"] for t in tickets.data]
        wins = statuses.count("win")
        losses = statuses.count("loss")
        pending = statuses.count("pending")
        total = wins + losses
        print(f"\n  Tickets:       {len(statuses)} total")
        print(f"  Record:        {wins}W / {losses}L / {pending} pending")
        if total > 0:
            print(f"  Hit rate:      {wins / total:.1%}")

    # ── Paper trading exit ────────────────────────────────────────────────
    exit_check = check_paper_trading_exit(db)
    ready = "✅ YES" if exit_check["ready_for_live"] else "❌ NOT YET"
    print(f"\n  Live ready:    {ready}")
    for key, cond in exit_check["conditions"].items():
        icon = "✅" if cond["passed"] else "❌"
        print(
            f"    {icon} {key}: {cond['value']} "
            f"(threshold: {cond['threshold']})"
        )

    # ── Active model ──────────────────────────────────────────────────────
    model = (
        db.table("model_params")
        .select("model_name, version, brier_score, trained_at")
        .eq("is_active", True)
        .execute()
    )
    if model.data:
        print("\n  Active models:")
        for m in model.data:
            print(
                f"    {m['model_name']} v{m['version']} "
                f"— Brier {m['brier_score']:.4f} "
                f"(trained {m['trained_at'][:10]})"
            )

    print("\n" + "═" * 50 + "\n")


if __name__ == "__main__":
    print_dashboard()
