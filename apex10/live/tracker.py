# Purpose: Log individual leg probabilities to track directional calibration during Paper Trading.
from typing import Any, Dict
from datetime import datetime, timezone

def log_prediction_leg(db_client: Any, ticket_id: str, leg: Dict[str, Any]) -> None:
    """
    Inserts a single predicted leg into the tracking database.
    'actual_outcome' initializes as NULL and is updated post-match by a settlement script.
    """
    payload = {
        "ticket_id": ticket_id,
        "fixture_id": leg["fixture_id"],
        "league": leg["league"],
        "market": leg["bet_type"],
        "market_odds": leg["odds"],
        "p_lgbm": round(leg["p_lgbm"], 4),
        "p_xgb": round(leg["p_xgb"], 4),
        "consensus_prob": round((leg["p_lgbm"] + leg["p_xgb"]) / 2, 4),
        "actual_outcome": None,  # 1.0 for Win, 0.0 for Loss, 0.5 for Push/Half-Loss
        "brier_contribution": None,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    try:
        # Assuming Supabase Python client syntax
        db_client.table("paper_trade_legs").insert(payload).execute()
    except Exception as e:
        print(f"⚠️ Failed to log prediction leg {leg['fixture_id']}: {e}")
