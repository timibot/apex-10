"""
Initialise bank_state table with starting capital.
Run once: python -m apex10.scripts.init_bank

Safe to re-run — uses upsert on id=1 (single-row table).
"""
from __future__ import annotations

import logging
import sys

from apex10.config import APEX_ENV
from apex10.db import get_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Starting bank for paper trading — notional capital, no real money at risk
DEFAULT_PAPER_BANK = 1000.00
DEFAULT_PRODUCTION_BANK = 1000.00


def init_bank(starting_bank: float | None = None) -> dict:
    """
    Insert or reset the bank_state row.
    Returns the bank state dict.
    """
    if starting_bank is None:
        starting_bank = (
            DEFAULT_PAPER_BANK if APEX_ENV == "PAPER_TRADE" else DEFAULT_PRODUCTION_BANK
        )

    db = get_client()

    bank_row = {
        "id": 1,
        "current_bank": starting_bank,
        "initial_bank": starting_bank,
        "extracted_profit": 0.0,
        "total_wins": 0,
        "total_losses": 0,
        "stake_multiplier": 1.0,
    }

    import time
    max_retries = 5
    for attempt in range(max_retries):
        try:
            db.table("bank_state").upsert(bank_row, on_conflict="id").execute()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"Supabase connection failed (attempt {attempt+1}), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise

    logger.info(f"✅ Bank initialised: £{starting_bank:.2f} ({APEX_ENV})")
    logger.info(f"   Environment: {APEX_ENV}")
    logger.info(f"   Stake multiplier: 1.0")

    if APEX_ENV == "PAPER_TRADE":
        logger.warning("⚠️  PAPER TRADE mode — all stakes will be £0.00 (notional only)")

    return bank_row


if __name__ == "__main__":
    # Accept optional starting bank from command line
    bank = float(sys.argv[1]) if len(sys.argv) > 1 else None
    result = init_bank(bank)
    print(f"Bank state: {result}")
