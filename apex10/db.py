"""
Supabase client singleton.
All DB access goes through this module — never instantiate the client elsewhere.
"""
from __future__ import annotations

from functools import lru_cache

from supabase import Client, create_client

from apex10.config import get_api_config


@lru_cache(maxsize=1)
def get_client() -> Client:
    """Returns a cached Supabase client. Thread-safe via lru_cache."""
    cfg = get_api_config()
    return create_client(cfg.SUPABASE_URL, cfg.SUPABASE_KEY)


def health_check() -> bool:
    """Verify DB connection is live. Used in CI smoke test."""
    try:
        client = get_client()
        # Ping with a lightweight query
        client.table("health_check").select("id").limit(1).execute()
        return True
    except Exception:
        return False
