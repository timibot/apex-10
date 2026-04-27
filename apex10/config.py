"""
APEX-10 Central Configuration
All system constants live here. Import from here. Never hardcode elsewhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# ── Environment mode ───────────────────────────────────────────────────────
# PAPER_TRADE: relaxed gates, zero-stake output
# PRODUCTION:  strict gates, live capital
APEX_ENV = os.getenv("APEX_ENV", "PAPER_TRADE").upper()


@dataclass(frozen=True)
class OddsConfig:
    MIN_ODDS: float = 1.25          # Raised from 1.20 — filters out ultra-cheap bets that inflate leg count
    MAX_ODDS: float = 1.49
    TARGET_PRODUCT: float = 10.0
    MIN_EDGE: float = 0.04          # 4% minimum edge, both models
    MAX_DIVERGENCE: float = 0.10    # >10% model disagreement → drop
    MAX_DRIFT: float = 0.08         # Line movement threshold → drop
    MAX_LEGS_PER_LEAGUE: int = 999  # Cap removed per user instruction
    MIN_CONSENSUS_PROB: float = 0.78  # Model must be ≥78% confident to qualify
    MAX_SAFE_LEGS: int = 8          # Cap safe ticket — don't dilute below 5% win rate


@dataclass(frozen=True)
class ModelConfig:
    BRIER_GATE_PRODUCTION: float = 0.20    # Strict gate for live capital
    BRIER_GATE_PAPER: float = 0.255        # Relaxed gate for paper trading
    BRIER_LIVE_ALERT: float = 0.24         # Live breach → halve stakes
    BRIER_VARIANCE_GATE: float = 0.02      # Paper trade stability gate
    TRAIN_YEARS: int = 5
    OPTUNA_TRIALS: int = 100
    WALK_FORWARD_TEST_YEAR: int = 5
    MIN_PAPER_TICKETS: int = 20
    ROLLING_BRIER_WINDOW: int = 15         # Live Brier monitoring window

    @property
    def BRIER_GATE(self) -> float:
        """Environment-aware Brier gate threshold."""
        return self.BRIER_GATE_PRODUCTION if APEX_ENV == "PRODUCTION" else self.BRIER_GATE_PAPER


@dataclass(frozen=True)
class StakingConfig:
    KELLY_FRACTION: float = 0.25    # Quarter-Kelly
    SIMULATED_ROI_FLOOR: float = -0.05  # Paper trade exit gate


@dataclass(frozen=True)
class LeagueConfig:
    # Live Deployment: All 5 major European leagues unlocked.
    ACTIVE_LEAGUES: tuple = ("EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1")
    ALL_LEAGUES: tuple = ("EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1")
    LEAGUE_IDS: dict = field(default_factory=lambda: {
        "EPL": 39,
        "La Liga": 140,
        "Bundesliga": 78,
        "Serie A": 135,
        "Ligue 1": 61,
    })


@dataclass(frozen=True)
class APIConfig:
    API_FOOTBALL_KEY: str = field(default_factory=lambda: os.getenv("API_FOOTBALL_KEY", ""))
    API_FOOTBALL_BASE: str = "https://v3.football.api-sports.io"
    ODDS_API_KEY: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))
    ODDS_API_BASE: str = "https://api.the-odds-api.com/v4"
    OPENWEATHER_KEY: str = field(default_factory=lambda: os.getenv("OPENWEATHER_KEY", ""))
    OPENWEATHER_BASE: str = "https://api.openweathermap.org/data/2.5"
    SUPABASE_URL: str = field(default_factory=lambda: _require_env("SUPABASE_URL"))
    SUPABASE_KEY: str = field(default_factory=lambda: _require_env("SUPABASE_KEY"))
    DISCORD_WEBHOOK: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK", ""))


def _require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Required environment variable '{key}' is not set.")
    return val


# Instantiate singletons — import these, not the dataclasses
ODDS = OddsConfig()
MODEL = ModelConfig()
STAKING = StakingConfig()
LEAGUES = LeagueConfig()
# Note: API config is instantiated lazily to avoid failing on import in test environments
# Use: from apex10.config import get_api_config
_api_config: APIConfig | None = None


def get_api_config() -> APIConfig:
    global _api_config
    if _api_config is None:
        _api_config = APIConfig()
    return _api_config
