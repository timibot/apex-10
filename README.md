# APEX-10

Edge-detection engine for algorithmic football accumulator betting. This system autonomously scrapes live tactical and market data, executes a dual-model machine learning prediction pipeline against a Supabase backend, and settles paper trades via mathematical calibration.

---

## 🧬 Core Architecture: Orthogonal Subspacing

APEX-10 operates via a strict dual-matrix machine learning pipeline designed to isolate betting market realities from on-pitch tactical geometry.

1. **LightGBM (The Tactical Engine)**
   * **Data Vector:** 23 On-pitch variables (xG, Goal Difference, PPDA, Injury impacts, Form, Travel). 
   * **Purpose:** Evaluates matches strictly on their fundamental physical and tactical realities, completely ignoring the consensus betting market.

2. **XGBoost (The Market Engine)**
   * **Data Vector:** 16 Market/Context variables (Current Odds, Line Movement, Fixture Congestion, Manager Days, ClubElo Ratings).
   * **Purpose:** Evaluates matches via the wisdom of the crowd, adjusting for psychological traps and scheduling fatigue.

Both models evaluate the same matches, but their feature sets are perfectly isolated. **`elo_diff` acts as the single Shared Anchor** bridging the two vectors. 
A ticket leg is only verified when both models converge to pass the 0.255 Paper Trade Brier Gate threshold.

---

## 🚀 Live Deployment & Blackout Rules

APEX-10 relies on a strict execution schedule to maintain its caching loops and model scoring. 
**Crucially, when deploying APEX-10 on a Linux VPS, standard `cron` does not preserve your execution environment variables.**

### The VPS Execution Wrapper (`run_apex.sh`)
You must route your cron triggers through a native bash wrapper to ensure `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` and `API_FOOTBALL_KEY` are successfully loaded into the bare-metal shell.

```bash
#!/bin/bash
# Purpose: Bulletproof execution wrapper for APEX-10 Cron Jobs

# 1. Load Environment Variables
export $(grep -v '^#' /path/to/apex10/.env | xargs)

# 2. Navigate to working directory
cd /path/to/apex10

# 3. Execute using the explicit virtual environment Python binary
/path/to/apex10/.venv/bin/python "$@"
```

### The Cron Schedule
Add these triggers to your `crontab -e`:

```text
# Make sure the logging directory exists first: mkdir -p /var/log/apex10

# 1. TUESDAY 08:00 UTC | The Global Fetch
# Pulls match results, resolves pending paper trades, recalculates Brier calibration, and updates the VW View
0 8 * * 2 /bin/bash /path/to/apex10/run_apex.sh -m apex10.live.settlement >> /var/log/apex10/settlement.log 2>&1

# 2. THURSDAY 14:00 UTC | The Cache Builder
# Fetches all incoming fixtures, market odds, and Understat xG metrics
0 14 * * 4 /bin/bash /path/to/apex10/run_apex.sh -m apex10.cache.cache >> /var/log/apex10/cache.log 2>&1

# 3. FRIDAY 12:00 UTC | The Ticket Generator
# Executes live inference and generates the S-Tier accumulator
0 12 * * 5 /bin/bash /path/to/apex10/run_apex.sh -m apex10.live.inference >> /var/log/apex10/inference.log 2>&1
```

---

## 🛠 Active Workflows & Operations

* **`apex10.live.retrain`**: Triggered strictly *manually* once a year to sweep parameters via Optuna and validate internal gate constraints using Platt Scaling (`LogisticRegression`).
* **`apex10.ticket.ticket`**: Run locally on Friday afternoons to pull the processed 1X2 outcomes, apply human logic / asymmetric vetoes, and stitch together the valid betting accumulator.

### Database Initialization
Before first execution, you **must** fire the tracking DB script inside your Supabase SQL editor:
`002_paper_trade_legs.sql`

If the SQL schema is absent, `settlement.py` will catastrophically fail with a `PGRST205` 404 cache error.
