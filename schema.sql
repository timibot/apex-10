-- ═══════════════════════════════════════════════
-- APEX-10 · Complete Database Schema · V1.0
-- Run once in Supabase SQL Editor
-- ═══════════════════════════════════════════════

-- Health check table (CI ping target)
CREATE TABLE IF NOT EXISTS health_check (
  id SERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
INSERT INTO health_check DEFAULT VALUES;

-- Raw match results (populated by cache.py)
CREATE TABLE IF NOT EXISTS matches (
  id BIGSERIAL PRIMARY KEY,
  api_match_id INTEGER UNIQUE NOT NULL,
  league VARCHAR(50) NOT NULL,
  season INTEGER NOT NULL,
  match_date DATE NOT NULL,
  home_team VARCHAR(100) NOT NULL,
  away_team VARCHAR(100) NOT NULL,
  home_goals INTEGER,
  away_goals INTEGER,
  status VARCHAR(20) DEFAULT 'scheduled',   -- scheduled | finished | cancelled
  raw_json JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_matches_league_season ON matches(league, season);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date);

-- Per-league Dixon-Coles rho values (computed in Phase 2)
CREATE TABLE IF NOT EXISTS league_rho (
  id BIGSERIAL PRIMARY KEY,
  league VARCHAR(50) UNIQUE NOT NULL,
  rho FLOAT NOT NULL,
  computed_at TIMESTAMPTZ DEFAULT NOW(),
  sample_size INTEGER
);

-- Historical odds (loaded from football-data.co.uk CSVs in Phase 2)
CREATE TABLE IF NOT EXISTS historical_odds (
  id BIGSERIAL PRIMARY KEY,
  api_match_id INTEGER REFERENCES matches(api_match_id),
  bookmaker VARCHAR(50),
  market VARCHAR(30),        -- 1X2 | over_2.5 | btts | etc.
  odds_home FLOAT,
  odds_draw FLOAT,
  odds_away FLOAT,
  opening_odds_home FLOAT,
  opening_odds_away FLOAT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trained model parameters (stored after Phase 3)
CREATE TABLE IF NOT EXISTS model_params (
  id BIGSERIAL PRIMARY KEY,
  model_name VARCHAR(50) NOT NULL,   -- lgbm | xgboost
  version INTEGER NOT NULL,
  params JSONB NOT NULL,
  brier_score FLOAT,
  trained_at TIMESTAMPTZ DEFAULT NOW(),
  is_active BOOLEAN DEFAULT FALSE
);

-- Weekly ticket log
CREATE TABLE IF NOT EXISTS tickets (
  id BIGSERIAL PRIMARY KEY,
  week_start DATE NOT NULL UNIQUE,
  legs JSONB NOT NULL,               -- Array of leg objects
  combined_odds FLOAT NOT NULL,
  stake FLOAT,
  status VARCHAR(20) DEFAULT 'pending',  -- pending | won | lost | void
  result_logged_at TIMESTAMPTZ,
  brier_score FLOAT,
  simulated_win_rate FLOAT,
  monte_carlo_ci_low FLOAT,
  monte_carlo_ci_high FLOAT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_tickets_week ON tickets(week_start);

-- Per-leg results (populated by score.py)
CREATE TABLE IF NOT EXISTS ticket_legs (
  id BIGSERIAL PRIMARY KEY,
  ticket_id BIGINT REFERENCES tickets(id),
  api_match_id INTEGER,
  bet_type VARCHAR(30),
  odds FLOAT,
  model_probability FLOAT,
  implied_probability FLOAT,
  edge FLOAT,
  result VARCHAR(10),    -- win | loss | void
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Live bank state (auto-updated by score.py)
CREATE TABLE IF NOT EXISTS bank_state (
  id INTEGER PRIMARY KEY DEFAULT 1,   -- Single-row table
  current_bank FLOAT NOT NULL,
  initial_bank FLOAT NOT NULL,
  extracted_profit FLOAT DEFAULT 0.0,
  total_wins INTEGER DEFAULT 0,
  total_losses INTEGER DEFAULT 0,
  stake_multiplier FLOAT DEFAULT 1.0,
  last_updated TIMESTAMPTZ DEFAULT NOW(),
  CONSTRAINT single_row CHECK (id = 1)
);

-- Brier score rolling tracker
CREATE TABLE IF NOT EXISTS brier_log (
  id BIGSERIAL PRIMARY KEY,
  ticket_id BIGINT REFERENCES tickets(id),
  brier_score FLOAT NOT NULL,
  rolling_brier_15 FLOAT,
  logged_at TIMESTAMPTZ DEFAULT NOW()
);

-- Understat xG data (populated by understat.py)
CREATE TABLE IF NOT EXISTS match_xg (
  id BIGSERIAL PRIMARY KEY,
  understat_id INTEGER UNIQUE NOT NULL,
  match_date DATE NOT NULL,
  home_team VARCHAR(100) NOT NULL,
  away_team VARCHAR(100) NOT NULL,
  home_xg FLOAT NOT NULL,
  away_xg FLOAT NOT NULL,
  home_goals INTEGER,
  away_goals INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_xg_date ON match_xg(match_date);

-- ClubElo ratings (populated by clubelo.py)
CREATE TABLE IF NOT EXISTS club_elo (
  id BIGSERIAL PRIMARY KEY,
  team_slug VARCHAR(100) NOT NULL,
  rating_date DATE NOT NULL,
  elo_rating FLOAT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(team_slug, rating_date)
);
CREATE INDEX IF NOT EXISTS idx_elo_team_date ON club_elo(team_slug, rating_date);

-- ── Phase 4: Upcoming fixtures for ticket generation ──────────────
CREATE TABLE IF NOT EXISTS upcoming_fixtures (
  id BIGSERIAL PRIMARY KEY,
  api_match_id INTEGER UNIQUE NOT NULL,
  league VARCHAR(50) NOT NULL,
  match_date DATE NOT NULL,
  home_team VARCHAR(100) NOT NULL,
  away_team VARCHAR(100) NOT NULL,
  best_bet_type VARCHAR(30),
  best_bet_odds FLOAT,
  opening_odds FLOAT,
  lgbm_prob FLOAT,
  xgb_prob FLOAT,
  key_player_absent_home INTEGER DEFAULT 0,
  key_player_absent_away INTEGER DEFAULT 0,
  features_complete BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Gap Patches: Cache run logger ─────────────────────────────────
CREATE TABLE IF NOT EXISTS cache_log (
  id BIGSERIAL PRIMARY KEY,
  run_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  fixtures_written INTEGER DEFAULT 0,
  xg_written INTEGER DEFAULT 0,
  odds_written INTEGER DEFAULT 0,
  ppda_written INTEGER DEFAULT 0,
  api_requests_used JSONB DEFAULT '{}',
  sources_failed JSONB DEFAULT '[]',
  runtime_seconds FLOAT,
  success BOOLEAN DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS idx_cache_log_timestamp ON cache_log(run_timestamp);

-- ── Gap Patches: Team PPDA from FBref ─────────────────────────────
CREATE TABLE IF NOT EXISTS team_ppda (
  id BIGSERIAL PRIMARY KEY,
  team VARCHAR(100) NOT NULL,
  league VARCHAR(50) NOT NULL,
  season INTEGER NOT NULL,
  ppda FLOAT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(team, season, league)
);
CREATE INDEX IF NOT EXISTS idx_team_ppda_lookup ON team_ppda(team, season);

-- ── Gap Patches: ALTER existing bank_state if it already exists ───
-- Run these only if bank_state was created BEFORE this schema update:
-- ALTER TABLE bank_state ADD COLUMN IF NOT EXISTS extracted_profit FLOAT DEFAULT 0.0;
-- ALTER TABLE bank_state ADD COLUMN IF NOT EXISTS total_wins INTEGER DEFAULT 0;
-- ALTER TABLE bank_state ADD COLUMN IF NOT EXISTS total_losses INTEGER DEFAULT 0;
-- ALTER TABLE bank_state ADD COLUMN IF NOT EXISTS stake_multiplier FLOAT DEFAULT 1.0;
