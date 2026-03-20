-- Migration: create paper_trade_legs
-- Run once in Supabase SQL editor before engaging paper trading.
-- ----------------------------------------------------------------

CREATE TABLE IF NOT EXISTS paper_trade_legs (
    id                UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    ticket_id         TEXT        NOT NULL,
    fixture_id        TEXT        NOT NULL,
    league            TEXT        NOT NULL,
    market            TEXT        NOT NULL,          -- 'HOME_WIN' | 'DRAW' | 'AWAY_WIN'
    market_odds       NUMERIC(6,3) NOT NULL,
    p_lgbm            NUMERIC(7,4) NOT NULL,
    p_xgb             NUMERIC(7,4) NOT NULL,
    consensus_prob    NUMERIC(7,4) NOT NULL,         -- (p_lgbm + p_xgb) / 2
    actual_outcome    NUMERIC(3,1),                  -- NULL until settled; 1.0 / 0.0 / 0.5
    brier_contribution NUMERIC(8,6),                 -- (consensus_prob - actual_outcome)^2
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    settled_at        TIMESTAMPTZ                    -- NULL until settlement run
);

-- Fast lookups for the settlement loop
CREATE INDEX IF NOT EXISTS idx_ptl_fixture    ON paper_trade_legs (fixture_id);
CREATE INDEX IF NOT EXISTS idx_ptl_unsettled  ON paper_trade_legs (actual_outcome) WHERE actual_outcome IS NULL;
CREATE INDEX IF NOT EXISTS idx_ptl_ticket     ON paper_trade_legs (ticket_id);

-- ----------------------------------------------------------------
-- Convenience view: weekly calibration snapshot
-- ----------------------------------------------------------------
CREATE OR REPLACE VIEW vw_calibration_summary AS
SELECT
    DATE_TRUNC('week', created_at)      AS week,
    COUNT(*)                            AS n_legs,
    ROUND(AVG((p_lgbm       - actual_outcome)^2)::NUMERIC, 6) AS brier_lgbm,
    ROUND(AVG((p_xgb        - actual_outcome)^2)::NUMERIC, 6) AS brier_xgb,
    ROUND(AVG((consensus_prob - actual_outcome)^2)::NUMERIC, 6) AS brier_consensus,
    ROUND(AVG(actual_outcome)::NUMERIC, 4)                     AS empirical_win_rate
FROM paper_trade_legs
WHERE actual_outcome IS NOT NULL
GROUP BY 1
ORDER BY 1 DESC;
