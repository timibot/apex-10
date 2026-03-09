"""
Walk-forward validation protocol (unchanged from spec):
  Train on seasons 1–3 → validate on season 4 → tune → test on season 5.
Season years are relative to the 5-year window, not calendar years.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


def get_season_splits(seasons: np.ndarray) -> dict:
    """
    Given an array of season years, return train/val/test index splits.
    Seasons are sorted chronologically. Most recent = test.
    """
    unique_seasons = sorted(np.unique(seasons))

    if len(unique_seasons) < 3:
        raise ValueError(f"Need at least 3 seasons, got {len(unique_seasons)}")

    # Walk-forward splits based on number of seasons available
    if len(unique_seasons) == 3:
        # 3 seasons: train on 1st, validate on 2nd, test on 3rd
        train_seasons = [unique_seasons[0]]
        val_seasons = [unique_seasons[1]]
        test_seasons = [unique_seasons[2]]
    elif len(unique_seasons) == 4:
        # 4 seasons: train on 1-2, validate on 3rd, test on 4th
        train_seasons = unique_seasons[:2]
        val_seasons = [unique_seasons[2]]
        test_seasons = [unique_seasons[3]]
    else:
        # 5+ seasons: train on first N-2, validate on N-1, test on N
        train_seasons = unique_seasons[:-2]
        val_seasons = [unique_seasons[-2]]
        test_seasons = [unique_seasons[-1]]

    train_idx = np.where(np.isin(seasons, train_seasons))[0]
    val_idx = np.where(np.isin(seasons, val_seasons))[0]
    test_idx = np.where(np.isin(seasons, test_seasons))[0]

    logger.info(
        f"Split: train={train_seasons}, "
        f"val={val_seasons}, "
        f"test={test_seasons}"
    )
    logger.info(
        f"Sizes: train={len(train_idx)}, "
        f"val={len(val_idx)}, "
        f"test={len(test_idx)}"
    )

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "train_seasons": train_seasons,
        "val_seasons": val_seasons,
        "test_seasons": test_seasons,
    }


def check_brier_gate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "model",
) -> dict:
    """
    Environment-aware Brier score validation.

    PAPER_TRADE: gate = 0.255, warns if > 0.20
    PRODUCTION:  gate = 0.20, hard reject
    """
    from apex10.config import MODEL, APEX_ENV

    gate = MODEL.BRIER_GATE  # Environment-aware via property
    score = brier_score_loss(y_true, y_prob)
    passed = score < gate

    result = {
        "brier_score": round(score, 4),
        "gate": gate,
        "environment": APEX_ENV,
        "passed": passed,
        "gap": round(gate - score, 4),
        "production_safe": score < 0.20,
    }

    if passed:
        logger.info(f"✅ [{model_name}] Brier gate PASSED: {score:.4f} < {gate} ({APEX_ENV})")
        if score >= 0.20 and APEX_ENV == "PAPER_TRADE":
            logger.warning(
                f"⚠️  [{model_name}] Passed paper gate but UNSAFE for live capital. "
                f"Stake locked to 0.00 until Brier < 0.20."
            )
    else:
        logger.error(
            f"❌ [{model_name}] Brier gate FAILED: {score:.4f} >= {gate} ({APEX_ENV}). "
            f"DO NOT DEPLOY."
        )

    return result

