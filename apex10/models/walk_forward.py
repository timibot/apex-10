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

    # Walk-forward: train=first 3, val=4th, test=5th (most recent)
    train_seasons = unique_seasons[:3]
    val_seasons = [unique_seasons[3]] if len(unique_seasons) >= 4 else []
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


def check_brier_gate(y_true: np.ndarray, y_prob: np.ndarray, gate: float = 0.20) -> dict:
    """
    Evaluate Brier score against deployment gate.
    Returns dict with score, passed bool, and gap to gate.
    """
    score = brier_score_loss(y_true, y_prob)
    passed = score < gate

    result = {
        "brier_score": round(score, 4),
        "gate": gate,
        "passed": passed,
        "gap": round(gate - score, 4),
    }

    if passed:
        logger.info(f"✅ Brier gate PASSED: {score:.4f} < {gate}")
    else:
        logger.error(f"❌ Brier gate FAILED: {score:.4f} >= {gate}. DO NOT DEPLOY.")

    return result
