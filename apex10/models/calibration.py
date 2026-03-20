"""
Platt scaling calibration (Logistic Regression sigmoid).
Applied after training — corrects systematic probability over/under-confidence.
Calibrator is fit on validation set, applied to test set and live predictions.
"""
from __future__ import annotations

import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)

def fit_calibrator(y_true: np.ndarray, y_prob: np.ndarray) -> LogisticRegression:
    """
    Fit Platt scaling (Logistic Regression) on validation set probabilities.
    Forces a parametric sigmoid curve, acting as a regularizer.
    """
    calibrator = LogisticRegression(C=999999, solver="lbfgs")
    calibrator.fit(y_prob.reshape(-1, 1), y_true)
    logger.info("Platt calibrator (LogisticRegression) fitted")
    return calibrator


def calibrate(calibrator: LogisticRegression, y_prob: np.ndarray) -> np.ndarray:
    """Apply calibration. Returns calibrated probabilities."""
    calibrated = calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    return np.clip(calibrated, 0.01, 0.99)


def calibration_improvement(
    y_true: np.ndarray,
    y_raw: np.ndarray,
    y_cal: np.ndarray,
) -> dict:
    """Log Brier score before and after calibration."""
    raw_brier = brier_score_loss(y_true, y_raw)
    cal_brier = brier_score_loss(y_true, y_cal)
    improved = cal_brier < raw_brier

    result = {
        "brier_raw": round(raw_brier, 4),
        "brier_calibrated": round(cal_brier, 4),
        "improvement": round(raw_brier - cal_brier, 4),
        "improved": improved,
    }

    logger.info(
        f"Calibration: {raw_brier:.4f} → {cal_brier:.4f} "
        f"({'improved' if improved else 'degraded'})"
    )
    return result
