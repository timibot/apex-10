"""
SHAP feature importance analysis.
Two purposes:
  1. Validate orthogonal subspacing — on-pitch features dominate LightGBM,
     market features dominate XGBoost.
  2. Identify near-zero contributors for potential pruning.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import shap
import xgboost as xgb

from apex10.models.features import MARKET_FEATURES, ONPITCH_FEATURES

logger = logging.getLogger(__name__)

NEAR_ZERO_THRESHOLD = 0.005  # Mean |SHAP| below this = candidate for pruning


def compute_shap_lgbm(
    model: lgb.Booster,
    X_val: np.ndarray,
) -> dict:
    """
    Compute SHAP values for LightGBM model.
    Returns dict with mean absolute SHAP per feature and pruning candidates.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # For binary classification, shap_values may be list — take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    feature_importance = {
        feat: round(float(val), 5)
        for feat, val in zip(ONPITCH_FEATURES, mean_abs_shap)
    }

    pruning_candidates = [
        feat for feat, val in feature_importance.items()
        if val < NEAR_ZERO_THRESHOLD
    ]

    logger.info(f"LightGBM SHAP — top 5: {_top_n(feature_importance, 5)}")
    if pruning_candidates:
        logger.warning(f"Pruning candidates (LightGBM): {pruning_candidates}")

    return {
        "model": "lgbm",
        "feature_importance": feature_importance,
        "pruning_candidates": pruning_candidates,
    }


def compute_shap_xgb(
    model: xgb.XGBClassifier,
    X_val: np.ndarray,
) -> dict:
    """Compute SHAP values for XGBoost model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    feature_importance = {
        feat: round(float(val), 5)
        for feat, val in zip(MARKET_FEATURES, mean_abs_shap)
    }

    pruning_candidates = [
        feat for feat, val in feature_importance.items()
        if val < NEAR_ZERO_THRESHOLD
    ]

    logger.info(f"XGBoost SHAP — top 5: {_top_n(feature_importance, 5)}")
    if pruning_candidates:
        logger.warning(f"Pruning candidates (XGBoost): {pruning_candidates}")

    return {
        "model": "xgboost",
        "feature_importance": feature_importance,
        "pruning_candidates": pruning_candidates,
    }


def validate_orthogonal_subspacing(
    lgbm_shap: dict,
    xgb_shap: dict,
    top_n: int = 5,
) -> dict:
    """
    Validate that the feature split is working as intended.
    LightGBM's top features should be on-pitch.
    XGBoost's top features should be market/context.
    Returns validation result with pass/fail and details.
    """
    lgbm_top = set(_top_n(lgbm_shap["feature_importance"], top_n).keys())
    xgb_top = set(_top_n(xgb_shap["feature_importance"], top_n).keys())

    # Check no cross-contamination (impossible with subspacing, but verify)
    lgbm_market_bleed = lgbm_top.intersection(set(MARKET_FEATURES))
    xgb_onpitch_bleed = xgb_top.intersection(set(ONPITCH_FEATURES))

    passed = len(lgbm_market_bleed) == 0 and len(xgb_onpitch_bleed) == 0

    result = {
        "passed": passed,
        "lgbm_top_features": list(lgbm_top),
        "xgb_top_features": list(xgb_top),
        "lgbm_market_bleed": list(lgbm_market_bleed),
        "xgb_onpitch_bleed": list(xgb_onpitch_bleed),
    }

    if passed:
        logger.info("✅ Orthogonal subspacing validated — no cross-bleed detected")
    else:
        logger.error(f"❌ Subspacing validation FAILED: {result}")

    return result


def _top_n(importance: dict, n: int) -> dict:
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:n])
