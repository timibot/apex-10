"""
Training entry point.
Run manually: python -m apex10.models.train
NOT triggered by GitHub Actions — training is a deliberate manual step.
"""
from __future__ import annotations

import json
import logging

from apex10.config import MODEL
from apex10.db import get_client
from apex10.models import lgbm_model, xgb_model
from apex10.models.calibration import calibrate, calibration_improvement, fit_calibrator
from apex10.models.features import get_feature_matrices, load_raw_features
from apex10.models.shap_analysis import (
    compute_shap_lgbm,
    compute_shap_xgb,
    validate_orthogonal_subspacing,
)
from apex10.models.walk_forward import check_brier_gate, get_season_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_training(league: str = "EPL", n_trials: int = MODEL.OPTUNA_TRIALS) -> dict:
    """
    Full training pipeline. Returns summary dict with all results.
    Hard-stops if either model fails the Brier gate.
    """
    logger.info("═══ APEX-10 Training Pipeline ═══")
    db = get_client()

    # ── 1. Load features ──────────────────────────────────────────────────
    logger.info("Step 1: Loading features")
    df = load_raw_features(league)
    X_onpitch, X_market, y, seasons = get_feature_matrices(df)

    # ── 2. Walk-forward splits ────────────────────────────────────────────
    logger.info("Step 2: Walk-forward splits")
    splits = get_season_splits(seasons)
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    X_op_train, X_op_val, X_op_test = (
        X_onpitch[train_idx], X_onpitch[val_idx], X_onpitch[test_idx]
    )
    X_mk_train, X_mk_val, X_mk_test = (
        X_market[train_idx], X_market[val_idx], X_market[test_idx]
    )
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # ── 3. Train LightGBM ─────────────────────────────────────────────────
    logger.info("Step 3: Training LightGBM (Optuna)")
    lgbm, lgbm_params = lgbm_model.tune_and_train(
        X_op_train, y_train, X_op_val, y_val, n_trials=n_trials
    )

    # ── 4. Train XGBoost ──────────────────────────────────────────────────
    logger.info("Step 4: Training XGBoost (Optuna)")
    xgb, xgb_params = xgb_model.tune_and_train(
        X_mk_train, y_train, X_mk_val, y_val, n_trials=n_trials
    )

    # ── 5. Calibrate on validation set ────────────────────────────────────
    logger.info("Step 5: Calibration")
    lgbm_val_probs = lgbm_model.predict_proba(lgbm, X_op_val)
    xgb_val_probs = xgb_model.predict_proba(xgb, X_mk_val)

    lgbm_cal = fit_calibrator(y_val, lgbm_val_probs)
    xgb_cal = fit_calibrator(y_val, xgb_val_probs)

    # ── 6. Test set evaluation (year 5 holdout) ───────────────────────────
    logger.info("Step 6: Test set evaluation")
    lgbm_test_raw = lgbm_model.predict_proba(lgbm, X_op_test)
    xgb_test_raw = xgb_model.predict_proba(xgb, X_mk_test)

    lgbm_test_cal = calibrate(lgbm_cal, lgbm_test_raw)
    xgb_test_cal = calibrate(xgb_cal, xgb_test_raw)

    lgbm_cal_info = calibration_improvement(y_test, lgbm_test_raw, lgbm_test_cal)
    xgb_cal_info = calibration_improvement(y_test, xgb_test_raw, xgb_test_cal)

    # ── 7. Brier gate check ───────────────────────────────────────────────
    logger.info("Step 7: Brier gate check")
    lgbm_gate = check_brier_gate(y_test, lgbm_test_cal)
    xgb_gate = check_brier_gate(y_test, xgb_test_cal)

    if not lgbm_gate["passed"] or not xgb_gate["passed"]:
        logger.error("❌ TRAINING FAILED — One or more models did not pass Brier gate.")
        logger.error("DO NOT PROCEED TO PHASE 4.")
        return {"passed": False, "lgbm": lgbm_gate, "xgb": xgb_gate}

    # ── 8. SHAP analysis ──────────────────────────────────────────────────
    logger.info("Step 8: SHAP analysis")
    lgbm_shap = compute_shap_lgbm(lgbm, X_op_val)
    xgb_shap = compute_shap_xgb(xgb, X_mk_val)
    subspace_validation = validate_orthogonal_subspacing(lgbm_shap, xgb_shap)

    # ── 9. Store params in Supabase ───────────────────────────────────────
    logger.info("Step 9: Storing model params")

    # Get current max version
    existing = db.table("model_params").select("version").order(
        "version", desc=True
    ).limit(1).execute()
    next_version = (existing.data[0]["version"] + 1) if existing.data else 1

    for model_name, params, brier in [
        ("lgbm", lgbm_params, lgbm_gate["brier_score"]),
        ("xgboost", xgb_params, xgb_gate["brier_score"]),
    ]:
        db.table("model_params").insert({
            "model_name": model_name,
            "version": next_version,
            "params": params,
            "brier_score": brier,
            "is_active": True,
        }).execute()

    # Deactivate previous versions
    db.table("model_params").update({"is_active": False}).lt(
        "version", next_version
    ).execute()

    summary = {
        "passed": True,
        "version": next_version,
        "lgbm_brier": lgbm_gate["brier_score"],
        "xgb_brier": xgb_gate["brier_score"],
        "lgbm_calibration": lgbm_cal_info,
        "xgb_calibration": xgb_cal_info,
        "shap_lgbm_pruning": lgbm_shap["pruning_candidates"],
        "shap_xgb_pruning": xgb_shap["pruning_candidates"],
        "subspacing_valid": subspace_validation["passed"],
        "splits": {k: v for k, v in splits.items() if "seasons" in k},
    }

    logger.info(f"═══ Training complete ═══\n{json.dumps(summary, indent=2)}")
    return summary


if __name__ == "__main__":
    result = run_training()
    if not result["passed"]:
        raise SystemExit(1)
