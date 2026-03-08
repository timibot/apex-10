"""
XGBoost classifier — market/context feature set only.
Identical Optuna interface to lgbm_model.py for clean orchestration.
"""
from __future__ import annotations

import logging

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import brier_score_loss

from apex10.config import MODEL

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def _objective(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
    }

    model = xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False,
    )

    preds = model.predict_proba(X_val)[:, 1]
    return brier_score_loss(y_val, preds)


def tune_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = MODEL.OPTUNA_TRIALS,
) -> tuple[xgb.XGBClassifier, dict]:
    """
    Run Optuna search, retrain on train+val with best params.
    Returns (trained_model, best_params).
    """
    logger.info(f"Starting Optuna search: {n_trials} trials (XGBoost)")

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_params.update({
        "objective": "binary:logistic",
        "random_state": 42,
        "use_label_encoder": False,
    })

    logger.info(f"Best params: {best_params}")
    logger.info(f"Best val Brier: {study.best_value:.4f}")

    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_full, y_full, verbose=False)

    return final_model, best_params


def predict_proba(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    """Return home win probabilities clipped to [0.01, 0.99]."""
    preds = model.predict_proba(X)[:, 1]
    return np.clip(preds, 0.01, 0.99)
