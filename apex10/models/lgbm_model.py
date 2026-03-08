"""
LightGBM classifier — on-pitch feature set only.
Optuna finds optimal hyperparameters on year-4 validation set.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import brier_score_loss

from apex10.config import MODEL

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def _objective(trial: optuna.Trial, X_train, y_train, X_val, y_val) -> float:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )

    preds = model.predict(X_val)
    return brier_score_loss(y_val, preds)


def tune_and_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = MODEL.OPTUNA_TRIALS,
) -> tuple[lgb.Booster, dict]:
    """
    Run Optuna search on validation set, then retrain on train+val
    with best params. Returns (trained_model, best_params).
    """
    logger.info(f"Starting Optuna search: {n_trials} trials (LightGBM)")

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: _objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_params.update({
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
    })

    logger.info(f"Best params: {best_params}")
    logger.info(f"Best val Brier: {study.best_value:.4f}")

    # Retrain on train + val combined with best params
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])

    full_data = lgb.Dataset(X_full, label=y_full)
    final_model = lgb.train(
        best_params,
        full_data,
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(-1)],
    )

    return final_model, best_params


def predict_proba(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """Return home win probabilities clipped to [0.01, 0.99]."""
    preds = model.predict(X)
    return np.clip(preds, 0.01, 0.99)
