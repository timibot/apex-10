"""
Dual line movement classifier.
Classifies whether odds drift is sharp professional money or public noise.
Trained as a binary classifier: 1 = sharp, 0 = noise.

Features:
  - drift_magnitude: abs(current - opening)
  - drift_velocity: magnitude / hours_since_open
  - is_pinnacle_adjacent: 1 if line moved on Pinnacle first
  - day_of_week: 0=Mon ... 6=Sun (early week = more likely sharp)
  - drift_direction: 1 = lengthening (fading), -1 = shortening
  - market_liquidity_proxy: implied_volume rank (higher = more liquid)

Labels: historically, drift > 0.08 that preceded a favourite losing
is labelled 1 (sharp). Otherwise 0.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)

LINE_MOVEMENT_FEATURES = [
    "drift_magnitude",
    "drift_velocity",
    "is_pinnacle_adjacent",
    "day_of_week",
    "drift_direction",
    "market_liquidity_proxy",
]


def build_training_data(
    historical_odds: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix and labels from historical odds movements.
    Label = 1 if the favourite lost (sharp money was right), else 0.
    """
    X, y = [], []

    for record in historical_odds:
        opening = record.get("opening_odds_home", 0)
        closing = record.get("odds_home", 0)
        home_goals = record.get("home_goals", 0)
        away_goals = record.get("away_goals", 0)
        hours_open = float(record.get("hours_since_open", 48))

        if opening <= 0 or closing <= 0:
            continue

        drift = closing - opening
        drift_mag = abs(drift)
        velocity = drift_mag / max(hours_open, 1.0)
        direction = 1.0 if drift > 0 else -1.0
        dow = int(record.get("day_of_week", 3))
        pinnacle_adj = int(record.get("is_pinnacle_adjacent", 0))
        liquidity = float(
            record.get("market_liquidity_proxy", 0.5)
        )

        features = [
            drift_mag,
            velocity,
            pinnacle_adj,
            dow,
            direction,
            liquidity,
        ]
        X.append(features)

        # Label: was the drift sharp?
        home_was_favourite = opening < 2.0
        home_lost = home_goals < away_goals
        label = (
            1
            if (
                home_was_favourite
                and home_lost
                and drift_mag > 0.05
            )
            else 0
        )
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def train_lgbm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> lgb.Booster:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
    }
    train_d = lgb.Dataset(X_train, label=y_train)
    val_d = lgb.Dataset(X_val, label=y_val, reference=train_d)
    model = lgb.train(
        params,
        train_d,
        num_boost_round=200,
        valid_sets=[val_d],
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    return model


def train_xgb_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


SHARP_THRESHOLD = 0.55


def classify_movement(
    lgbm_model,
    xgb_model,
    drift_magnitude: float,
    drift_velocity: float,
    is_pinnacle_adjacent: int,
    day_of_week: int,
    drift_direction: float,
    market_liquidity_proxy: float = 0.5,
) -> dict:
    """
    Classify a single line movement as sharp or noise.
    Returns: {both_sharp, one_sharp, lgbm_sharp_prob, xgb_sharp_prob}
    """
    features = np.array(
        [
            [
                drift_magnitude,
                drift_velocity,
                is_pinnacle_adjacent,
                day_of_week,
                drift_direction,
                market_liquidity_proxy,
            ]
        ],
        dtype=np.float32,
    )

    lgbm_prob = float(lgbm_model.predict(features)[0])
    xgb_prob = float(xgb_model.predict_proba(features)[0, 1])

    lgbm_sharp = lgbm_prob >= SHARP_THRESHOLD
    xgb_sharp = xgb_prob >= SHARP_THRESHOLD

    return {
        "both_sharp": lgbm_sharp and xgb_sharp,
        "one_sharp": lgbm_sharp != xgb_sharp,
        "neither_sharp": not lgbm_sharp and not xgb_sharp,
        "lgbm_sharp_prob": round(lgbm_prob, 4),
        "xgb_sharp_prob": round(xgb_prob, 4),
    }
