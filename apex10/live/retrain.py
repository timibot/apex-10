"""
Annual retrain orchestrator.
Runs every January via GitHub Actions.
Retrains on most recent 5 years of data, validates walk-forward,
runs SHAP drift check. Deploys only if new model outperforms current.
Never touches live system if new model fails Brier gate.
"""
from __future__ import annotations

import json
import logging
from datetime import date

from apex10.config import MODEL
from apex10.db import get_client
from apex10.live.notify import retrain_complete
from apex10.models.train import run_training

logger = logging.getLogger(__name__)


def get_active_model_brier(db) -> float | None:
    """Fetch the Brier score of the currently active model."""
    result = (
        db.table("model_params")
        .select("brier_score, version")
        .eq("is_active", True)
        .order("version", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    return float(result.data[0]["brier_score"])


def run_annual_retrain(
    n_trials: int = MODEL.OPTUNA_TRIALS,
) -> dict:
    """
    Full annual retrain pipeline.

    Steps:
      1. Train new model on latest 5 years of data
      2. Compare new model Brier vs current active model
      3. Deploy if better — keep current if not
      4. Log result and notify Discord
    """
    logger.info(f"═══ Annual Retrain — {date.today().year} ═══")
    db = get_client()

    # ── 1. Get current model performance baseline ─────────────────────────
    current_brier = get_active_model_brier(db)
    logger.info(f"Current active model Brier: {current_brier}")

    # ── 2. Run full training pipeline ─────────────────────────────────────
    training_result = run_training(league="EPL", n_trials=n_trials)

    if not training_result.get("passed"):
        logger.error(
            "New model failed Brier gate — keeping current model"
        )
        retrain_complete(
            version=training_result.get("version", 0),
            lgbm_brier=training_result.get("lgbm_brier", 1.0),
            xgb_brier=training_result.get("xgb_brier", 1.0),
            deployed=False,
        )
        return {
            "deployed": False,
            "reason": "New model failed Brier gate",
            "training_result": training_result,
        }

    new_lgbm_brier = training_result["lgbm_brier"]
    new_xgb_brier = training_result["xgb_brier"]
    new_version = training_result["version"]
    new_brier_benchmark = max(new_lgbm_brier, new_xgb_brier)

    # ── 3. Compare and deploy decision ────────────────────────────────────
    if current_brier is None:
        deploy = True
        reason = "No active model found — deploying new model"
    elif new_brier_benchmark < current_brier:
        deploy = True
        reason = (
            f"New model ({new_brier_benchmark:.4f}) outperforms "
            f"current ({current_brier:.4f})"
        )
    else:
        deploy = False
        reason = (
            f"New model ({new_brier_benchmark:.4f}) does not outperform "
            f"current ({current_brier:.4f}) — keeping current"
        )

    logger.info(f"Deploy decision: {deploy} — {reason}")

    # ── 4. If not deploying, deactivate the newly trained version ─────────
    if not deploy:
        db.table("model_params").update(
            {"is_active": False}
        ).eq("version", new_version).execute()

    # ── 5. Notify ─────────────────────────────────────────────────────────
    retrain_complete(
        version=new_version,
        lgbm_brier=new_lgbm_brier,
        xgb_brier=new_xgb_brier,
        deployed=deploy,
    )

    summary = {
        "deployed": deploy,
        "reason": reason,
        "new_version": new_version,
        "new_lgbm_brier": new_lgbm_brier,
        "new_xgb_brier": new_xgb_brier,
        "previous_brier": current_brier,
        "retrain_date": str(date.today()),
        "shap_pruning": {
            "lgbm": training_result.get("shap_lgbm_pruning", []),
            "xgb": training_result.get("shap_xgb_pruning", []),
        },
        "subspacing_valid": training_result.get("subspacing_valid"),
    }

    logger.info(f"Retrain complete:\n{json.dumps(summary, indent=2)}")
    return summary


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO)
    run_annual_retrain()
