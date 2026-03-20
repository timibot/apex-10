# Purpose: Extract and serialize feature importances to detect Optuna over-indexing.
import json
import os
import numpy as np
from typing import List, Any

def dump_feature_importances(
    model: Any, 
    feature_names: List[str], 
    model_name: str, 
    run_id: str, 
    passed_gate: bool
) -> None:
    """
    Extracts tree-based feature importances and writes them to a diagnostic JSON.
    Handles CalibratedClassifierCV wrappers by extracting the base estimator.
    """
    os.makedirs("apex10/diagnostics", exist_ok=True)
    
    # If wrapped in a CalibratedClassifierCV, we must extract the base estimator
    # Note: If cv='prefit', there is one base_estimator. If cv>1, there is a list of calibrated_classifiers_.
    if hasattr(model, 'calibrated_classifiers_'):
        base_model = model.calibrated_classifiers_[0].base_estimator
    elif hasattr(model, 'base_estimator'):
        base_model = model.base_estimator
    else:
        base_model = model

    if hasattr(base_model, 'feature_importances_'):
        importances = base_model.feature_importances_
    elif hasattr(base_model, 'feature_importance'):
        importances = base_model.feature_importance()
    else:
        print(f"WARNING: Cannot extract feature importances for {model_name}.")
        return
    
    # Normalize importances to percentages for easier reading
    total_importance = np.sum(importances)
    if total_importance > 0:
        importances = (importances / total_importance) * 100

    # Sort descending
    feat_imp = sorted(
        zip(feature_names, importances), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Convert to standard dict for JSON serialization
    importance_dict = {feat: round(float(imp), 4) for feat, imp in feat_imp}
    
    status = "PASSED" if passed_gate else "FAILED"
    filepath = f"apex10/diagnostics/{model_name}_{status}_{run_id}_features.json"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "status": status,
            "top_10_features": dict(list(importance_dict.items())[:10]),
            "all_features": importance_dict
        }, f, indent=4)
        
    print(f"INFO: Feature importances dumped to {filepath}")
