"""
MediNova Tabular Model Benchmarking Suite
=========================================
Implements rigorous, repeatable comparative benchmarking for tabular clinical screening.
Evaluates candidate estimators on identical splits using clinical classification metrics:
- Sensitivity / Recall (Detection of true positive pathology)
- Specificity (Correct identification of healthy/euthyroid patients)
- Precision / PPV (Positive Predictive Value)
- Negative Predictive Value (NPV)
- F1-Score
- ROC-AUC
- Brier Score (Calibration error)

Candidate Estimators:
1. Logistic Regression (Interpretable linear baseline with L2 regularization)
2. Random Forest (Standard tree ensemble)
3. XGBoost (Gradient-boosted decision trees)
4. HistGradientBoosting / LightGBM (Histogram-based gradient boosting)
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix
)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def calculate_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive clinical screening evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
        
    brier = brier_score_loss(y_true, y_prob)
    
    return {
        "accuracy": round(float(acc), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "precision": round(float(precision), 4),
        "npv": round(float(npv), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "brier_score": round(float(brier), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn)
    }


def benchmark_tabular_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """Runs stratified 5-fold cross-validation across all candidate architectures."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    models = {
        "LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced"),
        "RandomForest": lambda: RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, class_weight="balanced"),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(random_state=random_state, class_weight="balanced")
    }
    
    if XGB_AVAILABLE:
        pos_weight = float((len(y) - y.sum()) / max(1, y.sum()))
        models["XGBoost"] = lambda: xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=pos_weight,
            random_state=random_state,
            eval_metric="logloss"
        )
        
    results = {}
    
    for name, model_factory in models.items():
        fold_metrics = []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            clf = model_factory()
            clf.fit(X_train, y_train)
            
            preds = clf.predict(X_val)
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_val)[:, 1]
            else:
                probs = preds.astype(float)
                
            metrics = calculate_clinical_metrics(y_val.to_numpy(), preds, probs)
            fold_metrics.append(metrics)
            
        summary = {}
        metric_keys = ["accuracy", "sensitivity", "specificity", "precision", "npv", "f1_score", "roc_auc", "brier_score"]
        for k in metric_keys:
            vals = [m[k] for m in fold_metrics]
            summary[f"{k}_mean"] = round(float(np.mean(vals)), 4)
            summary[f"{k}_std"] = round(float(np.std(vals)), 4)
            
        results[name] = summary
        
    return {
        "dataset": dataset_name,
        "sample_count": len(X),
        "positive_rate": round(float(y.mean()), 4),
        "benchmark_results": results
    }


if __name__ == "__main__":
    print("Tabular benchmarking utility ready.")
