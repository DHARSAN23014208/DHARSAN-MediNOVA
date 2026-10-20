"""
MediNova Thyroid Dysfunction Training & Evaluation Pipeline
============================================================
Evidence-based training pipeline using the authentic UCI Thyroid Disease
(Garavan Institute / J. Ross Quinlan) benchmark dataset.

Features (8 Clinical Analytes & Demographics):
1. age (years)
2. sex (0 = Female, 1 = Male)
3. on_thyroxine (0 = No, 1 = Yes)
4. TSH (Thyroid-Stimulating Hormone, mIU/L) - Canonical marker for primary/subclinical hypothyroidism
5. T3 (Triiodothyronine, ng/mL or nmol/L)
6. TT4 (Total Thyroxine, ug/dL)
7. T4U (Thyroxine Uptake ratio)
8. FTI (Free Thyroxine Index)

Target Labels:
0 = Euthyroid (Normal Thyroid Function)
1 = Hypothyroidism (Primary, Compensated, or Secondary Thyroid Dysfunction)

Methodology:
- Separate training set (allhypo.data: 2,800 patients)
- Completely untouched independent test set (allhypo.test: 972 patients)
- Calibrated probability via CalibratedClassifierCV
- Full clinical metrics: Sensitivity, Specificity, Precision, NPV, F1, ROC-AUC, Brier Score
"""

import os
import hashlib
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "datasets", "thyroid")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

ATTRIBUTE_COLS = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH_measured', 'TSH', 'T3_measured', 'T3', 'TT4_measured', 'TT4',
    'T4U_measured', 'T4U', 'FTI_measured', 'FTI', 'TBG_measured', 'TBG',
    'referral_source', 'target'
]

SELECTED_FEATURES = ['age', 'sex', 'on_thyroxine', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']


def load_and_preprocess_split(filepath: str) -> tuple[pd.DataFrame, pd.Series]:
    """Loads and standardizes UCI thyroid data into clinical feature schema."""
    df = pd.read_csv(filepath, names=ATTRIBUTE_COLS, na_values=['?'], skipinitialspace=True)
    
    X = df[SELECTED_FEATURES].copy()
    
    # Clinical sanity bounds: age > 110 is entry typo; clip to NaN for median imputation
    X['age'] = X['age'].apply(lambda a: np.nan if (pd.notnull(a) and (a < 1 or a > 115)) else a)
    X['sex'] = X['sex'].map({'M': 1, 'F': 0}).fillna(0).astype(float)
    X['on_thyroxine'] = X['on_thyroxine'].map({'t': 1, 'f': 0}).fillna(0).astype(float)
    
    for num_col in ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']:
        X[num_col] = pd.to_numeric(X[num_col], errors='coerce')
        
    # Standardize target
    raw_target = df['target'].astype(str).apply(lambda s: s.split('.')[0].strip().lower())
    y = raw_target.apply(lambda t: 0 if t == 'negative' else 1)
    
    return X, y


def train_and_evaluate_thyroid_model() -> dict:
    train_path = os.path.join(DATA_DIR, "allhypo.data")
    test_path = os.path.join(DATA_DIR, "allhypo.test")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Thyroid dataset missing at {DATA_DIR}")
        
    print(f"Loading training data from {train_path}...")
    X_train, y_train = load_and_preprocess_split(train_path)
    print(f"Loading independent test data from {test_path}...")
    X_test, y_test = load_and_preprocess_split(test_path)
    
    print(f"Train cohort size: {len(X_train)} (Positive: {y_train.sum()} / {len(y_train)})")
    print(f"Test cohort size: {len(X_test)} (Positive: {y_test.sum()} / {len(y_test)})")
    
    # Build robust pipeline with median imputation and balanced Random Forest
    imputer = SimpleImputer(strategy="median")
    base_rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )
    
    pipeline = Pipeline([
        ("imputer", imputer),
        ("classifier", base_rf)
    ])
    
    # Probability calibration using CalibratedClassifierCV
    calibrated_model = CalibratedClassifierCV(
        estimator=pipeline,
        method="sigmoid",
        cv=5
    )
    
    print("Fitting calibrated thyroid classifier...")
    calibrated_model.fit(X_train, y_train)
    
    # Also fit base pipeline for feature importance extraction
    pipeline.fit(X_train, y_train)
    
    # Evaluate on untouched test set
    y_pred = calibrated_model.predict(X_test)
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    # Feature importances from underlying fitted random forest
    rf_estimator = pipeline.named_steps["classifier"]
    feature_importances = dict(zip(SELECTED_FEATURES, [round(float(w), 4) for w in rf_estimator.feature_importances_]))
    
    # Medians for reference/fallback imputation
    imputed_medians = dict(zip(SELECTED_FEATURES, [round(float(m), 4) for m in pipeline.named_steps["imputer"].statistics_]))
    
    model_artifact = {
        "model": calibrated_model,
        "base_pipeline": pipeline,
        "feature_names": SELECTED_FEATURES,
        "feature_importances": feature_importances,
        "feature_medians": imputed_medians,
        "version": "1.0.0",
        "modality": "laboratory",
        "condition": "thyroid_dysfunction",
        "target_mapping": {0: "Euthyroid (Normal Thyroid Function)", 1: "Elevated Thyroid Dysfunction (Hypothyroidism)"}
    }
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_model_path = os.path.join(MODELS_DIR, "thyroid_model.pkl")
    joblib.dump(model_artifact, out_model_path)
    
    with open(out_model_path, "rb") as f_hash:
        artifact_hash = hashlib.sha256(f_hash.read()).hexdigest()
        
    metrics = {
        "model_id": "medinova-thyroid-v1",
        "version": "1.0.0",
        "artifact_path": "models/thyroid_model.pkl",
        "artifact_sha256": artifact_hash,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_metrics": {
            "accuracy": round(float(acc), 4),
            "sensitivity": round(float(sensitivity), 4),
            "specificity": round(float(specificity), 4),
            "precision": round(float(precision), 4),
            "npv": round(float(npv), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "brier_score": round(float(brier), 4),
            "confusion_matrix": {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn)
            }
        },
        "feature_importances": feature_importances,
        "feature_medians": imputed_medians
    }
    
    print("\n--- INDEPENDENT TEST EVALUATION RESULTS ---")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"NPV:         {npv:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"SHA-256:     {artifact_hash}")
    
    return metrics


if __name__ == "__main__":
    train_and_evaluate_thyroid_model()
