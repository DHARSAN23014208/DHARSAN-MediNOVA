"""
MediNova Unified Clinical Diagnostic Prediction & Explainability Engine
=======================================================================
Shared prediction logic ensuring 100% mathematical and clinical parity
between manual web form submissions and Smart Lab Report Scanner predictions.
Integrates:
- Scikit-learn Random Forest model inference
- Confidence calculation
- Real trained feature importance attribution
- Clinical reference biomarker gauges
- Silent Google Sheets audit logging
- Cryptographic verification registration
"""

import os
import re
import uuid
import datetime
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Tuple, Optional

from utils.google_sheets_logger import log_patient_prediction
from utils.report_verifier import register_report

logger = logging.getLogger("medinova.predictor")

# ============================================================================
# CLINICAL REFERENCE RANGES FOR BIOMARKER GAUGES
# ============================================================================

CLINICAL_REFERENCE_RANGES = {
    # Glycemic
    "HbA1c_level": {"name": "HbA1c Level", "unit": "%", "optimal": (4.0, 5.6), "borderline": (5.7, 6.4), "max_disp": 14.0},
    "blood_glucose_level": {"name": "Blood Glucose", "unit": "mg/dL", "optimal": (70.0, 99.0), "borderline": (100.0, 125.0), "max_disp": 300.0},
    "bmi": {"name": "Body Mass Index", "unit": "kg/m²", "optimal": (18.5, 24.9), "borderline": (25.0, 29.9), "max_disp": 50.0},

    # Cardiovascular
    "bp": {"name": "Resting Blood Pressure", "unit": "mm Hg", "optimal": (90.0, 119.0), "borderline": (120.0, 139.0), "max_disp": 200.0},
    "chol": {"name": "Serum Cholesterol", "unit": "mg/dL", "optimal": (120.0, 199.0), "borderline": (200.0, 239.0), "max_disp": 400.0},
    "max_hr": {"name": "Maximum Heart Rate", "unit": "bpm", "optimal": (100.0, 175.0), "borderline": (75.0, 99.0), "max_disp": 220.0},
    "oldpeak": {"name": "ST Depression (Oldpeak)", "unit": "", "optimal": (0.0, 0.9), "borderline": (1.0, 1.9), "max_disp": 6.0},

    # Renal
    "sc": {"name": "Serum Creatinine", "unit": "mg/dL", "optimal": (0.6, 1.2), "borderline": (1.3, 1.5), "max_disp": 6.0},
    "bu": {"name": "Blood Urea", "unit": "mg/dL", "optimal": (15.0, 45.0), "borderline": (46.0, 60.0), "max_disp": 150.0},
    "bgr": {"name": "Blood Glucose (Random)", "unit": "mg/dL", "optimal": (70.0, 139.0), "borderline": (140.0, 199.0), "max_disp": 350.0},
    "hemo": {"name": "Hemoglobin", "unit": "g/dL", "optimal": (12.0, 16.5), "borderline": (10.0, 11.9), "max_disp": 20.0, "low_is_bad": True},
    "sod": {"name": "Serum Sodium", "unit": "mEq/L", "optimal": (135.0, 145.0), "borderline": (130.0, 134.0), "max_disp": 160.0},
    "pot": {"name": "Serum Potassium", "unit": "mEq/L", "optimal": (3.5, 5.0), "borderline": (5.1, 5.5), "max_disp": 8.0},

    # Hepatic
    "Total_Bilirubin": {"name": "Total Bilirubin", "unit": "mg/dL", "optimal": (0.2, 1.2), "borderline": (1.3, 2.0), "max_disp": 10.0},
    "Direct_Bilirubin": {"name": "Direct Bilirubin", "unit": "mg/dL", "optimal": (0.0, 0.3), "borderline": (0.4, 0.8), "max_disp": 6.0},
    "Alkaline_Phosphotase": {"name": "Alkaline Phosphatase (ALP)", "unit": "U/L", "optimal": (30.0, 120.0), "borderline": (121.0, 180.0), "max_disp": 600.0},
    "Alamine_Aminotransferase": {"name": "Alanine Aminotransferase (ALT)", "unit": "U/L", "optimal": (7.0, 56.0), "borderline": (57.0, 90.0), "max_disp": 300.0},
    "Aspartate_Aminotransferase": {"name": "Aspartate Aminotransferase (AST)", "unit": "U/L", "optimal": (10.0, 40.0), "borderline": (41.0, 75.0), "max_disp": 300.0},
    "Albumin": {"name": "Serum Albumin", "unit": "g/dL", "optimal": (3.4, 5.4), "borderline": (3.0, 3.3), "max_disp": 6.0, "low_is_bad": True},
    "Albumin_and_Globulin_Ratio": {"name": "A/G Ratio", "unit": "", "optimal": (1.0, 2.2), "borderline": (0.8, 0.99), "max_disp": 3.0, "low_is_bad": True},

    # Breast Cancer (FNA Nuclear)
    "mean_radius": {"name": "Nuclear Radius", "unit": "µm", "optimal": (6.0, 13.9), "borderline": (14.0, 16.0), "max_disp": 28.0},
    "mean_perimeter": {"name": "Nuclear Perimeter", "unit": "µm", "optimal": (40.0, 89.9), "borderline": (90.0, 105.0), "max_disp": 180.0},
    "mean_area": {"name": "Nuclear Area", "unit": "µm²", "optimal": (140.0, 499.0), "borderline": (500.0, 699.0), "max_disp": 1500.0},
    "mean_smoothness": {"name": "Nuclear Smoothness", "unit": "", "optimal": (0.06, 0.099), "borderline": (0.10, 0.12), "max_disp": 0.20},

    # Thyroid Function Panel
    "TSH": {"name": "Thyroid-Stimulating Hormone", "unit": "mIU/L", "optimal": (0.4, 4.0), "borderline": (4.1, 10.0), "max_disp": 25.0},
    "T3": {"name": "Total Triiodothyronine", "unit": "ng/mL", "optimal": (0.8, 2.0), "borderline": (0.6, 0.79), "max_disp": 4.0, "low_is_bad": True},
    "TT4": {"name": "Total Thyroxine (TT4)", "unit": "µg/dL", "optimal": (4.5, 12.0), "borderline": (3.5, 4.4), "max_disp": 20.0, "low_is_bad": True},
    "T4U": {"name": "T4 Uptake Ratio", "unit": "", "optimal": (0.8, 1.3), "borderline": (1.31, 1.5), "max_disp": 2.5},
    "FTI": {"name": "Free Thyroxine Index", "unit": "", "optimal": (1.2, 4.2), "borderline": (0.8, 1.19), "max_disp": 6.0, "low_is_bad": True}
}


def build_biomarker_gauges(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Constructs horizontal biomarker gauges comparing values against reference intervals."""
    gauges = []
    for key, spec in CLINICAL_REFERENCE_RANGES.items():
        if key in data and data[key] is not None:
            try:
                val = float(data[key])
                opt_min, opt_max = spec["optimal"]
                bord_min, bord_max = spec["borderline"]
                max_d = spec.get("max_disp", opt_max * 2.0)
                low_bad = spec.get("low_is_bad", False)

                # Determine status
                if opt_min <= val <= opt_max:
                    status = "optimal"
                    status_lbl = "Optimal Range"
                elif (bord_min <= val <= bord_max) or (low_bad and val < opt_min and val >= bord_min):
                    status = "borderline"
                    status_lbl = "Borderline / Review"
                else:
                    status = "elevated"
                    status_lbl = "Clinically Elevated" if not low_bad else "Clinically Deficient"

                # Gauge marker position percentage [0..100]
                pos_pct = min(100.0, max(0.0, (val / max_d) * 100.0))
                pct_rounded = round(pos_pct, 1)

                gauges.append({
                    "field": key,
                    "name": spec["name"],
                    "analyte": spec["name"],
                    "value": val,
                    "unit": spec["unit"],
                    "status": status,
                    "status_label": status_lbl,
                    "reference_interval": f"{opt_min} - {opt_max} {spec['unit']}".strip(),
                    "position_pct": pct_rounded,
                    "percent": pct_rounded,
                    "low": opt_min,
                    "high": opt_max
                })
            except (ValueError, TypeError):
                continue
    return gauges


def safe_percent(obj: Any, default: Any = 0) -> Any:
    """Safe dictionary or attribute access for percent across all model outputs."""
    if obj is None:
        return default
    return obj.get("percent", default) if isinstance(obj, dict) else getattr(obj, "percent", default)


def extract_top_contributing_factors(
    model: Any,
    feature_names: List[str],
    patient_vector: List[float],
    disease_key: str
) -> List[Dict[str, Any]]:
    """
    Extracts top contributing risk indicators using the actual trained Random Forest
    global feature importances and patient measurements.
    Strictly avoids fabricating pseudo-exact percentage contributions.
    """
    factors = []
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(feature_names):
        return []

    # Rank features by model global importance weight
    ranked_indices = np.argsort(importances)[::-1]

    for idx in ranked_indices:
        fname = feature_names[idx]
        weight = float(importances[idx])
        val = patient_vector[idx]

        # Check if this feature has a reference spec
        spec = CLINICAL_REFERENCE_RANGES.get(fname)
        if spec:
            opt_min, opt_max = spec["optimal"]
            is_atypical = (val < opt_min) or (val > opt_max)
            severity = "abnormal" if is_atypical else "normal"
            bench = f"Normal Target: {opt_min} - {opt_max} {spec['unit']}".strip()
        else:
            severity = "informative"
            bench = "Standard Clinical Target"

        lbl = spec["name"] if spec else fname.replace("_", " ").title()
        val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
        ctx = f"{bench} (Observed: {val_str})"
        f_pct = round(weight * 100.0, 1)
        w_round = round(weight, 3)
        factors.append({
            "feature": fname,
            "feature_name": lbl,
            "label": lbl,
            "value": val_str,
            "weight": w_round,
            "global_sensitivity_weight": w_round,
            "benchmark": bench,
            "clinical_context": ctx,
            "severity": severity,
            "percent": f_pct,
            "contribution_percent": f_pct,
            "importance": w_round
        })

    # Return top 4 most impactful features
    return factors[:4]


def generate_report_id(prefix: str) -> str:
    """Generates a non-guessable, professional Report Verification ID."""
    year = datetime.datetime.now().year
    rand_hex = uuid.uuid4().hex[:6].upper()
    return f"MNV-{prefix.upper()[:4]}-{year}-{rand_hex}"


# ============================================================================
# MODULE PREDICTORS
# ============================================================================

def predict_heart(
    data: Dict[str, Any],
    heart_model: Any,
    disease_insights: Dict[str, Any],
    is_scanner: bool = False
) -> Dict[str, Any]:
    """Shared prediction pipeline for Heart Disease."""
    raw_data = {
        'age': float(data['age']),
        'sex': float(data['sex']),
        'cp': float(data['cp']),
        'bp': float(data['bp']),
        'chol': float(data['chol']),
        'fbs': float(data['fbs']),
        'ekg': float(data['ekg']),
        'max_hr': float(data['max_hr']),
        'exang': float(data['exang']),
        'oldpeak': float(data['oldpeak']),
        'slope': float(data['slope']),
        'ca': float(data['ca']),
        'thal': float(data['thal'])
    }

    feature_names = ['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',
                     'EKG results', 'Max HR', 'Exercise angina', 'ST depression', 'Slope of ST',
                     'Number of vessels fluro', 'Thallium']

    features_vec = [
        raw_data['age'], raw_data['sex'], raw_data['cp'], raw_data['bp'],
        raw_data['chol'], raw_data['fbs'], raw_data['ekg'], raw_data['max_hr'],
        raw_data['exang'], raw_data['oldpeak'], raw_data['slope'], raw_data['ca'],
        raw_data['thal']
    ]

    import pandas as pd
    if hasattr(heart_model, "feature_names_in_"):
        X_df = pd.DataFrame([features_vec], columns=heart_model.feature_names_in_)
    else:
        X_df = [features_vec]

    prediction = int(heart_model.predict(X_df)[0])
    is_pos = bool(prediction == 1)
    result_text = "Presence of Cardiovascular Risk" if is_pos else "Absence of Significant Risk"
    insight_key = 'positive' if is_pos else 'negative'
    insights = disease_insights['heart_disease'][insight_key]

    conf_score = None
    try:
        if hasattr(heart_model, "predict_proba"):
            probs = heart_model.predict_proba(X_df)[0]
            conf_score = round(float(probs[1] if is_pos else probs[0]) * 100.0, 1)
    except Exception:
        conf_score = None

    # Silent Google Sheets logging
    try:
        log_patient_prediction(
            module_name="Heart_Disease",
            form_data=data,
            raw_prediction=prediction,
            is_positive=is_pos,
            confidence_score=conf_score
        )
    except Exception as e:
        logger.warning(f"Google sheets log error in heart: {e}")

    patient_summary = [
        {'label': 'Age', 'value': f"{int(raw_data['age'])} years"},
        {'label': 'Sex', 'value': 'Male' if raw_data['sex'] == 1 else 'Female'},
        {'label': 'Chest Pain Type', 'value': ['Typical Angina', 'Atypical Angina', 'Non-Anginal', 'Asymptomatic'][int(raw_data['cp'])]},
        {'label': 'Resting Blood Pressure', 'value': f"{int(raw_data['bp'])} mm Hg"},
        {'label': 'Serum Cholesterol', 'value': f"{int(raw_data['chol'])} mg/dL"},
        {'label': 'Fasting Blood Sugar > 120', 'value': 'Yes' if raw_data['fbs'] == 1 else 'No'},
        {'label': 'Resting ECG', 'value': ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'][int(raw_data['ekg'])]},
        {'label': 'Max Heart Rate', 'value': f"{int(raw_data['max_hr'])} bpm"},
        {'label': 'Exercise Induced Angina', 'value': 'Yes' if raw_data['exang'] == 1 else 'No'},
        {'label': 'ST Depression (Oldpeak)', 'value': f"{raw_data['oldpeak']}"},
        {'label': 'ST Slope', 'value': ['Upsloping', 'Flat', 'Downsloping'][int(raw_data['slope'])]},
        {'label': 'Major Vessels Fluro', 'value': f"{int(raw_data['ca'])}"},
        {'label': 'Thallium Stress Result', 'value': {3: 'Normal', 6: 'Fixed Defect', 7: 'Reversible Defect'}.get(int(raw_data['thal']), str(raw_data['thal']))}
    ]

    report_id = generate_report_id("HEAR")
    register_report(
        report_id=report_id,
        module_name="heart_disease",
        condition_title="Cardiovascular Disease Evaluation",
        prediction_text=result_text,
        is_positive=is_pos,
        confidence_score=conf_score,
        is_scanner_extracted=is_scanner,
        summary_data={"patient_summary": patient_summary}
    )

    gauges = build_biomarker_gauges(raw_data)
    risk_factors = extract_top_contributing_factors(heart_model, feature_names, features_vec, "heart_disease")

    norm_conf = float(conf_score) if conf_score is not None else None
    norm_pct = float(conf_score) if conf_score is not None else None
    norm_prob = round(norm_conf / 100.0, 4) if norm_conf is not None else None

    for g in gauges:
        if isinstance(g, dict):
            g["percent"] = g.get("percent", 0) if isinstance(g, dict) else getattr(g, "percent", 0)
            g["low"] = g.get("low", 0)
            g["high"] = g.get("high", 0)
            g["analyte"] = g.get("analyte", g.get("name", ""))

    for f in risk_factors:
        if isinstance(f, dict):
            f["percent"] = f.get("percent", 0) if isinstance(f, dict) else getattr(f, "percent", 0)
            f["weight"] = f.get("weight", f.get("global_sensitivity_weight", f.get("percent", 10.0) / 100.0))
            f["feature_name"] = f.get("feature_name", f.get("label", f.get("feature", "Clinical Indicator")))
            f["clinical_context"] = f.get("clinical_context", f.get("benchmark", ""))

    return {
        "prediction": result_text,
        "risk_label": result_text,
        "is_positive": is_pos,
        "confidence_score": norm_conf,
        "confidence": norm_conf,
        "percent": norm_pct,
        "probability": norm_prob,
        "insights": insights,
        "patient_summary": patient_summary,
        "report_id": report_id,
        "biomarker_gauges": gauges,
        "top_risk_factors": risk_factors,
        "feature_contributions": risk_factors
    }


def predict_diabetes(
    data: Dict[str, Any],
    diabetes_model: Any,
    disease_insights: Dict[str, Any],
    is_scanner: bool = False
) -> Dict[str, Any]:
    """Shared prediction pipeline for Diabetes."""
    gender = str(data.get('gender', 'Female'))
    smoking = str(data.get('smoking_history', 'never')).lower()
    age = float(data['age'])
    hypertension = int(data.get('hypertension', 0))
    heart_disease = int(data.get('heart_disease', 0))
    bmi = float(data.get('bmi', 25.0))
    HbA1c_level = float(data['HbA1c_level'])
    blood_glucose_level = float(data['blood_glucose_level'])

    encoded_data = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level,
        'gender_Male': 1 if gender == 'Male' else 0,
        'gender_Other': 1 if gender == 'Other' else 0,
        'smoking_history_current': 1 if smoking == 'current' else 0,
        'smoking_history_ever': 1 if smoking == 'ever' else 0,
        'smoking_history_former': 1 if smoking == 'former' else 0,
        'smoking_history_never': 1 if smoking in ('never', 'no info') else 0,
        'smoking_history_not current': 1 if smoking == 'not current' else 0
    }

    feature_order = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
                     'gender_Male', 'gender_Other', 'smoking_history_current', 'smoking_history_ever',
                     'smoking_history_former', 'smoking_history_never', 'smoking_history_not current']

    feature_vector = [encoded_data[col] for col in feature_order]

    import pandas as pd
    if hasattr(diabetes_model, "feature_names_in_"):
        X_df = pd.DataFrame([feature_vector], columns=diabetes_model.feature_names_in_)
    else:
        X_df = [feature_vector]

    prediction = int(diabetes_model.predict(X_df)[0])
    is_pos = bool(prediction == 1)
    result_text = "Positive - Elevated Diabetes Risk" if is_pos else "Negative - Normal Glycemic Indicators"
    insight_key = 'positive' if is_pos else 'negative'
    insights = disease_insights['diabetes'][insight_key]

    conf_score = None
    try:
        if hasattr(diabetes_model, "predict_proba"):
            probs = diabetes_model.predict_proba(X_df)[0]
            conf_score = round(float(probs[1] if is_pos else probs[0]) * 100.0, 1)
    except Exception:
        conf_score = None

    try:
        log_patient_prediction(
            module_name="Diabetes",
            form_data=data,
            raw_prediction=prediction,
            is_positive=is_pos,
            confidence_score=conf_score
        )
    except Exception as e:
        logger.warning(f"Google sheets log error in diabetes: {e}")

    patient_summary = [
        {'label': 'Gender', 'value': gender},
        {'label': 'Age', 'value': f"{int(age)} years"},
        {'label': 'Hypertension History', 'value': 'Yes' if hypertension == 1 else 'No'},
        {'label': 'Heart Disease History', 'value': 'Yes' if heart_disease == 1 else 'No'},
        {'label': 'Smoking History', 'value': smoking.capitalize()},
        {'label': 'Body Mass Index (BMI)', 'value': f"{bmi:.1f} kg/m²"},
        {'label': 'HbA1c Level', 'value': f"{HbA1c_level:.1f}%"},
        {'label': 'Blood Glucose Level', 'value': f"{int(blood_glucose_level)} mg/dL"}
    ]

    report_id = generate_report_id("DIAB")
    register_report(
        report_id=report_id,
        module_name="diabetes",
        condition_title="Diabetes & Glycemic Profile",
        prediction_text=result_text,
        is_positive=is_pos,
        confidence_score=conf_score,
        is_scanner_extracted=is_scanner,
        summary_data={"patient_summary": patient_summary}
    )

    gauges = build_biomarker_gauges({'HbA1c_level': HbA1c_level, 'blood_glucose_level': blood_glucose_level, 'bmi': bmi})
    risk_factors = extract_top_contributing_factors(diabetes_model, feature_order, feature_vector, "diabetes")

    norm_conf = float(conf_score) if conf_score is not None else None
    norm_pct = float(conf_score) if conf_score is not None else None
    norm_prob = round(norm_conf / 100.0, 4) if norm_conf is not None else None

    for g in gauges:
        if isinstance(g, dict):
            g["percent"] = g.get("percent", 0) if isinstance(g, dict) else getattr(g, "percent", 0)
            g["low"] = g.get("low", 0)
            g["high"] = g.get("high", 0)
            g["analyte"] = g.get("analyte", g.get("name", ""))

    for f in risk_factors:
        if isinstance(f, dict):
            f["percent"] = f.get("percent", 0) if isinstance(f, dict) else getattr(f, "percent", 0)
            f["weight"] = f.get("weight", f.get("global_sensitivity_weight", f.get("percent", 10.0) / 100.0))
            f["feature_name"] = f.get("feature_name", f.get("label", f.get("feature", "Clinical Indicator")))
            f["clinical_context"] = f.get("clinical_context", f.get("benchmark", ""))

    return {
        "prediction": result_text,
        "risk_label": result_text,
        "is_positive": is_pos,
        "confidence_score": norm_conf,
        "confidence": norm_conf,
        "percent": norm_pct,
        "probability": norm_prob,
        "insights": insights,
        "patient_summary": patient_summary,
        "report_id": report_id,
        "biomarker_gauges": gauges,
        "top_risk_factors": risk_factors,
        "feature_contributions": risk_factors
    }


def predict_kidney(
    data: Dict[str, Any],
    kidney_model: Any,
    disease_insights: Dict[str, Any],
    is_scanner: bool = False
) -> Dict[str, Any]:
    """Shared prediction pipeline for Kidney Disease."""
    field_keys = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
                  'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm',
                  'cad', 'appet', 'pe', 'ane']

    features = [float(data.get(k, 0)) for k in field_keys]

    if hasattr(kidney_model, "feature_names_in_"):
        X_df = pd.DataFrame([features], columns=kidney_model.feature_names_in_)
    else:
        X_df = [features]

    prediction = int(kidney_model.predict(X_df)[0])
    is_pos = bool(prediction == 1)
    result_text = "Positive - Chronic Kidney Disease Indicators" if is_pos else "Negative - Healthy Renal Function"
    insight_key = 'positive' if is_pos else 'negative'
    insights = disease_insights['kidney_disease'][insight_key]

    conf_score = None
    try:
        if hasattr(kidney_model, "predict_proba"):
            probs = kidney_model.predict_proba(X_df)[0]
            conf_score = round(float(probs[1] if is_pos else probs[0]) * 100.0, 1)
    except Exception:
        conf_score = None

    try:
        log_patient_prediction(
            module_name="Kidney_Disease",
            form_data=data,
            raw_prediction=prediction,
            is_positive=is_pos,
            confidence_score=conf_score
        )
    except Exception as e:
        logger.warning(f"Google sheets log error in kidney: {e}")

    patient_summary = [
        {'label': 'Age', 'value': f"{int(features[0])} years"},
        {'label': 'Blood Pressure', 'value': f"{int(features[1])} mm Hg"},
        {'label': 'Specific Gravity', 'value': f"{features[2]:.3f}"},
        {'label': 'Albumin (Urine)', 'value': f"Grade {int(features[3])}"},
        {'label': 'Sugar (Urine)', 'value': f"Grade {int(features[4])}"},
        {'label': 'Red Blood Cells', 'value': 'Normal' if features[5] == 1 else 'Abnormal'},
        {'label': 'Pus Cells', 'value': 'Normal' if features[6] == 1 else 'Abnormal'},
        {'label': 'Blood Glucose Random', 'value': f"{int(features[9])} mg/dL"},
        {'label': 'Blood Urea', 'value': f"{features[10]} mg/dL"},
        {'label': 'Serum Creatinine', 'value': f"{features[11]} mg/dL"},
        {'label': 'Hemoglobin', 'value': f"{features[14]} g/dL"},
        {'label': 'Packed Cell Volume', 'value': f"{int(features[15])}%"},
        {'label': 'Hypertension', 'value': 'Yes' if features[18] == 1 else 'No'},
        {'label': 'Diabetes Mellitus', 'value': 'Yes' if features[19] == 1 else 'No'}
    ]

    report_id = generate_report_id("KIDN")
    register_report(
        report_id=report_id,
        module_name="kidney_disease",
        condition_title="Chronic Kidney Disease Panel",
        prediction_text=result_text,
        is_positive=is_pos,
        confidence_score=conf_score,
        is_scanner_extracted=is_scanner,
        summary_data={"patient_summary": patient_summary}
    )

    gauges = build_biomarker_gauges({
        'sc': features[11], 'bu': features[10], 'bgr': features[9],
        'hemo': features[14], 'sod': features[12], 'pot': features[13]
    })
    risk_factors = extract_top_contributing_factors(kidney_model, field_keys, features, "kidney_disease")

    norm_conf = float(conf_score) if conf_score is not None else None
    norm_pct = float(conf_score) if conf_score is not None else None
    norm_prob = round(norm_conf / 100.0, 4) if norm_conf is not None else None

    for g in gauges:
        if isinstance(g, dict):
            g["percent"] = g.get("percent", 0) if isinstance(g, dict) else getattr(g, "percent", 0)
            g["low"] = g.get("low", 0)
            g["high"] = g.get("high", 0)
            g["analyte"] = g.get("analyte", g.get("name", ""))

    for f in risk_factors:
        if isinstance(f, dict):
            f["percent"] = f.get("percent", 0) if isinstance(f, dict) else getattr(f, "percent", 0)
            f["weight"] = f.get("weight", f.get("global_sensitivity_weight", f.get("percent", 10.0) / 100.0))
            f["feature_name"] = f.get("feature_name", f.get("label", f.get("feature", "Clinical Indicator")))
            f["clinical_context"] = f.get("clinical_context", f.get("benchmark", ""))

    return {
        "prediction": result_text,
        "risk_label": result_text,
        "is_positive": is_pos,
        "confidence_score": norm_conf,
        "confidence": norm_conf,
        "percent": norm_pct,
        "probability": norm_prob,
        "insights": insights,
        "patient_summary": patient_summary,
        "report_id": report_id,
        "biomarker_gauges": gauges,
        "top_risk_factors": risk_factors,
        "feature_contributions": risk_factors
    }


def predict_liver(
    data: Dict[str, Any],
    liver_model: Any,
    disease_insights: Dict[str, Any],
    is_scanner: bool = False
) -> Dict[str, Any]:
    """Shared prediction pipeline for Liver Disease."""
    gender_raw = data.get('Gender', 'Male')
    gender_val = 1 if gender_raw in ('Male', 1, '1') else 0

    features = [
        float(data['Age']),
        gender_val,
        float(data['Total_Bilirubin']),
        float(data['Direct_Bilirubin']),
        float(data['Alkaline_Phosphotase']),
        float(data['Alamine_Aminotransferase']),
        float(data['Aspartate_Aminotransferase']),
        float(data['Total_Protiens']),
        float(data['Albumin']),
        float(data['Albumin_and_Globulin_Ratio'])
    ]

    feature_names = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                     'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
                     'Albumin', 'Albumin_and_Globulin_Ratio']

    import pandas as pd
    if hasattr(liver_model, "feature_names_in_"):
        X_df = pd.DataFrame([features], columns=liver_model.feature_names_in_)
    else:
        X_df = [features]

    prediction = int(liver_model.predict(X_df)[0])
    is_pos = bool(prediction == 1)
    result_text = "High Probability of Liver Disease" if is_pos else "Low Probability of Liver Disease"
    insight_key = 'positive' if is_pos else 'negative'
    insights = disease_insights['liver_disease'][insight_key]

    conf_score = None
    try:
        if hasattr(liver_model, "predict_proba"):
            probs = liver_model.predict_proba(X_df)[0]
            conf_score = round(float(probs[1] if is_pos else probs[0]) * 100.0, 1)
    except Exception:
        conf_score = None

    try:
        log_patient_prediction(
            module_name="Liver_Disease",
            form_data=data,
            raw_prediction=prediction,
            is_positive=is_pos,
            confidence_score=conf_score
        )
    except Exception as e:
        logger.warning(f"Google sheets log error in liver: {e}")

    patient_summary = [
        {'label': 'Age', 'value': f"{int(features[0])} years"},
        {'label': 'Gender', 'value': 'Male' if gender_val == 1 else 'Female'},
        {'label': 'Total Bilirubin', 'value': f"{features[2]:.1f} mg/dL"},
        {'label': 'Direct Bilirubin', 'value': f"{features[3]:.1f} mg/dL"},
        {'label': 'Alkaline Phosphatase (ALP)', 'value': f"{int(features[4])} U/L"},
        {'label': 'Alamine Aminotransferase (ALT)', 'value': f"{int(features[5])} U/L"},
        {'label': 'Aspartate Aminotransferase (AST)', 'value': f"{int(features[6])} U/L"},
        {'label': 'Total Proteins', 'value': f"{features[7]:.1f} g/dL"},
        {'label': 'Albumin', 'value': f"{features[8]:.1f} g/dL"},
        {'label': 'A/G Ratio', 'value': f"{features[9]:.2f}"}
    ]

    report_id = generate_report_id("LIVR")
    register_report(
        report_id=report_id,
        module_name="liver_disease",
        condition_title="Hepatic Function Panel",
        prediction_text=result_text,
        is_positive=is_pos,
        confidence_score=conf_score,
        is_scanner_extracted=is_scanner,
        summary_data={"patient_summary": patient_summary}
    )

    gauges = build_biomarker_gauges({
        'Total_Bilirubin': features[2], 'Direct_Bilirubin': features[3],
        'Alkaline_Phosphotase': features[4], 'Alamine_Aminotransferase': features[5],
        'Aspartate_Aminotransferase': features[6], 'Albumin': features[8],
        'Albumin_and_Globulin_Ratio': features[9]
    })
    risk_factors = extract_top_contributing_factors(liver_model, feature_names, features, "liver_disease")

    norm_conf = float(conf_score) if conf_score is not None else None
    norm_pct = float(conf_score) if conf_score is not None else None
    norm_prob = round(norm_conf / 100.0, 4) if norm_conf is not None else None

    for g in gauges:
        if isinstance(g, dict):
            g["percent"] = g.get("percent", 0) if isinstance(g, dict) else getattr(g, "percent", 0)
            g["low"] = g.get("low", 0)
            g["high"] = g.get("high", 0)
            g["analyte"] = g.get("analyte", g.get("name", ""))

    for f in risk_factors:
        if isinstance(f, dict):
            f["percent"] = f.get("percent", 0) if isinstance(f, dict) else getattr(f, "percent", 0)
            f["weight"] = f.get("weight", f.get("global_sensitivity_weight", f.get("percent", 10.0) / 100.0))
            f["feature_name"] = f.get("feature_name", f.get("label", f.get("feature", "Clinical Indicator")))
            f["clinical_context"] = f.get("clinical_context", f.get("benchmark", ""))

    return {
        "prediction": result_text,
        "risk_label": result_text,
        "is_positive": is_pos,
        "confidence_score": norm_conf,
        "confidence": norm_conf,
        "percent": norm_pct,
        "probability": norm_prob,
        "insights": insights,
        "patient_summary": patient_summary,
        "report_id": report_id,
        "biomarker_gauges": gauges,
        "top_risk_factors": risk_factors,
        "feature_contributions": risk_factors
    }


def predict_breast_cancer(
    data: Dict[str, Any],
    breast_cancer_model: Any,
    disease_insights: Dict[str, Any],
    is_scanner: bool = False
) -> Dict[str, Any]:
    """Shared prediction pipeline for Breast Cancer FNA Biopsy."""
    features = [
        float(data['mean_radius']),
        float(data['mean_texture']),
        float(data['mean_perimeter']),
        float(data['mean_area']),
        float(data['mean_smoothness'])
    ]

    feature_names = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']
    cols = getattr(breast_cancer_model, 'feature_names_in_', feature_names)
    df_features = pd.DataFrame([features], columns=cols)

    prediction = int(breast_cancer_model.predict(df_features)[0])
    # Model classes: 0 = Malignant, 1 = Benign
    is_pos = bool(prediction == 0)
    result_text = "Malignant Neoplasm Indicators" if is_pos else "Benign Cellular Characteristics"
    insight_key = 'malignant' if is_pos else 'benign'
    insights = disease_insights['breast_cancer'][insight_key]

    conf_score = None
    try:
        if hasattr(breast_cancer_model, "predict_proba"):
            probs = breast_cancer_model.predict_proba(df_features)[0]
            cls_idx = list(breast_cancer_model.classes_).index(prediction) if hasattr(breast_cancer_model, 'classes_') else prediction
            conf_score = round(float(probs[cls_idx]) * 100.0, 1)
    except Exception:
        conf_score = None

    try:
        log_patient_prediction(
            module_name="Breast_Cancer",
            form_data=data,
            raw_prediction="Malignant" if is_pos else "Benign",
            is_positive=is_pos,
            confidence_score=conf_score
        )
    except Exception as e:
        logger.warning(f"Google sheets log error in breast cancer: {e}")

    patient_summary = [
        {'label': 'Mean Nuclear Radius', 'value': f"{features[0]:.2f} µm"},
        {'label': 'Mean Texture Variation', 'value': f"{features[1]:.2f}"},
        {'label': 'Mean Perimeter', 'value': f"{features[2]:.2f} µm"},
        {'label': 'Mean Area', 'value': f"{features[3]:.1f} µm²"},
        {'label': 'Mean Smoothness Index', 'value': f"{features[4]:.4f}"}
    ]

    report_id = generate_report_id("ONCO")
    register_report(
        report_id=report_id,
        module_name="breast_cancer",
        condition_title="Breast Oncology Biopsy Morphometry",
        prediction_text=result_text,
        is_positive=is_pos,
        confidence_score=conf_score,
        is_scanner_extracted=is_scanner,
        summary_data={"patient_summary": patient_summary}
    )

    gauges = build_biomarker_gauges({
        'mean_radius': features[0], 'mean_perimeter': features[2],
        'mean_area': features[3], 'mean_smoothness': features[4]
    })
    risk_factors = extract_top_contributing_factors(breast_cancer_model, feature_names, features, "breast_cancer")

    norm_conf = float(conf_score) if conf_score is not None else None
    norm_pct = float(conf_score) if conf_score is not None else None
    norm_prob = round(norm_conf / 100.0, 4) if norm_conf is not None else None

    for g in gauges:
        if isinstance(g, dict):
            g["percent"] = g.get("percent", 0) if isinstance(g, dict) else getattr(g, "percent", 0)
            g["low"] = g.get("low", 0)
            g["high"] = g.get("high", 0)
            g["analyte"] = g.get("analyte", g.get("name", ""))

    for f in risk_factors:
        if isinstance(f, dict):
            f["percent"] = f.get("percent", 0) if isinstance(f, dict) else getattr(f, "percent", 0)
            f["weight"] = f.get("weight", f.get("global_sensitivity_weight", f.get("percent", 10.0) / 100.0))
            f["feature_name"] = f.get("feature_name", f.get("label", f.get("feature", "Clinical Indicator")))
            f["clinical_context"] = f.get("clinical_context", f.get("benchmark", ""))

    return {
        "prediction": result_text,
        "risk_label": result_text,
        "is_positive": is_pos,
        "confidence_score": norm_conf,
        "confidence": norm_conf,
        "percent": norm_pct,
        "probability": norm_prob,
        "insights": insights,
        "patient_summary": patient_summary,
        "report_id": report_id,
        "biomarker_gauges": gauges,
        "top_risk_factors": risk_factors,
        "feature_contributions": risk_factors
    }


def predict_thyroid(
    data: Dict[str, Any],
    thyroid_artifact: Any,
    disease_insights: Dict[str, Any],
    is_scanner: bool = False
) -> Dict[str, Any]:
    """Shared prediction pipeline for Thyroid Dysfunction Screening."""
    model_obj = thyroid_artifact.get("model", thyroid_artifact) if isinstance(thyroid_artifact, dict) else thyroid_artifact
    feat_names = thyroid_artifact.get("feature_names", ['age', 'sex', 'on_thyroxine', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']) if isinstance(thyroid_artifact, dict) else ['age', 'sex', 'on_thyroxine', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    feat_importances = thyroid_artifact.get("feature_importances") if isinstance(thyroid_artifact, dict) else None
    feat_medians = thyroid_artifact.get("feature_medians", {
        'age': 54.0, 'sex': 0.0, 'on_thyroxine': 0.0, 'TSH': 1.4, 'T3': 2.0, 'TT4': 103.0, 'T4U': 0.98, 'FTI': 107.0
    }) if isinstance(thyroid_artifact, dict) else {}

    # Extract & impute missing with median baseline
    def _parse_val(k, default_val):
        v = data.get(k)
        if v is None or str(v).strip() == "":
            v = data.get(k.lower())
        if v is None or str(v).strip() == "":
            v = data.get(k.upper())
        if v is None or str(v).strip() == "":
            return float(feat_medians.get(k, default_val))
        try:
            return float(v)
        except (ValueError, TypeError):
            return float(feat_medians.get(k, default_val))

    age = _parse_val('age', 50.0)
    # Sex: support 'Male'/'Female' or 1/0
    sex_raw = str(data.get('sex', data.get('gender', '0'))).lower()
    sex = 1.0 if sex_raw in ('1', 'male', 'm') else 0.0
    on_thyroxine_raw = str(data.get('on_thyroxine', '0')).lower()
    on_thyroxine = 1.0 if on_thyroxine_raw in ('1', 'true', 't', 'yes', 'y') else 0.0

    tsh = _parse_val('TSH', 1.4)
    t3 = _parse_val('T3', 1.8)
    tt4 = _parse_val('TT4', 103.0)
    t4u = _parse_val('T4U', 0.98)
    fti = _parse_val('FTI', 107.0)

    feature_vector = [age, sex, on_thyroxine, tsh, t3, tt4, t4u, fti]
    X_df = pd.DataFrame([feature_vector], columns=feat_names)

    prediction = int(model_obj.predict(X_df)[0])
    is_pos = bool(prediction == 1)
    result_text = "Elevated Thyroid Dysfunction (Hypothyroidism Marker)" if is_pos else "Euthyroid - Normal Thyroid Function"
    insight_key = 'positive' if is_pos else 'negative'
    insights = disease_insights.get('thyroid', {}).get(insight_key, {
        "title": "Thyroid Evaluation",
        "description": "Biochemical assessment of serum thyroid panel analytes.",
        "specialist": "Endocrinologist / Thyroid Clinic",
        "urgency_window": "Schedule endocrine follow-up within 7 to 10 days." if is_pos else "Annual routine wellness checkup.",
        "recommended_tests": ["Repeat Serum TSH", "Free T4 (FT4)", "Anti-TPO Antibodies", "Thyroid Ultrasound"]
    })

    conf_score = None
    try:
        if hasattr(model_obj, "predict_proba"):
            probs = model_obj.predict_proba(X_df)[0]
            conf_score = round(float(probs[1] if is_pos else probs[0]) * 100.0, 1)
    except Exception:
        conf_score = None

    try:
        log_patient_prediction(
            module_name="Thyroid_Dysfunction",
            form_data=data,
            raw_prediction="Hypothyroid" if is_pos else "Euthyroid",
            is_positive=is_pos,
            confidence_score=conf_score
        )
    except Exception as e:
        logger.warning(f"Google sheets log error in thyroid: {e}")

    patient_summary = [
        {'label': 'Patient Age', 'value': f"{int(age)} years"},
        {'label': 'Biological Sex', 'value': 'Male' if sex == 1 else 'Female'},
        {'label': 'On Thyroxine Medication', 'value': 'Yes' if on_thyroxine == 1 else 'No'},
        {'label': 'Thyroid-Stimulating Hormone (TSH)', 'value': f"{tsh:.2f} mIU/L"},
        {'label': 'Total Triiodothyronine (T3)', 'value': f"{t3:.2f} ng/mL"},
        {'label': 'Total Thyroxine (TT4)', 'value': f"{tt4:.1f} µg/dL"},
        {'label': 'T4 Uptake Ratio (T4U)', 'value': f"{t4u:.2f}"},
        {'label': 'Free Thyroxine Index (FTI)', 'value': f"{fti:.1f}"}
    ]

    report_id = generate_report_id("THYR")
    register_report(
        report_id=report_id,
        module_name="thyroid",
        condition_title="Thyroid Function & Biochemical Panel",
        prediction_text=result_text,
        is_positive=is_pos,
        confidence_score=conf_score,
        is_scanner_extracted=is_scanner,
        summary_data={"patient_summary": patient_summary}
    )

    gauges = build_biomarker_gauges({
        'TSH': tsh, 'T3': t3, 'TT4': tt4, 'T4U': t4u, 'FTI': fti
    })

    if feat_importances:
        risk_factors = []
        ranked_feats = sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)[:4]
        for fn, wt in ranked_feats:
            idx = feat_names.index(fn) if fn in feat_names else 0
            val = feature_vector[idx]
            spec = CLINICAL_REFERENCE_RANGES.get(fn)
            opt_min, opt_max = spec["optimal"] if spec else (0, 100)
            is_atypical = (val < opt_min) or (val > opt_max)
            pct = round(float(wt) * 100.0, 1)
            risk_factors.append({
                "feature": fn,
                "feature_name": spec["name"] if spec else fn,
                "label": spec["name"] if spec else fn,
                "value": f"{val:.2f}",
                "weight": round(float(wt), 3),
                "benchmark": f"Target: {opt_min} - {opt_max} {spec['unit'] if spec else ''}".strip(),
                "clinical_context": f"Target: {opt_min} - {opt_max} (Observed: {val:.2f})",
                "severity": "abnormal" if is_atypical else "normal",
                "percent": pct,
                "contribution_percent": pct,
                "importance": round(float(wt), 3)
            })
    else:
        risk_factors = extract_top_contributing_factors(model_obj, feat_names, feature_vector, "thyroid")

    norm_conf = float(conf_score) if conf_score is not None else None
    norm_prob = round(norm_conf / 100.0, 4) if norm_conf is not None else None

    return {
        "prediction": result_text,
        "risk_label": result_text,
        "is_positive": is_pos,
        "confidence_score": norm_conf,
        "confidence": norm_conf,
        "percent": norm_conf,
        "probability": norm_prob,
        "insights": insights,
        "patient_summary": patient_summary,
        "report_id": report_id,
        "biomarker_gauges": gauges,
        "top_risk_factors": risk_factors,
        "feature_contributions": risk_factors
    }


def predict_arrhythmia(
    signal_data: Any,
    arrhythmia_artifact: Any,
    disease_insights: Dict[str, Any],
    sample_rate: float = 360.0
) -> Dict[str, Any]:
    """Shared prediction pipeline for ECG Arrhythmia Biosignal Screening."""
    from utils.signal_processor import (
        validate_ecg_signal_quality,
        filter_ecg_signal,
        resample_to_canonical,
        detect_r_peaks,
        extract_features_from_waveform,
        parse_ecg_input_data
    )

    # Ingest and parse signal array
    raw_sig, fs, parse_err = parse_ecg_input_data(signal_data)
    if parse_err or len(raw_sig) == 0:
        raise ValueError(parse_err or "Invalid or unparseable ECG signal payload.")

    # Validate signal readability and length
    is_valid, s_err, sig_metrics = validate_ecg_signal_quality(raw_sig, fs)
    if not is_valid:
        raise ValueError(s_err)

    # Resample to canonical 360 Hz if required
    resampled_sig = resample_to_canonical(raw_sig, fs, 360.0)
    clean_sig = filter_ecg_signal(resampled_sig, 360.0)

    # Detect QRS R-peaks
    r_peaks = detect_r_peaks(clean_sig, 360.0)
    if len(r_peaks) < 3:
        raise ValueError("Insufficient physiological R-peaks detected in the ECG segment. Please provide a clear lead recording.")

    # Extract morphological & interval beat vectors
    feature_matrix, beat_meta = extract_features_from_waveform(clean_sig, r_peaks, 360.0)
    if len(feature_matrix) == 0:
        raise ValueError("Could not segment sufficient complete heartbeats from recording window.")

    # Extract model and threshold
    model_obj = arrhythmia_artifact.get("model", arrhythmia_artifact) if isinstance(arrhythmia_artifact, dict) else arrhythmia_artifact
    threshold = arrhythmia_artifact.get("operating_threshold", 0.20) if isinstance(arrhythmia_artifact, dict) else 0.20

    # Execute beat-by-beat inference
    if hasattr(model_obj, "predict_proba"):
        probs = model_obj.predict_proba(feature_matrix)[:, 1]
    else:
        probs = model_obj.predict(feature_matrix).astype(float)

    beat_preds = (probs >= threshold).astype(int)
    arrhythmic_count = int(np.sum(beat_preds == 1))
    total_beats = len(beat_preds)
    arrhythmia_pct = round((arrhythmic_count / float(total_beats)) * 100.0, 1)

    is_pos = bool(arrhythmic_count > 0)
    result_text = f"Cardiac Arrhythmia Detected ({arrhythmic_count}/{total_beats} Ectopic Beats)" if is_pos else f"Normal Sinus Rhythm ({total_beats} Regular Beats)"

    # Compute Heart Rate Metrics
    rr_ms = [b["rr_pre_ms"] for b in beat_meta]
    mean_bpm = round(float(np.mean([b["instant_bpm"] for b in beat_meta])), 1)
    # RMSSD (Root Mean Square of Successive Differences for HRV)
    rmssd = round(float(np.sqrt(np.mean(np.diff(rr_ms) ** 2))), 1) if len(rr_ms) > 1 else 0.0

    insight_key = 'positive' if is_pos else 'negative'
    insights = disease_insights.get('arrhythmia', {}).get(insight_key, {
        "title": "Cardiac Rhythm Analysis",
        "description": "Automated computerized rhythm classification from surface ECG telemetry.",
        "specialist": "Cardiologist & Cardiac Electrophysiologist",
        "urgency_window": "Schedule clinical cardiology appointment within 24 to 48 hours." if is_pos else "Routine annual wellness screening.",
        "recommended_tests": ["12-Lead Diagnostic ECG", "24-Hour Holter Monitor", "Echocardiogram", "Serum Electrolytes (Potassium, Magnesium)"]
    })

    conf_score = round(float(np.max(probs) * 100.0), 1) if is_pos else round(float((1.0 - np.mean(probs)) * 100.0), 1)

    try:
        log_patient_prediction(
            module_name="Cardiac_Arrhythmia",
            form_data={"beats_analyzed": total_beats, "ectopic_beats": arrhythmic_count, "mean_bpm": mean_bpm, "rmssd_hrv": rmssd},
            raw_prediction="Arrhythmia" if is_pos else "Normal Sinus Rhythm",
            is_positive=is_pos,
            confidence_score=conf_score
        )
    except Exception as e:
        logger.warning(f"Google sheets log error in arrhythmia: {e}")

    patient_summary = [
        {'label': 'Recording Duration', 'value': f"{sig_metrics['duration_sec']} seconds"},
        {'label': 'Analyzed Heartbeats', 'value': f"{total_beats} beats"},
        {'label': 'Mean Heart Rate', 'value': f"{mean_bpm} bpm"},
        {'label': 'HRV (RMSSD)', 'value': f"{rmssd} ms"},
        {'label': 'Ectopic Arrhythmic Beats', 'value': f"{arrhythmic_count} beats ({arrhythmia_pct}%)"},
        {'label': 'Lead Configuration', 'value': 'Single-Lead Modified Lead II (MLII)'}
    ]

    report_id = generate_report_id("ARRH")
    register_report(
        report_id=report_id,
        module_name="arrhythmia",
        condition_title="ECG Biosignal Arrhythmia Screening",
        prediction_text=result_text,
        is_positive=is_pos,
        confidence_score=conf_score,
        is_scanner_extracted=False,
        summary_data={"patient_summary": patient_summary}
    )

    # Subsample rhythm points for client-side rhythm strip display (first 10 seconds or 1800 points)
    strip_len = min(1800, len(clean_sig))
    rhythm_strip = [round(float(y), 3) for y in clean_sig[:strip_len]]

    # Tag beat positions for strip visualization
    strip_peaks = [int(p) for p in r_peaks if p < strip_len]

    return {
        "prediction": result_text,
        "risk_label": result_text,
        "is_positive": is_pos,
        "confidence_score": conf_score,
        "confidence": conf_score,
        "percent": conf_score,
        "probability": round(conf_score / 100.0, 4),
        "insights": insights,
        "patient_summary": patient_summary,
        "report_id": report_id,
        "biomarker_gauges": [
            {"field": "bpm", "name": "Heart Rate", "value": mean_bpm, "unit": "bpm", "status": "optimal" if 60 <= mean_bpm <= 100 else "borderline", "status_label": "Normal Rate" if 60 <= mean_bpm <= 100 else "Tachycardia / Bradycardia", "reference_interval": "60 - 100 bpm", "position_pct": min(100.0, (mean_bpm / 200.0) * 100.0)},
            {"field": "hrv", "name": "Heart Rate Variability (RMSSD)", "value": rmssd, "unit": "ms", "status": "optimal" if rmssd >= 25 else "borderline", "status_label": "Favorable Autonomic Tone" if rmssd >= 25 else "Depressed HRV", "reference_interval": ">= 25 ms", "position_pct": min(100.0, (rmssd / 100.0) * 100.0)}
        ],
        "arrhythmic_beat_count": arrhythmic_count,
        "total_beats": total_beats,
        "rhythm_strip_samples": rhythm_strip,
        "r_peak_indices": strip_peaks,
        "top_risk_factors": [
            {"feature_name": "R-R Prematurity Index", "value": f"{arrhythmia_pct}% ectopic", "severity": "abnormal" if is_pos else "normal", "clinical_context": "Coupling interval premature beat clustering"}
        ]
    }
