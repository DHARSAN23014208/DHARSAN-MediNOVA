"""
MediNova Clinical Model Explanations & Biomarker Gauges
=======================================================
Generates evidence-grounded interpretability summaries for clinical decision support.

Design Governance:
1. No Pseudo-Exact Attribution: Feature importances reflect actual trained global tree weights
2. Reference Benchmarking: Compares patient measurements to established clinical intervals
3. Modality-Specific Context: Distinguishes biochemical analytes from biosignal intervals and pixel activations
4. Clinical Caution: Explicitly disclaims that feature weights do not establish medical etiology or certainty
"""

import numpy as np
from typing import Dict, Any, List, Optional

# Standard Clinical Reference Ranges and Physiological Benchmarks
CLINICAL_BENCHMARKS = {
    # Thyroid Panel
    "TSH": {
        "name": "Thyroid-Stimulating Hormone",
        "unit": "mIU/L",
        "optimal": (0.4, 4.0),
        "borderline": (4.1, 10.0),
        "max_disp": 25.0,
        "clinical_role": "Primary pituitaro-thyroid feedback regulator. Elevated TSH indicates primary/subclinical thyroid gland failure."
    },
    "T3": {
        "name": "Total Triiodothyronine",
        "unit": "ng/mL",
        "optimal": (0.8, 2.0),
        "borderline": (0.6, 0.79),
        "max_disp": 4.0,
        "low_is_bad": True,
        "clinical_role": "Most biologically active thyroid hormone. Subnormal levels corroborate advanced hypothyroidism."
    },
    "TT4": {
        "name": "Total Thyroxine",
        "unit": "µg/dL",
        "optimal": (4.5, 12.0),
        "borderline": (3.5, 4.4),
        "max_disp": 20.0,
        "low_is_bad": True,
        "clinical_role": "Primary prohormone synthesized by thyroid follicular cells. Low levels signify diminished gland output."
    },
    "T4U": {
        "name": "T4 Uptake Ratio",
        "unit": "",
        "optimal": (0.8, 1.3),
        "borderline": (1.31, 1.5),
        "max_disp": 2.5,
        "clinical_role": "Indirect measure of thyroxine-binding globulin (TBG) carrier protein saturation."
    },
    "FTI": {
        "name": "Free Thyroxine Index",
        "unit": "",
        "optimal": (1.2, 4.2),
        "borderline": (0.8, 1.19),
        "max_disp": 6.0,
        "low_is_bad": True,
        "clinical_role": "Normalized estimation of circulating unbound, bioavailable thyroxine independent of carrier proteins."
    },
    
    # ECG Arrhythmia Biosignal Features
    "rr_pre_ratio": {
        "name": "Pre-Ectopic RR Prematurity Ratio",
        "unit": "",
        "optimal": (0.85, 1.15),
        "borderline": (0.70, 0.84),
        "max_disp": 2.0,
        "clinical_role": "Coupling interval to preceding beat. Ratio < 0.80 indicates premature ectopic contraction."
    },
    "qrs_amplitude": {
        "name": "QRS Voltage Deflection",
        "unit": "mV",
        "optimal": (0.5, 2.5),
        "borderline": (2.51, 3.5),
        "max_disp": 5.0,
        "clinical_role": "Ventricular depolarization amplitude. Wide, bizarre high-voltage QRS is characteristic of ventricular ectopy."
    },
    "beat_kurtosis": {
        "name": "Waveform Kurtosis Moment",
        "unit": "",
        "optimal": (1.5, 5.0),
        "borderline": (5.1, 9.0),
        "max_disp": 15.0,
        "clinical_role": "Peakedness and tail heaviness of the beat waveform distribution."
    },
    "st_level": {
        "name": "ST Segment Deviation",
        "unit": "mV",
        "optimal": (-0.1, 0.1),
        "borderline": (0.11, 0.2),
        "max_disp": 0.5,
        "clinical_role": "Post-depolarization isoelectric level reflecting ventricular repolarization and ischemic deviation."
    },

    # Metabolic
    "HbA1c_level": {"name": "HbA1c Level", "unit": "%", "optimal": (4.0, 5.6), "borderline": (5.7, 6.4), "max_disp": 14.0},
    "blood_glucose_level": {"name": "Blood Glucose", "unit": "mg/dL", "optimal": (70.0, 99.0), "borderline": (100.0, 125.0), "max_disp": 300.0},
    "bmi": {"name": "Body Mass Index", "unit": "kg/m²", "optimal": (18.5, 24.9), "borderline": (25.0, 29.9), "max_disp": 50.0},

    # Cardiovascular
    "bp": {"name": "Resting Blood Pressure", "unit": "mm Hg", "optimal": (90.0, 119.0), "borderline": (120.0, 139.0), "max_disp": 200.0},
    "chol": {"name": "Serum Cholesterol", "unit": "mg/dL", "optimal": (120.0, 199.0), "borderline": (200.0, 239.0), "max_disp": 400.0},
    "max_hr": {"name": "Maximum Heart Rate", "unit": "bpm", "optimal": (100.0, 175.0), "borderline": (75.0, 99.0), "max_disp": 220.0},
    "oldpeak": {"name": "ST Depression (Oldpeak)", "unit": "mm", "optimal": (0.0, 0.9), "borderline": (1.0, 1.9), "max_disp": 6.0},

    # Renal
    "sc": {"name": "Serum Creatinine", "unit": "mg/dL", "optimal": (0.6, 1.2), "borderline": (1.3, 1.5), "max_disp": 6.0},
    "bu": {"name": "Blood Urea", "unit": "mg/dL", "optimal": (15.0, 45.0), "borderline": (46.0, 60.0), "max_disp": 150.0},
    "hemo": {"name": "Hemoglobin", "unit": "g/dL", "optimal": (12.0, 16.5), "borderline": (10.0, 11.9), "max_disp": 20.0, "low_is_bad": True},
    "sod": {"name": "Serum Sodium", "unit": "mEq/L", "optimal": (135.0, 145.0), "borderline": (130.0, 134.0), "max_disp": 160.0},
    "pot": {"name": "Serum Potassium", "unit": "mEq/L", "optimal": (3.5, 5.0), "borderline": (5.1, 5.5), "max_disp": 8.0},

    # Hepatic
    "Total_Bilirubin": {"name": "Total Bilirubin", "unit": "mg/dL", "optimal": (0.2, 1.2), "borderline": (1.3, 2.0), "max_disp": 10.0},
    "Direct_Bilirubin": {"name": "Direct Bilirubin", "unit": "mg/dL", "optimal": (0.0, 0.3), "borderline": (0.4, 0.8), "max_disp": 6.0},
    "Alkaline_Phosphotase": {"name": "Alkaline Phosphatase", "unit": "U/L", "optimal": (30.0, 120.0), "borderline": (121.0, 180.0), "max_disp": 600.0},
    "Alamine_Aminotransferase": {"name": "Alanine Aminotransferase (ALT)", "unit": "U/L", "optimal": (7.0, 56.0), "borderline": (57.0, 90.0), "max_disp": 300.0},
    "Aspartate_Aminotransferase": {"name": "Aspartate Aminotransferase (AST)", "unit": "U/L", "optimal": (10.0, 40.0), "borderline": (41.0, 75.0), "max_disp": 300.0},
    "Albumin": {"name": "Serum Albumin", "unit": "g/dL", "optimal": (3.4, 5.4), "borderline": (3.0, 3.3), "max_disp": 6.0, "low_is_bad": True},
    "Albumin_and_Globulin_Ratio": {"name": "A/G Ratio", "unit": "", "optimal": (1.0, 2.2), "borderline": (0.8, 0.99), "max_disp": 3.0, "low_is_bad": True},

    # Breast Cancer (FNA Nuclear)
    "mean_radius": {"name": "Nuclear Radius", "unit": "µm", "optimal": (6.0, 13.9), "borderline": (14.0, 16.0), "max_disp": 28.0},
    "mean_perimeter": {"name": "Nuclear Perimeter", "unit": "µm", "optimal": (40.0, 89.9), "borderline": (90.0, 105.0), "max_disp": 180.0},
    "mean_area": {"name": "Nuclear Area", "unit": "µm²", "optimal": (140.0, 499.0), "borderline": (500.0, 699.0), "max_disp": 1500.0},
    "mean_smoothness": {"name": "Nuclear Smoothness", "unit": "", "optimal": (0.06, 0.099), "borderline": (0.10, 0.12), "max_disp": 0.20}
}


def build_clinical_gauges(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Constructs physiological gauges comparing patient values against medical benchmarks."""
    gauges = []
    for key, spec in CLINICAL_BENCHMARKS.items():
        if key in data and data[key] is not None:
            try:
                val = float(data[key])
                opt_min, opt_max = spec["optimal"]
                bord_min, bord_max = spec["borderline"]
                max_d = spec.get("max_disp", opt_max * 2.0)
                low_bad = spec.get("low_is_bad", False)

                if opt_min <= val <= opt_max:
                    status = "optimal"
                    status_lbl = "Optimal Range"
                elif (bord_min <= val <= bord_max) or (low_bad and val < opt_min and val >= bord_min):
                    status = "borderline"
                    status_lbl = "Borderline / Clinical Review"
                else:
                    status = "elevated"
                    status_lbl = "Clinically Elevated" if not low_bad else "Clinically Deficient"

                pos_pct = min(100.0, max(0.0, (val / max(1.0, max_d)) * 100.0))
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
                    "high": opt_max,
                    "clinical_role": spec.get("clinical_role", "")
                })
            except (ValueError, TypeError):
                continue
    return gauges


def extract_model_contributions(
    model_or_weights: Any,
    feature_names: List[str],
    patient_vector: List[float],
    top_k: int = 4
) -> List[Dict[str, Any]]:
    """
    Extracts top contributing clinical features using actual model feature importances.
    Avoids pseudoscientific causality claims by providing clear clinical context.
    """
    weights = None
    if isinstance(model_or_weights, dict):
        weights = [model_or_weights.get(fn, 0.0) for fn in feature_names]
    elif hasattr(model_or_weights, "feature_importances_"):
        weights = list(model_or_weights.feature_importances_)
    elif hasattr(model_or_weights, "named_steps"):
        # Pipeline
        clf = model_or_weights.named_steps.get("classifier")
        if hasattr(clf, "feature_importances_"):
            weights = list(clf.feature_importances_)

    if not weights or len(weights) != len(feature_names):
        return []

    ranked = np.argsort(weights)[::-1]
    factors = []

    for idx in ranked[:top_k]:
        fname = feature_names[idx]
        w = float(weights[idx])
        val = patient_vector[idx]

        spec = CLINICAL_BENCHMARKS.get(fname)
        if spec:
            opt_min, opt_max = spec["optimal"]
            is_atypical = (val < opt_min) or (val > opt_max)
            severity = "abnormal" if is_atypical else "normal"
            bench = f"Normal Target: {opt_min} - {opt_max} {spec['unit']}".strip()
            lbl = spec["name"]
        else:
            severity = "informative"
            bench = "Standard Parameter Target"
            lbl = fname.replace("_", " ").title()

        val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
        ctx = f"{bench} (Observed: {val_str})"
        f_pct = round(w * 100.0, 1)

        factors.append({
            "feature": fname,
            "feature_name": lbl,
            "label": lbl,
            "value": val_str,
            "weight": round(w, 3),
            "global_sensitivity_weight": round(w, 3),
            "benchmark": bench,
            "clinical_context": ctx,
            "severity": severity,
            "percent": f_pct,
            "contribution_percent": f_pct,
            "importance": round(w, 3)
        })

    return factors
