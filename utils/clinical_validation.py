"""
MediNova Clinical Input Validation & OOD Safety Gateway
========================================================
Validates all patient telemetry, laboratory measurements, and biosignals
against strict clinical plausibility ranges and out-of-distribution (OOD) bounds.

Core Principles:
1. Strict Typology: Rejects non-numeric, malformed, or injected strings
2. Physiological Bounds: Detects values outside human biological limits (e.g. negative BP, HR > 250)
3. Out-of-Distribution (OOD) Detection: Flags extreme multivariable outliers to trigger CDSS abstention
4. Evidence-Based Normalization: Cleans units and formats consistently between training and inference
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Physiological Plausibility Envelopes
# (Values outside min_plausible..max_plausible are biologically impossible or fatal artifacts)
PHYSIOLOGICAL_BOUNDS: Dict[str, Dict[str, Any]] = {
    # Vitals & Demographics
    "age": {"min": 0.0, "max": 120.0, "unit": "years", "name": "Patient Age"},
    "bp": {"min": 40.0, "max": 260.0, "unit": "mm Hg", "name": "Resting Blood Pressure"},
    "max_hr": {"min": 35.0, "max": 240.0, "unit": "bpm", "name": "Maximum Heart Rate"},
    "bmi": {"min": 10.0, "max": 80.0, "unit": "kg/m²", "name": "Body Mass Index"},
    
    # Glycemic
    "blood_glucose_level": {"min": 20.0, "max": 600.0, "unit": "mg/dL", "name": "Blood Glucose Level"},
    "HbA1c_level": {"min": 3.0, "max": 18.0, "unit": "%", "name": "Glycated Hemoglobin (HbA1c)"},
    "bgr": {"min": 20.0, "max": 600.0, "unit": "mg/dL", "name": "Random Blood Glucose"},
    
    # Cardiovascular
    "chol": {"min": 50.0, "max": 600.0, "unit": "mg/dL", "name": "Serum Cholesterol"},
    "oldpeak": {"min": -2.0, "max": 10.0, "unit": "mm", "name": "ST Depression (Oldpeak)"},
    
    # Renal
    "sc": {"min": 0.1, "max": 25.0, "unit": "mg/dL", "name": "Serum Creatinine"},
    "bu": {"min": 2.0, "max": 250.0, "unit": "mg/dL", "name": "Blood Urea"},
    "hemo": {"min": 2.0, "max": 25.0, "unit": "g/dL", "name": "Hemoglobin"},
    "sod": {"min": 90.0, "max": 180.0, "unit": "mEq/L", "name": "Serum Sodium"},
    "pot": {"min": 1.5, "max": 10.0, "unit": "mEq/L", "name": "Serum Potassium"},
    "sg": {"min": 1.000, "max": 1.040, "unit": "", "name": "Specific Gravity"},
    "pcv": {"min": 10.0, "max": 65.0, "unit": "%", "name": "Packed Cell Volume"},
    
    # Hepatic
    "Total_Bilirubin": {"min": 0.1, "max": 40.0, "unit": "mg/dL", "name": "Total Bilirubin"},
    "Direct_Bilirubin": {"min": 0.0, "max": 30.0, "unit": "mg/dL", "name": "Direct Bilirubin"},
    "Alkaline_Phosphotase": {"min": 10.0, "max": 2000.0, "unit": "U/L", "name": "Alkaline Phosphatase"},
    "Alamine_Aminotransferase": {"min": 2.0, "max": 1500.0, "unit": "U/L", "name": "Alanine Aminotransferase (ALT)"},
    "Aspartate_Aminotransferase": {"min": 2.0, "max": 1500.0, "unit": "U/L", "name": "Aspartate Aminotransferase (AST)"},
    "Total_Protiens": {"min": 2.0, "max": 15.0, "unit": "g/dL", "name": "Total Proteins"},
    "Albumin": {"min": 1.0, "max": 7.0, "unit": "g/dL", "name": "Serum Albumin"},
    "Albumin_and_Globulin_Ratio": {"min": 0.1, "max": 5.0, "unit": "", "name": "A/G Ratio"},
    
    # Thyroid
    "TSH": {"min": 0.001, "max": 250.0, "unit": "mIU/L", "name": "Thyroid-Stimulating Hormone"},
    "T3": {"min": 0.05, "max": 15.0, "unit": "nmol/L", "name": "Triiodothyronine (T3)"},
    "TT4": {"min": 5.0, "max": 350.0, "unit": "nmol/L", "name": "Total Thyroxine (TT4)"},
    "T4U": {"min": 0.2, "max": 3.0, "unit": "", "name": "T4 Uptake Ratio"},
    "FTI": {"min": 5.0, "max": 400.0, "unit": "", "name": "Free Thyroxine Index"},
    
    # Breast FNA Nuclear
    "mean_radius": {"min": 5.0, "max": 35.0, "unit": "µm", "name": "Mean Nuclear Radius"},
    "mean_texture": {"min": 5.0, "max": 45.0, "unit": "", "name": "Mean Texture"},
    "mean_perimeter": {"min": 35.0, "max": 250.0, "unit": "µm", "name": "Mean Perimeter"},
    "mean_area": {"min": 100.0, "max": 3000.0, "unit": "µm²", "name": "Mean Area"},
    "mean_smoothness": {"min": 0.04, "max": 0.25, "unit": "", "name": "Mean Smoothness"}
}


def validate_clinical_inputs(data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, Dict[str, float], List[str]]:
    """
    Validates that all required fields are present, parses them to floats,
    and enforces physiological bounds.
    
    Returns:
        (is_valid, parsed_clean_data, error_messages)
    """
    clean_data: Dict[str, float] = {}
    errors: List[str] = []
    
    for field in required_fields:
        if field not in data or data[field] is None or str(data[field]).strip() == "":
            errors.append(f"Missing mandatory clinical parameter: '{field}'.")
            continue
            
        raw_val = data[field]
        try:
            val = float(raw_val)
            if math.isnan(val) or math.isinf(val):
                errors.append(f"Invalid non-numeric value for '{field}'.")
                continue
        except (ValueError, TypeError):
            errors.append(f"Field '{field}' must be a valid numerical value (received '{raw_val}').")
            continue
            
        # Check physiological plausibility
        bounds = PHYSIOLOGICAL_BOUNDS.get(field)
        if bounds:
            b_min, b_max = bounds["min"], bounds["max"]
            if val < b_min or val > b_max:
                errors.append(
                    f"Parameter '{bounds['name']}' ({val} {bounds['unit']}) is outside physiologically "
                    f"plausible limits ({b_min} - {b_max} {bounds['unit']}). Please verify laboratory units."
                )
                continue
                
        clean_data[field] = val
        
    is_valid = len(errors) == 0
    return is_valid, clean_data, errors


def check_out_of_distribution(vector: List[float], feature_names: List[str]) -> Tuple[bool, float, List[str]]:
    """
    Evaluates whether a patient's biomarker vector exhibits extreme out-of-distribution (OOD)
    characteristics that make statistical inference unreliable.
    
    Returns:
        (is_ood, max_deviation_z, anomalous_features)
    """
    anomalous = []
    max_z = 0.0
    
    for i, name in enumerate(feature_names):
        if i >= len(vector):
            break
        val = vector[i]
        bounds = PHYSIOLOGICAL_BOUNDS.get(name)
        if bounds:
            b_min = bounds["min"]
            b_max = bounds["max"]
            center = (b_min + b_max) / 2.0
            spread = (b_max - b_min) / 4.0 # approximate pseudo-std dev
            z = abs(val - center) / max(0.01, spread)
            if z > max_z:
                max_z = z
            if z > 3.5:
                anomalous.append(f"{bounds['name']} (Extreme value: {val} {bounds['unit']})")
                
    is_ood = max_z > 4.0 or len(anomalous) >= 3
    return is_ood, round(max_z, 2), anomalous


def validate_clinical_features(data: Dict[str, Any], condition: str) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Validates patient feature payload against clinical bounds, returning (clean_vals, errors, warnings)."""
    clean_vals: Dict[str, Any] = {}
    errors: List[str] = []
    warnings: List[str] = []
    
    for k, v in data.items():
        bounds = PHYSIOLOGICAL_BOUNDS.get(k)
        if bounds:
            try:
                fv = float(v)
                if fv < bounds["min"] or fv > bounds["max"]:
                    errors.append(f"Field '{k}' ({fv} {bounds['unit']}) is outside plausible range ({bounds['min']} - {bounds['max']})")
                else:
                    clean_vals[k] = fv
            except (ValueError, TypeError):
                errors.append(f"Invalid non-numeric value for '{k}': {v}")
        else:
            clean_vals[k] = v
            
    return clean_vals, errors, warnings


def detect_out_of_distribution(features: Dict[str, Any], condition: str) -> Tuple[bool, List[str]]:
    """OOD detection wrapper taking feature dictionary."""
    vec = []
    names = []
    for k, v in features.items():
        try:
            vec.append(float(v))
            names.append(k)
        except (ValueError, TypeError):
            continue
    is_ood, max_z, anomalous = check_out_of_distribution(vec, names)
    return is_ood, anomalous


try:
    from utils.model_explanations import CLINICAL_REFERENCE_RANGES
except ImportError:
    CLINICAL_REFERENCE_RANGES = PHYSIOLOGICAL_BOUNDS

