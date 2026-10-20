"""
MediNova Unified Clinical Inference Service
============================================
Provides a decoupled, robust inference dispatch pipeline across all diagnostic conditions.

Workflow per Request:
1. Validates condition registration and governance status (rejects research previews)
2. Validates input schema and physiological bounds via clinical_validation
3. Loads model artifact securely via model_registry with integrity checks
4. Executes model inference with execution timer
5. Records latency and success telemetry into model_monitoring
6. Returns normalized clinical screening dictionary
"""

import time
import logging
from typing import Dict, Any, Tuple, Optional

from utils.model_registry import (
    get_model_spec,
    load_model_artifact,
    verify_model_integrity
)
from utils.clinical_validation import (
    validate_clinical_inputs,
    check_out_of_distribution
)
from utils.model_monitoring import record_inference_telemetry
from utils.diagnostic_predictor import (
    predict_heart,
    predict_diabetes,
    predict_kidney,
    predict_liver,
    predict_breast_cancer,
    predict_thyroid,
    predict_arrhythmia
)

logger = logging.getLogger("medinova.inference")


def run_clinical_screening(
    condition_id: str,
    payload: Dict[str, Any],
    disease_insights: Dict[str, Any],
    is_scanner: bool = False
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Unified entry point for CDSS inference.
    
    Returns:
        (is_success, result_dict, error_message)
    """
    start_time = time.perf_counter()
    spec = get_model_spec(condition_id)
    
    if not spec:
        err = f"Unknown condition identifier: '{condition_id}'."
        record_inference_telemetry(condition_id, 0.0, False, err)
        return False, {}, err

    # Governance check: Fail closed if module is in Research Preview
    status = spec.get("status", "available")
    if status != "available":
        err = (
            f"The '{spec.get('condition_name', condition_id)}' module is currently in "
            f"'{status.upper()}' under MediNova clinical governance. Direct inference is disabled."
        )
        record_inference_telemetry(condition_id, 0.0, False, err)
        return False, {"is_preview": True, "condition_name": spec.get("condition_name")}, err

    # Load model artifact securely
    model_obj, load_err = load_model_artifact(condition_id)
    if load_err or model_obj is None:
        record_inference_telemetry(condition_id, 0.0, False, load_err)
        return False, {}, f"Model initialization failed: {load_err}"

    modality = spec.get("modality", "tabular")

    try:
        if condition_id == "heart_disease":
            # Tabular Validation
            required = ['age', 'sex', 'cp', 'bp', 'chol', 'fbs', 'ekg', 'max_hr', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            valid, clean_data, val_errs = validate_clinical_inputs(payload, required)
            if not valid:
                err = " | ".join(val_errs)
                record_inference_telemetry(condition_id, (time.perf_counter() - start_time) * 1000.0, False, err)
                return False, {}, err
            res = predict_heart(clean_data, model_obj, disease_insights, is_scanner)

        elif condition_id == "diabetes":
            required = ['age', 'gender', 'bmi', 'HbA1c_level', 'blood_glucose_level']
            # Numeric validation for age, bmi, hba1c, glucose
            num_req = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
            valid, clean_data, val_errs = validate_clinical_inputs(payload, num_req)
            if not valid:
                err = " | ".join(val_errs)
                record_inference_telemetry(condition_id, (time.perf_counter() - start_time) * 1000.0, False, err)
                return False, {}, err
            # Merge non-numeric fields
            clean_data['gender'] = payload.get('gender', 'Female')
            clean_data['smoking_history'] = payload.get('smoking_history', 'never')
            clean_data['hypertension'] = int(payload.get('hypertension', 0))
            clean_data['heart_disease'] = int(payload.get('heart_disease', 0))
            res = predict_diabetes(clean_data, model_obj, disease_insights, is_scanner)

        elif condition_id == "kidney_disease":
            required = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
                        'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm',
                        'cad', 'appet', 'pe', 'ane']
            valid, clean_data, val_errs = validate_clinical_inputs(payload, required)
            if not valid:
                err = " | ".join(val_errs)
                record_inference_telemetry(condition_id, (time.perf_counter() - start_time) * 1000.0, False, err)
                return False, {}, err
            res = predict_kidney(clean_data, model_obj, disease_insights, is_scanner)

        elif condition_id == "liver_disease":
            required = ['Age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                        'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
                        'Albumin', 'Albumin_and_Globulin_Ratio']
            valid, clean_data, val_errs = validate_clinical_inputs(payload, required)
            if not valid:
                err = " | ".join(val_errs)
                record_inference_telemetry(condition_id, (time.perf_counter() - start_time) * 1000.0, False, err)
                return False, {}, err
            clean_data['Gender'] = payload.get('Gender', 'Male')
            res = predict_liver(clean_data, model_obj, disease_insights, is_scanner)

        elif condition_id == "breast_cancer":
            required = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']
            valid, clean_data, val_errs = validate_clinical_inputs(payload, required)
            if not valid:
                err = " | ".join(val_errs)
                record_inference_telemetry(condition_id, (time.perf_counter() - start_time) * 1000.0, False, err)
                return False, {}, err
            res = predict_breast_cancer(clean_data, model_obj, disease_insights, is_scanner)

        elif condition_id == "thyroid":
            required = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
            valid, clean_data, val_errs = validate_clinical_inputs(payload, required)
            if not valid:
                err = " | ".join(val_errs)
                record_inference_telemetry(condition_id, (time.perf_counter() - start_time) * 1000.0, False, err)
                return False, {}, err
            clean_data['sex'] = payload.get('sex', payload.get('gender', 0))
            clean_data['on_thyroxine'] = payload.get('on_thyroxine', 0)
            res = predict_thyroid(clean_data, model_obj, disease_insights, is_scanner)

        elif condition_id == "arrhythmia":
            signal_input = payload.get("signal", payload)
            sample_rate = float(payload.get("sample_rate", 360.0))
            res = predict_arrhythmia(signal_input, model_obj, disease_insights, sample_rate)

        else:
            err = f"No predictor handler implemented for '{condition_id}'."
            record_inference_telemetry(condition_id, 0.0, False, err)
            return False, {}, err

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        record_inference_telemetry(condition_id, latency_ms, True)
        
        # Add common metadata contract
        res["model_version"] = spec.get("version", "1.0.0")
        res["model_modality"] = modality
        res["specialty_routing"] = spec.get("specialty_routing", "General Medicine")
        res["latency_ms"] = round(latency_ms, 1)
        
        return True, res, None

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        err_msg = str(e)
        logger.error(f"[INFERENCE ERROR] Error executing {condition_id}: {err_msg}")
        record_inference_telemetry(condition_id, latency_ms, False, err_msg)
        return False, {}, err_msg
