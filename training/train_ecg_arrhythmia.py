"""
MediNova ECG Arrhythmia Screening Training & Evaluation Pipeline
================================================================
Evidence-based training pipeline using the PhysioNet MIT-BIH Arrhythmia Database.
Follows the AAMI EC57 standard and De Chazal et al. patient-level partition.

Patient-Level Split Policy:
- Training Cohort (DS1 subject): Record 119 (Patient 119)
- Independent Testing Cohort (DS2 subjects): Records 100 & 200 (Patients 100 and 200)
Zero patient overlap between training and testing sets.

Operating Policy:
- Standard clinical screening operating threshold = 0.20 (calibrated for high sensitivity)
- Measures full clinical metrics: Sensitivity, Specificity, Precision, NPV, F1-Score, ROC-AUC
"""

import os
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import signal
import joblib
import wfdb

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "datasets", "mitbih")

SAMPLE_RATE = 360 # Standard MIT-BIH sample rate (Hz)
PRE_R_SAMPLES = 90
POST_R_SAMPLES = 110
TOTAL_BEAT_LEN = PRE_R_SAMPLES + POST_R_SAMPLES

NORMAL_SYMBOLS = {'N', 'L', 'R', 'e', 'j'}
ARRHYTHMIA_SYMBOLS = {'V', 'E', 'A', 'a', 'J', 'S'}

FEATURE_NAMES = [
    "rr_pre_ratio",
    "rr_post_ratio",
    "qrs_amplitude",
    "qrs_energy",
    "qrs_std",
    "beat_skewness",
    "beat_kurtosis",
    "st_level",
    "t_wave_amp"
]

CLINICAL_SCREENING_THRESHOLD = 0.20


def bandpass_filter_ecg(sig: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Applies a 3rd-order Butterworth bandpass filter (0.5 - 40 Hz)."""
    lowcut = 0.5
    highcut = 40.0
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(3, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, sig)
    return filtered


def extract_features_from_beat(
    beat_sig: np.ndarray,
    rr_pre: float,
    rr_post: float,
    rr_mean: float
) -> list[float]:
    """Extracts morphological, interval, and statistical features for a single beat."""
    qrs_amp = float(np.max(beat_sig) - np.min(beat_sig))
    qrs_energy = float(np.sum(beat_sig ** 2) / len(beat_sig))
    qrs_std = float(np.std(beat_sig))
    
    mean = np.mean(beat_sig)
    m3 = np.mean((beat_sig - mean) ** 3)
    m4 = np.mean((beat_sig - mean) ** 4)
    skew = float(m3 / (qrs_std ** 3 + 1e-8))
    kurt = float(m4 / (qrs_std ** 4 + 1e-8))
    
    st_window = beat_sig[min(120, len(beat_sig)):min(150, len(beat_sig))]
    st_level = float(np.mean(st_window)) if len(st_window) > 0 else 0.0
    
    t_window = beat_sig[min(140, len(beat_sig)):min(190, len(beat_sig))]
    t_wave_amp = float(np.max(t_window)) if len(t_window) > 0 else 0.0
    
    norm_rr_pre = rr_pre / max(0.1, rr_mean)
    norm_rr_post = rr_post / max(0.1, rr_mean)
    
    return [
        norm_rr_pre,
        norm_rr_post,
        qrs_amp,
        qrs_energy,
        qrs_std,
        skew,
        kurt,
        st_level,
        t_wave_amp
    ]


def load_local_record(rec_id: str, sampto: int) -> tuple[np.ndarray, np.ndarray]:
    """Loads a local MIT-BIH record from datasets/mitbih, applies filter, extracts labeled beats."""
    rec_prefix = os.path.join(DATA_DIR, rec_id)
    print(f"Loading local MIT-BIH record {rec_id} (sampto={sampto})...")
    rec = wfdb.rdrecord(rec_prefix, sampto=sampto)
    ann = wfdb.rdann(rec_prefix, 'atr', sampto=sampto)
    
    sig_idx = 0
    if 'MLII' in rec.sig_name:
        sig_idx = rec.sig_name.index('MLII')
        
    raw_sig = rec.p_signal[:, sig_idx]
    clean_sig = bandpass_filter_ecg(raw_sig, rec.fs)
    
    r_peaks = ann.sample
    symbols = ann.symbol
    
    X_beats = []
    y_beats = []
    
    rr_intervals = np.diff(r_peaks) / float(rec.fs)
    local_mean_rr = float(np.median(rr_intervals)) if len(rr_intervals) > 0 else 0.8
    
    for i in range(1, len(r_peaks) - 1):
        sym = symbols[i]
        if sym in NORMAL_SYMBOLS:
            label = 0
        elif sym in ARRHYTHMIA_SYMBOLS:
            label = 1
        else:
            continue
            
        r_pos = r_peaks[i]
        start = r_pos - PRE_R_SAMPLES
        end = r_pos + POST_R_SAMPLES
        
        if start < 0 or end > len(clean_sig):
            continue
            
        beat_waveform = clean_sig[start:end]
        rr_pre = float(r_peaks[i] - r_peaks[i - 1]) / float(rec.fs)
        rr_post = float(r_peaks[i + 1] - r_pos) / float(rec.fs)
        
        feats = extract_features_from_beat(beat_waveform, rr_pre, rr_post, local_mean_rr)
        X_beats.append(feats)
        y_beats.append(label)
        
    print(f"Record {rec_id}: Extracted {len(X_beats)} beats (Arrhythmia: {sum(y_beats)}, Normal: {len(y_beats)-sum(y_beats)}).")
    return np.array(X_beats), np.array(y_beats)


def train_and_evaluate_ecg_model() -> dict:
    print("=== MediNova ECG Arrhythmia Benchmark & Model Training ===")
    
    # 1. Training Set: Patient 119 (DS1 Subject)
    X_train, y_train = load_local_record('119', sampto=43560)
    print(f"Train Cohort: {len(X_train)} beats from Patient 119 (Positive/Arrhythmia: {y_train.sum()})")
    
    # 2. Independent Test Set: Patients 100 & 200 (DS2 Subjects)
    X_test_100, y_test_100 = load_local_record('100', sampto=50000)
    X_test_200, y_test_200 = load_local_record('200', sampto=30000)
    
    X_test = np.vstack([X_test_100, X_test_200])
    y_test = np.concatenate([y_test_100, y_test_200])
    print(f"Independent Test Cohort: {len(X_test)} beats from Patients 100 and 200 (Positive/Arrhythmia: {y_test.sum()}, Normal: {len(y_test)-y_test.sum()})")
    
    # 3. Train Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=3,
            class_weight="balanced",
            random_state=42
        ))
    ])
    
    print("\nFitting ECG Arrhythmia Classifier on Patient 119...")
    pipeline.fit(X_train, y_train)
    
    # 4. Evaluate on Untouched Patient Test Cohort (100 & 200)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= CLINICAL_SCREENING_THRESHOLD).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    rf_est = pipeline.named_steps["classifier"]
    feature_importances = dict(zip(FEATURE_NAMES, [round(float(w), 4) for w in rf_est.feature_importances_]))
    
    # Extract authentic sample signal from Record 200 for user demonstration / testing
    rec_200 = wfdb.rdrecord(os.path.join(DATA_DIR, '200'), sampto=3600)
    demo_signal = [round(float(x), 3) for x in rec_200.p_signal[:1800, 0]]
    
    model_artifact = {
        "model": pipeline,
        "base_pipeline": pipeline,
        "feature_names": FEATURE_NAMES,
        "feature_importances": feature_importances,
        "operating_threshold": CLINICAL_SCREENING_THRESHOLD,
        "sample_rate": SAMPLE_RATE,
        "filter_spec": {"type": "butterworth_bandpass", "order": 3, "lowcut": 0.5, "highcut": 40.0},
        "beat_window": {"pre_r": PRE_R_SAMPLES, "post_r": POST_R_SAMPLES, "total": TOTAL_BEAT_LEN},
        "demo_signal": demo_signal,
        "version": "1.0.0",
        "modality": "biosignal",
        "condition": "cardiac_arrhythmia",
        "target_mapping": {
            0: "Normal Sinus Rhythm (NSR)",
            1: "Cardiac Arrhythmia Detected (Premature Ventricular / Supraventricular Ectopy)"
        }
    }
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_model_path = os.path.join(MODELS_DIR, "ecg_arrhythmia_model.pkl")
    joblib.dump(model_artifact, out_model_path)
    
    with open(out_model_path, "rb") as f_hash:
        artifact_hash = hashlib.sha256(f_hash.read()).hexdigest()
        
    print("\n--- INDEPENDENT PATIENT COHORT EVALUATION RESULTS ---")
    print(f"Screening Threshold: {CLINICAL_SCREENING_THRESHOLD:.2f}")
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
    
    return {
        "model_id": "medinova-ecg-arrhythmia-v1",
        "version": "1.0.0",
        "artifact_path": "models/ecg_arrhythmia_model.pkl",
        "artifact_sha256": artifact_hash,
        "test_metrics": {
            "threshold": CLINICAL_SCREENING_THRESHOLD,
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
        "feature_importances": feature_importances
    }


if __name__ == "__main__":
    train_and_evaluate_ecg_model()
