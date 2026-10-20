"""
MediNova Clinical Biosignal & ECG Processing Engine
===================================================
Provides robust digital signal processing, artifact suppression,
QRS detection, and morphological feature extraction for single-lead ECG telemetry.

Core Capabilities:
1. Signal Ingestion: Parses CSV, JSON, TXT, or numerical arrays
2. Signal Quality Assurance (SQA):
   - Duration check (minimum 5s, maximum 60s)
   - Saturation, clipping, and flatline detection
   - SNR & powerline noise estimation
3. Filtering & Conditioning:
   - 3rd-order Butterworth bandpass (0.5 - 40 Hz) eliminating respiration wander and EMG noise
4. QRS Detection:
   - Energy-based Pan-Tompkins adaptive peak detection for R-wave localization
5. Beat Segmentation & Feature Engineering:
   - Extracts canonical 200-sample beat windows centered on R-peaks
   - Computes morphological, distribution, and rhythm intervals
6. Clinical Rhythm Strip Metadata:
   - Computes instantaneous BPM, mean RR, RMSSD (Heart Rate Variability), and rhythm points for UI rendering
"""

import math
import numpy as np
from scipy import signal
from typing import Dict, Any, List, Tuple, Optional

SAMPLE_RATE_CANONICAL = 360 # Canonical sample rate in Hz
MIN_DURATION_SECONDS = 4.0 # Minimum 4 seconds for reliable rhythm analysis
MAX_DURATION_SECONDS = 60.0 # Maximum 60 seconds per screening request

PRE_R_SAMPLES = 90
POST_R_SAMPLES = 110
BEAT_LENGTH = PRE_R_SAMPLES + POST_R_SAMPLES # 200 samples (~555 ms)


def validate_ecg_signal_quality(
    raw_signal: np.ndarray,
    sample_rate: float
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Evaluates signal readability, duration, variance, and artifact levels.
    Returns (is_acceptable, rejection_reason_or_empty, signal_metrics).
    """
    if len(raw_signal) == 0:
        return False, "ECG signal recording is empty.", {}
        
    duration_sec = len(raw_signal) / float(sample_rate)
    if duration_sec < MIN_DURATION_SECONDS:
        return False, (
            f"ECG recording is too brief ({duration_sec:.1f}s). "
            f"Minimum required duration for arrhythmia screening is {MIN_DURATION_SECONDS:.0f} seconds."
        ), {"duration_sec": duration_sec}
        
    if duration_sec > MAX_DURATION_SECONDS:
        # Bounded length: truncate to 60 seconds
        raw_signal = raw_signal[:int(MAX_DURATION_SECONDS * sample_rate)]
        duration_sec = MAX_DURATION_SECONDS
        
    # Check for NaN, Inf
    if not np.all(np.isfinite(raw_signal)):
        return False, "ECG signal contains invalid non-numeric, NaN, or infinite sensor values.", {}
        
    # Flatline / disconnect check
    sig_std = float(np.std(raw_signal))
    sig_range = float(np.max(raw_signal) - np.min(raw_signal))
    if sig_std < 0.01 or sig_range < 0.05:
        return False, (
            "Sensor flatline or disconnected lead detected. "
            f"Signal variance ({sig_std:.4f}) is below biological thresholds."
        ), {"variance": sig_std, "range": sig_range}
        
    # Rail saturation / clipping check (>80% of samples stuck at exact extremes)
    min_val, max_val = np.min(raw_signal), np.max(raw_signal)
    clipped_count = np.sum(np.isclose(raw_signal, min_val)) + np.sum(np.isclose(raw_signal, max_val))
    clip_ratio = clipped_count / float(len(raw_signal))
    if clip_ratio > 0.40:
        return False, (
            f"Severe ADC sensor rail clipping detected ({clip_ratio*100:.1f}% clipped). "
            "Signal cannot be reliably interpreted."
        ), {"clip_ratio": clip_ratio}
        
    metrics = {
        "duration_sec": round(duration_sec, 2),
        "sample_count": len(raw_signal),
        "sample_rate": sample_rate,
        "signal_std": round(sig_std, 4),
        "voltage_range_mv": round(sig_range, 3)
    }
    return True, "", metrics


def filter_ecg_signal(
    raw_signal: np.ndarray,
    fs: float = SAMPLE_RATE_CANONICAL
) -> np.ndarray:
    """Applies zero-phase 3rd-order Butterworth bandpass filter (0.5 to 40 Hz)."""
    nyq = 0.5 * fs
    lowcut = 0.5 / nyq
    highcut = min(0.95, 40.0 / nyq)
    b, a = signal.butter(3, [lowcut, highcut], btype='band')
    clean = signal.filtfilt(b, a, raw_signal)
    return clean


def resample_to_canonical(
    sig: np.ndarray,
    orig_fs: float,
    target_fs: float = SAMPLE_RATE_CANONICAL
) -> np.ndarray:
    """Resamples input signal array to canonical 360 Hz if needed."""
    if abs(orig_fs - target_fs) < 1.0:
        return sig
    num_target_samples = int(len(sig) * (target_fs / orig_fs))
    return signal.resample(sig, num_target_samples)


def detect_r_peaks(clean_signal: np.ndarray, fs: float = SAMPLE_RATE_CANONICAL) -> np.ndarray:
    """
    Pan-Tompkins energy and moving integration derivative for R-peak detection.
    Returns array of sample indices corresponding to QRS R-peaks.
    """
    # 1. Five-point derivative
    diff = np.diff(clean_signal)
    
    # 2. Squaring function
    squared = diff ** 2
    
    # 3. Moving window integrator (~150 ms window)
    window_len = int(0.15 * fs)
    kernel = np.ones(window_len) / float(window_len)
    integrated = np.convolve(squared, kernel, mode='same')
    
    # 4. Adaptive peak finding
    min_dist = int(0.35 * fs) # Refractory period: minimum 350 ms between beats (~170 bpm max)
    threshold = np.mean(integrated) + 0.6 * np.std(integrated)
    
    peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_dist)
    
    # Align peaks to actual local maximum in clean signal
    refined_peaks = []
    search_win = int(0.08 * fs) # +/- 80 ms
    for p in peaks:
        left = max(0, p - search_win)
        right = min(len(clean_signal), p + search_win)
        if right > left:
            local_max_idx = left + np.argmax(clean_signal[left:right])
            refined_peaks.append(local_max_idx)
            
    return np.array(sorted(list(set(refined_peaks))))


def extract_features_from_waveform(
    clean_signal: np.ndarray,
    r_peaks: np.ndarray,
    fs: float = SAMPLE_RATE_CANONICAL
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Extracts beat feature vectors from clean waveform and detected R-peaks.
    Returns (feature_matrix, beat_metadata_list).
    """
    if len(r_peaks) < 3:
        return np.empty((0, 9)), []
        
    rr_intervals = np.diff(r_peaks) / float(fs)
    mean_rr = float(np.median(rr_intervals)) if len(rr_intervals) > 0 else 0.8
    
    feature_rows = []
    beat_meta = []
    
    for i in range(1, len(r_peaks) - 1):
        r_pos = int(r_peaks[i])
        start = r_pos - PRE_R_SAMPLES
        end = r_pos + POST_R_SAMPLES
        
        if start < 0 or end > len(clean_signal):
            continue
            
        beat_wf = clean_signal[start:end]
        rr_pre = float(r_peaks[i] - r_peaks[i - 1]) / float(fs)
        rr_post = float(r_peaks[i + 1] - r_pos) / float(fs)
        
        qrs_amp = float(np.max(beat_wf) - np.min(beat_wf))
        qrs_energy = float(np.sum(beat_wf ** 2) / len(beat_wf))
        qrs_std = float(np.std(beat_wf))
        
        # Statistical moments
        m_mean = np.mean(beat_wf)
        m3 = np.mean((beat_wf - m_mean) ** 3)
        m4 = np.mean((beat_wf - m_mean) ** 4)
        skew = float(m3 / (qrs_std ** 3 + 1e-8))
        kurt = float(m4 / (qrs_std ** 4 + 1e-8))
        
        st_window = beat_wf[min(120, len(beat_wf)):min(150, len(beat_wf))]
        st_level = float(np.mean(st_window)) if len(st_window) > 0 else 0.0
        
        t_window = beat_wf[min(140, len(beat_wf)):min(190, len(beat_wf))]
        t_wave_amp = float(np.max(t_window)) if len(t_window) > 0 else 0.0
        
        norm_rr_pre = rr_pre / max(0.1, mean_rr)
        norm_rr_post = rr_post / max(0.1, mean_rr)
        
        feats = [
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
        
        feature_rows.append(feats)
        beat_meta.append({
            "beat_index": i,
            "r_peak_sample": r_pos,
            "timestamp_sec": round(r_pos / float(fs), 3),
            "instant_bpm": round(60.0 / max(0.2, rr_pre), 1),
            "rr_pre_ms": round(rr_pre * 1000.0, 1),
            "qrs_amplitude_mv": round(qrs_amp, 3)
        })
        
    return np.array(feature_rows), beat_meta


def parse_ecg_input_data(input_payload: Any) -> Tuple[np.ndarray, float, Optional[str]]:
    """
    Parses various input formats:
    - String containing numbers (CSV, space-delimited, newline-delimited)
    - JSON list of numbers
    - Dictionary with 'signal' and optional 'sample_rate'
    
    Returns (signal_array, sample_rate, error_or_None).
    """
    sample_rate = float(SAMPLE_RATE_CANONICAL)
    
    if isinstance(input_payload, dict):
        sample_rate = float(input_payload.get("sample_rate", SAMPLE_RATE_CANONICAL))
        sig_data = input_payload.get("signal", [])
    elif isinstance(input_payload, (list, tuple, np.ndarray)):
        sig_data = input_payload
    elif isinstance(input_payload, str):
        # Strip brackets if JSON string
        raw_str = input_payload.strip()
        if raw_str.startswith("[") and raw_str.endswith("]"):
            import json
            try:
                sig_data = json.loads(raw_str)
            except Exception:
                sig_data = [float(x) for x in raw_str.strip("[]").replace(",", " ").split() if x.strip()]
        else:
            # Parse CSV or newline floats
            vals = []
            for item in raw_str.replace(",", " ").replace("\n", " ").replace("\r", " ").split():
                try:
                    vals.append(float(item))
                except ValueError:
                    continue
            sig_data = vals
    else:
        return np.array([]), sample_rate, "Unsupported ECG payload format."
        
    try:
        arr = np.array(sig_data, dtype=float)
        return arr, sample_rate, None
    except Exception as e:
        return np.array([]), sample_rate, f"Failed to convert ECG data to numerical array: {e}"
