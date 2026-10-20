# Model Card: MediNova Biosignal ECG Arrhythmia Screening Engine

## 1. Model Details
- **Model Identifier:** `medinova-ecg-arrhythmia-v1`
- **Version:** `1.0.0`
- **Model Type:** Preprocessing Pipeline (`StandardScaler` + `RandomForestClassifier(n_estimators=100, class_weight='balanced')`)
- **Developer:** MediNova Healthcare AI & Electrophysiology Informatics
- **Artifact File:** `models/ecg_arrhythmia_model.pkl`
- **SHA-256 Checksum:** `8e9ed74991faf3c4621950e2e370fe7db8ee883f67d4cbad088b7465d0d1d99c`
- **Release Date:** October 2026
- **License:** Open Academic Clinical CDSS (PhysioNet Attribution)

---

## 2. Intended Use & Clinical Scope
- **Primary Clinical Purpose:** Automated biosignal screening of single-lead surface ECG telemetry to detect premature ventricular contractions (PVCs) and cardiac rhythm disturbances.
- **Intended Users:** Ambulatory cardiac telemetry centers, emergency department triage, wearable ECG screening, and outpatient cardiology clinics.
- **Excluded Uses:**
  - Definitive diagnosis of acute ST-elevation myocardial infarction (STEMI) or Brugada syndrome.
  - Automated control of implantable cardioverter-defibrillators (ICDs) or pacemakers.
  - Replacement for comprehensive 12-lead diagnostic electrocardiography interpreted by a board-certified cardiologist.

---

## 3. Signal Processing & Training Pipeline
- **Hardware & Telemetry Contract:**
  - Sample Rate: Canonical 360 Hz (automated polyphase resampling applied if non-360 Hz)
  - Lead Configuration: Modified Lead II (MLII)
  - Window Duration: 4.0 to 60.0 seconds
- **Digital Filter Pipeline:**
  - Baseline wander & high-frequency artifact suppression: 3rd-order digital Butterworth bandpass filter (0.5 – 40 Hz).
  - QRS Localization: Energy-squared Pan-Tompkins adaptive window integrator.
- **Feature Extraction (9 Features per Beat):**
  - Canonical 200-sample beat window (90 samples pre-R, 110 samples post-R).
  - `norm_rr_pre`: Preceding R-R interval normalized by mean heart rate.
  - `norm_rr_post`: Subsequent R-R interval normalized by mean heart rate.
  - `qrs_amp`: Peak-to-peak QRS amplitude (mV).
  - `qrs_energy`: Normalized root-mean-square electrical energy of depolarization.
  - `qrs_std`: Signal standard deviation within QRS window.
  - `skew`: Waveform asymmetry / morphological skewness.
  - `kurt`: Kurtosis (peakedness of ventricular depolarization).
  - `st_level`: Mean amplitude in early repolarization ST segment.
  - `t_wave_amp`: Amplitude of repolarization T-wave.

---

## 4. Strict Patient-Level Split & Evaluation Results
- **Training Set (DS1):** Patient 119 (444 annotated beats).
- **Independent Test Set (DS2):** Patients 100 & 200 (288 annotated beats from completely unseen patients).
- **Zero Leakage:** Complete patient isolation between training and evaluation splits.

Evaluated on **288 independent unseen beats at screening threshold 0.20**:

| Metric | Score | Clinical Interpretation |
|:---|:---|:---|
| **Accuracy** | **99.65%** | Overall beat classification concordance |
| **Sensitivity (Recall)** | **97.78%** | 44 out of 45 ventricular ectopic beats detected |
| **Specificity** | **100.0%** | Zero false alarms on 243 normal sinus beats |
| **Precision (PPV)** | **100.0%** | Perfect positive predictive value on unseen cohort |
| **Negative Predictive Value (NPV)** | **99.59%** | Reliable reassurance when no ectopy is flagged |
| **F1-Score** | **0.9888** | Outstanding trade-off between detection and false alarms |
| **ROC-AUC** | **0.9887** | High discriminative power across variable rhythm burdens |

---

## 5. Telemetry & Hardware Resource Profile
- **Peak Memory Usage:** < 45 MB RAM during active digital filtering.
- **Inference Latency:** ~110 ms for 10 seconds of 360 Hz continuous ECG recording.
- **Safety Abstention:** Flatline, excessive powerline noise (SNR < 3 dB), or signal duration < 4 seconds immediately triggers safe abstention with actionable guidance.
