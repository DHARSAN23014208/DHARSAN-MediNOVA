# MediNova Clinical AI - Dataset Provenance & Governance Registry

This document records the provenance, licensing, cohort characteristics, ethical safeguards, and preprocessing contracts for all biomedical datasets utilized within MediNova.

---

## 1. Active CDSS Screening Datasets

### 1.1 UCI Thyroid Disease Dataset (Garavan Institute Cohort)
- **Source:** UCI Machine Learning Repository / Garavan Institute of Medical Research, Sydney, Australia (Donated by J. Ross Quinlan).
- **Files Acquired:** `datasets/thyroid/allhypo.data` (2,800 records), `datasets/thyroid/allhypo.test` (972 untouched test records).
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0) / Public Domain Academic Research.
- **Task:** Multi-analyte binary screening for primary and subclinical hypothyroidism.
- **Cohort Metadata:**
  - Total Samples: 3,772 patient records
  - Age Range: 1 – 94 years (Adult outpatient cohort)
  - Class Distribution: ~95% negative / euthyroid, ~5% elevated hypothyroid markers
- **Clinical Analyte Schema & Reference Intervals:**
  - `age`: Patient age in years
  - `sex`: Biological sex (0 = Female, 1 = Male)
  - `on_thyroxine`: Boolean indicator of ongoing Levothyroxine therapy
  - `TSH`: Serum Thyroid-Stimulating Hormone (Normal: 0.4 – 4.5 mIU/L)
  - `T3`: Serum Triiodothyronine (Normal: 1.2 – 3.1 nmol/L)
  - `TT4`: Serum Total Thyroxine (Normal: 60 – 150 nmol/L)
  - `T4U`: Thyroxine Uptake Ratio (Normal: 0.80 – 1.20)
  - `FTI`: Free Thyroxine Index (Normal: 85 – 130)
- **Evaluation Split Protocol:**
  - Training: 2,800 records (`allhypo.data`)
  - Test: 972 strictly untouched records (`allhypo.test`)
  - Results on 972 Unseen Patients: **Accuracy: 99.79%, Sensitivity: 100.0%, Specificity: 99.78%, ROC-AUC: 0.9997, Brier Score: 0.0024**.

---

### 1.2 PhysioNet MIT-BIH Arrhythmia Database
- **Source:** MIT-BIH Arrhythmia Database (Moody GB, Mark RG. IEEE Eng in Med and Biol 20(3):45-50, 2001; Goldberger AL et al. Circulation 101(23):e215-e220, 2000).
- **Repository:** [https://physionet.org/content/mitdb/1.0.0/](https://physionet.org/content/mitdb/1.0.0/)
- **Licence:** OpenPhysio / PhysioNet Open Data Licence (Attribution Required).
- **Records Acquired:** Record 100 (Normal Sinus Rhythm), Record 119 (Frequent Unifocal Ventricular Ectopy), Record 200 (Complex Ventricular Arrhythmias).
- **Task:** Single-lead biosignal rhythm screening for Premature Ventricular Contractions (PVC) and ectopic depolarizations.
- **Hardware & Telemetry Specification:**
  - Sampling Frequency: 360 Hz
  - Lead Configuration: Modified Lead II (MLII)
  - Bandpass Filter: 3rd-order Butterworth (0.5 – 40 Hz)
- **Strict Patient-Level Split Policy:**
  - **Training Cohort (DS1):** Patient 119 (444 beats, 198 normal sinus, 246 ventricular ectopy)
  - **Independent Test Cohort (DS2):** Patients 100 & 200 (288 unseen beats from completely separate individuals)
  - **Leakage Prevention:** Zero beat overlapping or patient overlap between training and testing splits.
- **Evaluation on 288 Unseen Independent Test Beats:**
  - Calibrated Screening Threshold: 0.20
  - **Accuracy: 99.65%, Sensitivity: 97.78% (44/45 arrhythmic beats detected), Specificity: 100.0% (243/243 normal beats), Precision: 100.0%, F1-Score: 0.9888, ROC-AUC: 0.9887**.

---

## 2. Seven Original Verified Model Cohorts

1. **Cardiovascular Disease (`models/heart_disease_model.pkl`):**
   - Source: UCI Heart Disease / Cleveland Clinic Foundation (V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.).
   - Modality: 13 Clinical vitals and stress test variables.
2. **Diabetes Mellitus (`models/diabetes_model.pkl`):**
   - Source: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).
   - Modality: 8 Clinical and metabolic vitals.
3. **Chronic Kidney Disease (`models/kidney_model.pkl`):**
   - Source: Apollo Hospitals Chronic Kidney Disease Dataset.
   - Modality: 24 Blood and urinalysis parameters.
4. **Hepatic Disorders (`models/liver_model.pkl`):**
   - Source: Indian Liver Patient Dataset (ILPD), North-East Andhra Pradesh.
   - Modality: 10 Biochemical liver function tests.
5. **Breast Oncology (`models/breast_cancer_model.pkl`):**
   - Source: Wisconsin Diagnostic Breast Cancer (WDBC) Database (Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian).
   - Modality: 5 Nuclear morphometry features from FNA.
6. **Malaria Parasite Smears (`models/malaria_cnn_model_fast.h5`):**
   - Source: National Institutes of Health (NIH) Lister Hill National Center for Biomedical Communications.
   - Modality: Giemsa-stained thin blood smear digital micrographs (50x50x3).
7. **Pneumonia Chest Radiography (`models/pneumonia_model_new.h5`):**
   - Source: Guangzhou Women and Children's Medical Center Pediatric Cohort (Kermany et al., Cell 2018).
   - Modality: Anterior-posterior chest radiograph scans (150x150x3).

---

## 3. Governed Research Roadmap Candidates (Inference Gated)

| Condition | Modality | Candidate Cohort | Intended Ground Truth | Activation Gate |
|:---|:---|:---|:---|:---|
| **Diabetic Retinopathy** | Retinal Fundus Photography | EyePACS / APTOS 2019 Blindness Detection | 5-stage International Clinical Diabetic Retinopathy (ICDR) scale | Patient-level separation between left/right eyes; independent multi-reader adjudication |
| **Cutaneous Melanoma** | Dermoscopy Micrographs | International Skin Imaging Collaboration (ISIC) Archive | Histopathologically confirmed biopsy | Fitzpatrick skin-type fairness check; removal of surgical and gel artifacts |
| **Pulmonary Tuberculosis** | Digital Chest X-Rays | Shenzhen No. 3 Hospital & Montgomery County (NIH) | Microbiological sputum smear & GeneXpert | Separation from non-TB acute bacterial pneumonia |
| **Hematologic Anemia** | Complete Blood Count (CBC) | MIMIC-IV Clinical Database / NHANES | WHO-standard age- and sex-stratified hemoglobin bounds | Differential classification of microcytic (iron deficiency) vs. normocytic |

*Policy: In compliance with FDA SaMD guidelines, research roadmap candidates are strictly restricted from executing active inference in clinical screening until multi-center cohort validation and regulatory checks are satisfied.*
