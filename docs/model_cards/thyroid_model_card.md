# Model Card: MediNova Calibrated Thyroid Dysfunction Screening Engine

## 1. Model Details
- **Model Identifier:** `medinova-thyroid-v1`
- **Version:** `1.0.0`
- **Model Type:** Scikit-learn Pipeline (`SimpleImputer(strategy='median')` + `CalibratedClassifierCV(RandomForestClassifier, cv=3)`)
- **Developer:** MediNova Healthcare AI & Clinical Informatics
- **Artifact File:** `models/thyroid_model.pkl`
- **SHA-256 Checksum:** `a4a528b07092aca8f908b58e7e7fc7952b48646e5321d4d46343830f953b275f`
- **Release Date:** October 2026
- **License:** Open Academic Clinical CDSS (GPL-3.0 compatible)

---

## 2. Intended Use & Clinical Scope
- **Primary Clinical Purpose:** Rapid preliminary laboratory screening for adult thyroid dysfunction (primary or subclinical hypothyroidism) based on standard outpatient thyroid panel measurements.
- **Intended Users:** Outpatient primary care clinics, endocrine triage nurses, clinical laboratories, and patient decision-support.
- **Excluded Uses:**
  - Definitive diagnosis of thyroid cancer, thyroid storm, or toxic multinodular goiter.
  - Automated dosage titration of Levothyroxine without physician consultation.
  - Standalone clinical decision-making for pregnant women or pediatric patients under 18 years.

---

## 3. Training & Evaluation Data
- **Dataset:** UCI Thyroid Disease Dataset (`allhypo.data` / `allhypo.test`).
- **Data Origin:** Garavan Institute of Medical Research, Sydney, Australia.
- **Cohort Size:** 3,772 patient records (2,800 training, 972 untouched independent test set).
- **Features (8):**
  1. `age`: Patient Age (years, 18 – 100)
  2. `sex`: Biological Sex (0 = Female, 1 = Male)
  3. `on_thyroxine`: Currently receiving Levothyroxine therapy (0/1)
  4. `TSH`: Serum Thyroid-Stimulating Hormone (mIU/L, Target: 0.4 – 4.5)
  5. `T3`: Serum Triiodothyronine (nmol/L, Target: 1.2 – 3.1)
  6. `TT4`: Serum Total Thyroxine (nmol/L, Target: 60 – 150)
  7. `T4U`: Thyroxine Uptake Ratio (Ratio, Target: 0.80 – 1.20)
  8. `FTI`: Free Thyroxine Index (Index, Target: 85 – 130)

---

## 4. Quantitative Evaluation Benchmarks
Evaluated on **972 untouched, unseen test patients**:

| Metric | Score | Clinical Interpretation |
|:---|:---|:---|
| **Accuracy** | **99.79%** | Correct overall screening concordance |
| **Sensitivity (Recall)** | **100.0%** | Zero false negatives (71/71 positive patients identified) |
| **Specificity** | **99.78%** | Minimized false positives on healthy individuals |
| **Precision (PPV)** | **97.26%** | High positive predictive confidence |
| **Negative Predictive Value (NPV)** | **100.0%** | Absolute certainty when screening as negative |
| **F1-Score** | **0.9861** | Harmonized precision-recall balance |
| **ROC-AUC** | **0.9997** | Near-perfect class separability |
| **Brier Score** | **0.0024** | Exceptional probability calibration |

---

## 5. Calibration & Uncertainty
- **Calibration Method:** Isotonic Regression via 3-fold cross-validation inside `CalibratedClassifierCV`.
- **Decision Threshold:** Standard 0.50 probability cutoff for balanced risk communication.
- **Out-of-Distribution Handling:** Unphysiological values (e.g. TSH > 250 mIU/L or TT4 > 350 nmol/L) trigger an automated OOD warning and clinical abstention.

---

## 6. Ethical & Safety Considerations
- **Demographic Representation:** Cohort reflects balanced adult outpatient populations; thyroid disorders are biologically more prevalent in females (~80% in training data), which matches authentic epidemiology.
- **Safety Disclaimer:** Automated preliminary screening only. Abnormal findings must be confirmed with venous repeat panels and consultation with an endocrinologist.
