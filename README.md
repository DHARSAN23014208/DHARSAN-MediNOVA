# 🩺 CureSight - Advanced AI-Powered Medical Diagnosis & Clinical Decision Support Suite

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-185a9d?style=for-the-badge)](https://xgboost.ai)
[![License](https://img.shields.io/badge/License-Educational-06B6D4?style=for-the-badge)](LICENSE)

**CureSight** is a next-generation clinical decision support web application that bridges multi-parametric machine learning and computer vision to deliver rapid preliminary screening for **7 critical conditions**. 

Equipped with a modern dark medical interface, **automated emergency triage protocols for positive/high-risk tests**, **one-click demo sample automation**, and **1-click downloadable/printable hospital-grade medical reports**.

---

## 🌟 Key Features & Architectural Upgrades

### 1. 🚨 Automated Emergency Protocol Suite (When Tests Are Positive)
Whenever an AI model detects a high risk, infection, or positive pathology, CureSight automatically deploys a multi-tier clinical response protocol:
- **Critical Alert Hero Card & Severity Gauge**: High-visibility flashing banner with an interactive risk severity meter (e.g., *88% High Priority*).
- **Specialist Referral Matching**: Clear referral to the appropriate specialist department (Cardiologist, Endocrinologist, Nephrologist, Hepatologist, Surgical Oncologist, Infectious Disease Specialist, or Pulmonologist) with urgency timeframes (e.g., *Consult within 24 to 48 hours*).
- **Red-Flag Emergency Warning Signs**: Highlights critical life-threatening symptoms (e.g., crushing chest pressure radiating to arm, SpO2 hypoxemia below 92%, acute jaundice, ketoacidosis, cerebral malaria signs) that mandate immediate 911 / 112 emergency calls.
- **Cardioprotective & Organ-Specific Dietary Protocols**: Categorized dos and don'ts (Foods to Embrace vs. Foods to Strictly Avoid).
- **Essential Questions to Ask Your Doctor**: 4 to 5 targeted questions patients can bring to their physician appointment.
- **Interactive Patient Pre-Consultation Checklist**: Checkboxes enabling patients to prepare prior medical records, symptom timelines, and medication lists before their appointment.
- **📄 Download / Print Clinical Medical Report (PDF)**: Generates a hospital-grade printable summary containing patient inputs, AI confidence levels, reference ranges, and physician sign-off sections via `@media print`.

### 2. ⚡ One-Click Quick Test Automation (Pre-filled Demos)
- Every tabular diagnostic form features instant **"⚠️ Fill High-Risk (Positive) Sample"** and **"🛡️ Fill Healthy (Negative) Sample"** buttons.
- Every computer vision diagnostic (Malaria and Pneumonia) features instant **"⚠️ Test Parasitized/Infected Sample"** and **"🛡️ Test Uninfected/Normal Sample"** buttons.
- Evaluators and testers can test any condition and observe positive and negative outcomes in **1 click** without typing complex medical values.

### 3. 🎨 State-of-the-Art Clinical Design
- Deep obsidian dark theme (`#070b14`) with glassmorphism, luminous medical cyan/teal accents (`#06b6d4`), and glowing status indicators.
- Animated ECG heartbeat pulse logo and live statistics counters.
- Drag-and-drop image uploaders with instant client-side thumbnail previews.
- Fully responsive across desktop, tablet, and mobile devices.

### 4. 🚑 Emergency Care & Telehealth Directory (`/contact`)
- Instant access to emergency dispatch hotlines (112, 911, 108).
- 24/7 non-emergency triage nurse helpline (104, 1800-HEALTH-LINE).
- Priority specialist appointment request form.

---

## 🔬 Supported Diagnostic Models & Accuracies

| Condition | Modality / Input | Machine Learning Architecture | Clinical Accuracy | Output / Finding |
| :--- | :--- | :--- | :--- | :--- |
| **Heart Disease** | 13 Clinical Vitals (BP, Chol, EKG, ST Depression, etc.) | Random Forest Classifier | **~85%** | Presence / Absence of Cardiac Risk |
| **Diabetes** | 8 Metabolic Biomarkers (HbA1c, Glucose, BMI, etc.) | XGBoost Gradient Boosting | **~92%** | Positive / Negative Diabetes Indicator |
| **Kidney Disease** | 24 Urinalysis & Blood Chemistry Markers | Random Forest Ensemble | **~98%** | CKD Risk / Healthy Renal Function |
| **Liver Disease** | 10 Hepatic Parameters (ALT, AST, Bilirubin, Albumin, etc.) | Random Forest Classifier | **~75%** | High / Low Probability of Hepatic Disease |
| **Breast Cancer** | 5 Cellular Nucleus Morphometry Metrics | Fine-Needle Aspirate Classifier | **~95%** | Malignant Neoplasm / Benign Tissue |
| **Malaria** | Microscopic Thin Blood Smear Images (50x50) | Deep Convolutional Neural Network (CNN) | **~95%+** | Parasitized (Infected) / Uninfected |
| **Pneumonia** | Chest Radiograph X-Ray Scans (150x150) | Deep Convolutional Neural Network (CNN) | **~90%+** | Pulmonary Infiltrates / Clear Lungs |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.10 - 3.13)
- **Pip** package manager
- Recommended virtual environment

### 2. Installation & Setup

1. **Navigate to the Project Directory**:
   ```bash
   cd "c:\Users\admin\Desktop\MINI PROJECT\CureSight-Medical-diagnosis-app"
   ```

2. **Activate the Virtual Environment**:
   - Windows PowerShell:
     ```powershell
     ..\venv\Scripts\Activate.ps1
     ```
   - Windows Command Prompt:
     ```cmd
     ..\venv\Scripts\activate.bat
     ```
   - Linux / macOS:
     ```bash
     source ../venv/bin/activate
     ```

3. **Install Dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Application**:
   ```bash
   python app.py
   ```

5. **Open in Web Browser**:
   ```
   http://127.0.0.1:5000
   ```

---

## 📋 How to Use Every Feature (Step-by-Step)

### A. Navigating the Portal
- **Home (`/`)**: Overview of diagnostic capabilities, live statistics, clinical pipeline overview, and emergency banner.
- **Diagnostic Suite (`/features`)**: Filterable library of all 7 diagnostic tools. Use category buttons (*Vitals & Heart*, *Metabolic & Labs*, *Deep Learning Vision*) or the search bar (*"blood"*, *"lung"*, *"heart"*) to locate tests instantly.
- **Emergency & Care (`/contact`)**: Emergency dispatch numbers, triage nurse lines, and specialist appointment booking.

### B. Testing Positive & Negative Predictions (Form-Based Models)

#### 1. Heart Disease (`/heart`)
- **To test High-Risk (Positive)**:
  - Click **"⚠️ Fill High-Risk (Positive) Sample"** at the top of the form.
  - Form automatically fills: Age=60, Sex=Male, CP=Asymptomatic, BP=145, Chol=280, FBS=1, EKG=LVH, Max HR=130, ExAng=Yes, Oldpeak=2.5, Slope=Downsloping, CA=2, Thal=Reversible Defect.
  - Click **"Run AI Cardiovascular Analysis"**.
  - *Result*: Critical alert banner (*Cardiovascular Risk Indicators Detected*), 88% severity meter, Cardiologist referral within 24-48 hrs, crushing chest pain emergency warnings, Mediterranean cardioprotective diet guidelines, cardiologist questions, patient checklist, and full biomarker table.
- **To test Low-Risk (Negative)**:
  - Click **"🛡️ Fill Healthy (Negative) Sample"** -> Click **"Run AI Cardiovascular Analysis"**.
  - *Result*: Green optimal card (*Absence of Significant Risk*), wellness maintenance guidelines.

#### 2. Diabetes Risk (`/diabetes`)
- **To test High-Risk (Positive)**:
  - Click **"⚠️ Fill High-Risk (Positive) Sample"** (Age=65, BMI=34.5, HbA1c=8.5%, Blood Glucose=220 mg/dL, Hypertension=Yes).
  - Click **"Run AI Diabetes Analysis"**.
  - *Result*: Critical alert banner (*Elevated Glycemic & Metabolic Risk Detected*), Endocrinologist referral, Diabetic Ketoacidosis (DKA) warning signs, Low-GI dietary protocol, patient consultation checklist.
- **To test Low-Risk (Negative)**:
  - Click **"🛡️ Fill Healthy (Negative) Sample"** (Age=28, BMI=22.0, HbA1c=5.0%, Blood Glucose=85 mg/dL).
  - *Result*: Green optimal card (*Negative - Normal Glycemic Indicators*).

#### 3. Kidney Disease (`/kidney`)
- **To test High-Risk (Positive / CKD)**:
  - Click **"⚠️ Fill High-Risk (Positive/CKD) Sample"** (Auto-populates all 24 markers: Creatinine=3.5 mg/dL, Blood Urea=65 mg/dL, Albumin=Grade 3, Urine SG=1.010, Pedal Edema=Yes, Hypertension=Yes).
  - Click **"Run AI Kidney Disease Analysis"**.
  - *Result*: High-priority nephrology alert, NSAID avoidance mandate, uremia/oliguria warnings, renal dietary restrictions (controlled potassium, sodium, and phosphorus), patient checklist.
- **To test Low-Risk (Negative)**:
  - Click **"🛡️ Fill Healthy (Negative) Sample"** (Creatinine=0.9 mg/dL, Albumin=0, SG=1.025, Normal RBC/Pus cells).
  - *Result*: Green healthy renal physiology report.

#### 4. Liver Function (`/liver`)
- **To test High-Risk (Positive)**:
  - Click **"⚠️ Fill High-Risk (Positive) Sample"** (Total Bilirubin=4.5 mg/dL, Direct Bilirubin=2.3 mg/dL, ALP=380 U/L, ALT=95 U/L, AST=120 U/L, A/G Ratio=0.7).
  - Click **"Run AI Liver Disease Analysis"**.
  - *Result*: High probability alert, Hepatologist consultation within 3-5 days, Jaundice & ascites warnings, total alcohol cessation guidelines, liver diet dos & don'ts.
- **To test Low-Risk (Negative)**:
  - Click **"🛡️ Fill Healthy (Negative) Sample"** (Bilirubin=0.8 mg/dL, ALT=22 U/L, AST=24 U/L, Albumin=4.1 g/dL).
  - *Result*: Low probability healthy liver assessment.

#### 5. Breast Cancer Screening (`/breast_cancer`)
- **To test Malignant (High Risk)**:
  - Click **"⚠️ Fill Malignant (High Risk) Sample"** (Mean Radius=18.5, Perimeter=125.0, Area=1100.0, Texture=24.5, Smoothness=0.12).
  - Click **"Run AI Morphometry Analysis"**.
  - *Result*: High-priority surgical oncologist referral within 24-48 hrs, biopsy recommendations (Core Needle Biopsy, ER/PR/HER2 receptor status), clinical red-flag breast symptoms, oncologist discussion questions.
- **To test Benign (Low Risk)**:
  - Click **"🛡️ Fill Benign (Low Risk) Sample"** (Mean Radius=11.2, Perimeter=72.0, Area=380.0, Smoothness=0.08).
  - *Result*: Benign cellular architecture report with routine mammography schedule.

---

### C. Testing Computer Vision & Image Models

#### 6. Malaria Blood Smear Detection (`/malaria`)
- **Option 1 (Instant 1-Click Verification)**:
  - Click **"⚠️ Test Parasitized (Infected) Cell"** or **"🛡️ Test Uninfected (Healthy) Cell"**.
  - The deep CNN model scans the sample in < 0.4s and renders the classification with **99.8% confidence**, side-by-side microscopic preview, and urgent antimalarial (ACT) treatment guidelines.
- **Option 2 (Custom Upload)**:
  - Drag and drop any Giemsa-stained thin blood smear cell image into the upload box.
  - A real-time preview thumbnail appears.
  - Click **"Run Deep CNN Malaria Scan"**.

#### 7. Chest X-Ray Pneumonia Detection (`/pneumonia`)
- **Option 1 (Instant 1-Click Verification)**:
  - Click **"⚠️ Test Sample Pneumonia Detected X-Ray"**.
  - The deep radiograph model scans the scan in < 0.5s and renders **Pneumonia Detected (96.9% confidence)**, displaying the analyzed radiograph alongside Pulmonologist action steps, SpO2 hypoxemia warnings, and sputum culture recommendations.
- **Option 2 (Custom Upload)**:
  - Drag and drop any chest X-ray image into the upload zone.
  - Click **"Run Deep Learning Radiograph Analysis"**.

---

### D. 📄 Generating & Printing Clinical Medical Reports
1. Run any diagnostic test (Positive or Negative).
2. At the top of the result section, click **"📄 Download / Print Medical Report (PDF)"**.
3. Your browser's print dialog will open automatically.
4. The report includes:
   - Official Clinic Header: *CureSight Clinical Intelligence*
   - Patient Screening Report ID & Date/Time Stamp
   - Primary AI Diagnostic Assessment & Severity Gauge
   - Specialist Referral & Clinical Confirmatory Tests
   - Emergency Red-Flag Warning Signs & Dietary Action Plan
   - Comprehensive Table of all Patient Submitted Biomarkers with Standard Target Reference Ranges
   - Reviewing Physician Sign-off Box & Medical Disclaimer
5. Select **"Save as PDF"** or choose your physical printer for a clean, hospital-grade document!

---

## 📁 Project Directory Structure

```
CureSight-Medical-diagnosis-app/
├── app.py                      # Flask backend, routing, AI inference, and clinical protocols
├── requirements.txt            # Python dependencies (Flask, TensorFlow, Scikit-learn, etc.)
├── README.md                   # Comprehensive system documentation
├── models/                     # Trained ML & Deep Learning models
│   ├── heart_disease_model.pkl
│   ├── diabetes_model.pkl
│   ├── kidney_model.pkl
│   ├── liver_model.pkl
│   ├── breast_cancer_model.pkl
│   ├── malaria_cnn_model_fast.h5
│   └── pneumonia_model_new.h5
├── templates/                  # Modern Jinja2 HTML templates
│   ├── home.html               # Main landing page with live statistics
│   ├── features.html           # Diagnostic suite with live search and filter tabs
│   ├── heart.html              # Cardiovascular assessment with 1-click tester & printable report
│   ├── diabetes.html           # Diabetes metabolic check with 1-click tester & printable report
│   ├── kidney.html             # Renal panel (24 biomarkers) with 1-click tester & printable report
│   ├── liver.html              # Hepatic panel with 1-click tester & printable report
│   ├── breast_cancer.html      # Oncology morphometry with 1-click tester & printable report
│   ├── malaria.html            # Blood smear vision with live drag-and-drop & demo samples
│   ├── pneumonia.html          # Chest X-ray vision with live drag-and-drop & demo samples
│   ├── contact.html            # Emergency care directory and specialist appointment booking
│   └── index.html              # Clean redirect to /
├── static/                     # Static assets
│   ├── style.css               # Clinical dark design system, glassmorphism, and print stylesheet
│   ├── samples/                # Pre-verified sample images for 1-click testing
│   │   ├── malaria_infected.png
│   │   ├── malaria_uninfected.png
│   │   └── pneumonia_detected.jpeg
│   └── uploads/                # User uploaded medical images
└── scripts/                    # Verification and training scripts
```

---

## 🛡️ Medical Disclaimer & Clinical Advisory

> [!WARNING]
> **EDUCATIONAL & DECISION-SUPPORT USE ONLY**
> 
> CureSight is an artificial intelligence-assisted screening tool engineered for preliminary risk stratification and educational demonstration. It does **not** provide definitive medical diagnoses, prescriptions, or replace in-person consultation with licensed healthcare professionals. 
> 
> If you or someone you know is experiencing acute or life-threatening symptoms (such as acute chest pain, severe shortness of breath, sudden numbness, or loss of consciousness), contact emergency medical services (**112 / 911 / 108**) or visit an emergency room immediately.

---

<p align="center">
  Crafted with care by <strong>Dharsan and Engineering Team</strong>
</p>
