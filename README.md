# 🩺 MediNova - Advanced AI-Powered Medical Diagnosis & Clinical Decision Support Suite

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-185a9d?style=for-the-badge)](https://xgboost.ai)
[![License](https://img.shields.io/badge/License-Educational-06B6D4?style=for-the-badge)](LICENSE)

**MediNova** is a next-generation clinical decision support web application that bridges multi-parametric machine learning and computer vision to deliver rapid preliminary screening for **7 critical conditions**. 

Equipped with a modern dark medical interface, an **intelligent context-aware MediNova AI Clinical Copilot**, **automated emergency triage protocols for positive/high-risk tests**, **one-click demo sample automation**, and **1-click downloadable/printable hospital-grade medical reports**.

---

## 🌟 Key Features & Architectural Upgrades

### 1. 🤖 MediNova AI Clinical Copilot (Intelligent Medical Assistant)
Integrated across the Flask backend (`app.py`), Jinja2 templates, and obsidian glassmorphic frontend (`static/chatbot.js` + `static/style.css`):
- **Dual-Engine Intelligence Architecture**:
  - **Engine 1 (Live LLM Mode - Optional)**: Ingests `GEMINI_API_KEY` or `OPENAI_API_KEY` from environment variables, formatting dynamic system prompts with MediNova's 7 diagnostic models, `DISEASE_INSIGHTS`, and active patient biomarker findings.
  - **Engine 2 (Built-in Clinical Reasoning Engine - Zero-Key Fallback)**: 100% offline, guaranteed fast, zero-key fallback with structured clinical knowledge graph.
- **Biomarker Comparative Breakdown ("Explain My Results")**: Automatically reads active patient vitals, compares them against standard medical reference ranges (Blood Pressure, Cholesterol, HbA1c, Creatinine, Bilirubin, Enzymes), highlights out-of-range indicators, and details recommended specialist consultation urgency.
- **Interactive Symptom Triage & Navigation**: Natural-language symptom triage mapping queries to the exact MediNova screening module with 1-click action buttons (`/heart`, `/diabetes`, `/kidney`, `/liver`, `/breast_cancer`, `/malaria`, `/pneumonia`).
- **Red-Flag Emergency Detection**: Automatically detects acute life-threatening symptoms (crushing chest pain, severe dyspnea, seizures, syncope) and displays high-priority emergency alert callouts with 1-click links to hotlines (`112 / 911 / 108`) and `/contact`.
- **Conversational Form Auto-Fill**: Parses natural-language vitals typed into chat (or sample load requests) and enables 1-click auto-filling of diagnostic forms with smooth cyan highlight animations.
- **Hands-Free Voice Input & TTS**: Web Speech API speech-to-text microphone input and speech synthesis text-to-speech voice readout.
- **Workstation Maximize Toggle**: Smooth expansion from a compact drawer into a full clinical workstation view.

### 2. 🚨 Automated Emergency Protocol Suite (When Tests Are Positive)
Whenever an AI model detects a high risk, infection, or positive pathology, MediNova automatically deploys a multi-tier clinical response protocol:
- **Critical Alert Hero Card & Severity Gauge**: High-visibility flashing banner with an interactive risk severity meter (e.g., *88% High Priority*).
- **Specialist Referral Matching**: Clear referral to the appropriate specialist department (Cardiologist, Endocrinologist, Nephrologist, Hepatologist, Surgical Oncologist, Infectious Disease Specialist, or Pulmonologist) with urgency timeframes (e.g., *Consult within 24 to 48 hours*).
- **Red-Flag Emergency Warning Signs**: Highlights critical life-threatening symptoms that mandate immediate 911 / 112 emergency calls.
- **Organ-Specific Dietary Protocols**: Categorized dos and don'ts (Foods to Embrace vs. Foods to Strictly Avoid).
- **Essential Questions to Ask Your Doctor**: 4 to 5 targeted questions patients can bring to their physician appointment.
- **Interactive Patient Pre-Consultation Checklist**: Checkboxes enabling patients to prepare prior medical records, symptom timelines, and medication lists before their appointment.
- **🤖 Ask MediNova AI to Explain This Result**: 1-click button on every diagnostic result card opening the copilot with patient findings pre-loaded.
- **📄 Download / Print Clinical Medical Report (PDF)**: Generates a hospital-grade printable summary containing patient inputs, AI confidence levels, reference ranges, and physician sign-off sections via `@media print`.

### 3. ⚡ One-Click Quick Test Automation (Pre-filled Demos)
- Every tabular diagnostic form features instant **"⚠️ Fill High-Risk (Positive) Sample"** and **"🛡️ Fill Healthy (Negative) Sample"** buttons.
- Every computer vision diagnostic (Malaria and Pneumonia) features instant **"⚠️ Test Parasitized/Infected Sample"** and **"🛡️ Test Uninfected/Normal Sample"** buttons.
- Evaluators and testers can test any condition and observe positive and negative outcomes in **1 click** without typing complex medical values.

### 4. 🎨 State-of-the-Art Clinical Design
- Deep obsidian dark theme (`#070b14`) with glassmorphism, luminous medical cyan/teal accents (`#06b6d4`), and glowing status indicators.
- Animated ECG heartbeat pulse logo and live statistics counters.
- Clean printable reports hiding all UI elements and chatbot widgets during `@media print`.

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
- Python 3.10 to 3.13
- Pip package manager

### 2. Installation & Setup

1. **Navigate to the Project Directory**:
   ```bash
   cd MediNova-Medical-diagnosis-app
   ```

2. **Activate the Virtual Environment**:
   - Windows PowerShell:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - Windows Command Prompt:
     ```cmd
     .\venv\Scripts\activate.bat
     ```

3. **Install Dependencies** (if needed):
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

## 🔌 MediNova AI Copilot API Endpoints

### 1. Chat Completion Endpoint
- **URL**: `POST /api/chat`
- **Request Format**:
  ```json
  {
    "message": "Explain my heart screening results and abnormal biomarkers",
    "current_page": "/heart",
    "page_context": {
      "has_active_result": true,
      "is_positive": true
    }
  }
  ```
- **Response Format**:
  ```json
  {
    "reply": "Markdown formatted clinical response...",
    "confidence_badge": "Clinical Copilot • Context-Aware",
    "is_emergency": false,
    "suggested_actions": [
      {"label": "📄 Print Official Medical Report", "action": "print"},
      {"label": "🚨 Emergency & Care Directory", "url": "/contact"}
    ],
    "follow_up_chips": [
      "Which of my biomarkers are abnormal?",
      "What foods should I eat or avoid?",
      "What questions should I ask my doctor?"
    ],
    "form_fill_data": null
  }
  ```

### 2. Clear Session Endpoint
- **URL**: `POST /api/chat/clear`
- **Request Format**: `POST` empty body
- **Response Format**:
  ```json
  {
    "status": "cleared",
    "message": "Clinical diagnostic context and chat session reset successfully."
  }
  ```

---

## 📁 Project Directory Structure

```
MediNova-Medical-diagnosis-app/
├── app.py                      # Flask backend, AI Copilot dual engine, session tracking, protocols
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive system documentation
├── models/                     # Trained ML & Deep Learning models
│   ├── heart_disease_model.pkl
│   ├── diabetes_model.pkl
│   ├── kidney_model.pkl
│   ├── liver_model.pkl
│   ├── breast_cancer_model.pkl
│   ├── malaria_cnn_model_fast.h5
│   └── pneumonia_model_new.h5
├── templates/                  # Jinja2 HTML templates
│   ├── home.html               # Main landing page with live statistics
│   ├── features.html           # Diagnostic suite with live search and filter tabs
│   ├── heart.html              # Cardiovascular assessment with 1-click tester & copilot explain
│   ├── diabetes.html           # Diabetes metabolic check with 1-click tester & copilot explain
│   ├── kidney.html             # Renal panel (24 biomarkers) with 1-click tester & copilot explain
│   ├── liver.html              # Hepatic panel with 1-click tester & copilot explain
│   ├── breast_cancer.html      # Oncology morphometry with 1-click tester & copilot explain
│   ├── malaria.html            # Blood smear vision with live drag-and-drop & demo samples
│   ├── pneumonia.html          # Chest X-ray vision with live drag-and-drop & demo samples
│   ├── contact.html            # Emergency care directory and specialist appointment booking
│   └── index.html              # Clean redirect to /
├── static/                     # Static assets
│   ├── chatbot.js              # MediNova AI Copilot client controller (TTS, speech, auto-fill)
│   ├── style.css               # Clinical dark design system, glassmorphism, and print stylesheet
│   ├── samples/                # Sample images for 1-click testing
│   └── uploads/                # User uploaded medical images
└── scripts/                    # Verification and training scripts
```

---

## 🛡️ Medical Disclaimer & Clinical Advisory

> [!WARNING]
> **EDUCATIONAL & DECISION-SUPPORT USE ONLY**
> 
> MediNova is an artificial intelligence-assisted screening tool engineered for preliminary risk stratification and educational demonstration. It does **not** provide definitive medical diagnoses, prescriptions, or replace in-person consultation with licensed healthcare professionals. 
> 
> If you or someone you know is experiencing acute or life-threatening symptoms (such as acute chest pain, severe shortness of breath, sudden numbness, or loss of consciousness), contact emergency medical services (**112 / 911 / 108**) or visit an emergency room immediately.

---

<p align="center">
  Crafted with care by <strong>DHARSAN KUMAR R</strong>
</p>
