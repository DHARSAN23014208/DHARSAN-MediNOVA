# 🩺 MediNova - Advanced AI-Powered Medical Diagnosis & Clinical Decision Support Suite

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-185a9d?style=for-the-badge)](https://xgboost.ai)
[![License](https://img.shields.io/badge/License-Educational-06B6D4?style=for-the-badge)](LICENSE)

**MediNova** is a next-generation clinical decision support web application that bridges multi-parametric machine learning and computer vision to deliver rapid preliminary screening for **7 critical conditions**. 

Equipped with a modern dark medical interface, an **intelligent context-aware MediNova AI Clinical Copilot**, a **location-aware Smart Hospital Discovery & Priority Appointment Management System**, **automated emergency triage protocols for positive/high-risk tests**, **one-click demo sample automation**, and **1-click downloadable/printable hospital-grade medical reports**.

---

## 🌟 Key Features & Architectural Upgrades

### 1. 🏥 Smart Hospital Discovery & Appointment Management System (v4.0)
Upgrades the specialist consultation workflow into an end-to-end, location-aware healthcare discovery and booking system:
- **Live Geolocation & Manual Search**:
  - One-click **"📍 Use My Current Location"** leveraging the HTML5 Geolocation API with user permission.
  - Manual location search by city, postcode, or address geocoded via OpenStreetMap Nominatim.
  - Configurable search radius (**5 km, 10 km, 20 km default, 25 km**).
  - OpenStreetMap Overpass API nearby facility discovery (`amenity=hospital`, `amenity=clinic`, `healthcare=hospital`, etc.) with in-memory caching (30-min TTL), deduplication, and straight-line Haversine distance calculations.
- **Seven-Disease Specialty Mapping**:
  - Canonical mapping for all 7 conditions (`heart`, `diabetes`, `kidney`, `liver`, `cancer`, `malaria`, `pneumonia`) to primary specialties, sub-specialties, specialist physician titles, department names, hospital search keywords, and urgent clinical assessment flags.
- **Curated Global Specialty Referral Centres**:
  - Structured, source-attributed dataset of world-renowned regional and international specialty centres (including *Cleveland Clinic, Mayo Clinic, Apollo Hospitals, CMC Vellore, Tata Memorial Centre, MD Anderson, King's College Hospital, ILBS, NUH Singapore, AIIMS New Delhi, Royal Brompton, and National Jewish Health*).
  - Complete with official portals, direct appointment/referral URLs, verified accreditation credentials (JCI, NABH, Magnet, NCI), and last-verified timestamps.
- **Intelligent Recommendation Engine**:
  - Evaluates specialty match confidence, distance from patient location, verified departmental services, and consultation mode.
  - Transparent recommendation rationale attached to each facility (e.g., *"Recommended because the facility lists the relevant specialty and is approximately 4.2 km from your selected location"*).
- **Interactive Hospital Comparison Table**:
  - Columns: Select, Facility Name & Type, Specialty Match, Address & Distance, Consultation Mode, Verified Services, and Actions ("View Details" modal).
  - Categorized badges: `Nearby Facility`, `Global Specialty Centre`, `Specialty Verified`, `Specialty Not Verified`, `Telemedicine Verified`.
  - Responsive mobile design: automatically transforms into accessible cards on screens `< 768px`.
  - Selected Hospital Summary Card highlighting facility, address, straight-line distance, specialty match explanation, and chosen mode.
- **Consultation Modes & Scheduling**:
  - In-person consultation or telemedicine video consultation (disabled if unsupported by provider).
  - Date picker (past dates blocked) and configurable time slots (morning, mid-day, afternoon, evening).
  - Prominent scheduling status disclaimer distinguishing requested preference from verified hospital confirmation.
- **Persistent SQLite Database & Reference IDs**:
  - Thread-safe persistence in `medinova_appointments.db` with parameterized queries and indexes.
  - Cryptographically secure unique Reference IDs (e.g., `APT-2026-8D2185`).
  - Meaningful status workflow starting strictly with `Awaiting Hospital Confirmation`, followed by `Confirmed by Hospital`, `Rescheduling Required`, `Cancelled`, `Completed`, or `Rejected`.
- **Printable Medical Slip & Calendar Integration**:
  - Dedicated print-friendly receipt slip (`@media print`) with an option to hide sensitive clinical notes.
  - 1-click **Add to Google Calendar** web event link.
  - RFC 5545 compliant downloadable **iCalendar (`.ics`)** file.
- **Notification Service**:
  - Dispatches alerts to clinic care coordinators / developer mailbox via SMTP with delivery audit logging.
- **Patient Tracking & Admin Portals**:
  - **Patient Tracking Dashboard** (`/contact#trackingSection`): Lookup by Reference ID + Contact with live status, update timeline, and cancellation self-service.
  - **Clinic Administrator Dashboard** (`/developer/patient-records`): Authenticated portal allowing administrators to review incoming requests, change status, and attach clinical audit notes.

### 2. 🤖 MediNova AI Clinical Copilot (Intelligent Medical Assistant)
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

### 3. 🚨 Automated Emergency Protocol Suite (When Tests Are Positive)
Whenever an AI model detects a high risk, infection, or positive pathology, MediNova automatically deploys a multi-tier clinical response protocol:
- **Critical Alert Hero Card & Severity Gauge**: High-visibility flashing banner with an interactive risk severity meter (e.g., *88% High Priority*).
- **Specialist Referral Matching**: Clear referral to the appropriate specialist department (Cardiologist, Endocrinologist, Nephrologist, Hepatologist, Surgical Oncologist, Infectious Disease Specialist, or Pulmonologist) with urgency timeframes (e.g., *Consult within 24 to 48 hours*).
- **Red-Flag Emergency Warning Signs**: Highlights critical life-threatening symptoms that mandate immediate 911 / 112 emergency calls.
- **Organ-Specific Dietary Protocols**: Categorized dos and don'ts (Foods to Embrace vs. Foods to Strictly Avoid).
- **Essential Questions to Ask Your Doctor**: 4 to 5 targeted questions patients can bring to their physician appointment.
- **Interactive Patient Pre-Consultation Checklist**: Checkboxes enabling patients to prepare prior medical records, symptom timelines, and medication lists before their appointment.
- **🤖 Ask MediNova AI to Explain This Result**: 1-click button on every diagnostic result card opening the copilot with patient findings pre-loaded.
- **📄 Download / Print Clinical Medical Report (PDF)**: Generates a hospital-grade printable summary containing patient inputs, AI confidence levels, reference ranges, and physician sign-off sections via `@media print`.

### 4. ⚡ One-Click Quick Test Automation (Pre-filled Demos)
- Every tabular diagnostic form features instant **"⚠️ Fill High-Risk (Positive) Sample"** and **"🛡️ Fill Healthy (Negative) Sample"** buttons.
- Every computer vision diagnostic (Malaria and Pneumonia) features instant **"⚠️ Test Parasitized/Infected Sample"** and **"🛡️ Test Uninfected/Normal Sample"** buttons.
- Evaluators and testers can test any condition and observe positive and negative outcomes in **1 click** without typing complex medical values.

### 5. 🎨 State-of-the-Art Clinical Design
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

### 3. Hospital Discovery & Appointment Management Endpoints

| Endpoint | Method | Access | Description |
| :--- | :---: | :---: | :--- |
| `/api/hospitals/specialties` | `GET` | Public | Returns canonical 7-disease specialty mapping, specialist titles, and curated global centres |
| `/api/hospitals/search` | `GET` | Public | Discovers nearby facilities (Overpass API) & curated global centres with distance sorting |
| `/api/appointments/book` | `POST` | Public | Submits appointment, validates fields/dates, persists to SQLite, triggers coordinator email |
| `/api/appointments/lookup` | `POST` | Patient | Enforces patient record isolation: requires Reference ID and matching Phone or Email |
| `/api/appointments/cancel` | `POST` | Patient | Self-service cancellation of eligible pending appointment requests |
| `/api/appointments/<ref_id>/ics` | `GET` | Public | Generates and downloads RFC 5545 standard `.ics` calendar invitation |
| `/api/admin/appointments` | `GET` | Authenticated | Lists all appointment requests with status/disease filters for clinic coordinators |
| `/api/admin/appointments/update-status` | `POST` | Authenticated | Transitions appointment status (e.g. to Confirmed by Hospital) with audit history |

---

## 📁 Project Directory Structure

```
MediNova-Medical-diagnosis-app/
├── app.py                      # Flask backend, AI Copilot, hospital discovery APIs, appointment routes
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive system documentation
├── .env.example                # Sample environment configuration template
├── medinova_appointments.db    # SQLite persistent appointment database (auto-initialized, git-ignored)
├── models/                     # Trained ML & Deep Learning vision models
│   ├── heart_disease_model.pkl
│   ├── diabetes_model.pkl
│   ├── kidney_model.pkl
│   ├── liver_model.pkl
│   ├── breast_cancer_model.pkl
│   ├── malaria_cnn_model_fast.h5
│   └── pneumonia_model_new.h5
├── utils/                      # Modular application services & helpers
│   ├── hospital_service.py     # 7-disease specialty mapping, Overpass discovery & recommendation engine
│   ├── appointment_db.py       # Thread-safe SQLite persistence, reference IDs & status workflow
│   ├── calendar_service.py     # 1-click Google Calendar & RFC 5545 .ics calendar generation
│   ├── appointment_notifier.py # SMTP clinic coordinator notification & delivery audit logging
│   └── google_sheets_logger.py # Silent background thread worker logging to Google Sheets
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
│   ├── contact.html            # Smart Hospital Discovery & Priority Appointment Management Suite
│   ├── dev_access.html         # Authenticated Developer Portal & Appointment Scheduling Management
│   └── index.html              # Clean redirect to /
├── static/                     # Static assets
│   ├── chatbot.js              # MediNova AI Copilot client controller (TTS, speech, auto-fill)
│   ├── style.css               # Clinical dark design system, comparison table, and print stylesheet
│   ├── samples/                # Sample images for 1-click testing
│   └── uploads/                # User uploaded medical images
└── scripts/                    # Automated testing, verification, and model training
    ├── test_hospital_system.py # End-to-end test suite for hospital discovery & appointments
    ├── verify_chat_flow.py     # 11-route regression test suite for copilot & diagnostics
    ├── audit_repo.py           # Deep repository health audit
    └── train_*.py              # Training scripts for tabular and vision models
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
