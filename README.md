# 🩺 MediNova - Advanced AI-Powered Clinical Decision Support Suite (CDSS)

[![Python](https://img.shields.io/badge/Python-3.10%20--%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-185a9d?style=for-the-badge)](https://xgboost.ai)
[![Tests](https://img.shields.io/badge/Tests-54%20Passed%20%E2%80%A2%20100%25-brightgreen?style=for-the-badge)](tests/)
[![Security](https://img.shields.io/badge/Security-Fail--Closed%20%E2%80%A2%20HMAC--SHA256-blueviolet?style=for-the-badge)](ARCHITECTURE.md#4-security--privacy-framework)

**MediNova** is an enterprise-grade Clinical Decision Support System (CDSS) that bridges multi-parametric machine learning, deep convolutional neural networks, optical character recognition (OCR), geospatial healthcare discovery, priority appointment lifecycle tracking, cryptographic document authentication, and asynchronous cloud synchronization.

> 📖 **Full System Architecture & Workflow Specification:**  
> For detailed architectural diagrams, sequence diagrams, database schemas, and state-machine transitions, review the complete [MediNova System Architecture & End-to-End Workflow Specification](ARCHITECTURE.md).

---

## 🌟 Key Functional Pillars & Subsystems

### 1. 🔬 Seven Diagnostic Disease Modules
- **Heart Disease:** Random Forest classifier analyzing 13 cardiovascular vitals (Resting BP, Cholesterol, EKG, ST Depression, Thallium, etc.).
- **Diabetes:** XGBoost gradient booster evaluating 8 metabolic markers (HbA1c, Glucose, BMI, Blood Pressure, Insulin, etc.).
- **Kidney Disease:** Random Forest ensemble evaluating 24 renal biomarkers (Serum Creatinine, Blood Urea, Specific Gravity, Albumin, etc.).
- **Liver Disease:** Random Forest classifier screening 10 hepatic enzymes and parameters (Bilirubin, ALT, AST, Albumin, A/G Ratio).
- **Breast Cancer:** Cellular morphometry classifier assessing 5 fine-needle aspirate (FNA) nucleus parameters.
- **Malaria:** Deep Convolutional Neural Network (CNN) analyzing $50 \times 50 \times 3$ microscopic thin blood smear images.
- **Pneumonia:** Deep Convolutional Neural Network (CNN) analyzing $150 \times 150 \times 3$ chest radiograph (CXR) scans.
- **Confidence Integrity:** Eliminates hardcoded confidence fallbacks (no fake 95.0%). Calibrated probabilities are rendered when available; discrete models report *"Not Stated (Categorical Assessment)"* with `confidence_score = None`.

### 2. 📄 Medical Report Scanner & Extraction Engine
- **Multi-Format Ingestion:** Ingests digital PDFs and image lab reports (PNG, JPG, JPEG, WebP) up to 16 MB.
- **Text & OCR Extraction:** Uses `pypdf` for clean digital text extraction and `pytesseract` OCR for scanned paperwork.
- **Biomarker Matcher & Unit Normalizer:** Parses clinical synonyms and converts laboratory units (e.g., $\text{mmol/L} \to \text{mg/dL}$).
- **Instant Triage Mode:** Automatically populates diagnostic input fields or directly triggers predictive inference.

### 3. 🔐 Cryptographic Report Integrity & Dynamic QR Codes
- **HMAC-SHA256 Sealed:** Every generated screening report is stamped with a tamper-evident cryptographic signature.
- **Dynamic QR Code:** Generates high-resolution PNG QR codes pointing to official verification URLs (`/verify-report/<report_id>`).
- **Public Audit Portal:** Anyone can scan the report QR code or enter the Report ID to verify assessment authenticity against `report_verifications.db`.

### 4. 🏥 Verified Care Network & Disease-Specific Hospital Discovery
- **Strict Specialty Matching:** Matches conditions exclusively to verified departments (e.g., Pneumonia $\to$ Pulmonology/Respiratory care; generic Internal Medicine is strictly disallowed).
- **Geocoding & Bounded Search:** OpenStreetMap / Photon geocoding with server-side coordinate boundaries ($-90 \le \text{lat} \le 90$, $-180 \le \text{lon} \le 180$) and radius options ($1 \le \text{radius\_km} \le 100$).
- **Progressive Auto-Expansion:** Gradually expands search from 5 km up to 100 km when local facilities are scarce, accompanied by clear UI notice.
- **Rating & Non-Clinical Filters:** Excludes facilities with ratings $\le 3.0$ and discards alternative medicine, dental, and optical clinics.

### 5. 📅 Priority Appointment Lifecycle State Machine
- **State Machine Graph:** Enforces strict status transitions (`Awaiting Hospital Confirmation` $\to$ `Confirmed by Hospital` $\to$ `Completed`, etc.) with terminal state protection.
- **Persistent SQLite Outbox:** Parameterized queries, indexed reference IDs (`APT-YYYY-XXXXXX`), and timestamped audit logs in `medinova_appointments.db`.
- **Patient Privacy Isolation:** Appointment lookup requires Reference ID plus matching patient phone or email.
- **Digital Calendar Slip:** RFC 5545 standard `.ics` calendar download and 1-click Google Calendar web event creation.

### 6. 📊 Asynchronous Google Sheets Logging & SQLite Outbox
- **Zero Latency:** Patient inference responses return immediately while logging occurs via a background worker thread.
- **Durable Outbox Queue:** Records are inserted into `sheets_outbox` prior to memory queue dispatch, ensuring resilience across restarts.
- **Idempotency Guarantee:** Tracks condition-sheet and Master-Log appends independently; network retries never produce duplicate rows.
- **Formula Injection Defense:** Automatically neutralizes spreadsheet formulas by prepending single quotes (`'`) to strings starting with `=`, `+`, `-`, `@`, `\t`, `\r`.

### 7. 🛡️ Application Security & Privacy Architecture
- **Fail-Closed Secrets:** In production (`FLASK_ENV=production`), missing or default values for `SECRET_KEY`, `DEV_ACCESS_KEY`, and `REPORT_SIGNING_SECRET` immediately halt startup.
- **Hardened Developer 2FA:** 6-digit OTP dispatched via SMTP; never printed to console logs or HTTP responses; 60-second rate-limiting; 5-attempt lockout; constant-time comparison via `secrets.compare_digest()`.
- **Patient Image Privacy:** Uploaded scans are saved to `private_uploads/` outside the public web root. Images are accessed through ephemeral, session-bound preview tokens with 30-minute expiry.
- **Content Verification:** Uploads are validated via Pillow (`Image.open().verify()`), blocking polyglots and corrupt files.

### 8. 🌐 Multilingual Patient Summaries
- Real-time client-side language switcher supporting **English**, **தமிழ் (Tamil)**, **हिंदी (Hindi)**, and **Español (Spanish)** across diagnostic result cards.

---

## 🔬 Diagnostic Models & Accuracy Reference

| Condition | Input Modality | Model Architecture | Baseline Accuracy | Diagnostic Finding |
| :--- | :--- | :--- | :---: | :--- |
| **Heart Disease** | 13 Clinical Vitals | Random Forest Classifier | **~85%** | Elevated Cardiac Risk vs. Low Risk |
| **Diabetes** | 8 Metabolic Markers | XGBoost Gradient Boosting | **~92%** | Positive vs. Negative Diabetic Indicator |
| **Kidney Disease** | 24 Renal Biomarkers | Random Forest Ensemble | **~98%** | CKD Risk vs. Healthy Renal Function |
| **Liver Disease** | 10 Hepatic Parameters | Random Forest Classifier | **~75%** | Hepatic Risk Detected vs. Normal Function |
| **Breast Cancer** | 5 Cellular Nucleus Metrics | Fine-Needle Aspirate Classifier | **~95%** | Malignant Neoplasm vs. Benign Tissue |
| **Malaria** | Thin Blood Smear Images ($50\times50$) | Fast Deep CNN (Keras) | **~95%+** | Parasitized (Infected) vs. Uninfected |
| **Pneumonia** | Chest Radiographs ($150\times150$) | Deep Radiography CNN (Keras) | **~90%+** | Pulmonary Infiltrates vs. Clear Lungs |

---

## 📁 Repository Structure

```
DHARSAN-MediNOVA/
├── app.py                         # Main Flask application, routes, AI Copilot, auth gates
├── ARCHITECTURE.md                # End-to-end system architecture & workflow specification
├── requirements.txt               # Production Python dependencies
├── Dockerfile                     # Multi-stage production container definition
├── render.yaml                    # Cloud deployment specification for Render
├── Procfile                       # Gunicorn WSGI startup configuration
├── .env.example                   # Environment configuration template with safe placeholders
├── .gitignore                     # Git exclusions (private_uploads/, *.db, instance/, *.log)
├── .dockerignore                  # Docker build exclusions
├── medinova_appointments.db       # SQLite database (appointments & sheets_outbox)
├── report_verifications.db        # SQLite database (HMAC seals & report verification)
├── models/                        # Serialized ML & deep learning model weights
│   ├── heart_disease_model.pkl
│   ├── diabetes_model.pkl
│   ├── kidney_model.pkl
│   ├── liver_model.pkl
│   ├── breast_cancer_model.pkl
│   ├── malaria_cnn_model_fast.h5
│   └── pneumonia_model_new.h5
├── private_uploads/               # Session-gated patient scans (HIPAA-aligned privacy)
├── utils/                         # Modular business logic & services
│   ├── diagnostic_predictor.py    # Shared multi-parametric prediction & confidence scoring
│   ├── report_scanner.py          # PDF text extraction & Tesseract OCR pipeline
│   ├── report_verifier.py         # HMAC-SHA256 integrity sealing & QR code generator
│   ├── hospital_service.py        # Specialty mapping, geocoding & progressive radius search
│   ├── verified_hospital_registry.py # Authoritative verified hospital registry & credentials
│   ├── appointment_db.py          # Thread-safe SQLite database & state-machine transitions
│   ├── calendar_service.py        # RFC 5545 iCalendar (.ics) & Google Calendar links
│   ├── appointment_notifier.py    # SMTP coordinator notifications
│   └── google_sheets_logger.py    # Silent asynchronous logging with SQLite durable outbox
├── templates/                     # Jinja2 HTML templates
│   ├── home.html                  # Landing page with live statistics
│   ├── features.html              # Diagnostic suite selector
│   ├── heart.html                 # Cardiovascular assessment & multilingual summary
│   ├── diabetes.html              # Diabetes metabolic panel & multilingual summary
│   ├── kidney.html                # 24-biomarker renal panel & multilingual summary
│   ├── liver.html                 # Hepatic panel & multilingual summary
│   ├── breast_cancer.html         # Oncology FNA morphometry & multilingual summary
│   ├── malaria.html               # Microscopic vision screening & private preview
│   ├── pneumonia.html             # Chest radiograph vision screening & private preview
│   ├── contact.html               # Verified Care Network, Hospital Finder & Appointment Tracking
│   ├── verify_report.html         # Official cryptographic report verification view
│   └── dev_access.html            # Authenticated 2FA developer portal & admin dashboard
├── static/                        # Static assets (obsidian dark theme)
│   ├── style.css                  # Core design tokens, glassmorphism, responsive grid
│   ├── medical_report.js          # Client-side scanner integration, PDF preview, OCR parser
│   ├── chatbot.js                 # AI Clinical Copilot client (TTS, voice recognition)
│   └── samples/                   # Pre-bundled demo images and synthetic lab reports
└── tests/                         # Automated unit & integration regression test suite
    ├── test_audit_fixes.py        # Security, OTP, privacy, outbox, transitions, confidence (22 tests)
    ├── test_care_network.py       # Specialty matching, radius bounds, geocoding (8 tests)
    └── test_scanner_pipeline.py   # OCR, PDF extraction, unit conversions, instant predict (24 tests)
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10 to 3.13
- Pip package manager
- *(Optional)* Tesseract OCR for physical paper scan extraction

### 2. Local Installation

```bash
# 1. Clone repository and navigate to folder
git clone https://github.com/DHARSAN23014208/DHARSAN-MediNOVA.git
cd DHARSAN-MediNOVA

# 2. Create and activate virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure local environment variables
cp .env.example .env

# 5. Start development server
python app.py
```

Open your browser at `http://127.0.0.1:5000`.

---

## 🧪 Automated Testing

MediNova includes a comprehensive automated test suite covering all security, diagnostic, hospital discovery, appointment, and report verification pipelines:

```bash
# Run all test suites
python -m unittest discover tests
```

**Test Execution Summary:**
```text
Ran 54 tests in 3.685s
OK
```
- **`tests/test_audit_fixes.py`** (22 tests): Fail-closed secrets, OTP leak-free 2FA, 60s cooldown, 5-attempt lockout, image upload isolation, hospital coordinate limits, state machine transitions, `None` confidence handling, spreadsheet formula injection defense, HMAC report seals.
- **`tests/test_care_network.py`** (8 tests): Pulmonology isolation, rating $> 3.0$ filter, radius auto-expansion, geocoding fallback.
- **`tests/test_scanner_pipeline.py`** (24 tests): PDF parsing, OCR term matching, unit normalization, instant triage inference.

---

## ⚙️ Environment Variables Reference

| Variable Name | Required? | Default / Example | Purpose |
| :--- | :---: | :--- | :--- |
| `FLASK_ENV` | Prod | `production` / `development` | Enforces fail-closed secret checks when set to `production`. |
| `SECRET_KEY` | **Prod (Mandatory)** | `<64-char-hex>` | Signs Flask user sessions and prevents CSRF. Fails closed if default. |
| `DEV_ACCESS_KEY` | **Prod (Mandatory)** | `<secure-master-key>` | Developer emergency master key. Fails closed if default. |
| `REPORT_SIGNING_SECRET` | **Prod (Mandatory)** | `<256-bit-key>` | Cryptographic secret for HMAC-SHA256 report integrity seals. |
| `AUTHORIZED_DEV_EMAIL` | Yes | `dharshankumarkumar3@gmail.com` | Developer authorized for 2FA OTP delivery and dashboard access. |
| `SMTP_SERVER` / `SMTP_PORT` | Optional | `smtp.gmail.com` / `587` | Outbound mail server for 2FA OTP codes and clinic dispatch. |
| `SMTP_USERNAME` / `SMTP_PASSWORD`| Optional | `<user@gmail.com>` / `<app-pwd>` | SMTP authentication credentials. |
| `GOOGLE_CREDENTIALS_JSON` | Optional | `{"type": "service_account", ...}` | Service account credentials for silent Google Sheets logging. |
| `GOOGLE_SHEETS_SPREADSHEET_ID` | Optional | `<spreadsheet-id>` | Target Google Sheets spreadsheet ID. |

---

## 🐳 Docker Deployment

Build and run using the optimized Docker container:

```bash
# Build Docker image
docker build -t medinova:latest .

# Run container with production secrets
docker run -d -p 5000:5000 \
  -e FLASK_ENV=production \
  -e SECRET_KEY="<production-random-secret-key>" \
  -e DEV_ACCESS_KEY="<production-dev-access-key>" \
  -e REPORT_SIGNING_SECRET="<production-hmac-signing-key>" \
  -e AUTHORIZED_DEV_EMAIL="dharshankumarkumar3@gmail.com" \
  medinova:latest
```

---

## 🛡️ Clinical & Regulatory Advisory

> [!WARNING]
> **CLINICAL DECISION SUPPORT & SCREENING NOTICE**
> 
> MediNova is an artificial intelligence-assisted clinical decision-support and screening platform designed for educational, research, and preliminary risk stratification purposes. It does **not** provide definitive clinical diagnoses, prescribe pharmacological regimens, or supersede clinical judgment by licensed medical practitioners.
> 
> For acute, unstable, or life-threatening symptoms (e.g., crushing chest pain, acute respiratory distress, severe trauma, or altered mental status), contact emergency services (**112 / 911 / 108**) or proceed immediately to the nearest emergency medical facility.

---

<p align="center">
  Developed & Maintained by <strong>DHARSAN KUMAR R</strong>
</p>
