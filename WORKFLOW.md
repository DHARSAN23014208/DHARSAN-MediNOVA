# MediNova End-to-End Operational Workflows & Lifecycle Specification

**Version:** 5.0.0 (Next-Generation Clinical Decision Support & Location Intelligence Edition)  
**Classification:** AI-Assisted Clinical Decision Support Suite (CDSS) & Healthcare Discovery Engine  
**Architect & Maintainer:** DHARSAN KUMAR R  
**Audited & Validated:** October 2026  

---

## 🌟 Executive Workflow Overview

This document specifies the operational lifecycle, data contracts, and control flows governing MediNova. Every workflow is architected for strict clinical safety, defensive fail-closed execution, resource efficiency ($\le 512\,\text{MB}$ container memory), zero patient data leakage, and high reliability across edge conditions.

```mermaid
flowchart TD
    subgraph Ingestion ["1. Patient Triage & Data Ingestion"]
        P1["Manual Clinical Vitals Form"]
        P2["Medical Report Ingestion (PDF / Image OCR)"]
        P3["Direct Microscopic / CXR Upload"]
        P4["Digital ECG Waveform (Lead MLII)"]
    end

    subgraph Governance ["2. Validation & Inference Engine"]
        V1["OOD Physiological Envelope Validation"]
        V2["Unified Model Dispatcher (Lazy Loaded)"]
        V3["Calibrated Clinical Risk Assessment"]
    end

    subgraph PostInference ["3. Results, Sealing & Care Discovery"]
        R1["Multilingual Patient Result Card"]
        R2["HMAC-SHA256 Cryptographic Report Seal & QR"]
        R3["Asynchronous Durable Sheets Outbox Worker"]
        R4["Verified Specialist Care Network Discovery"]
    end

    subgraph HospitalCare ["4. Hospital Routing & Priority Booking"]
        H1["7-Tier Progressive Geocoding Engine"]
        H2["ACTIVE_ORIGIN Central State Manager"]
        H3["Haversine Straight-Line Distance & Auto-Radius"]
        H4["Dynamic Coordinate-First Google Maps Routing"]
        H5["Priority Consultation State Machine (SQLite)"]
    end

    P1 --> V1
    P2 --> V1
    P3 --> V2
    P4 --> V2
    V1 --> V2
    V2 --> V3
    V3 --> R1
    V3 --> R2
    V3 --> R3
    V3 --> R4
    R4 --> H1
    H1 --> H2
    H2 --> H3
    H3 --> H4
    H3 --> H5
```

---

## 📋 Comprehensive Workflow Index

1. [Workflow 1: Patient Clinical Screening & Diagnostic Triage](#workflow-1-patient-clinical-screening--diagnostic-triage)
2. [Workflow 2: Medical Report Ingestion & Multi-Format Scanner Pipeline](#workflow-2-medical-report-ingestion--multi-format-scanner-pipeline)
3. [Workflow 3: Unified 7-Tier Progressive Geocoding & Hospital Discovery](#workflow-3-unified-7-tier-progressive-geocoding--hospital-discovery)
4. [Workflow 4: Client-Side Location Management & ACTIVE_ORIGIN State Machine](#workflow-4-client-side-location-management--active_origin-state-machine)
5. [Workflow 5: Priority Consultation Lifecycle & State Transitions](#workflow-5-priority-consultation-lifecycle--state-transitions)
6. [Workflow 6: Asynchronous Google Sheets Outbox Logging](#workflow-6-asynchronous-google-sheets-outbox-logging)
7. [Workflow 7: Cryptographic Report Authentication & Verification](#workflow-7-cryptographic-report-authentication--verification)
8. [Workflow 8: Developer Portal & 2FA Security Governance](#workflow-8-developer-portal--2fa-security-governance)

---

## Workflow 1: Patient Clinical Screening & Diagnostic Triage

### Objective
Enable patients and clinicians to submit multi-parametric biomarkers or physiological data, validate clinical plausibility, perform calibrated machine learning inference, and deliver multi-lingual risk stratifications.

### Sequence Diagram
```mermaid
sequenceDiagram
    autonumber
    actor User as Patient / Clinician
    participant Browser as Clinical UI
    participant Route as Flask Route Handler
    participant Val as Clinical Validation Gateway
    participant Registry as Model Registry
    participant Model as Calibrated ML / CNN Model
    participant Outbox as SQLite Sheets Outbox
    participant View as Result Card View

    User->>Browser: Enters Biomarkers / Submits Form
    Browser->>Route: POST /<condition> (with CSRF Token)
    Route->>Route: Verify Session CSRF Token (secrets.compare_digest)
    Route->>Val: Validate Biomarker Envelope (utils/clinical_validation.py)
    alt Value Out of Bounds / Unphysical
        Val-->>Route: Validation Error (e.g., Blood Pressure > 300 mmHg)
        Route-->>Browser: Flash Error Banner (Safe Rejection)
    else Plausible Physiological Metrics
        Val-->>Route: Validated Parameter Vector
        Route->>Registry: Request Model (utils/model_registry.py)
        Registry->>Model: Lazy Load Artifact (if not cached)
        Route->>Model: Execute Prediction & Probability Calibration
        Model-->>Route: Raw Class, Calibrated Probability / Confidence
        Route->>Outbox: Enqueue Record in sheets_outbox (status='QUEUED')
        Route->>View: Render Result Card with Dynamic Gauges & Disclaimer
        View-->>Browser: Return HTML (obsidian theme, multilingual toggles)
        Browser-->>User: Visual Triage Result Displayed
    end
```

### Key Safety Invariants
1. **Zero Hardcoded Confidence Fallbacks:** Discrete categorical models explicitly return `confidence_score = None`, rendered as *"Not Stated (Categorical Assessment)"* without fabricated 95% figures.
2. **Physiological Envelope Gating:** Extreme values (e.g., negative age, blood pressure > 350, glucose > 1200) trigger fail-safe rejection before hitting model matrices.
3. **Emergency Red-Flag Escalation:** Extreme indicators (e.g., resting BP > 180, severe ST depression, troponin elevation) trigger emergency alert banners with instant telephone hotline dialing (`112 / 911 / 108`).

---

## Workflow 2: Medical Report Ingestion & Multi-Format Scanner Pipeline

### Objective
Convert unstructured laboratory reports (digital PDFs or photographed paperwork) into validated structured parameters and optionally trigger immediate zero-click diagnostic inference.

### Sequence Diagram
```mermaid
sequenceDiagram
    autonumber
    actor User as Patient / Clinician
    participant UI as Report Scanner Modal
    participant API as Report Processing API (/api/reports/process)
    participant Engine as Scanner Engine (utils/report_scanner.py)
    participant OCR as Tesseract OCR / pypdf Parser
    participant Form as Diagnostic Form UI

    User->>UI: Uploads Lab Report (PDF / PNG / JPG)
    UI->>API: POST /api/reports/process (Multipart Form Data)
    API->>API: Verify File Extension & Size (<= 16 MB)
    API->>Engine: extract_biomarkers_from_file(file_stream, condition)
    alt Digital PDF Document
        Engine->>OCR: Parse Digital Streams via pypdf
    else Scanned Document or Image
        Engine->>OCR: Grayscale Preprocessing & Tesseract OCR Analysis
    end
    OCR-->>Engine: Raw Extracted Text
    Engine->>Engine: Match Biomarker Regex Patterns & Synonyms
    Engine->>Engine: Normalize Laboratory Units (e.g., mmol/L -> mg/dL)
    Engine-->>API: Extracted Parameters Dict + Confidence Level
    alt Instant Predict Mode Requested
        API->>API: Dispatch to Diagnostic Predictor
        API-->>UI: JSON { success: true, instant_prediction: {...}, extracted: {...} }
        UI->>UI: Render Completed Result Card
    else Form Pre-fill Mode
        API-->>UI: JSON { success: true, extracted: {...} }
        UI->>Form: Auto-Populate Form Input Fields with Highlight Glow
    end
```

### Supported Biomarker Normalizations
- **Glucose / Blood Sugar:** $\text{mmol/L} \times 18.0182 \to \text{mg/dL}$
- **Serum Creatinine:** $\mu\text{mol/L} \div 88.42 \to \text{mg/dL}$
- **Total Bilirubin:** $\mu\text{mol/L} \div 17.1 \to \text{mg/dL}$
- **Cholesterol:** $\text{mmol/L} \times 38.67 \to \text{mg/dL}$
- **Hemoglobin:** $\text{mmol/L} \times 1.6114 \to \text{g/dL}$

---

## Workflow 3: Unified 7-Tier Progressive Geocoding & Hospital Discovery

### Objective
Ensure 100% resilient address resolution across noisy Indian road naming conventions (e.g., `Saveetha Engineering College, Saveetha Nagar, (NH-4), Sriperumbudur Taluk, Kanchipuram District, Chennai, Tamil Nadu, 602105`), match disease-specific medical specialties, enforce non-clinical exclusion rules, compute exact straight-line Haversine distances, and construct coordinate-first Google Maps navigation URLs.

### Address Decomposition & Fallback Hierarchy
```
Input Address: "Saveetha Engineering College, Saveetha Nagar, (NH-4), Sriperumbudur Taluk, Kanchipuram District, Chennai, Tamil Nadu, 602105"
                                      │
                                      ▼
                      [Tier 0: normalize_address()]
              Strip plus-codes, normalize Unicode dashes,
             clean highway parentheticals "(NH-4)" -> "NH-4",
                     remove duplicate commas/spaces
                                      │
                                      ▼
                      [Tier 1: Exact Normalized Query]
     "Saveetha Engineering College, Saveetha Nagar, NH-4, Sriperumbudur Taluk, ..."
                                      │ (If 404 / empty)
                                      ▼
                      [Tier 2: Country-Biased Query]
         "Saveetha Engineering College, Saveetha Nagar, NH-4, ..., India"
                                      │ (If 404 / empty)
                                      ▼
                  [Tier 3: Locality Without Road Noise]
        "Saveetha Nagar, Sriperumbudur Taluk, Kanchipuram District, Chennai, TN"
                                      │ (If 404 / empty)
                                      ▼
               [Tier 4: Locality + District/Town + State]
               "Saveetha Nagar, Sriperumbudur, Tamil Nadu"
                                      │ (If 404 / empty)
                                      ▼
                    [Tier 5: District/Town + State]
                     "Sriperumbudur, Tamil Nadu"
                                      │ (If 404 / empty)
                                      ▼
                       [Tier 6: 6-Digit Indian PIN]
                             "602105, India"
                                      │ (If 404 / empty)
                                      ▼
                 [Tier 7: Approximate Major City Center]
         "Chennai, Tamil Nadu" (Explicitly flagged: approximateCityCenter=true)
```

### Complete Hospital Discovery Flow
```mermaid
sequenceDiagram
    autonumber
    actor User as Patient
    participant UI as Hospital Discovery Table (contact.html)
    participant Proxy as Geocode Proxy API (/api/hospitals/geocode)
    participant GeoService as Geocoding Engine (utils/hospital_service.py)
    participant Nominatim as OpenStreetMap Nominatim
    participant Photon as Komoot Photon Geocoder
    participant Registry as Verified Hospital Registry
    participant GMaps as Google Maps Directions API

    User->>UI: Types Address & Clicks "Search Location"
    UI->>UI: Set Search State (Loading Spinner, Disable Inputs)
    UI->>Proxy: POST /api/hospitals/geocode { address: "..." }
    Proxy->>GeoService: resolve_any_address(address)
    GeoService->>GeoService: normalize_address() & generate_progressive_candidates()
    loop Iterate Tiers 1 through 7
        GeoService->>Nominatim: Query Candidate (timeout=6.0s, User-Agent compliant)
        alt Nominatim Succeeds (200 OK with Results)
            Nominatim-->>GeoService: [ { lat, lon, display_name } ]
        else Nominatim Fails / Times Out
            GeoService->>Photon: Query Candidate Fallback (timeout=5.0s)
            Photon-->>GeoService: [ { geometry: { coordinates: [lon, lat] } } ]
        end
    end
    alt Coordinates Resolved
        GeoService-->>Proxy: { success: true, latitude, longitude, queryUsed, resolutionLevel, ... }
        Proxy-->>UI: HTTP 200 JSON Response
        UI->>UI: Update ACTIVE_ORIGIN (coords, query, resolvedAddress, source='address_search')
        UI->>UI: fetchHospitals(condition, radius_km)
        UI->>Registry: Filter Hospitals by Specialty (e.g., Pneumonia -> Pulmonology)
        UI->>Registry: Exclude Non-Clinical Facilities (Ayurveda, Homeopathy, Dental, Optical)
        UI->>UI: Calculate Straight-Line Haversine Distance (from ACTIVE_ORIGIN coords)
        UI->>UI: Sort Hospitals Ascending by Distance
        UI->>UI: Generate Navigation Links: origin=LAT%2CLNG&destination=DEST&travelmode=driving
        UI-->>User: Render Verified Hospitals Table with Active Origin Badge & Live Distance
    else All Tiers Fail
        GeoService-->>Proxy: { success: false, error: "Address could not be resolved" }
        Proxy-->>UI: HTTP 404 JSON Response
        UI->>UI: clearActiveOrigin()
        UI->>UI: Empty Table: ALL_RECOMMENDATIONS = []
        UI-->>User: Display Clean "Location Not Found" Notice (Zero Stale Data)
    end
```

---

## Workflow 4: Client-Side Location Management & ACTIVE_ORIGIN State Machine

### Objective
Provide a unified, single source of truth for the patient's geographic origin in `templates/contact.html`, eliminating race conditions between manual text input, browser GPS geolocation, and quick-metro city buttons.

### Centralized State Schema (`ACTIVE_ORIGIN`)
```javascript
let ACTIVE_ORIGIN = {
    userOriginCoords: null,            // [lat, lng] or null
    userOriginAddress: '',             // Raw user input or button label
    userOriginResolvedAddress: '',     // Canonical geocoded address string
    userOriginSource: 'none',          // 'browser_geolocation' | 'address_search' | 'metro_quick_button' | 'approximate_city_center'
    userOriginPrecision: 'unresolved'  // 'gps_exact' | 'geocoded_high' | 'geocoded_locality' | 'geocoded_pincode' | 'city_approximate'
};
```

### Transition Matrix
| Trigger Action | Input Data | Target Origin Coordinates | Target Source | Target Precision | Recommendations Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Browser GPS** | `navigator.geolocation.getCurrentPosition` | `[coords.latitude, coords.longitude]` | `'browser_geolocation'` | `'gps_exact'` | Re-sort & render with exact distances |
| **Manual Search (Exact)** | Text Address string | Geocoded `[res.latitude, res.longitude]` | `'address_search'` | `'geocoded_high'` | Re-sort & render from resolved coordinates |
| **Manual Search (PIN)** | 6-Digit PIN Code | Geocoded `[res.latitude, res.longitude]` | `'address_search'` | `'geocoded_pincode'` | Re-sort & render from centroid |
| **Quick Metro Button** | Chennai, Delhi, Mumbai, etc. | Preset Metro Coordinates | `'metro_quick_button'` | `'geocoded_high'` | Populate address field, re-sort & render |
| **Lookup Failure** | Unresolvable gibberish | `null` | `'none'` | `'unresolved'` | **`clearActiveOrigin()`**: Clear table, set `ALL_RECOMMENDATIONS = []`, display error banner |

---

## Workflow 5: Priority Consultation Lifecycle & State Transitions

### Objective
Govern patient appointment requests through an auditable, persistent state machine in SQLite (`medinova_appointments.db`), enforcing authorization boundaries, patient privacy, and notification triggers.

### State Transition Diagram
```mermaid
stateDiagram-v2
    [*] --> AwaitingHospitalConfirmation: Patient Submits Consultation Form
    
    AwaitingHospitalConfirmation --> ConfirmedByHospital: Coordinator Confirms Date & Time
    AwaitingHospitalConfirmation --> ReschedulingRequired: Hospital Slot Conflict
    AwaitingHospitalConfirmation --> Cancelled: Patient Cancels Request
    AwaitingHospitalConfirmation --> Rejected: Out-of-Scope Condition / Overcapacity

    ReschedulingRequired --> ConfirmedByHospital: New Slot Agreed with Patient
    ReschedulingRequired --> AwaitingHospitalConfirmation: Patient Resubmits Preferences
    ReschedulingRequired --> Cancelled: Patient Declines Alternative

    ConfirmedByHospital --> Completed: Consultation Conducted
    ConfirmedByHospital --> ReschedulingRequired: Urgent Hospital Physician Conflict
    ConfirmedByHospital --> Cancelled: Patient Cancels Prior to Visit

    Cancelled --> [*]: Terminal State
    Completed --> [*]: Terminal State
    Rejected --> [*]: Terminal State
```

### Lifecycle Invariants
1. **Cryptographic Reference ID:** Every appointment is assigned an indexed, collision-resistant identifier formatted as `APT-YYYY-XXXXXX` (e.g., `APT-2026-B8E19F`).
2. **Dual-Factor Record Isolation:** Accessing appointment details or tracking status requires both the Reference ID and the exact matching patient phone number or email address.
3. **Digital Calendar Export:** Confirmed appointments generate standard RFC 5545 `.ics` downloadable files and dynamic 1-click Google Calendar web scheduling URLs.
4. **Terminal State Immutability:** Appointments in `Completed`, `Cancelled`, or `Rejected` states cannot transition back to active states without supervisor intervention.

---

## Workflow 6: Asynchronous Google Sheets Outbox Logging

### Objective
Ensure durable, zero-latency clinical logging to private Google Sheets worksheets (`Master_Log` and condition-specific sheets) without blocking diagnostic response times or losing records during network drops.

### Architecture & Sequence Flow
```mermaid
sequenceDiagram
    autonumber
    actor Patient as Patient Browser
    participant Flask as Flask Controller
    participant Outbox as SQLite Table (sheets_outbox)
    participant MemQueue as In-Memory Queue (max=2000)
    participant Worker as Background Daemon Thread
    participant Sheets as Google Sheets API v4

    Patient->>Flask: POST /<condition> (Inference Request)
    Flask->>Flask: Run Model & Calibrate Probability
    Flask->>Outbox: INSERT Record (status='QUEUED', module_appended=0, master_appended=0)
    Flask->>MemQueue: put_nowait(payload)
    Flask-->>Patient: HTTP 200 OK (Returns Diagnostic Result Instantly)

    Worker->>MemQueue: get() Task Payload
    Worker->>Outbox: UPDATE status='IN_PROGRESS'
    Worker->>Sheets: Append Row to Condition Sheet (e.g., Heart_Disease)
    Sheets-->>Worker: HTTP 200 OK (module_appended=1)
    Worker->>Outbox: UPDATE module_appended=1
    Worker->>Sheets: Append Row to Master_Log Sheet
    Sheets-->>Worker: HTTP 200 OK (master_appended=1)
    Worker->>Outbox: UPDATE status='SYNCHRONIZED', updated_at=CURRENT_TIMESTAMP
```

### Defensive Reliability Rules
- **Formula Injection Defense:** All string values starting with formula trigger tokens (`=`, `+`, `-`, `@`, `\t`, `\r`) are automatically escaped with a leading single quote (`'`), neutralizing spreadsheet calculation vulnerabilities.
- **Idempotent Retry Guarantee:** If appending to `Master_Log` fails after `Condition_Sheet` succeeds, subsequent retries only push to `Master_Log`, eliminating duplicate rows.
- **Queue Saturation Safety:** If the in-memory queue exceeds its 2,000-item buffer, records are saved directly to `sheets_outbox` with status `QUEUED` for sequential batch pickup.

---

## Workflow 7: Cryptographic Report Authentication & Verification

### Objective
Provide tamper-evident cryptographic sealing for clinical screening reports, enabling public verification via dynamic QR codes and anti-counterfeit audit registries.

### Verification Flow
```mermaid
sequenceDiagram
    autonumber
    actor Clinician as Clinician / Patient
    participant System as MediNova Core
    participant Verifier as HMAC-SHA256 Signer
    participant DB as SQLite (report_verifications.db)
    participant Auditor as External Auditor / Doctor

    Clinician->>System: Generates Assessment Report
    System->>Verifier: seal_report(report_id, condition, prediction, confidence)
    Verifier->>Verifier: Compute HMAC-SHA256(SECRET, payload)
    Verifier->>DB: INSERT Record (report_id, hash, prediction, confidence, timestamp)
    Verifier-->>System: Cryptographic Seal & Verification URL
    System->>System: Generate High-Resolution QR Code (/api/report-qr/<id>)
    System-->>Clinician: Downloadable PDF / Digital Report with Embedded QR Seal

    Auditor->>System: Scans QR Code / Navigates to /verify-report/<id>
    System->>DB: SELECT Verification Record by report_id
    alt Record Found & Seal Valid
        System-->>Auditor: Verified Report View (Timestamp, Biomarkers, Seal Evidence)
    else Invalid ID / Modified Seal
        System-->>Auditor: Cryptographic Tampering Warning Banner (Fail Closed)
    end
```

---

## Workflow 8: Developer Portal & 2FA Security Governance

### Objective
Protect administrative diagnostic controls and sensitive patient records through hardened two-factor authentication (2FA), constant-time key validation, rate-limiting, and zero-leakage logging.

### Developer Authentication State Machine
```mermaid
stateDiagram-v2
    [*] --> Unauthenticated: GET /developer
    Unauthenticated --> ValidatingKey: Submits Master Access Key
    
    ValidatingKey --> Unauthenticated: Key Mismatch (HTTP 403)
    ValidatingKey --> OTPSent: Key Matches (secrets.compare_digest)
    
    OTPSent --> VerifyingOTP: Submits 6-Digit Verification Code
    VerifyingOTP --> SessionActive: OTP Matches & Within 10m TTL
    VerifyingOTP --> OTPSent: Incorrect Code (Attempts < 5)
    VerifyingOTP --> LockedOut: 5 Consecutive Failed Attempts
    
    SessionActive --> EphemeralTimeout: 30 Minutes Inactivity
    EphemeralTimeout --> Unauthenticated: Session Invalidated
    LockedOut --> Unauthenticated: Requires Master Key Restart & New OTP
```

### Security Guardrails
1. **Zero Console Leaks:** Generated 6-digit OTP codes are dispatched solely via authenticated SMTP to `AUTHORIZED_DEV_EMAIL`. OTP values are strictly excluded from console logs, stdout, error traces, and HTTP responses.
2. **60-Second Cooldown:** Developers cannot request another OTP within 60 seconds of a previous request.
3. **5-Attempt Lockout:** 5 consecutive failed code submissions immediately invalidate the active token and require re-authentication from the master access key.
4. **Constant-Time Comparison:** All secret comparisons use `secrets.compare_digest()` to eliminate timing-attack vulnerabilities.

---

<p align="center">
  <strong>MediNova Clinical Decision Support Suite</strong> • Engineered for Precision, Resilience & Security
</p>
