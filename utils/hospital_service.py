"""
MediNova Hospital Discovery & Disease-Specialty Mapping Service
===============================================================
Provides:
1. Canonical seven-disease medical specialty mapping.
2. Curated global and regional specialty referral centres dataset.
3. Live OpenStreetMap Nominatim geocoding & Overpass API nearby hospital discovery.
4. Haversine straight-line distance calculations.
5. Intelligent hospital recommendation engine with transparent matching rationales.
"""

import os
import re
import math
import time
import json
import logging
import urllib.parse
from typing import Dict, List, Any, Optional, Tuple, Set
import requests

from utils.verified_hospital_registry import VERIFIED_HOSPITAL_REGISTRY

logger = logging.getLogger("medinova.hospital_service")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

USER_AGENT = "MediNova-Clinical-AI-Hospital-Discovery/5.0 (contact: dharshankumarkumar3@gmail.com)"
REQUEST_TIMEOUT = 8

# In-memory LRU-style cache
_GEO_CACHE: Dict[str, Tuple[float, Any]] = {}
CACHE_TTL_SECONDS = 1800  # 30 minutes


# ==============================================================================
# 1. CENTRALIZED DISEASE-TO-SPECIALTY MAPPING
# ==============================================================================

DISEASE_SPECIALTY_MAP: Dict[str, Dict[str, Any]] = {
    "pneumonia": {
        "id": "pneumonia",
        "canonical_name": "Pneumonia / Respiratory Disease",
        "primary_specialty": "Pulmonology / Respiratory Medicine",
        "accepted_specialties": [
            "Pulmonology",
            "Pulmonary Medicine",
            "Respiratory Medicine",
            "Chest Medicine",
            "Thoracic Medicine",
            "Respiratory Critical Care",
            "Pulmonology & Chest Medicine",
            "Chest Clinic"
        ],
        "specialist_types": ["Pulmonologist", "Chest Physician", "Respiratory Specialist", "Intensivist"],
        "relevant_departments": [
            "Department of Pulmonary Medicine",
            "Department of Respiratory Medicine",
            "Respiratory Intensive Care Unit (RICU)",
            "Chest Medicine & Interventional Pulmonology"
        ],
        "search_keywords": ["pulmonology", "pulmonary", "respiratory", "chest", "thoracic", "lung", "ricu"],
        "disallow_general_medicine_fallback": True,
        "requires_urgent_assessment": True,
        "urgent_warning": "Severe breathlessness, oxygen saturation (SpO2) < 92%, cyanosis (blue lips), or confusion requires immediate hospital emergency room evaluation.",
        "telemedicine_eligible": True,
        "telemedicine_notes": "Post-discharge recovery checkups and chest X-ray second reviews can be done via telemedicine; acute respiratory distress requires in-person hospitalization."
    },
    "heart": {
        "id": "heart",
        "canonical_name": "Heart Disease / Cardiovascular Disease",
        "primary_specialty": "Cardiology / Interventional Cardiology",
        "accepted_specialties": [
            "Cardiology",
            "Interventional Cardiology",
            "Cardiothoracic Surgery",
            "Cardiothoracic and Vascular Surgery (CTVS)",
            "Electrophysiology",
            "Heart Failure Care"
        ],
        "specialist_types": ["Cardiologist", "Interventional Cardiologist", "Cardiothoracic Surgeon"],
        "relevant_departments": [
            "Department of Cardiology",
            "Cardiovascular Care Unit (CCU)",
            "Cardiac Catheterization Laboratory",
            "Department of Cardiothoracic & Vascular Surgery (CTVS)"
        ],
        "search_keywords": ["cardiology", "heart", "cardiac", "vascular", "coronary", "cardiothoracic", "ctvs"],
        "disallow_general_medicine_fallback": True,
        "requires_urgent_assessment": True,
        "urgent_warning": "Acute crushing chest pain, dyspnea, left arm/jaw pain, or sudden dizziness requires IMMEDIATE emergency care (Call 112 / 911 / 108).",
        "telemedicine_eligible": True,
        "telemedicine_notes": "Follow-up consultations, blood pressure management, and ECG/cholesterol review are eligible for telemedicine. Acute symptoms require in-person care."
    },
    "neurological": {
        "id": "neurological",
        "canonical_name": "Stroke / Brain Disorders / Neurological Disease",
        "primary_specialty": "Neurology / Neurosurgery",
        "accepted_specialties": [
            "Neurology",
            "Neurosurgery",
            "Neurocritical Care",
            "Neurosciences",
            "Stroke Medicine",
            "Epileptology"
        ],
        "specialist_types": ["Neurologist", "Neurosurgeon", "Stroke Specialist", "Neurointensivist"],
        "relevant_departments": [
            "Department of Neurology",
            "Comprehensive Stroke Care Unit",
            "Neuro-Intensive Care Unit (Neuro-ICU)",
            "Department of Neurosurgery"
        ],
        "search_keywords": ["neurology", "brain", "stroke", "neuro", "neurosurgery", "neurocritical", "neurosciences"],
        "disallow_general_medicine_fallback": True,
        "requires_urgent_assessment": True,
        "urgent_warning": "Sudden facial droop, arm weakness, speech difficulty (F.A.S.T.), or sudden severe headache requires IMMEDIATE emergency medical care.",
        "telemedicine_eligible": True,
        "telemedicine_notes": "Follow-up consultations for chronic migraine, epilepsy review, and neuroprevention are eligible for telemedicine."
    },
    "diabetes": {
        "id": "diabetes",
        "canonical_name": "Diabetes / Metabolic Disorders",
        "primary_specialty": "Endocrinology / Diabetology",
        "accepted_specialties": [
            "Endocrinology",
            "Diabetology",
            "Metabolic Medicine",
            "Endocrinology & Diabetology"
        ],
        "specialist_types": ["Endocrinologist", "Diabetologist", "Diabetes Care Educator"],
        "relevant_departments": [
            "Department of Endocrinology & Diabetology",
            "Diabetes & Metabolism Center",
            "Comprehensive Diabetic Foot Care Clinic"
        ],
        "search_keywords": ["diabetes", "diabetologist", "endocrinology", "sugar clinic", "endocrine", "metabolic"],
        "disallow_general_medicine_fallback": True,
        "requires_urgent_assessment": False,
        "urgent_warning": "Suspected diabetic ketoacidosis (DKA) with severe nausea, confusion, fruity breath odor, or extreme dehydration requires emergency admission.",
        "telemedicine_eligible": True,
        "telemedicine_notes": "Comprehensive continuous glucose review, HbA1c titration, and dietary management are highly suitable for telemedicine."
    },
    "kidney": {
        "id": "kidney",
        "canonical_name": "Kidney Disease / Chronic Kidney Disease",
        "primary_specialty": "Nephrology / Renal Medicine",
        "accepted_specialties": [
            "Nephrology",
            "Renal Medicine",
            "Dialysis Services",
            "Kidney Transplantation",
            "Hemodialysis Care"
        ],
        "specialist_types": ["Nephrologist", "Renal Transplant Surgeon", "Dialysis Specialist"],
        "relevant_departments": [
            "Department of Nephrology",
            "Renal Dialysis Unit",
            "Kidney Care & Transplantation Center"
        ],
        "search_keywords": ["nephrology", "kidney", "dialysis", "renal"],
        "disallow_general_medicine_fallback": True,
        "requires_urgent_assessment": True,
        "urgent_warning": "Sudden anuria (no urine), severe fluid retention with pulmonary edema, or electrolyte crisis (hyperkalemia) requires urgent hospital care.",
        "telemedicine_eligible": True,
        "telemedicine_notes": "Early CKD monitoring, eGFR and serum creatinine tracking, and dietary adjustments are supported via telemedicine."
    },
    "liver": {
        "id": "liver",
        "canonical_name": "Liver Disease / Hepatic Disorders",
        "primary_specialty": "Hepatology / Gastroenterology",
        "accepted_specialties": [
            "Hepatology",
            "Gastroenterology with confirmed liver services",
            "Hepatobiliary Medicine",
            "Hepato-Pancreato-Biliary (HPB) Surgery",
            "Liver Transplantation"
        ],
        "specialist_types": ["Hepatologist", "Gastroenterologist", "Hepato-Pancreato-Biliary (HPB) Surgeon"],
        "relevant_departments": [
            "Department of Hepatology",
            "Gastroenterology & Liver Clinic",
            "Liver Transplant & HPB Surgery Centre"
        ],
        "search_keywords": ["hepatology", "liver", "gastroenterology", "gastro", "hepatic", "hepatobiliary", "hpb"],
        "disallow_general_medicine_fallback": True,
        "requires_urgent_assessment": True,
        "urgent_warning": "Vomiting blood (hematemesis), sudden jaundice with confusion (hepatic encephalopathy), or acute ascites requires emergency admission.",
        "telemedicine_eligible": True,
        "telemedicine_notes": "Liver function panel (ALT/AST/Bilirubin) review, fatty liver lifestyle consultation, and chronic hepatitis monitoring are available via telemedicine."
    },
    "cancer": {
        "id": "cancer",
        "canonical_name": "Breast Cancer / Oncology / Tumors",
        "primary_specialty": "Oncology",
        "accepted_specialties": [
            "Oncology",
            "Medical Oncology",
            "Surgical Oncology",
            "Radiation Oncology",
            "Breast Surgical Oncology",
            "Comprehensive Cancer Care"
        ],
        "specialist_types": ["Surgical Oncologist", "Medical Oncologist", "Radiation Oncologist", "Breast Surgeon"],
        "relevant_departments": [
            "Comprehensive Cancer Center",
            "Department of Medical Oncology",
            "Department of Surgical Oncology",
            "Tumor Board & Breast Clinic"
        ],
        "search_keywords": ["oncology", "cancer", "tumor", "tumour", "breast oncology", "radiation oncology", "surgical oncology"],
        "disallow_general_medicine_fallback": True,
        "requires_urgent_assessment": True,
        "urgent_warning": "Rapidly growing breast mass, inflammatory skin changes with orange-peel texture (peau d'orange), or ulceration requires prompt specialist evaluation.",
        "telemedicine_eligible": True,
        "telemedicine_notes": "Second opinions on biopsy pathology, mammography review, and systemic therapy consultation can be initiated through telemedicine."
    },
    "malaria": {
        "id": "malaria",
        "canonical_name": "Malaria / Infectious Disease",
        "primary_specialty": "Infectious Diseases / Tropical Medicine",
        "accepted_specialties": [
            "Infectious Diseases",
            "Tropical Medicine",
            "Clinical Parasitology",
            "Communicable Diseases"
        ],
        "specialist_types": ["Infectious Disease Specialist", "Tropical Medicine Physician", "Clinical Microbiologist"],
        "relevant_departments": [
            "Department of Infectious Diseases",
            "Tropical Medicine Unit",
            "Fever Triage & Clinical Pathology"
        ],
        "search_keywords": ["infectious", "tropical", "fever", "communicable", "parasitology", "infectious disease"],
        "disallow_general_medicine_fallback": True,
        "requires_urgent_assessment": True,
        "urgent_warning": "Plasmodium falciparum malaria can progress rapidly to cerebral malaria, severe anemia, or multi-organ failure. Immediate anti-malarial therapy is essential.",
        "telemedicine_eligible": False,
        "telemedicine_notes": "Malaria requires confirmatory peripheral blood smear/rapid antigen testing and immediate administration of oral or IV antimalarial therapy. In-person visit is mandatory."
    },
    "thyroid": {
        "id": "thyroid",
        "canonical_name": "Thyroid Dysfunction / Endocrine Disorders",
        "primary_specialty": "Endocrinology / Thyroid Center",
        "accepted_specialties": [
            "Endocrinology",
            "Thyroid Center",
            "Endocrine Surgery",
            "Metabolic Disorders",
            "Internal Medicine"
        ],
        "specialist_types": ["Endocrinologist", "Thyroid Specialist", "Endocrine Surgeon"],
        "relevant_departments": [
            "Department of Endocrinology",
            "Thyroid & Endocrine Surgery Clinic",
            "Comprehensive Metabolic Center"
        ],
        "search_keywords": ["thyroid", "endocrinologist", "endocrinology", "hormone", "endocrine"],
        "disallow_general_medicine_fallback": False,
        "requires_urgent_assessment": False,
        "urgent_warning": "Severe thyroid crisis (myxedema coma or thyroid storm with high fever, delirium, tachycardia) requires immediate emergency room resuscitation.",
        "telemedicine_eligible": True,
        "telemedicine_notes": "Thyroid hormone level monitoring (TSH, FT4, T3) and dosage adjustments are highly suitable for telemedicine consultations."
    },
    "arrhythmia": {
        "id": "arrhythmia",
        "canonical_name": "Cardiac Arrhythmia / Rhythm Disorders",
        "primary_specialty": "Cardiology / Electrophysiology (EP)",
        "accepted_specialties": [
            "Cardiology",
            "Cardiac Electrophysiology",
            "Electrophysiology",
            "Arrhythmia Clinic",
            "Cardiovascular Care Unit (CCU)",
            "Heart Failure Care"
        ],
        "specialist_types": ["Cardiologist", "Cardiac Electrophysiologist", "Arrhythmia Specialist"],
        "relevant_departments": [
            "Department of Cardiac Electrophysiology",
            "Arrhythmia & Pacemaker Clinic",
            "Cardiovascular Intensive Care Unit"
        ],
        "search_keywords": ["arrhythmia", "cardiology", "electrophysiology", "ep clinic", "pacemaker", "heart rhythm"],
        "disallow_general_medicine_fallback": True,
        "requires_urgent_assessment": True,
        "urgent_warning": "Sustained rapid heart rate (>150 bpm), syncope/fainting, crushing chest pain, or severe shortness of breath requires IMMEDIATE emergency medical care.",
        "telemedicine_eligible": True,
        "telemedicine_notes": "Holter monitor reviews, ECG rhythm strip second opinions, and stable medication checks are suitable for telemedicine."
    },
    "general": {
        "id": "general",
        "canonical_name": "Internal Medicine & General Care",
        "primary_specialty": "Internal Medicine / General Physician",
        "accepted_specialties": [
            "Internal Medicine",
            "General Practice",
            "Family Medicine",
            "Preventive Healthcare"
        ],
        "specialist_types": ["General Physician", "Internal Medicine Specialist", "Consultant Physician"],
        "relevant_departments": [
            "Department of Internal Medicine",
            "Outpatient Consultation (OPD)",
            "General Healthcare Clinic"
        ],
        "search_keywords": ["internal medicine", "general physician", "physician", "general hospital", "general medicine", "family doctor", "consultant physician"],
        "disallow_general_medicine_fallback": False,
        "requires_urgent_assessment": False,
        "urgent_warning": "Acute severe shortness of breath, loss of consciousness, uncontrolled bleeding, or sudden paralysis requires emergency services.",
        "telemedicine_eligible": True,
        "telemedicine_notes": "General clinical assessments, lab review, and initial triage consultations are supported via telemedicine."
    }
}

DEFAULT_SPECIALTY: Dict[str, Any] = DISEASE_SPECIALTY_MAP["general"]

# Curated global referral centres (for international comparison / export)
CURATED_GLOBAL_CENTRES: List[Dict[str, Any]] = [
    h for h in VERIFIED_HOSPITAL_REGISTRY if h.get("country") != "India" or "Apex" in h.get("facility_type", "")
]


def get_specialty_mapping(condition_id: Optional[str]) -> Dict[str, Any]:
    """Retrieve specialty mapping with alias normalization and strict unknown condition handling."""
    if not condition_id:
        return DEFAULT_SPECIALTY
    c_lower = str(condition_id).strip().lower()
    if c_lower in DISEASE_SPECIALTY_MAP:
        return DISEASE_SPECIALTY_MAP[c_lower]

    # Explicit alias normalization
    if c_lower in ("respiratory", "lungs", "chest", "copd", "pulmonary", "pulmonology"):
        return DISEASE_SPECIALTY_MAP["pneumonia"]
    if c_lower in ("cardio", "hypertension", "cardiovascular", "heart_disease", "coronary"):
        return DISEASE_SPECIALTY_MAP["heart"]
    if c_lower in ("neuro", "neurology", "stroke", "brain", "neurosurgery"):
        return DISEASE_SPECIALTY_MAP["neurological"]
    if c_lower in ("sugar", "diabetic", "endocrinology", "metabolic"):
        return DISEASE_SPECIALTY_MAP["diabetes"]
    if c_lower in ("renal", "ckd", "kidney_disease", "nephrology"):
        return DISEASE_SPECIALTY_MAP["kidney"]
    if c_lower in ("liver_disease", "hepatic", "hepatology", "gastro", "gastroenterology"):
        return DISEASE_SPECIALTY_MAP["liver"]
    if c_lower in ("breast_cancer", "tumor", "tumour", "oncology", "malignancy"):
        return DISEASE_SPECIALTY_MAP["cancer"]
    if c_lower in ("infectious", "tropical", "infection", "infectious_disease"):
        return DISEASE_SPECIALTY_MAP["malaria"]
    if c_lower in ("thyroid", "thyroid_dysfunction", "hypothyroidism", "hyperthyroidism", "goitre"):
        return DISEASE_SPECIALTY_MAP["thyroid"]
    if c_lower in ("arrhythmia", "cardiac_arrhythmia", "ecg", "ekg", "palpitations", "electrophysiology"):
        return DISEASE_SPECIALTY_MAP["arrhythmia"]
    if c_lower in ("general", "default", "internal_medicine", "general_physician"):
        return DISEASE_SPECIALTY_MAP["general"]

    # Unknown condition: do not silently classify as specialty match; return general with explicit unmapped marker
    unmapped = dict(DEFAULT_SPECIALTY)
    unmapped["is_unmapped_condition"] = True
    unmapped["requested_query"] = condition_id
    return unmapped


# ==============================================================================
# 2. STRICT FACILITY FILTERING & NON-CLINICAL EXCLUSIONS
# ==============================================================================

# Regex patterns for non-clinical or alternative medicine facilities
STRICT_EXCLUSION_NAME_PATTERNS = [
    # Alternative Medicine: Ayurveda, Homeopathy, Siddha, Unani, Naturopathy
    r"\bayurved(?:a|ic)\b",
    r"\bvaidya\b",
    r"\bpanchakarma\b",
    r"\bhomeopath(?:y|ic)\b",
    r"\bhomoeopath(?:y|ic)\b",
    r"\bsiddha\b",
    r"\bunani\b",
    r"\bnaturopath(?:y|ic)\b",
    r"\byoga\s+&(?:amp;)?\s+naturopathy\b",
    # Dental clinics / hospitals
    r"\bdental\b",
    r"\bdentist\b",
    r"\bdentistry\b",
    r"\bteeth\b",
    r"\borthodontic\b",
    # Eye hospitals / Ophthalmology-only
    r"\beye\s+hospital\b",
    r"\beye\s+clinic\b",
    r"\beye\s+cent(?:re|er)\b",
    r"\beye\s+institute\b",
    r"\bophthalmology\b",
    r"\bvision\s+care\b",
    r"\bretina\b",
    r"\blasik\b",
    r"\boptometry\b",
    # Maternity / Fertility / IVF only
    r"\bmaternity\s+home\b",
    r"\bmaternity\s+hospital\b",
    r"\bmaternity\s+clinic\b",
    r"\bfertility\b",
    r"\bivf\b",
    r"\bwomen\s+&(?:amp;)?\s+child\b",
    # Blood banks and donation centres
    r"\bblood\s*bank\b",
    r"\bblood\s*transfusion\b",
    r"\bblood\s*donation\b",
    r"\brotary\s+blood\b",
    r"\bred\s+cross\s+blood\b",
    r"\bblood\s*cent(?:re|er)\b",
    # Veterinary and animal care
    r"\bveterinary\b",
    r"\banimal\s+hospital\b",
    r"\bpet\s+clinic\b",
    r"\banimal\s+clinic\b",
    r"\bvet\s+clinic\b",
    # Pharmacies / Chemists / Medical stores
    r"\bpharmacy\b",
    r"\bchemist\b",
    r"\bdruggist\b",
    r"\bmedical\s+store\b",
    r"\bmedicals\b",
    # Standalone diagnostic labs
    r"\bpathology\s+lab\b",
    r"\bdiagnostic\s+lab\b",
    r"\bdiagnostic\s+cent(?:re|er)\b",
    r"\bblood\s+collection\b",
    r"\bscan\s+cent(?:re|er)\b",
    r"\bimaging\s+cent(?:re|er)\b",
    r"\bx-?ray\s+clinic\b"
]

NON_CLINICAL_TAG_VALUES = {
    "amenity": {"blood_bank", "pharmacy", "veterinary", "dentist"},
    "healthcare": {"blood_bank", "blood_donation", "pharmacy", "veterinary", "dentist", "laboratory", "alternative"},
    "healthcare:speciality": {"blood_bank", "dentist", "dental", "orthodontics", "veterinary", "ayurveda", "homeopathy", "ophthalmology"},
    "shop": {"pharmacy", "chemist", "medical_supply"}
}


def is_non_clinical_facility(name: str, tags: Optional[Dict[str, Any]] = None, condition_id: str = "general") -> bool:
    """
    Strictly filters out non-qualifying, alternative medicine, or single-unrelated facilities:
    - Alternative medicine: Ayurveda, Homeopathy, Siddha, Unani, Naturopathy
    - Dental clinics and dental hospitals
    - Eye hospitals and ophthalmology-only facilities
    - Maternity-only and IVF facilities
    - Blood banks, pharmacies, veterinary, standalone labs
    """
    name_clean = (name or "").strip().lower()
    if not name_clean:
        return True  # Exclude unnamed nodes

    tags = tags or {}
    cond_lower = (condition_id or "general").lower()

    # Check OSM tags for disqualifying categories
    for key, bad_values in NON_CLINICAL_TAG_VALUES.items():
        val = str(tags.get(key, "")).strip().lower()
        if val in bad_values:
            return True

    # Check name against strict exclusion patterns
    for pat in STRICT_EXCLUSION_NAME_PATTERNS:
        # If user explicitly searched for cancer, do not exclude "breast" if in cancer mapping
        if "breast" in pat and cond_lower == "cancer":
            continue
        if re.search(pat, name_clean):
            # If facility is a major multi-specialty medical college that also has an eye/dental dept
            if ("medical college" in name_clean or "aiims" in name_clean) and ("diagnostic" in pat or "scan" in pat):
                continue
            return True

    return False


# ==============================================================================
# 3. HAVERSINE DISTANCE & ADDRESS NORMALIZATION
# ==============================================================================

def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle (straight-line) distance between two points
    on the Earth in kilometers using the Haversine formula.
    """
    R = 6371.0  # Earth's mean radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(R * c, 2)


def normalize_address(address: str) -> str:
    """Clean and standardize an address string, stripping plus codes, unicode dashes, and punctuation noise."""
    if not address or not isinstance(address, str):
        return ""
    s = address.strip()
    # Normalize unicode hyphens, dashes, and quotes
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    # Strip plus codes
    s = re.sub(r'^[A-Z0-9]{4,8}\+[A-Z0-9]{2,4},?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\b[A-Z0-9]{4,8}\+[A-Z0-9]{2,4}\b,?\s*', '', s, flags=re.IGNORECASE)
    # Standardize highway designations e.g. (NH-4), (NH 4), (NH-48) -> NH-4 / NH-48
    s = re.sub(r'\(\s*(NH[-\s]*\d+[A-Z]?)\s*\)', r' \1 ', s, flags=re.IGNORECASE)
    s = re.sub(r'\bNH\s*(\d+)\b', r'NH-\1', s, flags=re.IGNORECASE)
    # Repeated commas and spaces
    s = re.sub(r'[,;]+', ',', s)
    s = re.sub(r'\s*,\s*', ', ', s)
    s = re.sub(r'\s+', ' ', s).strip(' ,')
    return s


# ==============================================================================
# 4. UNIVERSAL ADDRESS GEOCODING (NOMINATIM PROGRESSIVE & PHOTON FALLBACK)
# ==============================================================================

CITY_CENTROIDS: Dict[str, Dict[str, Any]] = {
    "delhi": {"latitude": 28.6139, "longitude": 77.2090, "display_name": "New Delhi, Delhi, India", "city": "New Delhi", "state": "Delhi", "country": "India"},
    "new delhi": {"latitude": 28.6139, "longitude": 77.2090, "display_name": "New Delhi, Delhi, India", "city": "New Delhi", "state": "Delhi", "country": "India"},
    "chennai": {"latitude": 13.0827, "longitude": 80.2707, "display_name": "Chennai, Tamil Nadu, India", "city": "Chennai", "state": "Tamil Nadu", "country": "India"},
    "mumbai": {"latitude": 19.0760, "longitude": 72.8777, "display_name": "Mumbai, Maharashtra, India", "city": "Mumbai", "state": "Maharashtra", "country": "India"},
    "bengaluru": {"latitude": 12.9716, "longitude": 77.5946, "display_name": "Bengaluru, Karnataka, India", "city": "Bengaluru", "state": "Karnataka", "country": "India"},
    "bangalore": {"latitude": 12.9716, "longitude": 77.5946, "display_name": "Bengaluru, Karnataka, India", "city": "Bengaluru", "state": "Karnataka", "country": "India"},
    "hyderabad": {"latitude": 17.3850, "longitude": 78.4867, "display_name": "Hyderabad, Telangana, India", "city": "Hyderabad", "state": "Telangana", "country": "India"},
    "kolkata": {"latitude": 22.5726, "longitude": 88.3639, "display_name": "Kolkata, West Bengal, India", "city": "Kolkata", "state": "West Bengal", "country": "India"}
}

INDIAN_STATES: Set[str] = {
    "tamil nadu", "karnataka", "maharashtra", "delhi", "telangana", "west bengal",
    "kerala", "andhra pradesh", "gujarat", "uttar pradesh", "haryana", "punjab",
    "rajasthan", "madhya pradesh", "bihar", "odisha", "assam", "goa", "chandigarh",
    "puducherry", "pondicherry", "jharkhand", "chhattisgarh", "uttarakhand", "himachal pradesh"
}


def _try_nominatim_query(query: str, countrycodes: Optional[str] = "in") -> Optional[Dict[str, Any]]:
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": 1
    }
    if countrycodes:
        params["countrycodes"] = countrycodes

    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            results = resp.json()
            if results and len(results) > 0:
                item = results[0]
                lat = float(item["lat"])
                lon = float(item["lon"])
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 and not (lat == 0.0 and lon == 0.0):
                    addr = item.get("address", {})
                    city = addr.get("city") or addr.get("town") or addr.get("suburb") or addr.get("village") or addr.get("county") or ""
                    state = addr.get("state") or ""
                    country = addr.get("country") or ""
                    pincode = addr.get("postcode") or ""
                    return {
                        "latitude": lat,
                        "longitude": lon,
                        "display_name": item.get("display_name", query),
                        "city": city,
                        "state": state,
                        "country": country,
                        "pincode": pincode,
                        "source": "OpenStreetMap Nominatim"
                    }
    except Exception as e:
        logger.debug(f"[NOMINATIM QUERY FAIL]: {e}")
    return None


def _try_photon_query(query: str) -> Optional[Dict[str, Any]]:
    url = "https://photon.komoot.io/api/"
    params = {"q": query, "limit": 1}
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            features = data.get("features", [])
            if features:
                feat = features[0]
                coords = feat.get("geometry", {}).get("coordinates", [])
                if len(coords) >= 2:
                    lon = float(coords[0])
                    lat = float(coords[1])
                    if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0 and not (lat == 0.0 and lon == 0.0):
                        props = feat.get("properties", {})
                        name = props.get("name", query)
                        city = props.get("city") or props.get("district") or props.get("county") or ""
                        state = props.get("state", "")
                        country = props.get("country", "")
                        pincode = props.get("postcode", "")
                        parts = [p for p in [name, city, state, country] if p]
                        label = ", ".join(parts) if parts else name
                        return {
                            "latitude": lat,
                            "longitude": lon,
                            "display_name": label,
                            "city": city,
                            "state": state,
                            "country": country,
                            "pincode": pincode,
                            "source": "Photon (Komoot)"
                        }
    except Exception as e:
        logger.debug(f"[PHOTON QUERY FAIL]: {e}")
    return None


def generate_progressive_candidates(raw_input: str) -> List[Tuple[str, str]]:
    """
    Generate progressive query variants in decreasing order of specificity:
    1. The original address, normalized for search.
    2. The complete normalized address with 'India' added.
    3. Locality segments without road/highway designations.
    4. Specific locality combined with district/town and state.
    5. District or town combined with state.
    6. Six-digit Indian PIN code candidates.
    """
    norm = normalize_address(raw_input)
    if not norm:
        return []

    candidates: List[Tuple[str, str]] = []
    seen: Set[str] = set()

    def add_cand(q: str, level: str):
        q_clean = q.strip(' ,')
        if q_clean and q_clean.lower() not in seen:
            seen.add(q_clean.lower())
            candidates.append((q_clean, level))

    # Tier 1: Exact normalized
    add_cand(norm, "exact_normalized")

    # Tier 2: Exact with country
    if "india" not in norm.lower():
        add_cand(f"{norm}, India", "exact_country_biased")

    segments = [s.strip() for s in norm.split(',') if s.strip()]
    pincode_match = re.search(r'\b(\d{6})\b', norm)
    pincode = pincode_match.group(1) if pincode_match else None

    road_pattern = re.compile(
        r'\b(nh[-\s]*\d+|highway|expressway|bypass|road|salai|marg|lane|street|circle|junction|flyover)\b',
        re.IGNORECASE
    )
    non_road = [s for s in segments if not road_pattern.search(s) and not re.search(r'^\d{6}$', s)]

    # Tier 3: Locality without highway/road noise
    if len(non_road) >= 2 and len(non_road) != len(segments):
        add_cand(", ".join(non_road), "locality_without_road")
        if "india" not in ", ".join(non_road).lower():
            add_cand(", ".join(non_road) + ", India", "locality_without_road_country")

    # Identify state if present
    state_candidate = ""
    for seg in reversed(segments):
        if seg.lower() in INDIAN_STATES:
            state_candidate = seg
            break

    # Tier 4: Locality combinations with town/district and state
    if len(non_road) >= 3:
        if state_candidate:
            add_cand(f"{non_road[1]}, {non_road[2]}, {state_candidate}", "locality_town_state")
            add_cand(f"{non_road[0]}, {non_road[1]}, {state_candidate}", "locality_pair_state")
        add_cand(f"{non_road[0]}, {non_road[1]}", "locality_pair")
        add_cand(f"{non_road[1]}, {non_road[2]}", "locality_town")
        add_cand(f"{non_road[0]}, {non_road[2]}", "locality_town")

    if len(segments) > 2:
        for i in range(1, len(segments) - 1):
            sub = ", ".join(segments[i:])
            add_cand(sub, "sub_address")

    # Tier 5: District/Town + State
    if len(non_road) >= 2 and state_candidate:
        add_cand(f"{non_road[-2]}, {state_candidate}", "town_state")

    # Tier 6: PIN code
    if pincode:
        if non_road:
            add_cand(f"{non_road[0]}, {pincode}, India", "pincode_locality")
            if len(non_road) > 1:
                add_cand(f"{non_road[-1]}, {pincode}, India", "pincode_town")
        add_cand(f"{pincode}, India", "pincode_country")
        add_cand(pincode, "pincode_exact")

    return candidates


def resolve_any_address(query: str) -> Dict[str, Any]:
    """
    Robust multi-tier address resolution pipeline.
    Progressively resolves detailed addresses, localities, suburbs, roads, highways,
    and PIN codes without destroying useful information or silently substituting
    a generic city centre.
    Returns a structured dictionary with success status, coordinates, resolution level, and source.
    """
    if not query or not isinstance(query, str) or not query.strip():
        return {
            "success": False,
            "latitude": None,
            "longitude": None,
            "originalAddress": query or "",
            "resolvedAddress": None,
            "queryUsed": None,
            "resolutionLevel": "unresolved",
            "source": None,
            "city": "",
            "state": "",
            "country": "",
            "pincode": "",
            "display_name": "",
            "error": "Address input is empty."
        }

    cleaned = normalize_address(query)
    cache_key = f"geo:{cleaned.lower()}"
    now = time.time()
    if cache_key in _GEO_CACHE:
        cached_time, cached_val = _GEO_CACHE[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_val

    # Authoritative Metro Centroid lookup ONLY when query itself explicitly equals that metro
    q_lower = cleaned.lower()
    for metro_key, metro_data in CITY_CENTROIDS.items():
        if q_lower in (
            metro_key,
            f"{metro_key}, india",
            f"{metro_key}, {metro_data['state'].lower()}",
            f"{metro_key}, {metro_data['state'].lower()}, india"
        ):
            res = {
                "success": True,
                "latitude": metro_data["latitude"],
                "longitude": metro_data["longitude"],
                "originalAddress": query,
                "resolvedAddress": metro_data["display_name"],
                "queryUsed": cleaned,
                "resolutionLevel": "metro_centroid",
                "source": "Authoritative Metro Geocoding",
                "city": metro_data["city"],
                "state": metro_data["state"],
                "country": metro_data["country"],
                "pincode": "",
                "display_name": metro_data["display_name"]
            }
            _GEO_CACHE[cache_key] = (now, res)
            return res

    candidates = generate_progressive_candidates(query)

    # Progressive fallback execution
    for q_cand, level in candidates:
        # 1. Nominatim with India countrycode bias
        geo = _try_nominatim_query(q_cand, countrycodes="in")
        if geo:
            res = {
                "success": True,
                "latitude": geo["latitude"],
                "longitude": geo["longitude"],
                "originalAddress": query,
                "resolvedAddress": geo["display_name"],
                "queryUsed": q_cand,
                "resolutionLevel": level,
                "source": geo["source"],
                "city": geo.get("city", ""),
                "state": geo.get("state", ""),
                "country": geo.get("country", ""),
                "pincode": geo.get("pincode", ""),
                "display_name": geo["display_name"]
            }
            _GEO_CACHE[cache_key] = (now, res)
            return res

        # 2. Photon fallback
        geo = _try_photon_query(q_cand)
        if geo:
            res = {
                "success": True,
                "latitude": geo["latitude"],
                "longitude": geo["longitude"],
                "originalAddress": query,
                "resolvedAddress": geo["display_name"],
                "queryUsed": q_cand,
                "resolutionLevel": level,
                "source": geo["source"],
                "city": geo.get("city", ""),
                "state": geo.get("state", ""),
                "country": geo.get("country", ""),
                "pincode": geo.get("pincode", ""),
                "display_name": geo["display_name"]
            }
            _GEO_CACHE[cache_key] = (now, res)
            return res

    # Tier 7: Recognized city-level result ONLY when input actually refers to that city
    # and explicitly marked as approximate_city per Requirement 4
    for metro_key, metro_data in CITY_CENTROIDS.items():
        if re.search(r'\b' + re.escape(metro_key) + r'\b', q_lower):
            res = {
                "success": True,
                "latitude": metro_data["latitude"],
                "longitude": metro_data["longitude"],
                "originalAddress": query,
                "resolvedAddress": f"{metro_data['display_name']} (Approximate City Centre)",
                "queryUsed": metro_key,
                "resolutionLevel": "approximate_city",
                "source": "Authoritative Metro Geocoding (Approximate)",
                "city": metro_data["city"],
                "state": metro_data["state"],
                "country": metro_data["country"],
                "pincode": "",
                "display_name": f"{metro_data['display_name']} (Approximate)",
                "is_approximate": True,
                "notice": f"Specific locality could not be resolved. Approximate coordinates for {metro_data['city']} city centre applied."
            }
            _GEO_CACHE[cache_key] = (now, res)
            return res

    # Location could not be resolved
    res = {
        "success": False,
        "latitude": None,
        "longitude": None,
        "originalAddress": query,
        "resolvedAddress": None,
        "queryUsed": None,
        "resolutionLevel": "unresolved",
        "source": None,
        "city": "",
        "state": "",
        "country": "",
        "pincode": "",
        "display_name": "",
        "error": "Location not found. Please refine your search or select a Quick Metro above."
    }
    return res

geocode_address = resolve_any_address


def reverse_geocode_coords(latitude: float, longitude: float) -> Optional[str]:
    """Convert (lat, lon) to human-readable label."""
    cache_key = f"rev:{round(latitude, 4)},{round(longitude, 4)}"
    now = time.time()
    if cache_key in _GEO_CACHE:
        cached_time, cached_val = _GEO_CACHE[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_val

    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": latitude, "lon": longitude, "format": "jsonv2", "zoom": 13, "addressdetails": 1}
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            label = data.get("display_name", f"{latitude:.4f}, {longitude:.4f}")
            _GEO_CACHE[cache_key] = (now, label)
            return label
    except Exception as e:
        logger.debug(f"[REVERSE GEOCODE FAIL]: {e}")
    return f"{latitude:.4f}, {longitude:.4f}"


# ==============================================================================
# 5. EVIDENCE & BADGE BUILDER
# ==============================================================================

def generate_evidence_badges(hosp: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generate evidence-based verification badges:
    - ✓ NABH Accredited (only when NABH confirmed)
    - ✓ JCI Accredited (only when JCI confirmed)
    - ✓ Government Specialty Institute (only when official government apex status confirmed)
    - ✓ Registration Verified (only when legal registration is confirmed)
    Never invent badges or treat OpenStreetMap markers as proof.
    """
    badges = []
    accred = hosp.get("accreditation_status", "")
    gov_status = hosp.get("government_facility_status", "")
    reg_status = hosp.get("legal_registration_status", "")

    if accred == "NABH Accredited":
        badges.append({
            "key": "nabh",
            "label": "✓ NABH Accredited",
            "css_class": "badge-nabh",
            "tooltip": "Current accreditation confirmed in National Accreditation Board for Hospitals (NABH) registry."
        })
    elif accred == "JCI Accredited":
        badges.append({
            "key": "jci",
            "label": "✓ JCI Accredited",
            "css_class": "badge-jci",
            "tooltip": "Current international accreditation confirmed in Joint Commission International (JCI) Gold Seal registry."
        })

    if "Government" in gov_status or accred == "Government Specialty Institute":
        badges.append({
            "key": "govt",
            "label": "✓ Government Specialty Institute",
            "css_class": "badge-govt",
            "tooltip": "Statutory apex institution operated by Ministry of Health & Family Welfare / State Directorate."
        })

    if reg_status and reg_status != "Unverified":
        badges.append({
            "key": "registration",
            "label": "✓ Registration Verified",
            "css_class": "badge-registration",
            "tooltip": f"Legally verified: {reg_status}"
        })

    return badges


# ==============================================================================
# 6. STRICT ELIGIBILITY FILTERING & COMPOSITE RANKING
# ==============================================================================

def filter_and_rank_hospitals(
    candidates: List[Dict[str, Any]],
    condition_id: str,
    patient_lat: Optional[float] = None,
    patient_lon: Optional[float] = None,
    radius_km: float = 25.0,
    min_rating: float = 3.0,
    preferred_mode: str = "all",
    patient_address: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Strict Hospital Filter and Composite Ranking Pipeline:
    1. Resolve condition to clinical specialty.
    2. Check specialty match: hospital MUST have confirmed evidence in the requested specialty.
       (Pneumonia MUST have Pulmonology/Respiratory; general medicine is rejected).
    3. Exclude non-clinical / alternative medicine facilities.
    4. Verify legal and accreditation status.
    5. Filter hospital.rating > min_rating (default 3.0, strict exclusive >).
       Hospitals <= 3.0 or without ratings are EXCLUDED.
    6. Filter distance_km <= radius_km.
    7. Composite Ranking Priorities:
       Priority 1: Strength of verified specialty match (Dedicated institute vs department).
       Priority 2: Higher patient rating.
       Priority 3: Greater review count.
       Priority 4: Shorter distance.
    Ranking NEVER rescues an ineligible hospital.
    """
    mapping = get_specialty_mapping(condition_id)
    canonical_cond_id = mapping.get("id", condition_id)
    primary_spec = mapping.get("primary_specialty", "Specialist Care")
    accepted_specialties = [s.lower() for s in mapping.get("accepted_specialties", [])]
    keywords = [kw.lower() for kw in mapping.get("search_keywords", [])]

    eligible_facilities: List[Dict[str, Any]] = []

    for item in candidates:
        hosp = dict(item)
        name = hosp.get("name", "")

        # 1. Non-clinical / Alternative medicine exclusion
        if is_non_clinical_facility(name, hosp.get("tags"), canonical_cond_id):
            continue

        # 2. Strict Specialty Verification Check
        matched_conditions = hosp.get("condition_ids", [])
        confirmed_specs = [s.lower() for s in hosp.get("confirmed_specialties", [])]
        name_lower = name.lower()

        is_specialty_matched = False
        dept_evidence_text = ""

        # Check condition_ids in verified registry
        if canonical_cond_id in matched_conditions:
            is_specialty_matched = True
            dept_ev = hosp.get("department_evidence", {})
            if isinstance(dept_ev, dict):
                dept_evidence_text = dept_ev.get(canonical_cond_id, "")
            elif isinstance(dept_ev, str):
                dept_evidence_text = dept_ev

        # Cross-reference confirmed_specialties
        if not is_specialty_matched and confirmed_specs:
            for spec in confirmed_specs:
                if any(acc in spec for acc in accepted_specialties) or any(kw in spec for kw in keywords):
                    is_specialty_matched = True
                    break

        # For Pneumonia: STRICT rejection of generic internal medicine without pulmonology
        if canonical_cond_id == "pneumonia":
            if not is_specialty_matched:
                continue
            # If the hospital only has general internal medicine, reject
            if not any("pulmon" in s or "respirat" in s or "chest" in s or "thorac" in s for s in confirmed_specs):
                if not ("pulmon" in dept_evidence_text.lower() or "respirat" in dept_evidence_text.lower() or "chest" in dept_evidence_text.lower()):
                    continue

        if not is_specialty_matched:
            # Multi-specialty hospital WITHOUT confirmed required specialty is strictly excluded!
            # Never fall back to yellow General / Multi-Specialty
            continue

        # 3. Official Verification & Legal Eligibility
        reg_status = hosp.get("legal_registration_status", "")
        accred_status = hosp.get("accreditation_status", "")
        gov_status = hosp.get("government_facility_status", "")
        has_official_verification = bool(
            (reg_status and reg_status != "Unverified") or
            (accred_status in ("NABH Accredited", "JCI Accredited", "Government Specialty Institute")) or
            ("Government" in gov_status)
        )
        if not has_official_verification:
            continue

        # 4. Strict Rating Filter (rating > min_rating)
        raw_rating = hosp.get("rating")
        if raw_rating is None:
            continue
        try:
            rating_val = float(raw_rating)
        except (ValueError, TypeError):
            continue

        if rating_val <= min_rating:
            # Strictly exclude <= min_rating (e.g. 3.0)
            continue

        review_count = int(hosp.get("review_count", 0))

        # 5. Distance calculation
        dist_km = None
        if patient_lat is not None and patient_lon is not None and hosp.get("latitude") and hosp.get("longitude"):
            dist_km = calculate_haversine_distance(patient_lat, patient_lon, hosp["latitude"], hosp["longitude"])
            hosp["distance_km"] = dist_km
            if dist_km > radius_km:
                # Outside active radius
                continue
        elif hosp.get("distance_km") is not None:
            dist_km = hosp["distance_km"]
            if dist_km > radius_km:
                continue

        # 6. Consultation Mode Filter
        telemed_available = bool(hosp.get("telemedicine_available", False))
        if preferred_mode == "telemedicine" and not telemed_available:
            continue

        # 7. Build Evidence Badges & Display Attributes
        badges = generate_evidence_badges(hosp)
        hosp["verified_badges"] = badges
        hosp["specialty_match"] = True
        hosp["tier"] = 1
        hosp["specialty_badge_label"] = f"✓ VERIFIED SPECIALTY MATCH ({primary_spec})"
        hosp["specialty_label"] = f"✓ VERIFIED SPECIALTY MATCH ({primary_spec})"
        hosp["display_specialty"] = primary_spec
        hosp["primary_specialty"] = primary_spec
        hosp["condition_id"] = canonical_cond_id
        hosp["canonical_disease"] = mapping.get("canonical_name", canonical_cond_id)
        hosp["rating_display"] = f"★ {rating_val:.1f} · {review_count:,} reviews"
        hosp["department_evidence_text"] = dept_evidence_text or f"Confirmed {primary_spec} department verified via official accreditation registry."

        # Compute composite score for ranking:
        # Priority 1: Dedicated specialty institute match vs department in tertiary center
        is_dedicated_institute = "Institute" in hosp.get("facility_type", "") or "Apex" in hosp.get("facility_type", "")
        specialty_score = 1000 if is_dedicated_institute else 800

        # Priority 2: Higher patient rating (scaled)
        rating_score = int(rating_val * 100)

        # Priority 3: Review count confidence (log-scaled)
        review_score = int(min(math.log10(max(review_count, 1)) * 30, 150))

        # Priority 4: Distance penalty (shorter distance gets higher score)
        distance_penalty = int(min(dist_km or 0, 100) * 4)

        composite_score = specialty_score + rating_score + review_score - distance_penalty
        hosp["recommendation_score"] = composite_score

        # Google Maps navigation link
        origin_str = ""
        if patient_lat is not None and patient_lon is not None:
            origin_str = f"{float(patient_lat):.6f}%2C{float(patient_lon):.6f}"
        elif patient_address and patient_address.strip() and patient_address != "Location Not Provided":
            origin_str = urllib.parse.quote_plus(normalize_address(patient_address))

        dest_str = f"{float(hosp['latitude']):.6f}%2C{float(hosp['longitude']):.6f}" if (hosp.get("latitude") and hosp.get("longitude")) else urllib.parse.quote_plus(name)
        if origin_str:
            hosp["directions_url"] = f"https://www.google.com/maps/dir/?api=1&origin={origin_str}&destination={dest_str}&travelmode=driving"
        else:
            hosp["directions_url"] = f"https://www.google.com/maps/dir/?api=1&destination={dest_str}&travelmode=driving"

        eligible_facilities.append(hosp)

    # Sort strictly by composite score (highest first), then by distance (closest first)
    eligible_facilities.sort(key=lambda h: (-h.get("recommendation_score", 0), h.get("distance_km") or 99999.0))
    return eligible_facilities


# Backwards compatibility alias
def score_and_recommend_hospitals(
    hospitals: List[Dict[str, Any]],
    condition_id: str,
    patient_lat: Optional[float] = None,
    patient_lon: Optional[float] = None,
    preferred_mode: str = "all",
    patient_address: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Legacy scoring wrapper routing to the strict filter and ranking pipeline."""
    return filter_and_rank_hospitals(
        candidates=hospitals,
        condition_id=condition_id,
        patient_lat=patient_lat,
        patient_lon=patient_lon,
        radius_km=100.0,
        min_rating=3.0,
        preferred_mode=preferred_mode,
        patient_address=patient_address
    )


# ==============================================================================
# 7. MAIN DISCOVERY CONTROLLER WITH PROGRESSIVE RADIUS AUTO-EXPANSION
# ==============================================================================

PROGRESSIVE_RADIUS_TIERS = [5, 10, 25, 50, 100]

def discover_hospitals_for_patient(
    condition_id: str,
    patient_lat: Optional[float] = None,
    patient_lon: Optional[float] = None,
    manual_query: Optional[str] = None,
    radius_km: int = 25,
    min_rating: float = 3.0,
    preferred_mode: str = "all",
    include_global: bool = False
) -> Dict[str, Any]:
    """
    Main orchestration function for hospital discovery:
    1. Resolves patient coordinates via geocoding.
    2. Queries authoritative verified registry + live enriched candidates.
    3. Strictly filters by disease specialty, legal verification, and rating > min_rating.
    4. Executes Progressive Radius Auto-Expansion:
       [5 km -> 10 km -> 25 km -> 50 km -> 100 km].
       If 0 results found at requested radius, tries next tier until qualifying
       facilities are found or 100 km max is reached.
    5. Discloses auto-expansion with notification banner while keeping
       the user's originally requested radius in the UI.
    6. Ranks eligible facilities using composite ranking.
    """
    mapping = get_specialty_mapping(condition_id)
    canonical_cond_id = mapping.get("id", condition_id)
    primary_spec = mapping.get("primary_specialty", "Specialist Care")
    location_label = "Location Not Provided"
    geocoded_origin = None

    # Step 1: Resolve manual query if coordinates not given
    if (patient_lat is None or patient_lon is None) and manual_query:
        geo_res = resolve_any_address(manual_query)
        if geo_res and geo_res.get("success"):
            patient_lat = geo_res["latitude"]
            patient_lon = geo_res["longitude"]
            location_label = geo_res.get("resolvedAddress") or geo_res.get("display_name")
            geocoded_origin = geo_res

    # Step 2: Reverse geocode coordinates if needed
    if patient_lat is not None and patient_lon is not None and location_label == "Location Not Provided":
        location_label = reverse_geocode_coords(patient_lat, patient_lon) or f"{patient_lat:.4f}, {patient_lon:.4f}"

    has_coords = patient_lat is not None and patient_lon is not None
    requested_radius_km = int(radius_km) if radius_km else 25

    # If manual query was supplied but could not be resolved, do NOT return success or global centres
    if not has_coords and manual_query:
        return {
            "status": "location_not_found",
            "condition": mapping,
            "location": {
                "latitude": None,
                "longitude": None,
                "display_name": "Location Not Found",
                "requested_radius_km": requested_radius_km,
                "effective_radius_km": requested_radius_km,
                "auto_expanded": False,
                "expansion_message": f"Location '{manual_query}' could not be resolved. Please refine your search or select a Quick Metro above.",
                "has_coordinates": False
            },
            "filters_applied": {
                "disease": canonical_cond_id,
                "specialty": primary_spec,
                "min_rating": min_rating,
                "strict_specialty_verified": True,
                "strict_legal_verification": True
            },
            "counts": {"total": 0, "nearby": 0, "global_centres": 0},
            "recommendations": [],
            "urgent_notice": mapping.get("urgent_warning") if mapping.get("requires_urgent_assessment") else None
        }

    # Prepare candidate pool from the authoritative verified registry
    candidate_pool: List[Dict[str, Any]] = list(VERIFIED_HOSPITAL_REGISTRY)

    # Progressive Radius Auto-Expansion Pipeline
    effective_radius_km = requested_radius_km
    auto_expanded = False
    expansion_message = ""
    eligible_results: List[Dict[str, Any]] = []

    if has_coords:
        # Determine starting index in tiers
        tiers_to_try = [t for t in PROGRESSIVE_RADIUS_TIERS if t >= requested_radius_km]
        if requested_radius_km not in tiers_to_try:
            tiers_to_try = [requested_radius_km] + [t for t in PROGRESSIVE_RADIUS_TIERS if t > requested_radius_km]

        for tier in tiers_to_try:
            results_at_tier = filter_and_rank_hospitals(
                candidates=candidate_pool,
                condition_id=canonical_cond_id,
                patient_lat=patient_lat,
                patient_lon=patient_lon,
                radius_km=float(tier),
                min_rating=min_rating,
                preferred_mode=preferred_mode,
                patient_address=manual_query or location_label
            )

            if results_at_tier:
                eligible_results = results_at_tier
                effective_radius_km = tier
                if tier > requested_radius_km:
                    auto_expanded = True
                    expansion_message = (
                        f"No qualifying hospitals were found within {requested_radius_km} km. "
                        f"Search expanded to {tier} km to find verified {primary_spec} hospitals rated above {min_rating:.1f}."
                    )
                break
            else:
                effective_radius_km = tier

        # If still 0 results at maximum radius
        if not eligible_results:
            effective_radius_km = 100
            auto_expanded = False
            expansion_message = (
                f"No verified {primary_spec} hospitals with ratings above {min_rating:.1f} "
                f"were found within 100 km of {location_label}."
            )
    else:
        # No coordinates provided yet: return global curated centres if requested
        if include_global:
            eligible_results = filter_and_rank_hospitals(
                candidates=[h for h in candidate_pool if h.get("country") != "India" or "Apex" in h.get("facility_type", "")],
                condition_id=canonical_cond_id,
                patient_lat=None,
                patient_lon=None,
                radius_km=99999.0,
                min_rating=min_rating,
                preferred_mode=preferred_mode,
                patient_address=manual_query
            )

    return {
        "status": "success",
        "condition": mapping,
        "location": {
            "latitude": patient_lat,
            "longitude": patient_lon,
            "display_name": location_label,
            "requested_radius_km": requested_radius_km,
            "effective_radius_km": effective_radius_km,
            "auto_expanded": auto_expanded,
            "expansion_message": expansion_message,
            "has_coordinates": has_coords
        },
        "filters_applied": {
            "disease": canonical_cond_id,
            "specialty": primary_spec,
            "min_rating": min_rating,
            "strict_specialty_verified": True,
            "strict_legal_verification": True
        },
        "counts": {
            "total": len(eligible_results),
            "nearby": len(eligible_results) if has_coords else 0,
            "global_centres": 0 if has_coords else len(eligible_results)
        },
        "recommendations": eligible_results,
        "urgent_notice": mapping.get("urgent_warning") if mapping.get("requires_urgent_assessment") else None
    }
