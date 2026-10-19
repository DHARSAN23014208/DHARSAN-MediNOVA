"""
MediNova Smart Medical Lab Report Scanner & Clinical Biomarker Extractor
========================================================================
Modular, production-grade clinical document extraction pipeline supporting:
1. Multimodal Vision APIs (Gemini / OpenAI) with strict schema validation.
2. Local text-based PDF parsing via pypdf (zero-external-dependency local extraction).
3. Local OCR for photographs and scanned documents via pytesseract and Pillow.
4. Robust regex parsing with clinical synonym dictionaries, reference interval
   disambiguation, unit normalization, and baseline imputation.
"""

import os
import re
import json
import logging
import tempfile
from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageEnhance, ImageOps

logger = logging.getLogger("medinova.report_scanner")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Supported module allowlist
SUPPORTED_MODULES = {"diabetes", "heart", "kidney", "liver", "breast_cancer"}

# Maximum upload file size in bytes (10 MB default)
MAX_FILE_SIZE = int(os.environ.get("MAX_SCANNER_FILE_SIZE", 10 * 1024 * 1024))
# Maximum PDF pages to process to preserve RAM
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", 3))

# Strict file extension allowlist
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "pdf"}

# ============================================================================
# CLINICAL FEATURE DEFINITIONS & SYNONYM DICTIONARIES
# ============================================================================

MODULE_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "diabetes": {
        "title": "Diabetes Mellitus Metabolic Panel",
        "fields": {
            "HbA1c_level": {
                "type": "float",
                "label": "Glycated Hemoglobin (HbA1c)",
                "unit": "%",
                "aliases": [
                    r"\bhba1c\b", r"\bhb\s*a1c\b", r"\ba1c\b",
                    r"glycated\s+hemoglobin", r"glycohemoglobin",
                    r"glycosylated\s+hemoglobin", r"hemoglobin\s+a1c"
                ],
                "range": (3.0, 18.0),
                "normal_range": (4.0, 5.6),
                "required_for_prediction": False,
                "is_core_analyte": True
            },
            "blood_glucose_level": {
                "type": "float",
                "label": "Blood Glucose Level",
                "unit": "mg/dL",
                "aliases": [
                    r"fasting\s+blood\s+sugar", r"\bfbs\b", r"fasting\s+glucose",
                    r"fasting\s+plasma\s+glucose", r"\bfpg\b", r"blood\s+glucose",
                    r"random\s+blood\s+sugar", r"\brbs\b", r"plasma\s+glucose",
                    r"serum\s+glucose", r"\bglucose\b", r"random\s+blood\s+glucose",
                    r"post\s*prandial\s+blood\s+sugar", r"\bppbs\b", r"blood\s+sugar"
                ],
                "range": (40.0, 600.0),
                "normal_range": (70.0, 99.0),
                "required_for_prediction": False,
                "is_core_analyte": True
            },
            "bmi": {
                "type": "float",
                "label": "Body Mass Index (BMI)",
                "unit": "kg/m²",
                "aliases": [
                    r"body\s+mass\s+index", r"\bbmi\b", r"body-mass-index", r"quetelet\s+index"
                ],
                "range": (12.0, 65.0),
                "normal_range": (18.5, 24.9),
                "required_for_prediction": False,
                "baseline_default": 25.0,
                "default_reason": "Clinical median adult baseline BMI (25.0 kg/m²)"
            },
            "age": {
                "type": "float",
                "label": "Patient Age",
                "unit": "years",
                "aliases": [r"\bage\b", r"patient\s+age", r"age\s+in\s+years", r"\byrs\b"],
                "range": (1.0, 120.0),
                "normal_range": (18.0, 85.0),
                "required_for_prediction": False,
                "baseline_default": 45.0,
                "default_reason": "Standard adult screening median age (45 years)"
            },
            "gender": {
                "type": "category",
                "label": "Gender",
                "unit": "",
                "aliases": [r"\bgender\b", r"\bsex\b", r"biological\s+sex"],
                "options": {"male": "Male", "m": "Male", "female": "Female", "f": "Female", "other": "Other"},
                "required_for_prediction": False,
                "baseline_default": "Female",
                "default_reason": "Baseline demographic category (Female)"
            },
            "hypertension": {
                "type": "int",
                "label": "Hypertension History",
                "unit": "",
                "aliases": [r"hypertension", r"\bhtn\b", r"high\s+blood\s+pressure", r"bp\s+history"],
                "options": {"yes": 1, "true": 1, "positive": 1, "present": 1, "1": 1, "no": 0, "false": 0, "negative": 0, "absent": 0, "0": 0},
                "required_for_prediction": False,
                "baseline_default": 0,
                "default_reason": "No documented hypertension history reported"
            },
            "heart_disease": {
                "type": "int",
                "label": "Heart Disease History",
                "unit": "",
                "aliases": [r"heart\s+disease", r"coronary\s+artery\s+disease", r"\bcad\b", r"cardiac\s+history", r"myocardial\s+infarction"],
                "options": {"yes": 1, "true": 1, "positive": 1, "present": 1, "1": 1, "no": 0, "false": 0, "negative": 0, "absent": 0, "0": 0},
                "required_for_prediction": False,
                "baseline_default": 0,
                "default_reason": "No documented coronary heart disease reported"
            },
            "smoking_history": {
                "type": "category",
                "label": "Smoking History",
                "unit": "",
                "aliases": [r"smoking\s+history", r"smoking\s+status", r"\bsmoking\b", r"tobacco", r"smoker"],
                "options": {
                    "never": "never", "non-smoker": "never", "nonsmoker": "never", "no": "never",
                    "current": "current", "active": "current", "yes": "current",
                    "former": "former", "ex-smoker": "former", "past": "former",
                    "ever": "ever", "not current": "not current", "no info": "never"
                },
                "required_for_prediction": False,
                "baseline_default": "never",
                "default_reason": "Non-smoking baseline assumed pending clinical intake"
            }
        },
        "minimum_rule": "At least one core glycemic analyte (HbA1c or Blood Glucose) must be extracted from the lab report."
    },

    "heart": {
        "title": "Cardiovascular & Coronary Panel",
        "fields": {
            "bp": {
                "type": "float",
                "label": "Resting Blood Pressure",
                "unit": "mm Hg",
                "aliases": [
                    r"resting\s+bp", r"systolic\s+bp", r"blood\s+pressure",
                    r"\bbp\b", r"resting\s+blood\s+pressure", r"systolic\s+pressure", r"r\.b\.p\."
                ],
                "range": (70.0, 220.0),
                "normal_range": (90.0, 120.0),
                "required_for_prediction": False,
                "is_core_analyte": True
            },
            "chol": {
                "type": "float",
                "label": "Serum Cholesterol",
                "unit": "mg/dL",
                "aliases": [
                    r"total\s+cholesterol", r"serum\s+cholesterol", r"\bcholesterol\b",
                    r"\bchol\b", r"total\s+chol", r"s\.\s*cholesterol"
                ],
                "range": (90.0, 600.0),
                "normal_range": (125.0, 200.0),
                "required_for_prediction": False,
                "is_core_analyte": True
            },
            "max_hr": {
                "type": "float",
                "label": "Maximum Heart Rate Achieved",
                "unit": "bpm",
                "aliases": [
                    r"maximum\s+heart\s+rate", r"max\s+heart\s+rate", r"max\s+hr",
                    r"\bthalach\b", r"peak\s+heart\s+rate", r"heart\s+rate", r"\bpulse\b", r"\bhr\b"
                ],
                "range": (50.0, 230.0),
                "normal_range": (60.0, 100.0),
                "required_for_prediction": False,
                "baseline_default": 150.0,
                "default_reason": "Model population median exercise peak heart rate (150 bpm)"
            },
            "oldpeak": {
                "type": "float",
                "label": "ST Depression (Oldpeak)",
                "unit": "",
                "aliases": [r"st\s+depression", r"\boldpeak\b", r"st\s+segment\s+depression", r"st-depression"],
                "range": (0.0, 7.0),
                "normal_range": (0.0, 1.0),
                "required_for_prediction": False,
                "baseline_default": 0.8,
                "default_reason": "Baseline normal physiological ST segment value (0.8)"
            },
            "fbs": {
                "type": "int",
                "label": "Fasting Blood Sugar > 120 mg/dL",
                "unit": "",
                "aliases": [r"fasting\s+blood\s+sugar", r"\bfbs\b", r"fasting\s+glucose", r"fbs\s+over\s+120"],
                "options": {"yes": 1, "true": 1, "1": 1, "positive": 1, "no": 0, "false": 0, "0": 0, "negative": 0},
                "range": (0, 1),
                "required_for_prediction": False,
                "baseline_default": 0,
                "default_reason": "Fasting blood sugar assumed <= 120 mg/dL unless elevated in report"
            },
            "age": {
                "type": "float",
                "label": "Age",
                "unit": "years",
                "aliases": [r"\bage\b", r"patient\s+age", r"\byrs\b"],
                "range": (18.0, 100.0),
                "required_for_prediction": False,
                "baseline_default": 55.0,
                "default_reason": "Standard adult cardiac baseline age (55 years)"
            },
            "sex": {
                "type": "int",
                "label": "Biological Sex",
                "unit": "",
                "aliases": [r"\bsex\b", r"\bgender\b"],
                "options": {"male": 1, "m": 1, "1": 1, "female": 0, "f": 0, "0": 0},
                "required_for_prediction": False,
                "baseline_default": 1,
                "default_reason": "Default clinical profile baseline (Male)"
            },
            "cp": {
                "type": "int",
                "label": "Chest Pain Type",
                "unit": "grade",
                "aliases": [r"chest\s+pain\s+type", r"chest\s+pain", r"\bcp\b", r"angina\s+type"],
                "options": {
                    "typical angina": 0, "typical": 0, "0": 0,
                    "atypical angina": 1, "atypical": 1, "1": 1,
                    "non-anginal": 2, "non anginal": 2, "2": 2,
                    "asymptomatic": 3, "none": 3, "3": 3
                },
                "range": (0, 3),
                "required_for_prediction": False,
                "baseline_default": 0,
                "default_reason": "Default asymptomatic/typical category (0) - clinical confirmation advised"
            },
            "ekg": {
                "type": "int",
                "label": "Resting ECG Results",
                "unit": "",
                "aliases": [r"resting\s+ecg", r"resting\s+ekg", r"\bekg\b", r"\becg\b"],
                "options": {
                    "normal": 0, "0": 0,
                    "st-t wave abnormality": 1, "st abnormality": 1, "1": 1,
                    "left ventricular hypertrophy": 2, "lvh": 2, "2": 2
                },
                "range": (0, 2),
                "required_for_prediction": False,
                "baseline_default": 0,
                "default_reason": "Normal sinus rhythm ECG baseline (0)"
            },
            "exang": {
                "type": "int",
                "label": "Exercise Induced Angina",
                "unit": "",
                "aliases": [r"exercise\s+induced\s+angina", r"exercise\s+angina", r"\bexang\b"],
                "options": {"yes": 1, "1": 1, "no": 0, "0": 0},
                "range": (0, 1),
                "required_for_prediction": False,
                "baseline_default": 0,
                "default_reason": "Absence of exercise-induced angina (0)"
            },
            "slope": {
                "type": "int",
                "label": "Slope of Peak Exercise ST",
                "unit": "",
                "aliases": [r"slope\s+of\s+st", r"st\s+slope", r"\bslope\b"],
                "options": {"upsloping": 0, "0": 0, "flat": 1, "1": 1, "downsloping": 2, "2": 2},
                "range": (0, 2),
                "required_for_prediction": False,
                "baseline_default": 0,
                "default_reason": "Normal upsloping ST segment baseline (0)"
            },
            "ca": {
                "type": "int",
                "label": "Major Vessels by Fluoroscopy",
                "unit": "vessels",
                "aliases": [r"number\s+of\s+vessels", r"major\s+vessels", r"\bca\b", r"vessels\s+fluro"],
                "range": (0, 3),
                "required_for_prediction": False,
                "baseline_default": 0,
                "default_reason": "Zero fluoroscopy calcified vessels (0)"
            },
            "thal": {
                "type": "int",
                "label": "Thallium Stress Test",
                "unit": "",
                "aliases": [r"thallium\s+stress", r"thallium", r"\bthal\b"],
                "options": {
                    "normal": 3, "3": 3,
                    "fixed defect": 6, "fixed": 6, "6": 6,
                    "reversible defect": 7, "reversible": 7, "7": 7
                },
                "range": (3, 7),
                "required_for_prediction": False,
                "baseline_default": 3,
                "default_reason": "Normal thallium perfusion baseline (3)"
            }
        },
        "minimum_rule": "At least Blood Pressure (BP) or Serum Cholesterol must be extracted from the laboratory report."
    },

    "kidney": {
        "title": "Renal Function & Urinalysis Panel",
        "fields": {
            "sc": {
                "type": "float",
                "label": "Serum Creatinine",
                "unit": "mg/dL",
                "aliases": [
                    r"serum\s+creatinine", r"s\.\s*creatinine", r"\bcreatinine\b",
                    r"\bsc\b", r"\bcreat\b"
                ],
                "range": (0.2, 20.0),
                "normal_range": (0.6, 1.2),
                "required_for_prediction": False,
                "is_core_analyte": True
            },
            "bu": {
                "type": "float",
                "label": "Blood Urea",
                "unit": "mg/dL",
                "aliases": [
                    r"blood\s+urea", r"blood\s+urea\s+nitrogen", r"\bbun\b", r"\burea\b", r"\bbu\b"
                ],
                "range": (5.0, 300.0),
                "normal_range": (15.0, 45.0),
                "required_for_prediction": False,
                "is_core_analyte": True
            },
            "bgr": {
                "type": "float",
                "label": "Blood Glucose Random",
                "unit": "mg/dL",
                "aliases": [
                    r"blood\s+glucose\s+random", r"random\s+blood\s+glucose",
                    r"random\s+blood\s+sugar", r"\bbgr\b", r"blood\s+glucose", r"\bglucose\b"
                ],
                "range": (40.0, 500.0),
                "normal_range": (70.0, 140.0),
                "required_for_prediction": False,
                "baseline_default": 120.0,
                "default_reason": "Baseline random glucose (120 mg/dL)"
            },
            "sod": {
                "type": "float",
                "label": "Serum Sodium",
                "unit": "mEq/L",
                "aliases": [r"serum\s+sodium", r"s\.\s*sodium", r"\bsodium\b", r"\bsod\b", r"\bna\b"],
                "range": (100.0, 180.0),
                "normal_range": (135.0, 145.0),
                "required_for_prediction": False,
                "baseline_default": 138.0,
                "default_reason": "Baseline serum sodium electrolyte (138 mEq/L)"
            },
            "pot": {
                "type": "float",
                "label": "Serum Potassium",
                "unit": "mEq/L",
                "aliases": [r"serum\s+potassium", r"s\.\s*potassium", r"\bpotassium\b", r"\bpot\b", r"\bk\b"],
                "range": (2.0, 10.0),
                "normal_range": (3.5, 5.0),
                "required_for_prediction": False,
                "baseline_default": 4.4,
                "default_reason": "Baseline serum potassium electrolyte (4.4 mEq/L)"
            },
            "hemo": {
                "type": "float",
                "label": "Hemoglobin",
                "unit": "g/dL",
                "aliases": [r"\bhemoglobin\b", r"\bhaemoglobin\b", r"\bhemo\b", r"\bhb\b", r"\bhgb\b"],
                "range": (3.0, 22.0),
                "normal_range": (12.0, 16.5),
                "required_for_prediction": False,
                "baseline_default": 13.5,
                "default_reason": "Normal adult hemoglobin baseline (13.5 g/dL)"
            },
            "pcv": {
                "type": "float",
                "label": "Packed Cell Volume (Hematocrit)",
                "unit": "%",
                "aliases": [r"packed\s+cell\s+volume", r"\bpcv\b", r"\bhematocrit\b", r"\bhct\b"],
                "range": (10.0, 65.0),
                "normal_range": (36.0, 50.0),
                "required_for_prediction": False,
                "baseline_default": 40.0,
                "default_reason": "Baseline packed cell volume (40%)"
            },
            "wc": {
                "type": "float",
                "label": "White Blood Cell Count",
                "unit": "cells/cumm",
                "aliases": [
                    r"white\s+blood\s+cell\s+count", r"white\s+blood\s+cells",
                    r"\bwbc\s*count\b", r"\bwbc\b", r"\bwc\b", r"total\s+leukocyte\s+count", r"\btlc\b"
                ],
                "range": (1500.0, 30000.0),
                "normal_range": (4000.0, 11000.0),
                "required_for_prediction": False,
                "baseline_default": 7800.0,
                "default_reason": "Median physiological leukocyte count (7,800 /cumm)"
            },
            "rc": {
                "type": "float",
                "label": "Red Blood Cell Count",
                "unit": "millions/cmm",
                "aliases": [
                    r"red\s+blood\s+cell\s+count", r"red\s+blood\s+cells",
                    r"\brbc\s*count\b", r"\brc\b", r"total\s+red\s+cell\s+count"
                ],
                "range": (1.5, 8.5),
                "normal_range": (4.2, 5.8),
                "required_for_prediction": False,
                "baseline_default": 4.8,
                "default_reason": "Median adult erythrocyte count (4.8 M/cmm)"
            },
            "bp": {
                "type": "float",
                "label": "Blood Pressure",
                "unit": "mm Hg",
                "aliases": [r"blood\s+pressure", r"resting\s+bp", r"\bbp\b"],
                "range": (60.0, 220.0),
                "normal_range": (80.0, 120.0),
                "required_for_prediction": False,
                "baseline_default": 80.0,
                "default_reason": "Standard resting diastolic/systolic baseline (80 mm Hg)"
            },
            "age": {
                "type": "float",
                "label": "Age",
                "unit": "years",
                "aliases": [r"\bage\b", r"patient\s+age", r"\byrs\b"],
                "range": (1.0, 110.0),
                "required_for_prediction": False,
                "baseline_default": 50.0,
                "default_reason": "Screening median age (50 years)"
            },
            "sg": {
                "type": "float",
                "label": "Specific Gravity",
                "unit": "",
                "aliases": [r"specific\s+gravity", r"sp\.\s*gravity", r"\bsg\b", r"urine\s+specific\s+gravity"],
                "range": (1.000, 1.040),
                "normal_range": (1.015, 1.025),
                "required_for_prediction": False,
                "baseline_default": 1.020,
                "default_reason": "Optimal physiological urine specific gravity (1.020)"
            },
            "al": {
                "type": "float",
                "label": "Urine Albumin",
                "unit": "grade",
                "aliases": [r"urine\s+albumin", r"\balbumin\s+urine\b", r"\bal\b", r"protein\s+urine"],
                "range": (0.0, 5.0),
                "normal_range": (0.0, 0.0),
                "required_for_prediction": False,
                "baseline_default": 0.0,
                "default_reason": "Negative urinalysis albumin grade (0)"
            },
            "su": {
                "type": "float",
                "label": "Urine Sugar",
                "unit": "grade",
                "aliases": [r"urine\s+sugar", r"\bsugar\s+urine\b", r"\bsu\b", r"urine\s+glucose"],
                "range": (0.0, 5.0),
                "normal_range": (0.0, 0.0),
                "required_for_prediction": False,
                "baseline_default": 0.0,
                "default_reason": "Negative urinalysis glucose grade (0)"
            },
            "rbc": {
                "type": "int",
                "label": "Urinary RBC",
                "unit": "",
                "aliases": [r"urine\s+rbc", r"microscopic\s+rbc"],
                "options": {"normal": 1, "1": 1, "abnormal": 0, "0": 0},
                "required_for_prediction": False,
                "baseline_default": 1,
                "default_reason": "Normal urine microscopy red blood cells (1)"
            },
            "pc": {
                "type": "int",
                "label": "Pus Cells",
                "unit": "",
                "aliases": [r"pus\s+cells", r"\bpc\b", r"pyuria"],
                "options": {"normal": 1, "1": 1, "abnormal": 0, "0": 0},
                "required_for_prediction": False,
                "baseline_default": 1,
                "default_reason": "Normal pus cells range (1)"
            },
            "pcc": {
                "type": "int",
                "label": "Pus Cell Clumps",
                "unit": "",
                "aliases": [r"pus\s+cell\s+clumps", r"\bpcc\b"],
                "options": {"present": 1, "1": 1, "not present": 0, "0": 0, "absent": 0},
                "required_for_prediction": False,
                "baseline_default": 0,
                "default_reason": "Absence of pus cell clumps (0)"
            },
            "ba": {
                "type": "int",
                "label": "Urinary Bacteria",
                "unit": "",
                "aliases": [r"bacteria", r"\bba\b", r"bacteriuria"],
                "options": {"present": 1, "1": 1, "not present": 0, "0": 0, "absent": 0},
                "required_for_prediction": False,
                "baseline_default": 0,
                "default_reason": "Absence of urinary bacteria (0)"
            },
            "htn": {
                "type": "int",
                "label": "Hypertension",
                "unit": "",
                "aliases": [r"hypertension", r"\bhtn\b"],
                "options": {"yes": 1, "1": 1, "no": 0, "0": 0},
                "required_for_prediction": False,
                "baseline_default": 0,
                "default_reason": "No documented hypertension (0)"
            },
            "dm": {
                "type": "int",
                "label": "Diabetes Mellitus",
                "unit": "",
                "aliases": [r"diabetes\s+mellitus", r"diabetes", r"\bdm\b"],
                "options": {"yes": 1, "1": 1, "no": 0, "0": 0},
                "required_for_prediction": False,
                "baseline_default": 0,
                "default_reason": "No documented diabetes mellitus (0)"
            },
            "cad": {
                "type": "int",
                "label": "Coronary Artery Disease",
                "unit": "",
                "aliases": [r"coronary\s+artery\s+disease", r"\bcad\b"],
                "options": {"yes": 1, "1": 1, "no": 0, "0": 0},
                "required_for_prediction": False,
                "baseline_default": 0,
                "default_reason": "No coronary artery disease history (0)"
            },
            "appet": {
                "type": "int",
                "label": "Appetite",
                "unit": "",
                "aliases": [r"appetite", r"\bappet\b"],
                "options": {"good": 0, "0": 0, "poor": 1, "1": 1},
                "required_for_prediction": False,
                "baseline_default": 0,
                "default_reason": "Good appetite baseline (0)"
            },
            "pe": {
                "type": "int",
                "label": "Pedal Edema",
                "unit": "",
                "aliases": [r"pedal\s+edema", r"edema", r"\bpe\b"],
                "options": {"yes": 1, "1": 1, "no": 0, "0": 0},
                "required_for_prediction": False,
                "baseline_default": 0,
                "default_reason": "Absence of pedal edema (0)"
            },
            "ane": {
                "type": "int",
                "label": "Anemia",
                "unit": "",
                "aliases": [r"anemia", r"\bane\b"],
                "options": {"yes": 1, "1": 1, "no": 0, "0": 0},
                "required_for_prediction": False,
                "baseline_default": 0,
                "default_reason": "No clinical anemia history (0)"
            }
        },
        "minimum_rule": "At least Serum Creatinine or Blood Urea must be extracted from the renal panel."
    },

    "liver": {
        "title": "Hepatic Function Panel (LFT)",
        "fields": {
            "Total_Bilirubin": {
                "type": "float",
                "label": "Total Bilirubin",
                "unit": "mg/dL",
                "aliases": [
                    r"total\s+bilirubin", r"t\.\s*bilirubin", r"bilirubin\s+total",
                    r"s\.\s*bilirubin", r"serum\s+bilirubin\s+total"
                ],
                "range": (0.1, 40.0),
                "normal_range": (0.2, 1.2),
                "required_for_prediction": False,
                "is_core_analyte": True
            },
            "Direct_Bilirubin": {
                "type": "float",
                "label": "Direct Bilirubin",
                "unit": "mg/dL",
                "aliases": [
                    r"direct\s+bilirubin", r"d\.\s*bilirubin", r"conjugated\s+bilirubin",
                    r"bilirubin\s+direct"
                ],
                "range": (0.0, 20.0),
                "normal_range": (0.0, 0.3),
                "required_for_prediction": False,
                "is_core_analyte": True
            },
            "Alkaline_Phosphotase": {
                "type": "float",
                "label": "Alkaline Phosphatase (ALP)",
                "unit": "U/L",
                "aliases": [
                    r"alkaline\s+phosphatase", r"alkaline\s+phosphotase", r"\balp\b",
                    r"s\.\s*alp", r"serum\s+alkaline\s+phosphatase"
                ],
                "range": (20.0, 2500.0),
                "normal_range": (30.0, 120.0),
                "required_for_prediction": False,
                "is_core_analyte": True
            },
            "Alamine_Aminotransferase": {
                "type": "float",
                "label": "Alanine Aminotransferase (ALT / SGPT)",
                "unit": "U/L",
                "aliases": [
                    r"alamine\s+aminotransferase", r"alanine\s+aminotransferase",
                    r"\balt\b", r"\bsgpt\b", r"s\.g\.p\.t\.", r"s\.\s*alt"
                ],
                "range": (5.0, 2000.0),
                "normal_range": (7.0, 56.0),
                "required_for_prediction": False,
                "is_core_analyte": True
            },
            "Aspartate_Aminotransferase": {
                "type": "float",
                "label": "Aspartate Aminotransferase (AST / SGOT)",
                "unit": "U/L",
                "aliases": [
                    r"aspartate\s+aminotransferase", r"\bast\b", r"\bsgot\b",
                    r"s\.g\.o\.t\.", r"s\.\s*ast"
                ],
                "range": (5.0, 2000.0),
                "normal_range": (10.0, 40.0),
                "required_for_prediction": False,
                "is_core_analyte": True
            },
            "Total_Protiens": {
                "type": "float",
                "label": "Total Proteins",
                "unit": "g/dL",
                "aliases": [
                    r"total\s+proteins", r"total\s+protein", r"total\s+protiens",
                    r"\btp\b", r"s\.\s*total\s+protein"
                ],
                "range": (2.0, 14.0),
                "normal_range": (6.0, 8.3),
                "required_for_prediction": False,
                "is_core_analyte": True
            },
            "Albumin": {
                "type": "float",
                "label": "Serum Albumin",
                "unit": "g/dL",
                "aliases": [r"\balbumin\b", r"serum\s+albumin", r"s\.\s*albumin"],
                "range": (1.0, 7.0),
                "normal_range": (3.4, 5.4),
                "required_for_prediction": False,
                "is_core_analyte": True
            },
            "Albumin_and_Globulin_Ratio": {
                "type": "float",
                "label": "Albumin and Globulin Ratio (A/G)",
                "unit": "",
                "aliases": [
                    r"albumin\s+and\s+globulin\s+ratio", r"albumin\s+globulin\s+ratio",
                    r"a/g\s+ratio", r"ag\s+ratio", r"a:g\s+ratio"
                ],
                "range": (0.2, 3.5),
                "normal_range": (1.0, 2.2),
                "required_for_prediction": False,
                "baseline_default": 1.0,
                "default_reason": "Physiological baseline Albumin/Globulin ratio (1.0)"
            },
            "Age": {
                "type": "float",
                "label": "Age",
                "unit": "years",
                "aliases": [r"\bage\b", r"patient\s+age", r"\byrs\b"],
                "range": (1.0, 110.0),
                "required_for_prediction": False,
                "baseline_default": 45.0,
                "default_reason": "Median adult baseline age (45 years)"
            },
            "Gender": {
                "type": "category",
                "label": "Gender",
                "unit": "",
                "aliases": [r"\bgender\b", r"\bsex\b"],
                "options": {"male": "Male", "m": "Male", "female": "Female", "f": "Female"},
                "required_for_prediction": False,
                "baseline_default": "Male",
                "default_reason": "Baseline demographic category (Male)"
            }
        },
        "minimum_rule": "At least two hepatic enzymes/proteins (e.g. ALT, AST, Bilirubin, or Albumin) must be extracted."
    },

    "breast_cancer": {
        "title": "Breast Biopsy Cellular Morphometry",
        "fields": {
            "mean_radius": {
                "type": "float",
                "label": "Mean Nuclear Radius",
                "unit": "µm",
                "aliases": [
                    r"mean\s+radius", r"nuclear\s+radius", r"radius\s*\(mean\)",
                    r"cell\s+radius", r"radius_mean", r"\bradius\b"
                ],
                "range": (5.0, 35.0),
                "normal_range": (6.0, 14.0),
                "required_for_prediction": True,
                "is_core_analyte": True
            },
            "mean_texture": {
                "type": "float",
                "label": "Mean Texture Variation",
                "unit": "",
                "aliases": [
                    r"mean\s+texture", r"nuclear\s+texture", r"texture\s*\(mean\)",
                    r"texture\s+variation", r"texture_mean", r"\btexture\b"
                ],
                "range": (5.0, 45.0),
                "normal_range": (9.0, 20.0),
                "required_for_prediction": True,
                "is_core_analyte": True
            },
            "mean_perimeter": {
                "type": "float",
                "label": "Mean Nuclear Perimeter",
                "unit": "µm",
                "aliases": [
                    r"mean\s+perimeter", r"nuclear\s+perimeter", r"perimeter\s*\(mean\)",
                    r"perimeter_mean", r"\bperimeter\b"
                ],
                "range": (35.0, 250.0),
                "normal_range": (40.0, 90.0),
                "required_for_prediction": True,
                "is_core_analyte": True
            },
            "mean_area": {
                "type": "float",
                "label": "Mean Nuclear Area",
                "unit": "µm²",
                "aliases": [
                    r"mean\s+area", r"nuclear\s+area", r"area\s*\(mean\)",
                    r"area_mean", r"\barea\b"
                ],
                "range": (100.0, 3000.0),
                "normal_range": (140.0, 600.0),
                "required_for_prediction": True,
                "is_core_analyte": True
            },
            "mean_smoothness": {
                "type": "float",
                "label": "Mean Smoothness Index",
                "unit": "",
                "aliases": [
                    r"mean\s+smoothness", r"nuclear\s+smoothness", r"smoothness\s*\(mean\)",
                    r"smoothness\s+index", r"smoothness_mean", r"\bsmoothness\b"
                ],
                "range": (0.04, 0.25),
                "normal_range": (0.06, 0.11),
                "required_for_prediction": True,
                "is_core_analyte": True
            }
        },
        "minimum_rule": "All 5 cytological nuclear morphometry measurements must be extracted from the FNA pathology report."
    }
}


# ============================================================================
# TIER 1: MULTIMODAL VISION API EXTRACTOR (GEMINI & OPENAI)
# ============================================================================

def call_vision_api_gemini(image_bytes: bytes, mime_type: str, module_name: str) -> Optional[Dict[str, Any]]:
    """
    Submits image to Google Gemini Vision API using pure requests (zero extra SDK overhead).
    Requires GEMINI_API_KEY. Configurable via GEMINI_MODEL (default gemini-1.5-flash).
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash").strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    schema_info = MODULE_SCHEMAS.get(module_name)
    if not schema_info:
        return None

    fields_desc = []
    for k, v in schema_info["fields"].items():
        unit_str = f" in {v['unit']}" if v.get("unit") else ""
        fields_desc.append(f"- '{k}' ({v['type']}{unit_str}, label: {v['label']})")
    fields_list = "\n".join(fields_desc)

    import base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        f"You are a specialized clinical lab report data extraction system.\n"
        f"Extract patient measurements for the '{schema_info['title']}' panel from this image.\n"
        f"Allowed target fields:\n{fields_list}\n\n"
        f"CRITICAL CLINICAL RULES:\n"
        f"1. Extract ONLY patient measurements. Do NOT confuse normal reference ranges with patient values.\n"
        f"2. Return JSON ONLY with this schema: {{\"values\": {{<field_key>: <numeric_or_string_value>}}, \"confidence\": {{<field_key>: <0.0_to_1.0>}}, \"original_text\": {{<field_key>: \"<snippet>\"}}}}\n"
        f"3. If a field is not present or ambiguous in the report, DO NOT fabricate or invent a value; omit it or set it to null.\n"
        f"4. Do not include markdown formatting or backticks around JSON."
    )

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": mime_type if mime_type.startswith("image/") else "image/png",
                        "data": b64_image
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json"
        }
    }

    try:
        import requests
        resp = requests.post(url, json=payload, timeout=20)
        if resp.status_code == 200:
            res_data = resp.json()
            candidates = res_data.get("candidates", [])
            if candidates:
                raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                cleaned = raw_text.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
                    cleaned = re.sub(r"\n?```$", "", cleaned)
                parsed = json.loads(cleaned)
                logger.info(f"[VISION API] Gemini extraction succeeded for {module_name}")
                return parsed
        else:
            logger.warning(f"[VISION API] Gemini API error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"[VISION API] Gemini request failed: {e}")

    return None


def call_vision_api_openai(image_bytes: bytes, mime_type: str, module_name: str) -> Optional[Dict[str, Any]]:
    """
    Submits image to OpenAI Vision API if OPENAI_API_KEY is configured.
    Configurable via OPENAI_MODEL (default gpt-4o-mini).
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    schema_info = MODULE_SCHEMAS.get(module_name)
    if not schema_info:
        return None

    fields_desc = []
    for k, v in schema_info["fields"].items():
        unit_str = f" in {v['unit']}" if v.get("unit") else ""
        fields_desc.append(f"- '{k}' ({v['type']}{unit_str})")
    fields_list = "\n".join(fields_desc)

    import base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    actual_mime = mime_type if mime_type.startswith("image/") else "image/png"

    prompt = (
        f"You are a clinical lab report extraction engine. Extract patient measurements for:\n{fields_list}\n"
        f"Return strictly JSON: {{\"values\": {{<key>: <val>}}, \"confidence\": {{<key>: <0.0_to_1.0>}}, \"original_text\": {{<key>: \"snippet\"}}}}\n"
        f"Do not confuse reference intervals with patient results. Do not invent missing values."
    )

    try:
        import requests
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{actual_mime};base64,{b64_image}"}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0
        }
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=20)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            logger.info(f"[VISION API] OpenAI extraction succeeded for {module_name}")
            return parsed
        else:
            logger.warning(f"[VISION API] OpenAI API error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"[VISION API] OpenAI request failed: {e}")

    return None


# ============================================================================
# TIER 2: LOCAL TEXT EXTRACTION & OCR
# ============================================================================

def extract_text_from_pdf(pdf_bytes: bytes) -> Tuple[str, int]:
    """
    Extracts text from a digital PDF using pypdf.
    Returns (extracted_text, page_count).
    """
    try:
        import pypdf
        reader = pypdf.PdfReader(BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        pages_to_process = min(total_pages, MAX_PDF_PAGES)
        text_chunks = []

        for i in range(pages_to_process):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text_chunks.append(page_text)

        full_text = "\n".join(text_chunks).strip()
        return full_text, total_pages
    except Exception as e:
        logger.warning(f"[PDF EXTRACTOR] pypdf extraction failed: {e}")
        return "", 0


def extract_text_from_image(image_bytes: bytes) -> Tuple[str, bool]:
    """
    Preprocesses photograph/scan with Pillow and runs pytesseract OCR.
    Handles contrast enhancement, grayscale conversion, and missing tesseract binary.
    Returns (extracted_text, tesseract_available).
    """
    try:
        import pytesseract

        # Check configured Tesseract executable path if provided
        tess_cmd = os.environ.get("TESSERACT_CMD")
        if tess_cmd and os.path.exists(tess_cmd):
            pytesseract.pytesseract.tesseract_cmd = tess_cmd
        elif os.name == "nt":
            # Common Windows default locations
            common_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
            ]
            for p in common_paths:
                if os.path.exists(p):
                    pytesseract.pytesseract.tesseract_cmd = p
                    break

        # Load and preprocess image
        img = Image.open(BytesIO(image_bytes))

        # Auto-orient based on EXIF
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        # Convert to Grayscale & enhance contrast
        gray = img.convert("L")
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.8)

        # Scale up small images for better OCR resolution
        if enhanced.width < 1200:
            scale = 1200 / float(enhanced.width)
            enhanced = enhanced.resize((int(enhanced.width * scale), int(enhanced.height * scale)), Image.Resampling.LANCZOS)

        # Run OCR with English language whitelist
        text = pytesseract.image_to_string(enhanced, lang="eng", config="--psm 6")
        return text.strip(), True

    except Exception as e:
        err_str = str(e).lower()
        if "tesseract is not installed" in err_str or "not found" in err_str:
            logger.warning("[OCR] Tesseract binary not found on host system.")
            return "", False
        logger.warning(f"[OCR] pytesseract processing failed: {e}")
        return "", True


# ============================================================================
# CLINICAL REGEX PARSER & REFERENCE INTERVAL DISAMBIGUATION
# ============================================================================

def clean_numeric_value(val_str: Any) -> Optional[float]:
    """
    Cleans raw numeric string with optional decimal commas, units, and noise.
    Returns float or None if invalid.
    """
    if val_str is None:
        return None
    if isinstance(val_str, (int, float)):
        return float(val_str)
    
    s = str(val_str).strip().replace(",", ".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def parse_clinical_text(text: str, module_name: str) -> Dict[str, Any]:
    """
    Analyzes raw text using domain-specific clinical aliases and regex patterns.
    Disambiguates between the patient's actual result and laboratory reference intervals.
    """
    schema = MODULE_SCHEMAS.get(module_name)
    if not schema:
        return {}

    extracted_values: Dict[str, Any] = {}
    field_metadata: Dict[str, Any] = {}

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for field_name, fdef in schema["fields"].items():
        aliases = fdef.get("aliases", [])
        ftype = fdef.get("type", "float")
        val_range = fdef.get("range")
        foptions = fdef.get("options", {})

        # Try matching each alias
        for alias_pattern in aliases:
            if field_name in extracted_values:
                break

            regex = re.compile(alias_pattern, re.IGNORECASE)

            for line_idx, line in enumerate(lines):
                if regex.search(line):
                    # Disambiguation strategy:
                    # Look for numerical or categorical values in the matched line,
                    # or in the immediately following line (for multi-row report layouts).
                    candidate_lines = [line]
                    if line_idx + 1 < len(lines):
                        candidate_lines.append(lines[line_idx + 1])

                    search_text = "  ".join(candidate_lines)

                    if ftype in ("float", "int"):
                        val, orig_snip, conf = _extract_numeric_with_reference_filtering(
                            search_text, alias_pattern, val_range, fdef.get("unit", "")
                        )
                        if val is not None:
                            extracted_values[field_name] = int(val) if ftype == "int" else float(val)
                            field_metadata[field_name] = {
                                "source": "extracted",
                                "confidence": conf,
                                "original_text": orig_snip
                            }
                            break
                    elif ftype == "category":
                        cat_val, orig_snip, conf = _extract_category_value(search_text, foptions)
                        if cat_val is not None:
                            extracted_values[field_name] = cat_val
                            field_metadata[field_name] = {
                                "source": "extracted",
                                "confidence": conf,
                                "original_text": orig_snip
                            }
                            break

    return {
        "values": extracted_values,
        "metadata": field_metadata
    }


def _extract_numeric_with_reference_filtering(
    text: str, alias_pattern: str, val_range: Optional[Tuple[float, float]], unit: str
) -> Tuple[Optional[float], str, float]:
    """
    Extracts the patient's result while ignoring reference interval ranges
    such as '(4.0 - 5.6)', 'Ref: 70-99', 'Normal: < 120'.
    """
    # 1. Mask out reference range patterns in the text
    # e.g. (70 - 99), (0.6 - 1.2), Ref: 10 - 40, Range: 3.5 to 5.0
    masked_text = re.sub(r"\(\s*\d+(?:\.\d+)?\s*(?:-|–|to)\s*\d+(?:\.\d+)?\s*\)", " [REF_RANGE] ", text, flags=re.IGNORECASE)
    masked_text = re.sub(r"(?:ref|reference|normal|norm|range|interval|target)\s*:?\s*[<>]?\s*\d+(?:\.\d+)?(?:\s*(?:-|–|to)\s*\d+(?:\.\d+)?)?", " [REF_LABEL] ", masked_text, flags=re.IGNORECASE)

    # 2. Extract numbers with optional decimal comma replacement (e.g. 7,8 -> 7.8)
    clean_text = masked_text.replace(",", ".")
    # Match numbers that may have a decimal point
    num_matches = list(re.finditer(r"(?<![A-Za-z0-9_])(\d+(?:\.\d+)?)(?![A-Za-z0-9_])", clean_text))

    if not num_matches:
        return None, "", 0.0

    # 3. Find the number closest to the matched analyte label that falls within plausible range
    alias_match = re.search(alias_pattern, clean_text, re.IGNORECASE)
    alias_pos = alias_match.end() if alias_match else 0

    candidates = []
    for m in num_matches:
        val_str = m.group(1)
        try:
            val = float(val_str)
            # Check plausible medical range
            if val_range:
                min_v, max_v = val_range
                if not (min_v <= val <= max_v):
                    continue

            dist = abs(m.start() - alias_pos)
            # Bonus if immediately followed by expected unit
            unit_bonus = 0
            if unit:
                after_snippet = clean_text[m.end():m.end() + 15].lower()
                if unit.lower() in after_snippet or "%" in after_snippet:
                    unit_bonus = -50  # effectively decreases distance

            candidates.append((dist + unit_bonus, val, m.group(0), m.start(), m.end()))
        except ValueError:
            continue

    if not candidates:
        return None, "", 0.0

    # Sort by closest distance to label
    candidates.sort(key=lambda x: x[0])
    best = candidates[0]
    best_val = best[1]
    best_raw = best[2]

    # Context snippet
    start_ctx = max(0, best[3] - 15)
    end_ctx = min(len(text), best[4] + 25)
    snippet = text[start_ctx:end_ctx].strip()

    confidence = 0.92 if best[0] < 50 else 0.78
    return best_val, snippet, confidence


def _extract_category_value(text: str, options: Dict[str, Any]) -> Tuple[Optional[Any], str, float]:
    """Matches categorical values from synonyms."""
    for opt_key, opt_val in options.items():
        pattern = r"\b" + re.escape(opt_key) + r"\b"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            snippet = text[max(0, m.start() - 10):min(len(text), m.end() + 10)].strip()
            return opt_val, snippet, 0.90
    return None, "", 0.0


# ============================================================================
# TIER 3: MISSING FIELD & BASELINE IMPUTATION ORCHESTRATOR
# ============================================================================

def process_report_pipeline(
    file_bytes: bytes,
    filename: str,
    module_name: str,
    mime_type: str = "application/octet-stream"
) -> Dict[str, Any]:
    """
    Main extraction orchestrator:
    1. Tier 1: Multimodal Vision API (if keys present).
    2. Tier 2: Local text-based PDF or local pytesseract OCR.
    3. Tier 3: Module validation, schema compliance, baseline imputation.
    """
    if module_name not in SUPPORTED_MODULES:
        return {
            "success": False,
            "error": f"Invalid or unsupported diagnostic module: '{module_name}'."
        }

    schema = MODULE_SCHEMAS[module_name]
    engine_used = "unknown"
    extracted_data: Dict[str, Any] = {"values": {}, "metadata": {}}

    is_pdf = filename.lower().endswith(".pdf") or "pdf" in mime_type.lower()

    # --- Tier 1: Vision API attempt ---
    has_gemini = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    has_openai = bool(os.environ.get("OPENAI_API_KEY", "").strip())

    if (has_gemini or has_openai) and not is_pdf:
        vision_res = None
        if has_gemini:
            vision_res = call_vision_api_gemini(file_bytes, mime_type, module_name)
            if vision_res:
                engine_used = "vision_api_gemini"

        if not vision_res and has_openai:
            vision_res = call_vision_api_openai(file_bytes, mime_type, module_name)
            if vision_res:
                engine_used = "vision_api_openai"

        if vision_res and isinstance(vision_res.get("values"), dict):
            # Validate schema
            v_vals = {}
            v_meta = {}
            for k, val in vision_res["values"].items():
                if k in schema["fields"] and val is not None:
                    v_vals[k] = val
                    conf = vision_res.get("confidence", {}).get(k, 0.95)
                    snip = vision_res.get("original_text", {}).get(k, f"Extracted via {engine_used}")
                    v_meta[k] = {"source": "extracted", "confidence": float(conf), "original_text": str(snip)}

            if v_vals:
                extracted_data = {"values": v_vals, "metadata": v_meta}

    # --- Tier 2: Local PDF / OCR fallback ---
    if not extracted_data.get("values"):
        if is_pdf:
            pdf_text, page_cnt = extract_text_from_pdf(file_bytes)
            if pdf_text and len(pdf_text) > 30:
                engine_used = "local_pdf_text"
                extracted_data = parse_clinical_text(pdf_text, module_name)
            else:
                engine_used = "local_pdf_empty"
        else:
            ocr_text, tess_avail = extract_text_from_image(file_bytes)
            if ocr_text:
                engine_used = "local_tesseract_ocr"
                extracted_data = parse_clinical_text(ocr_text, module_name)
            else:
                engine_used = "local_ocr_unavailable" if not tess_avail else "local_ocr_no_text"

    # --- Tier 3: Baseline Imputation & Module Input Validation ---
    extracted_vals = extracted_data.get("values", {})
    field_metadata = extracted_data.get("metadata", {})

    final_values: Dict[str, Any] = {}
    defaulted_fields: List[Dict[str, Any]] = []
    missing_fields: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for fname, fdef in schema["fields"].items():
        if fname in extracted_vals:
            final_values[fname] = extracted_vals[fname]
        elif "baseline_default" in fdef:
            # Baseline permitted
            base_val = fdef["baseline_default"]
            final_values[fname] = base_val
            defaulted_fields.append({
                "field": fname,
                "label": fdef["label"],
                "value": base_val,
                "source": "baseline_default",
                "reason": fdef["default_reason"],
                "requires_confirmation": True
            })
            field_metadata[fname] = {
                "source": "defaulted",
                "confidence": 0.50,
                "original_text": fdef["default_reason"]
            }
        else:
            missing_fields.append({
                "field": fname,
                "label": fdef["label"],
                "unit": fdef.get("unit", "")
            })
            field_metadata[fname] = {
                "source": "missing",
                "confidence": 0.0,
                "original_text": "Not present in report"
            }

    # Evaluate minimum-input prediction policy
    can_predict = False
    core_extracted_count = sum(
        1 for f, d in schema["fields"].items()
        if d.get("is_core_analyte") and f in extracted_vals
    )

    if module_name == "diabetes":
        # Needs at least HbA1c or Glucose
        can_predict = ("HbA1c_level" in extracted_vals) or ("blood_glucose_level" in extracted_vals)
        if not can_predict:
            warnings.append("Insufficient glycemic markers: HbA1c or Fasting Blood Glucose must be present to predict diabetes risk.")

    elif module_name == "heart":
        # Needs at least BP or Cholesterol
        can_predict = ("bp" in extracted_vals) or ("chol" in extracted_vals)
        if not can_predict:
            warnings.append("Insufficient cardiovascular markers: Resting Blood Pressure (BP) or Serum Cholesterol must be present to evaluate risk.")

    elif module_name == "kidney":
        # Needs at least Serum Creatinine or Blood Urea
        can_predict = ("sc" in extracted_vals) or ("bu" in extracted_vals)
        if not can_predict:
            warnings.append("Insufficient renal markers: Serum Creatinine or Blood Urea must be present to evaluate chronic kidney disease risk.")

    elif module_name == "liver":
        # Needs at least 2 liver enzymes or proteins
        can_predict = core_extracted_count >= 2
        if not can_predict:
            warnings.append("Insufficient hepatic markers: At least 2 liver enzymes (ALT, AST, Bilirubin, Albumin) must be present to evaluate liver disease.")

    elif module_name == "breast_cancer":
        # Needs all 5 morphometry parameters
        bc_required = ["mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness"]
        can_predict = all(k in extracted_vals for k in bc_required)
        if not can_predict:
            missing_bc = [schema["fields"][k]["label"] for k in bc_required if k not in extracted_vals]
            warnings.append(f"Incomplete fine-needle aspirate cytology panel. Missing required measurements: {', '.join(missing_bc)}.")

    if engine_used == "local_ocr_unavailable" and not extracted_vals:
        warnings.append(
            "Local OCR engine (Tesseract) is not detected on the host system. "
            "Please review/complete form fields manually, upload a digital text-based PDF report, "
            "or configure GEMINI_API_KEY / OPENAI_API_KEY."
        )

    return {
        "success": True,
        "module": module_name,
        "engine": engine_used,
        "values": final_values,
        "field_metadata": field_metadata,
        "defaulted_fields": defaulted_fields,
        "missing_fields": missing_fields,
        "warnings": warnings,
        "can_predict": can_predict,
        "prediction": None,
        "probability": None,
        "report_id": None,
        "insights": None
    }
