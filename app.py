import os
import sys
import gc

# Configure environment variables to restrict TensorFlow CPU thread overhead and reduce RAM usage
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
# Do NOT force TF_USE_LEGACY_KERAS=1 as our vision CNN models are serialized in Keras 3 schema
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
import pandas as pd
import datetime
import json
import re
import uuid
import secrets
import smtplib
import traceback
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file, abort
from werkzeug.utils import secure_filename

# Modern Keras 3 loader with backward compatibility fallback
try:
    import keras
    from keras.models import load_model as keras_load_model
    from keras.preprocessing import image as keras_image
except ImportError:
    from tensorflow.keras.models import load_model as keras_load_model
    # pyrefly: ignore[missing-import]
    from tensorflow.keras.preprocessing import image as keras_image

from utils.google_sheets_logger import (
    log_patient_prediction,
    get_sheets_status,
    SPREADSHEET_NAME
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def _load_env_file(env_path):
    """Zero-dependency .env loader for local developer and production environments."""
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_env_file(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'medinova-clinical-secret-key-2026')

# Developer Master Gate Configuration (Strictly restricted to dharshankumarkumar3@gmail.com)
AUTHORIZED_DEV_EMAIL = os.environ.get('AUTHORIZED_DEV_EMAIL', 'dharshankumarkumar3@gmail.com').strip().lower()
DEV_ACCESS_KEY = os.environ.get('DEV_ACCESS_KEY', 'medinova_dev_2026_dharshan_secret').strip()

# Ephemeral in-memory OTP storage for developer two-factor authentication
DEV_OTP_CACHE = {
    'otp': None,
    'expires_at': None,
    'email': None
}

def send_developer_email(subject, html_content, attachment_path=None):
    """Deliver emails directly to developer via SMTP."""
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))
    smtp_user = os.environ.get("SMTP_USERNAME")
    smtp_pass = os.environ.get("SMTP_PASSWORD")
    use_tls = os.environ.get("SMTP_USE_TLS", "true").lower() in ["true", "1", "yes"]

    if not smtp_user or not smtp_pass:
        print(f"[DEVELOPER NOTICE] SMTP credentials not configured. Email to {AUTHORIZED_DEV_EMAIL} skipped.")
        return False, "SMTP credentials (SMTP_USERNAME/SMTP_PASSWORD) are not set in environment or .env."

    try:
        msg = MIMEMultipart()
        msg["From"] = f"MediNova Clinical Diagnostics <{smtp_user}>"
        msg["To"] = AUTHORIZED_DEV_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(html_content, "html"))

        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            filename = os.path.basename(attachment_path)
            part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
            msg.attach(part)

        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=15)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=15)
            if use_tls:
                server.starttls()

        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
        server.quit()
        return True, "Email successfully delivered to developer."
    except Exception as e:
        print(f"[SMTP ERROR] Failed sending to {AUTHORIZED_DEV_EMAIL}: {e}")
        return False, str(e)

def to_python_type(obj):
    if isinstance(obj, dict):
        return {str(k): to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [to_python_type(x) for x in obj.tolist()]
    return obj


# Configure Upload & Sample Folders (Anchored to BASE_DIR for Cloud WSGI/Gunicorn)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
SAMPLES_FOLDER = os.path.join(BASE_DIR, 'static', 'samples')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLES_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAMPLES_FOLDER'] = SAMPLES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Load Pre-trained Models
print("Loading medical diagnostic models...")

def load_pkl_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None

heart_model = load_pkl_model('heart_disease_model.pkl')
diabetes_model = load_pkl_model('diabetes_model.pkl')
kidney_model = load_pkl_model('kidney_model.pkl')
liver_model = load_pkl_model('liver_model.pkl')
breast_cancer_model = load_pkl_model('breast_cancer_model.pkl')

# Absolute filesystem paths to deep learning vision models
pneumonia_model_path = os.path.abspath(os.path.join(MODELS_DIR, 'pneumonia_model_new.h5'))
malaria_model_path = os.path.abspath(os.path.join(MODELS_DIR, 'malaria_cnn_model_fast.h5'))

# Global model references and diagnostics state
pneumonia_model = None
pneumonia_model_error = None
malaria_model = None
malaria_model_error = None

def load_vision_model(model_path, model_name="Vision Model"):
    """
    Safely load a Keras/TensorFlow HDF5 deep learning model with multi-tier fallback architecture,
    absolute path resolution, and comprehensive traceback logging.
    
    Returns:
        tuple: (model_instance, error_message_or_None)
    """
    abs_path = os.path.abspath(model_path)
    if not os.path.exists(abs_path):
        err_msg = f"{model_name} weight file not found at: {abs_path}"
        app.logger.error(f"[MODEL ERROR] {err_msg}")
        print(f"[MODEL ERROR] {err_msg}")
        return None, err_msg

    print(f"[MODEL LOADER] Loading {model_name} from: {abs_path} ({os.path.getsize(abs_path):,} bytes)...")

    # Tier 1: Load via Keras 3 with compile=False (optimal for models serialized in Keras 3 format)
    try:
        import keras
        model = keras.models.load_model(abs_path, compile=False)
        print(f"[MODEL LOADER] SUCCESS: {model_name} loaded via Keras 3 engine.")
        return model, None
    except Exception as e_k3:
        tb_k3 = traceback.format_exc()
        app.logger.warning(f"Keras 3 loader failed for {model_name}: {e_k3}\n{tb_k3}")
        print(f"[MODEL LOADER] Keras 3 attempt failed for {model_name}: {e_k3}")

    # Tier 2: Fallback to tensorflow.keras loader
    try:
        from tensorflow.keras.models import load_model as tf_load_model
        model = tf_load_model(abs_path, compile=False)
        print(f"[MODEL LOADER] SUCCESS: {model_name} loaded via tensorflow.keras engine.")
        return model, None
    except Exception as e_tf:
        tb_tf = traceback.format_exc()
        app.logger.warning(f"tensorflow.keras loader failed for {model_name}: {e_tf}\n{tb_tf}")
        print(f"[MODEL LOADER] tensorflow.keras attempt failed for {model_name}: {e_tf}")

    # Tier 3: Fallback to tf_keras (legacy Keras 2) with PatchedInputLayer for batch_shape
    try:
        import tf_keras
        class PatchedInputLayer(tf_keras.layers.InputLayer):
            @classmethod
            def from_config(cls, config):
                cfg = config.copy()
                if "batch_shape" in cfg:
                    cfg["batch_input_shape"] = cfg.pop("batch_shape")
                return super().from_config(cfg)

        model = tf_keras.models.load_model(
            abs_path,
            compile=False,
            custom_objects={"InputLayer": PatchedInputLayer}
        )
        print(f"[MODEL LOADER] SUCCESS: {model_name} loaded via tf_keras legacy compatibility patch.")
        return model, None
    except Exception as e_tfk:
        tb_tfk = traceback.format_exc()
        app.logger.error(f"tf_keras fallback failed for {model_name}: {e_tfk}\n{tb_tfk}")
        print(f"[MODEL LOADER] tf_keras fallback failed for {model_name}: {e_tfk}")

    # All loader tiers failed
    full_err = f"Failed to load {model_name} from {abs_path}. All loader tiers (Keras 3, TF Keras, and legacy patch) raised exceptions."
    app.logger.error(f"[MODEL LOADER FATAL] {full_err}")
    print(f"[MODEL LOADER FATAL] {full_err}")
    return None, full_err

def get_pneumonia_model():
    """Lazy-load Pneumonia Keras model on demand and cache globally."""
    global pneumonia_model, pneumonia_model_error
    if pneumonia_model is None:
        pneumonia_model, pneumonia_model_error = load_vision_model(pneumonia_model_path, "Pneumonia Radiography CNN Model")
    return pneumonia_model

def get_malaria_model():
    """Lazy-load Malaria Keras model on demand and cache globally."""
    global malaria_model, malaria_model_error
    if malaria_model is None:
        malaria_model, malaria_model_error = load_vision_model(malaria_model_path, "Malaria Parasite Vision CNN Model")
    return malaria_model

def verify_vision_models_startup(preload=False):
    """
    Startup verification check: verifies model weight paths and file presence on disk.
    If preload=True or PRELOAD_VISION_MODELS=1, pre-warms the models into global memory.
    """
    print("--- [STARTUP VERIFICATION] Vision Deep Learning Models ---")
    for name, p in [("Malaria Vision CNN", malaria_model_path), ("Pneumonia Chest Radiography CNN", pneumonia_model_path)]:
        if os.path.exists(p):
            print(f"  [OK] {name}: Found at {p} ({os.path.getsize(p):,} bytes)")
        else:
            print(f"  [MISSING] {name}: NOT found at {p}")

    if preload or os.environ.get("PRELOAD_VISION_MODELS", "0").lower() in ("1", "true"):
        print("  [PRELOAD] Eagerly loading vision models into global state...")
        get_malaria_model()
        get_pneumonia_model()
    else:
        print("  [LAZY-LOAD] Vision models will be lazy-loaded on first diagnostic request (conserving RAM).")
    print("---------------------------------------------------------")

# Run startup filesystem verification for vision models
verify_vision_models_startup()

# ============================================================================
# COMPREHENSIVE CLINICAL INSIGHTS & EMERGENCY PROTOCOLS
# ============================================================================
DISEASE_INSIGHTS = {
    'heart_disease': {
        'positive': {
            'is_positive': True,
            'title': 'Cardiovascular Risk Indicators Detected',
            'severity': 'High Clinical Risk - Immediate Consultation Advised',
            'badge': 'ATTENTION REQUIRED',
            'status_color': 'danger',
            'description': 'AI algorithmic analysis identified significant cardiovascular markers (such as elevated ST depression, atypical chest pressure parameters, or abnormal resting ECG markers) that correlate with coronary artery disease risk.',
            'specialist': 'Board-Certified Cardiologist & Coronary Care Specialist',
            'urgency_window': 'Schedule clinical consultation within 24 to 48 hours.',
            'risk_factors': [
                'Elevated systolic or diastolic resting blood pressure',
                'Hypercholesterolemia (elevated serum cholesterol > 200 mg/dL)',
                'Exercise-induced angina or ischemic ST depression patterns',
                'Fluoroscopy showing multiple major vessels with calcification',
                'Underlying metabolic load or family history of myocardial infarction'
            ],
            'emergency_symptoms': [
                'Crushing, squeezing chest pressure or tightness radiating to left arm, neck, or jaw',
                'Sudden severe shortness of breath, cold sweats, or unprovoked nausea/dizziness',
                'Sudden irregular heart palpitations accompanied by lightheadedness or syncope (fainting)'
            ],
            'recommendations': [
                'Immediately schedule a comprehensive consultation with a cardiologist',
                'Undergo a 12-lead Electrocardiogram (ECG) and an Echocardiogram (Echo Doppler)',
                'Conduct cardiac enzyme biomarkers panel (Troponin I/T, CK-MB, hs-CRP)',
                'Refrain from high-intensity physical exertion or heavy lifting until cleared by a doctor',
                'Maintain an accurate twice-daily resting Blood Pressure and Heart Rate log'
            ],
            'doctor_questions': [
                'Do these clinical biomarkers indicate coronary artery stenosis or microvascular ischemia?',
                'Should I undergo a Coronary CT Angiography or nuclear myocardial perfusion scan?',
                'Are preventive cardioprotective medications (e.g., statins, ACE inhibitors, or beta blockers) indicated for me?',
                'What target resting systolic/diastolic blood pressure and LDL levels should I strictly maintain?'
            ],
            'diet_dos': [
                'Heart-healthy Mediterranean diet: high in leafy greens, avocados, walnuts, and extra virgin olive oil',
                'Omega-3 rich foods (salmon, flaxseeds, chia seeds) to lower systemic arterial inflammation',
                'High-soluble fiber foods (steel-cut oats, legumes, berries) to actively bind and lower serum cholesterol'
            ],
            'diet_donts': [
                'Strictly avoid high sodium (canned soups, processed meats, chips) - keep sodium < 1,500 mg/day',
                'Eliminate trans fats, hydrogenated oils, deep-fried snacks, and fatty cuts of red meat',
                'Avoid excessive caffeine, energy drinks, and tobacco products which cause acute arterial vasoconstriction'
            ],
            'recommended_tests': ['12-Lead ECG', 'Echocardiogram', 'Lipid Panel', 'Troponin-I', 'Coronary CT Angiogram']
        },
        'negative': {
            'is_positive': False,
            'title': 'Cardiovascular Profile Within Normal Limits',
            'severity': 'Low Risk Detected - Favorable Parameters',
            'badge': 'OPTIMAL HEALTH',
            'status_color': 'safe',
            'description': 'AI assessment detected no statistically significant risk factors for coronary artery disease based on the submitted clinical parameters.',
            'specialist': 'Primary Care Physician / Preventive Cardiologist',
            'urgency_window': 'Routine annual wellness screening recommended.',
            'tips': [
                'Maintain 150 minutes of moderate aerobic cardiovascular activity weekly (e.g., brisk walking, swimming)',
                'Continue maintaining balanced blood pressure below 120/80 mm Hg and cholesterol < 200 mg/dL',
                'Keep fasting blood glucose stable and maintain body mass index (BMI) within 18.5 - 24.9',
                'Perform an annual preventive wellness ECG and lipid panel screening'
            ],
            'diet_dos': [
                'Eat a vibrant rainbow of seasonal fruits, antioxidant-rich berries, and dark leafy vegetables',
                'Incorporate lean proteins (fish, poultry, legumes) and whole unrefined grains'
            ],
            'diet_donts': [
                'Avoid high-sodium fast food and sugar-sweetened beverages to preserve vascular elasticity'
            ],
            'recommended_tests': ['Annual Routine Wellness Lipid Panel', 'Routine Annual Resting ECG Checkup']
        }
    },
    'diabetes': {
        'positive': {
            'is_positive': True,
            'title': 'Elevated Glycemic & Metabolic Risk Detected',
            'severity': 'High Risk of Type 2 Diabetes / Dysglycemia',
            'badge': 'ATTENTION REQUIRED',
            'status_color': 'danger',
            'description': 'Based on submitted biomarkers (glycated hemoglobin HbA1c, fasting plasma glucose, BMI, and hypertension history), the AI model flagged positive indicators consistent with impaired glucose tolerance or clinical diabetes.',
            'specialist': 'Consult an Endocrinologist & Certified Diabetes Educator',
            'urgency_window': 'Schedule a medical consultation within 3 to 5 days.',
            'risk_factors': [
                'Elevated HbA1c level (indicating chronic hyperglycemia over the past 90 days)',
                'Elevated blood glucose levels exceeding normal fasting or post-prandial ranges',
                'Increased Body Mass Index (BMI > 25 kg/m² indicating excess visceral adiposity)',
                'Hypertension or co-existing cardiovascular strain amplifying insulin resistance'
            ],
            'emergency_symptoms': [
                'Sudden extreme thirst (polydipsia) paired with frequent urination (polyuria)',
                'Fruity breath odor, rapid deep breathing, confusion, or severe nausea (symptoms of Diabetic Ketoacidosis - DKA)',
                'Sudden blurred vision, unexplained dizziness, extreme fatigue, or numbness/tingling in hands/feet'
            ],
            'recommendations': [
                'Schedule a confirmatory venous blood draw for Fasting Plasma Glucose and Oral Glucose Tolerance Test (OGTT)',
                'Obtain a home glucometer to record fasting, 2-hour post-meal, and bedtime glucose values',
                'Consult a licensed clinical dietician for a personalized Low-Glycemic Index (GI) meal plan',
                'Engage in 20-30 minutes of daily post-meal moderate walking to boost peripheral insulin sensitivity',
                'Schedule baseline microvascular screenings (diabetic retinal exam and urine microalbuminuria test)'
            ],
            'doctor_questions': [
                'What is my definitive diagnosis: prediabetes, type 2 diabetes, or secondary dysglycemia?',
                'Do you recommend starting oral insulin-sensitizing therapy (such as Metformin) immediately?',
                'What is my individual target HbA1c percentage over the next 3 to 6 months?',
                'How often should I monitor my blood glucose throughout the day?'
            ],
            'diet_dos': [
                'Complex carbohydrates with high fiber: quinoa, brown rice, steel-cut oats, and legumes',
                'Non-starchy cruciferous vegetables: broccoli, spinach, kale, bell peppers, and cauliflower',
                'Healthy fats and lean proteins that stabilize postprandial glucose spikes'
            ],
            'diet_donts': [
                'Eliminate refined sugars, fruit juices, sodas, energy drinks, syrups, and sweetened bakery products',
                'Avoid refined carbohydrates: white bread, instant noodles, white rice, and sugary cereals',
                'Limit excessive tropical fruits with high fructose content (mangoes, ripe bananas, dried dates)'
            ],
            'recommended_tests': ['Venous Fasting Plasma Glucose', 'Oral Glucose Tolerance Test (OGTT)', 'C-Peptide Level', 'Lipid Panel', 'Urine Microalbumin/Creatinine Ratio']
        },
        'negative': {
            'is_positive': False,
            'title': 'Glycemic Markers Within Normal Range',
            'severity': 'Low Risk Detected - Optimal Glucose Regulation',
            'badge': 'OPTIMAL HEALTH',
            'status_color': 'safe',
            'description': 'Your clinical markers (HbA1c, glucose, and body mass metrics) show favorable insulin sensitivity with no indicators of active diabetes.',
            'specialist': 'General Physician / Preventive Health Specialist',
            'urgency_window': 'Annual routine metabolic panel recommended.',
            'tips': [
                'Maintain an active daily lifestyle with regular resistance and cardiovascular exercise',
                'Keep HbA1c below 5.7% and fasting blood sugar within 70 - 99 mg/dL',
                'Practice portion control and limit nighttime snacking on refined sweets',
                'Stay well-hydrated with pure water instead of sweetened sodas and juices'
            ],
            'diet_dos': [
                'Focus on whole foods, diverse dietary fibers, seeds, nuts, and clean lean proteins'
            ],
            'diet_donts': [
                'Avoid regular consumption of sugary beverages and processed ultra-refined snacks'
            ],
            'recommended_tests': ['Annual Fasting Blood Sugar Test', 'Annual HbA1c Screening']
        }
    },
    'kidney_disease': {
        'positive': {
            'is_positive': True,
            'title': 'Renal Functional Markers Abnormal - CKD Risk',
            'severity': 'High Clinical Risk - Nephrology Evaluation Required',
            'badge': 'ATTENTION REQUIRED',
            'status_color': 'danger',
            'description': 'AI multi-parametric analysis of urinalysis and blood biochemistry detected markers consistent with impaired glomerular filtration rate and chronic kidney disease (CKD).',
            'specialist': 'Consult a Board-Certified Nephrologist',
            'urgency_window': 'Urgent nephrology evaluation recommended within 48 to 72 hours.',
            'risk_factors': [
                'Proteinuria (Albumin present in urine reflecting glomerular leakage)',
                'Elevated Serum Creatinine (> 1.2 mg/dL) and Blood Urea Nitrogen (BUN)',
                'Abnormal urine specific gravity and microscopic sediment indicators (RBCs / Pus cells)',
                'Hypertension causing renal arteriolar nephrosclerosis',
                'Anemia or depleted hemoglobin associated with reduced erythropoietin secretion'
            ],
            'emergency_symptoms': [
                'Severe bilateral lower-extremity peripheral edema (swollen legs/ankles) or facial puffiness',
                'Marked reduction in urine output (oliguria/anuria) or foamy/cola-colored blood in urine',
                'Sudden severe shortness of breath due to pulmonary fluid overload',
                'Severe intractable nausea, metallic taste in mouth, cognitive fog, or muscle twitching (uremia signs)'
            ],
            'recommendations': [
                'Undergo immediate confirmatory Serum Creatinine, eGFR calculation, and 24-hour Urine Protein test',
                'Schedule a Renal Ultrasound (KUB - Kidney, Ureter, Bladder) to assess kidney size and cortical thickness',
                'Strictly avoid all Over-The-Counter Nephrotoxic drugs: Ibuprofen, Naproxen, Diclofenac (NSAIDs)',
                'Undergo strict blood pressure management aiming for systolic BP < 120-130 mm Hg',
                'Work with a renal dietician to regulate dietary protein, sodium, potassium, and phosphorus'
            ],
            'doctor_questions': [
                'What is my estimated Glomerular Filtration Rate (eGFR) and what stage of CKD does it correspond to?',
                'Are ACE inhibitors (like Enalapril) or ARBs (like Losartan) suitable to protect my renal microvasculature?',
                'Do I need to restrict dietary potassium (bananas, potatoes) or dietary phosphorus (dairy)?',
                'Are there reversible etiologies (such as renal stone obstruction, medication effect, or dehydration)?'
            ],
            'diet_dos': [
                'Low-sodium intake: keep sodium under 1,500 mg daily to reduce renal glomerular pressure',
                'Moderate, high-biological-value protein intake (strictly adhere to nephrologist-prescribed grams/day)',
                'Consume kidney-friendly fruits (apples, berries, cauliflower) based on serum potassium levels'
            ],
            'diet_donts': [
                'Strictly avoid excessive salt, processed canned goods, bouillon cubes, and MSG',
                'Avoid protein powders, high-dosage creatine supplements, and crash high-protein keto diets',
                'Avoid excessive dark colas and packaged foods with added phosphate preservatives'
            ],
            'recommended_tests': ['Serum Creatinine & eGFR', 'Blood Urea Nitrogen (BUN)', 'Urine Albumin-to-Creatinine Ratio (uACR)', 'Electrolyte Panel (K+, Na+, Ca2+)', 'Renal Ultrasound']
        },
        'negative': {
            'is_positive': False,
            'title': 'Renal Biomarkers Within Healthy Limits',
            'severity': 'Low Risk Detected - Healthy Kidney Function',
            'badge': 'OPTIMAL HEALTH',
            'status_color': 'safe',
            'description': 'Renal filtration, urinary parameters, serum creatinine, and electrolyte values indicate intact glomerular filtration and healthy renal physiology.',
            'specialist': 'General Practitioner / Family Physician',
            'urgency_window': 'Standard annual wellness checkup.',
            'tips': [
                'Drink adequate water throughout the day (approx. 2 to 2.5 liters) to support renal waste clearance',
                'Avoid habitual misuse of NSAID painkillers (e.g. ibuprofen); choose safer alternatives when needed',
                'Maintain healthy blood pressure and blood sugar, the two primary safeguards against renal damage',
                'Perform an annual routine urinalysis and basic metabolic panel'
            ],
            'diet_dos': [
                'Stay well-hydrated, consume fresh hydrating vegetables, and season foods with fresh herbs instead of excess salt'
            ],
            'diet_donts': [
                'Avoid excessive table salt and chronic dehydration during intense heat or exercise'
            ],
            'recommended_tests': ['Annual Routine Urinalysis', 'Annual Serum Creatinine Screening']
        }
    },
    'liver_disease': {
        'positive': {
            'is_positive': True,
            'title': 'Hepatic Enzyme & Function Indicators Abnormal',
            'severity': 'High Probability of Hepatic Dysfunction / Inflammation',
            'badge': 'ATTENTION REQUIRED',
            'status_color': 'danger',
            'description': 'Elevated transaminases (ALT/AST), alkaline phosphatase, or bilirubin ratios reflect potential hepatocellular injury, cholestasis, fatty liver changes, or hepatic inflammation.',
            'specialist': 'Consult a Hepatologist / Gastroenterologist',
            'urgency_window': 'Schedule a gastroenterologist/hepatologist visit within 3 to 5 days.',
            'risk_factors': [
                'Elevated Direct and Total Bilirubin indicating impaired hepatic clearance',
                'Elevated ALT (Alamine Aminotransferase) and AST indicating active hepatocyte necrosis',
                'Alkaline Phosphatase elevation suggesting biliary duct involvement or inflammation',
                'Altered Albumin and Albumin-to-Globulin (A/G) ratio indicating compromised protein synthesis',
                'Risk factors including metabolic syndrome, steatosis, viral exposure, or alcohol/toxin intake'
            ],
            'emergency_symptoms': [
                'Yellowing of skin or sclera of the eyes (jaundice) accompanied by dark amber urine and pale stools',
                'Abdominal distension or fluid retention in belly (ascites) and swelling in legs',
                'Vomiting blood or dark black tarry stools (melena - signs of esophageal variceal bleeding)',
                'Acute confusion, marked disorientation, slurred speech, or flapping hand tremors (hepatic encephalopathy)'
            ],
            'recommendations': [
                'Schedule a liver ultrasound or FibroScan to evaluate liver stiffness and steatosis stage',
                'Complete viral hepatitis serology panel (HBsAg for Hepatitis B, Anti-HCV for Hepatitis C)',
                'Strictly avoid 100% of alcoholic beverages and all non-essential hepatotoxic supplements/medications',
                'Review every regular prescription with your physician to ensure liver-safe dosing',
                'Undergo coagulation profile screening (Prothrombin Time / INR) to evaluate synthesis function'
            ],
            'doctor_questions': [
                'What is the underlying cause: metabolic dysfunction-associated steatohepatitis (MASH/NASH), viral, or toxic?',
                'Does my liver imaging show any signs of fibrosis or early portal hypertension?',
                'Are my current medications safe for my liver enzymes, or do dosages need adjustments?',
                'What lifestyle and weight loss targets will best reverse hepatic fat infiltration?'
            ],
            'diet_dos': [
                'Emphasize antioxidant-rich foods: cruciferous vegetables, green tea, black coffee (shown to protect hepatocytes)',
                'Anti-inflammatory Mediterranean diet with abundant omega-3s, artichokes, and fiber',
                'High-fiber plant-based proteins (beans, lentils, tofu) and lean poultry'
            ],
            'diet_donts': [
                'Strictly zero alcohol consumption - even small amounts accelerate ongoing hepatocyte damage',
                'Avoid high-fructose corn syrup, candy, sugary sodas (fructose drives hepatic de novo lipogenesis)',
                'Avoid saturated fats, fried foods, raw or undercooked shellfish (risk of Vibrio vulnificus)'
            ],
            'recommended_tests': ['Comprehensive Liver Function Panel (LFT)', 'Liver Ultrasound / FibroScan', 'Hepatitis B & C Serology', 'Coagulation PT/INR', 'Serum Ferritin']
        },
        'negative': {
            'is_positive': False,
            'title': 'Hepatic Function Within Normal Range',
            'severity': 'Low Risk Detected - Healthy Liver Profile',
            'badge': 'OPTIMAL HEALTH',
            'status_color': 'safe',
            'description': 'Serum bilirubin, alkaline phosphatase, transaminases, and protein synthesis parameters are balanced and indicative of normal hepatic metabolic function.',
            'specialist': 'Primary Care Physician',
            'urgency_window': 'Routine annual wellness screening.',
            'tips': [
                'Maintain a healthy weight to prevent metabolic liver infiltration and steatosis',
                'Avoid excessive or binge alcohol consumption to safeguard liver longevity',
                'Ensure Hepatitis A and B vaccinations are up-to-date',
                'Always use prescription and over-the-counter medications only according to directions'
            ],
            'diet_dos': [
                'Stay hydrated, eat fresh seasonal vegetables, and enjoy moderate black coffee for hepatic health'
            ],
            'diet_donts': [
                'Avoid heavy alcohol consumption, excessive fried foods, and untested herbal supplements'
            ],
            'recommended_tests': ['Annual Routine Liver Function Screening']
        }
    },
    'breast_cancer': {
        'malignant': {
            'is_positive': True,
            'title': 'Cellular Metrics Consistent with Malignancy',
            'severity': 'High Priority - Immediate Oncology Referral Required',
            'badge': 'ATTENTION REQUIRED',
            'status_color': 'danger',
            'description': 'AI feature vector analysis of cellular nucleus morphology (mean radius, texture irregularity, perimeter distortion, area expansion, and edge smoothness) matches characteristics of malignant breast neoplasms.',
            'specialist': 'Consult a Surgical Oncologist & Breast Oncology Specialist',
            'urgency_window': 'Urgent diagnostic confirmation within 24 to 48 hours.',
            'risk_factors': [
                'Increased nuclear mean radius and enlarged cellular area indicating rapid mitotic proliferation',
                'Nuclear border irregularity, high texture variance, and abnormal perimeter ratios',
                'Loss of smooth cellular boundary architecture typical of invasive neoplastic transformation',
                'Associated clinical factors: palpable lump, family history (BRCA1/BRCA2), or prior abnormal imaging'
            ],
            'emergency_symptoms': [
                'Hard, non-tender, immovable solitary breast mass or axillary lymph node enlargement',
                'Skin changes: peau d\'orange (orange-peel skin dimpling), breast skin retraction, or ulceration',
                'Spontaneous serosanguinous (bloody or clear pink) nipple discharge from a single duct',
                'Inverted nipple or newly developed erythema/warmth across the breast tissue'
            ],
            'recommendations': [
                'Consult an oncology center immediately for confirmatory clinical breast examination',
                'Schedule a Diagnostic Digital Bilateral Mammogram and high-resolution Breast Ultrasound',
                'Undergo a Core Needle Biopsy (CNB) of the suspicious lesion with histopathological review',
                'Request immunohistochemical staining (ER, PR, HER2-neu, and Ki-67 proliferation index)',
                'Prepare all prior imaging disks and clinical notes for the multidisciplinary tumor board review'
            ],
            'doctor_questions': [
                'What is the recommended biopsy method (ultrasound-guided core biopsy or stereotactic)?',
                'Do you recommend a Breast MRI to evaluate lesion multicentricity or contralateral tissue?',
                'Should I consult a genetic counselor for BRCA1, BRCA2, or multi-gene panel testing?',
                'What is the staging and anticipated timeline for treatment planning?'
            ],
            'diet_dos': [
                'Nutrient-dense whole-food diet: cruciferous vegetables (rich in sulforaphane), berries, flaxseed',
                'Adequate lean protein to maintain immune reserves and muscle mass during upcoming clinical workup',
                'Plant-based anti-inflammatory foods and hydration'
            ],
            'diet_donts': [
                'Strictly avoid alcohol consumption (alcohol increases circulating estrogens and oncogenic risk)',
                'Avoid ultra-processed meats, trans fats, and excess refined sugars',
                'Do not begin untested alternative supplements without oncologist clearance'
            ],
            'recommended_tests': ['Diagnostic Digital Mammogram', 'Breast Ultrasound', 'Image-Guided Core Needle Biopsy', 'Receptor Status (ER/PR/HER2)', 'Breast MRI']
        },
        'benign': {
            'is_positive': False,
            'title': 'Cellular Metrics Consistent with Benign Tissue',
            'severity': 'Low Risk Detected - Non-Malignant Characteristics',
            'badge': 'OPTIMAL HEALTH',
            'status_color': 'safe',
            'description': 'The nuclear dimensions, smoothness, and texture parameters align with benign, non-cancerous cellular structures (such as fibroadenoma or normal glandular tissue).',
            'specialist': 'Gynecologist / Women\'s Health Specialist',
            'urgency_window': 'Continue standard age-appropriate breast screening schedule.',
            'tips': [
                'Perform monthly breast self-examinations to stay familiar with your normal tissue texture',
                'Follow clinical guidelines for annual or biennial screening mammography (age 40+ or per doctor advice)',
                'Promptly report any new, persistent, or progressive palpable lumps to your physician',
                'Maintain a healthy body weight and active lifestyle'
            ],
            'diet_dos': [
                'Eat balanced, fiber-rich whole foods, antioxidant-rich fruits, and leafy greens'
            ],
            'diet_donts': [
                'Limit alcohol consumption and avoid smoking'
            ],
            'recommended_tests': ['Routine Screening Mammogram (as recommended by age)', 'Routine Annual Clinical Breast Exam']
        }
    },
    'malaria': {
        'infected': {
            'is_positive': True,
            'title': 'Parasitized Erythrocytes Detected (Malaria Positive)',
            'severity': 'Acute Parasitic Infection - Prompt Medical Treatment Mandated',
            'badge': 'ATTENTION REQUIRED',
            'status_color': 'danger',
            'description': 'Deep convolutional neural network image analysis of the Giemsa-stained thin blood smear detected intracellular trophozoite or ring-form inclusions consistent with Plasmodium parasite invasion.',
            'specialist': 'Consult an Infectious Disease Specialist / Emergency Physician',
            'urgency_window': 'Seek immediate medical evaluation today (within 6 to 12 hours).',
            'risk_factors': [
                'Recent travel to or residence in malaria-endemic tropical/subtropical regions',
                'Exposure to Anopheles mosquito bites without chemoprophylaxis or bed nets',
                'Immunosuppression, pregnancy, or lack of prior acquired immunity',
                'Delayed diagnosis risking progression to severe falciparum malaria complications'
            ],
            'emergency_symptoms': [
                'Paroxysmal shaking chills (rigors) followed by high spiking fever (103-105°F) and drenching sweats',
                'Altered mental status, confusion, seizures, or unresponsiveness (Cerebral Malaria warning!)',
                'Severe dark/black tea-colored urine (Blackwater fever due to massive intravascular hemolysis)',
                'Severe anemia, jaundice, persistent vomiting, or respiratory distress/acidotic breathing'
            ],
            'recommendations': [
                'Go to a clinic or hospital emergency room immediately for definitive blood smear confirmation and parasite density quantification',
                'Initiate prescribed Antimalarial therapy immediately (such as Artemisinin-based Combination Therapy - ACT)',
                'Maintain vigorous oral hydration with electrolyte solutions to support renal perfusion',
                'Do not discontinue antimalarial medication early; complete the exact full prescribed course',
                'Monitor for signs of hypoglycemia and severe hemolytic anemia'
            ],
            'doctor_questions': [
                'Which Plasmodium species is present (Plasmodium falciparum, vivax, ovale, or malariae)?',
                'What is the quantitative parasite count / percentage of infected red blood cells?',
                'If P. vivax or P. ovale, will Primaquine therapy be needed to eradicate dormant liver hypnozoites (after G6PD testing)?',
                'Do my platelet counts and hemoglobin levels require close in-hospital monitoring?'
            ],
            'diet_dos': [
                'High-electrolyte fluids: coconut water, oral rehydration salts (ORS), clear broths to replace sweat losses',
                'Easily digestible, nutrient-dense soft foods: rice congee, boiled potatoes, bananas, and yogurt',
                'Iron and vitamin C-rich foods during recovery to assist erythrocyte replenishment'
            ],
            'diet_donts': [
                'Avoid heavy, greasy, fried, or spicy foods that irritate a fever-stressed gastrointestinal tract',
                'Avoid caffeine, alcohol, and unpasteurized drinks',
                'Do not take unverified herbal remedies in place of verified antimalarial medication'
            ],
            'recommended_tests': ['Thick and Thin Blood Smears with Parasite Count', 'Malaria Rapid Diagnostic Test (RDT)', 'Complete Blood Count (CBC) with Platelets', 'Serum Bilirubin & Liver Panel', 'G6PD Enzyme Assay']
        },
        'uninfected': {
            'is_positive': False,
            'title': 'No Intra-Erythrocytic Parasites Detected',
            'severity': 'Negative for Malaria Parasites in Sample',
            'badge': 'OPTIMAL HEALTH',
            'status_color': 'safe',
            'description': 'Deep learning vision analysis found clean, normal erythrocyte morphology with no identifiable Plasmodium ring stages, schizonts, or trophozoites in the submitted blood smear image.',
            'specialist': 'General Practitioner / Travel Medicine Physician',
            'urgency_window': 'If active unexplained fever persists, follow up for repeat blood smear.',
            'tips': [
                'If you have an active fever after tropical travel, note that repeat blood smears over 48 hours may be needed as parasite levels fluctuate',
                'Continue mosquito bite prevention: DEET-based repellents, permethrin-treated clothing, bed nets',
                'Consider prescribed antimalarial prophylaxis if currently traveling in high-transmission areas',
                'Seek prompt medical evaluation if high fevers or rigors reoccur'
            ],
            'diet_dos': [
                'Maintain regular hydration, balanced nutrition, and adequate rest'
            ],
            'diet_donts': [
                'Avoid unprotected exposure to mosquitoes during dusk-to-dawn peak biting hours'
            ],
            'recommended_tests': ['Repeat Peripheral Blood Smear in 24-48 hours if febrile symptoms persist']
        }
    },
    'pneumonia': {
        'positive': {
            'is_positive': True,
            'title': 'Pulmonary Infiltrates / Consolidation Detected',
            'severity': 'Acute Lower Respiratory Infection - Clinical Action Required',
            'badge': 'ATTENTION REQUIRED',
            'status_color': 'danger',
            'description': 'Deep neural network chest radiograph visual analyzer detected radiographic opacification, alveolar consolidation, or interstitial infiltrates characteristic of bacterial or viral pneumonia.',
            'specialist': 'Consult a Pulmonologist / Respiratory Physician or Visit Urgent Care',
            'urgency_window': 'Obtain medical evaluation today (within 12 to 24 hours).',
            'risk_factors': [
                'Airspace consolidation or patchy opacity obscuring normal lung parenchyma on X-ray',
                'Recent viral upper respiratory infection (Influenza, RSV, COVID-19) predisposing to secondary pneumonia',
                'Age > 65 or underlying comorbidities (COPD, asthma, diabetes, heart failure, immunocompromise)',
                'History of smoking, vaping, or environmental pollutant inhalation'
            ],
            'emergency_symptoms': [
                'Severe shortness of breath or rapid labored breathing even while resting',
                'Pulse oximeter reading below 92% SpO2 (hypoxemia!) or blue/gray discoloration of lips/fingernails (cyanosis)',
                'Sharp stabbing pleuritic chest pain exacerbated by deep breathing or coughing',
                'Confusion, delirium, or severe lethargy, especially in older adults'
            ],
            'recommendations': [
                'Seek immediate clinical consultation for chest auscultation (stethoscope crackles/rhonchi evaluation)',
                'Have your blood oxygen saturation (SpO2) and respiratory rate measured',
                'Obtain sputum culture and Complete Blood Count (CBC) with differential to distinguish bacterial vs viral etiology',
                'Take all prescribed antibiotics or antivirals exactly on schedule; do not miss or discontinue doses early',
                'Practice deep breathing exercises or use an incentive spirometer to prevent alveolar collapse (atelectasis)'
            ],
            'doctor_questions': [
                'Is my pneumonia radiographic pattern bacterial (lobar) or viral/atypical (interstitial)?',
                'What is my CURB-65 or Pneumonia Severity Index (PSI) score, and is outpatient management safe for me?',
                'Do I require supplemental oxygen or nebulizer bronchodilator therapy?',
                'When should I schedule a follow-up chest radiograph to confirm complete resolution of infiltrates?'
            ],
            'diet_dos': [
                'Continuous warm hydration: broths, herbal teas with honey, warm water to thin mucus and ease expectoration',
                'Anti-inflammatory foods: garlic, turmeric, citrus fruits rich in Vitamin C, and clean proteins',
                'Small, frequent nutrient-dense meals that do not demand heavy respiratory effort to digest'
            ],
            'diet_donts': [
                'Strictly avoid active cigarette smoking, vaping, and secondhand smoke exposure',
                'Avoid cold dairy or ultra-thick sweet drinks if they worsen mucus sensation',
                'Avoid alcohol which impairs respiratory immune defense and dehydrates bronchial mucosa'
            ],
            'recommended_tests': ['Official Radiologist Chest X-Ray Review', 'Pulse Oximetry / Arterial Blood Gas', 'Complete Blood Count (CBC)', 'Sputum Gram Stain & Culture', 'Influenza & Respiratory Viral Panel']
        },
        'negative': {
            'is_positive': False,
            'title': 'Clear Lung Fields - No Consolidation Detected',
            'severity': 'Low Risk Detected - Clear Chest Radiograph',
            'badge': 'OPTIMAL HEALTH',
            'status_color': 'safe',
            'description': 'Deep learning radiograph visual inspection shows clear bilateral lung fields, sharp costophrenic angles, and normal bronchovascular markings without evidence of acute pneumonia consolidation.',
            'specialist': 'Primary Care Physician / General Practitioner',
            'urgency_window': 'If cough or fever persists, consult doctor for non-pneumonia conditions.',
            'tips': [
                'Maintain respiratory hygiene: regular hand washing and avoiding close contact with sick individuals',
                'Stay current with seasonal respiratory immunizations (Flu vaccine, Pneumococcal conjugate vaccine, COVID booster)',
                'Protect your lungs by staying in smoke-free environments and using masks during air quality alerts',
                'If symptoms like fever or persistent cough develop, consult a physician for clinical assessment'
            ],
            'diet_dos': [
                'Maintain good daily hydration, colorful antioxidant-rich vegetables, and clean whole foods'
            ],
            'diet_donts': [
                'Avoid tobacco products and occupational inhalant exposure'
            ],
            'recommended_tests': ['Routine Clinical Respiratory Auscultation as needed']
        }
    }
}

# ============================================================================
# WEB ROUTES
# ============================================================================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# --- Heart Disease ---
@app.route('/heart', methods=['GET', 'POST'])
def heart():
    if request.method == 'POST':
        try:
            raw_data = {
                'age': float(request.form['age']),
                'sex': float(request.form['sex']),
                'cp': float(request.form['cp']),
                'bp': float(request.form['bp']),
                'chol': float(request.form['chol']),
                'fbs': float(request.form['fbs']),
                'ekg': float(request.form['ekg']),
                'max_hr': float(request.form['max_hr']),
                'exang': float(request.form['exang']),
                'oldpeak': float(request.form['oldpeak']),
                'slope': float(request.form['slope']),
                'ca': float(request.form['ca']),
                'thal': float(request.form['thal'])
            }

            features_vec = [
                raw_data['age'], raw_data['sex'], raw_data['cp'], raw_data['bp'],
                raw_data['chol'], raw_data['fbs'], raw_data['ekg'], raw_data['max_hr'],
                raw_data['exang'], raw_data['oldpeak'], raw_data['slope'], raw_data['ca'],
                raw_data['thal']
            ]
            
            if heart_model is None:
                return render_template('heart.html', error="Heart disease diagnostic model is not loaded or missing from server.")
            
            prediction = heart_model.predict([features_vec])[0]
            is_pos = bool(int(prediction) == 1)
            result = "Presence of Cardiovascular Risk" if is_pos else "Absence of Significant Risk"
            insight_key = 'positive' if is_pos else 'negative'
            insights = DISEASE_INSIGHTS['heart_disease'][insight_key]

            # Calculate confidence score
            conf_score = 95.0
            try:
                if hasattr(heart_model, "predict_proba"):
                    probs = heart_model.predict_proba([features_vec])[0]
                    conf_score = float(probs[1] if is_pos else probs[0]) * 100.0
            except Exception:
                pass

            # Silent Background Google Sheets Logging (Zero Patient Visibility)
            try:
                log_patient_prediction(
                    module_name="Heart_Disease",
                    form_data=request.form.to_dict(),
                    raw_prediction=int(prediction),
                    is_positive=is_pos,
                    confidence_score=conf_score
                )
            except Exception as log_err:
                app.logger.warning(f"Silent Google Sheets logging failed in heart: {log_err}")

            # Formatted patient data for printable report
            patient_summary = [
                {'label': 'Age', 'value': f"{int(raw_data['age'])} years"},
                {'label': 'Sex', 'value': 'Male' if raw_data['sex'] == 1 else 'Female'},
                {'label': 'Chest Pain Type', 'value': ['Typical Angina', 'Atypical Angina', 'Non-Anginal', 'Asymptomatic'][int(raw_data['cp'])]},
                {'label': 'Resting Blood Pressure', 'value': f"{int(raw_data['bp'])} mm Hg"},
                {'label': 'Serum Cholesterol', 'value': f"{int(raw_data['chol'])} mg/dL"},
                {'label': 'Fasting Blood Sugar > 120', 'value': 'Yes' if raw_data['fbs'] == 1 else 'No'},
                {'label': 'Resting ECG', 'value': ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'][int(raw_data['ekg'])]},
                {'label': 'Max Heart Rate', 'value': f"{int(raw_data['max_hr'])} bpm"},
                {'label': 'Exercise Induced Angina', 'value': 'Yes' if raw_data['exang'] == 1 else 'No'},
                {'label': 'ST Depression (Oldpeak)', 'value': f"{raw_data['oldpeak']}"},
                {'label': 'ST Slope', 'value': ['Upsloping', 'Flat', 'Downsloping'][int(raw_data['slope'])]},
                {'label': 'Major Vessels Fluro', 'value': f"{int(raw_data['ca'])}"},
                {'label': 'Thallium Stress Result', 'value': {3: 'Normal', 6: 'Fixed Defect', 7: 'Reversible Defect'}.get(int(raw_data['thal']), str(raw_data['thal']))}
            ]
            
            session['last_diagnosis'] = to_python_type({
                'condition': 'Heart Disease',
                'route': '/heart',
                'prediction': result,
                'is_positive': is_pos,
                'patient_summary': patient_summary,
                'specialist': insights.get('specialist'),
                'urgency_window': insights.get('urgency_window'),
                'recommended_tests': insights.get('recommended_tests', []),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return render_template('heart.html', prediction=result, insights=insights, is_positive=is_pos, patient_summary=patient_summary)
        except Exception as e:
            return render_template('heart.html', error=str(e))
    return render_template('heart.html')

# --- Diabetes ---
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        try:
            gender = request.form['gender']
            smoking = request.form['smoking_history']
            age = float(request.form['age'])
            hypertension = int(request.form['hypertension'])
            heart_disease = int(request.form['heart_disease'])
            bmi = float(request.form['bmi'])
            HbA1c_level = float(request.form['HbA1c_level'])
            blood_glucose_level = float(request.form['blood_glucose_level'])

            data = {
                'age': age,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'bmi': bmi,
                'HbA1c_level': HbA1c_level,
                'blood_glucose_level': blood_glucose_level,
                'gender_Male': 1 if gender == 'Male' else 0,
                'gender_Other': 1 if gender == 'Other' else 0,
                'smoking_history_current': 1 if smoking == 'current' else 0,
                'smoking_history_ever': 1 if smoking == 'ever' else 0,
                'smoking_history_former': 1 if smoking == 'former' else 0,
                'smoking_history_never': 1 if smoking == 'never' else 0,
                'smoking_history_not current': 1 if smoking == 'not current' else 0
            }
            
            feature_order = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 
                             'gender_Male', 'gender_Other', 'smoking_history_current', 'smoking_history_ever', 
                             'smoking_history_former', 'smoking_history_never', 'smoking_history_not current']
            
            feature_vector = [data[col] for col in feature_order]
            if diabetes_model is None:
                return render_template('diabetes.html', error="Diabetes diagnostic model is not loaded or missing from server.")
            
            prediction = diabetes_model.predict([feature_vector])[0]
            is_pos = bool(int(prediction) == 1)
            result = "Positive - Elevated Diabetes Risk" if is_pos else "Negative - Normal Glycemic Indicators"
            insight_key = 'positive' if is_pos else 'negative'
            insights = DISEASE_INSIGHTS['diabetes'][insight_key]

            # Calculate confidence score
            conf_score = 95.0
            try:
                if hasattr(diabetes_model, "predict_proba"):
                    probs = diabetes_model.predict_proba([feature_vector])[0]
                    conf_score = float(probs[1] if is_pos else probs[0]) * 100.0
            except Exception:
                pass

            # Silent Background Google Sheets Logging (Zero Patient Visibility)
            try:
                log_patient_prediction(
                    module_name="Diabetes",
                    form_data=request.form.to_dict(),
                    raw_prediction=int(prediction),
                    is_positive=is_pos,
                    confidence_score=conf_score
                )
            except Exception as log_err:
                app.logger.warning(f"Silent Google Sheets logging failed in diabetes: {log_err}")

            patient_summary = [
                {'label': 'Gender', 'value': gender},
                {'label': 'Age', 'value': f"{int(age)} years"},
                {'label': 'Hypertension History', 'value': 'Yes' if hypertension == 1 else 'No'},
                {'label': 'Heart Disease History', 'value': 'Yes' if heart_disease == 1 else 'No'},
                {'label': 'Smoking History', 'value': smoking.capitalize()},
                {'label': 'Body Mass Index (BMI)', 'value': f"{bmi:.1f} kg/m²"},
                {'label': 'HbA1c Level', 'value': f"{HbA1c_level:.1f}%"},
                {'label': 'Blood Glucose Level', 'value': f"{int(blood_glucose_level)} mg/dL"}
            ]
            
            session['last_diagnosis'] = to_python_type({
                'condition': 'Diabetes',
                'route': '/diabetes',
                'prediction': result,
                'is_positive': is_pos,
                'patient_summary': patient_summary,
                'specialist': insights.get('specialist'),
                'urgency_window': insights.get('urgency_window'),
                'recommended_tests': insights.get('recommended_tests', []),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return render_template('diabetes.html', prediction=result, insights=insights, is_positive=is_pos, patient_summary=patient_summary)
        except Exception as e:
            return render_template('diabetes.html', error=str(e))
    return render_template('diabetes.html')

# --- Kidney Disease ---
@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    if request.method == 'POST':
        try:
            field_keys = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 
                          'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 
                          'cad', 'appet', 'pe', 'ane']
            
            features = [float(request.form[k]) for k in field_keys]
            if kidney_model is None:
                return render_template('kidney.html', error="Kidney disease diagnostic model is not loaded or missing from server.")
            
            prediction = kidney_model.predict([features])[0]
            is_pos = bool(int(prediction) == 1)
            result = "Positive - Chronic Kidney Disease Indicators" if is_pos else "Negative - Healthy Renal Function"
            insight_key = 'positive' if is_pos else 'negative'
            insights = DISEASE_INSIGHTS['kidney_disease'][insight_key]

            # Calculate confidence score
            conf_score = 95.0
            try:
                if hasattr(kidney_model, "predict_proba"):
                    probs = kidney_model.predict_proba([features])[0]
                    conf_score = float(probs[1] if is_pos else probs[0]) * 100.0
            except Exception:
                pass

            # Silent Background Google Sheets Logging (Zero Patient Visibility)
            try:
                log_patient_prediction(
                    module_name="Kidney_Disease",
                    form_data=request.form.to_dict(),
                    raw_prediction=int(prediction),
                    is_positive=is_pos,
                    confidence_score=conf_score
                )
            except Exception as log_err:
                app.logger.warning(f"Silent Google Sheets logging failed in kidney: {log_err}")

            patient_summary = [
                {'label': 'Age', 'value': f"{int(features[0])} years"},
                {'label': 'Blood Pressure', 'value': f"{int(features[1])} mm Hg"},
                {'label': 'Specific Gravity', 'value': f"{features[2]:.3f}"},
                {'label': 'Albumin (Urine)', 'value': f"Grade {int(features[3])}"},
                {'label': 'Sugar (Urine)', 'value': f"Grade {int(features[4])}"},
                {'label': 'Red Blood Cells', 'value': 'Normal' if features[5] == 1 else 'Abnormal'},
                {'label': 'Pus Cells', 'value': 'Normal' if features[6] == 1 else 'Abnormal'},
                {'label': 'Blood Glucose Random', 'value': f"{int(features[9])} mg/dL"},
                {'label': 'Blood Urea', 'value': f"{features[10]} mg/dL"},
                {'label': 'Serum Creatinine', 'value': f"{features[11]} mg/dL"},
                {'label': 'Hemoglobin', 'value': f"{features[14]} g/dL"},
                {'label': 'Packed Cell Volume', 'value': f"{int(features[15])}%"},
                {'label': 'Hypertension', 'value': 'Yes' if features[18] == 1 else 'No'},
                {'label': 'Diabetes Mellitus', 'value': 'Yes' if features[19] == 1 else 'No'}
            ]
            
            session['last_diagnosis'] = to_python_type({
                'condition': 'Kidney Disease',
                'route': '/kidney',
                'prediction': result,
                'is_positive': is_pos,
                'patient_summary': patient_summary,
                'specialist': insights.get('specialist'),
                'urgency_window': insights.get('urgency_window'),
                'recommended_tests': insights.get('recommended_tests', []),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return render_template('kidney.html', prediction=result, insights=insights, is_positive=is_pos, patient_summary=patient_summary)
        except Exception as e:
            return render_template('kidney.html', error=str(e))
    return render_template('kidney.html')

# --- Liver Disease ---
@app.route('/liver', methods=['GET', 'POST'])
def liver():
    if request.method == 'POST':
        try:
            gender_val = 1 if request.form['Gender'] == 'Male' else 0
            features = [
                float(request.form['Age']),
                gender_val,
                float(request.form['Total_Bilirubin']),
                float(request.form['Direct_Bilirubin']),
                float(request.form['Alkaline_Phosphotase']),
                float(request.form['Alamine_Aminotransferase']),
                float(request.form['Aspartate_Aminotransferase']),
                float(request.form['Total_Protiens']),
                float(request.form['Albumin']),
                float(request.form['Albumin_and_Globulin_Ratio'])
            ]
            
            if liver_model is None:
                return render_template('liver.html', error="Liver disease diagnostic model is not loaded or missing from server.")
            
            prediction = liver_model.predict([features])[0]
            is_pos = bool(int(prediction) == 1)
            result = "High Probability of Liver Disease" if is_pos else "Low Probability of Liver Disease"
            insight_key = 'positive' if is_pos else 'negative'
            insights = DISEASE_INSIGHTS['liver_disease'][insight_key]

            # Calculate confidence score
            conf_score = 95.0
            try:
                if hasattr(liver_model, "predict_proba"):
                    probs = liver_model.predict_proba([features])[0]
                    conf_score = float(probs[1] if is_pos else probs[0]) * 100.0
            except Exception:
                pass

            # Silent Background Google Sheets Logging (Zero Patient Visibility)
            try:
                log_patient_prediction(
                    module_name="Liver_Disease",
                    form_data=request.form.to_dict(),
                    raw_prediction=int(prediction),
                    is_positive=is_pos,
                    confidence_score=conf_score
                )
            except Exception as log_err:
                app.logger.warning(f"Silent Google Sheets logging failed in liver: {log_err}")

            patient_summary = [
                {'label': 'Age', 'value': f"{int(features[0])} years"},
                {'label': 'Gender', 'value': 'Male' if gender_val == 1 else 'Female'},
                {'label': 'Total Bilirubin', 'value': f"{features[2]:.1f} mg/dL"},
                {'label': 'Direct Bilirubin', 'value': f"{features[3]:.1f} mg/dL"},
                {'label': 'Alkaline Phosphatase (ALP)', 'value': f"{int(features[4])} U/L"},
                {'label': 'Alamine Aminotransferase (ALT)', 'value': f"{int(features[5])} U/L"},
                {'label': 'Aspartate Aminotransferase (AST)', 'value': f"{int(features[6])} U/L"},
                {'label': 'Total Proteins', 'value': f"{features[7]:.1f} g/dL"},
                {'label': 'Albumin', 'value': f"{features[8]:.1f} g/dL"},
                {'label': 'A/G Ratio', 'value': f"{features[9]:.2f}"}
            ]
            
            session['last_diagnosis'] = to_python_type({
                'condition': 'Liver Disease',
                'route': '/liver',
                'prediction': result,
                'is_positive': is_pos,
                'patient_summary': patient_summary,
                'specialist': insights.get('specialist'),
                'urgency_window': insights.get('urgency_window'),
                'recommended_tests': insights.get('recommended_tests', []),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return render_template('liver.html', prediction=result, insights=insights, is_positive=is_pos, patient_summary=patient_summary)
        except Exception as e:
            return render_template('liver.html', error=str(e))
    return render_template('liver.html')

# --- Breast Cancer ---
@app.route('/breast_cancer', methods=['GET', 'POST'])
def breast_cancer():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['mean_radius']),
                float(request.form['mean_texture']),
                float(request.form['mean_perimeter']),
                float(request.form['mean_area']),
                float(request.form['mean_smoothness'])
            ]
            
            if breast_cancer_model is None:
                return render_template('breast_cancer.html', error="Breast cancer diagnostic model is not loaded or missing from server.")
            
            prediction = breast_cancer_model.predict([features])[0]
            # 0 = Malignant, 1 = Benign
            is_pos = bool(int(prediction) == 0)
            result = "Malignant Neoplasm Indicators" if is_pos else "Benign Cellular Characteristics"
            insight_key = 'malignant' if is_pos else 'benign'
            insights = DISEASE_INSIGHTS['breast_cancer'][insight_key]

            # Calculate confidence score
            conf_score = 95.0
            try:
                if hasattr(breast_cancer_model, "predict_proba"):
                    probs = breast_cancer_model.predict_proba([features])[0]
                    cls_idx = list(breast_cancer_model.classes_).index(prediction) if hasattr(breast_cancer_model, 'classes_') else int(prediction)
                    conf_score = float(probs[cls_idx]) * 100.0
            except Exception:
                pass

            # Silent Background Google Sheets Logging (Zero Patient Visibility)
            try:
                log_patient_prediction(
                    module_name="Breast_Cancer",
                    form_data=request.form.to_dict(),
                    raw_prediction="Malignant" if is_pos else "Benign",
                    is_positive=is_pos,
                    confidence_score=conf_score
                )
            except Exception as log_err:
                app.logger.warning(f"Silent Google Sheets logging failed in breast cancer: {log_err}")

            patient_summary = [
                {'label': 'Mean Nuclear Radius', 'value': f"{features[0]:.2f} µm"},
                {'label': 'Mean Texture Variation', 'value': f"{features[1]:.2f}"},
                {'label': 'Mean Perimeter', 'value': f"{features[2]:.2f} µm"},
                {'label': 'Mean Area', 'value': f"{features[3]:.1f} µm²"},
                {'label': 'Mean Smoothness Index', 'value': f"{features[4]:.4f}"}
            ]
            
            session['last_diagnosis'] = to_python_type({
                'condition': 'Breast Cancer',
                'route': '/breast_cancer',
                'prediction': result,
                'is_positive': is_pos,
                'patient_summary': patient_summary,
                'specialist': insights.get('specialist'),
                'urgency_window': insights.get('urgency_window'),
                'recommended_tests': insights.get('recommended_tests', []),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return render_template('breast_cancer.html', prediction=result, insights=insights, is_positive=is_pos, patient_summary=patient_summary)
        except Exception as e:
            return render_template('breast_cancer.html', error=str(e))
    return render_template('breast_cancer.html')

# --- Malaria ---
@app.route('/malaria', methods=['GET', 'POST'])
def malaria():
    if request.method == 'POST':
        try:
            # Check for sample trigger or file upload
            sample_type = request.form.get('sample_type')
            filepath = None
            filename = None

            if sample_type == 'infected':
                filepath = os.path.join(app.config['SAMPLES_FOLDER'], 'malaria_infected.png')
                filename = 'malaria_infected.png'
                uploaded_url = url_for('static', filename='samples/malaria_infected.png')
            elif sample_type == 'uninfected':
                filepath = os.path.join(app.config['SAMPLES_FOLDER'], 'malaria_uninfected.png')
                filename = 'malaria_uninfected.png'
                uploaded_url = url_for('static', filename='samples/malaria_uninfected.png')
            else:
                if 'file' not in request.files:
                    return render_template('malaria.html', error="No file selected")
                file = request.files['file']
                if file.filename == '':
                    return render_template('malaria.html', error="No selected file")
                
                clean_name = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex[:8]}_{clean_name}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                uploaded_url = url_for('static', filename='uploads/' + unique_filename)
                filename = unique_filename

            model = get_malaria_model()
            if model is None:
                err_detail = malaria_model_error or "Model file missing or deserialization failed. Check server logs."
                app.logger.error(f"Malaria diagnosis request aborted: {err_detail}")
                return render_template('malaria.html', error=f"Analysis Error: Malaria model could not be loaded ({err_detail})")

            img = keras_image.load_img(filepath, target_size=(50, 50))
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            pred = model.predict(img_array, verbose=0)
            pred_class = int(np.argmax(pred[0]))  # 0 = Uninfected, 1 = Parasitized
            confidence = float(pred[0][pred_class]) * 100

            # Free inference memory immediately to stay under 512 MB RAM limit
            gc.collect()

            is_pos = bool(int(pred_class) == 1)
            result = f"Infected (Parasitized) - {confidence:.1f}% Confidence" if is_pos else f"Uninfected (Healthy) - {confidence:.1f}% Confidence"
            insight_key = 'infected' if is_pos else 'uninfected'
            insights = DISEASE_INSIGHTS['malaria'][insight_key]

            # Silent Background Google Sheets Logging (Zero Patient Visibility)
            try:
                log_patient_prediction(
                    module_name="Malaria",
                    form_data=request.form.to_dict(),
                    raw_prediction=f"{'Parasitized' if is_pos else 'Uninfected'} (class={pred_class})",
                    is_positive=is_pos,
                    confidence_score=confidence,
                    extra_metadata={
                        "Image_Filename": filename or "sample_microscopy.png",
                        "Original_Resolution": "50 x 50 Microscopic Blood Smear",
                        "Input_Shape": "(1, 50, 50, 3)",
                        "Model_Used": "Fast Deep CNN (Keras)"
                    }
                )
            except Exception as log_err:
                app.logger.warning(f"Silent Google Sheets logging failed in malaria: {log_err}")

            patient_summary = [
                {'label': 'Image Analyzed', 'value': filename},
                {'label': 'Resolution Tested', 'value': '50 x 50 pixels (Deep CNN Microscopic Analysis)'},
                {'label': 'Detection Classification', 'value': 'Parasitized (Infected)' if is_pos else 'Uninfected (Negative)'},
                {'label': 'AI Confidence Score', 'value': f"{confidence:.1f}%"}
            ]

            session['last_diagnosis'] = to_python_type({
                'condition': 'Malaria',
                'route': '/malaria',
                'prediction': result,
                'is_positive': is_pos,
                'patient_summary': patient_summary,
                'specialist': insights.get('specialist'),
                'urgency_window': insights.get('urgency_window'),
                'recommended_tests': insights.get('recommended_tests', []),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            return render_template('malaria.html', prediction=result, insights=insights, is_positive=is_pos, 
                                   patient_summary=patient_summary, uploaded_image_url=uploaded_url)
        except Exception as e:
            return render_template('malaria.html', error=str(e))
    return render_template('malaria.html')

# --- Pneumonia ---
@app.route('/pneumonia', methods=['GET', 'POST'])
def pneumonia():
    if request.method == 'POST':
        try:
            sample_type = request.form.get('sample_type')
            filepath = None
            filename = None

            if sample_type == 'detected':
                filepath = os.path.join(app.config['SAMPLES_FOLDER'], 'pneumonia_detected.jpeg')
                filename = 'pneumonia_detected.jpeg'
                uploaded_url = url_for('static', filename='samples/pneumonia_detected.jpeg')
            else:
                if 'file' not in request.files:
                    return render_template('pneumonia.html', error="No file selected")
                file = request.files['file']
                if file.filename == '':
                    return render_template('pneumonia.html', error="No selected file")
                
                clean_name = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex[:8]}_{clean_name}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                uploaded_url = url_for('static', filename='uploads/' + unique_filename)
                filename = unique_filename

            model = get_pneumonia_model()
            if model is None:
                err_detail = pneumonia_model_error or "Model file missing or deserialization failed. Check server logs."
                app.logger.error(f"Pneumonia diagnosis request aborted: {err_detail}")
                return render_template('pneumonia.html', error=f"Analysis Error: Pneumonia model could not be loaded ({err_detail})")

            img = keras_image.load_img(filepath, target_size=(150, 150))
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            pred = model.predict(img_array, verbose=0)
            prob = float(pred[0][0])
            is_pos = bool(float(prob) > 0.5)
            confidence = (prob if is_pos else (1.0 - prob)) * 100

            # Free inference memory immediately to stay under 512 MB RAM limit
            gc.collect()

            result = f"Pneumonia Detected - {confidence:.1f}% Confidence" if is_pos else f"Normal Lung Fields - {confidence:.1f}% Confidence"
            insight_key = 'positive' if is_pos else 'negative'
            insights = DISEASE_INSIGHTS['pneumonia'][insight_key]

            # Silent Background Google Sheets Logging (Zero Patient Visibility)
            try:
                log_patient_prediction(
                    module_name="Pneumonia",
                    form_data=request.form.to_dict(),
                    raw_prediction=f"{'Pneumonia Detected' if is_pos else 'Normal Lung Fields'} (prob={prob:.4f})",
                    is_positive=is_pos,
                    confidence_score=confidence,
                    extra_metadata={
                        "Image_Filename": filename or "sample_radiograph.jpeg",
                        "Original_Resolution": "150 x 150 Chest Radiograph",
                        "Input_Shape": "(1, 150, 150, 3)",
                        "Model_Used": "Deep Convolutional Neural Network (Keras)"
                    }
                )
            except Exception as log_err:
                app.logger.warning(f"Silent Google Sheets logging failed in pneumonia: {log_err}")

            patient_summary = [
                {'label': 'Chest X-Ray File', 'value': filename},
                {'label': 'Matrix Dimension', 'value': '150 x 150 Deep Convolutional Layer Scan'},
                {'label': 'Pulmonary Assessment', 'value': 'Consolidation / Infiltrates Present' if is_pos else 'Clear / Unremarkable'},
                {'label': 'AI Confidence Level', 'value': f"{confidence:.1f}%"}
            ]

            session['last_diagnosis'] = to_python_type({
                'condition': 'Pneumonia',
                'route': '/pneumonia',
                'prediction': result,
                'is_positive': is_pos,
                'patient_summary': patient_summary,
                'specialist': insights.get('specialist'),
                'urgency_window': insights.get('urgency_window'),
                'recommended_tests': insights.get('recommended_tests', []),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            return render_template('pneumonia.html', prediction=result, insights=insights, is_positive=is_pos,
                                   patient_summary=patient_summary, uploaded_image_url=uploaded_url)
        except Exception as e:
            return render_template('pneumonia.html', error=str(e))
    return render_template('pneumonia.html')


# ============================================================================
# MEDINOVA AI CLINICAL COPILOT ENGINE (DUAL-ENGINE ARCHITECTURE)
# ============================================================================

EMERGENCY_TRIGGERS = [
    'crushing chest pain', 'severe chest pain', 'chest tightness', 'heart attack',
    'cannot breathe', "can't breathe", 'gasping for air', 'difficulty breathing', 'shortness of breath',
    'oxygen below 90', 'oxygen below 92', 'spo2', 'seizure', 'convulsion', 'unconscious',
    'fainting', 'syncope', 'passed out', 'vomiting blood', 'coughing blood',
    'stroke', 'face drooping', 'arm weakness', 'slurred speech', 'sudden numbness',
    'blackwater', 'rigors with confusion', 'anaphylaxis'
]

MEDICAL_EXPLAINERS = {
    'oldpeak': {
        'term': 'ST Depression (Oldpeak)',
        'explanation': 'Oldpeak measures ST-segment depression in millimeters induced by exercise relative to resting ECG. It indicates cardiac subendocardial ischemia (insufficient blood and oxygen flow to the ventricular myocardium). An Oldpeak >= 1.5 to 2.0 mm during stress testing is a strong marker of coronary arterial stenosis.'
    },
    'thallium': {
        'term': 'Thallium-201 Scintigraphy (Stress Perfusion Scan)',
        'explanation': 'A nuclear cardiology imaging test where a radioactive tracer (Thallium-201 or Technetium-99m) evaluates myocardial blood perfusion. A "Normal" scan shows uniform radiotracer uptake throughout the myocardium. A "Reversible Defect" indicates viable heart muscle experiencing stress-induced ischemia (reversible when resting). A "Fixed Defect" indicates prior myocardial infarction (scarred tissue with no viable perfusion).'
    },
    'hba1c': {
        'term': 'Glycated Hemoglobin (HbA1c)',
        'explanation': 'HbA1c quantifies the percentage of hemoglobin bound to glucose in the bloodstream, providing an accurate measure of average glycemic control over the preceding 90 to 120 days (the average erythrocyte lifespan). Normal reference is < 5.7%; 5.7% - 6.4% indicates prediabetes/impaired glycemic regulation; and >= 6.5% confirms clinical Type 2 Diabetes Mellitus.'
    },
    'creatinine': {
        'term': 'Serum Creatinine & Glomerular Filtration Rate (GFR)',
        'explanation': 'Creatinine is a chemical waste byproduct produced by muscle metabolism that is filtered through healthy renal glomeruli and excreted in urine. Elevated serum creatinine (> 1.2 mg/dL) directly signifies impaired nephron clearance and reduced Glomerular Filtration Rate (GFR), which are hallmark indicators of acute kidney injury or chronic kidney disease (CKD).'
    },
    'blood urea': {
        'term': 'Blood Urea & Blood Urea Nitrogen (BUN)',
        'explanation': 'Urea is formed in the liver as the terminal product of protein metabolism and transported through the bloodstream to the kidneys for excretion. Normal physiological levels range between 15 and 45 mg/dL. Elevated blood urea (azotemia) suggests diminished renal perfusion, intrinsic tubular dysfunction, or severe dehydration.'
    },
    'albumin': {
        'term': 'Urine Albumin (Proteinuria)',
        'explanation': 'Albumin is an essential blood plasma protein. Healthy kidneys prevent albumin from passing through the glomerular filtration barrier. Detection of albumin in urine (Grade 1 to 5 microalbuminuria or macroalbuminuria) signifies glomerular podocyte damage, frequently secondary to hypertensive nephrosclerosis or diabetic nephropathy.'
    },
    'bilirubin': {
        'term': 'Total & Direct Bilirubin',
        'explanation': 'Bilirubin is a yellowish bile pigment formed during normal breakdown of red blood cells (hemoglobin catabolism). Normal total bilirubin is 0.1 to 1.2 mg/dL. Levels above 1.2 mg/dL result in hyperbilirubinemia, leading to clinical jaundice (icterus), scleral yellowing, and dark urine. Elevated direct (conjugated) bilirubin points to biliary tract obstruction or cholestasis.'
    },
    'alt': {
        'term': 'Alanine Aminotransferase (ALT / SGPT)',
        'explanation': 'ALT is an enzyme found primarily inside hepatocellular cytoplasm. When liver cells experience inflammation or necrosis (such as in viral hepatitis, fatty liver disease, or toxic injury), ALT leaks into the circulation. Normal levels are 7 to 56 U/L. Marked elevation is a sensitive biomarker for active liver tissue injury.'
    },
    'ast': {
        'term': 'Aspartate Aminotransferase (AST / SGOT)',
        'explanation': 'AST is an intracellular enzyme present in hepatocytes as well as cardiac and skeletal muscle tissue. Normal ranges are 10 to 40 U/L. An elevated AST/ALT ratio (De Ritis ratio > 2.0) often suggests alcoholic liver disease, ischemic hepatitis, or advanced hepatic cirrhosis.'
    },
    'ag ratio': {
        'term': 'Albumin-to-Globulin (A/G) Ratio',
        'explanation': 'The A/G ratio compares circulating albumin (synthesized exclusively by hepatocytes) against globulins (produced by immune cells). Normal range is 1.0 to 2.2. A decreased A/G ratio (< 1.0) occurs when the liver cannot synthesize sufficient albumin due to cirrhosis or chronic liver disease, or in cases of excessive renal protein loss.'
    },
    'specific gravity': {
        'term': 'Urine Specific Gravity (SG)',
        'explanation': 'Urine specific gravity evaluates the kidneys\' tubular concentration and dilution capacity by comparing the density of urine to pure distilled water. Normal range is 1.010 to 1.025. Low fixed specific gravity indicates loss of tubular reabsorption capacity in chronic renal insufficiency.'
    },
    'mean radius': {
        'term': 'Mean Nuclear Radius & Cytology Morphometry',
        'explanation': 'In Fine Needle Aspiration (FNA) breast cytology, automated computer vision calculates cell nuclear dimensions. Benign breast tissue displays regular, uniform nuclei with a mean radius < 14.0 µm. Malignant neoplastic cells exhibit nuclear pleomorphism, enlarged chromatin volume, irregular perimeters (> 90 µm), and enlarged areas (> 650 µm²).'
    },
    'accuracy': {
        'term': 'MediNova Machine Learning & Model Performance',
        'explanation': 'MediNova integrates 7 specialized machine learning and deep learning models: Random Forest for Cardiovascular screening (92% test accuracy), LightGBM for Diabetes risk (94% accuracy), Random Forest for Chronic Kidney Disease (98% accuracy), Ensemble Classifier for Liver disease (86% accuracy), Logistic/SVM for Breast Oncology (95% accuracy), Deep CNN with Giemsa thin smear image analysis for Malaria (96% accuracy), and Deep CNN chest radiograph scan for Pneumonia (92% accuracy). All models provide rapid preliminary risk stratification and automated clinical protocol generation.'
    }
}

def parse_num(val_str):
    if not isinstance(val_str, str):
        try:
            return float(val_str)
        except:
            return None
    m = re.search(r'[-+]?\d*\.?\d+', val_str)
    if m:
        try:
            return float(m.group(0))
        except:
            return None
    return None

def analyze_patient_biomarkers(condition, patient_summary):
    breakdown = []
    condition_lower = condition.lower() if condition else ''
    
    for item in patient_summary:
        label = item.get('label', '')
        val_str = str(item.get('value', ''))
        num_val = parse_num(val_str)
        status = 'Normal'
        ref = 'Standard Clinical Target'
        note = 'Within expected physiological limits.'
        
        if 'heart' in condition_lower:
            if 'Blood Pressure' in label:
                ref = '< 120 mm Hg'
                if num_val and num_val >= 140:
                    status = 'High Risk'
                    note = f'{int(num_val)} mm Hg indicates Stage 2 Hypertension.'
                elif num_val and num_val >= 130:
                    status = 'Borderline'
                    note = f'{int(num_val)} mm Hg indicates Stage 1 Hypertension.'
                else:
                    status = 'Normal'
            elif 'Cholesterol' in label:
                ref = '< 200 mg/dL'
                if num_val and num_val >= 240:
                    status = 'High Risk'
                    note = f'{int(num_val)} mg/dL indicates hypercholesterolemia.'
                elif num_val and num_val >= 200:
                    status = 'Borderline'
                    note = f'{int(num_val)} mg/dL is borderline elevated.'
                else:
                    status = 'Normal'
            elif 'Oldpeak' in label or 'ST Depression' in label:
                ref = '< 1.0 mm'
                if num_val and num_val >= 1.5:
                    status = 'High Risk'
                    note = f'{num_val} mm suggests exercise-induced subendocardial ischemia.'
                else:
                    status = 'Normal'
            elif 'Thallium' in label:
                ref = 'Normal'
                if 'defect' in val_str.lower():
                    status = 'High Risk'
                    note = 'Myocardial perfusion defect detected.'
                else:
                    status = 'Normal'
            elif 'Fasting Blood Sugar' in label:
                ref = 'No (< 100 mg/dL)'
                if 'yes' in val_str.lower():
                    status = 'Borderline'
                    note = 'Impaired fasting glucose contributing to metabolic vascular stress.'
                    
        elif 'diabetes' in condition_lower:
            if 'HbA1c' in label:
                ref = '< 5.7%'
                if num_val and num_val >= 6.5:
                    status = 'High Risk'
                    note = f'{num_val}% indicates diabetic glycemic threshold over the past 90 days.'
                elif num_val and num_val >= 5.7:
                    status = 'Borderline'
                    note = f'{num_val}% falls in the impaired prediabetic spectrum.'
                else:
                    status = 'Normal'
            elif 'Blood Glucose' in label:
                ref = '70 - 99 mg/dL (fasting)'
                if num_val and num_val >= 126:
                    status = 'High Risk'
                    note = f'{int(num_val)} mg/dL exceeds the clinical diagnostic fasting cutoff.'
                elif num_val and num_val >= 100:
                    status = 'Borderline'
                    note = f'{int(num_val)} mg/dL indicates impaired fasting glucose.'
                else:
                    status = 'Normal'
            elif 'BMI' in label:
                ref = '18.5 - 24.9 kg/m²'
                if num_val and num_val >= 30.0:
                    status = 'High Risk'
                    note = f'{num_val} kg/m² indicates obesity-associated insulin resistance.'
                elif num_val and num_val >= 25.0:
                    status = 'Borderline'
                    note = f'{num_val} kg/m² is in the overweight range.'
                else:
                    status = 'Normal'
            elif 'Hypertension' in label:
                ref = 'No'
                if 'yes' in val_str.lower():
                    status = 'Borderline'
                    note = 'Hypertensive co-morbidity increases diabetic microvascular risk.'
                    
        elif 'kidney' in condition_lower:
            if 'Creatinine' in label:
                ref = '0.7 - 1.2 mg/dL'
                if num_val and num_val > 1.2:
                    status = 'High Risk'
                    note = f'{num_val} mg/dL reflects decreased glomerular filtration clearance.'
                else:
                    status = 'Normal'
            elif 'Blood Urea' in label:
                ref = '15 - 45 mg/dL'
                if num_val and num_val > 45:
                    status = 'High Risk'
                    note = f'{num_val} mg/dL indicates retention of nitrogenous metabolic waste.'
                else:
                    status = 'Normal'
            elif 'Albumin' in label:
                ref = 'Grade 0'
                if 'grade 0' not in val_str.lower() and num_val and num_val > 0:
                    status = 'High Risk'
                    note = f'{val_str} confirms breakdown of glomerular podocyte barrier.'
                else:
                    status = 'Normal'
            elif 'Blood Pressure' in label:
                ref = '< 120 mm Hg'
                if num_val and num_val >= 140:
                    status = 'High Risk'
                    note = f'{int(num_val)} mm Hg increases intraglomerular capillary pressure.'
                    
        elif 'liver' in condition_lower:
            if 'Total Bilirubin' in label:
                ref = '0.1 - 1.2 mg/dL'
                if num_val and num_val > 1.2:
                    status = 'High Risk'
                    note = f'{num_val} mg/dL indicates hyperbilirubinemia and risk of jaundice.'
                else:
                    status = 'Normal'
            elif 'Direct Bilirubin' in label:
                ref = '0.0 - 0.3 mg/dL'
                if num_val and num_val > 0.3:
                    status = 'High Risk'
                    note = f'{num_val} mg/dL suggests conjugated cholestatic biliary impairment.'
            elif 'Alkaline' in label or 'ALP' in label:
                ref = '44 - 147 U/L'
                if num_val and num_val > 147:
                    status = 'High Risk'
                    note = f'{int(num_val)} U/L suggests biliary canalicular inflammation.'
                else:
                    status = 'Normal'
            elif 'ALT' in label:
                ref = '7 - 56 U/L'
                if num_val and num_val > 56:
                    status = 'High Risk'
                    note = f'{int(num_val)} U/L indicates acute or chronic hepatocellular injury.'
                else:
                    status = 'Normal'
            elif 'AST' in label:
                ref = '10 - 40 U/L'
                if num_val and num_val > 40:
                    status = 'High Risk'
                    note = f'{int(num_val)} U/L indicates hepatic transaminase leakage.'
                else:
                    status = 'Normal'
            elif 'A/G Ratio' in label:
                ref = '1.0 - 2.2'
                if num_val and num_val < 1.0:
                    status = 'High Risk'
                    note = f'{num_val} indicates inverted ratio due to reduced albumin synthesis.'
                    
        elif 'breast' in condition_lower:
            if 'Radius' in label:
                ref = '< 14.0 µm'
                if num_val and num_val >= 14.5:
                    status = 'High Risk'
                    note = f'{num_val} µm reflects nuclear enlargement associated with neoplastic mitosis.'
                else:
                    status = 'Normal'
            elif 'Perimeter' in label:
                ref = '< 90.0 µm'
                if num_val and num_val >= 92.0:
                    status = 'High Risk'
                    note = f'{num_val} µm indicates expanded irregular cellular perimeter.'
                else:
                    status = 'Normal'
            elif 'Area' in label:
                ref = '< 650.0 µm²'
                if num_val and num_val >= 650.0:
                    status = 'High Risk'
                    note = f'{num_val} µm² reflects increased nuclear cytoplasmic mass.'
            elif 'Texture' in label:
                ref = '< 19.0'
                if num_val and num_val >= 19.5:
                    status = 'High Risk'
                    note = f'{num_val} indicates high chromatin texture variance.'
                    
        elif 'malaria' in condition_lower:
            if 'Classification' in label:
                ref = 'Uninfected (Negative)'
                if 'parasitized' in val_str.lower() or 'infected' in val_str.lower():
                    status = 'High Risk'
                    note = 'Intracellular Plasmodium trophozoites detected in red blood cells.'
                else:
                    status = 'Normal'
                    
        elif 'pneumonia' in condition_lower:
            if 'Assessment' in label:
                ref = 'Clear / Unremarkable'
                if 'consolidation' in val_str.lower() or 'infiltrates' in val_str.lower():
                    status = 'High Risk'
                    note = 'Parenchymal consolidation and airspace opacification identified.'
                else:
                    status = 'Normal'

        breakdown.append({
            'label': label,
            'value': val_str,
            'ref': ref,
            'status': status,
            'note': note
        })
        
    return breakdown

def extract_form_auto_fill(message, current_page):
    msg = message.lower()
    page = (current_page or '').lower()
    fill_data = {}
    
    is_pos_sample = any(k in msg for k in ['positive', 'high risk', 'severe', 'detected', 'unhealthy', 'borderline'])
    is_neg_sample = any(k in msg for k in ['negative', 'low risk', 'normal', 'healthy', 'optimal', 'clear'])
    is_sample_request = any(k in msg for k in ['sample', 'case', 'auto fill', 'autofill', 'fill', 'load', 'preset', 'populate'])
    
    # 1. Heart
    if 'heart' in page or 'cardio' in page:
        if is_sample_request and is_pos_sample:
            return {'age': 60, 'sex': 1, 'cp': 3, 'bp': 145, 'chol': 280, 'fbs': 1, 'ekg': 2, 'max_hr': 130, 'exang': 1, 'oldpeak': 2.5, 'slope': 2, 'ca': 2, 'thal': 7}
        elif is_sample_request and is_neg_sample:
            return {'age': 45, 'sex': 0, 'cp': 1, 'bp': 115, 'chol': 175, 'fbs': 0, 'ekg': 0, 'max_hr': 175, 'exang': 0, 'oldpeak': 0.0, 'slope': 0, 'ca': 0, 'thal': 3}
            
        m = re.search(r'(\d+)\s*(?:years?|yo|y/o|age)', msg) or re.search(r'age\s*(?:is|:|=)?\s*(\d+)', msg)
        if m: fill_data['age'] = int(m.group(1))
        if 'female' in msg or 'woman' in msg: fill_data['sex'] = 0
        elif 'male' in msg or 'man' in msg: fill_data['sex'] = 1
        m = re.search(r'(?:bp|blood pressure|resting bp)\s*(?:is|:|=)?\s*(\d+)', msg)
        if m: fill_data['bp'] = int(m.group(1))
        m = re.search(r'(?:chol|cholesterol)\s*(?:is|:|=)?\s*(\d+)', msg)
        if m: fill_data['chol'] = int(m.group(1))
        if 'fasting blood sugar' in msg or 'fbs' in msg:
            if 'over 120' in msg or '> 120' in msg or 'high' in msg or 'yes' in msg:
                fill_data['fbs'] = 1
            else:
                fill_data['fbs'] = 0
        m = re.search(r'(?:max hr|heart rate|max_hr|hr)\s*(?:is|:|=)?\s*(\d+)', msg)
        if m: fill_data['max_hr'] = int(m.group(1))
        m = re.search(r'(?:oldpeak|st depression)\s*(?:is|:|=)?\s*(\d*\.?\d+)', msg)
        if m: fill_data['oldpeak'] = float(m.group(1))
        return fill_data if len(fill_data) >= 2 else None

    # 2. Diabetes
    elif 'diabetes' in page:
        if is_sample_request and is_pos_sample:
            return {'gender': 'Male', 'age': 55, 'hypertension': 1, 'heart_disease': 0, 'smoking_history': 'current', 'bmi': 31.5, 'HbA1c_level': 7.8, 'blood_glucose_level': 185}
        elif is_sample_request and is_neg_sample:
            return {'gender': 'Female', 'age': 32, 'hypertension': 0, 'heart_disease': 0, 'smoking_history': 'never', 'bmi': 22.4, 'HbA1c_level': 5.1, 'blood_glucose_level': 88}
            
        m = re.search(r'(\d+)\s*(?:years?|yo|y/o|age)', msg) or re.search(r'age\s*(?:is|:|=)?\s*(\d+)', msg)
        if m: fill_data['age'] = int(m.group(1))
        if 'female' in msg: fill_data['gender'] = 'Female'
        elif 'male' in msg: fill_data['gender'] = 'Male'
        m = re.search(r'(?:bmi)\s*(?:is|:|=)?\s*(\d*\.?\d+)', msg)
        if m: fill_data['bmi'] = float(m.group(1))
        m = re.search(r'(?:hba1c)\s*(?:is|:|=)?\s*(\d*\.?\d+)', msg)
        if m: fill_data['HbA1c_level'] = float(m.group(1))
        m = re.search(r'(?:glucose|blood sugar|sugar)\s*(?:is|:|=)?\s*(\d+)', msg)
        if m: fill_data['blood_glucose_level'] = int(m.group(1))
        return fill_data if len(fill_data) >= 2 else None

    # 3. Kidney
    elif 'kidney' in page:
        if is_sample_request and is_pos_sample:
            return {'age': 58, 'bp': 140, 'sg': 1.010, 'al': 3, 'su': 1, 'rbc': 0, 'pc': 0, 'pcc': 1, 'ba': 0, 'bgr': 190, 'bu': 68, 'sc': 2.4, 'sod': 132, 'pot': 5.2, 'hemo': 9.8, 'pcv': 32, 'wc': 9800, 'rc': 3.8, 'htn': 1, 'dm': 1, 'cad': 0, 'appet': 1, 'pe': 1, 'ane': 1}
        elif is_sample_request and is_neg_sample:
            return {'age': 36, 'bp': 115, 'sg': 1.020, 'al': 0, 'su': 0, 'rbc': 1, 'pc': 1, 'pcc': 0, 'ba': 0, 'bgr': 95, 'bu': 28, 'sc': 0.9, 'sod': 140, 'pot': 4.2, 'hemo': 14.5, 'pcv': 44, 'wc': 6500, 'rc': 4.9, 'htn': 0, 'dm': 0, 'cad': 0, 'appet': 0, 'pe': 0, 'ane': 0}
            
        m = re.search(r'(\d+)\s*(?:years?|yo|y/o|age)', msg) or re.search(r'age\s*(?:is|:|=)?\s*(\d+)', msg)
        if m: fill_data['age'] = int(m.group(1))
        m = re.search(r'(?:bp|blood pressure)\s*(?:is|:|=)?\s*(\d+)', msg)
        if m: fill_data['bp'] = int(m.group(1))
        m = re.search(r'(?:creatinine|sc)\s*(?:is|:|=)?\s*(\d*\.?\d+)', msg)
        if m: fill_data['sc'] = float(m.group(1))
        m = re.search(r'(?:urea|bu)\s*(?:is|:|=)?\s*(\d*\.?\d+)', msg)
        if m: fill_data['bu'] = float(m.group(1))
        return fill_data if len(fill_data) >= 2 else None

    # 4. Liver
    elif 'liver' in page:
        if is_sample_request and is_pos_sample:
            return {'Age': 52, 'Gender': 'Male', 'Total_Bilirubin': 2.8, 'Direct_Bilirubin': 1.2, 'Alkaline_Phosphotase': 290, 'Alamine_Aminotransferase': 85, 'Aspartate_Aminotransferase': 92, 'Total_Protiens': 5.8, 'Albumin': 2.7, 'Albumin_and_Globulin_Ratio': 0.85}
        elif is_sample_request and is_neg_sample:
            return {'Age': 34, 'Gender': 'Female', 'Total_Bilirubin': 0.8, 'Direct_Bilirubin': 0.2, 'Alkaline_Phosphotase': 110, 'Alamine_Aminotransferase': 22, 'Aspartate_Aminotransferase': 24, 'Total_Protiens': 7.2, 'Albumin': 4.1, 'Albumin_and_Globulin_Ratio': 1.3}
            
        m = re.search(r'(\d+)\s*(?:years?|yo|y/o|age)', msg) or re.search(r'age\s*(?:is|:|=)?\s*(\d+)', msg)
        if m: fill_data['Age'] = int(m.group(1))
        if 'female' in msg: fill_data['Gender'] = 'Female'
        elif 'male' in msg: fill_data['Gender'] = 'Male'
        m = re.search(r'(?:total bilirubin|bilirubin)\s*(?:is|:|=)?\s*(\d*\.?\d+)', msg)
        if m: fill_data['Total_Bilirubin'] = float(m.group(1))
        m = re.search(r'(?:alt)\s*(?:is|:|=)?\s*(\d+)', msg)
        if m: fill_data['Alamine_Aminotransferase'] = int(m.group(1))
        m = re.search(r'(?:ast)\s*(?:is|:|=)?\s*(\d+)', msg)
        if m: fill_data['Aspartate_Aminotransferase'] = int(m.group(1))
        return fill_data if len(fill_data) >= 2 else None

    # 5. Breast Cancer
    elif 'breast' in page:
        if is_sample_request and is_pos_sample:
            return {'mean_radius': 17.99, 'mean_texture': 20.38, 'mean_perimeter': 122.8, 'mean_area': 1001.0, 'mean_smoothness': 0.1184}
        elif is_sample_request and is_neg_sample:
            return {'mean_radius': 11.28, 'mean_texture': 15.68, 'mean_perimeter': 72.1, 'mean_area': 390.0, 'mean_smoothness': 0.082}
            
        m = re.search(r'(?:radius|mean radius)\s*(?:is|:|=)?\s*(\d*\.?\d+)', msg)
        if m: fill_data['mean_radius'] = float(m.group(1))
        m = re.search(r'(?:texture|mean texture)\s*(?:is|:|=)?\s*(\d*\.?\d+)', msg)
        if m: fill_data['mean_texture'] = float(m.group(1))
        m = re.search(r'(?:perimeter|mean perimeter)\s*(?:is|:|=)?\s*(\d*\.?\d+)', msg)
        if m: fill_data['mean_perimeter'] = float(m.group(1))
        return fill_data if len(fill_data) >= 2 else None

    return None

def call_live_llm(user_msg, current_page, page_context, last_diagnosis):
    gemini_key = os.environ.get('GEMINI_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')
    
    if not gemini_key and not openai_key:
        return None
        
    system_prompt = f"""You are MediNova AI Clinical Copilot, an advanced clinical decision support AI assistant built into MediNova.
You are strictly evidence-based, medically accurate, reassuring, and adhere to medical decision support protocols.
MediNova features 7 diagnostic screening models:
1. Cardiovascular Disease (Random Forest: 13 biomarkers)
2. Diabetes & Metabolic Risk (LightGBM: 8 biomarkers)
3. Chronic Kidney Disease (Random Forest: 24 biomarkers)
4. Hepatic / Liver Disease (Ensemble Classifier: 10 biomarkers)
5. Breast Oncology Cytology (Logistic/SVM: 5 parameters)
6. Malaria Smear Parasitology (Deep CNN: 50x50 cell smear classification)
7. Pneumonia Radiography (Deep CNN: 150x150 chest X-ray classification)

Current User Context:
- Active Page: {current_page}
- Active Diagnostic Session: {json.dumps(last_diagnosis if last_diagnosis else page_context, default=str)}

Guidelines:
- Explain medical terms and biomarkers clearly with reference ranges.
- If acute red-flag symptoms are reported (crushing chest pain, severe dyspnea, seizures, fainting), issue an immediate emergency alert.
- Format responses cleanly with Markdown (bold headings, bullet points, concise tables if helpful).
- Maintain clinical safety disclaimers: this is preliminary AI risk stratification and decision support, not definitive diagnosis.
"""

    if gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\nUser Question: {user_msg}"}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1000
                }
            }
            res = requests.post(url, json=payload, timeout=8)
            if res.status_code == 200:
                data = res.json()
                text = data['candidates'][0]['content']['parts'][0]['text']
                return {
                    "reply": text,
                    "confidence_badge": "MediNova Copilot • Gemini Live AI",
                    "engine": "gemini"
                }
        except Exception as e:
            print(f"Gemini API call failed, falling back to local engine: {e}")
            
    if openai_key:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            res = requests.post(url, json=payload, headers=headers, timeout=8)
            if res.status_code == 200:
                data = res.json()
                text = data['choices'][0]['message']['content']
                return {
                    "reply": text,
                    "confidence_badge": "MediNova Copilot • OpenAI Live AI",
                    "engine": "openai"
                }
        except Exception as e:
            print(f"OpenAI API call failed, falling back to local engine: {e}")
            
    return None

def medinova_clinical_copilot_engine(user_msg, current_page, page_context, last_diagnosis):
    msg = (user_msg or '').strip()
    msg_lower = msg.lower()
    page = (current_page or '').lower()
    
    # 1. EMERGENCY RED-FLAG DETECTION
    for trigger in EMERGENCY_TRIGGERS:
        if trigger in msg_lower:
            reply = (
                "### 🚨 HIGH-PRIORITY CLINICAL EMERGENCY ALERT\n\n"
                "**Immediate Action Protocol:**\n"
                "The acute symptoms you reported (*" + trigger + "*) indicate a potential life-threatening medical emergency.\n\n"
                "- **Call Emergency Services Immediately:** Dial **112 / 911 / 108** right now.\n"
                "- **Do Not Drive Yourself:** Request an urgent emergency ambulance.\n"
                "- **Sit in a Supported Position:** Remain seated upright with chest uncompressed.\n"
                "- **Loosen Restrictive Clothing:** Loosen collar, belt, and buttons to facilitate unlabored respiration.\n"
                "- **Calm Regulated Breathing:** Take slow, deep breaths while awaiting immediate paramedic arrival.\n\n"
                "For the accredited local emergency directory and 24/7 care helplines, consult our Emergency Portal."
            )
            return {
                "reply": reply,
                "confidence_badge": "Emergency Triage • Immediate Action",
                "is_emergency": True,
                "suggested_actions": [
                    {"label": "🚨 Emergency Directory & Care", "url": "/contact"},
                    {"label": "📞 Call 112 / 911", "url": "tel:911"}
                ],
                "follow_up_chips": [
                    "What emergency first aid should I administer?",
                    "View nearest hospital emergency room",
                    "Clear chat session"
                ]
            }

    # 2. CONVERSATIONAL FORM AUTO-FILL
    fill_data = extract_form_auto_fill(msg, page)
    if fill_data:
        param_lines = "\n".join([f"- **`{k}`**: `{v}`" for k, v in fill_data.items()])
        clean_page_name = page.replace('/', '').replace('_', ' ').capitalize()
        reply = (
            f"### ⚡ MediNova Form Auto-Fill Activated\n\n"
            f"I have parsed the patient clinical vitals from your message for **{clean_page_name} Screening**:\n\n"
            f"{param_lines}\n\n"
            "Click the **Auto-Fill Form** button below to populate the input parameters directly into the diagnostic form with a smooth highlight animation."
        )
        return {
            "reply": reply,
            "confidence_badge": "Form Assistant • Auto-Fill Ready",
            "is_emergency": False,
            "form_fill_data": fill_data,
            "suggested_actions": [
                {"label": "⚡ Auto-Fill Form with These Values", "action": "autofill"}
            ],
            "follow_up_chips": [
                "Run diagnostic analysis",
                "Load a normal reference case instead",
                "Explain the significance of these values"
            ]
        }

    # 2.5 NORMAL BIOMARKER REFERENCE RANGES
    is_ref_range_query = any(k in msg_lower for k in [
        'reference range', 'normal range', 'normal biomarker', 'standard target', 
        'reference ranges', 'normal values', 'what is normal', 'biomarker range'
    ])
    if is_ref_range_query:
        reply = (
            "### 🧪 Standard Clinical Biomarker Reference Ranges\n\n"
            "MediNova cross-references your submitted lab parameters against validated medical reference standards:\n\n"
            "| Diagnostic Domain | Clinical Biomarker | Normal / Optimal Target | Clinical Alert Threshold |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| **Cardiovascular** | Resting Blood Pressure | `< 120/80 mm Hg` | `≥ 130/80 mm Hg` (Hypertension) |\n"
            "| **Cardiovascular** | Total Serum Cholesterol | `< 200 mg/dL` | `≥ 240 mg/dL` (Hypercholesterolemia) |\n"
            "| **Cardiovascular** | ST Depression (Oldpeak) | `0.0 - 0.5 mm` | `> 1.5 mm` (Subendocardial Ischemia) |\n"
            "| **Glycemic Panel** | Fasting Blood Glucose | `70 - 99 mg/dL` | `≥ 126 mg/dL` (Diabetes Mellitus) |\n"
            "| **Glycemic Panel** | Glycated Hemoglobin (HbA1c) | `< 5.7%` | `≥ 6.5%` (Diabetes Mellitus) |\n"
            "| **Renal Panel** | Serum Creatinine | `0.7 - 1.2 mg/dL` (M) / `0.5 - 1.0` (F) | `> 1.5 mg/dL` (Impaired GFR) |\n"
            "| **Renal Panel** | Urine Albumin | `Grade 0` (Negative) | `≥ Grade 1` (Microalbuminuria) |\n"
            "| **Renal Panel** | Blood Urea | `15 - 45 mg/dL` | `> 50 mg/dL` (Azotemia) |\n"
            "| **Hepatic Panel** | Total Bilirubin | `0.2 - 1.2 mg/dL` | `> 2.0 mg/dL` (Hyperbilirubinemia) |\n"
            "| **Hepatic Panel** | ALT (Alanine Aminotransferase) | `< 40 U/L` | `> 50 U/L` (Hepatocellular Injury) |\n"
            "| **Hepatic Panel** | AST (Aspartate Aminotransferase)| `< 40 U/L` | `> 50 U/L` (Hepatic / Cardiac Strain) |\n"
            "| **Hepatic Panel** | Albumin-to-Globulin (A/G) Ratio| `1.0 - 2.2` | `< 1.0` (Impaired Protein Synthesis) |\n\n"
            "> 💡 **Clinical Guidance:** Lab test values should always be interpreted alongside complete clinical history and symptoms by a board-certified physician."
        )
        return {
            "reply": reply,
            "confidence_badge": "Clinical Standards • Reference Ranges",
            "is_emergency": False,
            "suggested_actions": [
                {"label": "🫀 Heart Screening", "url": "/heart"},
                {"label": "🩸 Diabetes Screening", "url": "/diabetes"},
                {"label": "🩺 Kidney Screening", "url": "/kidney"},
                {"label": "🧬 Liver Screening", "url": "/liver"}
            ],
            "follow_up_chips": [
                "Which biomarkers are most dangerous when elevated?",
                "Check symptoms and recommend a test",
                "Explain my latest test results"
            ]
        }

    # 2.6 RED-FLAG EMERGENCY WARNING SIGNS (INFORMATIONAL)
    is_emergency_info_query = any(k in msg_lower for k in [
        'emergency warning signs', 'emergency symptoms', 'red-flag', 'red flag',
        'warning signs', 'when to go to er', 'emergency room visit'
    ])
    if is_emergency_info_query:
        reply = (
            "### 🚨 Acute Clinical Red-Flag Warning Signs\n\n"
            "If you or a patient experience any of the following high-priority clinical presentations, **seek immediate emergency care (112 / 911 / 108)**:\n\n"
            "| Clinical Domain | Red-Flag Presentation | Immediate Emergency Threat |\n"
            "| :--- | :--- | :--- |\n"
            "| **Cardiovascular** | Crushing retrosternal chest pressure radiating to jaw/left arm with diaphoresis (sweating) | Acute Myocardial Infarction (Heart Attack) |\n"
            "| **Neurological** | Sudden facial droop, unilateral arm drift, or acute dysarthria (slurred speech) | Acute Ischemic Stroke (CVA) |\n"
            "| **Respiratory** | Severe shortness of breath, gasping, cyanosis (blue lips/fingers), SpO2 < 90% | Acute Respiratory Distress / Pulmonary Embolism |\n"
            "| **Metabolic** | High blood sugar (> 250 mg/dL) with fruity breath, rapid deep breathing, vomiting, confusion | Diabetic Ketoacidosis (DKA) / HHS |\n"
            "| **Infectious** | Spiking fevers with rigors, altered mental status, or black tea-colored urine | Cerebral Malaria / Sepsis |\n"
            "| **Renal / Hepatic** | Sudden anuria (no urine output), massive ascites, confusion, vomiting blood | Acute Kidney / Hepatic Failure |\n\n"
            "- **National Emergency Services:** Dial **112 / 911 / 108** immediately.\n"
            "- **Directory & Hospital Care:** Visit our Emergency Directory for accredited contacts."
        )
        return {
            "reply": reply,
            "confidence_badge": "Emergency Protocols • Life-Threatening Cues",
            "is_emergency": False,
            "suggested_actions": [
                {"label": "🚨 Emergency Directory & Care", "url": "/contact"},
                {"label": "📞 Call 112 / 911", "url": "tel:911"}
            ],
            "follow_up_chips": [
                "Check symptoms and recommend a test",
                "Normal biomarker reference ranges",
                "Explain my latest test results"
            ]
        }

    # 2.7 INTERACTIVE SYMPTOM OVERVIEW
    is_symptom_overview_query = any(k in msg_lower for k in [
        'check symptoms', 'recommend a test', 'symptom check', 'which test should i take',
        'triage symptoms', 'recommend test', 'symptom', 'recommend', 'screening test'
    ])
    if is_symptom_overview_query:
        reply = (
            "### 🩺 MediNova Interactive Symptom Triage\n\n"
            "Describe the symptoms you or a patient are experiencing (e.g., *chest tightness, frequent thirst, foamy urine, jaundice, breast lump, shaking chills, or persistent wet cough*), and I will route you to the appropriate diagnostic tool.\n\n"
            "**Available MediNova Screening Suites:**\n"
            "- **🫀 Cardiovascular (/heart):** Chest discomfort, dyspnea on exertion, hypertension\n"
            "- **🩸 Diabetes & Glycemic (/diabetes):** Polydipsia (thirst), polyuria (urination), fatigue\n"
            "- **🩺 Kidney Function (/kidney):** Foamy urine, peripheral edema (ankle swelling), flank pain\n"
            "- **🧬 Liver & Hepatic (/liver):** Scleral icterus (yellow eyes), dark urine, RUQ abdominal pain\n"
            "- **🔬 Breast Oncology (/breast_cancer):** Palpable breast nodule, skin dimpling, nipple inversion\n"
            "- **🦟 Malaria Parasitology (/malaria):** Shaking rigors, cyclic spiking fever, mosquito exposure\n"
            "- **🫁 Pneumonia Scan (/pneumonia):** Productive cough with sputum, pleuritic chest pain\n\n"
            "Select any screening tool below or type your symptoms to start automated triage!"
        )
        return {
            "reply": reply,
            "confidence_badge": "Symptom Triage • Navigation",
            "is_emergency": False,
            "suggested_actions": [
                {"label": "🫀 Heart Screening", "url": "/heart"},
                {"label": "🩸 Diabetes Screening", "url": "/diabetes"},
                {"label": "🩺 Kidney Screening", "url": "/kidney"},
                {"label": "🧬 Liver Screening", "url": "/liver"}
            ],
            "follow_up_chips": [
                "I have high blood sugar and frequent thirst",
                "I have chest tightness and high BP",
                "Normal biomarker reference ranges"
            ]
        }

    # 3. EXPLAIN MY RESULTS / BIOMARKER BREAKDOWN
    active_diag = last_diagnosis or (page_context.get('last_diagnosis') if page_context else None)
    if not active_diag and page_context and ('patient_summary' in page_context or 'prediction' in page_context):
        active_diag = page_context
        
    is_explain_query = any(k in msg_lower for k in [
        'explain', 'breakdown', 'my result', 'my test', 'my diagnosis', 
        'why was i diagnosed', 'why did i test', 'what caused my result', 'test outcome', 'report'
    ])
    
    if (is_explain_query and active_diag) or ('explain' in msg_lower and active_diag):
        condition = active_diag.get('condition', 'Screening Analysis')
        prediction = active_diag.get('prediction', 'Assessment Complete')
        is_pos = active_diag.get('is_positive', False)
        summary = active_diag.get('patient_summary', [])
        specialist = active_diag.get('specialist', 'Clinical Specialist')
        urgency = active_diag.get('urgency_window', 'Within 24 to 48 hours')
        recommended_tests = active_diag.get('recommended_tests', [])
        
        biomarker_analysis = analyze_patient_biomarkers(condition, summary)
        
        table_rows = []
        for b in biomarker_analysis:
            status_icon = "🚨 High Risk" if b['status'] == 'High Risk' else ("⚠️ Borderline" if b['status'] == 'Borderline' else "✅ Optimal")
            table_rows.append(f"| **{b['label']}** | `{b['value']}` | {b['ref']} | {status_icon} | {b['note']} |")
            
        table_md = "\n".join(table_rows) if table_rows else "| Parameter | Value | Reference |\n|---|---|---|\n| Standard Scan | Evaluated | Within Limits |"
        severity_badge = "High Clinical Alert" if is_pos else "Favorable Physiological Status"
        
        reply = (
            f"### 🩺 MediNova Clinical Copilot: Result Breakdown\n\n"
            f"**Diagnostic Assessment:** `{prediction}` ({severity_badge})\n"
            f"**Condition Evaluated:** {condition}\n\n"
            f"#### 📊 Biomarker & Clinical Parameters Breakdown\n\n"
            f"| Biomarker Parameter | Submitted Value | Standard Target | Clinical Status | Diagnostic Note |\n"
            f"| :--- | :--- | :--- | :--- | :--- |\n"
            f"{table_md}\n\n"
            f"#### 🏥 Specialist Referral & Protocol Timeline\n"
            f"- **Recommended Specialist:** **{specialist}**\n"
            f"- **Consultation Urgency Window:** **{urgency}**\n\n"
            f"#### 🧪 Confirmatory Clinical Workup Recommended\n"
            f"{', '.join(['`' + t + '`' for t in recommended_tests]) if recommended_tests else 'Routine physical evaluation'}\n\n"
            f"#### 📋 Clinical Action Guidance\n"
            f"{'Multiple high-risk markers triggered this positive screening alert. Schedule an in-person diagnostic confirmation with your specialist and assemble previous lab panels.' if is_pos else 'All submitted physiological parameters are within optimal ranges. Continue routine preventive wellness checkups.'}"
        )
        
        actions = [
            {"label": "📄 Print Official Medical Report", "action": "print"}
        ]
        if is_pos:
            actions.append({"label": "🚨 Emergency & Care Directory", "url": "/contact"})
            
        return {
            "reply": reply,
            "confidence_badge": "Clinical Copilot • Context-Aware",
            "is_emergency": False,
            "suggested_actions": actions,
            "follow_up_chips": [
                "Which of my biomarkers are abnormal?",
                "What foods should I eat or avoid?",
                "What questions should I ask my doctor?",
                "Print medical report"
            ]
        }

    # 3.5 NO ACTIVE RESULT CASE FOR EXPLAIN QUERIES
    if is_explain_query and not active_diag:
        reply = (
            "### 📊 MediNova Diagnostic Results Analysis\n\n"
            "No completed diagnostic screening was detected in your active session.\n\n"
            "**How to run a screening and receive an instant AI biomarker analysis:**\n"
            "1. Choose any of MediNova's **7 clinical diagnostic suites** below.\n"
            "2. Enter patient biomarkers, or click **⚡ Fill High-Risk / Healthy Sample** for instant test automation.\n"
            "3. Click **Submit** to run the validated machine learning model.\n"
            "4. On your result card, click **🤖 Ask MediNova AI to Explain This Result** for an automatic breakdown!\n\n"
            "Which diagnostic suite would you like to explore?"
        )
        return {
            "reply": reply,
            "confidence_badge": "Clinical Copilot • Session Context",
            "is_emergency": False,
            "suggested_actions": [
                {"label": "🫀 Heart Screening", "url": "/heart"},
                {"label": "🩸 Diabetes Screening", "url": "/diabetes"},
                {"label": "🩺 Kidney Screening", "url": "/kidney"},
                {"label": "🧬 Liver Screening", "url": "/liver"}
            ],
            "follow_up_chips": [
                "Check symptoms and recommend a test",
                "Normal biomarker reference ranges",
                "Red-flag emergency warning signs"
            ]
        }

    # 4. DIET & PRE-APPOINTMENT PREP
    is_diet_query = any(k in msg_lower for k in ['diet', 'food', 'eat', 'avoid', 'nutrition', 'lifestyle'])
    is_doctor_q_query = any(k in msg_lower for k in ['doctor', 'questions', 'ask', 'appointment', 'consultation', 'prep'])
    
    target_cond_key = None
    if 'heart' in msg_lower or 'cardio' in msg_lower or (active_diag and 'heart' in active_diag.get('condition', '').lower()):
        target_cond_key = 'heart_disease'
    elif 'diabet' in msg_lower or 'sugar' in msg_lower or (active_diag and 'diabet' in active_diag.get('condition', '').lower()):
        target_cond_key = 'diabetes'
    elif 'kidney' in msg_lower or 'renal' in msg_lower or (active_diag and 'kidney' in active_diag.get('condition', '').lower()):
        target_cond_key = 'kidney_disease'
    elif 'liver' in msg_lower or 'hepat' in msg_lower or (active_diag and 'liver' in active_diag.get('condition', '').lower()):
        target_cond_key = 'liver_disease'
    elif 'cancer' in msg_lower or 'breast' in msg_lower or (active_diag and 'breast' in active_diag.get('condition', '').lower()):
        target_cond_key = 'breast_cancer'
    elif 'malaria' in msg_lower or (active_diag and 'malaria' in active_diag.get('condition', '').lower()):
        target_cond_key = 'malaria'
    elif 'pneumonia' in msg_lower or 'lung' in msg_lower or (active_diag and 'pneumonia' in active_diag.get('condition', '').lower()):
        target_cond_key = 'pneumonia'

    if (is_diet_query or is_doctor_q_query) and target_cond_key:
        insight = DISEASE_INSIGHTS[target_cond_key].get('positive', {})
        dos = insight.get('diet_dos', ['Maintain balanced hydration and whole foods.'])
        donts = insight.get('diet_donts', ['Avoid processed foods and excess sodium.'])
        q_list = insight.get('doctor_questions', ['What are my target biomarkers?'])
        
        reply = (
            f"### 🥗 MediNova Clinical Protocol: {target_cond_key.replace('_', ' ').title()}\n\n"
            f"#### ✅ Evidence-Based Nutritional Dos\n"
            + "\n".join([f"- {item}" for item in dos]) + "\n\n"
            f"#### ❌ Foods & Habits to Strictly Avoid\n"
            + "\n".join([f"- {item}" for item in donts]) + "\n\n"
            f"#### ❓ High-Yield Questions for Your Physician\n"
            + "\n".join([f"- \"{item}\"" for item in q_list])
        )
        return {
            "reply": reply,
            "confidence_badge": "Clinical Protocol • Dietary & Consultation Guide",
            "is_emergency": False,
            "suggested_actions": [
                {"label": "📄 Print Medical Summary", "action": "print"},
                {"label": "🏥 Find Specialist", "url": "/contact"}
            ],
            "follow_up_chips": [
                "What confirmatory lab tests are recommended?",
                "Explain my diagnostic results",
                "What are the red-flag emergency symptoms?"
            ]
        }

    # 5. SYMPTOM TRIAGE & SMART ROUTING
    symptom_routes = [
        (['chest pain', 'chest pressure', 'heart palpitations', 'left arm pain', 'angina', 'shortness of breath on walking'],
         'Cardiovascular Disease', '/heart', '🫀 Open Heart Disease Screening',
         'Your symptoms correlate with cardiovascular strain or coronary insufficiency. Immediate multi-parametric screening (resting ECG, blood pressure, lipid profile) is strongly recommended.'),
        (['high blood sugar', 'frequent thirst', 'excessive urination', 'peeing a lot', 'polydipsia', 'polyuria', 'sweet smelling urine', 'fruity breath'],
         'Diabetes & Glycemic Risk', '/diabetes', '🩸 Open Diabetes Screening',
         'Your symptoms match classic indicators of hyperglycemia, impaired insulin sensitivity, or Type 2 Diabetes. A screening with HbA1c and plasma glucose is indicated.'),
        (['foamy urine', 'frothy urine', 'swollen ankles', 'swelling in feet', 'edema', 'puffiness around eyes', 'flank pain', 'kidney pain'],
         'Chronic Kidney Disease', '/kidney', '🩺 Open Kidney Disease Screening',
         'These symptoms may indicate glomerular filtration impairment, microalbuminuria, or renal dysfunction. We advise evaluating serum creatinine, blood urea, and urine albumin.'),
        (['yellow eyes', 'yellow skin', 'jaundice', 'dark urine', 'pale stools', 'right abdominal pain', 'elevated enzymes'],
         'Liver / Hepatic Disease', '/liver', '🧬 Open Liver Disease Screening',
         'These signs correlate with hyperbilirubinemia, hepatic parenchymal inflammation, or biliary stasis. Immediate liver function screening (Total Bilirubin, ALT, AST, ALP) is advised.'),
        (['breast lump', 'lump in breast', 'nipple discharge', 'breast skin dimpling', 'inverted nipple', 'peau d orange'],
         'Breast Oncology Cytology', '/breast_cancer', '🔬 Open Breast Oncology Screening',
         'A newly noted breast mass, texture change, or asymmetric discharge mandates urgent oncology screening and morphological evaluation.'),
        (['shaking chills', 'rigors', 'high fever spikes', 'tropical travel', 'mosquito bite', 'cyclic sweating'],
         'Malaria Parasitology', '/malaria', '🦟 Open Malaria Smear Screening',
         'Paroxysmal shaking chills and cyclic febrile episodes following mosquito exposure suggest intraerythrocytic Plasmodium infection. A Giemsa-stained thin blood smear analysis is recommended.'),
        (['cough with phlegm', 'coughing up mucus', 'chest pain when coughing', 'crackles', 'lung infection', 'breathing hurts'],
         'Pneumonia Pulmonary Scan', '/pneumonia', '🫁 Open Pneumonia X-Ray Screening',
         'Persistent wet cough with colored sputum, pleuritic chest pain, and fever suggest lower respiratory consolidation or pneumonia. Upload a chest X-ray for deep convolutional analysis.')
    ]
    
    for keywords, cond_name, route_url, action_label, triage_desc in symptom_routes:
        if any(kw in msg_lower for kw in keywords):
            reply = (
                f"### 🩺 MediNova Symptom Triage: {cond_name}\n\n"
                f"{triage_desc}\n\n"
                f"- **Recommended Screening Tool:** **MediNova {cond_name} Diagnostic**\n"
                f"- **Direct Route:** Click the button below to launch the automated screening model.\n\n"
                f"> 💡 **Clinical Reminder:** Early risk identification significantly improves treatment outcomes. If your symptoms escalate or you develop acute chest pain or respiratory distress, contact emergency responders (112 / 911 / 108) immediately."
            )
            return {
                "reply": reply,
                "confidence_badge": "Symptom Triage • Model Match",
                "is_emergency": False,
                "suggested_actions": [
                    {"label": action_label, "url": route_url},
                    {"label": "🚨 Emergency Support Directory", "url": "/contact"}
                ],
                "follow_up_chips": [
                    f"What biomarkers does the {cond_name} tool test?",
                    "What are the red-flag emergency symptoms?",
                    "How accurate is this MediNova model?"
                ]
            }

    # 6. BIOMARKER & MEDICAL TERM EXPLAINER
    for term_key, term_info in MEDICAL_EXPLAINERS.items():
        if term_key in msg_lower:
            reply = (
                f"### 📖 MediNova Clinical Explainer: {term_info['term']}\n\n"
                f"{term_info['explanation']}\n\n"
                f"> **Clinical Context:** In MediNova's diagnostic pipeline, this biomarker is cross-referenced with machine learning reference vectors to detect abnormal physiological shifts before irreversible pathology develops."
            )
            return {
                "reply": reply,
                "confidence_badge": "Clinical Knowledge Graph • Validated",
                "is_emergency": False,
                "suggested_actions": [
                    {"label": "🔍 Explore Diagnostic Suite", "url": "/features"}
                ],
                "follow_up_chips": [
                    "What are the normal reference ranges?",
                    "Explain my test results",
                    "What foods help improve these numbers?"
                ]
            }

    # 7. GENERAL CLINICAL COPILOT OVERVIEW
    reply = (
        "### 🤖 MediNova AI Clinical Copilot\n\n"
        "Welcome! I am your context-aware clinical assistant integrated directly into MediNova's diagnostic system. Here is what I can do for you right now:\n\n"
        "1. **Explain Diagnostic Results:** Provide an in-depth breakdown of your submitted biomarkers, flag abnormal values against clinical standards, and explain the AI classification.\n"
        "2. **Symptom Triage & Navigation:** Tell me what symptoms you or a patient are experiencing, and I will recommend the exact screening model across our 7 clinical suites.\n"
        "3. **Conversational Form Auto-Fill:** Type patient vitals into chat (e.g., *\"58 yo male with BP 145, cholesterol 240, max hr 130\"*) to auto-populate any diagnostic form with one click.\n"
        "4. **Clinical Protocols & Pre-Appointment Prep:** Access dietary guidelines (dos/don'ts), specialist referral urgency windows, and high-yield questions for your physician.\n"
        "5. **Biomarker Term Explainer:** Ask about any clinical test parameter (such as Oldpeak, ST depression, Thallium scans, HbA1c, A/G ratio, or Serum Creatinine).\n\n"
        "How can I assist your clinical workflow today?"
    )
    return {
        "reply": reply,
        "confidence_badge": "Clinical Copilot • Online",
        "is_emergency": False,
        "suggested_actions": [
            {"label": "🫀 Heart Screening", "url": "/heart"},
            {"label": "🩸 Diabetes Screening", "url": "/diabetes"},
            {"label": "🩺 Kidney Screening", "url": "/kidney"},
            {"label": "🧬 Liver Screening", "url": "/liver"}
        ],
        "follow_up_chips": [
            "Explain my test results",
            "What are the normal reference ranges for biomarkers?",
            "What symptoms require an emergency room visit?",
            "Auto fill a positive heart case"
        ]
    }


# ============================================================================
# INTERACTIVE CLINICAL DOCTOR ADVISOR & SITUATION-HANDLING ENGINE
# ============================================================================

CLINICAL_DOMAINS = {
    'heart': {
        'name': 'Cardiovascular Health & Chest Symptoms',
        'route': '/heart',
        'action_label': '🫀 Open Heart Screening',
        'insight_key': 'heart_disease',
        'keywords': [
            'chest pain', 'chest pressure', 'heart palpitations', 'left arm pain', 'angina',
            'palpitations', 'high bp', 'blood pressure', 'hypertension', 'shortness of breath on walking',
            'heart attack', 'cardiac', 'heart', 'bp'
        ],
        'q1_text': "To help you handle this safely, how severe are your chest symptoms or blood pressure readings right now, and how long have you had them?",
        'q1_chips': ["🔴 Severe / Started suddenly", "🟡 Moderate / Past few days", "🟢 Mild / Just checking", "📊 High Blood Pressure"],
        'q2_text': "Are you experiencing shortness of breath, unexplained cold sweating, dizziness, nausea, or pain radiating to your left arm, neck, or jaw?",
        'q2_chips': ["Yes, I have those symptoms", "No, none of those", "Only mild chest tightness", "I already ran a MediNova test"],
        'emergency_advice': (
            "🚨 **CRITICAL FIRST-AID — CALL 112 / 911 / 108 IMMEDIATELY:**\n"
            "- **Sit Down & Rest:** Sit upright and rest immediately in a well-ventilated room. Do not walk, climb stairs, or exert yourself.\n"
            "- **Loosen Tight Clothing:** Loosen collar, necktie, and waistband to reduce vascular resistance.\n"
            "- **Alert Someone Nearby:** Alert a family member, neighbor, or colleague right now. Do not drive yourself to the emergency department.\n"
            "- **Emergency Dispatch:** Call 112 / 911 / 108 right away and request an ambulance with cardiac life-support."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Rest in an Upright Posture:** Sit with your back supported. Take slow, regulated abdominal breaths.\n"
            "2. **Check Resting Vitals:** Measure your resting blood pressure and pulse rate with a home cuff if available. Write down the numbers.\n"
            "3. **Zero Stimulants:** Avoid all caffeine, energy drinks, nicotine, alcohol, and high-sodium meals.\n"
            "4. **Monitor Warning Signs:** If you notice sudden chest pressure radiating to your arm or jaw, seek emergency medical care immediately."
        )
    },
    'diabetes': {
        'name': 'Diabetes & Glycemic Regulation',
        'route': '/diabetes',
        'action_label': '🩸 Open Diabetes Screening',
        'insight_key': 'diabetes',
        'keywords': [
            'diabetes', 'sugar', 'blood sugar', 'glucose', 'hba1c', 'frequent thirst',
            'excessive urination', 'peeing a lot', 'polydipsia', 'polyuria', 'sweet smelling urine',
            'fruity breath', 'insulin'
        ],
        'q1_text': "To help you handle your blood sugar safely, what are your current glucose levels or symptoms, and how long have you noticed these changes?",
        'q1_chips': ["🔴 Very High (>250 mg/dL) / Sudden", "🟡 Moderately High / Past few days", "🟢 Mild / Routine checking", "📉 Low Blood Sugar (<70 mg/dL)"],
        'q2_text': "Are you having extreme unquenchable thirst, frequent urination at night, sudden blurred vision, nausea, or fruity-smelling breath?",
        'q2_chips': ["Yes, I have those symptoms", "No, none of those", "Just thirst & fatigue", "I already ran a MediNova test"],
        'emergency_advice': (
            "🚨 **CRITICAL FIRST-AID — DKA / ACUTE HYPERGLYCEMIA ALERT:**\n"
            "- **Immediate Glucose & Ketone Test:** Test blood glucose and urine ketones if a home meter is accessible.\n"
            "- **Call Emergency (112 / 911 / 108):** If glucose is >250–300 mg/dL accompanied by nausea, vomiting, confusion, or fruity breath, seek emergency care to rule out Diabetic Ketoacidosis (DKA).\n"
            "- **Do Not Strenuously Exercise:** Strenuous exertion with high ketones accelerates dehydration and metabolic acidosis."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Hydrate with Plain Water:** Drink 2 to 3 glasses of clean room-temperature water to help your kidneys clear excess glucose.\n"
            "2. **Strictly Cut Simple Sugars:** Avoid fruit juices, sodas, sweets, refined flours, and high-glycemic carbohydrates.\n"
            "3. **Gentle Walking:** A light 15-minute flat walk after meals activates peripheral muscle GLUT-4 glucose transporters.\n"
            "4. **Log Biomarkers:** Record fasting and 2-hour post-prandial blood glucose along with recent HbA1c values."
        )
    },
    'kidney': {
        'name': 'Kidney & Renal Health',
        'route': '/kidney',
        'action_label': '🩺 Open Kidney Screening',
        'insight_key': 'kidney_disease',
        'keywords': [
            'kidney', 'foamy urine', 'frothy urine', 'swollen ankles', 'swelling in feet',
            'edema', 'puffiness around eyes', 'flank pain', 'kidney pain', 'creatinine', 'blood urea', 'renal'
        ],
        'q1_text': "To assess your kidney health safely, how severe is the swelling or urine change, and when did you first notice it?",
        'q1_chips': ["🔴 Severe swelling / Sudden drop in urine", "🟡 Moderate swelling / Foamy urine", "🟢 Mild puffiness / Just checking", "🧪 Abnormal lab creatinine"],
        'q2_text': "Have you noticed persistent foamy/bubbly urine, cola-colored or bloody urine, swelling around your ankles/eyes, or lower back flank pain?",
        'q2_chips': ["Yes, swelling and foamy urine", "No, none of those", "Only foamy urine", "I already ran a MediNova test"],
        'emergency_advice': (
            "🚨 **CRITICAL FIRST-AID — ACUTE RENAL / FLUID OVERLOAD ALERT:**\n"
            "- A sudden severe decrease in urine output paired with breathing difficulty or rapid full-body edema suggests acute renal compromise or pulmonary fluid overload.\n"
            "- Seek immediate emergency medical assessment (112 / 911 / 108) or proceed to the nearest acute dialysis/nephrology hospital."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Discontinue OTC NSAID Painkillers:** Completely avoid ibuprofen, naproxen, and diclofenac—they constrict renal blood flow.\n"
            "2. **Strict Low-Sodium Diet:** Keep sodium intake under 1,500 mg/day; avoid table salt and processed foods to reduce fluid retention.\n"
            "3. **Monitor Urine Output & Fluid Intake:** Record your daily fluid intake and track urine color and volume.\n"
            "4. **Control Blood Pressure:** Keep your resting blood pressure well below 130/80 mm Hg to prevent glomerular capillary injury."
        )
    },
    'liver': {
        'name': 'Liver & Hepatic Function',
        'route': '/liver',
        'action_label': '🧬 Open Liver Screening',
        'insight_key': 'liver_disease',
        'keywords': [
            'liver', 'yellow eyes', 'yellow skin', 'jaundice', 'dark urine', 'pale stools',
            'right abdominal pain', 'bilirubin', 'alt', 'ast', 'hepatic', 'cirrhosis', 'fatty liver'
        ],
        'q1_text': "To evaluate your liver status safely, how noticeable is the jaundice, abdominal pain, or fatigue, and how quickly did it develop?",
        'q1_chips': ["🔴 Sudden deep yellowing / Upper pain", "🟡 Noticeable jaundice / Past few days", "🟢 Mild fatigue / Routine check", "🧪 Elevated SGOT/SGPT enzymes"],
        'q2_text': "Have you observed yellowing in the whites of your eyes, tea-colored dark urine, clay-colored pale stools, or right upper abdominal swelling?",
        'q2_chips': ["Yes, yellow eyes and dark urine", "No, none of those", "Only fatigue & abdominal ache", "I already ran a MediNova test"],
        'emergency_advice': (
            "🚨 **CRITICAL FIRST-AID — ACUTE HEPATIC ENCEPHALOPATHY ALERT:**\n"
            "- Acute jaundice accompanied by mental confusion, severe drowsiness, slurred speech, or persistent vomiting requires immediate emergency hospitalization."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Complete Alcohol Cessation:** Eliminate 100% of alcohol intake immediately to protect hepatocytes from acute toxic stress.\n"
            "2. **Avoid Hepatotoxic Drugs:** Avoid paracetamol in high doses and discontinue unverified herbal remedies.\n"
            "3. **Clean Low-Fat Nutrition:** Eat light, easily digestible meals (boiled vegetables, lean chicken, lentils) and stay hydrated with clean water.\n"
            "4. **Complete Liver Panel (LFT):** Test Total Bilirubin, Direct Bilirubin, ALT, AST, ALP, and Albumin/Globulin ratio."
        )
    },
    'breast_cancer': {
        'name': 'Breast Oncology & Tissue Screening',
        'route': '/breast_cancer',
        'action_label': '🔬 Open Breast Oncology Screening',
        'insight_key': 'breast_cancer',
        'keywords': [
            'breast lump', 'lump in breast', 'nipple discharge', 'breast skin dimpling',
            'inverted nipple', 'peau d orange', 'breast pain', 'breast mass', 'breast cancer', 'mammogram'
        ],
        'q1_text': "To guide your clinical evaluation safely, how long have you noticed this breast lump or tissue change, and does it feel tender or hard?",
        'q1_chips': ["🔴 Hard / Fixed / Rapidly growing", "🟡 New palpable lump / Past weeks", "🟢 Cyclical tenderness / Hormonal", "🔬 Have biopsy/mammogram numbers"],
        'q2_text': "Have you observed any skin dimpling, spontaneous bloody/clear nipple discharge, recent nipple retraction, or swollen armpit lymph nodes?",
        'q2_chips': ["Yes, I have noticed those signs", "No, none of those", "Only a localized movable lump", "I already have cytology values"],
        'emergency_advice': (
            "🚨 **CLINICAL ONCOLOGY ADVICE:**\n"
            "- Rapidly developing inflammatory skin changes (red, hot, swollen breast skin with an orange-peel texture) or rapidly growing hard masses should be evaluated by a breast oncology specialist within 24–48 hours."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Do Not Squeeze or Massage:** Avoid repetitive squeezing or palpating, which causes local tissue edema and irritation.\n"
            "2. **Record Symptom Timeline:** Note the exact date noticed, relationship to menstrual cycles, and any family oncological history.\n"
            "3. **Schedule Dedicated Imaging:** Arrange a clinical breast exam (CBE) along with bilateral breast ultrasound or digital mammogram.\n"
            "4. **Run MediNova Cytology Analysis:** Screen your fine-needle aspiration (FNA) cell morphometrics with MediNova's machine learning model."
        )
    },
    'malaria': {
        'name': 'Malaria Parasitology & Febrile Illness',
        'route': '/malaria',
        'action_label': '🦟 Open Malaria Smear Screening',
        'insight_key': 'malaria',
        'keywords': [
            'malaria', 'shaking chills', 'rigors', 'high fever spikes', 'tropical travel',
            'mosquito bite', 'cyclic sweating', 'fever with chills', 'intermittent fever'
        ],
        'q1_text': "To assess possible malaria parasite infection, what is your highest fever temperature, and when did the chills or fever begin?",
        'q1_chips': ["🔴 High fever (>103°F) + Shaking chills", "🟡 Moderate fever (100-102°F) / Past 3 days", "🟢 Low-grade fever / Mild body ache", "🦟 Recent mosquito bites / Travel"],
        'q2_text': "Are the fevers occurring in periodic cycles (e.g., every 24–48 hours) with teeth-chattering chills followed by heavy sweating, severe headache, or vomiting?",
        'q2_chips': ["Yes, cyclic chills and sweating", "No, constant unremitting fever", "Fever with severe body aches", "I have a blood smear slide ready"],
        'emergency_advice': (
            "🚨 **CRITICAL FIRST-AID — SEVERE FALCIPARUM MALARIA WARNING:**\n"
            "- High fever accompanied by delirium, extreme lethargy, dark/blackish urine (hemoglobinuria), or shortness of breath requires immediate emergency hospital admission for intravenous antiparasitic therapy."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Aggressive Hydration:** Fevers and rigors cause severe dehydration; drink Oral Rehydration Salts (ORS), coconut water, and clean fluids.\n"
            "2. **Tepid Sponging & Antipyretics:** Sponge forehead and limbs with lukewarm water; take doctor-recommended paracetamol to reduce fever. Avoid aspirin.\n"
            "3. **Avoid Random Antibiotics:** Malaria is an intraerythrocytic protozoan parasite (Plasmodium), not a bacteria. Antibiotics will not treat it.\n"
            "4. **Prompt Smear Test:** Get a thin and thick peripheral Giemsa blood smear or Rapid Diagnostic Antigen Test (RDT) immediately."
        )
    },
    'pneumonia': {
        'name': 'Pneumonia & Respiratory Health',
        'route': '/pneumonia',
        'action_label': '🫁 Open Pneumonia X-Ray Screening',
        'insight_key': 'pneumonia',
        'keywords': [
            'pneumonia', 'cough with phlegm', 'coughing up mucus', 'chest pain when coughing',
            'crackles', 'lung infection', 'breathing hurts', 'shortness of breath', 'difficulty breathing',
            'persistent cough', 'wheezing', 'spo2', 'oxygen'
        ],
        'q1_text': "To evaluate your respiratory status, how severe is your cough or shortness of breath, and what color is the mucus?",
        'q1_chips': ["🔴 Severe shortness of breath / High fever", "🟡 Deep cough with yellow/green phlegm", "🟢 Mild dry cough / Past few days", "📉 Low SpO2 (<94%) on oximeter"],
        'q2_text': "Are you experiencing sharp chest pain when inhaling deeply or coughing, breathing difficulty while resting, or pulse oximeter SpO2 below 94%?",
        'q2_chips': ["Yes, chest pain & low oxygen", "No, just cough and mild fever", "Breathing is harder when lying down", "I already have a chest X-Ray image"],
        'emergency_advice': (
            "🚨 **CRITICAL FIRST-AID — RESPIRATORY COMPROMISE ALERT:**\n"
            "- If your pulse oximeter SpO2 is below 92–94%, breathing is rapid or shallow, lips/fingernails appear cyanotic (bluish), or confusion occurs, call **112 / 911 / 108** immediately.\n"
            "- Sit propped upright. Do not lie flat on your back."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Maintain Upright Seated Posture:** Sit with your back elevated at 45–90 degrees to assist diaphragmatic expansion and air entry.\n"
            "2. **Check Oxygen Saturation (SpO2):** Use a fingertip pulse oximeter every 3–4 hours and record each reading.\n"
            "3. **Inhale Warm Steam & Hydrate:** Inhaling steam loosens thick pulmonary secretions. Sip warm water with honey and lemon.\n"
            "4. **Chest X-Ray Analysis:** Upload a PA chest radiograph to MediNova's deep convolutional radiography scanner for lung consolidation analysis."
        )
    },
    'general': {
        'name': 'General Health & Clinical Triage',
        'route': '/features',
        'action_label': '🔍 Explore MediNova Diagnostic Suite',
        'insight_key': 'general',
        'keywords': [
            'dizzy', 'dizziness', 'lightheaded', 'faint', 'weakness', 'fatigue', 'nausea',
            'vomiting', 'headache', 'body ache', 'pain', 'unwell', 'sick'
        ],
        'q1_text': "To help guide your situation safely, how intense are your symptoms and did they start suddenly or gradually?",
        'q1_chips': ["🔴 Started suddenly & feels intense", "🟡 Moderate discomfort for a few days", "🟢 Mild / Just feeling run down", "📊 I want to check a specific organ"],
        'q2_text': "Are you having any focal red flags such as chest tightness, sudden shortness of breath, extreme thirst, jaundice, or focal weakness?",
        'q2_chips': ["Yes, chest or breathing issues", "Yes, blood sugar or thirst issues", "Yes, fever and shaking chills", "No specific red flags"],
        'emergency_advice': (
            "🚨 **GENERAL EMERGENCY REMINDER:**\n"
            "- If you experience sudden crushing chest pressure, severe shortness of breath, sudden facial droop or arm weakness, or loss of consciousness, call **112 / 911 / 108** immediately."
        ),
        'immediate_care': (
            "✅ **Immediate Action — How to Handle This Right Now:**\n"
            "1. **Rest & Hydrate:** Sit or lie down in a quiet, cool space. Sip water with electrolytes slowly.\n"
            "2. **Monitor Resting Vitals:** Check your temperature, resting pulse, and blood pressure if home devices are available.\n"
            "3. **Avoid Sudden Posture Shifts:** Rise slowly from sitting or lying down to prevent orthostatic dizziness.\n"
            "4. **Screen Prominent Symptoms:** Use MediNova's organ-specific diagnostic suites to screen targeted biomarkers."
        )
    }
}

def detect_clinical_domain(message):
    msg = (message or '').lower()
    for domain_key in ['heart', 'diabetes', 'kidney', 'liver', 'breast_cancer', 'malaria', 'pneumonia', 'general']:
        domain_info = CLINICAL_DOMAINS[domain_key]
        if any(kw in msg for kw in domain_info['keywords']):
            return domain_key
    return None

def generate_situation_action_plan(domain_key, severity, associated_symptoms):
    d_info = CLINICAL_DOMAINS.get(domain_key, CLINICAL_DOMAINS['general'])
    insight_key = d_info.get('insight_key', 'general')
    
    sev_str = (severity or '').lower()
    assoc_str = (associated_symptoms or '').lower()
    is_high_risk = any(w in sev_str for w in ['severe', 'sudden', 'high', 'critical', 'red', '🔴', '103', '250', 'bad']) or \
                   any(w in assoc_str for w in ['yes', 'shortness of breath', 'severe', 'pain', 'sweating', 'arm', 'jaw', 'low oxygen', 'chills'])
    
    immediate_section = d_info['emergency_advice'] if is_high_risk else d_info['immediate_care']
    
    diet_section = ""
    if insight_key in DISEASE_INSIGHTS:
        disease_dict = DISEASE_INSIGHTS[insight_key]
        sub_key = 'malignant' if 'malignant' in disease_dict else ('infected' if 'infected' in disease_dict else 'positive')
        insight_data = disease_dict.get(sub_key, {})
        
        dos = insight_data.get('diet_dos', [])
        donts = insight_data.get('diet_donts', [])
        recs = insight_data.get('recommendations', [])
        doc_qs = insight_data.get('doctor_questions', [])
        
        diet_section = (
            "\n\n#### 🥗 3. Safe Diet & Care Protocol\n"
            "**✅ Recommended Evidence-Based Nutritional Dos:**\n"
            + "\n".join([f"- {item}" for item in dos[:3]]) + "\n\n"
            "**❌ Foods & Habits to Strictly Avoid:**\n"
            + "\n".join([f"- {item}" for item in donts[:3]]) + "\n\n"
            "**📋 Clinical Recommendations:**\n"
            + "\n".join([f"- {item}" for item in recs[:3]]) + "\n\n"
            "**❓ High-Yield Questions for Your Physician:**\n"
            + "\n".join([f"- \"{item}\"" for item in doc_qs[:3]])
        )
    
    reply = (
        f"### 🩺 MediNova Situation-Handling Action Plan: {d_info['name']}\n\n"
        f"Based on your clinical consultation (*Severity: {severity or 'Noted'}* | *Associated Indicators: {associated_symptoms or 'Evaluated'}*), here is your tailored step-by-step action plan:\n\n"
        f"#### 🚨 1. Immediate Action — How to Handle This Right Now\n"
        f"{immediate_section}\n\n"
        f"#### 🩺 2. Recommended MediNova AI Screening\n"
        f"- **Recommended Modality:** **{d_info['name']}**\n"
        f"- **Direct Route:** Click the screening button below to launch the automated AI model and test your clinical biomarkers.\n"
        f"{diet_section}\n\n"
        f"> 💡 **Clinical Reminder:** MediNova is an AI clinical decision support tool. If your symptoms worsen or you feel acute distress, contact emergency responders (**112 / 911 / 108**) immediately."
    )
    
    actions = [
        {"label": d_info['action_label'], "url": d_info['route']},
        {"label": "🏥 Emergency Directory & Specialists", "url": "/contact"}
    ]
    
    quick_replies = [
        "📊 Explain my latest test result",
        "🫀 Check cardiovascular symptoms",
        "🩸 Check blood sugar / diabetes",
        "🌡️ Check fever or cough",
        "↺ Reset Consultation"
    ]
    
    return {
        "reply": reply,
        "quick_replies": quick_replies,
        "actions": actions,
        "confidence_badge": f"Action Plan • {d_info['name']}",
        "is_emergency": is_high_risk,
        "suggested_actions": actions,
        "follow_up_chips": quick_replies
    }

def handle_explain_test_result(last_diagnosis):
    if not last_diagnosis:
        reply = (
            "### 📊 MediNova Diagnostic Results Analysis\n\n"
            "No completed diagnostic screening was detected in your active session.\n\n"
            "**How to run a screening and receive an instant AI biomarker analysis:**\n"
            "1. Choose any of MediNova's **7 clinical diagnostic suites** below.\n"
            "2. Enter patient biomarkers, or click **⚡ Fill High-Risk / Healthy Sample** for instant test automation.\n"
            "3. Click **Submit** to run the validated machine learning model.\n"
            "4. On your result card, click **🤖 Ask MediNova AI to Explain This Result** for an automatic breakdown!\n\n"
            "Which diagnostic suite would you like to explore?"
        )
        actions = [
            {"label": "🫀 Heart Screening", "url": "/heart"},
            {"label": "🩸 Diabetes Screening", "url": "/diabetes"},
            {"label": "🩺 Kidney Screening", "url": "/kidney"},
            {"label": "🧬 Liver Screening", "url": "/liver"},
            {"label": "🔍 All 7 Suites", "url": "/features"}
        ]
        chips = [
            "🫀 Chest pain or high BP",
            "🩸 High blood sugar advice",
            "🌡️ Fever or cough check",
            "🧪 Kidney / Liver check"
        ]
        return {
            "reply": reply,
            "quick_replies": chips,
            "actions": actions,
            "confidence_badge": "Session Context • No Active Test",
            "is_emergency": False,
            "suggested_actions": actions,
            "follow_up_chips": chips
        }
        
    condition = last_diagnosis.get('condition', 'Diagnostic Screening')
    route = last_diagnosis.get('route', '/features')
    prediction = last_diagnosis.get('prediction', 'Evaluated')
    is_pos = bool(last_diagnosis.get('is_positive', False))
    patient_summary = last_diagnosis.get('patient_summary', [])
    specialist = last_diagnosis.get('specialist', 'Clinical Specialist')
    urgency = last_diagnosis.get('urgency_window', 'Within 24-48 hours')
    recommended_tests = last_diagnosis.get('recommended_tests', [])
    
    summary_lines = []
    for item in patient_summary:
        summary_lines.append(f"- **{item.get('label')}**: `{item.get('value')}`")
    summary_md = "\n".join(summary_lines) if summary_lines else "- General scan parameters evaluated within clinical thresholds."
    
    status_badge = "🚨 High Clinical Alert / Elevated Risk" if is_pos else "✅ Optimal / Low Risk Detected"
    
    reply = (
        f"### 🩺 MediNova AI Doctor: Diagnostic Result Explanation\n\n"
        f"**Screening Condition:** {condition}\n"
        f"**Algorithmic Assessment:** `{prediction}`\n"
        f"**Clinical Status:** **{status_badge}**\n\n"
        f"#### 📊 Biomarker Breakdown Recorded for Patient:\n"
        f"{summary_md}\n\n"
        f"#### 🏥 Recommended Specialist & Urgency:\n"
        f"- **Specialist:** **{specialist}**\n"
        f"- **Urgency Window:** **{urgency}**\n"
        f"- **Confirmatory Clinical Tests:** {', '.join(['`' + t + '`' for t in recommended_tests]) if recommended_tests else 'Routine primary care checkup'}\n\n"
        f"#### 📋 How to Handle This Result:\n"
        f"{'Our machine learning models detected elevated physiological risk markers. Schedule an in-person clinical consultation with your specialist and assemble previous medical records for review.' if is_pos else 'All submitted biomarkers and imaging features align with healthy physiological reference ranges. Continue routine preventive wellness checkups.'}"
    )
    
    actions = [
        {"label": f"📄 Re-open {condition} Suite", "url": route},
        {"label": "🏥 Specialist & Emergency Directory", "url": "/contact"}
    ]
    quick_replies = [
        "What diet should I follow?",
        "What questions should I ask my doctor?",
        "Check another symptom",
        "↺ Reset Consultation"
    ]
    
    return {
        "reply": reply,
        "quick_replies": quick_replies,
        "actions": actions,
        "confidence_badge": f"Result Breakdown • {condition}",
        "is_emergency": is_pos,
        "suggested_actions": actions,
        "follow_up_chips": quick_replies
    }


# ============================================================================
# API ENDPOINTS: CHATBOT INTEGRATION
# ============================================================================

@app.route('/api/chat', methods=['POST'])
def api_chat():
    try:
        data = request.get_json(force=True) or {}
        user_message = data.get('message', '').strip()
        current_page = data.get('current_page', '')
        page_context = data.get('page_context', {})
        
        initial_chips = [
            "🫀 Chest pain or high BP",
            "🩸 High blood sugar / Diabetes advice",
            "🌡️ Fever, chills, or persistent cough",
            "🧪 Kidney or Liver symptom check",
            "📊 Explain my latest test result"
        ]
        initial_actions = [
            {"label": "🫀 Heart Screening", "url": "/heart"},
            {"label": "🩸 Diabetes Screening", "url": "/diabetes"},
            {"label": "🩺 Kidney Screening", "url": "/kidney"},
            {"label": "🔍 All 7 Suites", "url": "/features"}
        ]
        
        if not user_message:
            return jsonify({
                "reply": "Hi! I'm your **MediNova Clinical AI Advisor**. Tell me how you're feeling or pick a situation below—I'll ask a couple of quick clinical questions and guide you step-by-step on how to handle it safely.",
                "quick_replies": initial_chips,
                "actions": initial_actions,
                "confidence_badge": "Clinical Advisor Online",
                "is_emergency": False,
                "suggested_actions": initial_actions,
                "follow_up_chips": initial_chips
            })
            
        msg_lower = user_message.lower()
        
        # 1. Reset / restart check
        if msg_lower in ['reset', 'restart', 'clear', 'start over', '↺ reset consultation', 'reset chat', 'start new']:
            session['chat_state'] = {'stage': 'idle', 'topic': None, 'severity': None, 'associated_symptoms': None}
            session.modified = True
            return jsonify({
                "reply": "I have reset our consultation. How are you feeling right now? Pick a symptom or situation below, or describe how you feel in your own words:",
                "quick_replies": initial_chips,
                "actions": initial_actions,
                "confidence_badge": "Advisor Reset",
                "is_emergency": False,
                "suggested_actions": initial_actions,
                "follow_up_chips": initial_chips
            })
            
        # 2. Check "Explain my test result"
        is_explain_query = any(k in msg_lower for k in [
            'explain', 'latest test', 'my test', 'my result', 'my diagnosis',
            'explain my results', 'explain my test result', 'explain my latest test result',
            'biomarker analysis'
        ])
        if is_explain_query:
            last_diag = session.get('last_diagnosis')
            resp = handle_explain_test_result(last_diag)
            return jsonify(resp)
            
        # 3. Retrieve conversation state
        chat_state = session.get('chat_state')
        if not isinstance(chat_state, dict):
            chat_state = {'stage': 'idle', 'topic': None, 'severity': None, 'associated_symptoms': None}
            
        current_stage = chat_state.get('stage', 'idle')
        current_topic = chat_state.get('topic')
        
        # Check if user is answering Turn 1 (Severity & Duration)
        if current_stage == 'turn1_asked' and current_topic and current_topic in CLINICAL_DOMAINS:
            d_info = CLINICAL_DOMAINS[current_topic]
            chat_state['severity'] = user_message
            chat_state['stage'] = 'turn2_asked'
            session['chat_state'] = chat_state
            session.modified = True
            
            is_emergency = any(trig in msg_lower for trig in EMERGENCY_TRIGGERS)
            
            reply = (
                f"### 🩺 MediNova Clinical AI Advisor\n"
                f"**Noted Severity & Duration:** *\"{user_message}\"*\n\n"
                f"Thank you for sharing that important context. To check for critical complications or red-flag warning signs:\n\n"
                f"👉 **Follow-Up Question 2 of 2:**\n"
                f"{d_info['q2_text']}"
            )
            actions = [{"label": d_info['action_label'], "url": d_info['route']}]
            if is_emergency:
                actions.insert(0, {"label": "🚨 Call 112 / 911 / 108 Immediately", "url": "/contact"})
                
            return jsonify({
                "reply": reply,
                "quick_replies": d_info['q2_chips'],
                "actions": actions,
                "confidence_badge": f"Triage Step 2/2 • {d_info['name']}",
                "is_emergency": is_emergency,
                "suggested_actions": actions,
                "follow_up_chips": d_info['q2_chips']
            })
            
        # Check if user is answering Turn 2 (Red-Flag / Associated Symptoms -> Action Plan)
        elif current_stage == 'turn2_asked' and current_topic and current_topic in CLINICAL_DOMAINS:
            chat_state['associated_symptoms'] = user_message
            chat_state['stage'] = 'completed'
            session['chat_state'] = chat_state
            session.modified = True
            
            plan_response = generate_situation_action_plan(
                current_topic,
                chat_state.get('severity'),
                user_message
            )
            return jsonify(plan_response)
            
        # Otherwise, check if user describes a symptom / picks a clinical domain
        matched_domain = detect_clinical_domain(user_message)
        
        # If user directly asks for diet or physician questions
        if any(k in msg_lower for k in ['diet', 'food', 'eat', 'avoid', 'doctor questions', 'nutrition']) and matched_domain:
            plan_response = generate_situation_action_plan(matched_domain, "Routine Dietary Inquiry", "General Guidance")
            return jsonify(plan_response)
            
        if matched_domain:
            # Turn 1: Start multi-turn triage consultation
            d_info = CLINICAL_DOMAINS[matched_domain]
            session['chat_state'] = {
                'stage': 'turn1_asked',
                'topic': matched_domain,
                'severity': None,
                'associated_symptoms': None
            }
            session.modified = True
            
            is_emergency = any(trig in msg_lower for trig in EMERGENCY_TRIGGERS)
            
            reply = (
                f"### 🩺 MediNova Clinical AI Advisor\n"
                f"**Symptom Domain Identified:** {d_info['name']}\n\n"
                f"I hear your concern, and I will help you handle this safely and step-by-step. To recommend the right immediate first-aid, safe diet protocol, and matching MediNova AI screening:\n\n"
                f"👉 **Follow-Up Question 1 of 2:**\n"
                f"{d_info['q1_text']}"
            )
            actions = [{"label": d_info['action_label'], "url": d_info['route']}]
            if is_emergency:
                actions.insert(0, {"label": "🚨 Call Emergency (112 / 911 / 108)", "url": "/contact"})
                
            return jsonify({
                "reply": reply,
                "quick_replies": d_info['q1_chips'],
                "actions": actions,
                "confidence_badge": f"Triage Step 1/2 • {d_info['name']}",
                "is_emergency": is_emergency,
                "suggested_actions": actions,
                "follow_up_chips": d_info['q1_chips']
            })
            
        # Check LLM mode if configured
        last_diagnosis = session.get('last_diagnosis')
        llm_response = call_live_llm(user_message, current_page, page_context, last_diagnosis)
        if llm_response:
            is_emergency = any(trig in msg_lower for trig in EMERGENCY_TRIGGERS)
            actions = [{"label": "📄 Print Medical Report", "action": "print"}]
            if is_emergency:
                actions = [{"label": "🚨 Emergency Directory", "url": "/contact"}]
            chips = ["Explain my results", "What are my next steps?", "Questions for my doctor"]
            return jsonify({
                "reply": llm_response['reply'],
                "confidence_badge": llm_response['confidence_badge'],
                "is_emergency": is_emergency,
                "actions": actions,
                "quick_replies": chips,
                "suggested_actions": actions,
                "follow_up_chips": chips
            })
            
        # Fallback to local clinical copilot engine
        response_data = medinova_clinical_copilot_engine(user_message, current_page, page_context, last_diagnosis)
        if 'quick_replies' not in response_data:
            response_data['quick_replies'] = response_data.get('follow_up_chips', initial_chips)
        if 'actions' not in response_data:
            response_data['actions'] = response_data.get('suggested_actions', initial_actions)
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in /api/chat: {e}")
        return jsonify({
            "reply": f"An error occurred while analyzing your clinical request: {str(e)}",
            "confidence_badge": "System Alert",
            "is_emergency": False,
            "actions": [{"label": "🏠 Return to Home", "url": "/"}],
            "quick_replies": ["🫀 Chest pain or high BP", "🩸 High blood sugar advice", "🌡️ Fever or cough", "↺ Reset Consultation"]
        }), 500


@app.route('/api/chat/reset', methods=['POST'])
@app.route('/api/chat/clear', methods=['POST'])
def api_chat_reset():
    session['chat_state'] = {'stage': 'idle', 'topic': None, 'severity': None, 'associated_symptoms': None}
    session.pop('last_diagnosis', None)
    session.modified = True
    return jsonify({
        "status": "success",
        "message": "Chat session and diagnostic state reset successfully.",
        "reply": "I've reset our conversation. How can I help you today? Tell me how you're feeling or choose a situation below:",
        "quick_replies": [
            "🫀 Chest pain or high BP",
            "🩸 High blood sugar / Diabetes advice",
            "🌡️ Fever, chills, or persistent cough",
            "🧪 Kidney or Liver symptom check",
            "📊 Explain my latest test result"
        ],
        "actions": [
            {"label": "🫀 Heart Screening", "url": "/heart"},
            {"label": "🩸 Diabetes Screening", "url": "/diabetes"},
            {"label": "🩺 Kidney Screening", "url": "/kidney"},
            {"label": "🔍 All 7 Diagnostic Suites", "url": "/features"}
        ]
    })


# ============================================================================
# DEVELOPER-ONLY PORTAL & GOOGLE SHEETS AUDIT GATEWAY
# Strict Security Access: dharshankumarkumar3@gmail.com Only
# ============================================================================

def is_developer_authenticated() -> bool:
    """Check if current session has authenticated as the authorized developer."""
    return bool(
        session.get('dev_authenticated') is True and 
        session.get('dev_email', '').strip().lower() == AUTHORIZED_DEV_EMAIL
    )


@app.route('/developer/patient-records', methods=['GET', 'POST'])
def developer_patient_records():
    """
    Developer-only portal for inspecting Google Sheets synchronization status.
    Accessible only to dharshankumarkumar3@gmail.com with secret key or OTP.
    Zero patient visibility: Not exposed in any patient template.
    """
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)

    # 1. If already authenticated and GET request, render Dashboard view
    if request.method == 'GET':
        if is_developer_authenticated():
            stats = get_sheets_status()
            return render_template(
                'dev_access.html',
                is_authenticated=True,
                developer_email=session.get('dev_email'),
                stats=stats
            )
        return render_template('dev_access.html', is_authenticated=False, prefill_email=AUTHORIZED_DEV_EMAIL)

    # 2. Handle POST: Developer Authentication Gate
    email = request.form.get('email', '').strip().lower()
    auth_method = request.form.get('auth_method', 'key').strip().lower()
    access_key = request.form.get('access_key', '').strip()
    otp_entered = request.form.get('otp', '').strip()

    # Layer 1: Strict Developer Email Verification
    if email != AUTHORIZED_DEV_EMAIL:
        app.logger.warning(
            f"[SECURITY 403] Unauthorized access attempt blocked from IP {client_ip} with email '{email}'. "
            f"Authorized target is '{AUTHORIZED_DEV_EMAIL}'."
        )
        return render_template(
            'dev_access.html',
            error=f"403 Forbidden: Email '{email}' is not authorized. Access strictly reserved for Developer ({AUTHORIZED_DEV_EMAIL}).",
            error_code=403,
            is_authenticated=False,
            prefill_email=email
        ), 403

    # Layer 2: Verification of Key or OTP
    authenticated = False
    auth_failure_reason = ""

    if auth_method == 'otp' or otp_entered:
        # OTP verification
        cached_otp = DEV_OTP_CACHE.get('otp')
        expires_at = DEV_OTP_CACHE.get('expires_at')
        if not cached_otp or not expires_at:
            auth_failure_reason = "No active OTP found. Please click 'Send OTP' to request a new verification code."
        elif datetime.datetime.now() > expires_at:
            auth_failure_reason = "OTP has expired. Please request a new 6-digit code."
        elif otp_entered != cached_otp:
            auth_failure_reason = "Invalid OTP verification code. Please check your developer inbox or server log."
        else:
            authenticated = True
            DEV_OTP_CACHE['otp'] = None
            DEV_OTP_CACHE['expires_at'] = None
    else:
        # Access Key verification
        if access_key and access_key == DEV_ACCESS_KEY:
            authenticated = True
        else:
            auth_failure_reason = "Invalid Developer Master Access Key."

    if not authenticated:
        app.logger.warning(
            f"[SECURITY 403] Failed developer authentication attempt from IP {client_ip} for '{email}'. Reason: {auth_failure_reason}"
        )
        return render_template(
            'dev_access.html',
            error=f"403 Forbidden: {auth_failure_reason}",
            error_code=403,
            is_authenticated=False,
            prefill_email=email
        ), 403

    # Success: Establish authenticated developer session
    session['dev_authenticated'] = True
    session['dev_email'] = AUTHORIZED_DEV_EMAIL
    session['dev_login_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    app.logger.info(f"[SECURITY AUDIT] Developer '{AUTHORIZED_DEV_EMAIL}' successfully authenticated from IP {client_ip}.")

    return redirect(url_for('developer_patient_records'))


@app.route('/developer/request-otp', methods=['POST'])
def developer_request_otp():
    """Generates and sends a 6-digit OTP to AUTHORIZED_DEV_EMAIL."""
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    data = request.get_json(silent=True) or {}
    email = data.get('email', '').strip().lower()

    if email != AUTHORIZED_DEV_EMAIL:
        app.logger.warning(f"[SECURITY 403] OTP generation blocked for unauthorized email '{email}' from IP {client_ip}")
        return jsonify({"success": False, "message": "403 Forbidden: Email is not authorized for developer access."}), 403

    # Generate cryptographically secure 6-digit OTP
    otp_val = f"{secrets.randbelow(900000) + 100000}"
    DEV_OTP_CACHE['otp'] = otp_val
    DEV_OTP_CACHE['expires_at'] = datetime.datetime.now() + datetime.timedelta(minutes=10)
    DEV_OTP_CACHE['email'] = email

    print(f"\n====================================================================")
    print(f"[DEVELOPER SECURITY GATEWAY] Generated 6-digit OTP: {otp_val}")
    print(f"Target Developer: {AUTHORIZED_DEV_EMAIL}")
    print(f"Valid for: 10 minutes (Expires: {DEV_OTP_CACHE['expires_at'].strftime('%H:%M:%S')})")
    print(f"====================================================================\n")

    subject = f"[MediNova Developer Core] One-Time Verification Code: {otp_val}"
    body_html = f"""
    <div style="font-family: Arial, sans-serif; background: #0b1120; color: #f8fafc; padding: 24px; border-radius: 8px;">
        <h2 style="color: #38bdf8; margin-top: 0;">MediNova Developer Security Gate</h2>
        <p>You requested developer access to the MediNova Patient Diagnostic Records database.</p>
        <div style="background: #1e293b; padding: 18px; border-radius: 6px; text-align: center; margin: 20px 0;">
            <span style="font-size: 32px; font-weight: bold; letter-spacing: 6px; color: #38bdf8;">{otp_val}</span>
        </div>
        <p style="color: #94a3b8; font-size: 13px;">This code is valid for 10 minutes. If you did not request this code, no action is required.</p>
    </div>
    """
    email_sent, email_msg = send_developer_email(subject, body_html)
    if email_sent:
        msg = f"OTP successfully sent to {AUTHORIZED_DEV_EMAIL}."
    else:
        msg = f"OTP active! (Console log active: {otp_val}. Notice: {email_msg})"

    return jsonify({"success": True, "message": msg})


@app.route('/developer/logout', methods=['GET'])
def developer_logout():
    """Clear developer session and lock access."""
    session.pop('dev_authenticated', None)
    session.pop('dev_email', None)
    session.pop('dev_login_time', None)
    return redirect(url_for('developer_patient_records'))


@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)



