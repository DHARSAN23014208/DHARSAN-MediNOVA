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
import time
import threading
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
from utils.hospital_service import (
    DISEASE_SPECIALTY_MAP,
    CURATED_GLOBAL_CENTRES,
    get_specialty_mapping,
    discover_hospitals_for_patient,
    resolve_any_address,
    geocode_address,
    calculate_haversine_distance
)
from utils.appointment_db import (
    create_appointment,
    get_appointment_by_ref,
    get_appointment_by_token,
    list_appointments_admin,
    update_appointment_status,
    cancel_appointment_patient,
    VALID_STATUSES
)
from utils.calendar_service import (
    generate_google_calendar_url,
    generate_ics_content
)
from utils.appointment_notifier import notify_appointment_request
from utils.report_scanner import process_report_pipeline, SUPPORTED_MODULES, MAX_FILE_SIZE
from utils.report_verifier import (
    init_report_verifier_db,
    get_verified_report,
    generate_qr_code_image,
    register_report
)
from utils.diagnostic_predictor import (
    predict_heart,
    predict_diabetes,
    predict_kidney,
    predict_liver,
    predict_breast_cancer,
    predict_thyroid,
    predict_arrhythmia,
    generate_report_id
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

# Check environment and enforce secret strength in production
IS_PRODUCTION = (
    os.environ.get("FLASK_ENV") == "production"
    or os.environ.get("ENV") == "production"
    or os.environ.get("RENDER") == "true"
    or os.environ.get("PRODUCTION", "").lower() in ["true", "1", "yes"]
)

RAW_SECRET_KEY = os.environ.get('SECRET_KEY')
RAW_DEV_ACCESS_KEY = os.environ.get('DEV_ACCESS_KEY')
DEFAULT_INSECURE_SECRET = 'medinova-clinical-secret-key-2026'
DEFAULT_INSECURE_DEV_KEY = 'medinova_dev_2026_dharshan_secret'

if IS_PRODUCTION:
    if not RAW_SECRET_KEY or RAW_SECRET_KEY == DEFAULT_INSECURE_SECRET:
        raise RuntimeError("CRITICAL SECURITY ERROR: In production, SECRET_KEY must be set in environment variables and cannot use default value.")
    if not RAW_DEV_ACCESS_KEY or RAW_DEV_ACCESS_KEY == DEFAULT_INSECURE_DEV_KEY:
        raise RuntimeError("CRITICAL SECURITY ERROR: In production, DEV_ACCESS_KEY must be set in environment variables and cannot use default value.")

app = Flask(__name__)
app.secret_key = RAW_SECRET_KEY or DEFAULT_INSECURE_SECRET
init_report_verifier_db()

# Clinical Security & Session Hardening
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=IS_PRODUCTION,
    PERMANENT_SESSION_LIFETIME=datetime.timedelta(hours=4)
)

def get_csrf_token():
    """Generates or retrieves a cryptographically secure session-bound CSRF token."""
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_hex(32)
    return session['_csrf_token']

@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=get_csrf_token)

@app.after_request
def set_security_headers(response):
    """Applies strict clinical defense-in-depth security headers."""
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    
    csp = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    response.headers['Content-Security-Policy'] = csp
    if IS_PRODUCTION:
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

@app.before_request
def validate_csrf():
    """Validates session-bound CSRF token on state-changing requests."""
    if request.method in ('POST', 'PUT', 'DELETE', 'PATCH'):
        if app.config.get('TESTING'):
            return None
        exempt_prefixes = ('/api/scan-report', '/api/sample-report', '/api/appointments', '/api/patient-images', '/api/chat', '/api/report-qr')
        if any(request.path.startswith(p) for p in exempt_prefixes):
            return None
            
        token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
        expected_token = session.get('_csrf_token')
        if expected_token and token:
            if not secrets.compare_digest(token, expected_token):
                if request.is_json or 'application/json' in request.headers.get('Accept', ''):
                    return jsonify({'success': False, 'error': 'Security verification failed: Invalid CSRF token.'}), 403
                return render_template('home.html', error="Security verification failed: Invalid CSRF token."), 403

# Developer Master Gate Configuration (Strictly restricted to authorized developer email)
AUTHORIZED_DEV_EMAIL = os.environ.get('AUTHORIZED_DEV_EMAIL', 'dharshankumarkumar3@gmail.com').strip().lower()
DEV_ACCESS_KEY = RAW_DEV_ACCESS_KEY or DEFAULT_INSECURE_DEV_KEY

# Thread-safe in-memory OTP storage for developer two-factor authentication
DEV_OTP_LOCK = threading.Lock()
DEV_OTP_CACHE = {
    'otp': None,
    'expires_at': None,
    'email': None,
    'attempts': 0,
    'last_requested_at': 0,
    'max_attempts': 5
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
# Patient medical scans are stored privately outside static directory to preserve HIPAA-aligned privacy
PRIVATE_UPLOADS_DIR = os.path.join(BASE_DIR, 'private_uploads')
SAMPLES_FOLDER = os.path.join(BASE_DIR, 'static', 'samples')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(PRIVATE_UPLOADS_DIR, exist_ok=True)
os.makedirs(SAMPLES_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = PRIVATE_UPLOADS_DIR
app.config['SAMPLES_FOLDER'] = SAMPLES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max


def validate_and_save_patient_image(uploaded_file):
    """
    Validates uploaded patient image:
    1. Size <= 16MB.
    2. Extension in allowed list: png, jpg, jpeg, webp.
    3. Content validation via PIL.Image to ensure file is not corrupted, unsupported, or a polyglot.
    4. Saves to private_uploads directory (outside static/).
    5. Generates short-lived preview token bound to the current session.
    
    Returns:
        tuple: (filepath, preview_url, error_message)
    """
    if not uploaded_file or uploaded_file.filename == '':
        return None, None, "No file selected."

    filename = secure_filename(uploaded_file.filename)
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in {'png', 'jpg', 'jpeg', 'webp'}:
        return None, None, "Invalid image format. Please upload a PNG, JPG, JPEG, or WebP medical scan."

    file_bytes = uploaded_file.read()
    if len(file_bytes) > 16 * 1024 * 1024:
        return None, None, "File size exceeds 16 MB limit."

    # Validate image content with Pillow
    try:
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(file_bytes))
        img.verify()
        if img.format.lower() not in ['png', 'jpeg', 'webp']:
            return None, None, "Unsupported or corrupted medical image content."
    except Exception:
        return None, None, "The uploaded file is not a valid or readable medical image."

    # Generate secure random filename in private_uploads
    unique_name = f"{uuid.uuid4().hex}_{secrets.token_hex(4)}.{ext}"
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    try:
        with open(saved_path, 'wb') as f:
            f.write(file_bytes)
    except Exception as e:
        return None, None, f"Failed to save upload safely: {str(e)}"

    # Generate session-bound preview token
    token = secrets.token_urlsafe(32)
    preview_tokens = session.get('image_preview_tokens', {})
    if not isinstance(preview_tokens, dict):
        preview_tokens = {}
    preview_tokens[token] = {
        'filepath': saved_path,
        'created_at': time.time(),
        'mimetype': uploaded_file.mimetype or f"image/{ext}"
    }
    session['image_preview_tokens'] = preview_tokens

    preview_url = url_for('patient_image_preview', token=token)
    return saved_path, preview_url, None

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
thyroid_model = load_pkl_model('thyroid_model.pkl')
arrhythmia_model = load_pkl_model('ecg_arrhythmia_model.pkl')

def get_thyroid_model():
    """Lazy loader for thyroid dysfunction model."""
    global thyroid_model
    if thyroid_model is None:
        thyroid_model = load_pkl_model('thyroid_model.pkl')
    return thyroid_model

def get_arrhythmia_model():
    """Lazy loader for ECG arrhythmia model."""
    global arrhythmia_model
    if arrhythmia_model is None:
        arrhythmia_model = load_pkl_model('ecg_arrhythmia_model.pkl')
    return arrhythmia_model

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
    },
    'thyroid': {
        'positive': {
            'is_positive': True,
            'title': 'Serum Thyroid Dysfunction Detected (Hypothyroidism Marker)',
            'severity': 'Moderate Clinical Priority - Endocrine Follow-up Indicated',
            'badge': 'ATTENTION REQUIRED',
            'status_color': 'danger',
            'description': 'Laboratory biochemical panel analysis identified elevated serum TSH accompanied by suppressed or borderline free thyroxine (FT4/FTI) parameters, characteristic of primary or subclinical hypothyroidism.',
            'specialist': 'Board-Certified Endocrinologist & Thyroid Specialist',
            'urgency_window': 'Schedule clinical endocrine consultation within 7 to 10 days.',
            'risk_factors': [
                'Elevated Serum Thyroid-Stimulating Hormone (TSH > 4.5 mIU/L)',
                'Decreased Free Thyroxine Index (FTI < 85) or Total Thyroxine (TT4 < 60 nmol/L)',
                'Subnormal serum Triiodothyronine (T3 < 1.2 nmol/L)',
                'Prior history of thyroiditis, auto-antibodies, or thyroid surgery'
            ],
            'emergency_symptoms': [
                'Severe hypothermia, profound lethargy, confusion, or periorbital edema (Myxedema crisis)',
                'Severe bradycardia (heart rate < 45 bpm) or hemodynamic instability',
                'Acute onset of hoarseness, dysphagia, or rapidly enlarging anterior neck mass'
            ],
            'recommendations': [
                'Confirm results with a repeat venous Thyroid Panel (Serum TSH, Free T4, and Free T3)',
                'Test serum Anti-Thyroid Peroxidase (Anti-TPO) and Anti-Thyroglobulin antibodies to screen for Hashimoto thyroiditis',
                'Undergo high-resolution Thyroid Ultrasound if palpable goiter or nodules are present',
                'Consult an endocrinologist regarding appropriate Levothyroxine replacement titration'
            ],
            'doctor_questions': [
                'Does this panel reflect subclinical hypothyroidism or overt clinical thyroid failure?',
                'Should an autoimmune antibody panel (Anti-TPO / Anti-TG) be performed?',
                'Is daily Levothyroxine sodium therapy recommended at this stage?',
                'At what frequency should TSH levels be rechecked to monitor therapeutic titration?'
            ],
            'diet_dos': [
                'Incorporate adequate dietary iodine sources (iodized salt, sea vegetables, eggs, dairy)',
                'Consume selenium-rich foods (Brazil nuts, sunflower seeds, fish) to support deiodinase enzyme conversion',
                'Include zinc-rich proteins (lean poultry, pumpkin seeds, lentils) for thyroid hormone synthesis'
            ],
            'diet_donts': [
                'Avoid consuming raw cruciferous vegetables (broccoli, cabbage, kale) in large quantities without cooking',
                'Avoid taking iron or calcium supplements within 4 hours of thyroid medications',
                'Limit ultra-processed soy isolates which can interfere with thyroid hormone uptake'
            ],
            'recommended_tests': ['Repeat Serum TSH', 'Free T4 (FT4)', 'Free T3 (FT3)', 'Anti-TPO Antibodies', 'High-Resolution Thyroid Ultrasound']
        },
        'negative': {
            'is_positive': False,
            'title': 'Serum Thyroid Analytes Within Physiological Limits',
            'severity': 'Optimal Endocrine Profile - Euthyroid Status',
            'badge': 'OPTIMAL HEALTH',
            'status_color': 'safe',
            'description': 'Serum Thyroid-Stimulating Hormone (TSH), Total Thyroxine (TT4), and Free Thyroxine Index (FTI) values fall within recognized standard clinical laboratory reference intervals.',
            'specialist': 'Primary Care Physician / General Practitioner',
            'urgency_window': 'Routine annual wellness laboratory screening.',
            'tips': [
                'Maintain a balanced, nutrient-dense diet supporting endocrine homeostasis',
                'Stay active and get 7 to 8 hours of restorative sleep nightly to regulate hypothalamic-pituitary-thyroid axis',
                'Perform routine preventive wellness screenings if constitutional fatigue develops'
            ],
            'diet_dos': [
                'Balanced diet with adequate dietary iodine and micronutrients'
            ],
            'diet_donts': [
                'Avoid excessive iodine megadoses from unverified kelp supplements'
            ],
            'recommended_tests': ['Routine Annual Wellness Chemistry Panel']
        }
    },
    'arrhythmia': {
        'positive': {
            'is_positive': True,
            'title': 'Cardiac Arrhythmia / Ectopic Ventricular Activity Detected',
            'severity': 'High Clinical Priority - Cardiology Evaluation Advised',
            'badge': 'ARRHYTHMIA DETECTED',
            'status_color': 'danger',
            'description': 'Computerized digital biosignal analysis of single-lead ECG telemetry detected Premature Ventricular Contractions (PVCs) or ectopic rhythm disturbance with abnormal R-R interval coupling and altered QRS morphology.',
            'specialist': 'Cardiologist & Cardiac Electrophysiologist',
            'urgency_window': 'Schedule formal diagnostic cardiology appointment within 24 to 48 hours.',
            'risk_factors': [
                'Premature ectopic ventricular depolarizations (PVCs)',
                'Marked R-R interval variability and prematurity index',
                'Altered QRS complex width and elevated morphological skewness',
                'Underlying ischemic heart disease, electrolyte imbalance, or autonomic dysregulation'
            ],
            'emergency_symptoms': [
                'Sustained palpitations accompanied by dizziness, near-syncope, or loss of consciousness',
                'Acute onset of chest pressure, squeezing pain, or diaphoresis (cold sweats)',
                'Severe sudden dyspnea (shortness of breath) or acute confusion'
            ],
            'recommendations': [
                'Obtain a comprehensive standard clinical 12-lead Electrocardiogram (ECG)',
                'Wear a 24-48 hour ambulatory continuous Holter ECG monitor to quantify total ectopic burden',
                'Check basic serum metabolic panel for electrolyte anomalies (Serum Potassium, Magnesium, Calcium)',
                'Undergo a 2D Doppler Echocardiogram to evaluate left ventricular ejection fraction and structural heart health'
            ],
            'doctor_questions': [
                'What is my 24-hour total ventricular premature contraction burden percentage?',
                'Is there evidence of structural or ischemic substrate predisposing to arrhythmia?',
                'Are antiarrhythmic pharmacotherapies or catheter ablation indicated?',
                'Should I avoid specific triggers like caffeine, decongestants, or strenuous athletic exertion?'
            ],
            'diet_dos': [
                'Maintain optimal hydration and consume electrolyte-rich foods (bananas, spinach, sweet potatoes, coconut water)',
                'Incorporate magnesium-rich foods (pumpkin seeds, almonds, dark leafy greens, whole grains)',
                'Practice daily vagal tone optimization techniques (mindful breathing, meditation, stress management)'
            ],
            'diet_donts': [
                'Avoid high doses of caffeine, pre-workout supplements, and energy drinks containing taurine/guarana',
                'Strictly avoid recreational stimulants, excessive alcohol, and tobacco/nicotine products',
                'Avoid OTC decongestants containing pseudoephedrine or phenylephrine without doctor approval'
            ],
            'recommended_tests': ['12-Lead Diagnostic ECG', '24-Hour Ambulatory Holter Monitor', '2D Doppler Echocardiogram', 'Serum Potassium & Magnesium Panel', 'Exercise Stress ECG']
        },
        'negative': {
            'is_positive': False,
            'title': 'Regular Sinus Rhythm - No Ectopic Beats Detected',
            'severity': 'Normal Rhythm - Stable Physiological Electrophysiology',
            'badge': 'NORMAL SINUS RHYTHM',
            'status_color': 'safe',
            'description': 'Surface ECG biosignal evaluation demonstrates regular P-QRS-T complexes, uniform R-R intervals, and absence of significant ectopic or premature ventricular depolarizations during the recorded window.',
            'specialist': 'Primary Care Physician / Preventive Cardiologist',
            'urgency_window': 'Routine annual cardiovascular checkup.',
            'tips': [
                'Continue regular cardiovascular aerobic conditioning (150 minutes weekly)',
                'Manage lifestyle stress and maintain balanced electrolyte hydration',
                'If symptoms such as sudden fluttering or dizzy spells occur, request an on-demand rhythm strip'
            ],
            'diet_dos': [
                'Heart-healthy balanced diet with plenty of water and natural dietary electrolytes'
            ],
            'diet_donts': [
                'Avoid excessive consumption of energy drinks or high-stimulant dietary supplements'
            ],
            'recommended_tests': ['Routine Annual Wellness Physical & Resting ECG Checkup']
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
    prefill_condition = request.args.get('condition', '').strip().lower()
    prefill_result = request.args.get('result', '').strip()
    tracking_ref = request.args.get('track', '').strip().upper()
    return render_template(
        'contact.html',
        prefill_condition=prefill_condition,
        prefill_result=prefill_result,
        tracking_ref=tracking_ref
    )

@app.route('/api/hospitals/specialties', methods=['GET'])
def api_hospital_specialties():
    """Return canonical seven-disease specialty mappings and curated global specialty centres."""
    return jsonify({
        "status": "success",
        "specialties": DISEASE_SPECIALTY_MAP,
        "global_centres": CURATED_GLOBAL_CENTRES
    })

@app.route('/api/hospitals/geocode', methods=['GET'])
def api_hospital_geocode():
    """Universal address geocoding endpoint (Nominatim progressive + Photon backup)."""
    raw_query = request.args.get('query', '').strip()
    if not raw_query:
        return jsonify({"status": "error", "message": "Location search query is required."}), 400
    res = resolve_any_address(raw_query)
    if res and res.get("success"):
        return jsonify({"status": "success", "location": res})
    return jsonify({
        "status": "not_found",
        "message": res.get("error", "Location not found. Please refine your search or select a Quick Metro above.") if res else "Location not found."
    }), 404

@app.route('/api/hospitals/search', methods=['GET'])
def api_hospital_search():
    """
    Search healthcare facilities and specialist referral centres:
    Strictly filters nearby facilities within the selected radius (distanceKm <= radius_km).
    Never substitutes worldwide hospitals for local searches.
    """
    raw_condition = request.args.get('condition')
    if not raw_condition or not raw_condition.strip():
        return jsonify({
            "status": "error",
            "message": "Diagnostic condition is required. Please select a valid clinical condition."
        }), 400

    condition = raw_condition.strip().lower()
    spec_map = get_specialty_mapping(condition)
    if spec_map.get("is_unmapped_condition"):
        return jsonify({
            "status": "error",
            "message": f"Unsupported diagnostic condition '{condition}'. Please select a valid supported condition."
        }), 400

    # Coordinate validation
    lat_val = request.args.get('lat')
    lon_val = request.args.get('lon')
    lat = None
    lon = None

    if lat_val is not None or lon_val is not None:
        if lat_val is None or lon_val is None:
            return jsonify({
                "status": "error",
                "message": "Both latitude and longitude must be provided."
            }), 400
        try:
            lat = float(lat_val)
            lon = float(lon_val)
        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "Invalid coordinates format. Latitude and longitude must be numbers."
            }), 400

        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            return jsonify({
                "status": "error",
                "message": "Coordinates out of bounds. Latitude must be in [-90, 90] and longitude in [-180, 180]."
            }), 400

    query = request.args.get('query', '').strip()
    
    # Radius validation
    radius_val = request.args.get('radius_km', default=25)
    try:
        radius_km = int(radius_val)
    except (ValueError, TypeError):
        return jsonify({
            "status": "error",
            "message": "Invalid radius. Must be an integer between 1 and 100 km."
        }), 400

    if radius_km < 1 or radius_km > 100:
        return jsonify({
            "status": "error",
            "message": "Search radius must be between 1 km and 100 km."
        }), 400

    min_rating = request.args.get('min_rating', default=3.0, type=float)
    mode = request.args.get('mode', 'all').strip().lower()

    # If user provided coordinates or query, default include_global is False (strict nearby search)
    has_location_input = bool((lat is not None and lon is not None) or query)
    default_include_global = 'false' if has_location_input else 'true'
    include_global = request.args.get('include_global', default_include_global).lower() in ['true', '1', 'yes']

    try:
        results = discover_hospitals_for_patient(
            condition_id=condition,
            patient_lat=lat,
            patient_lon=lon,
            manual_query=query,
            radius_km=radius_km,
            min_rating=min_rating,
            preferred_mode=mode,
            include_global=include_global
        )
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"[HOSPITAL SEARCH ERROR]: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to discover facilities. Please refine location."
        }), 500

@app.route('/api/appointments/book', methods=['POST'])
def api_book_appointment():
    """
    Submit and validate a new priority specialist consultation appointment request.
    Creates persistent database record and sets initial status to 'Awaiting Hospital Confirmation'.
    """
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        if not data:
            return jsonify({"success": False, "message": "Invalid request: missing appointment payload."}), 400

        # Check for acute red-flag symptoms in clinical notes
        notes = str(data.get('clinical_notes', '')).lower()
        emergency_keywords = [
            "crushing chest pain", "cannot breathe", "severe dyspnea", "stroke", "loss of consciousness",
            "vomiting blood", "coughing blood", "anaphylaxis", "passed out", "seizure", "unresponsive"
        ]
        is_emergency = any(kw in notes for kw in emergency_keywords)
        if is_emergency:
            data['is_urgent'] = 1

        success, record, err = create_appointment(data)
        if not success:
            return jsonify({"success": False, "message": err}), 400

        # Dispatch notification to clinic care coordinator / developer
        email_sent, email_msg = notify_appointment_request(record, send_developer_email)

        # Generate Google Calendar and ICS links
        gcal_url = generate_google_calendar_url(record)
        ics_url = f"/api/appointments/{record['reference_id']}/ics"

        return jsonify({
            "success": True,
            "message": "Priority consultation request submitted successfully.",
            "appointment": record,
            "google_calendar_url": gcal_url,
            "ics_url": ics_url,
            "notification_dispatched": email_sent,
            "notification_notice": email_msg,
            "urgent_alert": is_emergency
        }), 201
    except Exception as e:
        app.logger.error(f"[BOOK APPOINTMENT ERROR]: {e}")
        return jsonify({"success": False, "message": "A system error occurred while processing your request. Please try again."}), 500

@app.route('/api/appointments/lookup', methods=['POST'])
def api_lookup_appointment():
    """
    Patient appointment lookup and status tracking.
    Enforces strict access control: requires Reference ID and matching Phone or Email.
    """
    data = request.get_json(silent=True) or {}
    ref_id = data.get('reference_id', '').strip().upper()
    contact = data.get('contact', '').strip()

    if not ref_id or not contact:
        return jsonify({"success": False, "message": "Reference ID and contact email/phone are both required."}), 400

    record = get_appointment_by_ref(ref_id, verification_contact=contact)
    if not record:
        return jsonify({"success": False, "message": "No matching appointment found with the provided Reference ID and contact details."}), 404

    gcal_url = generate_google_calendar_url(record)
    ics_url = f"/api/appointments/{record['reference_id']}/ics"

    return jsonify({
        "success": True,
        "appointment": record,
        "google_calendar_url": gcal_url,
        "ics_url": ics_url
    })

@app.route('/api/appointments/cancel', methods=['POST'])
def api_cancel_appointment():
    """Allows patient to cancel an appointment if in an eligible pending state."""
    data = request.get_json(silent=True) or {}
    ref_id = data.get('reference_id', '').strip().upper()
    contact = data.get('contact', '').strip()
    reason = data.get('reason', '').strip()

    if not ref_id or not contact:
        return jsonify({"success": False, "message": "Reference ID and contact verification are required."}), 400

    success, msg = cancel_appointment_patient(ref_id, verification_contact=contact, reason=reason)
    if success:
        return jsonify({"success": True, "message": msg})
    return jsonify({"success": False, "message": msg}), 400

@app.route('/api/appointments/<reference_id>/ics', methods=['GET'])
def api_appointment_ics(reference_id):
    """Download RFC 5545 compliant iCalendar (.ics) file for an appointment."""
    record = get_appointment_by_ref(reference_id)
    if not record:
        abort(404)
    ics_text = generate_ics_content(record)
    from flask import Response
    return Response(
        ics_text,
        mimetype="text/calendar",
        headers={"Content-Disposition": f"attachment; filename=appointment_{reference_id}.ics"}
    )

# --- Heart Disease ---
@app.route('/heart', methods=['GET', 'POST'])
def heart():
    if request.method == 'POST':
        is_json_req = bool(request.is_json or 'application/json' in request.headers.get('Accept', '') or (request.content_type and 'application/json' in request.content_type))
        try:
            req_data = request.get_json(silent=True) if request.is_json else None
            if not req_data:
                req_data = request.form.to_dict()

            if heart_model is None:
                err_msg = "Heart disease diagnostic model is not loaded or missing from server."
                if is_json_req:
                    return jsonify({"success": False, "error": err_msg}), 503
                return render_template('heart.html', error=err_msg)
            
            res = predict_heart(req_data, heart_model, DISEASE_INSIGHTS, is_scanner=False)
            
            # Normalize confidence, percent, risk_label, probability, and feature contributions
            conf_val = res.get("confidence_score") or res.get("confidence") or res.get("percent")
            risk_label = res.get("risk_label", res.get("prediction", "Evaluated"))
            prob_val = round(float(conf_val) / 100.0, 4) if conf_val is not None else None
            feat_contribs = res.get("feature_contributions", res.get("top_risk_factors", []))
            gauges = res.get("biomarker_gauges", [])

            # Safe validation for each biomarker gauge and feature contribution
            for g in gauges:
                if isinstance(g, dict):
                    g["percent"] = g.get("percent", 0) if isinstance(g, dict) else getattr(g, "percent", 0)
                    g["low"] = g.get("low", 0)
                    g["high"] = g.get("high", 0)
                    g["analyte"] = g.get("analyte", g.get("name", ""))

            for f in feat_contribs:
                if isinstance(f, dict):
                    f["percent"] = f.get("percent", 0) if isinstance(f, dict) else getattr(f, "percent", 0)
                    f["weight"] = f.get("weight", f.get("global_sensitivity_weight", f.get("percent", 10.0) / 100.0))
                    f["feature_name"] = f.get("feature_name", f.get("label", f.get("feature", "Clinical Indicator")))
                    f["clinical_context"] = f.get("clinical_context", f.get("benchmark", ""))

            session['last_diagnosis'] = to_python_type({
                'condition': 'Heart Disease',
                'route': '/heart',
                'prediction': res['prediction'],
                'risk_label': risk_label,
                'is_positive': res['is_positive'],
                'confidence_score': conf_val,
                'percent': conf_val,
                'probability': prob_val,
                'patient_summary': res['patient_summary'],
                'report_id': res['report_id'],
                'specialist': res['insights'].get('specialist'),
                'urgency_window': res['insights'].get('urgency_window'),
                'recommended_tests': res['insights'].get('recommended_tests', []),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            if is_json_req:
                return jsonify({
                    "success": True,
                    "prediction": res['prediction'],
                    "risk_label": risk_label,
                    "is_positive": res['is_positive'],
                    "confidence": conf_val,
                    "confidence_score": conf_val,
                    "percent": conf_val,
                    "probability": prob_val,
                    "patient_summary": res['patient_summary'],
                    "report_id": res['report_id'],
                    "biomarker_gauges": gauges,
                    "top_risk_factors": feat_contribs,
                    "feature_contributions": feat_contribs,
                    "insights": res['insights']
                }), 200

            return render_template(
                'heart.html',
                prediction=res['prediction'],
                insights=res['insights'],
                is_positive=res['is_positive'],
                confidence_score=conf_val,
                patient_summary=res['patient_summary'],
                report_id=res['report_id'],
                biomarker_gauges=gauges,
                top_risk_factors=feat_contribs
            )
        except Exception as e:
            app.logger.error(f"[HEART EVALUATION ERROR] {e}", exc_info=True)
            if is_json_req:
                return jsonify({"success": False, "error": str(e)}), 400
            return render_template('heart.html', error=str(e))
    return render_template('heart.html')

# --- Diabetes ---
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        try:
            if diabetes_model is None:
                return render_template('diabetes.html', error="Diabetes diagnostic model is not loaded or missing from server.")
            
            res = predict_diabetes(request.form.to_dict(), diabetes_model, DISEASE_INSIGHTS, is_scanner=False)
            
            session['last_diagnosis'] = to_python_type({
                'condition': 'Diabetes',
                'route': '/diabetes',
                'prediction': res['prediction'],
                'is_positive': res['is_positive'],
                'confidence_score': res['confidence_score'],
                'patient_summary': res['patient_summary'],
                'report_id': res['report_id'],
                'specialist': res['insights'].get('specialist'),
                'urgency_window': res['insights'].get('urgency_window'),
                'recommended_tests': res['insights'].get('recommended_tests', []),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return render_template(
                'diabetes.html',
                prediction=res['prediction'],
                insights=res['insights'],
                is_positive=res['is_positive'],
                confidence_score=res['confidence_score'],
                patient_summary=res['patient_summary'],
                report_id=res['report_id'],
                biomarker_gauges=res['biomarker_gauges'],
                top_risk_factors=res['top_risk_factors']
            )
        except Exception as e:
            return render_template('diabetes.html', error=str(e))
    return render_template('diabetes.html')

# --- Kidney Disease ---
@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    if request.method == 'POST':
        try:
            if kidney_model is None:
                return render_template('kidney.html', error="Kidney disease diagnostic model is not loaded or missing from server.")
            
            res = predict_kidney(request.form.to_dict(), kidney_model, DISEASE_INSIGHTS, is_scanner=False)
            
            session['last_diagnosis'] = to_python_type({
                'condition': 'Kidney Disease',
                'route': '/kidney',
                'prediction': res['prediction'],
                'is_positive': res['is_positive'],
                'confidence_score': res['confidence_score'],
                'patient_summary': res['patient_summary'],
                'report_id': res['report_id'],
                'specialist': res['insights'].get('specialist'),
                'urgency_window': res['insights'].get('urgency_window'),
                'recommended_tests': res['insights'].get('recommended_tests', []),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return render_template(
                'kidney.html',
                prediction=res['prediction'],
                insights=res['insights'],
                is_positive=res['is_positive'],
                confidence_score=res['confidence_score'],
                patient_summary=res['patient_summary'],
                report_id=res['report_id'],
                biomarker_gauges=res['biomarker_gauges'],
                top_risk_factors=res['top_risk_factors']
            )
        except Exception as e:
            return render_template('kidney.html', error=str(e))
    return render_template('kidney.html')

# --- Liver Disease ---
@app.route('/liver', methods=['GET', 'POST'])
def liver():
    if request.method == 'POST':
        try:
            if liver_model is None:
                return render_template('liver.html', error="Liver disease diagnostic model is not loaded or missing from server.")
            
            res = predict_liver(request.form.to_dict(), liver_model, DISEASE_INSIGHTS, is_scanner=False)
            
            session['last_diagnosis'] = to_python_type({
                'condition': 'Liver Disease',
                'route': '/liver',
                'prediction': res['prediction'],
                'is_positive': res['is_positive'],
                'confidence_score': res['confidence_score'],
                'patient_summary': res['patient_summary'],
                'report_id': res['report_id'],
                'specialist': res['insights'].get('specialist'),
                'urgency_window': res['insights'].get('urgency_window'),
                'recommended_tests': res['insights'].get('recommended_tests', []),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return render_template(
                'liver.html',
                prediction=res['prediction'],
                insights=res['insights'],
                is_positive=res['is_positive'],
                confidence_score=res['confidence_score'],
                patient_summary=res['patient_summary'],
                report_id=res['report_id'],
                biomarker_gauges=res['biomarker_gauges'],
                top_risk_factors=res['top_risk_factors']
            )
        except Exception as e:
            return render_template('liver.html', error=str(e))
    return render_template('liver.html')

# --- Breast Cancer ---
@app.route('/breast_cancer', methods=['GET', 'POST'])
def breast_cancer():
    if request.method == 'POST':
        try:
            if breast_cancer_model is None:
                return render_template('breast_cancer.html', error="Breast cancer diagnostic model is not loaded or missing from server.")
            
            res = predict_breast_cancer(request.form.to_dict(), breast_cancer_model, DISEASE_INSIGHTS, is_scanner=False)
            
            session['last_diagnosis'] = to_python_type({
                'condition': 'Breast Cancer',
                'route': '/breast_cancer',
                'prediction': res['prediction'],
                'is_positive': res['is_positive'],
                'confidence_score': res['confidence_score'],
                'patient_summary': res['patient_summary'],
                'report_id': res['report_id'],
                'specialist': res['insights'].get('specialist'),
                'urgency_window': res['insights'].get('urgency_window'),
                'recommended_tests': res['insights'].get('recommended_tests', []),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return render_template(
                'breast_cancer.html',
                prediction=res['prediction'],
                insights=res['insights'],
                is_positive=res['is_positive'],
                confidence_score=res['confidence_score'],
                patient_summary=res['patient_summary'],
                report_id=res['report_id'],
                biomarker_gauges=res['biomarker_gauges'],
                top_risk_factors=res['top_risk_factors']
            )
        except Exception as e:
            return render_template('breast_cancer.html', error=str(e))
    return render_template('breast_cancer.html')

# --- Thyroid Dysfunction ---
@app.route('/thyroid', methods=['GET', 'POST'])
def thyroid():
    if request.method == 'POST':
        is_json_req = bool(request.is_json or 'application/json' in request.headers.get('Accept', '') or (request.content_type and 'application/json' in request.content_type))
        try:
            req_data = request.get_json(silent=True) if request.is_json else None
            if not req_data:
                req_data = request.form.to_dict()

            t_model = get_thyroid_model()
            if t_model is None:
                err_msg = "Thyroid diagnostic model is not loaded or missing from server."
                if is_json_req:
                    return jsonify({"success": False, "error": err_msg}), 503
                return render_template('thyroid.html', error=err_msg)

            res = predict_thyroid(req_data, t_model, DISEASE_INSIGHTS, is_scanner=False)

            conf_val = res.get("confidence_score") or res.get("confidence") or res.get("percent")
            risk_label = res.get("risk_label", res.get("prediction", "Evaluated"))
            prob_val = res.get("probability")
            feat_contribs = res.get("feature_contributions", res.get("top_risk_factors", []))
            gauges = res.get("biomarker_gauges", [])

            session['last_diagnosis'] = to_python_type({
                'condition': 'Thyroid Dysfunction',
                'route': '/thyroid',
                'prediction': res['prediction'],
                'risk_label': risk_label,
                'is_positive': res['is_positive'],
                'confidence_score': conf_val,
                'percent': conf_val,
                'probability': prob_val,
                'patient_summary': res['patient_summary'],
                'report_id': res['report_id'],
                'specialist': res['insights'].get('specialist'),
                'urgency_window': res['insights'].get('urgency_window'),
                'recommended_tests': res['insights'].get('recommended_tests', []),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            if is_json_req:
                return jsonify({
                    "success": True,
                    "prediction": res['prediction'],
                    "risk_label": risk_label,
                    "is_positive": res['is_positive'],
                    "confidence": conf_val,
                    "confidence_score": conf_val,
                    "percent": conf_val,
                    "probability": prob_val,
                    "patient_summary": res['patient_summary'],
                    "report_id": res['report_id'],
                    "biomarker_gauges": gauges,
                    "top_risk_factors": feat_contribs,
                    "feature_contributions": feat_contribs,
                    "insights": res['insights']
                }), 200

            return render_template(
                'thyroid.html',
                prediction=res['prediction'],
                insights=res['insights'],
                is_positive=res['is_positive'],
                confidence_score=conf_val,
                patient_summary=res['patient_summary'],
                report_id=res['report_id'],
                biomarker_gauges=gauges,
                top_risk_factors=feat_contribs
            )
        except Exception as e:
            app.logger.error(f"[THYROID EVALUATION ERROR] {e}", exc_info=True)
            if is_json_req:
                return jsonify({"success": False, "error": str(e)}), 400
            return render_template('thyroid.html', error=str(e))
    return render_template('thyroid.html')



# ============================================================================
# SMART MEDICAL LAB REPORT SCANNER & VERIFICATION APIS
# ============================================================================

@app.route('/api/sample-report/<sample_type>', methods=['GET'])
@app.route('/api/sample-report/<module_name>/<sample_type>', methods=['GET'])
def get_sample_report(sample_type, module_name=None):
    """
    Dedicated GET endpoint serving realistic clinical laboratory report PDFs.
    Returns 200 OK with Content-Type: application/pdf.
    """
    clean_type = 'positive' if 'pos' in sample_type.lower() else 'negative'
    samples_dir = os.path.join(BASE_DIR, 'static', 'samples')
    
    candidates = []
    if module_name:
        candidates.append(f"{module_name}_sample_{clean_type}.pdf")
    candidates.append(f"sample_{clean_type}.pdf")
    candidates.append(f"diabetes_sample_{clean_type}.pdf")
    
    for candidate in candidates:
        pdf_path = os.path.join(samples_dir, candidate)
        if os.path.isfile(pdf_path):
            return send_file(pdf_path, mimetype='application/pdf', as_attachment=False)
            
    return jsonify({"error": f"Sample report '{sample_type}' not found"}), 404


@app.route('/api/scan-report/<module_name>', methods=['POST'])
def scan_report(module_name):
    """
    Unified Smart Medical Lab Report Scanner Endpoint.
    Accepts image/PDF uploads, runs 3-tier extraction, returns structured metadata,
    and optionally executes validated instant prediction.
    """
    if module_name not in SUPPORTED_MODULES:
        return jsonify({
            'success': False,
            'error': f"Unsupported diagnostic module: '{module_name}'. Allowed: {sorted(list(SUPPORTED_MODULES))}"
        }), 400

    uploaded_file = request.files.get('report_file') or request.files.get('file')
    if not uploaded_file:
        return jsonify({'success': False, 'error': "No report file provided in upload request."}), 400

    raw_filename = uploaded_file.filename or ''
    clean_filename = secure_filename(raw_filename)
    if not clean_filename:
        clean_filename = f"report_upload_{uuid.uuid4().hex[:6]}.pdf"

    allowed_exts = {'.jpg', '.jpeg', '.png', '.webp', '.pdf'}
    _, ext = os.path.splitext(clean_filename.lower())
    if ext not in allowed_exts:
        return jsonify({
            'success': False,
            'error': f"Unsupported file type '{ext}'. Please upload an image (.jpg, .png, .webp) or PDF report."
        }), 400

    file_bytes = uploaded_file.read()
    if not file_bytes:
        return jsonify({'success': False, 'error': "Uploaded file is empty (0 bytes)."}), 400

    if len(file_bytes) > MAX_FILE_SIZE:
        max_mb = MAX_FILE_SIZE // (1024 * 1024)
        return jsonify({'success': False, 'error': f"File size exceeds maximum upload limit of {max_mb} MB."}), 413

    mode = request.form.get('mode', 'autofill').strip().lower()

    # Execute modular extraction pipeline
    scan_res = process_report_pipeline(
        file_bytes=file_bytes,
        filename=clean_filename,
        module_name=module_name,
        mime_type=uploaded_file.mimetype or 'application/octet-stream'
    )

    if not scan_res.get('success'):
        return jsonify(scan_res), 400

    # If instant predict requested and module input policy permits:
    if mode == 'instant_predict':
        if not scan_res.get('can_predict'):
            return jsonify(scan_res), 200

        model_map = {
            'heart': (heart_model, predict_heart),
            'diabetes': (diabetes_model, predict_diabetes),
            'kidney': (kidney_model, predict_kidney),
            'liver': (liver_model, predict_liver),
            'breast_cancer': (breast_cancer_model, predict_breast_cancer),
            'thyroid': (get_thyroid_model(), predict_thyroid)
        }

        m_inst, p_func = model_map[module_name]
        if m_inst is None:
            return jsonify({
                'success': False,
                'error': f"Diagnostic model for {module_name} is currently not loaded on server."
            }), 503

        try:
            pred_res = p_func(scan_res['values'], m_inst, DISEASE_INSIGHTS, is_scanner=True)
            scan_res['prediction'] = pred_res['prediction']
            scan_res['risk_label'] = pred_res.get('risk_label', pred_res['prediction'])
            scan_res['is_positive'] = pred_res['is_positive']
            conf_val = pred_res.get("confidence_score") or pred_res.get("confidence") or pred_res.get("percent")
            scan_res['confidence'] = conf_val
            scan_res['confidence_score'] = conf_val
            scan_res['percent'] = conf_val
            scan_res['probability'] = round(float(conf_val) / 100.0, 4) if conf_val is not None else None
            scan_res['report_id'] = pred_res['report_id']
            scan_res['insights'] = pred_res['insights']
            scan_res['biomarker_gauges'] = pred_res['biomarker_gauges']
            scan_res['top_risk_factors'] = pred_res['top_risk_factors']
            scan_res['feature_contributions'] = pred_res.get('feature_contributions', pred_res['top_risk_factors'])
            scan_res['patient_summary'] = pred_res['patient_summary']
        except Exception as e:
            app.logger.error(f"[SCANNER INSTANT PREDICT ERROR] {e}", exc_info=True)
            scan_res['warnings'].append(f"Instant prediction could not complete: {str(e)}")

    return jsonify(scan_res), 200


@app.route('/api/patient-images/preview/<token>', methods=['GET'])
def patient_image_preview(token):
    """
    Protected preview mechanism for patient-uploaded medical images.
    Enforces session isolation: only the session that uploaded the image can preview it.
    Tokens expire after 30 minutes.
    """
    preview_tokens = session.get('image_preview_tokens', {})
    if not isinstance(preview_tokens, dict) or token not in preview_tokens:
        abort(404)

    token_info = preview_tokens[token]
    filepath = token_info.get('filepath')
    created_at = token_info.get('created_at', 0)

    # Check 30-minute expiry
    if time.time() - created_at > 1800:
        abort(404)

    if not filepath or not os.path.exists(filepath):
        abort(404)

    return send_file(filepath, mimetype=token_info.get('mimetype', 'image/png'))


@app.route('/api/report-qr/<report_id>', methods=['GET'])
def report_qr_code(report_id):
    """Dynamically generates high-resolution PNG QR code for verification."""
    clean_id = secure_filename(report_id)
    public_base = os.environ.get('PUBLIC_BASE_URL', '').strip()
    if not public_base:
        public_base = request.host_url.rstrip('/')
    qr_png = generate_qr_code_image(clean_id, public_base)
    from io import BytesIO
    return send_file(BytesIO(qr_png), mimetype='image/png')


@app.route('/verify-report', methods=['GET'])
@app.route('/verify-report/<report_id>', methods=['GET'])
def verify_report_page(report_id=None):
    """Official public verification page for authenticating clinical screening reports."""
    raw_id = report_id or request.args.get('id') or request.args.get('report_id') or request.args.get('verify') or ''
    clean_id = secure_filename(raw_id).strip()
    if not clean_id:
        return render_template('verify_report.html', verified=False, query_id="None"), 404
    record = get_verified_report(clean_id)
    if record:
        return render_template('verify_report.html', verified=True, report=record)
    return render_template('verify_report.html', verified=False, query_id=clean_id), 404

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
                
                filepath, uploaded_url, val_err = validate_and_save_patient_image(file)
                if val_err:
                    return render_template('malaria.html', error=val_err)
                filename = os.path.basename(filepath)

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

            report_id = generate_report_id("MALA")
            register_report(
                report_id=report_id,
                module_name="malaria",
                condition_title="Malaria Microscopic Vision Screening",
                prediction_text=result,
                is_positive=is_pos,
                confidence_score=confidence,
                is_scanner_extracted=False,
                summary_data={"patient_summary": patient_summary}
            )

            session['last_diagnosis'] = to_python_type({
                'condition': 'Malaria',
                'route': '/malaria',
                'prediction': result,
                'is_positive': is_pos,
                'confidence_score': confidence,
                'patient_summary': patient_summary,
                'report_id': report_id,
                'specialist': insights.get('specialist'),
                'urgency_window': insights.get('urgency_window'),
                'recommended_tests': insights.get('recommended_tests', []),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            return render_template('malaria.html', prediction=result, insights=insights, is_positive=is_pos, 
                                   patient_summary=patient_summary, uploaded_image_url=uploaded_url, report_id=report_id)
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
                
                filepath, uploaded_url, val_err = validate_and_save_patient_image(file)
                if val_err:
                    return render_template('pneumonia.html', error=val_err)
                filename = os.path.basename(filepath)

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

            report_id = generate_report_id("PNEU")
            register_report(
                report_id=report_id,
                module_name="pneumonia",
                condition_title="Pneumonia Chest Radiography",
                prediction_text=result,
                is_positive=is_pos,
                confidence_score=confidence,
                is_scanner_extracted=False,
                summary_data={"patient_summary": patient_summary}
            )

            session['last_diagnosis'] = to_python_type({
                'condition': 'Pneumonia',
                'route': '/pneumonia',
                'prediction': result,
                'is_positive': is_pos,
                'confidence_score': confidence,
                'patient_summary': patient_summary,
                'report_id': report_id,
                'specialist': insights.get('specialist'),
                'urgency_window': insights.get('urgency_window'),
                'recommended_tests': insights.get('recommended_tests', []),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            return render_template('pneumonia.html', prediction=result, insights=insights, is_positive=is_pos,
                                   patient_summary=patient_summary, uploaded_image_url=uploaded_url, report_id=report_id)
        except Exception as e:
            return render_template('pneumonia.html', error=str(e))
    return render_template('pneumonia.html')

# --- ECG Arrhythmia Biosignal Screening ---
@app.route('/arrhythmia', methods=['GET', 'POST'])
def arrhythmia():
    if request.method == 'POST':
        is_json_req = bool(request.is_json or 'application/json' in request.headers.get('Accept', '') or (request.content_type and 'application/json' in request.content_type))
        try:
            a_model = get_arrhythmia_model()
            if a_model is None:
                err_msg = "ECG Arrhythmia diagnostic model artifact is not loaded or missing from server."
                if is_json_req:
                    return jsonify({"success": False, "error": err_msg}), 503
                return render_template('arrhythmia.html', error=err_msg)

            signal_data = None
            sample_rate = 360.0

            demo_type = request.form.get('demo_sample') if not request.is_json else None
            if not demo_type and request.is_json:
                req_json = request.get_json(silent=True) or {}
                demo_type = req_json.get('demo_sample')

            if demo_type in ('normal_sinus', 'mitbih_100'):
                demo_path = os.path.join(BASE_DIR, 'datasets', 'mitbih', '100.dat')
                if os.path.exists(demo_path):
                    import wfdb
                    rec = wfdb.rdrecord(os.path.join(BASE_DIR, 'datasets', 'mitbih', '100'), sampto=3600)
                    signal_data = rec.p_signal[:, 0].tolist()
                    sample_rate = float(rec.fs)
            elif demo_type in ('pvc_arrhythmia', 'mitbih_119'):
                demo_path = os.path.join(BASE_DIR, 'datasets', 'mitbih', '119.dat')
                if os.path.exists(demo_path):
                    import wfdb
                    rec = wfdb.rdrecord(os.path.join(BASE_DIR, 'datasets', 'mitbih', '119'), sampto=3600)
                    signal_data = rec.p_signal[:, 0].tolist()
                    sample_rate = float(rec.fs)

            if signal_data is None:
                if 'ecg_file' in request.files and request.files['ecg_file'].filename:
                    f = request.files['ecg_file']
                    content = f.read().decode('utf-8', errors='ignore')
                    signal_data = content
                elif request.is_json:
                    req_json = request.get_json(silent=True) or {}
                    signal_data = req_json.get('signal') or req_json.get('signal_data')
                    sample_rate = float(req_json.get('sample_rate', 360.0))
                else:
                    signal_data = request.form.get('signal_data') or request.form.get('signal')

            if not signal_data:
                err_msg = "Please provide an ECG signal recording (upload CSV/TXT, paste telemetry data, or select a demo MIT-BIH recording)."
                if is_json_req:
                    return jsonify({"success": False, "error": err_msg}), 400
                return render_template('arrhythmia.html', error=err_msg)

            res = predict_arrhythmia(signal_data, a_model, DISEASE_INSIGHTS, sample_rate=sample_rate)

            session['last_diagnosis'] = to_python_type({
                'condition': 'ECG Arrhythmia',
                'route': '/arrhythmia',
                'prediction': res['prediction'],
                'risk_label': res['risk_label'],
                'is_positive': res['is_positive'],
                'confidence_score': res['confidence_score'],
                'patient_summary': res['patient_summary'],
                'report_id': res['report_id'],
                'specialist': res['insights'].get('specialist'),
                'urgency_window': res['insights'].get('urgency_window'),
                'recommended_tests': res['insights'].get('recommended_tests', []),
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            if is_json_req:
                return jsonify({
                    "success": True,
                    "prediction": res['prediction'],
                    "risk_label": res['risk_label'],
                    "is_positive": res['is_positive'],
                    "confidence_score": res['confidence_score'],
                    "patient_summary": res['patient_summary'],
                    "report_id": res['report_id'],
                    "biomarker_gauges": res['biomarker_gauges'],
                    "arrhythmic_beat_count": res['arrhythmic_beat_count'],
                    "total_beats": res['total_beats'],
                    "rhythm_strip_samples": res['rhythm_strip_samples'],
                    "r_peak_indices": res['r_peak_indices'],
                    "top_risk_factors": res['top_risk_factors'],
                    "insights": res['insights']
                }), 200

            return render_template(
                'arrhythmia.html',
                prediction=res['prediction'],
                insights=res['insights'],
                is_positive=res['is_positive'],
                confidence_score=res['confidence_score'],
                patient_summary=res['patient_summary'],
                report_id=res['report_id'],
                biomarker_gauges=res['biomarker_gauges'],
                arrhythmic_beat_count=res['arrhythmic_beat_count'],
                total_beats=res['total_beats'],
                rhythm_strip_samples=json.dumps(res['rhythm_strip_samples']),
                r_peak_indices=json.dumps(res['r_peak_indices']),
                top_risk_factors=res['top_risk_factors']
            )
        except Exception as e:
            app.logger.error(f"[ARRHYTHMIA EVALUATION ERROR] {e}", exc_info=True)
            if is_json_req:
                return jsonify({"success": False, "error": str(e)}), 400
            return render_template('arrhythmia.html', error=str(e))
    return render_template('arrhythmia.html')

# --- Model Governance, Monitoring, & Research Gating Endpoints ---
@app.route('/api/catalog', methods=['GET'])
def api_catalog():
    """Returns verified model catalog with real provenance and evaluation benchmarks."""
    from utils.model_registry import get_catalog_summary
    return jsonify(get_catalog_summary()), 200

@app.route('/api/monitoring', methods=['GET'])
def api_monitoring():
    """Returns runtime telemetry, latency, error rates, and system resource metrics."""
    from utils.model_monitoring import get_system_telemetry
    return jsonify(get_system_telemetry()), 200

@app.route('/diabetic_retinopathy', methods=['GET', 'POST'])
@app.route('/melanoma', methods=['GET', 'POST'])
@app.route('/tuberculosis', methods=['GET', 'POST'])
@app.route('/anemia', methods=['GET', 'POST'])
def roadmap_gated_modules():
    """Governed research-preview gate: real inference is disabled until clinical validation."""
    module_name = request.path.strip('/')
    if request.method == 'POST' or request.is_json:
        return jsonify({
            "success": False,
            "status": "research_preview",
            "active": False,
            "condition": module_name,
            "error": f"The '{module_name}' module is currently under governed clinical research. Active screening inference is restricted until dataset acquisition and clinical protocol validation are completed."
        }), 403
    return render_template('home.html', research_preview_notice=f"Module '{module_name}' is currently in governed research roadmap. Clinical inference is disabled.")




# ============================================================================

# MEDINOVA AI CLINICAL COPILOT ENGINE (DUAL-ENGINE ARCHITECTURE)

# ============================================================================

from utils.clinical_copilot import (

    EMERGENCY_TRIGGERS,

    MEDICAL_EXPLAINERS,

    CLINICAL_DOMAINS,

    EXTENDED_SYMPTOMS_KB,

    MEDICATIONS_KB,

    GENERAL_CONVERSATION_RESPONSES,

    call_live_llm,

    medinova_clinical_copilot_engine,

    detect_clinical_domain,
    is_clinical_query,

    extract_form_auto_fill,

    analyze_patient_biomarkers,

    parse_num,

    generate_situation_action_plan as copilot_generate_action_plan,

    handle_explain_test_result as copilot_explain_test_result

)



def generate_situation_action_plan(domain_key, severity, associated_symptoms):

    """Wrapper ensuring backward compatibility with DISEASE_INSIGHTS in app.py."""

    return copilot_generate_action_plan(domain_key, severity, associated_symptoms, disease_insights=DISEASE_INSIGHTS)



def handle_explain_test_result(last_diagnosis):

    """Wrapper ensuring backward compatibility with last_diagnosis in app.py."""

    return copilot_explain_test_result(last_diagnosis)





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

            '🫀 Chest pain or high BP',

            '🩸 High blood sugar advice',

            '🦋 Thyroid or fatigue check',

            '⚡ Heart rhythm or palpitations',

            '🧪 Kidney or Liver symptom check',

            '📊 Explain my latest test result'

        ]

        initial_actions = [

            {'label': '🫀 Heart Screening', 'url': '/heart'},

            {'label': '🩸 Diabetes Screening', 'url': '/diabetes'},

            {'label': '🦋 Thyroid Screening', 'url': '/thyroid'},

            {'label': '⚡ ECG Arrhythmia', 'url': '/arrhythmia'},

            {'label': '🔍 All 9 Suites', 'url': '/features'}

        ]

        def respond(d, code=200):
            if isinstance(d, dict):
                d.setdefault('success', True)
                if 'reply' in d and 'response' not in d:
                    d['response'] = d['reply']
                elif 'response' in d and 'reply' not in d:
                    d['reply'] = d['response']
                if 'is_medical' not in d:
                    d['is_medical'] = is_clinical_query(user_message)
                if 'domain' not in d:
                    dom = detect_clinical_domain(user_message)
                    if dom and dom in CLINICAL_DOMAINS:
                        d['domain'] = CLINICAL_DOMAINS[dom]['name'].split('&')[0].strip()
                    else:
                        d['domain'] = 'Clinical Guidance' if d['is_medical'] else 'General Conversation'
                if 'quick_replies' not in d:
                    d['quick_replies'] = d.get('follow_up_chips', initial_chips)
                if 'actions' not in d:
                    d['actions'] = d.get('suggested_actions', initial_actions)
                if 'is_emergency' not in d:
                    d['is_emergency'] = any(trig in msg_lower for trig in EMERGENCY_TRIGGERS)
            return jsonify(d), code




        if not user_message:

            return respond({

                'reply': "Hi! I'm your **MediNova Clinical AI Advisor & Medical Copilot**. Tell me how you're feeling, ask any medical or general question, or choose a situation below—I'm here to provide evidence-based guidance and assist you step-by-step.",

                'quick_replies': initial_chips,

                'actions': initial_actions,

                'confidence_badge': 'Clinical Advisor Online',

                'is_emergency': False,

                'suggested_actions': initial_actions,

                'follow_up_chips': initial_chips

            })



        msg_lower = user_message.lower()



        # 1. Reset / restart check

        if msg_lower in ['reset', 'restart', 'clear', 'start over', '↺ reset consultation', 'reset chat', 'start new']:

            session['chat_state'] = {'stage': 'idle', 'topic': None, 'severity': None, 'associated_symptoms': None}

            session.pop('last_diagnosis', None)

            session['chat_history'] = []

            session.modified = True

            return respond({

                'status': 'success',

                'message': 'Chat session and diagnostic state reset successfully.',

                'reply': 'I have reset our consultation. How are you feeling right now? Pick a symptom or situation below, or describe how you feel in your own words:',

                'quick_replies': initial_chips,

                'actions': initial_actions,

                'confidence_badge': 'Advisor Reset',

                'is_emergency': False,

                'suggested_actions': initial_actions,

                'follow_up_chips': initial_chips

            })



        # 2. Check Explain my test result
        is_explain_query = any(k in msg_lower for k in [
            'explain my result', 'explain my test', 'explain my diagnosis', 'explain test result',
            'explain latest test', 'explain my latest test', 'explain results', 'my test result',
            'what caused my result', 'biomarker breakdown', 'biomarker analysis'
        ])

        if is_explain_query:
            last_diag = session.get('last_diagnosis')
            resp = handle_explain_test_result(last_diag)
            if isinstance(resp, dict):
                resp.setdefault('success', True)
                resp.setdefault('is_medical', True)
                resp.setdefault('domain', last_diag.get('condition', 'Diagnosis Breakdown') if last_diag else 'Diagnosis Breakdown')
            return respond(resp)



        # 3. Emergency Red-Flag Protocol (Preempts all conversational turns)
        if any(trig in msg_lower for trig in EMERGENCY_TRIGGERS):
            return respond(medinova_clinical_copilot_engine(user_message, current_page, page_context, session.get('last_diagnosis')))

        # 4. Retrieve conversation state

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



            reply = f"### 🩺 MediNova Clinical AI Advisor\n**Noted Severity & Duration:** *\"{user_message}\"*\n\nThank you for sharing that important context. To check for critical complications or red-flag warning signs:\n\n👉 **Follow-Up Question 2 of 2:**\n{d_info['q2_text']}"

            actions = [{'label': d_info['action_label'], 'url': d_info['route']}]

            if is_emergency:

                actions.insert(0, {'label': '🚨 Call 112 / 911 / 108 Immediately', 'url': '/contact'})



            return respond({

                'reply': reply,

                'quick_replies': d_info['q2_chips'],

                'actions': actions,

                'confidence_badge': f"Triage Step 2/2 • {d_info['name']}",

                'is_emergency': is_emergency,

                'suggested_actions': actions,

                'follow_up_chips': d_info['q2_chips']

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

            return respond(plan_response)



        # 4. Check Emergency Red Flags directly

        is_emergency = any(trig in msg_lower for trig in EMERGENCY_TRIGGERS)

        if is_emergency:

            return respond(medinova_clinical_copilot_engine(user_message, current_page, page_context, session.get('last_diagnosis')))



        # 5. Check if user explicitly presents clinical domain symptoms

        matched_domain = detect_clinical_domain(user_message)

        is_symptom_presentation = any(w in msg_lower for w in [

            'i have', 'i feel', 'suffering from', 'experiencing', 'my symptoms are', 'pain in my',

            'started having', 'high bp', 'high sugar', 'chest pain', 'palpitations', 'chills',

            'coughing', 'jaundice', 'swelling', 'lump in'

        ])



        # If user directly asks for diet or nutrition

        if any(k in msg_lower for k in ['diet', 'food', 'eat', 'avoid', 'doctor questions', 'nutrition']) and matched_domain:

            plan_response = generate_situation_action_plan(matched_domain, 'Routine Dietary Inquiry', 'General Guidance')

            return respond(plan_response)



        # If user presents symptoms and wants triage questionnaire

        if matched_domain and (is_symptom_presentation or any(chip.lower() in msg_lower for chip in initial_chips)):

            d_info = CLINICAL_DOMAINS[matched_domain]

            session['chat_state'] = {

                'stage': 'turn1_asked',

                'topic': matched_domain,

                'severity': None,

                'associated_symptoms': None

            }

            session.modified = True



            reply = f"### 🩺 MediNova Clinical AI Advisor\n**Symptom Domain Identified:** {d_info['name']}\n\nI hear your concern, and I will help you handle this safely and step-by-step. To recommend the right immediate first-aid, safe diet protocol, and matching MediNova AI screening:\n\n👉 **Follow-Up Question 1 of 2:**\n{d_info['q1_text']}"

            actions = [{'label': d_info['action_label'], 'url': d_info['route']}]

            return respond({

                'reply': reply,

                'quick_replies': d_info['q1_chips'],

                'actions': actions,

                'confidence_badge': f"Triage Step 1/2 • {d_info['name']}",

                'is_emergency': False,

                'suggested_actions': actions,

                'follow_up_chips': d_info['q1_chips']

            })



        # 6. Check LLM Mode (Gemini 3.8 Flash / OpenAI)
        last_diagnosis = session.get('last_diagnosis')
        chat_history = session.get('chat_history', [])
        llm_response = call_live_llm(user_message, current_page, page_context, last_diagnosis, chat_history)
        if llm_response:
            if not isinstance(chat_history, list):
                chat_history = []
            chat_history.append({'role': 'user', 'content': user_message})
            chat_history.append({'role': 'model', 'content': llm_response['reply']})
            session['chat_history'] = chat_history[-8:]
            session.modified = True

            actions = [{'label': '🔍 Explore 9 Suites', 'url': '/features'}]
            if any(k in msg_lower for k in ['heart', 'cardiac', 'chest', 'bp']):
                actions.insert(0, {'label': '🫀 Heart Screening', 'url': '/heart'})
            elif any(k in msg_lower for k in ['diabetes', 'sugar', 'glucose', 'hba1c']):
                actions.insert(0, {'label': '🩸 Diabetes Screening', 'url': '/diabetes'})
            elif any(k in msg_lower for k in ['thyroid', 'tsh', 't3', 't4']):
                actions.insert(0, {'label': '🦋 Thyroid Screening', 'url': '/thyroid'})
            elif any(k in msg_lower for k in ['ecg', 'arrhythmia', 'palpitation']):
                actions.insert(0, {'label': '⚡ ECG Arrhythmia', 'url': '/arrhythmia'})
            elif any(k in msg_lower for k in ['kidney', 'creatinine', 'urea']):
                actions.insert(0, {'label': '🩺 Kidney Screening', 'url': '/kidney'})
            elif any(k in msg_lower for k in ['liver', 'bilirubin', 'jaundice']):
                actions.insert(0, {'label': '🧬 Liver Screening', 'url': '/liver'})
            elif any(k in msg_lower for k in ['doctor', 'hospital', 'appointment', 'specialist', 'clinic']):
                actions.insert(0, {'label': '🏥 Book Hospital Appointment', 'url': '/contact'})

            chips = ['Explain my results', 'Normal biomarker reference ranges', 'What are my next steps?', 'Check another symptom']
            is_med = is_clinical_query(user_message)
            return respond({
                'success': True,
                'reply': llm_response['reply'],
                'confidence_badge': llm_response['confidence_badge'],
                'is_emergency': False,
                'is_medical': is_med,
                'domain': detect_clinical_domain(user_message) or ('Clinical Guidance' if is_med else 'General Conversation'),
                'actions': actions,
                'quick_replies': chips,
                'suggested_actions': actions,
                'follow_up_chips': chips
            })

        # 7. Fallback to Local Comprehensive Clinical & Conversational Engine
        response_data = medinova_clinical_copilot_engine(user_message, current_page, page_context, last_diagnosis)
        if isinstance(response_data, dict):
            response_data.setdefault('success', True)
            if 'is_medical' not in response_data:
                response_data['is_medical'] = is_clinical_query(user_message)
            if 'domain' not in response_data:
                response_data['domain'] = detect_clinical_domain(user_message) or ('Clinical Guidance' if response_data['is_medical'] else 'General Conversation')
            if 'quick_replies' not in response_data:
                response_data['quick_replies'] = response_data.get('follow_up_chips', initial_chips)
            if 'actions' not in response_data:
                response_data['actions'] = response_data.get('suggested_actions', initial_actions)
        return respond(response_data)



    except Exception as e:

        app.logger.error(f"Error in /api/chat: {e}", exc_info=True)

        return jsonify({

            'reply': f"An unexpected error occurred while processing your request: {str(e)}",

            'confidence_badge': 'System Alert',

            'is_emergency': False,

            'actions': [{'label': '🏠 Return to Home', 'url': '/'}],

            'quick_replies': ['🫀 Chest pain or high BP', '🩸 High blood sugar advice', '🌡️ Fever or cough', '↺ Reset Consultation']

        }), 500





@app.route('/api/chat/reset', methods=['POST'])

@app.route('/api/chat/clear', methods=['POST'])

def api_chat_reset():

    session['chat_state'] = {'stage': 'idle', 'topic': None, 'severity': None, 'associated_symptoms': None}

    session.pop('last_diagnosis', None)

    session['chat_history'] = []

    session.modified = True

    return jsonify({

        'status': 'success',

        'message': 'Chat session and diagnostic state reset successfully.',

        'reply': "I've reset our conversation. How can I help you today? Tell me how you're feeling or choose a situation below:",

        'quick_replies': [

            '🫀 Chest pain or high BP',

            '🩸 High blood sugar advice',

            '🦋 Thyroid or fatigue check',

            '⚡ Heart rhythm or palpitations',

            '🌡️ Fever, chills, or persistent cough',

            '🧪 Kidney or Liver symptom check',

            '📊 Explain my latest test result'

        ],

        'actions': [

            {'label': '🫀 Heart Screening', 'url': '/heart'},

            {'label': '🩸 Diabetes Screening', 'url': '/diabetes'},

            {'label': '🦋 Thyroid Screening', 'url': '/thyroid'},

            {'label': '⚡ ECG Arrhythmia', 'url': '/arrhythmia'},

            {'label': '🩺 Kidney Screening', 'url': '/kidney'},

            {'label': '🔍 All 9 Diagnostic Suites', 'url': '/features'}

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
            appointments = list_appointments_admin()
            return render_template(
                'dev_access.html',
                is_authenticated=True,
                developer_email=session.get('dev_email'),
                stats=stats,
                appointments=appointments
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
        with DEV_OTP_LOCK:
            cached_otp = DEV_OTP_CACHE.get('otp')
            expires_at = DEV_OTP_CACHE.get('expires_at')
            attempts = DEV_OTP_CACHE.get('attempts', 0)
            max_attempts = DEV_OTP_CACHE.get('max_attempts', 5)

            if attempts >= max_attempts:
                DEV_OTP_CACHE['otp'] = None
                DEV_OTP_CACHE['expires_at'] = None
                auth_failure_reason = "Too many failed attempts. Verification code invalidated. Please request a new OTP."
            elif not cached_otp or not expires_at:
                auth_failure_reason = "No active OTP found. Please click 'Send OTP' to request a new verification code."
            elif datetime.datetime.now() > expires_at:
                DEV_OTP_CACHE['otp'] = None
                DEV_OTP_CACHE['expires_at'] = None
                auth_failure_reason = "OTP has expired. Please request a new 6-digit code."
            elif not secrets.compare_digest(otp_entered, str(cached_otp)):
                DEV_OTP_CACHE['attempts'] = attempts + 1
                if DEV_OTP_CACHE['attempts'] >= max_attempts:
                    DEV_OTP_CACHE['otp'] = None
                    DEV_OTP_CACHE['expires_at'] = None
                    auth_failure_reason = "Too many failed attempts. Verification code invalidated. Please request a new OTP."
                else:
                    remaining = max_attempts - (attempts + 1)
                    auth_failure_reason = f"Invalid OTP verification code. {remaining} attempt(s) remaining."
            else:
                authenticated = True
                DEV_OTP_CACHE['otp'] = None
                DEV_OTP_CACHE['expires_at'] = None
                DEV_OTP_CACHE['attempts'] = 0
    else:
        # Access Key verification with constant-time comparison
        if access_key and secrets.compare_digest(access_key, DEV_ACCESS_KEY):
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

    now_ts = time.time()
    with DEV_OTP_LOCK:
        last_req = DEV_OTP_CACHE.get('last_requested_at', 0)
        cooldown = 60
        if now_ts - last_req < cooldown:
            wait_sec = int(cooldown - (now_ts - last_req))
            return jsonify({
                "success": False,
                "message": f"Rate limit reached. Please wait {wait_sec} seconds before requesting a new code."
            }), 429

        # Generate cryptographically secure 6-digit OTP
        otp_val = f"{secrets.randbelow(900000) + 100000}"
        DEV_OTP_CACHE['otp'] = otp_val
        DEV_OTP_CACHE['expires_at'] = datetime.datetime.now() + datetime.timedelta(minutes=10)
        DEV_OTP_CACHE['email'] = email
        DEV_OTP_CACHE['attempts'] = 0
        DEV_OTP_CACHE['last_requested_at'] = now_ts

    subject = "[MediNova Developer Core] One-Time Verification Code"
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
    if not email_sent:
        with DEV_OTP_LOCK:
            DEV_OTP_CACHE['otp'] = None
            DEV_OTP_CACHE['expires_at'] = None
        app.logger.error(f"[SECURITY OTP] Failed to send developer OTP via SMTP: {email_msg}")
        return jsonify({
            "success": False,
            "message": "Failed to deliver verification code via SMTP. Please ensure SMTP configuration is active on the server."
        }), 502

    app.logger.info(f"[SECURITY OTP] One-time verification code dispatched to developer email.")
    return jsonify({
        "success": True,
        "message": f"A 6-digit verification code has been dispatched to {AUTHORIZED_DEV_EMAIL}."
    }), 200


@app.route('/developer/logout', methods=['GET'])
def developer_logout():
    """Clear developer session and lock access."""
    session.pop('dev_authenticated', None)
    session.pop('dev_email', None)
    session.pop('dev_login_time', None)
    return redirect(url_for('developer_patient_records'))


@app.route('/api/admin/appointments', methods=['GET'])
def api_admin_list_appointments():
    """Retrieve appointment records for authenticated clinic administrators."""
    if not is_developer_authenticated():
        return jsonify({"success": False, "message": "403 Forbidden: Authorized developer session required."}), 403
    status = request.args.get('status')
    condition = request.args.get('condition')
    appointments = list_appointments_admin(status_filter=status, condition_filter=condition)
    return jsonify({"success": True, "appointments": appointments})


@app.route('/api/admin/appointments/update-status', methods=['POST'])
def api_admin_update_status():
    """Perform an authenticated status transition on an appointment record with an audit entry."""
    if not is_developer_authenticated():
        return jsonify({"success": False, "message": "403 Forbidden: Authorized developer session required."}), 403
    data = request.get_json(silent=True) or {}
    ref_id = data.get('reference_id', '').strip().upper()
    new_status = data.get('status', '').strip()
    notes = data.get('admin_notes', '').strip()

    if not ref_id or not new_status:
        return jsonify({"success": False, "message": "Reference ID and target status are required."}), 400

    admin_actor = f"Clinic Admin ({session.get('dev_email', 'dharshan')})"
    success, err = update_appointment_status(ref_id, new_status, admin_actor, admin_notes=notes)
    if success:
        return jsonify({"success": True, "message": f"Appointment {ref_id} status updated to '{new_status}'."})
    return jsonify({"success": False, "message": err or "Failed to update status."}), 400


@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)



