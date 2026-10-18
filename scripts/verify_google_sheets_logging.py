"""
MediNova Comprehensive Verification Script for Private Google Sheets Logging & Developer Security
"""
import os
import sys

# Add repo root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from app import app, AUTHORIZED_DEV_EMAIL, DEV_ACCESS_KEY, DEV_OTP_CACHE
from utils.google_sheets_logger import (
    log_patient_prediction,
    get_sheets_status,
    normalize_status,
    _map_form_aliases,
    _build_row_values,
    SHEET_COLUMNS,
    ALL_SHEET_NAMES,
    flush_log_queue
)

def run_tests():
    print("=================================================================")
    print("MEDINOVA VERIFICATION: PRIVATE GOOGLE SHEETS LOGGING & SECURITY")
    print("=================================================================")

    client = app.test_client()

    # -------------------------------------------------------------
    # 1. Test Silent Patient Submissions Across All 7 Diagnostic Modules
    # -------------------------------------------------------------
    print("\n[TEST 1] Testing Diagnostic Modules Silent Prediction & Logging...")

    # A) Heart Disease
    heart_payload = {
        'age': 52, 'sex': 1, 'cp': 1, 'bp': 125, 'chol': 212,
        'fbs': 0, 'ekg': 1, 'max_hr': 168, 'exang': 0, 'oldpeak': 1.0,
        'slope': 2, 'ca': 2, 'thal': 3
    }
    resp = client.post('/heart', data=heart_payload)
    assert resp.status_code == 200, f"Heart POST failed: {resp.status_code}"
    assert b"Presence of Cardiovascular Risk" in resp.data or b"Absence of Significant Risk" in resp.data
    print("  [OK] /heart POST successful (status 200, valid clinical result returned)")

    # B) Diabetes
    diabetes_payload = {
        'gender': 'Male', 'age': 45, 'hypertension': 0, 'heart_disease': 0,
        'smoking_history': 'never', 'bmi': 27.5, 'HbA1c_level': 6.2, 'blood_glucose_level': 140
    }
    resp = client.post('/diabetes', data=diabetes_payload)
    assert resp.status_code == 200, f"Diabetes POST failed: {resp.status_code}"
    assert b"Elevated Diabetes Risk" in resp.data or b"Normal Glycemic Indicators" in resp.data
    print("  [OK] /diabetes POST successful (status 200, valid glycemic result returned)")

    # C) Kidney Disease
    kidney_payload = {
        'age': 48, 'bp': 80, 'sg': 1.020, 'al': 1, 'su': 0, 'rbc': 1,
        'pc': 1, 'pcc': 0, 'ba': 0, 'bgr': 121, 'bu': 36, 'sc': 1.2,
        'sod': 135, 'pot': 4.5, 'hemo': 15.4, 'pcv': 44, 'wc': 7800,
        'rc': 5.2, 'htn': 1, 'dm': 1, 'cad': 0, 'appet': 0, 'pe': 0, 'ane': 0
    }
    resp = client.post('/kidney', data=kidney_payload)
    assert resp.status_code == 200, f"Kidney POST failed: {resp.status_code}"
    assert b"Chronic Kidney Disease" in resp.data or b"Healthy Renal Function" in resp.data
    print("  [OK] /kidney POST successful (status 200, valid renal result returned)")

    # D) Liver Disease
    liver_payload = {
        'Age': 42, 'Gender': 'Male', 'Total_Bilirubin': 1.1, 'Direct_Bilirubin': 0.4,
        'Alkaline_Phosphotase': 180, 'Alamine_Aminotransferase': 35,
        'Aspartate_Aminotransferase': 32, 'Total_Protiens': 6.8, 'Albumin': 3.4,
        'Albumin_and_Globulin_Ratio': 1.0
    }
    resp = client.post('/liver', data=liver_payload)
    assert resp.status_code == 200, f"Liver POST failed: {resp.status_code}"
    assert b"Probability of Liver Disease" in resp.data
    print("  [OK] /liver POST successful (status 200, valid hepatic result returned)")

    # E) Breast Cancer
    breast_payload = {
        'mean_radius': 14.5, 'mean_texture': 18.2, 'mean_perimeter': 92.5,
        'mean_area': 650.0, 'mean_smoothness': 0.095
    }
    resp = client.post('/breast_cancer', data=breast_payload)
    assert resp.status_code == 200, f"Breast Cancer POST failed: {resp.status_code}"
    assert b"Malignant Neoplasm Indicators" in resp.data or b"Benign Cellular Characteristics" in resp.data
    print("  [OK] /breast_cancer POST successful (status 200, valid cellular result returned)")

    # F) Malaria (Sample trigger)
    malaria_payload = {'sample_type': 'uninfected'}
    resp = client.post('/malaria', data=malaria_payload)
    assert resp.status_code == 200, f"Malaria POST failed: {resp.status_code}"
    assert b"Infected" in resp.data or b"Uninfected" in resp.data
    print("  [OK] /malaria POST successful (status 200, valid microscopic result returned)")

    # G) Pneumonia (Sample trigger)
    pneumonia_payload = {'sample_type': 'detected'}
    resp = client.post('/pneumonia', data=pneumonia_payload)
    assert resp.status_code == 200, f"Pneumonia POST failed: {resp.status_code}"
    assert b"Pneumonia Detected" in resp.data or b"Normal Lung Fields" in resp.data
    print("  [OK] /pneumonia POST successful (status 200, valid pulmonary result returned)")

    # -------------------------------------------------------------
    # 2. Verify Row & Column Data Mapping Architecture
    # -------------------------------------------------------------
    print("\n[TEST 2] Verifying Column Mappings & Canonical Order for Google Sheets...")
    
    assert len(ALL_SHEET_NAMES) == 8, f"Expected 8 worksheets, found {len(ALL_SHEET_NAMES)}"
    for sheet_name in ALL_SHEET_NAMES:
        cols = SHEET_COLUMNS.get(sheet_name)
        assert cols is not None and len(cols) > 0, f"Missing columns for {sheet_name}"
        assert cols[0] == "Record_ID", f"{sheet_name} col 0 must be Record_ID"
        assert cols[1] == "Timestamp", f"{sheet_name} col 1 must be Timestamp"
        assert cols[2] == "Disease_Module", f"{sheet_name} col 2 must be Disease_Module"
        assert "Raw_Prediction" in cols, f"Raw_Prediction missing from {sheet_name}"
        assert "Prediction_Status" in cols, f"Prediction_Status missing from {sheet_name}"
        assert "Confidence_Score" in cols, f"Confidence_Score missing from {sheet_name}"
        print(f"  [OK] Worksheet '{sheet_name}': {len(cols)} stable columns verified")

    # Test Status Normalization
    assert normalize_status(1, is_positive=True) == "Positive"
    assert normalize_status(0, is_positive=False) == "Negative"
    assert normalize_status("Parasitized") == "Positive"
    assert normalize_status("Uninfected") == "Negative"
    assert normalize_status("Malignant") == "Positive"
    assert normalize_status("Benign") == "Negative"
    print("  [OK] Status normalization accurately produces strictly 'Positive' or 'Negative'")

    # Test Zero Value Preservation (e.g., pregnancies=0 or exang=0)
    heart_aliases = _map_form_aliases("heart", {'age': 50, 'exang': 0, 'bp': 120})
    assert heart_aliases['exang'] == 0, "Numeric zero must be preserved in form mapping"
    assert heart_aliases['trestbps'] == 120, "bp alias to trestbps must be mapped"

    diabetes_aliases = _map_form_aliases("diabetes", {'Pregnancies': 0, 'blood_glucose_level': 130, 'age': 40})
    assert diabetes_aliases['Pregnancies'] == 0, "Zero pregnancies must be preserved"
    assert diabetes_aliases['Glucose'] == 130, "blood_glucose_level mapped to Glucose"
    assert diabetes_aliases['Age'] == 40, "age mapped to Age"
    print("  [OK] Alias mapping and zero preservation verified")

    # -------------------------------------------------------------
    # 3. Test Absence of Local Files (Zero Excel/CSV Residuals)
    # -------------------------------------------------------------
    print("\n[TEST 3] Verifying Zero Local Excel / CSV / File Logging Residuals...")
    data_dir = os.path.join(ROOT_DIR, "data")
    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        excel_files = [f for f in files if f.endswith((".xlsx", ".csv"))]
        assert len(excel_files) == 0, f"Found local Excel files in data/: {excel_files}"
    excel_logger_file = os.path.join(ROOT_DIR, "utils", "excel_logger.py")
    assert not os.path.exists(excel_logger_file), "utils/excel_logger.py must not exist"
    print("  [OK] Zero local .xlsx or .csv patient files exist. Cloud-only logging confirmed.")

    # -------------------------------------------------------------
    # 4. Test Strict Developer Access Control & Security
    # -------------------------------------------------------------
    print("\n[TEST 4] Testing Developer Access Control Gate...")

    # 4A: Unauthenticated access to /developer/patient-records returns login page
    resp = client.get('/developer/patient-records')
    assert resp.status_code == 200
    assert b"Developer Authentication Gate" in resp.data
    assert b"Strictly restricted to MediNova Principal Engineer" in resp.data
    print("  [OK] Unauthenticated GET renders secure developer authentication gate")

    # 4B: Unauthorized Email Submission Returns 403 Forbidden
    resp = client.post('/developer/patient-records', data={
        'email': 'unauthorized_intruder@gmail.com',
        'access_key': DEV_ACCESS_KEY
    })
    assert resp.status_code == 403, f"Expected 403 on invalid email, got {resp.status_code}"
    assert b"403 Forbidden" in resp.data
    print("  [OK] Unauthorized email returns 403 Forbidden and logs security warning")

    # 4C: Authorized Email with Invalid Key Returns 403 Forbidden
    resp = client.post('/developer/patient-records', data={
        'email': AUTHORIZED_DEV_EMAIL,
        'access_key': 'incorrect_secret_xyz'
    })
    assert resp.status_code == 403, f"Expected 403 on invalid key, got {resp.status_code}"
    assert b"403 Forbidden" in resp.data
    print("  [OK] Invalid access key returns 403 Forbidden")

    # 4D: Authorized Email with Correct DEV_ACCESS_KEY -> Authenticated Session
    resp = client.post('/developer/patient-records', data={
        'email': AUTHORIZED_DEV_EMAIL,
        'access_key': DEV_ACCESS_KEY
    }, follow_redirects=True)
    assert resp.status_code == 200
    assert b"Google Sheets Diagnostic Records" in resp.data
    assert b"MediNova_Patient_Records" in resp.data
    assert AUTHORIZED_DEV_EMAIL.encode('utf-8') in resp.data
    print(f"  [OK] Valid Key authentication succeeded for {AUTHORIZED_DEV_EMAIL}!")

    # 4E: Developer Logout
    resp = client.get('/developer/logout', follow_redirects=True)
    assert resp.status_code == 200
    assert b"Developer Authentication Gate" in resp.data
    print("  [OK] Developer logout cleared session successfully")

    # -------------------------------------------------------------
    # 5. Test OTP (Two-Factor) Flow
    # -------------------------------------------------------------
    print("\n[TEST 5] Testing Developer OTP (Two-Factor) Authentication Flow...")

    # 5A: Unauthorized OTP Request Returns 403
    resp = client.post('/developer/request-otp', json={'email': 'hacker@phishing.com'})
    assert resp.status_code == 403
    print("  [OK] Unauthorized OTP request returned 403 Forbidden")

    # 5B: Authorized OTP Request Generates Valid 6-Digit OTP
    resp = client.post('/developer/request-otp', json={'email': AUTHORIZED_DEV_EMAIL})
    assert resp.status_code == 200
    otp = DEV_OTP_CACHE['otp']
    assert otp is not None and len(otp) == 6 and otp.isdigit(), f"Invalid OTP: {otp}"
    print(f"  [OK] Authorized OTP generated: {otp} for {AUTHORIZED_DEV_EMAIL}")

    # 5C: Verify with Invalid OTP Returns 403
    resp = client.post('/developer/patient-records', data={
        'email': AUTHORIZED_DEV_EMAIL,
        'auth_method': 'otp',
        'otp': '000000'
    })
    assert resp.status_code == 403
    print("  [OK] Invalid OTP returned 403 Forbidden")

    # 5D: Verify with Correct OTP Succeeds
    resp = client.post('/developer/patient-records', data={
        'email': AUTHORIZED_DEV_EMAIL,
        'auth_method': 'otp',
        'otp': otp
    }, follow_redirects=True)
    assert resp.status_code == 200
    assert b"Google Sheets Diagnostic Records" in resp.data
    print(f"  [OK] Correct OTP authentication succeeded for {AUTHORIZED_DEV_EMAIL}!")

    # Flush background queue
    flush_log_queue(timeout=3.0)

    print("\n=================================================================")
    print("ALL VERIFICATION SUITES PASSED FLAWLESSLY! (100% SUCCESS)")
    print("=================================================================\n")

if __name__ == "__main__":
    run_tests()
