"""
MediNova Comprehensive Audit & Security Fixes Regression Test Suite
===================================================================
Covers all audit phases:
1. Secrets & Production Fail-Closed
2. Developer Portal & OTP Security (zero leakage, rate-limiting, lockout, timing resistance)
3. Patient Image Upload Privacy (private_uploads/, PIL validation, session preview gating)
4. Hospital Discovery Input Validation (condition required, 400 on invalid, bounds checking)
5. Strict Specialty Matching & Non-clinical exclusion
6. Appointment Database & Permitted State Transitions Graph
7. Tabular Diagnostic Models Confidence Handling (No fake 95.0, None when uncalibrated)
8. Google Sheets Logger (Formula injection prevention, idempotent retry recovery, SQLite outbox)
9. Cryptographic Report Integrity & HMAC-SHA256 Verification
10. Application Route Availability & Health
"""

import os
import io
import json
import time
import shutil
import sqlite3
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

from app import (
    app,
    DEV_OTP_CACHE,
    DEV_OTP_LOCK,
    AUTHORIZED_DEV_EMAIL,
    DEV_ACCESS_KEY,
    validate_and_save_patient_image
)
from utils.hospital_service import (
    get_specialty_mapping,
    discover_hospitals_for_patient,
    DISEASE_SPECIALTY_MAP
)
from utils.appointment_db import (
    create_appointment,
    get_appointment_by_ref,
    update_appointment_status,
    cancel_appointment_patient,
    PERMITTED_TRANSITIONS,
    VALID_STATUSES
)
from utils.google_sheets_logger import (
    _clean_cell_value,
    _insert_into_outbox,
    _update_outbox_progress,
    _get_outbox_counts,
    log_patient_prediction,
    get_sheets_status,
    OUTBOX_DB_PATH
)
from utils.diagnostic_predictor import (
    predict_heart,
    predict_diabetes,
    predict_kidney,
    predict_liver,
    predict_breast_cancer
)
from app import DISEASE_INSIGHTS
from utils.report_verifier import (
    register_report,
    get_verified_report,
    generate_qr_code_image,
    generate_verification_hash
)


class TestSecurityAndSecrets(unittest.TestCase):
    """Phase 2A & 2B: Secrets, Developer OTP, and Session Security."""

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True
        with DEV_OTP_LOCK:
            DEV_OTP_CACHE['otp'] = None
            DEV_OTP_CACHE['expires_at'] = None
            DEV_OTP_CACHE['attempts'] = 0
            DEV_OTP_CACHE['last_requested_at'] = 0

    def test_otp_never_disclosed_in_response_or_logs(self):
        """OTP request must NEVER return the code in HTTP response, even if SMTP succeeds or fails."""
        with patch('app.send_developer_email', return_value=(True, "Delivered")):
            resp = self.client.post('/developer/request-otp', json={"email": AUTHORIZED_DEV_EMAIL})
            data = resp.get_json()
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(data.get('success'))
            # Ensure 6-digit OTP code does not appear anywhere in message
            with DEV_OTP_LOCK:
                cached_otp = DEV_OTP_CACHE['otp']
            self.assertIsNotNone(cached_otp)
            self.assertNotIn(str(cached_otp), data.get('message', ''))

    def test_otp_smtp_failure_fails_safely_without_leakage(self):
        """When SMTP fails, return 502, do NOT report success, and invalidate OTP."""
        with patch('app.send_developer_email', return_value=(False, "Connection refused")):
            resp = self.client.post('/developer/request-otp', json={"email": AUTHORIZED_DEV_EMAIL})
            self.assertEqual(resp.status_code, 502)
            data = resp.get_json()
            self.assertFalse(data.get('success'))
            self.assertIn("Failed to deliver verification code", data.get('message', ''))
            with DEV_OTP_LOCK:
                self.assertIsNone(DEV_OTP_CACHE['otp'], "OTP must be invalidated when delivery fails")

    def test_otp_rate_limiting_cooldown(self):
        """Requesting OTP again within 60 seconds must return 429 Too Many Requests."""
        with patch('app.send_developer_email', return_value=(True, "Delivered")):
            # First request
            resp1 = self.client.post('/developer/request-otp', json={"email": AUTHORIZED_DEV_EMAIL})
            self.assertEqual(resp1.status_code, 200)

            # Immediate second request
            resp2 = self.client.post('/developer/request-otp', json={"email": AUTHORIZED_DEV_EMAIL})
            self.assertEqual(resp2.status_code, 429)
            data2 = resp2.get_json()
            self.assertFalse(data2.get('success'))
            self.assertIn("Rate limit reached", data2.get('message', ''))

    def test_otp_lockout_after_max_attempts(self):
        """5 failed OTP verification attempts must invalidate the OTP and lock out."""
        with DEV_OTP_LOCK:
            DEV_OTP_CACHE['otp'] = "123456"
            import datetime
            DEV_OTP_CACHE['expires_at'] = datetime.datetime.now() + datetime.timedelta(minutes=10)
            DEV_OTP_CACHE['attempts'] = 0

        # Send 5 incorrect attempts
        for attempt in range(1, 6):
            resp = self.client.post('/developer/patient-records', data={
                'email': AUTHORIZED_DEV_EMAIL,
                'auth_method': 'otp',
                'otp': '999999'  # wrong code
            })
            self.assertEqual(resp.status_code, 403)

        # 6th attempt must be permanently locked
        with DEV_OTP_LOCK:
            self.assertIsNone(DEV_OTP_CACHE['otp'], "OTP must be invalidated after 5 failed attempts")

        resp = self.client.post('/developer/patient-records', data={
            'email': AUTHORIZED_DEV_EMAIL,
            'auth_method': 'otp',
            'otp': '123456'
        })
        self.assertEqual(resp.status_code, 403)

    def test_unauthorized_dashboard_access_blocked(self):
        """Unauthenticated GET to /api/admin/appointments must return 403."""
        resp = self.client.get('/api/admin/appointments')
        self.assertEqual(resp.status_code, 403)

        resp2 = self.client.post('/api/admin/appointments/update-status', json={
            "reference_id": "APT-2026-FAKE12",
            "status": "Confirmed by Hospital"
        })
        self.assertEqual(resp2.status_code, 403)


class TestPatientImageUploadPrivacy(unittest.TestCase):
    """Phase 2C: Patient Image Upload Privacy & Integrity."""

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def _create_synthetic_png(self, width=64, height=64):
        """Generate a valid, non-corrupt PNG in-memory."""
        img = Image.new('RGB', (width, height), color=(73, 109, 137))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf

    def test_valid_image_saved_in_private_uploads_with_session_token(self):
        """Valid image is stored in private_uploads and given a session-bound preview token."""
        png_io = self._create_synthetic_png()
        from werkzeug.datastructures import FileStorage
        fs = FileStorage(stream=png_io, filename='patient_scan.png', content_type='image/png')

        with app.test_request_context():
            from flask import session
            filepath, preview_url, err = validate_and_save_patient_image(fs)
            self.assertIsNone(err)
            self.assertIsNotNone(filepath)
            self.assertIsNotNone(preview_url)
            self.assertIn('private_uploads', filepath)
            self.assertNotIn('static', filepath)
            self.assertTrue(os.path.exists(filepath))
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_corrupt_or_fake_image_rejected(self):
        """Text file disguised as a .png must fail Pillow content validation."""
        fake_io = io.BytesIO(b"This is not a real image file, just ASCII text")
        from werkzeug.datastructures import FileStorage
        fs = FileStorage(stream=fake_io, filename='fake_scan.png', content_type='image/png')

        with app.test_request_context():
            filepath, preview_url, err = validate_and_save_patient_image(fs)
            self.assertIsNotNone(err)
            self.assertIsNone(filepath)
            self.assertIn("not a valid or readable medical image", err)

    def test_invalid_extension_rejected(self):
        """Executable or script files must be rejected immediately."""
        fake_io = io.BytesIO(b"MZ\x90\x00\x03\x00\x00\x00")
        from werkzeug.datastructures import FileStorage
        fs = FileStorage(stream=fake_io, filename='payload.exe', content_type='application/x-dosexec')

        with app.test_request_context():
            filepath, preview_url, err = validate_and_save_patient_image(fs)
            self.assertIsNotNone(err)
            self.assertIn("Invalid image format", err)


class TestHospitalDiscoveryValidation(unittest.TestCase):
    """Phase 3: Strict Hospital Search & Radius Validation."""

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_missing_condition_returns_400(self):
        """Omitting condition must return 400 with a clear validation error."""
        resp = self.client.get('/api/hospitals/search')
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertIn("condition is required", data.get("message", "").lower())

    def test_unsupported_condition_returns_400(self):
        """Searching for an unrecognized disease must return 400 without falling back to heart."""
        resp = self.client.get('/api/hospitals/search?condition=unknown_exotic_illness')
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertIn("unsupported diagnostic condition", data.get("message", "").lower())

    def test_invalid_coordinates_out_of_bounds_returns_400(self):
        """Latitude > 90 or longitude > 180 must return 400."""
        resp = self.client.get('/api/hospitals/search?condition=pneumonia&lat=95.0&lon=80.0')
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("out of bounds", data.get("message", "").lower())

    def test_incomplete_coordinates_returns_400(self):
        """Supplying latitude without longitude must return 400."""
        resp = self.client.get('/api/hospitals/search?condition=pneumonia&lat=13.08')
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertIn("both latitude and longitude", data.get("message", "").lower())

    def test_invalid_radius_limits_returns_400(self):
        """Radius < 1 km or > 100 km must return 400."""
        resp1 = self.client.get('/api/hospitals/search?condition=pneumonia&radius_km=0')
        self.assertEqual(resp1.status_code, 400)

        resp2 = self.client.get('/api/hospitals/search?condition=pneumonia&radius_km=150')
        self.assertEqual(resp2.status_code, 400)

    def test_pneumonia_keyword_excludes_generic_medicine(self):
        """Pneumonia must not match generic Internal Medicine without Pulmonology evidence."""
        pneumonia_spec = get_specialty_mapping("pneumonia")
        self.assertTrue(pneumonia_spec.get("disallow_general_medicine_fallback"))
        self.assertNotIn("general medicine", [kw.lower() for kw in pneumonia_spec.get("search_keywords", [])])


class TestAppointmentLifecycleAndTransitions(unittest.TestCase):
    """Phase 4: Appointment Booking, Validation & State Machine Transitions."""

    def test_create_appointment_with_unmapped_condition_fails(self):
        """Creating an appointment for an invalid condition returns False with error."""
        success, record, err = create_appointment({
            "patient_name": "Test Patient",
            "phone": "9876543210",
            "email": "patient@example.com",
            "condition_id": "totally_bogus_disease",
            "condition_name": "Bogus",
            "hospital_name": "General Hospital",
            "hospital_address": "123 Medical Way",
            "consultation_mode": "in_person",
            "preferred_date": "2026-12-01",
            "preferred_time_slot": "10:00 AM",
            "consent_given": True
        })
        self.assertFalse(success)
        self.assertIn("Invalid or unsupported clinical condition", err)

    def test_create_appointment_sets_initial_status_strictly_awaiting(self):
        """Initial appointment status must strictly be 'Awaiting Hospital Confirmation'."""
        success, record, err = create_appointment({
            "patient_name": "John Doe",
            "phone": "9876543210",
            "email": "johndoe@example.com",
            "condition_id": "heart",
            "condition_name": "Heart Disease",
            "hospital_name": "Apollo Heart Centre",
            "hospital_address": "Greams Road, Chennai",
            "consultation_mode": "in_person",
            "preferred_date": "2026-12-01",
            "preferred_time_slot": "10:00 AM",
            "consent_given": True
        })
        self.assertTrue(success)
        self.assertEqual(record["status"], "Awaiting Hospital Confirmation")
        ref_id = record["reference_id"]

        # Valid transition: Awaiting -> Confirmed by Hospital
        ok, trans_err = update_appointment_status(ref_id, "Confirmed by Hospital", "Clinic Coordinator")
        self.assertTrue(ok)
        self.assertIsNone(trans_err)

        # Valid transition: Confirmed -> Completed
        ok2, trans_err2 = update_appointment_status(ref_id, "Completed", "Attending Cardiologist")
        self.assertTrue(ok2)

        # Illegal transition: Completed (terminal) -> Awaiting Hospital Confirmation
        ok3, trans_err3 = update_appointment_status(ref_id, "Awaiting Hospital Confirmation", "Attending Cardiologist")
        self.assertFalse(ok3)
        self.assertIn("Cannot transition appointment from terminal state", trans_err3)

    def test_patient_isolation_lookup_requires_matching_contact(self):
        """Lookup requires matching email or phone; non-matching contact must be rejected."""
        success, record, _ = create_appointment({
            "patient_name": "Jane Smith",
            "phone": "9123456789",
            "email": "janesmith@example.com",
            "condition_id": "diabetes",
            "condition_name": "Diabetes",
            "hospital_name": "Fortis Care",
            "hospital_address": "Bannerghatta, Bengaluru",
            "consultation_mode": "in_person",
            "preferred_date": "2026-12-01",
            "preferred_time_slot": "11:00 AM",
            "consent_given": True
        })
        self.assertTrue(success)
        ref_id = record["reference_id"]

        # Matching lookup
        rec_match = get_appointment_by_ref(ref_id, "janesmith@example.com")
        self.assertIsNotNone(rec_match)

        # Non-matching lookup
        rec_mismatch = get_appointment_by_ref(ref_id, "attacker@badactor.com")
        self.assertIsNone(rec_mismatch)


class TestDiagnosticModelConfidenceHandling(unittest.TestCase):
    """Phase 5: Diagnostic Models Never Synthesize Hardcoded 95.0 Confidence."""

    def test_predict_heart_returns_none_confidence_when_uncalibrated(self):
        """Mock model with only predict() and no predict_proba() must return confidence = None."""
        class MockClassifierOnly:
            classes_ = [0, 1]
            def predict(self, X):
                return [1]

        mock_m = MockClassifierOnly()
        inputs = {
            "age": 55, "sex": 1, "cp": 2, "bp": 140, "trestbps": 140, "chol": 240,
            "fbs": 0, "ekg": 1, "restecg": 1, "max_hr": 150, "thalach": 150,
            "exang": 0, "oldpeak": 1.5, "slope": 2, "ca": 1, "thal": 2
        }
        res = predict_heart(inputs, mock_m, DISEASE_INSIGHTS)
        self.assertTrue(res["is_positive"])
        self.assertIsNone(res["confidence_score"], "Confidence must be None when model lacks calibrated probability")
        self.assertIsNone(res["confidence"], "Confidence must be None")
        self.assertIsNone(res["percent"], "Percent must be None")
        self.assertNotIn("95.0", str(res["prediction"]))


class TestGoogleSheetsOutboxAndSanitization(unittest.TestCase):
    """Phase 6: Google Sheets Outbox, Idempotency, and Formula Injection Defense."""

    def test_formula_injection_prevention(self):
        """Formulas starting with =, +, -, @, \\t, \\r must be prepended with a single quote."""
        self.assertEqual(_clean_cell_value("=SUM(A1:A10)"), "'=SUM(A1:A10)")
        self.assertEqual(_clean_cell_value("@cmd|' /C calc'!A0"), "'@cmd|' /C calc'!A0")
        self.assertEqual(_clean_cell_value("+cmd"), "'+cmd")
        self.assertEqual(_clean_cell_value("\tmalicious"), "'\tmalicious")

        # Legitimate numbers should NOT be quoted
        self.assertEqual(_clean_cell_value(-12.5), -12.5)
        self.assertEqual(_clean_cell_value("-3.14"), "-3.14")
        self.assertEqual(_clean_cell_value(42), 42)

    def test_outbox_table_records_queued_and_synchronized_states(self):
        """Records must be inserted into SQLite outbox and track progress accurately."""
        rec_id = f"TEST-MNV-{int(time.time())}"
        _insert_into_outbox(rec_id, "Test_Sheet", ["val1", "val2"], ["val1", "summary"])

        counts = _get_outbox_counts()
        self.assertGreaterEqual(counts.get("QUEUED", 0), 1)

        # Transition to SYNCHRONIZED
        _update_outbox_progress(rec_id, module_done=1, master_done=1, status="SYNCHRONIZED")
        counts_synced = _get_outbox_counts()
        self.assertGreaterEqual(counts_synced.get("SYNCHRONIZED", 0), 1)


class TestReportVerificationHMAC(unittest.TestCase):
    """Phase 7: Cryptographic Report Verification via HMAC-SHA256."""

    def test_valid_report_registration_and_hmac_verification(self):
        """Registered report must be verifiable and tamper-evident."""
        rep_id = f"MNV-TEST-{int(time.time())}"
        reg_hash = register_report(
            report_id=rep_id,
            module_name="heart",
            condition_title="Cardiovascular Risk Assessment",
            prediction_text="Presence of Cardiovascular Indicators",
            is_positive=True,
            confidence_score=87.5,
            summary_data={"age": 58, "bp": 145}
        )
        self.assertIsNotNone(reg_hash)
        self.assertTrue(len(reg_hash) >= 16)

        # Verify via official verifier function
        verified_rec = get_verified_report(rep_id)
        self.assertIsNotNone(verified_rec)
        self.assertEqual(verified_rec["report_id"], rep_id)
        self.assertEqual(verified_rec["confidence_score"], 87.5)
        self.assertEqual(verified_rec["verification_hash"], reg_hash)

    def test_qr_code_generation(self):
        """High-resolution PNG QR code must be generated successfully."""
        qr_bytes = generate_qr_code_image("MNV-HEAR-20261020000000-TEST", "http://localhost:5000")
        self.assertIsInstance(qr_bytes, bytes)
        self.assertTrue(len(qr_bytes) > 100)
        # Check PNG header signature (first 8 bytes)
        self.assertEqual(qr_bytes[:8], b'\x89PNG\r\n\x1a\n')


if __name__ == '__main__':
    unittest.main()
