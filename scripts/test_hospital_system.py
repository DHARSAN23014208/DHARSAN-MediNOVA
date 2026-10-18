"""
Comprehensive Automated Test Suite for MediNova Hospital Discovery & Appointment Management System
===================================================================================================
Tests:
1. Seven-Disease Medical Specialty Mapping.
2. Curated Global Referral Centres Dataset.
3. Haversine Distance Calculation.
4. Recommendation & Scoring Engine.
5. SQLite Database Persistence, Validation & Lifecycle Workflow.
6. Patient Isolation & Access Control.
7. Google Calendar URL & RFC 5545 .ics Generation.
8. Flask API Endpoints (Discovery, Booking, Tracking, Cancellation, Calendar, Admin).
"""

import os
import sys
import json
import unittest
import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app
from utils.hospital_service import (
    DISEASE_SPECIALTY_MAP,
    CURATED_GLOBAL_CENTRES,
    calculate_haversine_distance,
    score_and_recommend_hospitals,
    discover_hospitals_for_patient
)
from utils.appointment_db import (
    create_appointment,
    get_appointment_by_ref,
    update_appointment_status,
    cancel_appointment_patient,
    list_appointments_admin,
    VALID_STATUSES
)
from utils.calendar_service import (
    generate_google_calendar_url,
    generate_ics_content
)


class TestHospitalDiscoveryAndAppointmentSystem(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    # --------------------------------------------------------------------------
    # 1. SEVEN-DISEASE SPECIALTY MAPPING
    # --------------------------------------------------------------------------
    def test_01_all_seven_conditions_mapped_correctly(self):
        expected_conditions = ["heart", "diabetes", "kidney", "liver", "cancer", "malaria", "pneumonia"]
        for cond in expected_conditions:
            self.assertIn(cond, DISEASE_SPECIALTY_MAP, f"Condition '{cond}' missing from mapping")
            data = DISEASE_SPECIALTY_MAP[cond]
            self.assertTrue(data.get("canonical_name"), f"Missing canonical_name for {cond}")
            self.assertTrue(data.get("primary_specialty"), f"Missing primary_specialty for {cond}")
            self.assertTrue(len(data.get("specialist_types", [])) > 0, f"Missing specialist_types for {cond}")
            self.assertTrue(len(data.get("search_keywords", [])) > 0, f"Missing search_keywords for {cond}")
        print("✅ Test 1 Passed: All 7 diagnostic conditions mapped to canonical specialties.")

    # --------------------------------------------------------------------------
    # 2. CURATED GLOBAL SPECIALTY CENTRES
    # --------------------------------------------------------------------------
    def test_02_curated_global_centres_data_integrity(self):
        self.assertTrue(len(CURATED_GLOBAL_CENTRES) >= 10, "Should have at least 10 curated global centres")
        for centre in CURATED_GLOBAL_CENTRES:
            self.assertTrue(centre.get("name"), "Centre must have a name")
            self.assertTrue(centre.get("website_url", "").startswith("http"), "Centre must have a valid website URL")
            self.assertTrue(centre.get("source_attribution"), "Centre must have source attribution")
            self.assertTrue(centre.get("last_verified"), "Centre must have a last_verified date")
            self.assertTrue(len(centre.get("condition_ids", [])) > 0, "Centre must link to at least one condition")
        print("✅ Test 2 Passed: Curated global specialty centres dataset integrity verified.")

    # --------------------------------------------------------------------------
    # 3. HAVERSINE DISTANCE CALCULATION
    # --------------------------------------------------------------------------
    def test_03_haversine_distance_calculation(self):
        # Known coordinates: New York (40.7128, -74.0060) to London (51.5074, -0.1278) ~5570 km
        dist = calculate_haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
        self.assertAlmostEqual(dist, 5570, delta=50)

        # Zero distance for identical points
        zero_dist = calculate_haversine_distance(13.0604, 80.2514, 13.0604, 80.2514)
        self.assertEqual(zero_dist, 0.0)
        print("✅ Test 3 Passed: Haversine straight-line distance calculations verified.")

    # --------------------------------------------------------------------------
    # 4. RECOMMENDATION & SCORING ENGINE
    # --------------------------------------------------------------------------
    def test_04_recommendation_scoring_engine(self):
        sample_hospitals = [
            {
                "id": "hosp-1",
                "name": "General Healthcare Center",
                "facility_type": "Clinic",
                "category": "nearby",
                "latitude": 13.0600,
                "longitude": 80.2500,
                "distance_km": 2.0,
                "telemedicine_available": False,
                "tags": {}
            },
            {
                "id": "hosp-2",
                "name": "Metropolitan Cardiac and Heart Hospital",
                "facility_type": "Multi-Specialty Hospital",
                "category": "nearby",
                "latitude": 13.0650,
                "longitude": 80.2550,
                "distance_km": 3.0,
                "telemedicine_available": True,
                "tags": {"healthcare:speciality": "cardiology"}
            }
        ]

        scored = score_and_recommend_hospitals(sample_hospitals, condition_id="heart", patient_lat=13.0600, patient_lon=80.2500)
        self.assertEqual(len(scored), 2)
        # hosp-2 matches cardiology keywords so it should rank first
        self.assertEqual(scored[0]["id"], "hosp-2")
        self.assertTrue(scored[0]["specialty_match"])
        self.assertIn("cardiology", scored[0]["recommendation_rationale"].lower())
        print("✅ Test 4 Passed: Recommendation engine accurately prioritizes matching specialty facilities.")

    # --------------------------------------------------------------------------
    # 5. PERSISTENT APPOINTMENT DATABASE LIFECYCLE
    # --------------------------------------------------------------------------
    def test_05_appointment_database_crud_and_status_workflow(self):
        tomorrow = (datetime.datetime.now() + datetime.timedelta(days=2)).strftime("%Y-%m-%d")

        apt_data = {
            "patient_name": "Test Patient",
            "phone": "+1 555-019-2834",
            "email": "testpatient@medinova.ai",
            "condition_id": "heart",
            "condition_name": "Cardiovascular / Heart Disease",
            "clinical_notes": "Follow-up consultation for elevated blood pressure.",
            "hospital_id": "centre-apollo-chennai-cardio",
            "hospital_name": "Apollo Hospitals Heart Centre",
            "hospital_address": "21 Greams Lane, Chennai, India",
            "consultation_mode": "in_person",
            "preferred_date": tomorrow,
            "preferred_time_slot": "10:00 - 11:00 AM",
            "consent_given": 1
        }

        # 1. Create Appointment
        success, record, err = create_appointment(apt_data)
        self.assertTrue(success, f"Failed to create appointment: {err}")
        self.assertIn("APT-", record["reference_id"])
        self.assertEqual(record["status"], "Awaiting Hospital Confirmation")

        ref_id = record["reference_id"]

        # 2. Retrieve Appointment with Matching Contact (Patient Isolation)
        fetched = get_appointment_by_ref(ref_id, verification_contact="testpatient@medinova.ai")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["patient_name"], "Test Patient")

        # 3. Retrieve Appointment with Mismatched Contact (Must return None)
        blocked = get_appointment_by_ref(ref_id, verification_contact="hacker@intruder.com")
        self.assertIsNone(blocked, "Isolation failed: record exposed to mismatched contact")

        # 4. Status Transition: Change to 'Confirmed by Hospital'
        updated, update_err = update_appointment_status(ref_id, "Confirmed by Hospital", "Dr. Coordinator", "Hospital scheduling approved")
        self.assertTrue(updated, f"Update status failed: {update_err}")

        verified_rec = get_appointment_by_ref(ref_id)
        self.assertEqual(verified_rec["status"], "Confirmed by Hospital")
        self.assertTrue(len(verified_rec["status_history"]) >= 2)

        # 5. Invalid Past Date Rejection
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        bad_apt_data = dict(apt_data, preferred_date=yesterday)
        bad_success, _, bad_err = create_appointment(bad_apt_data)
        self.assertFalse(bad_success)
        self.assertIn("past", bad_err.lower())

        print("✅ Test 5 Passed: SQLite appointment database CRUD, isolation, and status transitions verified.")

    # --------------------------------------------------------------------------
    # 6. CALENDAR EVENT GENERATION
    # --------------------------------------------------------------------------
    def test_06_calendar_google_and_ics_generation(self):
        sample_apt = {
            "reference_id": "APT-2026-TEST01",
            "patient_name": "Eleanor Vance",
            "condition_name": "Cardiovascular / Heart Disease",
            "hospital_name": "Royal Brompton Heart Centre",
            "hospital_address": "Sydney St, London, UK",
            "consultation_mode": "in_person",
            "preferred_date": "2026-11-15",
            "preferred_time_slot": "10:00 - 11:00 AM",
            "status": "Awaiting Hospital Confirmation",
            "directions_url": "https://maps.google.com"
        }

        # Google Calendar Link
        gcal_url = generate_google_calendar_url(sample_apt)
        self.assertIn("calendar.google.com/calendar/render", gcal_url)
        self.assertIn("APT-2026-TEST01", gcal_url)

        # RFC 5545 .ics String
        ics_text = generate_ics_content(sample_apt)
        self.assertIn("BEGIN:VCALENDAR", ics_text)
        self.assertIn("BEGIN:VEVENT", ics_text)
        self.assertIn("APT-2026-TEST01@medinova.ai", ics_text)
        self.assertIn("END:VCALENDAR", ics_text)
        print("✅ Test 6 Passed: Google Calendar web URL and RFC 5545 .ics generator verified.")

    # --------------------------------------------------------------------------
    # 7. FLASK HTTP ENDPOINTS INTEGRATION
    # --------------------------------------------------------------------------
    def test_07_flask_hospital_discovery_and_booking_routes(self):
        # 1. GET /contact renders
        res = self.client.get('/contact')
        self.assertEqual(res.status_code, 200)
        html = res.data.decode('utf-8')
        self.assertIn("Hospital Discovery & Specialist Appointment Management", html)

        # 2. GET /api/hospitals/specialties
        res_spec = self.client.get('/api/hospitals/specialties')
        self.assertEqual(res_spec.status_code, 200)
        data_spec = res_spec.get_json()
        self.assertEqual(data_spec["status"], "success")
        self.assertEqual(len(data_spec["specialties"]), 7)

        # 3. GET /api/hospitals/search
        res_search = self.client.get('/api/hospitals/search?condition=diabetes&radius_km=20')
        self.assertEqual(res_search.status_code, 200)
        data_search = res_search.get_json()
        self.assertEqual(data_search["status"], "success")
        self.assertTrue(len(data_search["recommendations"]) > 0)

        # 4. POST /api/appointments/book
        tomorrow = (datetime.datetime.now() + datetime.timedelta(days=3)).strftime("%Y-%m-%d")
        payload = {
            "patient_name": "API Test User",
            "phone": "+1-800-555-1212",
            "email": "apitest@medinova.ai",
            "condition_id": "kidney",
            "condition_name": "Chronic Kidney Disease",
            "clinical_notes": "Mild edema, routine creatinine check.",
            "hospital_name": "Christian Medical College Nephrology",
            "hospital_address": "Vellore, Tamil Nadu, India",
            "consultation_mode": "in_person",
            "preferred_date": tomorrow,
            "preferred_time_slot": "02:00 - 03:00 PM",
            "consent_given": True
        }
        res_book = self.client.post('/api/appointments/book', json=payload)
        self.assertEqual(res_book.status_code, 201)
        data_book = res_book.get_json()
        self.assertTrue(data_book["success"])
        ref_id = data_book["appointment"]["reference_id"]
        self.assertIn("APT-", ref_id)

        # 5. POST /api/appointments/lookup
        res_lookup = self.client.post('/api/appointments/lookup', json={
            "reference_id": ref_id,
            "contact": "apitest@medinova.ai"
        })
        self.assertEqual(res_lookup.status_code, 200)
        data_lookup = res_lookup.get_json()
        self.assertTrue(data_lookup["success"])
        self.assertEqual(data_lookup["appointment"]["patient_name"], "API Test User")

        # 6. GET /api/appointments/<ref_id>/ics
        res_ics = self.client.get(f'/api/appointments/{ref_id}/ics')
        self.assertEqual(res_ics.status_code, 200)
        self.assertEqual(res_ics.mimetype, "text/calendar")
        self.assertIn("BEGIN:VCALENDAR", res_ics.data.decode('utf-8'))

        # 7. POST /api/appointments/cancel
        res_cancel = self.client.post('/api/appointments/cancel', json={
            "reference_id": ref_id,
            "contact": "apitest@medinova.ai",
            "reason": "Test user rescheduled externally."
        })
        self.assertEqual(res_cancel.status_code, 200)

        # 8. Admin Endpoint Access Control (Unauthenticated -> 403)
        res_admin_unauth = self.client.get('/api/admin/appointments')
        self.assertEqual(res_admin_unauth.status_code, 403)

        # 9. Admin Endpoint Access Control (Authenticated Session -> 200)
        with self.client.session_transaction() as sess:
            sess['dev_authenticated'] = True
            sess['dev_email'] = 'dharshankumarkumar3@gmail.com'

        res_admin_auth = self.client.get('/api/admin/appointments')
        self.assertEqual(res_admin_auth.status_code, 200)
        data_admin = res_admin_auth.get_json()
        self.assertTrue(data_admin["success"])
        self.assertTrue(len(data_admin["appointments"]) >= 1)

        print("✅ Test 7 Passed: All Flask endpoints, routes, access controls, and ICS download verified.")


if __name__ == "__main__":
    unittest.main()
