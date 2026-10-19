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
    discover_hospitals_for_patient,
    normalize_address,
    resolve_any_address,
    is_non_clinical_facility,
    get_specialty_mapping
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
        self.assertTrue(len(data_spec["specialties"]) >= 7)

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

    # --------------------------------------------------------------------------
    # 8. UNIVERSAL ADDRESS NORMALIZATION & PROGRESSIVE GEOCODING
    # --------------------------------------------------------------------------
    def test_08_universal_address_normalization_and_geocoding(self):
        # 1. Normalization tests (en dash, em dash, hyphen, commas, whitespace)
        raw_dash = "Saveetha Nagar, Thandalam, Kanchipuram – Chennai Road, Chennai — Tamil Nadu 602105, India"
        norm = normalize_address(raw_dash)
        self.assertNotIn("–", norm)
        self.assertNotIn("—", norm)
        self.assertIn("602105", norm)
        self.assertIn("Thandalam", norm)
        self.assertIn("Chennai", norm)

        # 2. Empty input validation
        self.assertEqual(normalize_address(""), "")
        self.assertEqual(normalize_address("   \t  \n  "), "")
        self.assertIsNone(resolve_any_address(""))
        self.assertIsNone(resolve_any_address("   "))

        # 3. Invalid address returns None (no uncontrolled request loop or crash)
        invalid_res = resolve_any_address("xyz999zzzaaa111nonexistentlocation987654321")
        self.assertIsNone(invalid_res)

        # 4. Dynamic Progressive Geocoding for Prompt's Chennai Test Address
        # Address: Saveetha Nagar, Thandalam, Kanchipuram – Chennai Road, Chennai, Tamil Nadu 602105, India
        res = resolve_any_address(raw_dash)
        self.assertIsNotNone(res, "Universal geocoder should resolve Chennai/Thandalam address via progressive fallback")
        lat = res.get("latitude")
        lon = res.get("longitude")
        display_name = res.get("display_name")

        self.assertIsInstance(lat, float)
        self.assertIsInstance(lon, float)
        self.assertTrue(res.get("source"), "Resolved record must have a source")
        self.assertTrue(display_name, "Resolved record must have a human-readable display name")

        # Validate finite geographic coordinate ranges (-90..90, -180..180)
        self.assertTrue(-90.0 <= lat <= 90.0)
        self.assertTrue(-180.0 <= lon <= 180.0)

        # Verify resolved coordinates are in the expected Chennai/Tamil Nadu region (~12.8 - 13.3 N, ~79.8 - 80.3 E)
        self.assertTrue(12.5 <= lat <= 13.5, f"Latitude {lat} outside expected Tamil Nadu region")
        self.assertTrue(79.5 <= lon <= 80.5, f"Longitude {lon} outside expected Tamil Nadu region")

        # 5. Locality and City without street address
        res_locality = resolve_any_address("Thandalam, Chennai, Tamil Nadu, India")
        self.assertIsNotNone(res_locality)
        self.assertTrue(-90.0 <= res_locality["latitude"] <= 90.0)
        self.assertTrue(-180.0 <= res_locality["longitude"] <= 180.0)

        # 6. Postal Code search with country context
        res_pin = resolve_any_address("602105, Tamil Nadu, India")
        self.assertIsNotNone(res_pin)
        self.assertTrue(-90.0 <= res_pin["latitude"] <= 90.0)

        print("✅ Test 8 Passed: Universal address normalization, progressive geocoding, and coordinate validation verified.")

    # --------------------------------------------------------------------------
    # 9. STRICT RADIUS FILTERING & EXCLUSION OF GLOBAL HOSPITALS FOR LOCAL SEARCHES
    # --------------------------------------------------------------------------
    def test_09_strict_radius_and_no_global_fallback(self):
        # Using resolved coordinates for Thandalam / Chennai (~13.0143, 79.9181)
        test_lat = 13.0143
        test_lon = 79.9181

        # Test radius options: 5, 10, 20, 25, 50 km
        for radius in [5, 10, 20, 25, 50]:
            results = discover_hospitals_for_patient(
                condition_id="heart",
                patient_lat=test_lat,
                patient_lon=test_lon,
                radius_km=radius,
                include_global=False
            )

            # Strict requirement: Every returned facility MUST have distance_km <= radius
            self.assertEqual(results.get("status"), "success")
            hospitals = results.get("recommendations", [])
            for h in hospitals:
                dist = h.get("distance_km")
                self.assertIsNotNone(dist, f"Hospital {h.get('name')} missing distance_km")
                self.assertLessEqual(
                    dist, radius,
                    f"Hospital '{h.get('name')}' distance {dist} km exceeds radius {radius} km!"
                )

                # Strict requirement: Global distant referral centres must NOT be substituted
                name = h.get("name", "")
                self.assertNotIn("Cleveland Clinic", name, "Distant US hospital returned for Chennai search!")
                self.assertNotIn("Royal Brompton", name, "Distant UK hospital returned for Chennai search!")
                self.assertNotIn("Mayo Clinic", name, "Distant US hospital returned for Chennai search!")
                self.assertNotIn("Charité", name, "Distant German hospital returned for Chennai search!")

        print("✅ Test 9 Passed: Strict geographic radius filtering and complete exclusion of global fallback centres verified.")

    # --------------------------------------------------------------------------
    # 10. SPECIALTY MATCHING & VERIFICATION STATUS
    # --------------------------------------------------------------------------
    def test_10_specialty_matching_and_verification_status(self):
        sample_facilities = [
            {
                "id": "hosp-cardio-verified",
                "name": "Sri Balaji Medical Centre",
                "facility_type": "Multi-Specialty Hospital",
                "category": "nearby",
                "latitude": 13.0150,
                "longitude": 79.9200,
                "distance_km": 1.2,
                "telemedicine_available": False,
                "tags": {"healthcare:speciality": "cardiology"}
            },
            {
                "id": "hosp-cardio-inferred",
                "name": "Metropolitan Heart & Cardiac Institute",
                "facility_type": "Multi-Specialty Hospital",
                "category": "nearby",
                "latitude": 13.0200,
                "longitude": 79.9250,
                "distance_km": 2.5,
                "telemedicine_available": True,
                "tags": {}
            },
            {
                "id": "hosp-general-unverified",
                "name": "Thandalam Community Healthcare Clinic",
                "facility_type": "Clinic",
                "category": "nearby",
                "latitude": 13.0180,
                "longitude": 79.9190,
                "distance_km": 1.8,
                "telemedicine_available": False,
                "tags": {}
            }
        ]

        scored = score_and_recommend_hospitals(sample_facilities, condition_id="heart")
        lookup = {h["id"]: h for h in scored}

        # 1. Verified specialty has official tag -> Tier 1
        verified = lookup["hosp-cardio-verified"]
        self.assertTrue(verified["specialty_match"])
        self.assertEqual(verified["tier"], 1)
        self.assertIn("VERIFIED SPECIALTY MATCH (Cardiology)", verified["specialty_label"])
        self.assertEqual(verified["specialty"], "Cardiology")

        # 2. Inferred specialty clearly says VERIFIED SPECIALTY MATCH (Cardiology) based on matching keywords
        inferred = lookup["hosp-cardio-inferred"]
        self.assertTrue(inferred["specialty_match"])
        self.assertEqual(inferred["tier"], 1)
        self.assertIn("VERIFIED SPECIALTY MATCH (Cardiology)", inferred["specialty_label"])
        self.assertEqual(inferred["specialty"], "Cardiology")

        # 3. Unverified facility is marked GENERAL / MULTI-SPECIALTY (INTERNAL MEDICINE) - Tier 2
        unverified = lookup["hosp-general-unverified"]
        self.assertFalse(unverified["specialty_match"])
        self.assertEqual(unverified["tier"], 2)
        self.assertIn("GENERAL / MULTI-SPECIALTY (INTERNAL MEDICINE)", unverified["specialty_label"])
        self.assertEqual(unverified["specialty"], "Internal Medicine / General Medicine")
        self.assertNotEqual(unverified["specialty"], "Cardiology", "Unverified hospital must NOT be stamped with Cardiology")

        print("✅ Test 10 Passed: Reliable distinction between verified and general hospitals verified.")

    # --------------------------------------------------------------------------
    # 11. FLASK GEOCODING PROXY & DYNAMIC SEARCH ENDPOINTS
    # --------------------------------------------------------------------------
    def test_11_flask_geocoding_api_and_search_endpoint(self):
        # 1. GET /api/hospitals/geocode with valid address
        test_addr = "Saveetha Nagar, Thandalam, Kanchipuram – Chennai Road, Chennai, Tamil Nadu 602105, India"
        res_geo = self.client.get(f'/api/hospitals/geocode?query={test_addr}')
        self.assertEqual(res_geo.status_code, 200)
        data_geo = res_geo.get_json()
        self.assertEqual(data_geo["status"], "success")
        loc = data_geo["location"]
        self.assertIsNotNone(loc["latitude"])
        self.assertIsNotNone(loc["longitude"])
        self.assertIn("display_name", loc)

        # 2. GET /api/hospitals/geocode with empty query -> 400
        res_bad = self.client.get('/api/hospitals/geocode?query=')
        self.assertEqual(res_bad.status_code, 400)

        # 3. GET /api/hospitals/search with address query -> resolves location and searches nearby
        res_search = self.client.get(f'/api/hospitals/search?condition=heart&radius_km=25&query={test_addr}')
        self.assertEqual(res_search.status_code, 200)
        data_search = res_search.get_json()
        self.assertEqual(data_search["status"], "success")
        self.assertIsNotNone(data_search.get("location"))

        # Verify no faraway global hospitals are present in recommendations
        for rec in data_search.get("recommendations", []):
            self.assertLessEqual(rec["distance_km"], 25.0)
            self.assertNotIn("Cleveland Clinic", rec["name"])
            self.assertNotIn("Royal Brompton", rec["name"])

        print("✅ Test 11 Passed: Flask geocoding proxy and dynamic address search endpoints verified.")

    # --------------------------------------------------------------------------
    # 12. NON-CLINICAL FACILITY FILTERING (BLOOD BANKS, PHARMACIES, DENTAL, VET)
    # --------------------------------------------------------------------------
    def test_12_non_clinical_facility_filtering(self):
        # Excluded non-clinical facility names
        self.assertTrue(is_non_clinical_facility("Rotary Blood Bank"))
        self.assertTrue(is_non_clinical_facility("Apollo Pharmacy & Chemists"))
        self.assertTrue(is_non_clinical_facility("Happy Paws Veterinary Clinic"))
        self.assertTrue(is_non_clinical_facility("Smile Line Dental Clinic"))
        self.assertTrue(is_non_clinical_facility("Metropolis Pathology & Diagnostic Blood Collection"))

        # Retained hospitals
        self.assertFalse(is_non_clinical_facility("Saveetha Medical College Hospital"))
        self.assertFalse(is_non_clinical_facility("Apollo Speciality Hospital"))
        self.assertFalse(is_non_clinical_facility("Government General Hospital"))
        self.assertFalse(is_non_clinical_facility("Dr. Mohan's Diabetes Specialities Centre"))

        # Ensure recommendation engine strips them out completely
        mixed_facilities = [
            {"id": "hosp-legit", "name": "City Care General Hospital", "facility_type": "Hospital", "distance_km": 2.0},
            {"id": "blood-bank", "name": "Rotary Blood Bank", "facility_type": "Blood Bank", "distance_km": 1.0},
            {"id": "pharma", "name": "Apollo Pharmacy 24x7", "facility_type": "Pharmacy", "distance_km": 0.5},
            {"id": "vet", "name": "Pet Care Veterinary Hospital", "facility_type": "Veterinary", "distance_km": 1.5},
        ]
        results = score_and_recommend_hospitals(mixed_facilities, condition_id="diabetes")
        result_ids = [h["id"] for h in results]
        self.assertIn("hosp-legit", result_ids)
        self.assertNotIn("blood-bank", result_ids, "Blood bank should have been strictly excluded")
        self.assertNotIn("pharma", result_ids, "Pharmacy should have been strictly excluded")
        self.assertNotIn("vet", result_ids, "Veterinary should have been strictly excluded")
        print("✅ Test 12 Passed: Non-clinical facilities (blood banks, pharmacies, vet, dental) strictly excluded.")

    # --------------------------------------------------------------------------
    # 13. GOOGLE PLUS CODE STRIPPING & ADDRESS NORMALIZATION
    # --------------------------------------------------------------------------
    def test_13_google_plus_code_address_normalization(self):
        addr1 = "3QMG+W9F, Sekkalai, Karaikudi, Tamil Nadu 630002"
        clean1 = normalize_address(addr1)
        self.assertNotIn("3QMG+W9F", clean1)
        self.assertIn("Sekkalai", clean1)
        self.assertIn("Karaikudi", clean1)
        self.assertIn("630002", clean1)

        addr2 = "8FVC9G8F+5X Mumbai, Maharashtra 400001"
        clean2 = normalize_address(addr2)
        self.assertNotIn("8FVC9G8F+5X", clean2)
        self.assertIn("Mumbai", clean2)
        self.assertIn("400001", clean2)

        addr3 = "4Q2P+3X, Anna Nagar, Chennai - 600040"
        clean3 = normalize_address(addr3)
        self.assertNotIn("4Q2P+3X", clean3)
        self.assertIn("Anna Nagar", clean3)
        self.assertIn("Chennai", clean3)
        print("✅ Test 13 Passed: Google Plus Codes stripped and address strings sanitized.")

    # --------------------------------------------------------------------------
    # 14. DYNAMIC CONDITION-TO-SPECIALTY MAPPING
    # --------------------------------------------------------------------------
    def test_14_dynamic_condition_to_specialty_mapping(self):
        # Diabetes & Glycemic Profile
        diab = get_specialty_mapping("diabetes")
        self.assertEqual(diab["primary_specialty"], "Endocrinology / Diabetology")
        for kw in ["diabetes", "diabetologist", "endocrinology", "sugar clinic", "endocrine"]:
            self.assertIn(kw, diab["search_keywords"])

        # Pneumonia / Respiratory Infections
        pneu = get_specialty_mapping("pneumonia")
        self.assertEqual(pneu["primary_specialty"], "Pulmonology / Chest Medicine")
        for kw in ["pulmonology", "chest", "respiratory", "lung", "thoracic", "general medicine"]:
            self.assertIn(kw, pneu["search_keywords"])

        # Cardiovascular / Hypertension
        heart = get_specialty_mapping("heart")
        self.assertEqual(heart["primary_specialty"], "Cardiology")
        for kw in ["cardiology", "heart", "cardiac", "vascular"]:
            self.assertIn(kw, heart["search_keywords"])

        # Kidney / Renal Profile
        kidney = get_specialty_mapping("kidney")
        self.assertEqual(kidney["primary_specialty"], "Nephrology")
        for kw in ["nephrology", "kidney", "dialysis", "renal"]:
            self.assertIn(kw, kidney["search_keywords"])

        # Neurological
        neuro = get_specialty_mapping("neurological")
        self.assertEqual(neuro["primary_specialty"], "Neurology")
        for kw in ["neurology", "brain", "stroke", "neuro"]:
            self.assertIn(kw, neuro["search_keywords"])

        # Default / General
        gen = get_specialty_mapping("unknown_disease_xyz")
        self.assertEqual(gen["primary_specialty"], "Internal Medicine / General Physician")

        print("✅ Test 14 Passed: Dynamic condition-to-specialty mappings verified across all clinical domains.")

    # --------------------------------------------------------------------------
    # 15. GOOGLE MAPS DIRECTIONS URL FROM USER ENTERED ADDRESS
    # --------------------------------------------------------------------------
    def test_15_google_maps_directions_url_with_user_address(self):
        sample_facility = [{
            "id": "hosp-nav-test",
            "name": "Apollo Speciality Hospital",
            "facility_type": "Multi-Specialty Hospital",
            "latitude": 13.0150,
            "longitude": 79.9200,
            "address": "Vanagaram - Ambattur Main Rd, Chennai",
            "distance_km": 3.5,
            "tags": {}
        }]

        # 1. Manual entered address as origin
        user_addr = "Saveetha Nagar, Thandalam, Chennai"
        recs = score_and_recommend_hospitals(
            sample_facility,
            condition_id="heart",
            patient_lat=13.0143,
            patient_lon=79.9181,
            patient_address=user_addr
        )
        self.assertTrue(len(recs) > 0)
        url = recs[0].get("directions_url", "")
        self.assertIn("google.com/maps/dir/", url)
        self.assertIn("api=1", url)
        self.assertIn("origin=", url)
        self.assertIn("destination=13.015%2C79.92", url.replace(",", "%2C"))
        self.assertIn("Saveetha", url)
        self.assertIn("Thandalam", url)
        self.assertIn("travelmode=driving", url)

        # 2. GPS coordinates as origin when manual text is not provided
        recs_gps = score_and_recommend_hospitals(
            sample_facility,
            condition_id="heart",
            patient_lat=13.0143,
            patient_lon=79.9181,
            patient_address=None
        )
        url_gps = recs_gps[0].get("directions_url", "")
        self.assertIn("origin=13.0143%2C79.9181", url_gps.replace(",", "%2C"))

        # 3. Google Plus Code sanitized in origin
        recs_plus = score_and_recommend_hospitals(
            sample_facility,
            condition_id="diabetes",
            patient_lat=10.0823,
            patient_lon=78.7762,
            patient_address="3QMG+W9F, Sekkalai, Karaikudi, Tamil Nadu 630002"
        )
        url_plus = recs_plus[0].get("directions_url", "")
        self.assertNotIn("3QMG+W9F", url_plus)
        self.assertNotIn("3QMG%2BW9F", url_plus)
        self.assertIn("Sekkalai", url_plus)
        self.assertIn("Karaikudi", url_plus)

        print("✅ Test 15 Passed: Google Maps directions URL correctly generates with user entered address as origin.")


if __name__ == "__main__":
    unittest.main()

