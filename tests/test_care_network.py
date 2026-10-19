"""
Comprehensive Acceptance Test Suite for Strictly Verified Disease-Specific Hospital Discovery
=============================================================================================
Tests:
- Test A: Pneumonia in nearby area (Verified Pulmonology, no generic medicine, rating > 3.0)
- Test B: Pneumonia in Delhi (Geocoded Delhi, radius 25km, no Chennai leakage, authentic data)
- Test C: Radius switching & Progressive auto-expansion (5 -> 10 -> 25 -> 50 -> 100 km)
- Test D: Strict rating filtering (hospital.rating > 3.0 strict exclusive)
- Test E: Verification filtering (legal registration, accreditation, govt status separated)
- Test F: Specialty isolation (Pneumonia, Heart, Neuro, Kidney, Diabetes, Cancer, Liver, Malaria)
- Test G: No eligible results (Stops at 100 km, informative empty state, no fabricated filler)
- Test H: API reliability & layout contract (400 on empty, invalid queries, response contract)
"""

import unittest
import json
from app import app
from utils.hospital_service import (
    DISEASE_SPECIALTY_MAP,
    get_specialty_mapping,
    filter_and_rank_hospitals,
    discover_hospitals_for_patient,
    is_non_clinical_facility,
    resolve_any_address,
    calculate_haversine_distance
)
from utils.verified_hospital_registry import VERIFIED_HOSPITAL_REGISTRY


class TestCareNetworkAcceptance(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    # --------------------------------------------------------------------------
    # Test A: Pneumonia in a nearby area (Chennai Central test anchor)
    # --------------------------------------------------------------------------
    def test_a_pneumonia_nearby_filtering(self):
        """Pneumonia must only return verified Pulmonology/Respiratory departments with rating > 3.0."""
        # Chennai Central anchor (13.0827, 80.2707)
        res = discover_hospitals_for_patient(
            condition_id="pneumonia",
            patient_lat=13.0827,
            patient_lon=80.2707,
            radius_km=25,
            min_rating=3.0
        )
        self.assertEqual(res["status"], "success")
        self.assertGreater(len(res["recommendations"]), 0, "Should find verified respiratory facilities in Chennai")

        for hosp in res["recommendations"]:
            # Every hospital must have confirmed pulmonology/respiratory services
            has_pulmonology = any(
                "pulmon" in s.lower() or "respirat" in s.lower() or "chest" in s.lower() or "thorac" in s.lower()
                for s in hosp.get("confirmed_specialties", [])
            )
            dept_text = hosp.get("department_evidence_text", "").lower()
            self.assertTrue(
                has_pulmonology or "pulmon" in dept_text or "respirat" in dept_text or "chest" in dept_text,
                f"Hospital {hosp['name']} does not have confirmed Pulmonology services!"
            )
            # Rating strictly > 3.0
            self.assertGreater(hosp["rating"], 3.0, f"Hospital {hosp['name']} rating {hosp['rating']} not > 3.0")

            # Must have official accreditation
            self.assertIn(
                hosp["accreditation_status"],
                ["NABH Accredited", "JCI Accredited", "Government Specialty Institute"],
                f"Hospital {hosp['name']} lacks official accreditation"
            )

            # Must not be an alternative medicine or dental clinic
            self.assertFalse(is_non_clinical_facility(hosp["name"], hosp.get("tags"), "pneumonia"))

    # --------------------------------------------------------------------------
    # Test B: Pneumonia in Delhi
    # --------------------------------------------------------------------------
    def test_b_pneumonia_in_delhi(self):
        """Pneumonia in Delhi must return Delhi respiratory centres without GPS/Chennai leakage."""
        # Geocode Delhi
        geo = resolve_any_address("Delhi")
        self.assertIsNotNone(geo, "Delhi must be successfully geocoded")
        self.assertAlmostEqual(geo["latitude"], 28.6139, delta=0.2)
        self.assertAlmostEqual(geo["longitude"], 77.2090, delta=0.2)

        res = discover_hospitals_for_patient(
            condition_id="pneumonia",
            patient_lat=geo["latitude"],
            patient_lon=geo["longitude"],
            manual_query="Delhi",
            radius_km=25,
            min_rating=3.0
        )
        self.assertEqual(res["status"], "success")
        recs = res["recommendations"]
        self.assertGreater(len(recs), 0, "Should find Delhi pulmonology hospitals within 25 km")

        for hosp in recs:
            # Must be in Delhi NCR region (distance within 25 km from Delhi center)
            self.assertLessEqual(hosp["distance_km"], 25.0)
            self.assertIn("Delhi", hosp["address"] + " " + hosp.get("city", "") + " " + hosp.get("state", ""))
            # Must NOT be a Chennai hospital (no leakage)
            self.assertNotIn("Chennai", hosp.get("city", ""))
            self.assertNotIn("Tamil Nadu", hosp.get("state", ""))
            # Rating > 3.0
            self.assertGreater(hosp["rating"], 3.0)

        # Confirm prestigious Delhi respiratory centres are present
        hosp_names = [h["name"] for h in recs]
        found_delhi_pulm = any("NITRD" in n or "AIIMS" in n or "Ganga Ram" in n or "VPCI" in n or "Max" in n or "Apollo" in n for n in hosp_names)
        self.assertTrue(found_delhi_pulm, "Delhi respiratory centres should be returned")

    # --------------------------------------------------------------------------
    # Test C: Radius switching & Progressive Auto-Expansion
    # --------------------------------------------------------------------------
    def test_c_radius_switching_and_auto_expansion(self):
        """Radius switching returns hospitals within that distance, auto-expanding if 0 found."""
        # Location in Gurgaon / South Delhi border (28.4800, 77.0800)
        # Search at 5 km
        res_5 = discover_hospitals_for_patient(
            condition_id="heart",
            patient_lat=28.4800,
            patient_lon=77.0800,
            radius_km=5,
            min_rating=3.0
        )
        self.assertEqual(res_5["status"], "success")
        if len(res_5["recommendations"]) == 0:
            # Auto-expansion should have triggered
            self.assertTrue(res_5["location"]["auto_expanded"])
            self.assertGreater(res_5["location"]["effective_radius_km"], 5)
            self.assertIn("Search expanded to", res_5["location"]["expansion_message"])

        # Manual search at 50 km must return facilities within 50 km
        res_50 = discover_hospitals_for_patient(
            condition_id="heart",
            patient_lat=28.6139,
            patient_lon=77.2090,
            radius_km=50,
            min_rating=3.0
        )
        for h in res_50["recommendations"]:
            self.assertLessEqual(h["distance_km"], 50.0)

    # --------------------------------------------------------------------------
    # Test D: Strict rating filtering (hospital.rating > 3.0)
    # --------------------------------------------------------------------------
    def test_d_strict_rating_filter(self):
        """Hospitals with rating <= 3.0 or missing rating must be strictly excluded."""
        # Synthetic test fixtures
        mock_candidates = [
            {
                "id": "hosp-good",
                "name": "Excellent Pulmonology Hospital",
                "facility_type": "Tertiary Hospital",
                "condition_ids": ["pneumonia"],
                "confirmed_specialties": ["Pulmonology"],
                "department_evidence": {"pneumonia": "Respiratory ICU on site"},
                "legal_registration_status": "Licensed Hospital #123",
                "accreditation_status": "NABH Accredited",
                "government_facility_status": "Private",
                "rating": 4.2,
                "review_count": 500,
                "latitude": 28.6139,
                "longitude": 77.2090
            },
            {
                "id": "hosp-boundary-3-0",
                "name": "Marginal Hospital Exactly 3.0",
                "facility_type": "Tertiary Hospital",
                "condition_ids": ["pneumonia"],
                "confirmed_specialties": ["Pulmonology"],
                "department_evidence": {"pneumonia": "Pulmonology ward"},
                "legal_registration_status": "Licensed Hospital #124",
                "accreditation_status": "NABH Accredited",
                "government_facility_status": "Private",
                "rating": 3.0,  # Boundary case: exactly 3.0 must be EXCLUDED
                "review_count": 120,
                "latitude": 28.6139,
                "longitude": 77.2090
            },
            {
                "id": "hosp-bad-2-8",
                "name": "Low Rated Hospital",
                "facility_type": "Tertiary Hospital",
                "condition_ids": ["pneumonia"],
                "confirmed_specialties": ["Pulmonology"],
                "department_evidence": {"pneumonia": "Pulmonology ward"},
                "legal_registration_status": "Licensed Hospital #125",
                "accreditation_status": "NABH Accredited",
                "government_facility_status": "Private",
                "rating": 2.8,  # Below 3.0 must be EXCLUDED
                "review_count": 40,
                "latitude": 28.6139,
                "longitude": 77.2090
            },
            {
                "id": "hosp-unrated",
                "name": "Unrated Hospital Without Review Score",
                "facility_type": "Tertiary Hospital",
                "condition_ids": ["pneumonia"],
                "confirmed_specialties": ["Pulmonology"],
                "department_evidence": {"pneumonia": "Pulmonology ward"},
                "legal_registration_status": "Licensed Hospital #126",
                "accreditation_status": "NABH Accredited",
                "government_facility_status": "Private",
                "rating": None,  # Missing rating must be EXCLUDED
                "review_count": 0,
                "latitude": 28.6139,
                "longitude": 77.2090
            }
        ]

        filtered = filter_and_rank_hospitals(
            candidates=mock_candidates,
            condition_id="pneumonia",
            patient_lat=28.6139,
            patient_lon=77.2090,
            radius_km=25.0,
            min_rating=3.0
        )

        filtered_ids = [h["id"] for h in filtered]
        self.assertIn("hosp-good", filtered_ids)
        self.assertNotIn("hosp-boundary-3-0", filtered_ids, "Hospital with rating 3.0 must be excluded (strict > 3.0)")
        self.assertNotIn("hosp-bad-2-8", filtered_ids, "Hospital with rating 2.8 must be excluded")
        self.assertNotIn("hosp-unrated", filtered_ids, "Hospital with None rating must be excluded")

    # --------------------------------------------------------------------------
    # Test E: Verification filtering & Separate fields
    # --------------------------------------------------------------------------
    def test_e_verification_filtering(self):
        """Unverified facilities without legal/accreditation evidence must be excluded."""
        mock_candidates = [
            {
                "id": "hosp-unverified-osm",
                "name": "Random Clinic Found on Map",
                "facility_type": "Clinic",
                "condition_ids": ["pneumonia"],
                "confirmed_specialties": ["Pulmonology"],
                "legal_registration_status": "Unverified",
                "accreditation_status": "Unverified",
                "government_facility_status": "None",
                "rating": 4.5,
                "review_count": 200,
                "latitude": 28.6139,
                "longitude": 77.2090
            }
        ]

        filtered = filter_and_rank_hospitals(
            candidates=mock_candidates,
            condition_id="pneumonia",
            patient_lat=28.6139,
            patient_lon=77.2090,
            radius_km=25.0,
            min_rating=3.0
        )
        self.assertEqual(len(filtered), 0, "Unverified facility without accreditation/registration must be excluded")

        # Verify separation of fields in verified registry
        for hosp in VERIFIED_HOSPITAL_REGISTRY:
            self.assertIn("legal_registration_status", hosp)
            self.assertIn("accreditation_status", hosp)
            self.assertIn("government_facility_status", hosp)
            self.assertIn("specialty_verification_status", hosp)
            self.assertIn("verification_source", hosp)
            self.assertIn("verification_source_url", hosp)
            self.assertIn("verification_date", hosp)
            self.assertIn("verification_confidence", hosp)

    # --------------------------------------------------------------------------
    # Test F: Specialty Isolation across 8 Conditions
    # --------------------------------------------------------------------------
    def test_f_specialty_isolation(self):
        """Each condition must isolate and return only its clinically verified specialty."""
        conditions_to_test = {
            "pneumonia": ["pulmon", "respirat", "chest"],
            "heart": ["cardio", "heart", "ctvs"],
            "neurological": ["neuro", "stroke", "brain"],
            "diabetes": ["endocrin", "diabet"],
            "kidney": ["nephro", "renal", "dialysis"],
            "liver": ["hepato", "liver", "gastro"],
            "cancer": ["oncol", "cancer", "tumor"],
            "malaria": ["infectious", "tropical"]
        }

        for cond_id, expected_stems in conditions_to_test.items():
            res = discover_hospitals_for_patient(
                condition_id=cond_id,
                patient_lat=28.6139,
                patient_lon=77.2090,
                radius_km=50,
                min_rating=3.0
            )
            self.assertEqual(res["status"], "success", f"Search failed for {cond_id}")
            recs = res["recommendations"]
            self.assertGreater(len(recs), 0, f"Expected verified facilities for {cond_id}")

            for h in recs:
                # Must match at least one of the clinical stems for this condition
                specs = " ".join(h.get("confirmed_specialties", [])).lower()
                dept = h.get("department_evidence_text", "").lower()
                matched = any(stem in specs or stem in dept for stem in expected_stems)
                self.assertTrue(matched, f"Hospital {h['name']} returned for {cond_id} without matching {expected_stems}")

    # --------------------------------------------------------------------------
    # Test G: No eligible results (Stops at 100km, informative empty state)
    # --------------------------------------------------------------------------
    def test_g_no_eligible_results(self):
        """In a remote location with 0 facilities, system expands to 100km then returns empty state."""
        # Remote coordinate in the Indian Ocean (0.0, 75.0)
        res = discover_hospitals_for_patient(
            condition_id="pneumonia",
            patient_lat=0.0,
            patient_lon=75.0,
            radius_km=5,
            min_rating=3.0
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(len(res["recommendations"]), 0, "No hospitals should be returned in ocean coordinates")
        self.assertEqual(res["location"]["effective_radius_km"], 100, "Should have expanded up to 100 km before stopping")
        self.assertFalse(res["location"]["auto_expanded"], "auto_expanded should be False when 0 results found")
        self.assertIn("No verified", res["location"]["expansion_message"])

    # --------------------------------------------------------------------------
    # Test H: API reliability, Geocoding & Layout Contract
    # --------------------------------------------------------------------------
    def test_h_api_reliability_and_contract(self):
        """API endpoints must return proper structure, handle missing inputs, and disallow invalid conditions."""
        # 1. Geocode endpoint
        resp = self.client.get('/api/hospitals/geocode?query=Delhi')
        self.assertEqual(resp.status_code, 200)
        geo_data = json.loads(resp.data)
        self.assertEqual(geo_data["status"], "success")
        self.assertIn("latitude", geo_data["location"])

        # Empty geocode query
        resp_bad = self.client.get('/api/hospitals/geocode?query=')
        self.assertEqual(resp_bad.status_code, 400)

        # 2. Search endpoint contract
        resp_search = self.client.get('/api/hospitals/search?condition=pneumonia&lat=28.6139&lon=77.2090&radius_km=25')
        self.assertEqual(resp_search.status_code, 200)
        search_data = json.loads(resp_search.data)
        self.assertEqual(search_data["status"], "success")
        self.assertIn("recommendations", search_data)
        self.assertIn("location", search_data)
        self.assertIn("filters_applied", search_data)

        # Check required fields in each recommendation
        if search_data["recommendations"]:
            first = search_data["recommendations"][0]
            for key in ["id", "name", "rating", "review_count", "distance_km", "verified_badges", "legal_registration_status", "accreditation_status"]:
                self.assertIn(key, first, f"Missing required response field {key}")

        # 3. Non-clinical exclusion of dental / ayurvedic
        self.assertTrue(is_non_clinical_facility("Dr. Smile Dental Care", {}, "pneumonia"))
        self.assertTrue(is_non_clinical_facility("Ayurvedic Wellness Hospital", {}, "pneumonia"))
        self.assertTrue(is_non_clinical_facility("Rotary Blood Bank", {}, "pneumonia"))
        self.assertTrue(is_non_clinical_facility("City Eye Institute", {}, "pneumonia"))


if __name__ == "__main__":
    unittest.main()
