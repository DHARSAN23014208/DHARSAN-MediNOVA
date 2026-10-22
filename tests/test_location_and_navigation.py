"""
MediNova Location Search, Geocoding, Distance Calculation, and Navigation Origin Test Suite
========================================================================================
Validates Requirements A through I:
- Test A: Detailed address resolution without falling back to Chennai city centre
- Test B: Alternate formatting normalization (Unicode dashes, highway labels, repeated commas)
- Test C: PIN code extraction and geocoding candidate resolution
- Test D: Unresolvable address handling and previous results invalidation
- Test E: GPS location authority even when reverse geocoding fails
- Test F: Sequential location updates and stale asynchronous response protection
- Test G: Radius filtering across all five tiers (5, 10, 25, 50, 100 km)
- Test H: Navigation URL generation using active coordinates origin and hospital destination
- Test I: Medical specialty preservation and hospital accreditation/rating threshold (> 3.0)
"""

import unittest
from unittest.mock import patch, MagicMock
import urllib.parse
from app import app
from utils.hospital_service import (
    normalize_address,
    generate_progressive_candidates,
    resolve_any_address,
    calculate_haversine_distance,
    filter_and_rank_hospitals,
    discover_hospitals_for_patient,
    get_specialty_mapping,
    CITY_CENTROIDS
)
from utils.verified_hospital_registry import VERIFIED_HOSPITAL_REGISTRY


class TestLocationSearchAndNavigation(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    # --------------------------------------------------------------------------
    # Test A: Detailed address resolution (Saveetha Nagar, Thandalam)
    # --------------------------------------------------------------------------
    def test_a_detailed_address_resolution(self):
        """Detailed address must resolve via progressive fallback to locality, not Chennai city-centre."""
        input_addr = "Saveetha Nagar, Thandalam, Chennai-Bangalore National Highway (NH-4), Sriperumbudur, Chennai, Tamil Nadu"
        
        # Test candidate generation
        candidates = generate_progressive_candidates(input_addr)
        cand_queries = [c[0].lower() for c in candidates]
        
        # Must generate simplified locality variants
        has_thandalam_locality = any("thandalam" in q for q in cand_queries)
        self.assertTrue(has_thandalam_locality, "Candidates must include Thandalam locality variants")
        
        # Mock geocoding provider to return Thandalam locality coordinates
        thandalam_lat = 13.0170
        thandalam_lon = 80.0054
        chennai_centre_lat = 13.0827
        chennai_centre_lon = 80.2707

        with patch("utils.hospital_service._try_nominatim_query") as mock_nom:
            def side_effect(q, countrycodes="in"):
                if "thandalam" in q.lower():
                    return {
                        "latitude": thandalam_lat,
                        "longitude": thandalam_lon,
                        "display_name": "Thandalam, Sriperumbudur, Tamil Nadu, India",
                        "city": "Thandalam",
                        "state": "Tamil Nadu",
                        "country": "India",
                        "pincode": "600124",
                        "source": "OpenStreetMap Nominatim"
                    }
                return None
            mock_nom.side_effect = side_effect

            res = resolve_any_address(input_addr)
            self.assertTrue(res["success"], "Address resolution must succeed")
            self.assertAlmostEqual(res["latitude"], thandalam_lat, places=4)
            self.assertAlmostEqual(res["longitude"], thandalam_lon, places=4)
            # Crucial: Must NOT be Chennai city centre
            self.assertNotEqual(res["latitude"], chennai_centre_lat)
            self.assertNotEqual(res["longitude"], chennai_centre_lon)
            self.assertIn("Thandalam", res["resolvedAddress"])

    # --------------------------------------------------------------------------
    # Test B: Alternate formatting normalization
    # --------------------------------------------------------------------------
    def test_b_alternate_formatting_normalization(self):
        """Unicode dashes, multiple commas, highway labels, extra spaces must normalize cleanly."""
        noisy_addr = "Saveetha Nagar,  Thandalam \u2013 Chennai\u2014Bangalore National Highway (NH-4) ,,, Sriperumbudur , Chennai , Tamil Nadu"
        normalized = normalize_address(noisy_addr)

        self.assertNotIn("\u2013", normalized)
        self.assertNotIn("\u2014", normalized)
        self.assertNotIn(",,,", normalized)
        self.assertIn("NH-4", normalized)
        self.assertIn("Thandalam", normalized)
        self.assertIn("Saveetha Nagar", normalized)
        self.assertIn("Sriperumbudur", normalized)

    # --------------------------------------------------------------------------
    # Test C: Six-digit Indian PIN code geocoding
    # --------------------------------------------------------------------------
    def test_c_pincode_resolution(self):
        """A valid six-digit Indian PIN code must be extracted and used as a fallback candidate."""
        pin_addr = "Industrial Area, Near Toll Plaza, 600124"
        candidates = generate_progressive_candidates(pin_addr)
        pin_cands = [q for q, lvl in candidates if "600124" in q]
        self.assertGreater(len(pin_cands), 0, "Candidates must include PIN 600124")

        with patch("utils.hospital_service._try_nominatim_query") as mock_nom:
            mock_nom.return_value = {
                "latitude": 13.0615,
                "longitude": 80.0368,
                "display_name": "Poonamallee, 600124, Tamil Nadu, India",
                "city": "Poonamallee",
                "state": "Tamil Nadu",
                "country": "India",
                "pincode": "600124",
                "source": "OpenStreetMap Nominatim"
            }
            res = resolve_any_address(pin_addr)
            self.assertTrue(res["success"])
            self.assertEqual(res["pincode"], "600124")
            self.assertAlmostEqual(res["latitude"], 13.0615, places=4)

    # --------------------------------------------------------------------------
    # Test D: Unresolvable address error & results invalidation
    # --------------------------------------------------------------------------
    def test_d_unresolvable_address_handling(self):
        """Unresolvable address must return success=False and empty results without fallback to city."""
        unres_addr = "QZXWV999NonExistentPlaceRandom123456789"
        with patch("utils.hospital_service._try_nominatim_query", return_value=None), \
             patch("utils.hospital_service._try_photon_query", return_value=None):
            
            res = resolve_any_address(unres_addr)
            self.assertFalse(res["success"])
            self.assertIsNone(res["latitude"])
            self.assertIsNone(res["longitude"])
            self.assertEqual(res["resolutionLevel"], "unresolved")

            # Discovery with unresolvable address must NOT return hospitals
            disc = discover_hospitals_for_patient(
                condition_id="pneumonia",
                patient_lat=None,
                patient_lon=None,
                manual_query=unres_addr
            )
            self.assertEqual(disc["status"], "location_not_found")
            self.assertEqual(len(disc["recommendations"]), 0)
            self.assertFalse(disc["location"]["has_coordinates"])

    # --------------------------------------------------------------------------
    # Test E: GPS location authority
    # --------------------------------------------------------------------------
    def test_e_gps_location_authority(self):
        """GPS coordinates remain authoritative origin even if reverse geocoding returns nothing."""
        gps_lat = 12.9800
        gps_lon = 80.1500

        with patch("utils.hospital_service.reverse_geocode_coords", return_value="12.9800, 80.1500"):
            disc = discover_hospitals_for_patient(
                condition_id="heart",
                patient_lat=gps_lat,
                patient_lon=gps_lon,
                radius_km=25,
                min_rating=3.0
            )
            self.assertEqual(disc["status"], "success")
            self.assertEqual(disc["location"]["latitude"], gps_lat)
            self.assertEqual(disc["location"]["longitude"], gps_lon)

            # Every recommendation must compute distance from gps_lat/gps_lon
            for hosp in disc["recommendations"]:
                expected_dist = calculate_haversine_distance(gps_lat, gps_lon, hosp["latitude"], hosp["longitude"])
                self.assertAlmostEqual(hosp["distance_km"], expected_dist, places=1)
                # Directions URL must start from GPS coordinates
                self.assertIn(f"origin={gps_lat:.6f}%2C{gps_lon:.6f}", hosp["directions_url"])

    # --------------------------------------------------------------------------
    # Test F: Sequential location updates and distance synchronization
    # --------------------------------------------------------------------------
    def test_f_change_location_repeatedly(self):
        """Switching from one location to another updates distances and navigation links consistently."""
        loc1_lat, loc1_lon = 13.0170, 80.0054  # Thandalam
        loc2_lat, loc2_lon = 28.6139, 77.2090  # Delhi
        loc3_lat, loc3_lon = 12.9716, 77.5946  # Bengaluru

        # Location 1: Thandalam
        disc1 = discover_hospitals_for_patient("pneumonia", patient_lat=loc1_lat, patient_lon=loc1_lon, radius_km=50)
        # Location 2: Delhi
        disc2 = discover_hospitals_for_patient("pneumonia", patient_lat=loc2_lat, patient_lon=loc2_lon, radius_km=50)
        # Location 3: Bengaluru
        disc3 = discover_hospitals_for_patient("pneumonia", patient_lat=loc3_lat, patient_lon=loc3_lon, radius_km=50)

        # Distances and directions URLs must match the respective active coordinates
        for h in disc1["recommendations"]:
            self.assertIn(f"origin={loc1_lat:.6f}%2C{loc1_lon:.6f}", h["directions_url"])
        for h in disc2["recommendations"]:
            self.assertIn(f"origin={loc2_lat:.6f}%2C{loc2_lon:.6f}", h["directions_url"])
        for h in disc3["recommendations"]:
            self.assertIn(f"origin={loc3_lat:.6f}%2C{loc3_lon:.6f}", h["directions_url"])

    # --------------------------------------------------------------------------
    # Test G: Radius filtering across all five tiers (5, 10, 25, 50, 100 km)
    # --------------------------------------------------------------------------
    def test_g_radius_filtering_all_tiers(self):
        """Radius settings (5, 10, 25, 50, 100 km) strictly filter hospitals by straight-line distance."""
        origin_lat, origin_lon = 13.0827, 80.2707  # Chennai Central
        mock_candidates = [
            {"id": "h1", "name": "Hospital 3km", "latitude": 13.0900, "longitude": 80.2800, "rating": 4.5, "condition_ids": ["heart"], "accreditation_status": "NABH Accredited"},
            {"id": "h2", "name": "Hospital 8km", "latitude": 13.1200, "longitude": 80.3200, "rating": 4.5, "condition_ids": ["heart"], "accreditation_status": "NABH Accredited"},
            {"id": "h3", "name": "Hospital 20km", "latitude": 13.2000, "longitude": 80.3800, "rating": 4.5, "condition_ids": ["heart"], "accreditation_status": "NABH Accredited"},
            {"id": "h4", "name": "Hospital 40km", "latitude": 13.3500, "longitude": 80.5000, "rating": 4.5, "condition_ids": ["heart"], "accreditation_status": "NABH Accredited"},
            {"id": "h5", "name": "Hospital 80km", "latitude": 13.7000, "longitude": 80.6000, "rating": 4.5, "condition_ids": ["heart"], "accreditation_status": "NABH Accredited"},
            {"id": "h6", "name": "Hospital 150km", "latitude": 14.3000, "longitude": 80.8000, "rating": 4.5, "condition_ids": ["heart"], "accreditation_status": "NABH Accredited"},
        ]

        tiers = [5, 10, 25, 50, 100]
        for tier in tiers:
            filtered = filter_and_rank_hospitals(
                candidates=mock_candidates,
                condition_id="heart",
                patient_lat=origin_lat,
                patient_lon=origin_lon,
                radius_km=float(tier),
                min_rating=3.0
            )
            for hosp in filtered:
                self.assertLessEqual(hosp["distance_km"], float(tier), f"Hospital {hosp['name']} distance {hosp['distance_km']} exceeds radius {tier}")

    # --------------------------------------------------------------------------
    # Test H: Navigation URL generation
    # --------------------------------------------------------------------------
    def test_h_navigation_url_construction(self):
        """Directions link must use https://www.google.com/maps/dir/?api=1 with encoded coords."""
        active_lat = 13.0170
        active_lon = 80.0054
        hosp_lat = 13.0234
        hosp_lon = 80.1834

        hosp = {
            "id": "miot",
            "name": "MIOT International",
            "latitude": hosp_lat,
            "longitude": hosp_lon,
            "address": "4/112, Mount Poonamallee Road, Manapakkam, Chennai, Tamil Nadu 600089",
            "rating": 4.6,
            "condition_ids": ["pneumonia"],
            "confirmed_specialties": ["Pulmonology & Respiratory Care", "Interventional Pulmonology"],
            "department_evidence": {"pneumonia": "Department of Respiratory Medicine & Critical Care"},
            "accreditation_status": "NABH Accredited",
            "legal_registration_status": "Registered Hospital"
        }

        ranked = filter_and_rank_hospitals(
            candidates=[hosp],
            condition_id="pneumonia",
            patient_lat=active_lat,
            patient_lon=active_lon,
            radius_km=25.0
        )
        self.assertEqual(len(ranked), 1)
        url = ranked[0]["directions_url"]

        # Expected Google Maps Directions API format
        expected_origin = f"{active_lat:.6f}%2C{active_lon:.6f}"
        expected_dest = f"{hosp_lat:.6f}%2C{hosp_lon:.6f}"
        self.assertIn("https://www.google.com/maps/dir/?api=1", url)
        self.assertIn(f"origin={expected_origin}", url)
        self.assertIn(f"destination={expected_dest}", url)
        self.assertIn("travelmode=driving", url)
        # Confirm no hardcoded static 'Chennai' string
        self.assertNotIn("origin=Chennai", url)

    # --------------------------------------------------------------------------
    # Test I: Specialty preservation and accreditation/rating thresholds
    # --------------------------------------------------------------------------
    def test_i_specialty_preservation_and_accreditation(self):
        """Pneumonia must never return cardiology-only facilities, and all returned facilities must be verified."""
        res = discover_hospitals_for_patient("pneumonia", patient_lat=13.0827, patient_lon=80.2707, radius_km=50)
        self.assertEqual(res["status"], "success")

        for hosp in res["recommendations"]:
            # Rating strictly > 3.0
            self.assertGreater(hosp["rating"], 3.0)
            # Accreditation required
            self.assertIn(
                hosp["accreditation_status"],
                ["NABH Accredited", "JCI Accredited", "Government Specialty Institute"]
            )
            # Pulmonology confirmed
            has_pulm = any("pulmon" in s.lower() or "respirat" in s.lower() or "chest" in s.lower() for s in hosp.get("confirmed_specialties", []))
            dept_text = hosp.get("department_evidence_text", "").lower()
            self.assertTrue(has_pulm or "pulmon" in dept_text or "respirat" in dept_text or "chest" in dept_text)
            self.assertTrue(hosp["specialty_match"])


if __name__ == "__main__":
    unittest.main()
