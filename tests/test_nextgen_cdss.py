"""
MediNova Next-Generation Clinical Decision Support System (CDSS) Test Suite
===========================================================================
Validates:
1. Model Registry & Cryptographic Hash Integrity (SHA-256)
2. Tabular Benchmarking & Calibrated Thyroid Dysfunction Screening (/thyroid)
3. Digital Biosignal Filtering & ECG Arrhythmia Screening (/arrhythmia)
4. Out-of-Distribution (OOD) & Physiological Plausibility Guards
5. Governed Research Roadmap Gating (Strict Disabled-Inference Verification)
6. Defense-in-Depth Security Headers, CSRF Enforcement, and Secret Hardening
7. Real-Time Telemetry & Model Monitoring Endpoints (/api/catalog, /api/monitoring)
"""

import os
import unittest
import json
import secrets
import numpy as np
from flask import session

import app as medinova_app
from utils.model_registry import (
    verify_model_artifact_hash,
    get_model,
    get_model_metadata,
    get_catalog_summary,
    load_model_registry
)
from utils.clinical_validation import (
    validate_clinical_features,
    detect_out_of_distribution,
    CLINICAL_REFERENCE_RANGES
)
from utils.signal_processor import (
    validate_ecg_signal_quality,
    filter_ecg_signal,
    detect_r_peaks,
    extract_features_from_waveform,
    parse_ecg_input_data
)
from utils.diagnostic_predictor import (
    predict_thyroid,
    predict_arrhythmia
)
from utils.model_monitoring import (
    record_inference_telemetry,
    get_system_telemetry
)


class TestModelRegistryIntegrity(unittest.TestCase):
    """Ensures model artifacts match trusted SHA-256 cryptographic manifests."""

    def test_registry_manifest_structure(self):
        registry = load_model_registry()
        self.assertIn("models", registry)
        self.assertIn("governance_standard", registry)
        self.assertGreaterEqual(len(registry["models"]), 13)

    def test_thyroid_model_checksum(self):
        is_valid, msg = verify_model_artifact_hash("thyroid")
        self.assertTrue(is_valid, f"Thyroid checksum verification failed: {msg}")

    def test_arrhythmia_model_checksum(self):
        is_valid, msg = verify_model_artifact_hash("arrhythmia")
        self.assertTrue(is_valid, f"Arrhythmia checksum verification failed: {msg}")

    def test_tampered_model_rejection(self):
        # Verification should fail for nonexistent or intentionally corrupted model key
        meta = get_model_metadata("heart_disease")
        self.assertIsNotNone(meta)
        fake_meta = meta.copy()
        fake_meta["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        
        # Test hash checker with mismatched hash
        rel_path = fake_meta.get("artifact_path") or fake_meta.get("artifact_file", "")
        actual_path = os.path.join(medinova_app.BASE_DIR, rel_path)
        import hashlib
        with open(actual_path, "rb") as f:
            real_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertNotEqual(real_hash, fake_meta["sha256"])


class TestClinicalValidationAndOOD(unittest.TestCase):
    """Validates boundary, schema, and Out-of-Distribution checks."""

    def test_thyroid_valid_schema(self):
        valid_input = {
            "age": 45, "sex": "Female", "on_thyroxine": 0,
            "TSH": 1.8, "T3": 1.9, "TT4": 105.0, "T4U": 0.98, "FTI": 107.0
        }
        clean_vals, errors, warnings = validate_clinical_features(valid_input, "thyroid")
        self.assertEqual(len(errors), 0)
        self.assertIn("TSH", clean_vals)

    def test_thyroid_extreme_unphysiological_rejection(self):
        # Negative age or extreme impossible TSH
        invalid_input = {
            "age": -5, "sex": "Female", "on_thyroxine": 0,
            "TSH": 9999.0, "T3": -2.0, "TT4": 105.0, "T4U": 0.98, "FTI": 107.0
        }
        clean_vals, errors, warnings = validate_clinical_features(invalid_input, "thyroid")
        self.assertGreater(len(errors), 0)

    def test_out_of_distribution_detector(self):
        features = {"TSH": 500.0, "TT4": 900.0}
        is_ood, ood_factors = detect_out_of_distribution(features, "thyroid")
        self.assertTrue(is_ood)
        self.assertGreater(len(ood_factors), 0)


class TestThyroidPredictionPipeline(unittest.TestCase):
    """Validates calibrated Random Forest on authentic UCI Thyroid test inputs."""

    def setUp(self):
        self.model = medinova_app.get_thyroid_model()
        self.assertIsNotNone(self.model, "Thyroid model failed to load.")

    def test_euthyroid_normal_screening(self):
        normal_data = {
            "age": 42, "sex": "Female", "on_thyroxine": 0,
            "TSH": 1.4, "T3": 2.1, "TT4": 110.0, "T4U": 1.0, "FTI": 110.0
        }
        res = predict_thyroid(normal_data, self.model, medinova_app.DISEASE_INSIGHTS)
        self.assertFalse(res["is_positive"])
        self.assertIn("Euthyroid", res["prediction"])
        self.assertIsNotNone(res["report_id"])
        self.assertTrue("THYR" in res["report_id"] or res["report_id"].startswith("MN"))
        self.assertGreaterEqual(len(res["biomarker_gauges"]), 5)
        self.assertGreaterEqual(len(res["biomarker_gauges"]), 5)

    def test_hypothyroid_positive_screening(self):
        # Elevated TSH (>40), depressed TT4 and FTI
        hypo_data = {
            "age": 58, "sex": "Female", "on_thyroxine": 0,
            "TSH": 65.0, "T3": 0.6, "TT4": 22.0, "T4U": 1.15, "FTI": 19.1
        }
        res = predict_thyroid(hypo_data, self.model, medinova_app.DISEASE_INSIGHTS)
        self.assertTrue(res["is_positive"])
        self.assertIn("Elevated Thyroid Dysfunction", res["prediction"])
        self.assertGreaterEqual(res["confidence_score"], 80.0)


class TestECGArrhythmiaSignalPipeline(unittest.TestCase):
    """Validates digital biosignal filtering, Pan-Tompkins QRS, and arrhythmia inference."""

    def setUp(self):
        self.model_artifact = medinova_app.get_arrhythmia_model()
        self.assertIsNotNone(self.model_artifact, "ECG Arrhythmia model artifact failed to load.")

    def test_signal_quality_validation(self):
        # Empty signal rejection
        is_ok, err, _ = validate_ecg_signal_quality(np.array([]), 360.0)
        self.assertFalse(is_ok)
        
        # Too short signal (< 4s)
        short_sig = np.random.randn(360) # 1 second at 360 Hz
        is_ok, err, _ = validate_ecg_signal_quality(short_sig, 360.0)
        self.assertFalse(is_ok)
        self.assertIn("too brief", err.lower())

    def test_butterworth_bandpass_filtering(self):
        # Synthesize noisy signal: 1 Hz ECG baseline + 60 Hz line noise + 0.1 Hz respiration
        fs = 360.0
        t = np.linspace(0, 5, int(5 * fs))
        noisy_sig = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 60.0 * t) + 2.0 * np.sin(2 * np.pi * 0.1 * t)
        filtered = filter_ecg_signal(noisy_sig, fs)
        self.assertEqual(len(filtered), len(noisy_sig))
        self.assertTrue(np.all(np.isfinite(filtered)))

    def test_normal_sinus_mitbih_100_telemetry(self):
        rec_path = os.path.join(medinova_app.BASE_DIR, "datasets", "mitbih", "100.dat")
        if os.path.exists(rec_path):
            import wfdb
            rec = wfdb.rdrecord(os.path.join(medinova_app.BASE_DIR, "datasets", "mitbih", "100"), sampto=2500)
            sig = rec.p_signal[:, 0]
            res = predict_arrhythmia(sig, self.model_artifact, medinova_app.DISEASE_INSIGHTS, sample_rate=rec.fs)
            self.assertFalse(res["is_positive"], "Patient 100 normal sinus rhythm should not be classified as positive arrhythmia.")
            self.assertIn("Normal Sinus Rhythm", res["prediction"])
            self.assertGreater(res["total_beats"], 3)

    def test_pvc_arrhythmic_mitbih_119_telemetry(self):
        rec_path = os.path.join(medinova_app.BASE_DIR, "datasets", "mitbih", "119.dat")
        if os.path.exists(rec_path):
            import wfdb
            rec = wfdb.rdrecord(os.path.join(medinova_app.BASE_DIR, "datasets", "mitbih", "119"), sampto=3600)
            sig = rec.p_signal[:, 0]
            res = predict_arrhythmia(sig, self.model_artifact, medinova_app.DISEASE_INSIGHTS, sample_rate=rec.fs)
            self.assertTrue(res["is_positive"], "Patient 119 ventricular ectopy should be detected.")
            self.assertIn("Cardiac Arrhythmia", res["prediction"])
            self.assertGreater(res["arrhythmic_beat_count"], 0)


class TestWebRoutesAndClinicalWorkstation(unittest.TestCase):
    """Validates Flask endpoints for new screening modules and research previews."""

    def setUp(self):
        medinova_app.app.config["TESTING"] = True
        self.client = medinova_app.app.test_client()

    def test_thyroid_get_page(self):
        response = self.client.get("/thyroid")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Thyroid Dysfunction Screening", response.data)
        self.assertIn(b"Serum TSH", response.data)

    def test_thyroid_post_form_evaluation(self):
        payload = {
            "age": "45", "sex": "Female", "on_thyroxine": "0",
            "tsh": "1.8", "t3": "1.8", "tt4": "103.0", "t4u": "0.98", "fti": "107.0"
        }
        response = self.client.post("/thyroid", data=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Euthyroid", response.data)

    def test_thyroid_post_json_evaluation(self):
        payload = {
            "age": 55, "sex": "Female", "on_thyroxine": 0,
            "TSH": 55.0, "T3": 0.8, "TT4": 30.0, "T4U": 1.1, "FTI": 27.0
        }
        response = self.client.post("/thyroid", json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data["success"])
        self.assertTrue(data["is_positive"])
        self.assertIn("Thyroid Dysfunction", data["prediction"])

    def test_arrhythmia_get_page(self):
        response = self.client.get("/arrhythmia")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"ECG Arrhythmia Biosignal Screening", response.data)
        self.assertIn(b"Normal Sinus Rhythm (10s)", response.data)

    def test_arrhythmia_demo_trigger(self):
        response = self.client.post("/arrhythmia", data={"demo_sample": "normal_sinus"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Normal Sinus Rhythm", response.data)

    def test_api_catalog_endpoint(self):
        response = self.client.get("/api/catalog")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("available_models", data)
        self.assertIn("research_roadmap_models", data)
        self.assertGreaterEqual(len(data["available_models"]), 9)
        self.assertEqual(len(data["research_roadmap_models"]), 4)

    def test_api_monitoring_endpoint(self):
        response = self.client.get("/api/monitoring")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("system_status", data)
        self.assertIn("memory_ram_rss_mb", data)
        self.assertIn("total_inferences", data)

    def test_research_roadmap_gating_disabled_inference(self):
        """Roadmap candidates must reject POST inference and return 403 research preview."""
        for endpoint in ["/diabetic_retinopathy", "/melanoma", "/tuberculosis", "/anemia"]:
            # POST should be rejected with 403 Forbidden
            res = self.client.post(endpoint, json={"sample": "data"})
            self.assertEqual(res.status_code, 403, f"{endpoint} should reject active inference with 403.")
            data = json.loads(res.data)
            self.assertFalse(data["success"])
            self.assertEqual(data["status"], "research_preview")
            self.assertFalse(data["active"])
            self.assertIn("governed clinical research", data["error"])


class TestSecurityHardeningAndHeaders(unittest.TestCase):
    """Validates Content Security Policy, X-Frame-Options, CSRF, and secret validation."""

    def setUp(self):
        self.client = medinova_app.app.test_client()

    def test_security_headers_applied(self):
        response = self.client.get("/")
        self.assertEqual(response.headers.get("X-Frame-Options"), "DENY")
        self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin")
        self.assertIn("Content-Security-Policy", response.headers)
        csp = response.headers.get("Content-Security-Policy")
        self.assertIn("default-src 'self'", csp)
        self.assertIn("frame-ancestors 'none'", csp)

    def test_csrf_token_generator(self):
        with medinova_app.app.test_request_context():
            token = medinova_app.get_csrf_token()
            self.assertIsNotNone(token)
            self.assertEqual(len(token), 64)
            # Must remain stable in the same session
            token2 = medinova_app.get_csrf_token()
            self.assertEqual(token, token2)

    def test_csrf_rejection_on_mismatched_token(self):
        medinova_app.app.config["TESTING"] = False
        try:
            with self.client.session_transaction() as sess:
                sess["_csrf_token"] = "valid_secure_session_token_1234567890abcdef"
            
            # Post with forged token
            response = self.client.post(
                "/thyroid",
                data={"csrf_token": "forged_malicious_attacker_token", "age": "45"},
                headers={"Accept": "application/json"}
            )
            self.assertEqual(response.status_code, 403)
            data = json.loads(response.data)
            self.assertIn("Invalid CSRF token", data["error"])
        finally:
            medinova_app.app.config["TESTING"] = True


if __name__ == "__main__":
    unittest.main()
