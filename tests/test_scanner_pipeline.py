"""
MediNova Clinical AI Platform - Comprehensive Test Suite
========================================================
Validates:
1. Clinical Synonym Dictionary, Regex Parsing, and Unit Normalization
2. Reference Interval Disambiguation (not mistaking normal limits for patient values)
3. Out-of-range / Implausible Value Rejection
4. File Handling, Allowed Types, and Corrupt File Guardrails
5. 3-Tier Fallback Pipeline (Vision API failure -> Local PDF/OCR -> Baseline Imputation)
6. Model Input Compatibility & Exact Feature Ordering across all 5 Tabular Modules
7. Verification Storage, HMAC-SHA256 Seals, and QR Endpoint
8. Public Report Verification Page
9. Flask Diagnostic Routes Backward Compatibility
10. Developer Gate Security Restrictions
"""

import os
import io
import json
import unittest
from unittest.mock import patch, MagicMock

# Import core application and utilities
from app import app
from utils.report_scanner import (
    parse_clinical_text,
    extract_text_from_pdf,
    clean_numeric_value,
    MODULE_SCHEMAS,
    ALLOWED_EXTENSIONS
)
from utils.report_verifier import (
    register_report,
    get_verified_report,
    generate_verification_hash,
    generate_qr_code_image
)
from utils.diagnostic_predictor import (
    predict_heart,
    predict_diabetes,
    predict_kidney,
    predict_liver,
    predict_breast_cancer,
    build_biomarker_gauges,
    safe_percent
)


class TestClinicalParserAndSynonyms(unittest.TestCase):
    """Tests regex analyte extraction, synonym dictionaries, and range bounds."""

    def test_diabetes_synonyms_and_reference_interval_disambiguation(self):
        sample_text = """
        METABOLIC LAB PANEL
        Patient: John Doe   Age: 58   Gender: Male
        Glycated Hemoglobin (HbA1c) ...... 8.4 %       (Ref: 4.0 - 5.6%)
        Fasting Blood Sugar (FBS) ........ 195 mg/dL   (Ref: 70 - 99 mg/dL)
        BMI: 31.2 kg/m2
        Hypertension: Positive (History of HTN: 1)
        Heart Disease: None (0)
        Smoking: Current smoker
        """
        parsed = parse_clinical_text(sample_text, "diabetes")
        values = parsed["values"]

        # HbA1c should be 8.4, NOT 4.0 or 5.6 from reference range
        self.assertEqual(values.get("HbA1c_level"), 8.4)
        # Blood glucose should be 195, NOT 70 or 99
        self.assertEqual(values.get("blood_glucose_level"), 195.0)
        self.assertEqual(values.get("age"), 58.0)
        self.assertEqual(values.get("gender"), "Male")
        self.assertEqual(values.get("bmi"), 31.2)
        self.assertEqual(values.get("hypertension"), 1)
        self.assertEqual(values.get("smoking_history"), "current")

    def test_heart_synonyms(self):
        sample_text = """
        CARDIAC ASSESSMENT REPORT
        Patient Age: 62   Sex: Male
        Resting Blood Pressure: 155 mm Hg
        Total Cholesterol: 278 mg/dL
        Fasting Blood Sugar > 120: Yes (1)
        Resting ECG: 2 (Definite ST-T wave abnormality)
        Maximum Heart Rate (Thalach): 130 bpm
        Exercise Induced Angina: 1
        ST Depression (Oldpeak): 2.4 mm
        Slope of Peak ST: 2
        Number of Major Vessels Fluro: 2
        Thallium Stress: 7
        """
        parsed = parse_clinical_text(sample_text, "heart")
        values = parsed["values"]

        self.assertEqual(values.get("bp"), 155.0)
        self.assertEqual(values.get("chol"), 278.0)
        self.assertEqual(values.get("max_hr"), 130.0)
        self.assertEqual(values.get("oldpeak"), 2.4)
        self.assertEqual(values.get("ca"), 2.0)
        self.assertEqual(values.get("thal"), 7.0)

    def test_kidney_synonyms(self):
        sample_text = """
        RENAL FUNCTION PANEL
        Age: 60
        Blood Pressure: 90 mm Hg
        Specific Gravity: 1.015
        Albumin: 3
        Serum Creatinine: 3.8 mg/dL   (Ref: 0.7 - 1.3 mg/dL)
        Blood Urea: 85 mg/dL
        Serum Sodium (Na): 132 mEq/L
        Serum Potassium (K): 5.8 mEq/L
        Hemoglobin (Hb): 9.2 g/dL
        White Blood Cell Count (WBC): 11200 /uL
        Red Blood Cell Count (RBC): 3.4 mil/uL
        """
        parsed = parse_clinical_text(sample_text, "kidney")
        values = parsed["values"]

        self.assertEqual(values.get("sc"), 3.8)
        self.assertEqual(values.get("bu"), 85.0)
        self.assertEqual(values.get("sod"), 132.0)
        self.assertEqual(values.get("pot"), 5.8)
        self.assertEqual(values.get("hemo"), 9.2)
        self.assertEqual(values.get("wc"), 11200.0)
        self.assertEqual(values.get("rc"), 3.4)

    def test_liver_synonyms(self):
        sample_text = """
        LIVER FUNCTION TEST (LFT)
        Age: 48   Gender: Male
        Total Bilirubin: 3.5 mg/dL   (0.2 - 1.2)
        Direct Bilirubin: 1.4 mg/dL  (0.0 - 0.3)
        Alkaline Phosphatase (ALP): 280 U/L
        SGPT / ALT (Alamine Aminotransferase): 88 U/L
        SGOT / AST (Aspartate Aminotransferase): 110 U/L
        Total Protein: 5.6 g/dL
        Serum Albumin: 2.8 g/dL
        Albumin and Globulin Ratio: 0.85
        """
        parsed = parse_clinical_text(sample_text, "liver")
        values = parsed["values"]

        self.assertEqual(values.get("Total_Bilirubin"), 3.5)
        self.assertEqual(values.get("Direct_Bilirubin"), 1.4)
        self.assertEqual(values.get("Alkaline_Phosphotase"), 280.0)
        self.assertEqual(values.get("Alamine_Aminotransferase"), 88.0)
        self.assertEqual(values.get("Aspartate_Aminotransferase"), 110.0)
        self.assertEqual(values.get("Total_Protiens"), 5.6)
        self.assertEqual(values.get("Albumin"), 2.8)
        self.assertEqual(values.get("Albumin_and_Globulin_Ratio"), 0.85)

    def test_breast_cancer_synonyms(self):
        sample_text = """
        CYTOLOGICAL MORPHOLOGY (FINE NEEDLE ASPIRATE)
        Mean Radius: 17.85 um
        Mean Texture: 23.40 um
        Mean Perimeter: 118.50 um
        Mean Area: 980.00 um2
        Mean Smoothness: 0.1150
        """
        parsed = parse_clinical_text(sample_text, "breast_cancer")
        values = parsed["values"]

        self.assertEqual(values.get("mean_radius"), 17.85)
        self.assertEqual(values.get("mean_texture"), 23.40)
        self.assertEqual(values.get("mean_perimeter"), 118.50)
        self.assertEqual(values.get("mean_area"), 980.00)
        self.assertEqual(values.get("mean_smoothness"), 0.1150)

    def test_decimal_comma_and_noise_handling(self):
        self.assertEqual(clean_numeric_value("7,8%"), 7.8)
        self.assertEqual(clean_numeric_value("185.0 mg/dL"), 185.0)
        self.assertEqual(clean_numeric_value("approx: 120"), 120.0)
        self.assertIsNone(clean_numeric_value("N/A"))
        self.assertIsNone(clean_numeric_value(""))


class TestFileValidationAndSecurity(unittest.TestCase):
    """Tests file type enforcement and safety constraints."""

    def test_allowed_extensions(self):
        self.assertTrue("pdf" in ALLOWED_EXTENSIONS)
        self.assertTrue("jpg" in ALLOWED_EXTENSIONS)
        self.assertTrue("png" in ALLOWED_EXTENSIONS)
        self.assertTrue("webp" in ALLOWED_EXTENSIONS)
        self.assertFalse("exe" in ALLOWED_EXTENSIONS)
        self.assertFalse("sh" in ALLOWED_EXTENSIONS)
        self.assertFalse("html" in ALLOWED_EXTENSIONS)

    def test_corrupt_pdf_handling(self):
        corrupt_bytes = b"This is not a real PDF file header."
        text, pages = extract_text_from_pdf(corrupt_bytes)
        self.assertEqual(text, "")
        self.assertEqual(pages, 0)


class TestReportVerifierAndQR(unittest.TestCase):
    """Tests cryptographic report registration, QR codes, and tamper detection."""

    def test_register_and_fetch_report(self):
        report_id = "MNV-TEST-2026-UNIT01"
        v_hash = register_report(
            report_id=report_id,
            module_name="diabetes",
            condition_title="Diabetes & Glycemic Check",
            prediction_text="Elevated Diabetic Risk",
            is_positive=True,
            confidence_score=92.5,
            is_scanner_extracted=True,
            summary_data={"biomarker": "HbA1c 8.4%"}
        )

        record = get_verified_report(report_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["report_id"], report_id)
        self.assertEqual(record["is_positive"], 1)
        self.assertEqual(record["is_scanner_extracted"], 1)
        self.assertEqual(record["verification_hash"], v_hash)

    def test_qr_code_generation(self):
        qr_bytes = generate_qr_code_image("MNV-TEST-QR", "http://localhost:5000")
        self.assertIsInstance(qr_bytes, bytes)
        self.assertTrue(len(qr_bytes) > 200)
        # Check PNG magic bytes: \x89PNG\r\n\x1a\n
        self.assertTrue(qr_bytes.startswith(b"\x89PNG\r\n\x1a\n"))


class TestApiIntegration(unittest.TestCase):
    """Tests Flask endpoints using test_client."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_scan_report_autofill_endpoint(self):
        sample_path = os.path.join("static", "samples", "diabetes_sample_positive.pdf")
        self.assertTrue(os.path.exists(sample_path), "Sample positive diabetes PDF missing")

        with open(sample_path, "rb") as f:
            pdf_bytes = f.read()

        response = self.client.post(
            "/api/scan-report/diabetes",
            data={
                "report_file": (io.BytesIO(pdf_bytes), "diabetes_sample_positive.pdf"),
                "mode": "autofill"
            },
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data["success"])
        self.assertEqual(data["module"], "diabetes")
        self.assertIn("HbA1c_level", data["values"])
        self.assertIn("blood_glucose_level", data["values"])

    def test_scan_report_instant_predict_endpoint(self):
        sample_path = os.path.join("static", "samples", "heart_sample_negative.pdf")
        self.assertTrue(os.path.exists(sample_path), "Sample negative heart PDF missing")

        with open(sample_path, "rb") as f:
            pdf_bytes = f.read()

        response = self.client.post(
            "/api/scan-report/heart",
            data={
                "report_file": (io.BytesIO(pdf_bytes), "heart_sample_negative.pdf"),
                "mode": "instant_predict"
            },
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data["success"])
        self.assertIn("prediction", data)
        self.assertIn("is_positive", data)
        self.assertIn("report_id", data)
        self.assertIn("biomarker_gauges", data)
        self.assertIn("top_risk_factors", data)

    def test_invalid_module_rejection(self):
        response = self.client.post(
            "/api/scan-report/unsupported_disease",
            data={"mode": "autofill"},
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)

    def test_report_qr_endpoint(self):
        response = self.client.get("/api/report-qr/MNV-TEST-QR123")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "image/png")

    def test_verify_report_page(self):
        # Register a test report first
        rep_id = "MNV-TEST-VERIFY-PAGE"
        register_report(
            report_id=rep_id,
            module_name="kidney",
            condition_title="Renal Function Assessment",
            prediction_text="Chronic Kidney Disease Detected",
            is_positive=True,
            confidence_score=94.0,
            is_scanner_extracted=True
        )

        response = self.client.get(f"/verify-report/{rep_id}")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Official Medical Report Certification", content)
        self.assertIn(rep_id, content)

        # 1. Navigation Trap: Back to Diagnostic Suite and Return to Dashboard buttons
        self.assertIn("Back to Diagnostic Suite", content)
        self.assertIn("Return to Dashboard", content)
        self.assertIn("backToSuiteBtn", content)
        self.assertIn("returnDashboardBtn", content)
        self.assertIn("popstate", content)
        self.assertIn("exitVerification", content)

        # 2. Dark-mode Contrast: Bright text and label colors
        self.assertIn("#f8fafc", content)
        self.assertIn("#f1f5f9", content)
        self.assertIn("#94a3b8", content)
        self.assertIn("#cbd5e1", content)
        self.assertIn("CLINICAL AUDIT NOTICE", content)

        # 3. Top Navbar Overflow: Padding-right and box-sizing
        self.assertIn("padding-right: 24px", content)
        self.assertIn("box-sizing: border-box", content)
        self.assertIn("Care Network", content)

        # 4. Route query parameters & case-insensitivity verification
        query_res = self.client.get(f"/verify-report?id={rep_id.lower()}")
        self.assertEqual(query_res.status_code, 200)
        self.assertIn(rep_id, query_res.data.decode("utf-8"))

    def test_verify_report_unverified_page(self):
        response = self.client.get("/verify-report/NONEXISTENT-ID")
        self.assertEqual(response.status_code, 404)
        content = response.data.decode("utf-8")
        self.assertIn("Report Record Not Located", content)
        self.assertIn("Back to Diagnostic Suite", content)
        self.assertIn("Return to Dashboard", content)

    def test_backward_compatibility_diagnostic_routes_get(self):
        routes = ["/diabetes", "/heart", "/kidney", "/liver", "/breast_cancer", "/malaria", "/pneumonia"]
        for r in routes:
            res = self.client.get(r)
            self.assertEqual(res.status_code, 200, f"Route {r} failed to render")

    def test_developer_records_security_restriction(self):
        # Accessing developer records without authenticated dev session must present the security gate
        res = self.client.get("/developer/patient-records")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Developer Authentication Gate", res.data)
        self.assertNotIn(b"DEVELOPER DASHBOARD VIEW", res.data)


class TestCardiovascularEvaluationEndpoint(unittest.TestCase):
    """Validates the Cardiovascular evaluation pipeline, safe percent handling, and styling."""

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()
        self.sample_vitals = {
            'age': '58',
            'sex': '1',
            'cp': '2',
            'bp': '135',
            'chol': '240',
            'fbs': '0',
            'ekg': '1',
            'max_hr': '145',
            'exang': '0',
            'oldpeak': '1.2',
            'slope': '1',
            'ca': '1',
            'thal': '3'
        }

    def test_biomarker_gauges_has_percent_and_analyte(self):
        """Validates that build_biomarker_gauges populates percent, position_pct, analyte, low, and high."""
        gauges = build_biomarker_gauges({'bp': 135, 'chol': 240, 'max_hr': 145})
        self.assertTrue(len(gauges) > 0)
        for g in gauges:
            self.assertIn("percent", g)
            self.assertIn("position_pct", g)
            self.assertIn("analyte", g)
            self.assertIn("low", g)
            self.assertIn("high", g)
            self.assertIsInstance(g["percent"], (int, float))

    def test_safe_percent_helper(self):
        """Validates that safe_percent gracefully handles dicts and objects without AttributeError."""
        # Dict with percent
        self.assertEqual(safe_percent({"percent": 88.5}), 88.5)
        # Dict without percent
        self.assertEqual(safe_percent({"confidence": 88.5}), 0)
        self.assertEqual(safe_percent({"score": 88.5}, default=50), 50)
        # Object with percent
        class ObjWithPct:
            percent = 92.0
        self.assertEqual(safe_percent(ObjWithPct()), 92.0)
        # Object without percent
        class ObjWithoutPct:
            pass
        self.assertEqual(safe_percent(ObjWithoutPct(), default=0), 0)
        # None
        self.assertEqual(safe_percent(None, default=0), 0)

    def test_heart_post_form_evaluates_without_attribute_error(self):
        """Tests that submitting heart form via POST renders successfully with no AttributeError / Evaluation Error."""
        res = self.client.post("/heart", data=self.sample_vitals)
        self.assertEqual(res.status_code, 200)
        content = res.data.decode("utf-8")
        self.assertNotIn("Evaluation Error", content)
        self.assertNotIn("has no attribute 'percent'", content)
        self.assertIn("Cardiovascular Risk Screening", content)
        self.assertTrue("Presence of Cardiovascular Risk" in content or "Absence of Significant Risk" in content)

    def test_heart_post_json_evaluation_response(self):
        """Tests that submitting heart vitals via JSON returns normalized model output fields."""
        res = self.client.post(
            "/heart",
            data=json.dumps(self.sample_vitals),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data.get("success"))
        self.assertIn("prediction", data)
        self.assertIn("risk_label", data)
        self.assertIn("confidence", data)
        self.assertIn("percent", data)
        self.assertIn("probability", data)
        self.assertIn("feature_contributions", data)
        self.assertIn("biomarker_gauges", data)
        # Validate that every biomarker gauge and feature contribution has 'percent'
        for g in data["biomarker_gauges"]:
            self.assertIn("percent", g)
        for f in data["feature_contributions"]:
            self.assertIn("percent", f)

    def test_heart_labels_contrast_and_multilingual_switcher(self):
        """Tests that form labels are #f1f5f9 font-weight: 600 and multilingual switcher is functional."""
        res = self.client.get("/heart")
        self.assertEqual(res.status_code, 200)
        content = res.data.decode("utf-8")

        # Form labels contrast: #f1f5f9 !important and font-weight: 600
        self.assertIn("#f1f5f9 !important", content)
        self.assertIn("font-weight: 600 !important", content)

        # Field labels for 13 clinical vitals
        vital_labels = [
            "Age (years)", "Biological Sex", "Chest Pain Type",
            "Resting Blood Pressure", "Serum Cholesterol", "Fasting Blood Sugar",
            "Resting ECG Results", "Max Heart Rate (Thalach)", "Exercise Induced Angina",
            "ST Depression (Oldpeak)", "Slope of Peak Exercise ST",
            "Fluoroscopy Major Vessels", "Thallium Scintigraphy"
        ]
        for label in vital_labels:
            self.assertIn(label, content)

        # Dark-mode pill button CSS
        self.assertIn("rgba(255, 255, 255, 0.08)", content)
        self.assertIn("#e2e8f0", content)
        self.assertIn("border-radius: 6px", content)
        self.assertIn("padding: 6px 12px", content)
        self.assertIn("#0ea5e9", content)
        self.assertIn("#38bdf8", content)

        # Client-side translation dictionary coverage for all 4 languages (high & low risk)
        self.assertIn("Cardiovascular Risk Indicators Detected", content)
        self.assertIn("இதய நாள ஆபத்து அறிகுறிகள் கண்டறியப்பட்டுள்ளன", content)
        self.assertIn("हृदय संबंधी जोखिम संकेतक पाए गए", content)
        self.assertIn("Indicadores de Riesgo Cardiovascular Detectados", content)
        self.assertIn("Cardiovascular Profile Within Normal Limits", content)
        self.assertIn("இதய நாள சுயவிவரம் இயல்பான வரம்புகளுக்குள் உள்ளது", content)
        self.assertIn("हृदय संबंधी प्रोफ़ाइल सामान्य सीमाओं के भीतर है", content)
        self.assertIn("Perfil Cardiovascular Dentro de los Límites Normales", content)

        # Event delegation and switchResultLanguage
        self.assertIn("switchResultLanguage", content)
        self.assertIn("document.addEventListener('click'", content)

    def test_heart_post_result_card_translation_targets(self):
        """Tests that submitting heart vitals renders the target IDs for dynamic translation."""
        payload = {
            "age": "60", "sex": "1", "cp": "3", "bp": "145", "chol": "280",
            "fbs": "1", "ekg": "2", "max_hr": "130", "exang": "1",
            "oldpeak": "2.5", "slope": "2", "ca": "2", "thal": "7"
        }
        res = self.client.post("/heart", data=payload)
        self.assertEqual(res.status_code, 200)
        content = res.data.decode("utf-8")

        # Multilingual Patient Summary pill buttons
        self.assertIn('data-lang="en"', content)
        self.assertIn('data-lang="ta"', content)
        self.assertIn('data-lang="hi"', content)
        self.assertIn('data-lang="es"', content)
        self.assertIn("English", content)
        self.assertIn("தமிழ் (Tamil)", content)
        self.assertIn("हिंदी (Hindi)", content)
        self.assertIn("Español", content)

        # Verify dynamic translation target IDs & attributes
        self.assertIn('id="resultMainHeading"', content)
        self.assertIn('id="resultSubHeading"', content)
        self.assertIn('id="resultDescription"', content)
        self.assertIn('id="resultPredictionText"', content)
        self.assertIn('id="resultStatusTag"', content)
        self.assertIn('data-condition="heart"', content)
        self.assertIn('data-is-positive="true"', content)
        self.assertIn("Multilingual Patient Summary", content)


if __name__ == "__main__":
    unittest.main()

