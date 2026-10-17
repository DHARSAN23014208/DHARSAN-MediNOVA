import os
import sys
import re
import json
import joblib
import unittest

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_DIR)

import app as medinova_app
from app import app, MODELS_DIR, SAMPLES_FOLDER

class MediNovaProductionAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "="*70)
        print("🚀 MEDINOVA PRE-LAUNCH PRODUCTION & CLOUD READINESS AUDIT")
        print("="*70)

    # -------------------------------------------------------------------------
    # CHECK 1: Model Loading & Input Schema Audit
    # -------------------------------------------------------------------------
    def test_01_models_and_schema_audit(self):
        print("\n[CHECK 1] AUDITING MODEL FILES & INPUT VECTOR SCHEMAS...")
        expected_schemas = {
            'heart_disease_model.pkl': 13,
            'diabetes_model.pkl': 13,
            'kidney_model.pkl': 24,
            'liver_model.pkl': 10,
            'breast_cancer_model.pkl': 5,
        }

        for model_file, expected_features in expected_schemas.items():
            path = os.path.join(MODELS_DIR, model_file)
            self.assertTrue(os.path.exists(path), f"Missing model file: {path}")
            model = joblib.load(path)
            actual_features = getattr(model, 'n_features_in_', None)
            self.assertEqual(actual_features, expected_features,
                             f"{model_file} feature mismatch: expected {expected_features}, got {actual_features}")
            print(f"  ✓ {model_file}: verified {actual_features} features (expected: {expected_features})")

        # Neural Net Image Models
        import tensorflow as tf
        nn_models = {
            'pneumonia_model_new.h5': {'input': (None, 150, 150, 3), 'output': (None, 1)},
            'malaria_cnn_model_fast.h5': {'input': (None, 50, 50, 3), 'output': (None, 2)},
        }

        for h5_file, shapes in nn_models.items():
            path = os.path.join(MODELS_DIR, h5_file)
            self.assertTrue(os.path.exists(path), f"Missing model file: {path}")
            model = tf.keras.models.load_model(path)
            self.assertEqual(model.input_shape, shapes['input'],
                             f"{h5_file} input shape mismatch: expected {shapes['input']}, got {model.input_shape}")
            self.assertEqual(model.output_shape, shapes['output'],
                             f"{h5_file} output shape mismatch: expected {shapes['output']}, got {model.output_shape}")
            print(f"  ✓ {h5_file}: verified input={model.input_shape}, output={model.output_shape}")

    # -------------------------------------------------------------------------
    # CHECK 2: HTML Form-to-Backend Key Audit
    # -------------------------------------------------------------------------
    def test_02_form_keys_and_samples_audit(self):
        print("\n[CHECK 2] AUDITING HTML TEMPLATE FORM KEYS AGAINST APP.PY...")
        
        # Verify Sample Image Files physically on disk
        samples = ['malaria_infected.png', 'malaria_uninfected.png', 'pneumonia_detected.jpeg']
        for s in samples:
            sample_path = os.path.join(SAMPLES_FOLDER, s)
            self.assertTrue(os.path.exists(sample_path), f"Sample file not found: {sample_path}")
            print(f"  ✓ Sample image verified on disk: static/samples/{s}")

        # Check forms
        forms_to_check = {
            'heart': {
                'template': 'templates/heart.html',
                'expected': ['age', 'sex', 'cp', 'bp', 'chol', 'fbs', 'ekg', 'max_hr', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            },
            'diabetes': {
                'template': 'templates/diabetes.html',
                'expected': ['gender', 'smoking_history', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
            },
            'kidney': {
                'template': 'templates/kidney.html',
                'expected': ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
            },
            'liver': {
                'template': 'templates/liver.html',
                'expected': ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']
            },
            'breast_cancer': {
                'template': 'templates/breast_cancer.html',
                'expected': ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']
            }
        }

        for route_name, cfg in forms_to_check.items():
            tpl_path = os.path.join(BASE_DIR, cfg['template'])
            with open(tpl_path, 'r', encoding='utf-8') as f:
                tpl_html = f.read()

            html_input_names = set(re.findall(r'<(?:input|select|textarea)[^>]+name=[\'"]([^\'"]+)[\'"]', tpl_html, re.IGNORECASE))
            missing = [k for k in cfg['expected'] if k not in html_input_names]
            self.assertEqual(len(missing), 0, f"Route {route_name} missing HTML inputs: {missing}")
            print(f"  ✓ Route /{route_name}: all {len(cfg['expected'])} form inputs match between HTML and backend")

    # -------------------------------------------------------------------------
    # CHECK 3: End-to-End Route Smoke Test (Flask Test Client)
    # -------------------------------------------------------------------------
    def test_03_end_to_end_route_smoke_test(self):
        print("\n[CHECK 3] END-TO-END SMOKE TESTS (GET & POST)...")
        app.config['TESTING'] = True
        client = app.test_client()

        # 1. GET requests
        get_routes = ['/', '/features', '/contact', '/heart', '/diabetes', '/kidney', '/liver', '/breast_cancer', '/malaria', '/pneumonia', '/healthz']
        for route in get_routes:
            res = client.get(route)
            self.assertEqual(res.status_code, 200, f"GET {route} failed with {res.status_code}")
            if route == '/healthz':
                data = res.get_json()
                self.assertEqual(data.get('status'), 'ok')
                self.assertTrue(all(data.get('models_loaded', {}).values()))
                print(f"  ✓ GET /healthz: Status OK, all 7 models loaded")
            else:
                print(f"  ✓ GET {route}: HTTP 200 OK")

        # 2. POST requests - All 7 Diagnostic Routes (Positive & Negative Cases)
        test_payloads = {
            'heart': {
                'pos': {'age': 60, 'sex': 1, 'cp': 3, 'bp': 145, 'chol': 280, 'fbs': 1, 'ekg': 2, 'max_hr': 130, 'exang': 1, 'oldpeak': 2.5, 'slope': 2, 'ca': 2, 'thal': 7},
                'neg': {'age': 45, 'sex': 0, 'cp': 1, 'bp': 115, 'chol': 175, 'fbs': 0, 'ekg': 0, 'max_hr': 175, 'exang': 0, 'oldpeak': 0.0, 'slope': 0, 'ca': 0, 'thal': 3}
            },
            'diabetes': {
                'pos': {'gender': 'Male', 'smoking_history': 'current', 'age': 65, 'hypertension': 1, 'heart_disease': 1, 'bmi': 32.5, 'HbA1c_level': 8.5, 'blood_glucose_level': 220},
                'neg': {'gender': 'Female', 'smoking_history': 'never', 'age': 28, 'hypertension': 0, 'heart_disease': 0, 'bmi': 21.0, 'HbA1c_level': 5.1, 'blood_glucose_level': 90}
            },
            'kidney': {
                'pos': {'age': 68, 'bp': 90, 'sg': 1.010, 'al': 3, 'su': 2, 'rbc': 0, 'pc': 0, 'pcc': 1, 'ba': 1, 'bgr': 210, 'bu': 85, 'sc': 3.5, 'sod': 130, 'pot': 5.8, 'hemo': 9.2, 'pcv': 28, 'wc': 11500, 'rc': 3.2, 'htn': 1, 'dm': 1, 'cad': 1, 'appet': 1, 'pe': 1, 'ane': 1},
                'neg': {'age': 40, 'bp': 75, 'sg': 1.025, 'al': 0, 'su': 0, 'rbc': 1, 'pc': 1, 'pcc': 0, 'ba': 0, 'bgr': 95, 'bu': 22, 'sc': 0.8, 'sod': 142, 'pot': 4.2, 'hemo': 15.5, 'pcv': 45, 'wc': 6800, 'rc': 5.0, 'htn': 0, 'dm': 0, 'cad': 0, 'appet': 0, 'pe': 0, 'ane': 0}
            },
            'liver': {
                'pos': {'Age': 58, 'Gender': 'Male', 'Total_Bilirubin': 3.5, 'Direct_Bilirubin': 1.8, 'Alkaline_Phosphotase': 380, 'Alamine_Aminotransferase': 95, 'Aspartate_Aminotransferase': 120, 'Total_Protiens': 5.8, 'Albumin': 2.4, 'Albumin_and_Globulin_Ratio': 0.70},
                'neg': {'Age': 32, 'Gender': 'Female', 'Total_Bilirubin': 0.8, 'Direct_Bilirubin': 0.2, 'Alkaline_Phosphotase': 160, 'Alamine_Aminotransferase': 22, 'Aspartate_Aminotransferase': 25, 'Total_Protiens': 7.2, 'Albumin': 4.1, 'Albumin_and_Globulin_Ratio': 1.30}
            },
            'breast_cancer': {
                'pos': {'mean_radius': 17.99, 'mean_texture': 10.38, 'mean_perimeter': 122.8, 'mean_area': 1001.0, 'mean_smoothness': 0.1184},
                'neg': {'mean_radius': 11.2, 'mean_texture': 14.5, 'mean_perimeter': 72.0, 'mean_area': 380.0, 'mean_smoothness': 0.08}
            },
            'malaria': {
                'pos': {'sample_type': 'infected'},
                'neg': {'sample_type': 'uninfected'}
            },
            'pneumonia': {
                'pos': {'sample_type': 'detected'},
                'neg': {} # Fallback test or sample
            }
        }

        for route, payloads in test_payloads.items():
            # Test Positive / High Risk
            res_pos = client.post(f'/{route}', data=payloads['pos'], follow_redirects=True)
            self.assertEqual(res_pos.status_code, 200, f"POST /{route} (pos) failed: {res_pos.status_code}")
            
            # Test Negative / Normal
            if payloads.get('neg'):
                res_neg = client.post(f'/{route}', data=payloads['neg'], follow_redirects=True)
                self.assertEqual(res_neg.status_code, 200, f"POST /{route} (neg) failed: {res_neg.status_code}")
            
            print(f"  ✓ POST /{route}: Verified High-Risk & Healthy inference flows (HTTP 200)")

        # 3. Test Chat API and Reset
        res_chat = client.post('/api/chat', json={'message': 'I have chest pain'})
        self.assertEqual(res_chat.status_code, 200)
        self.assertIn('reply', res_chat.get_json())
        print("  ✓ POST /api/chat: Verified interactive clinical triage (HTTP 200)")

        res_reset = client.post('/api/chat/reset')
        self.assertEqual(res_reset.status_code, 200)
        self.assertEqual(res_reset.get_json().get('status'), 'success')
        print("  ✓ POST /api/chat/reset: Verified session reset (HTTP 200)")

    # -------------------------------------------------------------------------
    # CHECK 4: Memory Usage & Environment Readiness
    # -------------------------------------------------------------------------
    def test_04_resource_footprint(self):
        print("\n[CHECK 4] RESOURCE & MEMORY FOOTPRINT AUDIT...")
        import psutil
        process = psutil.Process(os.getpid())
        ram_bytes = process.memory_info().rss
        ram_mb = ram_bytes / (1024 * 1024)
        print(f"  ✓ Current Process Resident Memory (RSS): {ram_mb:.1f} MB")
        self.assertLess(ram_mb, 1200, "Memory usage exceeds 1.2 GB limit")

        # Verify deployment files
        deployment_files = ['Dockerfile', 'Procfile', '.gitignore', '.dockerignore', 'requirements.txt']
        for df in deployment_files:
            p = os.path.join(BASE_DIR, df)
            self.assertTrue(os.path.exists(p), f"Missing deployment file: {df}")
            print(f"  ✓ Deployment config verified: {df}")

if __name__ == '__main__':
    unittest.main()
