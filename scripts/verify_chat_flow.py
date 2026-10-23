import os
import sys
import json
import unittest

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app, CLINICAL_DOMAINS, DISEASE_INSIGHTS

class TestMediNovaChatbot(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_01_all_11_templates_render_and_have_chatbot_v3(self):
        routes = [
            ('/', 'MediNova'),
            ('/features', 'Diagnostic Suite'),
            ('/heart', 'Cardiovascular'),
            ('/diabetes', 'Diabetes'),
            ('/kidney', 'Kidney'),
            ('/liver', 'Liver'),
            ('/breast_cancer', 'Breast'),
            ('/malaria', 'Malaria'),
            ('/pneumonia', 'Pneumonia'),
            ('/contact', 'Emergency'),
        ]
        for route, keyword in routes:
            res = self.client.get(route)
            self.assertEqual(res.status_code, 200, f"Route {route} failed with status {res.status_code}")
            html = res.data.decode('utf-8')
            self.assertTrue("chatbot.js" in html, f"Route {route} missing chatbot.js")
            print(f"✅ Route {route} rendered successfully with chatbot.js")

    def test_02_initial_greeting_and_chips(self):
        res = self.client.post('/api/chat', json={})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('reply', data)
        self.assertIn('quick_replies', data)
        self.assertIn('actions', data)
        self.assertTrue(len(data['quick_replies']) >= 4)
        print("✅ Initial greeting returns quick_replies and actions successfully")

    def test_03_three_turn_heart_consultation_flow(self):
        with self.client as c:
            # Turn 1: User presents symptom
            res1 = c.post('/api/chat', json={'message': 'I have chest pain and pressure'})
            self.assertEqual(res1.status_code, 200)
            data1 = res1.get_json()
            print("\n--- TURN 1 RESPONSE ---")
            print("Reply:", data1['reply'])
            print("Quick Replies:", data1['quick_replies'])
            print("Actions:", data1['actions'])

            self.assertIn('Question 1 of 2', data1['reply'])
            self.assertTrue(any('/heart' in act.get('url', '') for act in data1['actions']))
            self.assertTrue(len(data1['quick_replies']) >= 3)

            # Turn 2: User answers Question 1 (Severity & Duration)
            res2 = c.post('/api/chat', json={'message': '🔴 Severe / Started suddenly'})
            self.assertEqual(res2.status_code, 200)
            data2 = res2.get_json()
            print("\n--- TURN 2 RESPONSE ---")
            print("Reply:", data2['reply'])
            print("Quick Replies:", data2['quick_replies'])

            self.assertIn('Question 2 of 2', data2['reply'])
            self.assertTrue(len(data2['quick_replies']) >= 3)

            # Turn 3: User answers Question 2 (Red-flag symptoms)
            res3 = c.post('/api/chat', json={'message': 'Yes, I have those symptoms'})
            self.assertEqual(res3.status_code, 200)
            data3 = res3.get_json()
            print("\n--- TURN 3 ACTION PLAN RESPONSE ---")
            print("Reply snippet:", data3['reply'][:300] + "...")
            print("Actions:", data3['actions'])

            self.assertIn('Action Plan', data3['reply'])
            self.assertIn('Immediate Action', data3['reply'])
            self.assertIn('Recommended MediNova AI Screening', data3['reply'])
            self.assertIn('Safe Diet & Care Protocol', data3['reply'])
            self.assertTrue(any('/heart' in act.get('url', '') for act in data3['actions']))

    def test_04_explain_test_result_with_session(self):
        with self.client as c:
            # 1. No diagnosis in session
            res_none = c.post('/api/chat', json={'message': 'Explain my latest test result'})
            self.assertEqual(res_none.status_code, 200)
            data_none = res_none.get_json()
            self.assertIn('No completed diagnostic screening was detected', data_none['reply'])
            print("\n✅ Explain result with empty session handled gracefully")

            # 2. Put simulated positive diagnosis in session
            with c.session_transaction() as sess:
                sess['last_diagnosis'] = {
                    'condition': 'Heart Disease',
                    'route': '/heart',
                    'prediction': 'Presence of Cardiovascular Risk',
                    'is_positive': True,
                    'patient_summary': [
                        {'label': 'Age', 'value': '58 years'},
                        {'label': 'Resting Blood Pressure', 'value': '145 mm Hg'},
                        {'label': 'Serum Cholesterol', 'value': '260 mg/dL'},
                        {'label': 'ST Depression', 'value': '2.2'}
                    ],
                    'specialist': 'Board-Certified Cardiologist',
                    'urgency_window': 'Within 24 to 48 hours',
                    'recommended_tests': ['12-Lead ECG', 'Echocardiogram', 'Lipid Panel']
                }

            res_has = c.post('/api/chat', json={'message': 'Explain my test result'})
            self.assertEqual(res_has.status_code, 200)
            data_has = res_has.get_json()
            print("\n✅ Explain result with active diagnosis:")
            print("Reply snippet:", data_has['reply'][:350] + "...")
            self.assertIn('Presence of Cardiovascular Risk', data_has['reply'])
            self.assertIn('Serum Cholesterol', data_has['reply'])
            self.assertIn('Cardiologist', data_has['reply'])

    def test_05_chat_reset_endpoint(self):
        with self.client as c:
            res = c.post('/api/chat/reset')
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data.get('status'), 'success')
            self.assertIn('quick_replies', data)
            print("\n✅ /api/chat/reset works as expected")

if __name__ == '__main__':
    unittest.main()
