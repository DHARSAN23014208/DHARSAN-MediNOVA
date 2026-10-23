"""
Test suite for MediNova Clinical Copilot and General Chatbot capability.
Tests both Medical Priority handling and Universal Non-Medical response capabilities.
"""
import sys
import os
import json

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from utils.clinical_copilot import medinova_clinical_copilot_engine

def test_engine_directly():
    print("--- [1] Testing Engine Directly ---")
    
    # 1. Emergency Query
    res_emer = medinova_clinical_copilot_engine("I have crushing chest pain radiating to my left arm and jaw")
    assert res_emer["is_emergency"] is True, "Emergency not detected for chest pain"
    assert "EMERGENCY PROTOCOL" in res_emer["reply"], "Emergency banner missing"
    print("  [PASS] Emergency detection & protocol verified.")
    
    # 2. Medical Suite Query - Diabetes
    res_diab = medinova_clinical_copilot_engine("My fasting blood glucose is 185 mg/dL and HbA1c is 8.5%")
    assert res_diab["is_medical"] is True, "Diabetes not flagged as medical"
    assert res_diab["domain"] == "Diabetes", f"Expected Diabetes domain, got {res_diab['domain']}"
    assert "/diabetes" in res_diab["reply"] or any("/diabetes" in a.get("url", "") for a in res_diab.get("suggested_actions", [])), "Diabetes suite link missing"
    print("  [PASS] Diabetes clinical inquiry & routing verified.")

    # 3. Medical Suite Query - Heart
    res_heart = medinova_clinical_copilot_engine("What does high troponin or oldpeak ST depression mean?")
    assert res_heart["is_medical"] is True, "Heart/biomarker not flagged as medical"
    assert "/heart" in res_heart["reply"] or any("/heart" in a.get("url", "") for a in res_heart.get("suggested_actions", [])), "Heart routing missing"
    print("  [PASS] Cardiac biomarker explainer verified.")

    # 4. Out-of-suite Clinical Symptom - Migraine
    res_migraine = medinova_clinical_copilot_engine("I have a throbbing one-sided headache with nausea and light sensitivity")
    assert res_migraine["is_medical"] is True, "Migraine not flagged as medical"
    assert "Migraine" in res_migraine["reply"] or "Headache" in res_migraine["reply"], "Migraine triage missing"
    print("  [PASS] Extended symptom (migraine) verified.")

    # 5. Medication KB Query - Paracetamol
    res_med = medinova_clinical_copilot_engine("What is the safe dosage and contraindication for paracetamol?")
    assert res_med["is_medical"] is True, "Medication question not flagged as medical"
    assert "Paracetamol" in res_med["reply"] or "acetaminophen" in res_med["reply"].lower(), "Paracetamol KB missing"
    print("  [PASS] Medication pharmacology safety verified.")

    # 6. Non-Medical Query - Science / Photosynthesis
    res_photo = medinova_clinical_copilot_engine("Explain the process of photosynthesis in plants")
    assert res_photo["is_medical"] is False, "Photosynthesis should not be flagged as medical"
    assert "chlorophyll" in res_photo["reply"].lower() or "light" in res_photo["reply"].lower(), "Photosynthesis answer missing expected content"
    print("  [PASS] Non-medical general query (Photosynthesis) handled conversationally.")

    # 7. Non-Medical Query - Literature / General
    res_lit = medinova_clinical_copilot_engine("Who wrote Romeo and Juliet?")
    assert res_lit["is_medical"] is False, "Literature question flagged as medical"
    assert "shakespeare" in res_lit["reply"].lower(), "Shakespeare answer missing"
    print("  [PASS] General knowledge query (Shakespeare) handled accurately.")

    # 8. Conversational greeting
    res_greet = medinova_clinical_copilot_engine("Hello! What can you help me with?")
    assert "MediNova" in res_greet["reply"], "Greeting missing MediNova intro"
    print("  [PASS] Conversational greeting handled properly.")


def test_api_chat_endpoint():
    print("\n--- [2] Testing /api/chat HTTP Endpoint via Flask Client ---")
    client = app.test_client()
    
    # 1. Reset chat
    res = client.post("/api/chat/reset")
    assert res.status_code == 200, f"Reset failed: {res.status_code}"
    
    # 2. General Query
    res = client.post("/api/chat", json={"message": "Can you explain how gravity works?"})
    assert res.status_code == 200, f"Chat failed: {res.status_code}"
    data = json.loads(res.data)
    assert data["success"] is True, "Response success != True"
    assert "reply" in data and len(data["reply"]) > 20, "Reply empty or too short"
    assert data["is_medical"] is False, "Gravity flagged as medical"
    print(f"  [PASS] General query response verified. is_medical={data['is_medical']}")

    # 3. Medical Clinical Query
    res = client.post("/api/chat", json={"message": "I am experiencing jaundice, dark urine, and abdominal swelling."})
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["success"] is True
    assert data["is_medical"] is True, "Liver query not flagged as medical"
    assert data["domain"] == "Liver", f"Expected domain Liver, got {data.get('domain')}"
    assert "/liver" in data["reply"] or any("/liver" in a.get("url", "") for a in data.get("actions", [])), "Liver suite routing missing"
    print(f"  [PASS] Clinical query response verified. is_medical={data['is_medical']}, domain={data.get('domain')}")

    # 4. Emergency Query
    res = client.post("/api/chat", json={"message": "Help! Sudden facial drooping and arm weakness cannot speak!"})
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["is_emergency"] is True, "Stroke not flagged as emergency"
    assert "EMERGENCY PROTOCOL" in data["reply"]
    print(f"  [PASS] Emergency triage endpoint verified. is_emergency={data['is_emergency']}")

    # 5. Multi-turn Session History
    res = client.post("/api/chat", json={"message": "What specialist should I see for this?"})
    assert res.status_code == 200
    data = json.loads(res.data)
    assert data["success"] is True
    print("  [PASS] Multi-turn follow-up handled successfully.")

if __name__ == "__main__":
    try:
        test_engine_directly()
        test_api_chat_endpoint()
        print("\n=======================================================")
        print("ALL CLINICAL COPILOT & CHATBOT ENHANCEMENT TESTS PASSED!")
        print("=======================================================")
    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
