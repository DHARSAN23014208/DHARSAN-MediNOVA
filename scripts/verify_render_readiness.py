import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_render_readiness():
    print("Testing Render Readiness & Lazy-Load Mechanics...")
    
    # 1. Import app
    from app import app, pneumonia_model, malaria_model, get_pneumonia_model, get_malaria_model
    
    # Check env vars
    print("TF_CPP_MIN_LOG_LEVEL:", os.environ.get("TF_CPP_MIN_LOG_LEVEL"))
    print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
    print("TF_NUM_INTRAOP_THREADS:", os.environ.get("TF_NUM_INTRAOP_THREADS"))
    print("TF_NUM_INTEROP_THREADS:", os.environ.get("TF_NUM_INTEROP_THREADS"))
    
    assert os.environ.get("TF_CPP_MIN_LOG_LEVEL") == "2"
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    assert os.environ.get("TF_NUM_INTRAOP_THREADS") == "1"
    assert os.environ.get("TF_NUM_INTEROP_THREADS") == "1"
    print("[PASS] Environment variables verified.")

    # 2. Verify models are None at startup (Lazy-Loading)
    import app as app_module
    assert app_module.pneumonia_model is None, "Pneumonia model should NOT be loaded on boot!"
    assert app_module.malaria_model is None, "Malaria model should NOT be loaded on boot!"
    print("[PASS] Lazy loading verified: Keras models are NOT loaded on startup.")

    # 3. Test Flask Test Client
    client = app.test_client()

    # Test /healthz
    resp = client.get('/healthz')
    print("GET /healthz response:", resp.status_code, resp.get_json())
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}
    print("[PASS] /healthz endpoint returned 200 OK with {'status': 'ok'}.")

    # Test /malaria sample trigger (lazy-loads malaria model)
    print("Testing /malaria POST with sample_type='infected'...")
    resp_malaria = client.post('/malaria', data={'sample_type': 'infected'})
    assert resp_malaria.status_code == 200
    assert app_module.malaria_model is not None, "Malaria model should be loaded after POST /malaria"
    print("[PASS] Malaria lazy-load and inference succeeded.")

    # Test /pneumonia sample trigger (lazy-loads pneumonia model)
    print("Testing /pneumonia POST with sample_type='detected'...")
    resp_pneumonia = client.post('/pneumonia', data={'sample_type': 'detected'})
    assert resp_pneumonia.status_code == 200
    assert app_module.pneumonia_model is not None, "Pneumonia model should be loaded after POST /pneumonia"
    print("[PASS] Pneumonia lazy-load and inference succeeded.")

    print("\nALL RENDER DEPLOYMENT & 512MB RAM OPTIMIZATION TESTS PASSED!")

if __name__ == "__main__":
    test_render_readiness()
