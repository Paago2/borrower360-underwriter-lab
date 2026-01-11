from fastapi.testclient import TestClient

def test_health_ok():
    from app.main import app  # change only if needed
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    # Accept either {"status":"ok"} or a richer response
    body = r.json()
    assert "status" in body
