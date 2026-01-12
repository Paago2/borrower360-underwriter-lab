from fastapi.testclient import TestClient

def test_health_ok():
    from app.main import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert "x-request-id" in r.headers
    assert r.json().get("status") in ("ok", "ready")
