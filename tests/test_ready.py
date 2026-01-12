from fastapi.testclient import TestClient

def test_ready_ok():
    from app.main import app
    client = TestClient(app)
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json().get("status") in ("ready", "ok")
