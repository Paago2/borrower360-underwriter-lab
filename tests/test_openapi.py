from fastapi.testclient import TestClient

def test_openapi_json_serves():
    from app.main import app  # change only if needed
    client = TestClient(app)
    r = client.get("/openapi.json")
    assert r.status_code == 200
    data = r.json()
    assert data.get("openapi")
    assert data.get("paths")
