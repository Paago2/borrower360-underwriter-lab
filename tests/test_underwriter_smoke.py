from fastapi.testclient import TestClient

def test_underwriter_requires_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "dev-preview-key")
    from app.main import app
    client = TestClient(app)

    # no key -> 401
    r = client.post("/underwriter/run", json={"person_name": "John Doe"})
    assert r.status_code == 401

    # with key -> 200
    r = client.post(
        "/underwriter/run",
        json={"person_name": "John Doe", "sanctions_flag": False, "risk_tier": "low"},
        headers={"X-API-Key": "dev-preview-key"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "decision_result" in body
    assert body["decision_result"]["decision"] in ("false_positive", "approved_match", "needs_more_info", "open")
