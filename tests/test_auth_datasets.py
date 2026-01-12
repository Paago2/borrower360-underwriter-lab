import os
from fastapi.testclient import TestClient

def test_datasets_requires_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "dev-preview-key")
    from app.main import app
    client = TestClient(app)

    r = client.get("/datasets")
    assert r.status_code == 401

    r = client.get("/datasets", headers={"X-API-Key": "dev-preview-key"})
    assert r.status_code == 200
