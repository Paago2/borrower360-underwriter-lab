import os
from fastapi.testclient import TestClient

def test_preview_requires_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "dev-preview-key")
    from app.main import app
    client = TestClient(app)

    r = client.get("/preview/lending_club")
    assert r.status_code == 401

    r = client.get("/preview/lending_club", headers={"X-API-Key": "dev-preview-key"})
    assert r.status_code in (200, 404)  # 404 if dataset not present in test env
