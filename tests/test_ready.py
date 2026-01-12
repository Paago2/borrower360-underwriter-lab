from fastapi.testclient import TestClient

def test_ready_contract():
    from app.main import app
    client = TestClient(app)
    r = client.get("/ready")
    assert r.status_code in (200, 503)

    body = r.json()
    assert body["status"] in ("ready", "not_ready")
    assert "checks" in body

    checks = body["checks"]
    for key in ["configs.contracts", "configs.datasets", "data.mount", "sanctions.input_dir"]:
        assert key in checks
        assert checks[key]["status"] in ("ok", "fail")
        assert "detail" in checks[key]

