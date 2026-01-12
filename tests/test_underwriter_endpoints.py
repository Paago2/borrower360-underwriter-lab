from fastapi.testclient import TestClient
import os

def test_underwriter_requires_api_key():
    os.environ["API_KEY"] = "dev-preview-key"
    from app.main import app
    client = TestClient(app)

    r = client.get("/underwriter/cases/underwriter_case_packets_person/0")
    assert r.status_code == 401

    r2 = client.get(
        "/underwriter/cases/underwriter_case_packets_person/0",
        headers={"X-API-Key": "dev-preview-key"},
    )
    assert r2.status_code == 200


def test_underwriter_contract_has_trace_steps():
    os.environ["API_KEY"] = "dev-preview-key"
    from app.main import app
    client = TestClient(app)

    r = client.get(
        "/underwriter/cases/underwriter_review_queue_person_top/0",
        headers={"X-API-Key": "dev-preview-key"},
    )
    assert r.status_code == 200
    data = r.json()

    assert "decision_result" in data
    dr = data["decision_result"]
    assert dr["decision"] in ("false_positive", "needs_more_info", "approved_match")
    assert isinstance(dr["decision_notes"], str)

    trace = dr["rationale"]["trace"]
    assert trace["steps"] == [
        "LoadCase",
        "CheckCorroboration",
        "AssessEvidence",
        "RetrieveMoreEvidence",
        "PolicyDecision",
        "Finalize",
    ]
