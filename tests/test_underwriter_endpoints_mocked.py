import os
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# --- minimal stand-in for DatasetLocation (only fields route uses) ---
@dataclass(frozen=True)
class FakeLoc:
    key: str = "fake"
    tier: str = "underwriter_queue"
    storage: str = "parquet"
    dir_ref: str = "data/05_underwriter_queue/fake"
    dir_uri: str = "/tmp/fake"
    parquet_files: List[str] = None
    index_jsonl: Optional[str] = None


@pytest.fixture
def client(monkeypatch):
    # Ensure server-side API_KEY exists for require_api_key
    monkeypatch.setenv("API_KEY", "dev-preview-key")
    from app.main import app
    return TestClient(app)


def _fake_df():
    # Include columns that your agent/graph code may touch
    return pd.DataFrame([{
        "person_name": "Test User",
        "sanctions_flag": True,
        "risk_tier": "high",
        "match_score": 99,
        "birth_date": "1990-01-01",
        "state": "VA",
        "zip": "22150",
        "case_reasons_json": "[]",
        "evidence_matches_json": "[]",
    }])


def test_underwriter_requires_api_key(client):
    # Without key => 401
    r = client.get("/underwriter/cases/underwriter_case_packets_person/0")
    assert r.status_code == 401

    # With key + mocked loader => 200
    with patch("app.api.routes_underwriter.load_dataset_df") as mock_load:
        mock_load.return_value = (_fake_df(), FakeLoc())

        r2 = client.get(
            "/underwriter/cases/underwriter_case_packets_person/0",
            headers={"X-API-Key": "dev-preview-key"},
        )
        assert r2.status_code == 200


def test_underwriter_contract_has_trace_steps(client):
    with patch("app.api.routes_underwriter.load_dataset_df") as mock_load:
        mock_load.return_value = (_fake_df(), FakeLoc())

        r = client.get(
            "/underwriter/cases/underwriter_review_queue_person_top/0",
            headers={"X-API-Key": "dev-preview-key"},
        )
        assert r.status_code == 200
        data = r.json()

        dr = data["decision_result"]
        trace = dr["rationale"]["trace"]

        assert trace["steps"] == [
            "LoadCase",
            "CheckCorroboration",
            "AssessEvidence",
            "RetrieveMoreEvidence",
            "PolicyDecision",
            "Finalize",
        ]
