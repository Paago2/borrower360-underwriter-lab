# app/api/routes_underwriter.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.auth import require_api_key
from app.core.paths import project_root
from app.services.dataset_loader import load_dataset_df
from app.agents.underwriter_graph import UnderwriterTools, run_underwriter_agent, DecisionResult

router = APIRouter(
    prefix="/underwriter",
    tags=["Underwriter"],
    dependencies=[Depends(require_api_key)],
)


def _decision_to_json(dr: DecisionResult) -> Dict[str, Any]:
    """
    Convert DecisionResult dataclass -> JSON-safe dict.
    """
    if is_dataclass(dr):
        return asdict(dr)
    # fail-safe
    return {
        "decision": getattr(dr, "decision", None),
        "decision_notes": getattr(dr, "decision_notes", None),
        "needs_more_info_fields": getattr(dr, "needs_more_info_fields", []),
        "rationale": getattr(dr, "rationale", {}),
    }

def _row_as_dict(df, row_index: int) -> Dict[str, Any]:
    if row_index < 0 or row_index >= len(df):
        raise HTTPException(status_code=404, detail=f"row_index {row_index} out of range (0..{len(df)-1}).")
    row = df.iloc[row_index].to_dict()
    # Convert NaN -> None so JSON is clean
    return {k: (None if v != v else v) for k, v in row.items()}



@router.get("/cases/{dataset_key}/{row_index}", operation_id="run_case_by_row_index")
def run_case_by_row_index(
    dataset_key: str,
    row_index: int,
    agent_id: str = Query("underwriter_agent_v1"),
    policy_id: str = Query("policy_v1"),
) -> Dict[str, Any]:
    """
    Minimal “demo-ready” endpoint:
    - Reads ONE row from a tabular parquet dataset (by index)
    - Runs the Underwriter agent
    - Returns DecisionResult + rationale/trace
    """
    pr = project_root()

    # Load only enough rows to include row_index
    # (simple and reliable for now; later we can implement paging by shard)
    df, loc = load_dataset_df(pr, dataset_key, max_parts=1, max_rows=row_index + 1)

    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No rows found for dataset '{dataset_key}' (tier={loc.tier}).")

    if row_index < 0 or row_index >= len(df):
        raise HTTPException(status_code=404, detail=f"row_index {row_index} out of range (0..{len(df)-1}).")

    # Convert row to python types
    case_row: Dict[str, Any] = _row_as_dict(df, row_index)


    tools = UnderwriterTools(pr)
    dr = run_underwriter_agent(case_row=case_row, tools=tools, agent_id=agent_id, policy_id=policy_id)

    return {
        "dataset_key": dataset_key,
        "tier": loc.tier,
        "storage": loc.storage,
        "row_index": row_index,
        "decision_result": _decision_to_json(dr),
    }


@router.post("/run", operation_id="run_case_payload")
def run_case_payload(
    payload: Dict[str, Any],
    agent_id: str = Query("underwriter_agent_v1"),
    policy_id: str = Query("policy_v1"),
) -> Dict[str, Any]:
    """
    “Enterprise-style” endpoint:
    - Caller provides the case_row payload directly
    - We run the agent and return DecisionResult
    """
    pr = project_root()
    tools = UnderwriterTools(pr)

    if not isinstance(payload, dict) or not payload:
        raise HTTPException(status_code=400, detail="Payload must be a non-empty JSON object representing a case_row.")

    dr = run_underwriter_agent(case_row=payload, tools=tools, agent_id=agent_id, policy_id=policy_id)

    return {"decision_result": _decision_to_json(dr)}


@router.post("/run_from_case", operation_id="run_from_case")
def run_from_case(
    dataset_key: str = Query(..., description="Dataset key, e.g. underwriter_review_queue_person_top"),
    row_index: int = Query(0, ge=0),
    agent_id: str = Query("underwriter_agent_v1"),
    policy_id: str = Query("policy_v1"),
) -> Dict[str, Any]:
    # Reuse your existing GET handler (same logic, nicer UX for UI callers)
    return run_case_by_row_index(dataset_key, row_index, agent_id=agent_id, policy_id=policy_id)
