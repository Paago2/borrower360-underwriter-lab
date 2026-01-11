# app/agents/underwriter_graph.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, cast

import pandas as pd
from langgraph.graph import StateGraph, END


# ----------------------------
# Decision output
# ----------------------------

@dataclass
class DecisionResult:
    decision: str
    decision_notes: str
    needs_more_info_fields: List[str]
    rationale: Dict[str, Any]


# ----------------------------
# Tools interface
# ----------------------------

class UnderwriterTools:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()

    def get_watchlist_details(self, watchlist_id: str) -> Dict[str, Any]:
        return {}

    def get_person_application_signals(self, person_id: Any) -> Dict[str, Any]:
        return {}


# ----------------------------
# LangGraph State Schema (LangGraph 1.0.x)
# ----------------------------

class TraceState(TypedDict, total=False):
    steps: List[str]
    tools_used: List[str]
    llm_used: Optional[str]
    fallback: bool


class UnderwriterState(TypedDict, total=False):
    case_row: Dict[str, Any]
    reasons: List[str]
    evidence: List[Dict[str, Any]]
    signals: Dict[str, Any]
    tool_outputs: Dict[str, Any]
    decision_result: DecisionResult
    trace: TraceState

    tools: UnderwriterTools
    agent_id: str
    policy_id: str


# ----------------------------
# Safe parsing helpers
# ----------------------------

def _safe_json_loads(x: Any, default: Any) -> Any:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return default
    if isinstance(x, (dict, list)):
        return x
    s = str(x).strip()
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


def _as_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x)
    return "" if s.lower() in ("nan", "none") else s


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except Exception:
        return default


def _bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = _as_str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def _trace_push(state: UnderwriterState, step: str) -> TraceState:
    trace = dict(state.get("trace") or {})
    steps = list(trace.get("steps") or [])
    steps.append(step)
    trace["steps"] = steps
    trace.setdefault("tools_used", [])
    trace.setdefault("llm_used", None)
    # fallback True is reserved for a manual sequential fallback path (not used here)
    trace.setdefault("fallback", False)
    return cast(TraceState, trace)


# ----------------------------
# Extractors / signals
# ----------------------------

def _extract_reasons(case_row: Dict[str, Any]) -> List[str]:
    reasons = _safe_json_loads(case_row.get("case_reasons_json"), default=[])
    if isinstance(reasons, list):
        return [str(x) for x in reasons]
    txt = _as_str(case_row.get("qa_reasons", ""))
    return [t.strip() for t in txt.split(",") if t.strip()]


def _extract_evidence(case_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    ev = _safe_json_loads(case_row.get("evidence_matches_json"), default=[])
    if isinstance(ev, list):
        return [x for x in ev if isinstance(x, dict)]
    return []


def _infer_fuzzy(evidence: List[Dict[str, Any]], reasons: List[str]) -> bool:
    if "fuzzy_match" in set(reasons):
        return True
    for m in evidence:
        mm = _as_str(m.get("match_method", "")).lower()
        if "fuzzy" in mm:
            return True
        qa = m.get("qa_reasons") or []
        if isinstance(qa, list) and any(str(x) == "fuzzy_match" for x in qa):
            return True
    return False


def _infer_borderline(evidence: List[Dict[str, Any]], reasons: List[str]) -> bool:
    if "borderline_score" in set(reasons):
        return True
    for m in evidence:
        qa = m.get("qa_reasons") or []
        if isinstance(qa, list) and any(str(x) == "borderline_score" for x in qa):
            return True
    return False


def _summarize_evidence(
    evidence: List[Dict[str, Any]],
    fallback_score: float,
    reasons: List[str],
) -> Dict[str, Any]:
    scores = [_as_float(m.get("match_score", 0.0), 0.0) for m in evidence if isinstance(m, dict)]
    best_score = max(scores) if scores else float(fallback_score)
    any_fuzzy = _infer_fuzzy(evidence, reasons)
    any_borderline = _infer_borderline(evidence, reasons)
    return {
        "best_score": float(best_score),
        "any_fuzzy": bool(any_fuzzy),
        "any_borderline": bool(any_borderline),
        "evidence_count": int(len(evidence)),
    }


def _corroboration_strength(case_row: Dict[str, Any]) -> int:
    # +2 birth_date, +1 state, +1 zip, +1 any one of (ssn_last4/email_hash/phone_hash)
    score = 0
    if _as_str(case_row.get("birth_date", "")).strip():
        score += 2
    if _as_str(case_row.get("state", "")).strip():
        score += 1
    if _as_str(case_row.get("zip", "")).strip():
        score += 1
    if (
        _as_str(case_row.get("ssn_last4", "")).strip()
        or _as_str(case_row.get("email_hash", "")).strip()
        or _as_str(case_row.get("phone_hash", "")).strip()
    ):
        score += 1
    return int(score)


# ----------------------------
# Nodes (LangGraph 1.0.x contract: return dict of updates only)
# ----------------------------

def n_load_case(state: UnderwriterState) -> Dict[str, Any]:
    trace = _trace_push(state, "LoadCase")
    case_row = state.get("case_row") or {}
    reasons = _extract_reasons(case_row)
    evidence = _extract_evidence(case_row)
    return {
        "trace": trace,
        "reasons": reasons,
        "evidence": evidence,
        "signals": dict(state.get("signals") or {}),
        "tool_outputs": dict(state.get("tool_outputs") or {}),
    }


def n_check_corroboration(state: UnderwriterState) -> Dict[str, Any]:
    trace = _trace_push(state, "CheckCorroboration")
    case_row = state.get("case_row") or {}
    strength = _corroboration_strength(case_row)
    signals = dict(state.get("signals") or {})
    signals["corroboration_strength"] = int(strength)
    return {"trace": trace, "signals": signals}


def n_assess_evidence(state: UnderwriterState) -> Dict[str, Any]:
    trace = _trace_push(state, "AssessEvidence")
    case_row = state.get("case_row") or {}
    reasons = list(state.get("reasons") or [])
    evidence = list(state.get("evidence") or [])

    summary = _summarize_evidence(
        evidence=evidence,
        fallback_score=_as_float(case_row.get("match_score"), 0.0),
        reasons=reasons,
    )
    signals = dict(state.get("signals") or {})
    signals.update(summary)
    return {"trace": trace, "signals": signals}


def n_retrieve_more(state: UnderwriterState) -> Dict[str, Any]:
    trace = _trace_push(state, "RetrieveMoreEvidence")
    tools = state.get("tools")
    if tools is None:
        return {"trace": trace}

    case_row = state.get("case_row") or {}
    evidence = list(state.get("evidence") or [])
    signals = dict(state.get("signals") or {})

    best_score = float(signals.get("best_score", 0.0))
    any_fuzzy = bool(signals.get("any_fuzzy", False))
    any_borderline = bool(signals.get("any_borderline", False))

    need_more = (any_fuzzy or any_borderline) and best_score >= 90
    if not need_more:
        return {"trace": trace}

    tool_outputs = dict(state.get("tool_outputs") or {})
    tools_used = list(trace.get("tools_used") or [])

    details: List[Dict[str, Any]] = []
    for ev in evidence[:3]:
        wl_id = _as_str(ev.get("watchlist_id"))
        if not wl_id:
            continue
        d = tools.get_watchlist_details(wl_id) or {}
        if d:
            details.append({"watchlist_id": wl_id, "details": d})

    pid = case_row.get("person_id")
    person_sig = tools.get_person_application_signals(pid) or {}

    if details and "get_watchlist_details" not in tools_used:
        tools_used.append("get_watchlist_details")
    if person_sig and "get_person_application_signals" not in tools_used:
        tools_used.append("get_person_application_signals")

    trace["tools_used"] = tools_used
    tool_outputs["watchlist_details"] = details
    tool_outputs["person_signals"] = person_sig

    return {"trace": trace, "tool_outputs": tool_outputs}


def n_policy_decision(state: UnderwriterState) -> Dict[str, Any]:
    trace = _trace_push(state, "PolicyDecision")

    case_row = state.get("case_row") or {}
    reasons = list(state.get("reasons") or [])
    reasons_set = set(str(x) for x in reasons)

    signals = dict(state.get("signals") or {})

    sanctions_flag = _bool(case_row.get("sanctions_flag", False))
    risk_tier = _as_str(case_row.get("risk_tier", ""))
    person_name = _as_str(case_row.get("person_name", ""))

    best_score = float(signals.get("best_score", 0.0))
    any_fuzzy = bool(signals.get("any_fuzzy", False))
    any_borderline = bool(signals.get("any_borderline", False))
    corroboration_strength = int(signals.get("corroboration_strength", 0))

    # ---- Ambiguity flags (audit-friendly) ----
    ambiguous_name = "common_or_ambiguous_name" in reasons_set
    missing_loc = "missing_state_or_zip" in reasons_set
    ambiguous_match = any_fuzzy or any_borderline or ambiguous_name or missing_loc

    # 1) If sanctions + strong score but ambiguous -> needs more info (do NOT auto-approve)
    if sanctions_flag and best_score >= 99 and ambiguous_match:
        dr = DecisionResult(
            decision="needs_more_info",
            decision_notes=(
                f"Potential match for {person_name} (score={best_score}, corroboration={corroboration_strength}) "
                f"but ambiguity flags present (fuzzy/borderline/common-name/missing-loc). Request corroboration."
            ),
            needs_more_info_fields=["birth_date", "state", "zip", "ssn_last4_or_contact"],
            rationale={},
        )
        return {"trace": trace, "decision_result": dr}

    # 2) Auto-approve ONLY when sanctions + very strong score + strong corroboration + NOT ambiguous
    if sanctions_flag and best_score >= 99 and corroboration_strength >= 4 and not ambiguous_match:
        dr = DecisionResult(
            decision="approved_match",
            decision_notes=(
                f"Approved match for {person_name}: sanctions_flag={sanctions_flag}, "
                f"score={best_score}, corroboration={corroboration_strength}, risk_tier={risk_tier}."
            ),
            needs_more_info_fields=[],
            rationale={},
        )
        return {"trace": trace, "decision_result": dr}

    # 3) Default false positive
    dr = DecisionResult(
        decision="false_positive",
        decision_notes=f"Insufficient evidence to confirm match for {person_name}.",
        needs_more_info_fields=[],
        rationale={},
    )
    return {"trace": trace, "decision_result": dr}


def n_finalize(state: UnderwriterState) -> Dict[str, Any]:
    trace = _trace_push(state, "Finalize")

    case_row = state.get("case_row") or {}
    reasons = list(state.get("reasons") or [])
    signals = dict(state.get("signals") or {})
    tool_outputs = dict(state.get("tool_outputs") or {})

    sanctions_flag = _bool(case_row.get("sanctions_flag", False))
    risk_tier = _as_str(case_row.get("risk_tier", ""))

    dr = state.get("decision_result")
    if not isinstance(dr, DecisionResult):
        # Fail-safe: never return "open/init" silently
        dr = DecisionResult("false_positive", "Finalize missing decision_result; defaulted.", [], {})

    rationale = {
        "producer": "agent",
        "agent_id": state.get("agent_id"),
        "policy_id": state.get("policy_id"),
        "best_score": float(signals.get("best_score", 0.0)),
        "any_fuzzy": bool(signals.get("any_fuzzy", False)),
        "any_borderline": bool(signals.get("any_borderline", False)),
        "evidence_count": int(signals.get("evidence_count", 0)),
        "sanctions_flag": sanctions_flag,
        "risk_tier": risk_tier,
        "corroboration_strength": int(signals.get("corroboration_strength", 0)),
        "missing_fields": [],
        "reasons": list(reasons),
        "tool_outputs": tool_outputs,
        "trace": dict(trace),
    }

    dr.rationale = rationale
    return {"trace": trace, "decision_result": dr}


# ----------------------------
# Build graph + runner
# ----------------------------

_GRAPH = None


def build_underwriter_graph():
    g = StateGraph(UnderwriterState)

    g.add_node("LoadCase", n_load_case)
    g.add_node("CheckCorroboration", n_check_corroboration)
    g.add_node("AssessEvidence", n_assess_evidence)
    g.add_node("RetrieveMoreEvidence", n_retrieve_more)
    g.add_node("PolicyDecision", n_policy_decision)
    g.add_node("Finalize", n_finalize)

    g.set_entry_point("LoadCase")
    g.add_edge("LoadCase", "CheckCorroboration")
    g.add_edge("CheckCorroboration", "AssessEvidence")
    g.add_edge("AssessEvidence", "RetrieveMoreEvidence")
    g.add_edge("RetrieveMoreEvidence", "PolicyDecision")
    g.add_edge("PolicyDecision", "Finalize")
    g.add_edge("Finalize", END)

    # LangGraph 1.0.3: compile() takes runtime config only (no output/final_state)
    return g.compile()


def run_underwriter_agent(
    case_row: Dict[str, Any],
    tools: UnderwriterTools,
    agent_id: str,
    policy_id: str,
) -> DecisionResult:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_underwriter_graph()

    init: UnderwriterState = {
        "case_row": dict(case_row),
        "reasons": [],
        "evidence": [],
        "signals": {},
        "tool_outputs": {},
        "tools": tools,
        "agent_id": str(agent_id),
        "policy_id": str(policy_id),
        "trace": {"steps": [], "tools_used": [], "llm_used": None, "fallback": False},
        "decision_result": DecisionResult("open", "init", [], {}),
    }

    out = _GRAPH.invoke(init)

    # In 1.0.3 this should be the final state dict
    if isinstance(out, dict) and isinstance(out.get("decision_result"), DecisionResult):
        return cast(DecisionResult, out["decision_result"])

    raise RuntimeError(f"LangGraph invoke returned unexpected output: {type(out)} {repr(out)[:200]}")
