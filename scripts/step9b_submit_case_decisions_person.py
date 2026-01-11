# scripts/step9b_submit_case_decisions_person.py
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from app.services.dataset_loader import load_dataset_df
from app.services.dataset_writer import write_parquet_dataset

# LangGraph agent runner + tools interface
from app.agents.underwriter_graph import run_underwriter_agent, UnderwriterTools


# ----------------------------
# Helpers / constants
# ----------------------------

VALID_DECISIONS = {"approved_match", "false_positive", "needs_more_info", "open"}

DEFAULT_POLICY_ID = "policy_rules_v1"
DEFAULT_AGENT_ID = "langgraph_v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_project_root() -> Path:
    import os

    pr = os.environ.get("PROJECT_ROOT")
    if pr:
        return Path(pr).resolve()
    return Path(__file__).resolve().parents[1]


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


def _json_dumps(x: Any, fallback: str) -> str:
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return fallback


# ----------------------------
# Ingest mode (human/agent submitted)
# ----------------------------

def _load_decisions_file(path: Path) -> pd.DataFrame:
    """
    Accepts:
    - JSONL: each line is a JSON object
    - CSV: must have at least case_id, decision
    Optional: decision_notes, reviewer, needs_more_info_fields_json, rationale_json
    """
    if not path.exists():
        raise FileNotFoundError(f"decisions file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported decisions file type. Use .jsonl or .csv")

    if "case_id" not in df.columns or "decision" not in df.columns:
        raise ValueError("Decisions file must include columns: case_id, decision")

    df["case_id"] = df["case_id"].astype("string")
    df["decision"] = df["decision"].astype("string").str.strip()

    bad = df[~df["decision"].isin(sorted(list(VALID_DECISIONS)))]
    if len(bad) > 0:
        raise ValueError(
            f"Invalid decisions found: {sorted(bad['decision'].dropna().unique().tolist())}. "
            f"Valid: {sorted(list(VALID_DECISIONS))}"
        )

    if "decision_notes" not in df.columns:
        df["decision_notes"] = ""
    if "reviewer" not in df.columns:
        df["reviewer"] = "underwriter_human"

    if "needs_more_info_fields_json" not in df.columns:
        df["needs_more_info_fields_json"] = "[]"
    if "rationale_json" not in df.columns:
        df["rationale_json"] = "{}"

    return df


# ----------------------------
# Expand to row-level decisions (analytics)
# ----------------------------

def _expand_to_row_level_decisions(case_packets: pd.DataFrame, case_decisions: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (person_id, watchlist_id) evidence item, carrying the case decision.
    """
    merged = case_packets.merge(
        case_decisions[["case_id", "decision", "decision_notes", "reviewer", "reviewed_at_utc", "decision_version"]],
        on="case_id",
        how="inner",
    )

    out_rows: List[Dict[str, Any]] = []

    for _, r in merged.iterrows():
        evidence = _safe_json_loads(r.get("evidence_matches_json"), default=[])
        evidence = evidence if isinstance(evidence, list) else []

        case_reasons = _safe_json_loads(r.get("case_reasons_json"), default=[])
        case_reasons = case_reasons if isinstance(case_reasons, list) else []

        for ev in evidence:
            if not isinstance(ev, dict):
                continue
            out_rows.append(
                {
                    "person_id": r.get("person_id"),
                    "person_name": r.get("person_name"),
                    "state": r.get("state"),
                    "zip": r.get("zip"),
                    "watchlist_id": ev.get("watchlist_id"),
                    "source_dataset_wl": ev.get("source_dataset_wl"),
                    "program": ev.get("program"),
                    "country": ev.get("country"),
                    "remarks": ev.get("remarks_preview", ""),
                    "match_method": ev.get("match_method"),
                    "match_score": ev.get("match_score"),
                    "qa_reasons": ",".join([str(x) for x in case_reasons]),
                    "decision": r.get("decision"),
                    "decision_notes": r.get("decision_notes"),
                    "reviewer": r.get("reviewer"),
                    "reviewed_at_utc": r.get("reviewed_at_utc"),
                    "decision_version": r.get("decision_version"),
                    "source_used": r.get("source_queue_dataset", "underwriter_case_packets_person"),
                    "clean_edges_count": pd.NA,
                    "clean_unique_watchlists": pd.NA,
                }
            )

    return pd.DataFrame(out_rows)


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs")

    ap.add_argument("--mode", choices=["simulate", "ingest", "agent"], default="simulate")
    ap.add_argument("--decisions-file", type=str, default=None, help="Path to .jsonl or .csv for ingest mode")

    ap.add_argument("--max-cases", type=int, default=None, help="Limit cases processed (fast demos)")
    ap.add_argument("--seed", type=int, default=7, help="Seed for simulate/agent mode")
    ap.add_argument("--reviewer", type=str, default="underwriter_sim", help="Reviewer name")
    ap.add_argument("--decision-version", type=str, default="v1", help="Decision schema version")

    ap.add_argument("--policy-id", type=str, default=DEFAULT_POLICY_ID, help="Policy id for evaluation comparisons")
    ap.add_argument("--agent-id", type=str, default=DEFAULT_AGENT_ID, help="Agent id for evaluation comparisons")

    ap.add_argument(
        "--decide-fraction",
        type=float,
        default=1.0,
        help="Fraction of cases to decide in simulate/agent mode (rest marked open). Range 0..1",
    )

    ap.add_argument(
        "--emit-row-level-decisions",
        action="store_true",
        help="Also emit underwriter_decisions_person_from_cases for analytics",
    )

    args = ap.parse_args()

    decide_fraction = float(args.decide_fraction or 1.0)
    if decide_fraction < 0.0 or decide_fraction > 1.0:
        raise ValueError("--decide-fraction must be within [0.0, 1.0]")

    project_root = infer_project_root()
    generated_at = utc_now_iso()

    # Input: case packets (from Step9A)
    case_packets_df, case_packets_loc = load_dataset_df(
        project_root,
        "underwriter_case_packets_person",
        max_parts=None,
        max_rows=None,
    )
    if len(case_packets_df) == 0:
        print("[WARN] No case packets found. Run Step9A first.")
        return 0

    if args.max_cases is not None and args.max_cases > 0:
        case_packets_df = case_packets_df.head(int(args.max_cases)).copy()

    reviewed_at = utc_now_iso()

    # ----------------------------
    # Build decisions
    # ----------------------------
    if args.mode == "ingest":
        if not args.decisions_file:
            raise ValueError("--decisions-file is required for --mode ingest")
        decisions_in = _load_decisions_file(Path(args.decisions_file))

        # Keep only decisions for known cases
        decisions = case_packets_df[["case_id", "person_id"]].merge(decisions_in, on="case_id", how="inner")

        missing_cases = set(decisions_in["case_id"].tolist()) - set(decisions["case_id"].tolist())
        if missing_cases:
            ex = next(iter(missing_cases))
            print(f"[WARN] {len(missing_cases)} decision rows had unknown case_id (ignored). Example: {ex}")

        decisions["reviewed_at_utc"] = reviewed_at
        decisions["decision_version"] = args.decision_version

    else:
        rng = random.Random(int(args.seed))

        n = len(case_packets_df)
        decide_n = int(round(n * decide_fraction))
        decide_n = max(0, min(n, decide_n))

        idx_all = list(range(n))
        rng.shuffle(idx_all)
        decide_idx = set(idx_all[:decide_n])

        # Tools passed into agent (can stay empty for now)
        tools = UnderwriterTools(project_root=project_root)

        rows: List[Dict[str, Any]] = []
        for i, (_, r) in enumerate(case_packets_df.iterrows()):
            if i not in decide_idx:
                rows.append(
                    {
                        "case_id": r.get("case_id"),
                        "person_id": r.get("person_id"),
                        "decision": "open",
                        "decision_notes": "Not yet reviewed (work-in-progress queue).",
                        "reviewer": args.reviewer,
                        "reviewed_at_utc": reviewed_at,
                        "decision_version": args.decision_version,
                        "needs_more_info_fields_json": "[]",
                        "rationale_json": "{}",
                    }
                )
                continue

            if args.mode == "agent":
                # ✅ REAL agent call (LangGraph)
                dr = run_underwriter_agent(
                    case_row=r.to_dict(),
                    tools=tools,
                    agent_id=str(args.agent_id),
                    policy_id=str(args.policy_id),
                )
                rows.append(
                    {
                        "case_id": r.get("case_id"),
                        "person_id": r.get("person_id"),
                        "decision": dr.decision,
                        "decision_notes": dr.decision_notes,
                        "reviewer": args.reviewer,
                        "reviewed_at_utc": reviewed_at,
                        "decision_version": args.decision_version,
                        "needs_more_info_fields_json": _json_dumps(dr.needs_more_info_fields, "[]"),
                        "rationale_json": _json_dumps(dr.rationale, "{}"),
                    }
                )
            else:
                # simulate mode: keep backwards-compatible (rules-like) behavior
                # If you want to keep your old rule engine, you can re-add it here.
                # For now we call the agent too, but stamp producer=simulate to keep clear.
                dr = run_underwriter_agent(
                    case_row=r.to_dict(),
                    tools=tools,
                    agent_id="simulate_rules_v1",
                    policy_id=str(args.policy_id),
                )
                # overwrite producer so it’s obvious it’s simulate
                if isinstance(dr.rationale, dict):
                    dr.rationale["producer"] = "simulate"
                rows.append(
                    {
                        "case_id": r.get("case_id"),
                        "person_id": r.get("person_id"),
                        "decision": dr.decision,
                        "decision_notes": dr.decision_notes,
                        "reviewer": args.reviewer,
                        "reviewed_at_utc": reviewed_at,
                        "decision_version": args.decision_version,
                        "needs_more_info_fields_json": _json_dumps(dr.needs_more_info_fields, "[]"),
                        "rationale_json": _json_dumps(dr.rationale, "{}"),
                    }
                )

        decisions = pd.DataFrame(rows)

    # ----------------------------
    # Output 1: case decisions (enriched)
    # ----------------------------
    enrich_cols = [
        "case_id",
        "case_type",
        "source_queue_dataset",
        "case_priority_score",
        "case_priority_rank",
        "evidence_match_count",
        "queue_status",
        "risk_tier",
        "sanctions_flag",
    ]
    enrich_cols = [c for c in enrich_cols if c in case_packets_df.columns]

    decisions_out = decisions.merge(case_packets_df[enrich_cols], on="case_id", how="left")

    out_case_decisions = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/05_underwriter_queue",
        dataset_key="underwriter_case_decisions_person",
        df=decisions_out,
        meta={
            "producer_step": "step9b_submit_case_decisions_person",
            "generated_at_utc": generated_at,
            "mode": args.mode,
            "inputs": [str(getattr(case_packets_loc, "dir_ref", "underwriter_case_packets_person"))],
            "counts": {
                "cases_in": int(len(case_packets_df)),
                "decisions_out": int(len(decisions_out)),
                "open": int((decisions_out["decision"] == "open").sum()),
                "approved_match": int((decisions_out["decision"] == "approved_match").sum()),
                "false_positive": int((decisions_out["decision"] == "false_positive").sum()),
                "needs_more_info": int((decisions_out["decision"] == "needs_more_info").sum()),
            },
            "args": {
                "max_cases": args.max_cases,
                "seed": args.seed,
                "reviewer": args.reviewer,
                "decision_version": args.decision_version,
                "decide_fraction": decide_fraction,
                "policy_id": args.policy_id,
                "agent_id": args.agent_id,
            },
        },
        overwrite=bool(args.overwrite),
    )

    print("[OK] Step9B complete (case decisions written)")
    print("input_case_packets_dir:", getattr(case_packets_loc, "dir_uri", str(case_packets_loc)))
    print("output_case_decisions_dir:", out_case_decisions)
    print("decisions rows:", len(decisions_out))
    print("breakdown:", decisions_out["decision"].value_counts(dropna=False).to_dict())

    # ----------------------------
    # Output 2: row-level decisions (optional)
    # ----------------------------
    if bool(args.emit_row_level_decisions):
        row_level = _expand_to_row_level_decisions(case_packets_df, decisions)
        out_row_level = write_parquet_dataset(
            project_root=project_root,
            tier_dir="data/05_underwriter_queue",
            dataset_key="underwriter_decisions_person_from_cases",
            df=row_level,
            meta={
                "producer_step": "step9b_submit_case_decisions_person",
                "generated_at_utc": generated_at,
                "mode": args.mode,
                "inputs": [
                    str(getattr(case_packets_loc, "dir_ref", "underwriter_case_packets_person")),
                    "underwriter_case_decisions_person",
                ],
                "counts": {
                    "row_level_rows": int(len(row_level)),
                    "unique_people": int(row_level["person_id"].nunique()) if "person_id" in row_level.columns else 0,
                    "unique_watchlists": int(row_level["watchlist_id"].nunique()) if "watchlist_id" in row_level.columns else 0,
                },
            },
            overwrite=bool(args.overwrite),
        )
        print("[OK] Row-level decisions emitted for analytics:")
        print("output_row_level_dir:", out_row_level)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
