from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from app.services.dataset_loader import load_dataset_df
from app.services.dataset_writer import write_parquet_dataset


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_project_root() -> Path:
    import os

    pr = os.environ.get("PROJECT_ROOT")
    if pr:
        return Path(pr).resolve()
    return Path(__file__).resolve().parents[1]


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        s = str(x)
    except Exception:
        return ""
    return "" if s.lower() in ("nan", "none") else s


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _split_reasons(s: Any) -> List[str]:
    txt = _as_str(s).strip()
    if not txt:
        return []
    return [t.strip() for t in txt.split(",") if t.strip()]


def _pick_person_cols(df: pd.DataFrame) -> List[str]:
    # Keep it stable and enterprise-friendly (don’t explode huge columns).
    want = [
        "person_id",
        "name",
        "state",
        "zip",
        "birth_date",
        "source_dataset",
        "name_norm",
        "sanctions_flag",
        "risk_tier",
        "matched_watchlist_count",
        "max_match_score",
        "matched_sources",
        # new corroboration fields from Step5C
        "ssn_last4",
        "email_hash",
        "phone_hash",
    ]
    return [c for c in want if c in df.columns]


def _pack_evidence_rows(rows: pd.DataFrame, max_matches_per_person: int) -> List[Dict[str, Any]]:
    """
    Convert multiple queue rows for one person into compact evidence list.
    """
    out: List[Dict[str, Any]] = []
    if rows.empty:
        return out

    # Sort by priority if present, otherwise by match_score desc
    if "priority_rank" in rows.columns:
        rows = rows.sort_values(["priority_rank"], ascending=True)
    elif "priority_score" in rows.columns:
        rows = rows.sort_values(["priority_score"], ascending=False)
    elif "match_score" in rows.columns:
        rows = rows.sort_values(["match_score"], ascending=False)

    take = rows.head(max_matches_per_person)

    for _, r in take.iterrows():
        ev = {
            "watchlist_id": _as_str(r.get("watchlist_id")),
            "source_dataset_wl": _as_str(r.get("source_dataset_wl")),
            "program": _as_str(r.get("program")),
            "country": _as_str(r.get("country")),
            "match_method": _as_str(r.get("match_method")),
            "match_score": _safe_float(r.get("match_score")),
            "qa_reasons": _split_reasons(r.get("qa_reasons")),
            "remarks_preview": (_as_str(r.get("remarks"))[:240] + "...") if _as_str(r.get("remarks")) else "",
        }

        # Add explainability components if present (from Step8B upgraded version)
        for c in (
            "priority_base",
            "priority_match_score",
            "priority_fuzzy",
            "priority_common_name",
            "priority_missing_location",
            "priority_borderline",
            "priority_score",
            "priority_rank",
        ):
            if c in rows.columns:
                ev[c] = r.get(c)

        out.append(ev)

    return out


def _make_case_id(person_id: Any, ranked_at_utc: str) -> str:
    # Stable-enough id for local lab (enterprise would use UUID + run id)
    return f"case_person_{_as_str(person_id)}_{ranked_at_utc.replace(':','').replace('-','').replace('.','')}"


def _coalesce_str(a: Any, b: Any) -> str:
    """
    Prefer a if it is non-empty, else b.
    """
    sa = _as_str(a).strip()
    if sa:
        return sa
    sb = _as_str(b).strip()
    return sb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs")
    ap.add_argument("--top-only", action="store_true", help="Use underwriter_review_queue_person_top instead of ranked full")
    ap.add_argument("--max-cases", type=int, default=None, help="Optional cap on number of cases emitted")
    ap.add_argument("--max-matches-per-person", type=int, default=5, help="How many watchlist matches to include per person")
    ap.add_argument("--min-priority", type=float, default=None, help="Optional minimum priority_score to include")
    args = ap.parse_args()

    project_root = infer_project_root()
    generated_at = utc_now_iso()

    # Inputs
    queue_key = "underwriter_review_queue_person_top" if args.top_only else "underwriter_review_queue_person_ranked"
    queue_df, queue_loc = load_dataset_df(project_root, queue_key, max_parts=None, max_rows=None)

    profile_df, profile_loc = load_dataset_df(project_root, "borrower360_person_profile", max_parts=None, max_rows=None)

    # Defensive: ensure person_id exists
    if "person_id" not in queue_df.columns:
        raise SystemExit(f"[ERROR] {queue_key} missing required column 'person_id'")
    if "person_id" not in profile_df.columns:
        raise SystemExit("[ERROR] borrower360_person_profile missing required column 'person_id'")

    # Optional filter by min_priority
    if args.min_priority is not None and "priority_score" in queue_df.columns:
        queue_df = queue_df[queue_df["priority_score"].apply(_safe_float) >= float(args.min_priority)].copy()

    # Determine per-person priority summary (use best/first row)
    sort_cols: List[Tuple[str, bool]] = []
    if "priority_rank" in queue_df.columns:
        sort_cols.append(("priority_rank", True))
    elif "priority_score" in queue_df.columns:
        sort_cols.append(("priority_score", False))
    elif "match_score" in queue_df.columns:
        sort_cols.append(("match_score", False))

    if sort_cols:
        col, asc = sort_cols[0]
        queue_df = queue_df.sort_values([col], ascending=asc)

    # Build a person-level table of "best row" for case-level fields
    first_rows = queue_df.groupby("person_id", as_index=False).head(1).copy()

    # Join profile (left join: only people in queue produce cases)
    person_cols = _pick_person_cols(profile_df)
    profile_small = profile_df[person_cols].copy()
    cases = first_rows.merge(profile_small, on="person_id", how="left", suffixes=("_q", "_p"))

    # --- Canonicalize person attributes to avoid merge collisions ---
    # Prefer profile fields (_p) over queue fields (_q) because Step5C is authoritative.
    # If queue doesn't have those, pandas may not create _q/_p columns; we handle both cases.

    # Helper to resolve potential suffix columns into a canonical column name
    def _canon(col: str) -> None:
        q = f"{col}_q"
        p = f"{col}_p"
        if q in cases.columns and p in cases.columns:
            cases[col] = cases.apply(lambda r: _coalesce_str(r.get(p), r.get(q)), axis=1)
        elif p in cases.columns and col not in cases.columns:
            cases[col] = cases[p].apply(lambda x: _as_str(x).strip())
        elif q in cases.columns and col not in cases.columns:
            cases[col] = cases[q].apply(lambda x: _as_str(x).strip())
        elif col in cases.columns:
            # normalize to string-ish consistency
            cases[col] = cases[col].apply(lambda x: _as_str(x).strip())

    for c in ["name", "name_norm", "state", "zip", "birth_date", "source_dataset"]:
        _canon(c)

    # New corroboration fields should just come from profile if present
    for c in ["ssn_last4", "email_hash", "phone_hash"]:
        if c in cases.columns:
            cases[c] = cases[c].apply(lambda x: _as_str(x).strip())

    # Case-level derived fields
    ranked_at = utc_now_iso()
    cases["case_id"] = cases["person_id"].apply(lambda x: _make_case_id(x, ranked_at))
    cases["case_type"] = "person_sanctions_review"
    cases["generated_at_utc"] = generated_at
    cases["ranked_at_utc"] = ranked_at
    cases["source_queue_dataset"] = queue_key

    # Explainability summary at case level
    cases["case_priority_score"] = cases["priority_score"] if "priority_score" in cases.columns else pd.NA
    cases["case_priority_rank"] = cases["priority_rank"] if "priority_rank" in cases.columns else pd.NA
    cases["case_reasons"] = cases.get("qa_reasons", pd.Series([""] * len(cases))).apply(_split_reasons)

    # Build evidence lists
    evidence_map: Dict[str, List[Dict[str, Any]]] = {}
    for pid, grp in queue_df.groupby("person_id"):
        evidence_map[_as_str(pid)] = _pack_evidence_rows(grp, max_matches_per_person=int(args.max_matches_per_person))

    cases["evidence_matches"] = cases["person_id"].apply(lambda x: evidence_map.get(_as_str(x), []))
    cases["evidence_match_count"] = cases["evidence_matches"].apply(lambda xs: int(len(xs)))

    # Add reviewer-ready “ask” fields (what the human/agent should do)
    cases["review_action"] = "review_identity_and_sanctions_match"
    cases["review_questions"] = cases["case_reasons"].apply(
        lambda rs: [
            "Is this the same individual as the watchlist entry?",
            "Do we have enough corroborating attributes (state/zip/DOB) to confirm?",
            "If not, what additional evidence is needed?",
        ]
        + (["Does fuzzy matching appear justified?"] if "fuzzy_match" in rs else [])
    )

    # If you want: quick recommended decision heuristic (NOT a final decision)
    def _recommend(row: pd.Series) -> str:
        rs = set(row.get("case_reasons", []))
        score = _safe_float(row.get("match_score"))
        if "borderline_score" in rs or (90 <= score <= 98):
            return "needs_review"
        if "missing_state_or_zip" in rs or "common_or_ambiguous_name" in rs:
            return "needs_review"
        return "needs_review"

    cases["recommendation"] = cases.apply(_recommend, axis=1)

    # Keep output columns tight and consistent
    keep = [
        "case_id",
        "case_type",
        "generated_at_utc",
        "ranked_at_utc",
        "source_queue_dataset",
        "person_id",
        "person_name",
        "name",
        "name_norm",
        "state",
        "zip",
        "birth_date",
        "ssn_last4",
        "email_hash",
        "phone_hash",
        "source_dataset",
        "sanctions_flag",
        "risk_tier",
        "matched_watchlist_count",
        "max_match_score",
        "matched_sources",
        "queue_status",
        "match_method",
        "match_score",
        "case_priority_score",
        "case_priority_rank",
        "case_reasons",
        "evidence_match_count",
        "evidence_matches",
        "review_action",
        "review_questions",
        "recommendation",
    ]
    keep = [c for c in keep if c in cases.columns]
    cases_out = cases[keep].copy()

    # Optional cap
    if args.max_cases is not None:
        cases_out = cases_out.head(int(args.max_cases)).copy()

    # Ensure evidence fields are JSON-serializable for parquet roundtrip
    def _to_json_str(x: Any) -> str:
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return "[]"

    cases_out["case_reasons_json"] = cases_out["case_reasons"].apply(_to_json_str)
    cases_out["evidence_matches_json"] = cases_out["evidence_matches"].apply(_to_json_str)
    cases_out["review_questions_json"] = cases_out["review_questions"].apply(_to_json_str)

    # Drop python-object columns
    cases_out = cases_out.drop(columns=["case_reasons", "evidence_matches", "review_questions"], errors="ignore")

    out_dir = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/05_underwriter_queue",
        dataset_key="underwriter_case_packets_person",
        df=cases_out,
        meta={
            "producer_step": "step9a_build_underwriter_case_packets_person",
            "generated_at_utc": generated_at,
            "inputs": [str(queue_loc.dir_ref), str(profile_loc.dir_ref)],
            "counts": {
                "queue_rows_in": int(len(queue_df)),
                "case_rows_out": int(len(cases_out)),
                "unique_people": int(cases_out["person_id"].nunique()) if "person_id" in cases_out.columns else 0,
                "max_matches_per_person": int(args.max_matches_per_person),
                "top_only": bool(args.top_only),
                "min_priority": args.min_priority,
                "max_cases": args.max_cases,
            },
        },
        overwrite=bool(args.overwrite),
    )

    print("[OK] Step9A complete (underwriter case packets built)")
    print("input_queue:", queue_loc.dir_uri)
    print("input_profile:", profile_loc.dir_uri)
    print("output_dir:", out_dir)
    print("queue_rows_in:", int(len(queue_df)))
    print("cases_out:", int(len(cases_out)))
    if "person_id" in cases_out.columns:
        print("unique_people:", int(cases_out["person_id"].nunique()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
