from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

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


def _safe_mean(s: pd.Series) -> float:
    try:
        return float(pd.to_numeric(s, errors="coerce").dropna().mean())
    except Exception:
        return float("nan")


def _safe_min(s: pd.Series) -> float:
    try:
        return float(pd.to_numeric(s, errors="coerce").dropna().min())
    except Exception:
        return float("nan")


def _safe_max(s: pd.Series) -> float:
    try:
        return float(pd.to_numeric(s, errors="coerce").dropna().max())
    except Exception:
        return float("nan")


def _explode_reasons(df: pd.DataFrame, col: str = "qa_reasons") -> pd.DataFrame:
    if df is None or df.empty or col not in df.columns:
        return pd.DataFrame({"reason": pd.Series(dtype="string"), "rows": pd.Series(dtype="int64")})

    s = df[col].fillna("").astype("string")
    tokens = s.str.split(",")
    exploded = tokens.explode().astype("string").str.strip()
    exploded = exploded[exploded.str.len() > 0]

    if exploded.empty:
        return pd.DataFrame({"reason": pd.Series(dtype="string"), "rows": pd.Series(dtype="int64")})

    out = exploded.value_counts().reset_index()
    out.columns = ["reason", "rows"]
    out["rows"] = out["rows"].astype("int64")
    out["reason"] = out["reason"].astype("string")
    return out


def _breakdown(df: pd.DataFrame, dim: str) -> pd.DataFrame:
    if df is None or df.empty or dim not in df.columns:
        return pd.DataFrame(
            {
                "dimension": pd.Series(dtype="string"),
                "value": pd.Series(dtype="string"),
                "rows": pd.Series(dtype="int64"),
            }
        )

    vc = df[dim].fillna("").astype("string").value_counts().reset_index()
    vc.columns = ["value", "rows"]
    vc.insert(0, "dimension", dim)
    vc["dimension"] = vc["dimension"].astype("string")
    vc["value"] = vc["value"].astype("string")
    vc["rows"] = vc["rows"].astype("int64")
    return vc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output datasets")
    args = ap.parse_args()

    project_root = infer_project_root()
    generated_at = utc_now_iso()

    # Inputs
    decisions_df, decisions_loc = load_dataset_df(
        project_root, "underwriter_decisions_person", max_parts=None, max_rows=None
    )

    # Basic counts
    total_rows = int(len(decisions_df))
    unique_people = int(decisions_df["person_id"].nunique()) if "person_id" in decisions_df.columns else 0
    unique_watchlists = int(decisions_df["watchlist_id"].nunique()) if "watchlist_id" in decisions_df.columns else 0

    # Decision breakdown
    if "decision" in decisions_df.columns:
        decision_counts = decisions_df["decision"].fillna("").astype("string").value_counts().to_dict()
    else:
        decision_counts = {}

    approved = int(decision_counts.get("approved_match", 0))
    false_pos = int(decision_counts.get("false_positive", 0))
    needs_more = int(decision_counts.get("needs_more_info", 0))

    reviewed = total_rows
    approval_rate = (approved / reviewed) if reviewed else 0.0
    false_positive_rate = (false_pos / reviewed) if reviewed else 0.0
    needs_more_info_rate = (needs_more / reviewed) if reviewed else 0.0

    # Match score stats
    score_mean = _safe_mean(decisions_df.get("match_score", pd.Series([], dtype="float")))
    score_min = _safe_min(decisions_df.get("match_score", pd.Series([], dtype="float")))
    score_max = _safe_max(decisions_df.get("match_score", pd.Series([], dtype="float")))

    # KPIs dataset (1 row per run)
    kpis = pd.DataFrame(
        [
            {
                "generated_at_utc": generated_at,
                "decisions_rows": total_rows,
                "decisions_unique_people": unique_people,
                "decisions_unique_watchlists": unique_watchlists,
                "approved_match_rows": approved,
                "false_positive_rows": false_pos,
                "needs_more_info_rows": needs_more,
                "approval_rate": float(approval_rate),
                "false_positive_rate": float(false_positive_rate),
                "needs_more_info_rate": float(needs_more_info_rate),
                "avg_match_score": float(score_mean),
                "min_match_score": float(score_min),
                "max_match_score": float(score_max),
            }
        ]
    )

    # Breakdown dataset (avoid FutureWarning by concatenating only non-empty frames)
    parts = [
        _breakdown(decisions_df, "decision"),
        _breakdown(decisions_df, "queue_status"),
        _breakdown(decisions_df, "match_method"),
        _breakdown(decisions_df, "source_dataset_wl"),
        _breakdown(decisions_df, "program"),
        _breakdown(decisions_df, "country"),
        _breakdown(decisions_df, "reviewer"),
        _breakdown(decisions_df, "decision_version"),
    ]
    parts = [p for p in parts if p is not None and not p.empty]

    if parts:
        breakdown = pd.concat(parts, ignore_index=True)
    else:
        breakdown = pd.DataFrame(
            {
                "dimension": pd.Series(dtype="string"),
                "value": pd.Series(dtype="string"),
                "rows": pd.Series(dtype="int64"),
            }
        )

    # Reasons dataset
    reasons_all = _explode_reasons(decisions_df, "qa_reasons")

    if "decision" in decisions_df.columns and not decisions_df.empty:
        fp = decisions_df[decisions_df["decision"].astype("string") == "false_positive"].copy()
        reasons_fp = _explode_reasons(fp, "qa_reasons")
        if not reasons_fp.empty:
            reasons_fp = reasons_fp.rename(columns={"rows": "false_positive_rows"})
            reasons_all = reasons_all.merge(reasons_fp, on="reason", how="left")
        else:
            reasons_all["false_positive_rows"] = pd.NA
    else:
        reasons_all["false_positive_rows"] = pd.NA

    # Write folder datasets
    out_kpis = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/06_monitoring",
        dataset_key="underwriter_decisions_person_kpis",
        df=kpis,
        meta={
            "producer_step": "step6d_underwriter_decisions_feedback_analytics",
            "generated_at_utc": generated_at,
            "inputs": [str(decisions_loc.dir_ref)],
            "counts": {
                "decisions_rows": total_rows,
                "approved_match_rows": approved,
                "false_positive_rows": false_pos,
                "needs_more_info_rows": needs_more,
            },
        },
        overwrite=bool(args.overwrite),
    )

    out_breakdown = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/06_monitoring",
        dataset_key="underwriter_decisions_person_breakdown",
        df=breakdown,
        meta={
            "producer_step": "step6d_underwriter_decisions_feedback_analytics",
            "generated_at_utc": generated_at,
            "inputs": [str(decisions_loc.dir_ref)],
            "counts": {"rows": int(len(breakdown))},
        },
        overwrite=bool(args.overwrite),
    )

    out_reasons = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/06_monitoring",
        dataset_key="underwriter_decisions_person_reasons",
        df=reasons_all,
        meta={
            "producer_step": "step6d_underwriter_decisions_feedback_analytics",
            "generated_at_utc": generated_at,
            "inputs": [str(decisions_loc.dir_ref)],
            "counts": {"rows": int(len(reasons_all))},
        },
        overwrite=bool(args.overwrite),
    )

    print("[OK] Step6D complete (decisions feedback analytics)")
    print("input_decisions_dir:", decisions_loc.dir_uri)
    print("outputs:")
    print(" - kpis:", out_kpis)
    print(" - breakdown:", out_breakdown)
    print(" - reasons:", out_reasons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
