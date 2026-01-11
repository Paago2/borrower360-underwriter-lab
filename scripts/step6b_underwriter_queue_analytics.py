# scripts/step6b_underwriter_queue_analytics.py
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

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


def _safe_split_reasons(s: Any) -> list[str]:
    """Split qa_reasons like 'a,b,c' -> ['a','b','c'] safely."""
    if s is None:
        return []
    if isinstance(s, float) and pd.isna(s):
        return []
    txt = str(s).strip()
    if not txt:
        return []
    return [x.strip() for x in txt.split(",") if x.strip()]


def _add_reasons_exploded(queue_df: pd.DataFrame) -> pd.DataFrame:
    """Return a 2-col df: (queue_row_id, reason) for counting reasons."""
    if "qa_reasons" not in queue_df.columns:
        return pd.DataFrame(columns=["queue_row_id", "reason"])

    q = queue_df.reset_index(drop=True).copy()
    q["queue_row_id"] = q.index.astype("int64")
    q["__reasons"] = q["qa_reasons"].apply(_safe_split_reasons)

    exploded = q[["queue_row_id", "__reasons"]].explode("__reasons")
    exploded = exploded.rename(columns={"__reasons": "reason"})
    exploded = exploded.dropna(subset=["reason"])
    exploded["reason"] = exploded["reason"].astype("string")
    exploded = exploded[exploded["reason"].str.len() > 0]
    return exploded[["queue_row_id", "reason"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output datasets if they exist")
    ap.add_argument("--max-rows", type=int, default=None, help="Optional cap for faster local runs")
    args = ap.parse_args()

    project_root = infer_project_root()

    # ---- Load inputs (enterprise folder datasets) ----
    queue_df, queue_loc = load_dataset_df(
        project_root,
        "underwriter_review_queue_person",
        max_parts=None,
        max_rows=args.max_rows,
    )

    clean_edges_df, clean_loc = load_dataset_df(
        project_root,
        "edges_person_sanctions_clean",
        max_parts=None,
        max_rows=None,  # typically small (yours is 12)
    )

    # Normalize a few expected columns
    queue_df = queue_df.copy()

    for col in ["queue_status", "match_method", "source_dataset_wl", "program", "country"]:
        if col in queue_df.columns:
            queue_df[col] = queue_df[col].astype("string").fillna("")

    if "match_score" in queue_df.columns:
        queue_df["match_score"] = pd.to_numeric(queue_df["match_score"], errors="coerce")

    # ---- 1) KPIs dataset (single-row, reusable for dashboards) ----
    total_queue_rows = int(len(queue_df))

    def _nunique_if_present(df: pd.DataFrame, col: str) -> int:
        return int(df[col].nunique()) if col in df.columns else 0

    kpis: Dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "queue_rows": total_queue_rows,
        "queue_unique_people": _nunique_if_present(queue_df, "person_id"),
        "queue_unique_watchlists": _nunique_if_present(queue_df, "watchlist_id"),
        "queue_status_present": int("queue_status" in queue_df.columns),
        "queue_auto_filtered_rows": int((queue_df["queue_status"] == "auto_filtered").sum()) if "queue_status" in queue_df.columns else 0,
        "queue_needs_review_rows": int((queue_df["queue_status"] == "needs_review").sum()) if "queue_status" in queue_df.columns else 0,
        "clean_edges_rows": int(len(clean_edges_df)),
        "clean_edges_unique_people": _nunique_if_present(clean_edges_df, "person_id"),
        "clean_edges_unique_watchlists": _nunique_if_present(clean_edges_df, "watchlist_id"),
        "avg_match_score_queue": float(queue_df["match_score"].mean()) if "match_score" in queue_df.columns and total_queue_rows > 0 else None,
        "min_match_score_queue": float(queue_df["match_score"].min()) if "match_score" in queue_df.columns and total_queue_rows > 0 else None,
        "max_match_score_queue": float(queue_df["match_score"].max()) if "match_score" in queue_df.columns and total_queue_rows > 0 else None,
    }

    kpis_df = pd.DataFrame([kpis])

    # ---- 2) Breakdown dataset (status / method / source / program) ----
    breakdown_rows = []

    def _add_breakdown(df: pd.DataFrame, dim: str, label: str) -> None:
        if dim not in df.columns:
            return
        counts = (
            df.groupby(dim, dropna=False)
              .size()
              .reset_index(name="rows")
              .rename(columns={dim: "value"})
        )
        counts["dimension"] = label
        counts["value"] = counts["value"].astype("string").fillna("")
        breakdown_rows.append(counts[["dimension", "value", "rows"]])

    _add_breakdown(queue_df, "queue_status", "queue_status")
    _add_breakdown(queue_df, "match_method", "match_method")
    _add_breakdown(queue_df, "source_dataset_wl", "source_dataset_wl")
    _add_breakdown(queue_df, "program", "program")
    _add_breakdown(queue_df, "country", "country")

    if breakdown_rows:
        breakdown_df = pd.concat(breakdown_rows, ignore_index=True)
        breakdown_df["rows"] = breakdown_df["rows"].astype("int64")
        breakdown_df = breakdown_df.sort_values(["dimension", "rows"], ascending=[True, False]).reset_index(drop=True)
    else:
        breakdown_df = pd.DataFrame(columns=["dimension", "value", "rows"])

    # ---- 3) QA reasons dataset (exploded counts) ----
    exploded = _add_reasons_exploded(queue_df)
    if len(exploded) > 0:
        reasons_df = (
            exploded.groupby("reason")
            .size()
            .reset_index(name="rows")
            .sort_values("rows", ascending=False)
            .reset_index(drop=True)
        )
        reasons_df["rows"] = reasons_df["rows"].astype("int64")
    else:
        reasons_df = pd.DataFrame(columns=["reason", "rows"])

    # ---- Write outputs as folder datasets (enterprise pattern) ----
    meta_base = {
        "producer_step": "step6b_underwriter_queue_analytics",
        "inputs": [
            queue_loc.dir_ref,
            clean_loc.dir_ref,
        ],
        "counts": {
            "queue_rows": int(len(queue_df)),
            "clean_edges_rows": int(len(clean_edges_df)),
        },
    }

    out_kpis = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/06_monitoring",
        dataset_key="underwriter_queue_person_kpis",
        df=kpis_df,
        meta={**meta_base, "dataset": "kpis"},
        overwrite=bool(args.overwrite),
    )

    out_breakdown = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/06_monitoring",
        dataset_key="underwriter_queue_person_breakdown",
        df=breakdown_df,
        meta={**meta_base, "dataset": "breakdown"},
        overwrite=bool(args.overwrite),
    )

    out_reasons = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/06_monitoring",
        dataset_key="underwriter_queue_person_reasons",
        df=reasons_df,
        meta={**meta_base, "dataset": "reasons"},
        overwrite=bool(args.overwrite),
    )

    print("[OK] Step6B complete (underwriter queue analytics)")
    print("inputs:")
    print(" - queue:", queue_loc.dir_uri)
    print(" - clean_edges:", clean_loc.dir_uri)
    print("outputs:")
    print(" - kpis:", out_kpis)
    print(" - breakdown:", out_breakdown)
    print(" - reasons:", out_reasons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
