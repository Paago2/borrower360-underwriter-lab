from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
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


DECISIONS = ("approved_match", "false_positive", "needs_more_info")


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ensure df has these cols (enterprise defensive programming)."""
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def _pick_source_rows(
    queue_df: pd.DataFrame,
    clean_edges_df: pd.DataFrame,
    sample_n: int,
    seed: int,
) -> Tuple[pd.DataFrame, str]:
    """
    Prefer queue rows (true underwriter workflow).
    If queue is empty, fall back to clean edges so pipeline stays runnable.
    """
    rng = np.random.default_rng(seed)

    if queue_df is not None and len(queue_df) > 0:
        src = queue_df.copy()
        src_name = "underwriter_review_queue_person"
    else:
        # fallback source
        src = clean_edges_df.copy()
        src_name = "edges_person_sanctions_clean"

        # Map clean_edges -> queue-like schema so downstream logic stays consistent
        # (keep only fields that Step6C needs)
        keep = [
            "person_id",
            "person_name",
            "state",
            "zip",
            "watchlist_id",
            "source_dataset_wl",
            "program",
            "country",
            "remarks",
            "match_method",
            "match_score",
            "qa_reasons",
        ]
        src = _ensure_cols(src, keep)
        src = src[keep].copy()

        # For fallback we mark as needs_review so the simulator has something to "review"
        src["queue_status"] = "needs_review"
        src["generated_at_utc"] = utc_now_iso()

    if len(src) == 0:
        # Nothing anywhere -> return empty with correct schema
        empty = pd.DataFrame(columns=[
            "person_id", "person_name", "state", "zip", "watchlist_id",
            "source_dataset_wl", "program", "country", "remarks",
            "match_method", "match_score", "qa_reasons", "queue_status", "generated_at_utc"
        ])
        return empty, src_name

    n = min(int(sample_n), int(len(src)))
    idx = rng.choice(np.arange(len(src)), size=n, replace=False)
    return src.iloc[idx].reset_index(drop=True), src_name


def _decision_policy(row: pd.Series, rng: np.random.Generator) -> Tuple[str, str]:
    """
    Simple deterministic-ish policy:
      - If qa_reasons present -> mostly false_positive
      - If match_score < 95 -> needs_more_info sometimes
      - Otherwise -> approved_match sometimes, but mostly false_positive in demo
    Adjust to taste.
    """
    reasons = str(row.get("qa_reasons", "") or "").strip()
    score = row.get("match_score", pd.NA)
    try:
        score_f = float(score)
    except Exception:
        score_f = float("nan")

    # If any QA reasons exist, likely false positive / low confidence
    if reasons and reasons.lower() != "nan":
        # 95% false_positive, 5% needs_more_info
        if rng.random() < 0.95:
            return "false_positive", "auto: qa_reasons_present"
        return "needs_more_info", "auto: qa_reasons_present"

    # No reasons: use score
    if not np.isfinite(score_f):
        # unknown score -> needs more info
        return "needs_more_info", "auto: missing_score"

    if score_f < 95:
        # lower score -> more needs_more_info
        if rng.random() < 0.70:
            return "needs_more_info", "auto: low_score"
        return "false_positive", "auto: low_score"

    # high score no reasons: mix
    r = rng.random()
    if r < 0.05:
        return "approved_match", "auto: high_score_approve"
    if r < 0.10:
        return "needs_more_info", "auto: high_score_needs_more"
    return "false_positive", "auto: high_score_false_positive"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output dataset")
    ap.add_argument("--sample-n", type=int, default=200, help="How many rows to simulate decisions for")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed for reproducibility")
    args = ap.parse_args()

    project_root = infer_project_root()
    generated_at = utc_now_iso()
    rng = np.random.default_rng(int(args.seed))

    # Inputs
    queue_df, queue_loc = load_dataset_df(project_root, "underwriter_review_queue_person", max_parts=None, max_rows=None)
    clean_df, clean_loc = load_dataset_df(project_root, "edges_person_sanctions_clean", max_parts=None, max_rows=None)

    # Choose source rows (queue preferred, else clean fallback)
    src_rows, source_used = _pick_source_rows(queue_df, clean_df, int(args.sample_n), int(args.seed))

    if len(src_rows) == 0:
        # Write empty decisions dataset (still enterprise-consistent folder dataset)
        decisions = pd.DataFrame(columns=[
            "person_id", "person_name", "state", "zip",
            "watchlist_id", "source_dataset_wl", "program", "country", "remarks",
            "match_method", "match_score", "qa_reasons",
            "decision", "decision_notes", "reviewer", "reviewed_at_utc",
            "decision_version", "source_used",
            "clean_edges_count", "clean_unique_watchlists",
        ])
    else:
        # Build evidence from clean edges (per person)
        if "person_id" in clean_df.columns and len(clean_df) > 0:
            evidence = clean_df.groupby("person_id").agg(
                clean_edges_count=("watchlist_id", "count"),
                clean_unique_watchlists=("watchlist_id", "nunique"),
            ).reset_index()
        else:
            evidence = pd.DataFrame(columns=["person_id", "clean_edges_count", "clean_unique_watchlists"])

        decisions = src_rows.copy()
        decisions = _ensure_cols(decisions, [
            "person_id", "person_name", "state", "zip",
            "watchlist_id", "source_dataset_wl", "program", "country", "remarks",
            "match_method", "match_score", "qa_reasons"
        ])

        # Decide
        dec = []
        notes = []
        for _, r in decisions.iterrows():
            d, n = _decision_policy(r, rng)
            dec.append(d)
            notes.append(n)

        decisions["decision"] = pd.Series(dec, dtype="string")
        decisions["decision_notes"] = pd.Series(notes, dtype="string")
        decisions["reviewer"] = "underwriter_sim"
        decisions["reviewed_at_utc"] = generated_at
        decisions["decision_version"] = "v1"
        decisions["source_used"] = source_used

        # Safe merge evidence
        if "person_id" in decisions.columns and "person_id" in evidence.columns and len(evidence) > 0:
            decisions = decisions.merge(evidence, on="person_id", how="left")
        else:
            decisions["clean_edges_count"] = pd.NA
            decisions["clean_unique_watchlists"] = pd.NA

        # Keep stable column order
        decisions = decisions[[
            "person_id", "person_name", "state", "zip",
            "watchlist_id", "source_dataset_wl", "program", "country", "remarks",
            "match_method", "match_score", "qa_reasons",
            "decision", "decision_notes", "reviewer", "reviewed_at_utc",
            "decision_version", "source_used",
            "clean_edges_count", "clean_unique_watchlists",
        ]].copy()

    # Write folder dataset
    out_dir = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/05_underwriter_queue",
        dataset_key="underwriter_decisions_person",
        df=decisions,
        meta={
            "producer_step": "step6c_underwriter_decisions_simulator",
            "generated_at_utc": generated_at,
            "inputs": [
                str(queue_loc.dir_ref),
                str(clean_loc.dir_ref),
            ],
            "source_used": source_used,
            "counts": {
                "reviewed_rows": int(len(decisions)),
                "queue_rows": int(len(queue_df)),
                "clean_edges_rows": int(len(clean_df)),
            },
            "params": {"sample_n": int(args.sample_n), "seed": int(args.seed)},
        },
        overwrite=bool(args.overwrite),
    )

    # Print summary
    print("[OK] Step6C complete (underwriter decisions simulated)")
    print("input_queue_dir:", queue_loc.dir_uri)
    print("input_clean_edges_dir:", clean_loc.dir_uri)
    print("source_used:", source_used)
    print("output_dir:", out_dir)
    print("reviewed rows:", int(len(decisions)))
    if len(decisions) > 0 and "decision" in decisions.columns:
        print("decisions breakdown:")
        print(decisions["decision"].value_counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

