from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List

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


def _token_count(s: str) -> int:
    if not s:
        return 0
    return len([t for t in str(s).split(" ") if t])


def _add_reason(row_reasons: List[str], reason: str) -> None:
    if reason not in row_reasons:
        row_reasons.append(reason)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output datasets")
    ap.add_argument("--emit-queue", action="store_true", help="Emit underwriter review queue dataset")

    # Auto-accept rules
    ap.add_argument(
        "--auto-accept-exact-score",
        type=int,
        default=100,
        help="Auto-accept exact matches if score >= this (typically 100).",
    )
    ap.add_argument(
        "--auto-accept-fuzzy-min-score",
        type=int,
        default=99,
        help="Auto-accept fuzzy matches only if score >= this AND not ambiguous. (default: 99)",
    )

    # Borderline routing (queue)
    ap.add_argument("--borderline-min", type=int, default=90, help="Borderline lower bound (route to queue)")
    ap.add_argument("--borderline-max", type=int, default=98, help="Borderline upper bound (route to queue)")

    args = ap.parse_args()

    project_root = infer_project_root()
    generated_at = utc_now_iso()

    edges_df, edges_loc = load_dataset_df(project_root, "edges_person_sanctions", max_parts=None, max_rows=None)
    edges_in = int(len(edges_df))

    df = edges_df.copy()

    # Ensure columns exist
    for c in [
        "match_score", "match_method", "person_name", "state", "zip",
        "name_commonness_bucket", "is_fuzzy"
    ]:
        if c not in df.columns:
            df[c] = pd.NA

    # Normalize types
    df["match_score"] = pd.to_numeric(df["match_score"], errors="coerce")
    df["match_method"] = df["match_method"].fillna("").astype("string")
    df["person_name"] = df["person_name"].fillna("").astype("string")
    df["state"] = df["state"].fillna("").astype("string")
    df["zip"] = df["zip"].fillna("").astype("string")
    df["name_commonness_bucket"] = df["name_commonness_bucket"].fillna("").astype("string")

    # Derive is_fuzzy if missing
    if df["is_fuzzy"].isna().all():
        df["is_fuzzy"] = df["match_method"].astype("string").str.startswith("fuzzy_")
    else:
        df["is_fuzzy"] = df["is_fuzzy"].fillna(False).astype(bool)

    # Helper flags
    df["score_i"] = pd.to_numeric(df["match_score"], errors="coerce").fillna(-1).astype(int)
    df["is_exact"] = df["match_method"].astype("string").eq("exact_name_norm")
    df["is_fuzzy2"] = df["is_fuzzy"].astype(bool) | df["match_method"].astype("string").str.startswith("fuzzy_")

    # Build QA reasons (explainable)
    reasons_list: List[List[str]] = []
    for _, r in df.iterrows():
        reasons: List[str] = []
        score_i = int(r.get("score_i", -1))
        name = str(r.get("person_name", "") or "")
        state = str(r.get("state", "") or "")
        zipc = str(r.get("zip", "") or "")
        bucket = str(r.get("name_commonness_bucket", "") or "")
        is_fuzzy = bool(r.get("is_fuzzy2", False))

        # Name quality / ambiguity
        if _token_count(name) <= 1:
            _add_reason(reasons, "single_token_name")
        if len(name.strip()) < 8:
            _add_reason(reasons, "name_too_short<8")

        # "Common/ambiguous" is useful, but should NOT automatically block exact=100
        if bucket in ("single_token", "two_token", "short_name"):
            _add_reason(reasons, "common_or_ambiguous_name")

        # Geo missing: common in LendingClub demo, so keep as info not a hard block for exact=100
        if (not state.strip()) or (not zipc.strip()):
            _add_reason(reasons, "missing_state_or_zip")

        if is_fuzzy:
            _add_reason(reasons, "fuzzy_match")

        if args.borderline_min <= score_i <= args.borderline_max:
            _add_reason(reasons, "borderline_score")

        reasons_list.append(reasons)

    df["qa_reasons"] = [",".join(r) for r in reasons_list]
    df["qa_has_reasons"] = df["qa_reasons"].astype("string").str.len() > 0

    # Routing logic (enterprise calibration)
    # 1) AUTO-ACCEPT (clean)
    #    A) exact match at 100 -> accept (even if ambiguous/missing zip)
    #    B) fuzzy match only if very high AND not ambiguous
    ambiguous = df["qa_reasons"].astype("string").str.contains("common_or_ambiguous_name", na=False)
    borderline = df["qa_reasons"].astype("string").str.contains("borderline_score", na=False)

    auto_accept_exact = df["is_exact"] & (df["score_i"] >= int(args.auto_accept_exact_score))

    auto_accept_fuzzy = (
        df["is_fuzzy2"]
        & (df["score_i"] >= int(args.auto_accept_fuzzy_min_score))
        & (~ambiguous)
        & (~borderline)
    )

    clean_mask = auto_accept_exact | auto_accept_fuzzy

    clean_df = df[clean_mask].drop(columns=["score_i", "is_exact", "is_fuzzy2"]).reset_index(drop=True)
    queue_df = df[~clean_mask].drop(columns=["score_i", "is_exact", "is_fuzzy2"]).reset_index(drop=True)

    edges_clean = int(len(clean_df))
    edges_queue = int(len(queue_df))

    # Write clean edges dataset
    out_clean = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/03_entities",
        dataset_key="edges_person_sanctions_clean",
        df=clean_df,
        meta={
            "producer_step": "step6a_quality_gate_person_sanctions",
            "generated_at_utc": generated_at,
            "inputs": [str(edges_loc.dir_ref)],
            "counts": {"edges_in": edges_in, "edges_clean": edges_clean, "edges_routed_to_queue": edges_queue},
            "params": {
                "auto_accept_exact_score": int(args.auto_accept_exact_score),
                "auto_accept_fuzzy_min_score": int(args.auto_accept_fuzzy_min_score),
                "borderline_min": int(args.borderline_min),
                "borderline_max": int(args.borderline_max),
            },
        },
        overwrite=bool(args.overwrite),
    )

    out_queue = None
    if args.emit_queue:
        q = queue_df[[
            "person_id", "person_name", "state", "zip",
            "watchlist_id", "source_dataset_wl", "program", "country",
            "remarks", "match_method", "match_score", "qa_reasons"
        ]].copy()
        q["queue_status"] = "needs_review"
        q["generated_at_utc"] = generated_at

        out_queue = write_parquet_dataset(
            project_root=project_root,
            tier_dir="data/05_underwriter_queue",
            dataset_key="underwriter_review_queue_person",
            df=q,
            meta={
                "producer_step": "step6a_quality_gate_person_sanctions",
                "generated_at_utc": generated_at,
                "inputs": [str(edges_loc.dir_ref)],
                "counts": {"queue_rows": int(len(q))},
            },
            overwrite=bool(args.overwrite),
        )

    print("[OK] Step6A complete (quality gate)")
    print("edges_in:", edges_in)
    print("edges_clean:", edges_clean)
    print("edges_dropped:", edges_queue)
    print("output_dir_clean:", out_clean)
    if out_queue:
        print("output_dir_queue:", out_queue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
