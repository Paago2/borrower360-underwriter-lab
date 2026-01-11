from __future__ import annotations

import argparse
import json
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


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == "bool":
        return s
    return s.fillna(False).astype(bool)


def _as_str(s: pd.Series) -> pd.Series:
    return s.fillna("").astype("string")


def _num(s: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    return out.fillna(default)


def _has_reason(df: pd.DataFrame, reason: str, col: str = "qa_reasons") -> pd.Series:
    """
    Return True if qa_reasons contains token `reason` in comma-separated list.
    Uses non-capturing groups to avoid pandas "match groups" warning.
    """
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = _as_str(df[col])
    # token boundary: start or comma + optional spaces, then reason, then optional spaces + comma or end
    pat = rf"(?:^|,)\s*{reason}\s*(?:,|$)"
    return s.str.contains(pat, regex=True)


def _commonness_bucket(df: pd.DataFrame) -> pd.Series:
    if "name_commonness_bucket" in df.columns:
        return _as_str(df["name_commonness_bucket"])
    # fallback based on token counts if bucket missing
    name = _as_str(df.get("person_name", pd.Series("", index=df.index, dtype="string")))
    tokens = name.str.split().map(lambda x: len(x) if isinstance(x, list) else 0)
    out = pd.Series("unknown", index=df.index, dtype="string")
    out[tokens <= 1] = "single_token"
    out[tokens == 2] = "two_token"
    out[tokens >= 3] = "three_plus"
    return out


def _missing_location(df: pd.DataFrame) -> pd.Series:
    state = _as_str(df.get("state", pd.Series("", index=df.index, dtype="string"))).str.strip()
    z = _as_str(df.get("zip", pd.Series("", index=df.index, dtype="string"))).str.strip()
    return (state.str.len() == 0) | (z.str.len() == 0)


def compute_priority(df: pd.DataFrame) -> pd.DataFrame:
    """
    Priority is higher => should be reviewed sooner (goes to top of queue).
    It is intentionally explainable and decomposed into additive components.
    """
    out = df.copy()

    match_score = _num(out.get("match_score", pd.Series(0, index=out.index)), default=0.0)
    is_fuzzy = _as_bool(out.get("is_fuzzy", pd.Series(False, index=out.index)))

    common_bucket = _commonness_bucket(out)
    missing_loc = _missing_location(out)

    # Reasons (from Step6A gate)
    r_common = _has_reason(out, "common_or_ambiguous_name")
    r_missing = _has_reason(out, "missing_state_or_zip")
    r_fuzzy = _has_reason(out, "fuzzy_match")
    r_border = _has_reason(out, "borderline_score")

    # 1) Base
    priority_base = pd.Series(0.0, index=out.index)

    # 2) Match score contribution (bigger score => more urgent)
    # Keep it linear and easy to explain.
    priority_match_score = match_score.astype(float)

    # 3) Fuzzy bonus (fuzzy matches need human review more often)
    priority_fuzzy = is_fuzzy.map(lambda x: 8.0 if x else 0.0)

    # 4) Common name bonus (common names create risk of false positives)
    # Higher bonus => earlier review to clear or confirm quickly.
    common_bonus_map = {
        "single_token": 12.0,   # very ambiguous
        "two_token": 6.0,       # still common
        "three_plus": 2.0,
        "unknown": 4.0,
    }
    priority_common_name = common_bucket.map(lambda b: common_bonus_map.get(str(b), 4.0))

    # 5) Missing location bonus (lack of corroborating evidence => needs review)
    priority_missing_location = missing_loc.map(lambda x: 6.0 if x else 0.0)

    # 6) Borderline bonus (90–98 region; often the "real work")
    # Note: if current strategy is “high fuzzy 100s first”, keep this smaller.
    # If borderline-preference first, increase this value.
    priority_borderline = r_border.map(lambda x: 5.0 if x else 0.0)

    # Optional: small boosts if reasons present (keeps aligned with QA gate rationale)
    priority_common_name = priority_common_name + r_common.map(lambda x: 2.0 if x else 0.0)
    priority_missing_location = priority_missing_location + r_missing.map(lambda x: 2.0 if x else 0.0)
    priority_fuzzy = priority_fuzzy + r_fuzzy.map(lambda x: 2.0 if x else 0.0)

    # Total
    priority_total = (
        priority_base
        + priority_match_score
        + priority_fuzzy
        + priority_common_name
        + priority_missing_location
        + priority_borderline
    )

    # Pack components as a JSON string for audit/debug (easy to store in parquet)
    components = []
    for i in range(len(out)):
        components.append(
            json.dumps(
                {
                    "base": float(priority_base.iat[i]),
                    "match_score": float(priority_match_score.iat[i]),
                    "fuzzy": float(priority_fuzzy.iat[i]),
                    "common_name": float(priority_common_name.iat[i]),
                    "missing_location": float(priority_missing_location.iat[i]),
                    "borderline": float(priority_borderline.iat[i]),
                    "total": float(priority_total.iat[i]),
                },
                separators=(",", ":"),
            )
        )

    out["priority_base"] = priority_base.astype(float)
    out["priority_match_score"] = priority_match_score.astype(float)
    out["priority_fuzzy"] = priority_fuzzy.astype(float)
    out["priority_common_name"] = priority_common_name.astype(float)
    out["priority_missing_location"] = priority_missing_location.astype(float)
    out["priority_borderline"] = priority_borderline.astype(float)
    out["priority_score"] = priority_total.astype(float)  # keep existing column name
    out["priority_components"] = pd.Series(components, index=out.index, dtype="string")

    return out


def apply_dedupe(df: pd.DataFrame, mode: str, per_person_cap: int | None) -> pd.DataFrame:
    """
    Dedupe to control surge:
      - none: keep everything
      - person_watchlist: dedupe exact pair (person_id, watchlist_id)
      - person: keep only best row per person_id (highest priority_score)
    """
    if mode == "none":
        out = df
    elif mode == "person_watchlist":
        keys = [c for c in ["person_id", "watchlist_id"] if c in df.columns]
        if len(keys) == 2:
            out = df.drop_duplicates(keys, keep="first")
        else:
            out = df
    elif mode == "person":
        if "person_id" in df.columns:
            out = df.sort_values(["person_id", "priority_score"], ascending=[True, False])
            out = out.drop_duplicates(["person_id"], keep="first")
        else:
            out = df
    else:
        raise ValueError("Invalid --dedupe. Use one of: none, person, person_watchlist")

    # Optional: cap watchlists per person (top K per person)
    if per_person_cap is not None and per_person_cap > 0 and "person_id" in out.columns:
        out = out.sort_values(["person_id", "priority_score"], ascending=[True, False])
        out = out.groupby("person_id", as_index=False, group_keys=False).head(per_person_cap)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output datasets")
    ap.add_argument("--top-n", type=int, default=500, help="Emit top-N dataset cut (ranked queue)")
    ap.add_argument("--max-rows", type=int, default=None, help="Cap input rows read (surge protection)")
    ap.add_argument("--min-priority", type=float, default=None, help="Drop rows below this priority_score")
    ap.add_argument(
        "--dedupe",
        type=str,
        default="person_watchlist",
        choices=["none", "person", "person_watchlist"],
        help="Dedupe mode to control queue volume",
    )
    ap.add_argument(
        "--per-person-cap",
        type=int,
        default=None,
        help="Keep only top K rows per person_id (after dedupe). Example: 3",
    )
    args = ap.parse_args()

    project_root = infer_project_root()
    ranked_at = utc_now_iso()

    # Load queue
    queue_df, queue_loc = load_dataset_df(
        project_root, "underwriter_review_queue_person", max_parts=None, max_rows=args.max_rows
    )

    if len(queue_df) == 0:
        # Still write empty datasets (enterprise behavior)
        empty = queue_df.copy()
        empty["priority_score"] = pd.Series([], dtype="float")
        empty["ranked_at_utc"] = pd.Series([], dtype="string")
        empty["priority_rank"] = pd.Series([], dtype="int64")
        empty["priority_components"] = pd.Series([], dtype="string")

        out_ranked = write_parquet_dataset(
            project_root=project_root,
            tier_dir="data/05_underwriter_queue",
            dataset_key="underwriter_review_queue_person_ranked",
            df=empty,
            meta={
                "producer_step": "step8b_rank_underwriter_queue_person",
                "ranked_at_utc": ranked_at,
                "inputs": [str(queue_loc.dir_ref)],
                "counts": {"queue_rows_in": 0, "ranked_rows": 0, "top_rows": 0},
                "surge_controls": {
                    "max_rows": args.max_rows,
                    "min_priority": args.min_priority,
                    "dedupe": args.dedupe,
                    "per_person_cap": args.per_person_cap,
                    "top_n": args.top_n,
                },
            },
            overwrite=bool(args.overwrite),
        )

        out_top = write_parquet_dataset(
            project_root=project_root,
            tier_dir="data/05_underwriter_queue",
            dataset_key="underwriter_review_queue_person_top",
            df=empty,
            meta={
                "producer_step": "step8b_rank_underwriter_queue_person",
                "ranked_at_utc": ranked_at,
                "inputs": [str(queue_loc.dir_ref)],
                "counts": {"rows": 0},
                "surge_controls": {"top_n": args.top_n},
            },
            overwrite=bool(args.overwrite),
        )

        print("[OK] Step8B complete (ranked underwriter queue)")
        print("input_queue_dir:", queue_loc.dir_uri)
        print("queue rows: 0")
        print("outputs:")
        print(" - ranked:", out_ranked)
        print(" - top:", out_top)
        print("top_n:", args.top_n)
        return 0

    # Compute explainable priority
    scored = compute_priority(queue_df)

    # Dedupe + caps
    scored = scored.sort_values("priority_score", ascending=False)
    before_dedupe = int(len(scored))
    scored = apply_dedupe(scored, mode=args.dedupe, per_person_cap=args.per_person_cap)
    after_dedupe = int(len(scored))

    # Optional filter by min priority (surge control)
    if args.min_priority is not None:
        scored = scored[scored["priority_score"] >= float(args.min_priority)].copy()

    # Rank
    scored = scored.sort_values("priority_score", ascending=False).reset_index(drop=True)
    scored["ranked_at_utc"] = ranked_at
    scored["priority_rank"] = pd.RangeIndex(start=1, stop=len(scored) + 1, step=1)

    # Top cut
    top_n = int(args.top_n) if args.top_n is not None else 0
    top = scored.head(top_n).copy() if top_n > 0 else scored.head(0).copy()

    # Write datasets
    out_ranked = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/05_underwriter_queue",
        dataset_key="underwriter_review_queue_person_ranked",
        df=scored,
        meta={
            "producer_step": "step8b_rank_underwriter_queue_person",
            "ranked_at_utc": ranked_at,
            "inputs": [str(queue_loc.dir_ref)],
            "counts": {
                "queue_rows_in": int(len(queue_df)),
                "scored_rows": before_dedupe,
                "ranked_rows": int(len(scored)),
                "dedupe_before": before_dedupe,
                "dedupe_after": after_dedupe,
                "top_rows": int(len(top)),
            },
            "surge_controls": {
                "max_rows": args.max_rows,
                "min_priority": args.min_priority,
                "dedupe": args.dedupe,
                "per_person_cap": args.per_person_cap,
                "top_n": top_n,
            },
        },
        overwrite=bool(args.overwrite),
    )

    out_top = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/05_underwriter_queue",
        dataset_key="underwriter_review_queue_person_top",
        df=top,
        meta={
            "producer_step": "step8b_rank_underwriter_queue_person",
            "ranked_at_utc": ranked_at,
            "inputs": [str(queue_loc.dir_ref)],
            "counts": {"rows": int(len(top))},
            "surge_controls": {"top_n": top_n},
        },
        overwrite=bool(args.overwrite),
    )

    print("[OK] Step8B complete (ranked underwriter queue)")
    print("input_queue_dir:", queue_loc.dir_uri)
    print("queue rows:", int(len(queue_df)))
    print("dedupe:", args.dedupe, "| per_person_cap:", args.per_person_cap)
    print("rows ranked:", int(len(scored)), "| top_n:", top_n)
    print("outputs:")
    print(" - ranked:", out_ranked)
    print(" - top:", out_top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
