from __future__ import annotations

import argparse
import re
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


def normalize_name(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--prefer-real-names", action="store_true", help="If set, replace existing name with real_name")
    args = ap.parse_args()

    project_root = infer_project_root()

    entities, _ = load_dataset_df(project_root, "entities_person", max_parts=None, max_rows=None)
    realnames, _ = load_dataset_df(project_root, "person_real_names", max_parts=None, max_rows=None)

    # Left join to keep LendingClub intact
    realnames = realnames[["person_id", "real_name"]].copy()
    out = entities.merge(realnames, on="person_id", how="left")

    # Strategy:
    # - If prefer-real-names: overwrite name with real_name when available
    # - Else: only fill missing/empty names
    if args.prefer_real_names:
        out["name"] = out["real_name"].fillna(out["name"])
    else:
        name_str = out["name"].astype("string").fillna("")
        mask_empty = name_str.str.len().eq(0) | name_str.isna()
        out.loc[mask_empty, "name"] = out.loc[mask_empty, "real_name"]

    out["name"] = out["name"].astype("string")
    out["name_norm"] = out["name"].map(normalize_name)

    # Keep schema tidy
    out = out.drop(columns=["real_name"], errors="ignore")

    out_dir = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/03_entities",
        dataset_key="entities_person",
        df=out,
        meta={
            "producer_step": "step7b_enrich_entities_person_names",
            "generated_at_utc": utc_now_iso(),
            "inputs": ["data/03_entities/entities_person", "data/03_entities/person_real_names"],
            "counts": {"entities_person_rows": int(len(out))},
            "strategy": "prefer_real_names" if args.prefer_real_names else "fill_empty_only",
        },
        overwrite=bool(args.overwrite),
    )

    print("[OK] Step7B-2 complete (entities_person enriched)")
    print("output_dir:", out_dir)
    print("rows:", len(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
