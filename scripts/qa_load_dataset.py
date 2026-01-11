from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from app.services.dataset_loader import load_dataset_df


def infer_project_root() -> Path:
    import os
    pr = os.environ.get("PROJECT_ROOT")
    if pr:
        return Path(pr).resolve()
    return Path(__file__).resolve().parents[1]


def _safe_loc_str(loc: Any) -> str:
    """
    DatasetLocation may vary across versions. This avoids AttributeError
    by checking common attribute names and falling back safely.
    """
    # Try common path-ish attributes
    for attr in (
        "dir_path",
        "dir",
        "path",
        "dataset_dir",
        "resolved_path",
        "base_dir",
        "location",
    ):
        v = getattr(loc, attr, None)
        if v:
            return str(v)

    # Fallback: show a small set of attributes
    keys = []
    for k in ("tier", "storage"):
        if hasattr(loc, k):
            keys.append(f"{k}={getattr(loc, k)}")
    return "unknown" if not keys else " | ".join(keys)


def _print_basic_checks(df: pd.DataFrame, key_cols: Optional[list[str]] = None) -> None:
    print("\n--- QA CHECKS ---")
    print("rows:", len(df))
    print("cols:", len(df.columns))

    # Null % for selected key cols (if present)
    if key_cols:
        print("\nnull % per key col:")
        for c in key_cols:
            if c in df.columns:
                pct = float(df[c].isna().mean() * 100.0)
                print(f"  {c}: {pct:.3f}%")
            else:
                print(f"  {c}: (missing)")

    # Duplicates for obvious ID-like columns
    for id_col in ("person_id", "watchlist_id", "loan_id", "id", "uid"):
        if id_col in df.columns:
            dup = int(df.duplicated(id_col).sum())
            nunq = int(df[id_col].nunique(dropna=False))
            print(f"\nuniqueness check for {id_col}:")
            print(f"  unique {id_col}: {nunq} / rows: {len(df)}")
            print(f"  duplicate {id_col} rows: {dup}")

    # Empty-string checks for common text keys
    for txt_col in ("name_norm", "name", "state", "zip", "country"):
        if txt_col in df.columns:
            s = df[txt_col].astype("string")
            empty = int((s.fillna("").str.strip().str.len() == 0).sum())
            print(f"empty '{txt_col}' rows:", empty)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Dataset key")
    ap.add_argument("--max-parts", type=int, default=1, help="How many parquet shards to read")
    ap.add_argument("--max-rows", type=int, default=50, help="How many rows to load (sample)")
    ap.add_argument("--columns", nargs="*", default=None, help="Optional list of columns to load")
    ap.add_argument("--show-loc-attrs", action="store_true", help="Print DatasetLocation public attrs (debug)")
    args = ap.parse_args()

    project_root = infer_project_root()

    df, loc = load_dataset_df(
        project_root,
        args.dataset,
        columns=args.columns,
        max_parts=args.max_parts,
        max_rows=args.max_rows,
    )

    tier = getattr(loc, "tier", "unknown")
    storage = getattr(loc, "storage", "unknown")
    loc_str = _safe_loc_str(loc)

    print(f"[OK] Loaded '{args.dataset}' from tier={tier} storage={storage} location={loc_str}")

    if args.show_loc_attrs:
        print("\nDatasetLocation attrs (public):")
        print([a for a in dir(loc) if not a.startswith("_")])

    # shard counting works for BOTH parquet + jsonl
    if getattr(loc, "storage", "parquet") == "parquet":
        total_shards = len(getattr(loc, "parquet_files", []) or [])
        shards_read = min(total_shards, args.max_parts)
    else:
        total_shards = 1
        shards_read = 1

    print(f"      shards_read={shards_read} total_shards={total_shards}")
    print("shape:", df.shape)
    print("cols:", df.columns.tolist())

    # QA checks (useful defaults)
    default_key_cols = ["person_id", "watchlist_id", "loan_id", "uid", "id", "name", "name_norm", "source_dataset"]
    _print_basic_checks(df, key_cols=default_key_cols)

    # --- SAFE PREVIEW (avoid dumping binary image bytes) ---
    preview = df.head(min(len(df), 5)).copy()

    # If synthdog-like: show lengths instead of raw bytes
    if "image" in preview.columns:
        def _image_summary(x):
            try:
                b = x.get("bytes", b"") if isinstance(x, dict) else b""
                return {"bytes_len": len(b)}
            except Exception:
                return {"bytes_len": None}

        preview["image"] = preview["image"].apply(_image_summary)

    # Truncate long text-ish columns
    for c in ("ground_truth", "text", "remarks"):
        if c in preview.columns:
            preview[c] = preview[c].astype("string").fillna("").str.slice(0, 200) + "..."

    print("\n--- PREVIEW (first rows) ---")
    print(preview.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
