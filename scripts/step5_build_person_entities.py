from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.services.dataset_writer import write_parquet_dataset


def infer_project_root() -> Path:
    import os
    pr = os.environ.get("PROJECT_ROOT")
    if pr:
        return Path(pr).resolve()
    return Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_parquet_parts(dir_path: Path, max_parts: int | None = 5) -> pd.DataFrame:
    parts = sorted(dir_path.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts found under: {dir_path}")
    if max_parts is not None:
        parts = parts[:max_parts]
    dfs = [pd.read_parquet(p) for p in parts]
    return pd.concat(dfs, ignore_index=True)


def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-parts", type=int, default=5, help="How many parquet shards to read for entity build")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output dataset folder")
    args = ap.parse_args()

    project_root = infer_project_root()
    std_dir = project_root / "data" / "02_standardized" / "lending_club"

    # Load standardized LendingClub (folder dataset with part-*.parquet)
    df = load_parquet_parts(std_dir, max_parts=args.max_parts)

    # LendingClub datasets vary by year/source.
    # Try common identifiers:
    id_col = pick_first_existing(df, ["id", "member_id", "loan_id"])
    if id_col is None:
        # fallback stable row id
        df["_row_id"] = pd.RangeIndex(start=1, stop=len(df) + 1, step=1)
        id_col = "_row_id"

    # LendingClub usually does NOT include full legal names (PII). So for demo:
    # - we build a "person-like entity" from fields that exist
    # - we normalize names for matching
    name_like_col = pick_first_existing(df, ["applicant_name", "name", "borrower_name", "emp_title", "title"])
    if name_like_col is None:
        # last resort: synth name based on id
        df["__name_like"] = "BORROWER " + df[id_col].astype("string")
        name_like_col = "__name_like"

    state_col = pick_first_existing(df, ["addr_state", "state", "property_state"])
    zip_col = pick_first_existing(df, ["zip_code", "zip", "addr_zip"])
    dob_col = pick_first_existing(df, ["dob", "birth_date"])

    ent = pd.DataFrame({
        "person_id": df[id_col].astype("string"),
        "name": df[name_like_col].astype("string"),
        "state": df[state_col].astype("string") if state_col else pd.Series("", index=df.index, dtype="string"),
        "zip": df[zip_col].astype("string") if zip_col else pd.Series("", index=df.index, dtype="string"),
        "birth_date": df[dob_col].astype("string") if dob_col else pd.Series("", index=df.index, dtype="string"),
        "source_dataset": "lending_club",
    })

    ent["name_norm"] = ent["name"].map(normalize_name)

    # Drop empty and dedupe
    ent = ent[ent["name_norm"].astype("string").str.len() > 0].copy()
    ent = ent.drop_duplicates("person_id").reset_index(drop=True)

    
    # data/03_entities/entities_person/{_meta.json,_SUCCESS,part-00001.parquet}
    out_dataset_dir = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/03_entities",
        dataset_key="entities_person",
        df=ent,
        meta={
            "producer_step": "step5_build_person_entities",
            "generated_at_utc": utc_now_iso(),
            "inputs": ["data/02_standardized/lending_club"],
            "params": {"max_parts": int(args.max_parts)},
            "cols": ent.columns.tolist(),
            "counts": {"rows_in": int(len(df)), "entities_person_rows": int(len(ent))},
            "selected_columns": {
                "id_col": id_col,
                "name_like_col": name_like_col,
                "state_col": state_col or "",
                "zip_col": zip_col or "",
                "dob_col": dob_col or "",
            },
        },
        overwrite=bool(args.overwrite),
    )

    # Keep step report  for auditing
    report = {
        "generated_at_utc": utc_now_iso(),
        "source_dataset": "lending_club",
        "max_parts": int(args.max_parts),
        "rows_in": int(len(df)),
        "entities_person_rows": int(len(ent)),
        "output_dir": str(out_dataset_dir),
        "id_col": id_col,
        "name_like_col": name_like_col,
        "state_col": state_col,
        "zip_col": zip_col,
        "dob_col": dob_col,
    }
    (project_root / "data" / "03_entities" / "_STEP5A_REPORT.json").write_text(
        pd.Series(report).to_json(indent=2),
        encoding="utf-8",
    )

    print("[OK] Step5A complete (enterprise dataset written)")
    print("entities_person rows:", len(ent))
    print("output_dir:", out_dataset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



if __name__ == "__main__":
    raise SystemExit(main())
