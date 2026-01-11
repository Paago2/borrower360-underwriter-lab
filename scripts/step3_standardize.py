from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
import yaml

# Reuse your existing registry loader
from app.services.dataset_registry import load_registry, Dataset


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_project_root() -> Path:
    """
    Prefer env PROJECT_ROOT if present; otherwise assume repo root is parent of scripts/.
    """
    import os
    pr = os.environ.get("PROJECT_ROOT")
    if pr:
        return Path(pr).resolve()
    return Path(__file__).resolve().parents[1]


def load_contracts(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "contracts": {}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"version": 1, "contracts": {}}


def find_contract_for_dataset(contracts: Dict[str, Any], dataset_key: str) -> Optional[Dict[str, Any]]:
    root = contracts.get("contracts", {})
    spec = root.get(dataset_key)
    if isinstance(spec, dict):
        return spec
    return None


def list_parquet_files(curated_dir: Path) -> List[Path]:
    # Find any parquet files under curated dir, ignore meta markers
    files = sorted([
        p for p in curated_dir.rglob("*.parquet")
        if p.is_file() and not p.name.startswith("_")
    ])
    return files

def parse_yyyymm_to_date(series: pd.Series) -> pd.Series:
    """
    Convert YYYYMM (string/int) to datetime64[ns] at first day of month.
    """
    s = series.astype("string")
    s = s.str.replace(r"[^0-9]", "", regex=True)
    s = s.where(s.str.len() == 6, other=pd.NA)
    return pd.to_datetime(s, format="%Y%m", errors="coerce")


def parse_mmyyyy_to_date(s: pd.Series) -> pd.Series:
    """
    Convert MMYYYY (string/int) to datetime64[ns] at first day of month.

    Example: "042025" -> 2025-04-01
    """
    s = s.astype("string").str.strip()
    s = s.str.replace(r"\D+", "", regex=True)  # keep digits only
    s = s.where(s.str.len() == 6)

    mm = pd.to_numeric(s.str[:2], errors="coerce")
    yyyy = pd.to_numeric(s.str[2:], errors="coerce")

    valid = mm.between(1, 12) & yyyy.between(1900, 2100)

    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    y = yyyy[valid].astype("Int64").astype("string")
    m = mm[valid].astype("Int64").astype("string").str.zfill(2)
    out.loc[valid] = pd.to_datetime(y + "-" + m + "-01", errors="coerce")
    return out


def apply_contract_transforms(df: pd.DataFrame, contract: Dict[str, Any]) -> pd.DataFrame:
    # 1) Drop columns that are all-null
    drop_all_null = contract.get("drop_if_all_null") or []
    if isinstance(drop_all_null, list) and drop_all_null:
        to_drop: List[str] = []
        for c in drop_all_null:
            if c in df.columns and df[c].isna().all():
                to_drop.append(c)
        if to_drop:
            df = df.drop(columns=to_drop)

    # 2) Rename
    rename_map = contract.get("rename") or {}
    if isinstance(rename_map, dict) and rename_map:
        valid = {k: v for k, v in rename_map.items() if k in df.columns}
        if valid:
            df = df.rename(columns=valid)

    # 3) Keep-as-string (force dtype)
    keep_as_string = contract.get("keep_as_string") or []
    if isinstance(keep_as_string, list):
        for c in keep_as_string:
            if c in df.columns:
                df[c] = df[c].astype("string")

    # 4a) Parse YYYYMM columns + create *_date
    parse_cols_yyyymm = contract.get("parse_yyyymm") or []
    if isinstance(parse_cols_yyyymm, list):
        for c in parse_cols_yyyymm:
            if c in df.columns:
                df[c] = df[c].astype("string")
                df[f"{c}_date"] = parse_yyyymm_to_date(df[c])

    # 4b) Parse MMYYYY columns + create *_date
    parse_cols_mmyyyy = contract.get("parse_mmyyyy") or []
    if isinstance(parse_cols_mmyyyy, list):
        for c in parse_cols_mmyyyy:
            if c in df.columns:
                df[c] = df[c].astype("string")
                df[f"{c}_date"] = parse_mmyyyy_to_date(df[c])

    return df


def standardize_one(ds, contract: Optional[Dict[str, Any]], project_root: Path) -> Dict[str, Any]:
    curated_dir = project_root / "data" / "01_curated" / ds.key
    out_dir = project_root / "data" / "02_standardized" / ds.key

    if not curated_dir.exists():
        return {"key": ds.key, "status": "skipped", "reason": f"Missing curated dir: {curated_dir}"}

    parquet_files = list_parquet_files(curated_dir)
    if not parquet_files:
        # best practice: this is SKIP, not ERROR (dataset may be text/image/shapefile, etc.)
        return {"key": ds.key, "status": "skipped", "reason": f"No parquet files found in {curated_dir}"}

    # Prepare output dir fresh (overwrite mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    final_cols = None
    parts_written = 0

    # Process shard-by-shard (scales to huge datasets)
    for idx, p in enumerate(parquet_files):
        df = pd.read_parquet(p)

        before_cols = df.columns.tolist()

        if isinstance(contract, dict):
            df = apply_contract_transforms(df, contract)

        after_cols = df.columns.tolist()

        # Write standardized shard with consistent part naming
        out_part = out_dir / f"part-{idx:05d}.parquet"
        df.to_parquet(out_part, index=False)

        parts_written += 1
        total_rows += int(df.shape[0])
        final_cols = int(df.shape[1])

        # Optional: write a tiny per-shard meta (handy for debugging)
        # (comment out if you want fewer files)
        shard_meta = {
            "dataset_key": ds.key,
            "input_file": str(p),
            "output_file": str(out_part),
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "before_cols_sample": before_cols[:20],
            "after_cols_sample": after_cols[:20],
            "contract_applied": bool(isinstance(contract, dict)),
        }
        (out_dir / f"_meta_part_{idx:05d}.json").write_text(json.dumps(shard_meta, indent=2), encoding="utf-8")

    # Write dataset-level meta + success marker
    meta = {
        "key": ds.key,
        "input_curated_dir": str(curated_dir),
        "parquet_files_read": [str(p) for p in parquet_files[:50]],  # cap for readability
        "parquet_files_read_count": len(parquet_files),
        "parts_written": parts_written,
        "rows_written": total_rows,
        "cols_written": final_cols,
        "contract_applied": bool(isinstance(contract, dict)),
        "contract_keys": list(contract.keys()) if isinstance(contract, dict) else None,
        "output_dir": str(out_dir),
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "_SUCCESS").write_text("", encoding="utf-8")

    return {
        "key": ds.key,
        "status": "ok",
        "output_dir": str(out_dir),
        "rows_written": total_rows,
        "cols_written": final_cols,
        "parts_written": parts_written,
        "contract_applied": bool(isinstance(contract, dict)),
        # generic flags kept for your earlier checks:
        "has_loan_id": False if not hasattr(df, "columns") else ("loan_id" in df.columns),
        "has_property_state": False if not hasattr(df, "columns") else ("property_state" in df.columns),
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None, help="Dataset key (optional). If omitted, run all.")
    args = ap.parse_args()

    project_root = infer_project_root()
    manifest_path = project_root / "configs" / "datasets.yaml"
    contracts_path = project_root / "configs" / "contracts.yaml"

    registry = load_registry(manifest_path=manifest_path, project_root=project_root)
    contracts = load_contracts(contracts_path)

    keys = [args.dataset] if args.dataset else list(registry.keys())

    results = []
    for key in keys:
        if key not in registry:
            results.append({"key": key, "status": "error", "reason": "Unknown dataset key"})
            continue

        ds = registry[key]
        contract = find_contract_for_dataset(contracts, key)

        # Standardize anything that has curated parquet parts.
        res = standardize_one(ds, contract, project_root)
        results.append(res)

        if res.get("status") == "ok":
            print(f"[OK] {key} -> {res['output_dir']}")
        else:
            print(f"[ERR] {key} -> {res.get('reason')}")

    reports_dir = project_root / "reports" / "standardization"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "generated_at_utc": utc_now_iso(),
        "project_root": str(project_root),
        "manifest_path": str(manifest_path),
        "contracts_path": str(contracts_path),
        "results": results,
    }
    summary_path = reports_dir / "_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote summary: {summary_path}")

    any_errors = any(r.get("status") != "ok" for r in results)
    return 1 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
