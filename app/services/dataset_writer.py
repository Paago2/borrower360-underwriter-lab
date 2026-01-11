# app/services/dataset_writer.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_parquet_dataset(
    *,
    project_root: Path,
    tier_dir: str,          # e.g. "data/04_borrower360"
    dataset_key: str,       # e.g. "borrower360_person_profile"
    df: pd.DataFrame,
    meta: Optional[Dict[str, Any]] = None,
    part_name: str = "part-00001.parquet",
    overwrite: bool = True,
) -> Path:
    """
    Enterprise dataset writer:
      data/<tier>/<dataset_key>/part-00001.parquet + _meta.json + _SUCCESS

    - Keeps loader simple (directory-based datasets only)
    - Works on local FS (and can be extended later for S3)
    """
    out_dir = project_root / tier_dir / dataset_key
    ensure_dir(out_dir)

    part_path = out_dir / part_name
    meta_path = out_dir / "_meta.json"
    success_path = out_dir / "_SUCCESS"

    if overwrite:
        if part_path.exists():
            part_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        if success_path.exists():
            success_path.unlink()

    df.to_parquet(part_path, index=False)

    meta_out: Dict[str, Any] = {
        "dataset_key": dataset_key,
        "tier_dir": tier_dir,
        "generated_at_utc": utc_now_iso(),
        "rows": int(len(df)),
        "cols": df.columns.tolist(),
        "part": part_name,
    }
    if meta:
        meta_out.update(meta)

    meta_path.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    success_path.write_text("ok\n", encoding="utf-8")

    return out_dir
