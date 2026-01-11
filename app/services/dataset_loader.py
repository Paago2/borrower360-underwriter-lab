# app/services/dataset_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from app.services.storage import Storage, get_storage_settings


@dataclass(frozen=True)
class DatasetLocation:
    key: str
    tier: str          # "standardized" | "curated" | "entities" | "borrower360" | "underwriter_queue"
    storage: str       # "parquet" or "jsonl"
    dir_ref: str       # e.g. "data/02_standardized/funsd"
    dir_uri: str       # local absolute path or s3://bucket/prefix/...
    parquet_files: List[str]          # list of relative file refs
    index_jsonl: Optional[str] = None # relative ref


def _is_parquet_data_file_name(name: str) -> bool:
    if not name.lower().endswith(".parquet"):
        return False
    base = Path(name).name.lower()
    if base.startswith("_meta") or base.startswith("_"):
        return False
    return True


def _list_parquet_files(storage: Storage, dir_ref: str) -> List[str]:
    """
    Prefer part-*.parquet, else any *.parquet.
    Works for local and s3.
    """
    files = storage.list_files(dir_ref, suffix=".parquet")

    part = [f for f in files if Path(f).name.startswith("part-") and _is_parquet_data_file_name(f)]
    if part:
        return part

    allpq = [f for f in files if _is_parquet_data_file_name(f)]
    return allpq


def _find_index_jsonl(storage: Storage, dir_ref: str) -> Optional[str]:
    p = f"{dir_ref.rstrip('/')}/index.jsonl"
    return p if storage.exists_file(p) else None


def _resolve_from_dir(storage: Storage, dataset_key: str, tier_name: str, dir_ref: str) -> Optional[DatasetLocation]:
    """
    Resolve dataset in a single tier directory.
    Priority:
      1) parquet shards
      2) index.jsonl
    """
    pq = _list_parquet_files(storage, dir_ref)
    if pq:
        return DatasetLocation(
            key=dataset_key,
            tier=tier_name,
            storage="parquet",
            dir_ref=dir_ref,
            dir_uri=storage.uri(dir_ref),
            parquet_files=pq,
            index_jsonl=None,
        )

    j = _find_index_jsonl(storage, dir_ref)
    if j:
        return DatasetLocation(
            key=dataset_key,
            tier=tier_name,
            storage="jsonl",
            dir_ref=dir_ref,
            dir_uri=storage.uri(dir_ref),
            parquet_files=[],
            index_jsonl=j,
        )

    return None


def resolve_dataset_location(project_root: Path, dataset_key: str) -> DatasetLocation:
    """
    Enterprise tier search order (most "derived" first):
      1) borrower360 (data/04_borrower360)
      2) entities (data/03_entities)
      3) underwriter queue (data/05_underwriter_queue)
      4) standardized (data/02_standardized)
      5) curated (data/01_curated)

    Rationale:
      - borrower360 / entities are produced artifacts you usually want to read by key
      - underwriter_queue is also a produced artifact
      - standardized/curated are upstream raw/clean tiers
    """
    storage = Storage(get_storage_settings(project_root))

    tier_dirs = [
        ("borrower360",        f"data/04_borrower360/{dataset_key}"),
        ("entities",           f"data/03_entities/{dataset_key}"),
        ("underwriter_queue",  f"data/05_underwriter_queue/{dataset_key}"),
        ("standardized",       f"data/02_standardized/{dataset_key}"),
        ("curated",            f"data/01_curated/{dataset_key}"),
        ("monitoring",         f"data/06_monitoring/{dataset_key}"),
    ]

    for tier_name, dir_ref in tier_dirs:
        loc = _resolve_from_dir(storage, dataset_key, tier_name, dir_ref)
        if loc is not None:
            return loc

    # Build friendly error message listing checked locations
    checked = "\n".join([f"  - {storage.uri(d)}" for _, d in tier_dirs])
    raise FileNotFoundError(
        f"No parquet shards or index.jsonl found for dataset '{dataset_key}' in:\n{checked}"
    )


def _read_parquet_parts_as_df(
    storage: Storage,
    parquet_files: Sequence[str],
    *,
    columns: Optional[Sequence[str]] = None,
    max_parts: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    files = list(parquet_files)
    if max_parts is not None:
        files = files[: max_parts]

    dfs: List[pd.DataFrame] = []
    rows_so_far = 0

    for rel in files:
        local_path = storage.as_local_path(rel)  # local or downloaded cache
        df_part = pd.read_parquet(local_path, columns=list(columns) if columns else None)
        dfs.append(df_part)
        rows_so_far += len(df_part)

        if max_rows is not None and rows_so_far >= max_rows:
            break

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).copy()

    return df


def _read_index_jsonl_as_df(
    storage: Storage,
    index_jsonl: str,
    *,
    columns: Optional[Sequence[str]] = None,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    import json

    records: List[Dict[str, Any]] = []
    limit = max_rows if max_rows is not None else None

    p = storage.as_local_path(index_jsonl)  # local or downloaded cache
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

            if limit is not None and (i + 1) >= limit:
                break

    df = pd.DataFrame.from_records(records)
    if columns:
        keep = [c for c in columns if c in df.columns]
        df = df[keep].copy()

    return df


def load_dataset_df(
    project_root: Path,
    dataset_key: str,
    *,
    columns: Optional[Sequence[str]] = None,
    max_parts: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, DatasetLocation]:
    """
    Unified loader across tiers.
    Supports:
      - parquet datasets
      - document datasets with index.jsonl
    """
    storage = Storage(get_storage_settings(project_root))
    loc = resolve_dataset_location(project_root, dataset_key)

    if loc.storage == "parquet":
        df = _read_parquet_parts_as_df(
            storage,
            loc.parquet_files,
            columns=columns,
            max_parts=max_parts,
            max_rows=max_rows,
        )
        return df, loc

    assert loc.index_jsonl is not None
    df = _read_index_jsonl_as_df(
        storage,
        loc.index_jsonl,
        columns=columns,
        max_rows=max_rows,
    )
    return df, loc


def load_dataset_meta(project_root: Path, dataset_key: str) -> Dict[str, Any]:
    """
    Load _meta.json from the first tier where it exists (same order as resolve_dataset_location).
    Works for local or s3.
    """
    import json

    storage = Storage(get_storage_settings(project_root))

    candidates = [
        f"data/04_borrower360/{dataset_key}/_meta.json",
        f"data/03_entities/{dataset_key}/_meta.json",
        f"data/05_underwriter_queue/{dataset_key}/_meta.json",
        f"data/02_standardized/{dataset_key}/_meta.json",
        f"data/01_curated/{dataset_key}/_meta.json",
        f"data/06_monitoring/{dataset_key}/_meta.json",
    ]

    for rel in candidates:
        if storage.exists_file(rel):
            txt = storage.read_text(rel, encoding="utf-8")
            return json.loads(txt)

    return {}
