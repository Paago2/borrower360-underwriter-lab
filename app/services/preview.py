from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from app.services.dataset_loader import load_dataset_df, resolve_dataset_location


def _safe_read_text(path: Path, max_chars: int = 2000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


def _safe_read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _to_abs_path(project_root: Path, rel: Optional[str]) -> Optional[Path]:
    if not rel:
        return None
    # index.jsonl stores repo-relative paths like data/00_raw/...
    return project_root / rel


def _sanitize_preview_df(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """
    Make a small preview JSON-safe:
      - limit rows
      - avoid dumping binary bytes (e.g., synthdog image dicts)
      - truncate long text-like fields
    """
    preview = df.head(max_rows).copy()

    # If synthdog-like: "image" column is often dict with {"bytes": b"..."}
    if "image" in preview.columns:
        def _image_summary(x: Any) -> Any:
            try:
                if isinstance(x, dict) and "bytes" in x and isinstance(x["bytes"], (bytes, bytearray)):
                    return {"bytes_len": len(x["bytes"])}
                # sometimes it can be nested/other types
                return {"bytes_len": None}
            except Exception:
                return {"bytes_len": None}

        preview["image"] = preview["image"].apply(_image_summary)

    # Truncate very long text columns commonly seen in OCR datasets
    for col in preview.columns:
        if col in ("ground_truth", "text", "ocr", "qas", "annotation"):
            preview[col] = preview[col].astype("string").str.slice(0, 300).fillna("") + "..."

    # Convert NaN/NaT to None-friendly values when serializing
    preview = preview.where(pd.notna(preview), None)

    return preview


def preview_dataset(
    project_root: Path,
    dataset_key: str,
    *,
    max_rows: int = 50,
    max_parts: int = 1,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Unified preview for both:
      - parquet datasets (tabular) via load_dataset_df
      - jsonl document index datasets via load_dataset_df
    Returns a JSON-serializable dict.
    """
    df, loc = load_dataset_df(
        project_root,
        dataset_key,
        columns=columns,
        max_parts=max_parts,
        max_rows=max_rows,
    )

    # shard counting works for BOTH parquet + jsonl
    if loc.storage == "parquet":
        total_shards = len(loc.parquet_files)
        shards_read = min(total_shards, max_parts)
    else:
        total_shards = 1
        shards_read = 1

    preview_df = _sanitize_preview_df(df, max_rows=min(max_rows, 50))
    rows = preview_df.to_dict(orient="records")

    return {
        "dataset_key": dataset_key,
        "tier": loc.tier,
        "storage": loc.storage,
        "dir": str(loc.dir_path),
        "shards_read": shards_read,
        "total_shards": total_shards,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": df.columns.tolist(),
        "rows": rows,
    }


def get_document_record(
    project_root: Path,
    dataset_key: str,
    record_id: str,
    *,
    include_annotation: bool = False,
    include_text: bool = False,
    include_boxes: bool = False,
    include_entities: bool = False,
    include_qas: bool = False,
    include_ocr: bool = False,
) -> Dict[str, Any]:
    """
    Return a single record from a document dataset index.jsonl (Step2b/3b outputs).
    Streams the jsonl so it doesn't load the full file into memory.
    """
    loc = resolve_dataset_location(project_root, dataset_key)

    if loc.storage != "jsonl" or loc.index_jsonl is None:
        raise ValueError(f"Dataset '{dataset_key}' is not a document index dataset (jsonl). storage={loc.storage}")

    found: Optional[Dict[str, Any]] = None
    with loc.index_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("id") == record_id:
                found = obj
                break

    if not found:
        raise KeyError(f"Record id '{record_id}' not found in {loc.index_jsonl}")

    # Optionally include linked artifacts
    if include_annotation:
        p = _to_abs_path(project_root, found.get("annotation_path"))
        if p and p.exists():
            found["annotation"] = _safe_read_json(p)

    if include_text:
        p = _to_abs_path(project_root, found.get("text_path"))
        if p and p.exists():
            found["text"] = _safe_read_text(p)

    if include_boxes:
        p = _to_abs_path(project_root, found.get("boxes_path"))
        if p and p.exists():
            found["boxes"] = _safe_read_text(p)

    if include_entities:
        p = _to_abs_path(project_root, found.get("entities_path"))
        if p and p.exists():
            found["entities"] = _safe_read_text(p)

    if include_qas:
        p = _to_abs_path(project_root, found.get("qas_path"))
        if p and p.exists():
            found["qas"] = _safe_read_json(p)

    if include_ocr:
        p = _to_abs_path(project_root, found.get("ocr_path"))
        if p and p.exists():
            found["ocr"] = _safe_read_json(p)

    return found
