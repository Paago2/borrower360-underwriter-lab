# app/api/routes_preview.py
from __future__ import annotations

import csv
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from app.core.auth import require_api_key
from app.services.dataset_loader import load_dataset_df, resolve_dataset_location
from app.services.storage import Storage, get_storage_settings

router = APIRouter(
    prefix="/preview",
    tags=["Preview"],
    dependencies=[Depends(require_api_key)],
)

# ----------------------------
# Helpers
# ----------------------------

def infer_project_root() -> Path:
    pr = os.environ.get("PROJECT_ROOT")
    if pr:
        return Path(pr).resolve()
    # routes_preview.py is app/api/routes_preview.py -> parents[2] = repo root
    return Path(__file__).resolve().parents[2]


def preview_csv(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            rows.append(dict(row))
    return rows


def preview_text(path: Path, limit: int) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            lines.append(line.rstrip("\n"))
    return lines


def _safe_preview_df(df, *, max_preview_rows: int = 50) -> Tuple[List[str], List[Dict[str, Any]]]:
    import pandas as pd

    preview = df.head(min(len(df), max_preview_rows)).copy()

    # If a column stores raw bytes dicts, summarize instead of dumping bytes
    if "image" in preview.columns:
        def _image_summary(x):
            try:
                if isinstance(x, dict) and "bytes" in x and isinstance(x["bytes"], (bytes, bytearray)):
                    return {"bytes_len": len(x["bytes"])}
                return x
            except Exception:
                return {"bytes_len": None}

        preview["image"] = preview["image"].apply(_image_summary)

    # Truncate very long strings
    for col in preview.columns:
        if pd.api.types.is_string_dtype(preview[col]) or preview[col].dtype == "object":
            preview[col] = preview[col].map(
                lambda v: (str(v)[:300] + "...") if isinstance(v, str) and len(v) > 300 else v
            )

    return preview.columns.tolist(), preview.to_dict(orient="records")


def _find_record_in_jsonl(index_jsonl: Path, record_id: str) -> Optional[Dict[str, Any]]:
    with index_jsonl.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("id") == record_id:
                return obj
    return None


def _resolve_repo_path(project_root: Path, p: str) -> Path:
    """
    Resolve a path safely inside repo root (prevents path traversal).
    """
    root = project_root.resolve()
    raw = Path(p)
    abs_path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()

    try:
        abs_path.relative_to(root)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Refusing to access path outside project root. path='{p}' resolved='{abs_path}' root='{root}'",
        )
    return abs_path


def _safe_read_text_path(path: Path, *, max_chars: int = 20_000) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"exists": False, "path": str(path)}
    txt = path.read_text(encoding="utf-8", errors="replace")
    if len(txt) > max_chars:
        return {"exists": True, "path": str(path), "truncated": True, "text": txt[:max_chars]}
    return {"exists": True, "path": str(path), "truncated": False, "text": txt}


def _safe_read_json_path(path: Path, *, max_bytes: int = 2_000_000) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {"exists": False, "path": str(path)}
    size = path.stat().st_size
    if size > max_bytes:
        return {"exists": True, "path": str(path), "too_large": True, "size_bytes": size}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return {"exists": True, "path": str(path), "too_large": False, "json": json.load(f)}

# ----------------------------
# Endpoints
# ----------------------------

@router.get("/{dataset_key}", operation_id="preview_dataset")
def preview_dataset(
    dataset_key: str,
    max_rows: int = Query(50, ge=1, le=500),
    max_parts: int = Query(1, ge=1, le=50),
    columns: Optional[List[str]] = Query(None),
) -> Dict[str, Any]:
    project_root = infer_project_root()

    try:
        df, loc = load_dataset_df(
            project_root,
            dataset_key,
            columns=columns,
            max_parts=max_parts,
            max_rows=max_rows,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if loc.storage == "parquet":
        total_shards = len(loc.parquet_files)
        shards_read = min(total_shards, max_parts)
    else:
        total_shards = 1
        shards_read = 1

    cols, rows = _safe_preview_df(df, max_preview_rows=max_rows)

    return {
        "dataset_key": dataset_key,
        "tier": loc.tier,
        "storage": loc.storage,
        "dir_ref": loc.dir_ref,
        "dir_uri": loc.dir_uri,
        "shards_read": shards_read,
        "total_shards": total_shards,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": cols,
        "rows": rows,
    }


@router.get("/{dataset_key}/records/{record_id}", operation_id="get_document_record")
def get_document_record(
    dataset_key: str,
    record_id: str,
    include_annotation: bool = Query(False),
    include_boxes: bool = Query(False),
    include_entities: bool = Query(False),
    include_qas: bool = Query(False),
    include_ocr: bool = Query(False),
    include_text: bool = Query(False),
) -> Dict[str, Any]:
    """
    Fetch ONE record from a document dataset (index.jsonl).
    By default returns only the index record (paths + metadata).
    Optionally loads referenced files safely (size-limited).
    """
    project_root = infer_project_root()
    storage = Storage(get_storage_settings(project_root))

    try:
        loc = resolve_dataset_location(project_root, dataset_key)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if loc.storage != "jsonl" or not loc.index_jsonl:
        raise HTTPException(status_code=400, detail=f"Dataset '{dataset_key}' is not a jsonl document dataset.")

    # index.jsonl may be local OR cached-from-s3 (later)
    index_local = storage.as_local_path(loc.index_jsonl)
    rec = _find_record_in_jsonl(index_local, record_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Record id '{record_id}' not found.")

    out: Dict[str, Any] = {
        "location": {"tier": loc.tier, "dir_ref": loc.dir_ref, "dir_uri": loc.dir_uri},
        "record": rec,
    }

    def _asset_local_path(rel: Optional[str]) -> Optional[Path]:
        if not rel:
            return None
        # safety: must be inside repo root
        _resolve_repo_path(project_root, rel)
        # map through storage (local now, s3 later)
        return storage.as_local_path(rel)

    if include_annotation:
        p = _asset_local_path(rec.get("annotation_path"))
        if p:
            out["annotation"] = _safe_read_json_path(p)

    if include_boxes:
        p = _asset_local_path(rec.get("boxes_path"))
        if p:
            out["boxes"] = _safe_read_text_path(p)

    if include_entities:
        p = _asset_local_path(rec.get("entities_path"))
        if p:
            out["entities"] = _safe_read_text_path(p)

    if include_qas:
        p = _asset_local_path(rec.get("qas_path"))
        if p:
            out["qas"] = _safe_read_json_path(p)

    if include_ocr:
        p = _asset_local_path(rec.get("ocr_path"))
        if p:
            out["ocr"] = _safe_read_json_path(p)

    if include_text:
        p = _asset_local_path(rec.get("text_path"))
        if p:
            out["text"] = _safe_read_text_path(p)

    return out


@router.get("/{dataset_key}/records/{record_id}/image", operation_id="get_document_image")
def get_document_image(dataset_key: str, record_id: str):
    """
    Return the image bytes for a record.
    - local backend: FileResponse from disk
    - s3 backend: StreamingResponse from S3
    """
    project_root = infer_project_root()
    storage = Storage(get_storage_settings(project_root))

    try:
        loc = resolve_dataset_location(project_root, dataset_key)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if loc.storage != "jsonl" or not loc.index_jsonl:
        raise HTTPException(status_code=400, detail=f"Dataset '{dataset_key}' is not a jsonl document dataset.")

    index_local = storage.as_local_path(loc.index_jsonl)
    rec = _find_record_in_jsonl(index_local, record_id)
    if not rec:
        raise HTTPException(status_code=404, detail=f"Record id '{record_id}' not found.")

    image_rel = rec.get("image_path")
    if not image_rel:
        raise HTTPException(status_code=404, detail=f"Record '{record_id}' has no image_path.")

    # Safety check (prevents traversal)
    _resolve_repo_path(project_root, image_rel)

    # Local => file response
    if storage.backend == "local":
        image_abs = storage.as_local_path(image_rel)
        if not image_abs.exists() or not image_abs.is_file():
            raise HTTPException(status_code=404, detail=f"Image file not found: {image_abs}")

        media_type, _ = mimetypes.guess_type(str(image_abs))
        return FileResponse(
            path=str(image_abs),
            media_type=media_type or "application/octet-stream",
            filename=image_abs.name,
        )

    # S3 => stream (when you’re ready later)
    if not storage.exists_file(image_rel):
        raise HTTPException(status_code=404, detail=f"Image not found in S3: {storage.uri(image_rel)}")

    it, media_type, filename = storage.open_s3_stream(image_rel)
    return StreamingResponse(
        it,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
