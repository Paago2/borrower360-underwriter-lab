from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import yaml

from app.services.dataset_registry import load_registry, Dataset


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_project_root() -> Path:
    pr = os.environ.get("PROJECT_ROOT")
    if pr:
        return Path(pr).resolve()
    return Path(__file__).resolve().parents[1]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rel_to_project_root(project_root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(p.resolve())


def abs_from_index(project_root: Path, rel_or_abs: Optional[str]) -> Optional[Path]:
    if not rel_or_abs:
        return None
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def sha256_file(p: Path, max_bytes: int = 2_000_000) -> str:
    """
    Hash up to max_bytes for speed (still good for stable ID-ish fingerprints).
    """
    h = hashlib.sha256()
    with p.open("rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()


def safe_file_meta(project_root: Path, relpath: Optional[str]) -> Dict[str, Any]:
    p = abs_from_index(project_root, relpath)
    if not p or not p.exists():
        return {"path": relpath, "exists": False}

    try:
        size = p.stat().st_size
    except Exception:
        size = None

    out = {
        "path": relpath,
        "exists": True,
        "size_bytes": size,
    }

    # Only hash for smaller-ish files; images are fine, huge files not needed
    try:
        if size is not None and size <= 50_000_000:
            out["sha256_head"] = sha256_file(p)
    except Exception:
        out["sha256_head"] = None

    return out


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def standardize_doc_index(project_root: Path, rows: List[Dict[str, Any]], dataset_key: str) -> List[Dict[str, Any]]:
    """
    Normalize to a consistent schema across document datasets.

    Standard keys we aim for:
      - id, dataset_key, split, modality
      - image_path, text_path, annotation_path, boxes_path, entities_path, qas_path, ocr_path
      - meta: file sizes/hashes/existence flags
    """
    out: List[Dict[str, Any]] = []

    for r in rows:
        split = r.get("split") or "unknown"
        rid = r.get("id") or f"{split}:{r.get('image_path') or r.get('text_path') or 'item'}"

        # detect modality
        modality = "mixed"
        has_image = bool(r.get("image_path"))
        has_text = bool(r.get("text_path"))
        if has_image and not has_text:
            modality = "image"
        elif has_text and not has_image:
            modality = "text"

        std = {
            "id": rid,
            "dataset_key": dataset_key,
            "split": split,
            "modality": modality,
            "image_path": r.get("image_path"),
            "text_path": r.get("text_path"),
            "annotation_path": r.get("annotation_path"),
            "boxes_path": r.get("boxes_path"),
            "entities_path": r.get("entities_path"),
            "qas_path": r.get("qas_path"),
            "ocr_path": r.get("ocr_path"),
        }

        # attach file metadata for things that exist
        std["meta"] = {
            "image": safe_file_meta(project_root, std["image_path"]) if std["image_path"] else None,
            "text": safe_file_meta(project_root, std["text_path"]) if std["text_path"] else None,
            "annotation": safe_file_meta(project_root, std["annotation_path"]) if std["annotation_path"] else None,
            "boxes": safe_file_meta(project_root, std["boxes_path"]) if std["boxes_path"] else None,
            "entities": safe_file_meta(project_root, std["entities_path"]) if std["entities_path"] else None,
            "qas": safe_file_meta(project_root, std["qas_path"]) if std["qas_path"] else None,
            "ocr": safe_file_meta(project_root, std["ocr_path"]) if std["ocr_path"] else None,
        }

        out.append(std)

    return out


def standardize_one(ds: Dataset, project_root: Path, max_items: Optional[int]) -> Dict[str, Any]:
    curated_dir = project_root / "data" / "01_curated" / ds.key
    in_index = curated_dir / "index.jsonl"
    if not in_index.exists():
        return {"key": ds.key, "status": "error", "reason": f"Missing curated index.jsonl at {in_index}"}

    out_dir = project_root / "data" / "02_standardized" / ds.key
    ensure_dir(out_dir)
    out_index = out_dir / "index.jsonl"

    rows = read_jsonl(in_index)
    if max_items is not None:
        rows = rows[:max_items]

    std_rows = standardize_doc_index(project_root, rows, dataset_key=ds.key)
    n = write_jsonl(out_index, std_rows)

    meta = {
        "key": ds.key,
        "generated_at_utc": utc_now_iso(),
        "input_index": str(in_index),
        "output_index": str(out_index),
        "rows_written": n,
        "max_items": max_items,
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "_SUCCESS").write_text("", encoding="utf-8")

    return {"key": ds.key, "status": "ok", "output_dir": str(out_dir), "rows_written": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None, help="Dataset key (optional). If omitted, run all doc/mixed/text with curated index.jsonl.")
    ap.add_argument("--max-items", type=int, default=2000, help="Max standardized items per dataset (safety cap). Use 0 for no cap.")
    args = ap.parse_args()

    project_root = infer_project_root()
    manifest_path = project_root / "configs" / "datasets.yaml"
    registry = load_registry(manifest_path=manifest_path, project_root=project_root)

    def has_curated_index(key: str) -> bool:
        return (project_root / "data" / "01_curated" / key / "index.jsonl").exists()

    keys = [args.dataset] if args.dataset else [k for k in registry.keys() if has_curated_index(k)]
    max_items = None if args.max_items == 0 else args.max_items

    results: List[Dict[str, Any]] = []
    for key in keys:
        if key not in registry:
            results.append({"key": key, "status": "error", "reason": "Unknown dataset key"})
            continue
        ds = registry[key]
        try:
            res = standardize_one(ds, project_root, max_items=max_items)
            results.append(res)
            if res.get("status") == "ok":
                print(f"[OK] {key} -> {res['output_dir']}")
            else:
                print(f"[ERR] {key} -> {res.get('reason')}")
        except Exception as e:
            results.append({"key": key, "status": "error", "reason": str(e)})
            print(f"[ERR] {key} -> {e}")

    reports_dir = project_root / "reports" / "standardization_documents"
    ensure_dir(reports_dir)

    summary = {
        "generated_at_utc": utc_now_iso(),
        "project_root": str(project_root),
        "manifest_path": str(manifest_path),
        "mode": "doc_index",
        "max_items": max_items,
        "results": results,
    }
    summary_path = reports_dir / "_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote summary: {summary_path}")

    any_errors = any(r.get("status") != "ok" for r in results)
    return 1 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
