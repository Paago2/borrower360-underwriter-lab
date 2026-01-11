from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def read_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def rel_to_project_root(project_root: Path, p: Path) -> str:
    # portable paths in index.jsonl (relative to repo root)
    try:
        return str(p.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(p.resolve())


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def find_files(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    out: List[Path] = []
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


def guess_split_from_path(p: Path) -> str:
    s = str(p).lower()
    if "train" in s or "training" in s:
        return "train"
    if "val" in s or "valid" in s or "validation" in s:
        return "val"
    if "test" in s or "testing" in s:
        return "test"
    return "unknown"


# -----------------------
# Dataset-specific curators
# -----------------------

def curate_funsd(project_root: Path, raw_dir: Path, max_items: Optional[int]) -> List[Dict[str, Any]]:
    # FUNSD often includes images + json annotations with same stem.
    # We scan for image files and try to find annotation json with same filename stem.
    images = find_files(raw_dir, (".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    rows: List[Dict[str, Any]] = []

    for img in images:
        stem = img.stem
        split = guess_split_from_path(img)

        # best-effort: find json with same stem somewhere under raw_dir
        ann = None
        candidates = list(raw_dir.rglob(f"{stem}.json"))
        if candidates:
            ann = candidates[0]

        rows.append(
            {
                "id": f"{split}:{stem}",
                "dataset": "funsd",
                "split": split,
                "image_path": rel_to_project_root(project_root, img),
                "annotation_path": rel_to_project_root(project_root, ann) if ann else None,
            }
        )

        if max_items and len(rows) >= max_items:
            break

    return rows


def curate_sroie(project_root: Path, raw_dir: Path, max_items: Optional[int]) -> List[Dict[str, Any]]:
    # SROIE2019 often has:
    # - images/
    # - box/ (or bounding_box) txt
    # - entities/ txt or json
    images = find_files(raw_dir, (".jpg", ".jpeg", ".png"))
    rows: List[Dict[str, Any]] = []

    for img in images:
        stem = img.stem
        split = guess_split_from_path(img)

        # best-effort matching
        box = None
        ent = None

        box_candidates = list(raw_dir.rglob(f"{stem}.txt"))
        # heuristic: choose a "box" path if present
        if box_candidates:
            box_candidates_sorted = sorted(box_candidates, key=lambda p: ("box" not in str(p).lower(), str(p)))
            box = box_candidates_sorted[0]

        # entities sometimes txt or json
        ent_candidates = list(raw_dir.rglob(f"{stem}.json")) + list(raw_dir.rglob(f"{stem}.txt"))
        if ent_candidates:
            ent_candidates_sorted = sorted(ent_candidates, key=lambda p: ("entit" not in str(p).lower(), str(p)))
            ent = ent_candidates_sorted[0]

        rows.append(
            {
                "id": f"{split}:{stem}",
                "dataset": "sroie",
                "split": split,
                "image_path": rel_to_project_root(project_root, img),
                "boxes_path": rel_to_project_root(project_root, box) if box else None,
                "entities_path": rel_to_project_root(project_root, ent) if ent else None,
            }
        )

        if max_items and len(rows) >= max_items:
            break

    return rows


def curate_docvqa(project_root: Path, raw_dir: Path, max_items: Optional[int]) -> List[Dict[str, Any]]:
    # DocVQA layouts vary a lot. To keep this robust:
    # - index each image as an "asset"
    # - attach nearest qas/ocr json if any exist (best-effort)
    images = find_files(raw_dir, (".jpg", ".jpeg", ".png"))
    jsons = find_files(raw_dir, (".json",))

    rows: List[Dict[str, Any]] = []

    # pick some likely “global” jsons (qas/ocr) to reference
    qas_like = [p for p in jsons if any(t in p.name.lower() for t in ("qa", "qas", "question", "answers"))]
    ocr_like = [p for p in jsons if "ocr" in p.name.lower()]

    qas_path = qas_like[0] if qas_like else None
    ocr_path = ocr_like[0] if ocr_like else None

    for img in images:
        stem = img.stem
        split = guess_split_from_path(img)

        rows.append(
            {
                "id": f"{split}:{stem}",
                "dataset": "docvqa",
                "split": split,
                "image_path": rel_to_project_root(project_root, img),
                "qas_path": rel_to_project_root(project_root, qas_path) if qas_path else None,
                "ocr_path": rel_to_project_root(project_root, ocr_path) if ocr_path else None,
            }
        )

        if max_items and len(rows) >= max_items:
            break

    return rows


def curate_freddie_mac_crt(project_root: Path, raw_dir: Path, max_items: Optional[int]) -> List[Dict[str, Any]]:
    # Treat each text-ish file as a document.
    files = find_files(raw_dir, (".txt", ".csv", ".tsv"))
    rows: List[Dict[str, Any]] = []

    for p in files:
        split = guess_split_from_path(p)
        rows.append(
            {
                "id": f"{split}:{p.stem}",
                "dataset": "freddie_mac_crt",
                "split": split,
                "text_path": rel_to_project_root(project_root, p),
            }
        )
        if max_items and len(rows) >= max_items:
            break

    return rows


def curate_documents_for_key(project_root: Path, ds: Dataset, max_items: Optional[int]) -> Dict[str, Any]:
    raw_path = Path(ds.path).expanduser()
    # ds.path may contain ${raw_root} etc; registry loader should already resolve.
    raw_dir = raw_path if raw_path.is_dir() else raw_path.parent

    out_dir = project_root / "data" / "01_curated" / ds.key
    ensure_dir(out_dir)

    index_path = out_dir / "index.jsonl"

    if ds.key == "funsd":
        rows = curate_funsd(project_root, raw_dir, max_items)
    elif ds.key == "sroie":
        rows = curate_sroie(project_root, raw_dir, max_items)
    elif ds.key == "docvqa":
        rows = curate_docvqa(project_root, raw_dir, max_items)
    elif ds.key == "freddie_mac_crt_2025_12":
        rows = curate_freddie_mac_crt(project_root, raw_dir, max_items)
    else:
        # generic: index any images + json files as assets
        images = find_files(raw_dir, (".jpg", ".jpeg", ".png"))
        rows = []
        for img in images[: (max_items or len(images))]:
            rows.append(
                {
                    "id": f"{guess_split_from_path(img)}:{img.stem}",
                    "dataset": ds.key,
                    "split": guess_split_from_path(img),
                    "image_path": rel_to_project_root(project_root, img),
                }
            )

    n = write_jsonl(index_path, rows)

    meta = {
        "key": ds.key,
        "generated_at_utc": utc_now_iso(),
        "raw_dir": str(raw_dir),
        "output_dir": str(out_dir),
        "index_path": str(index_path),
        "rows_indexed": n,
        "max_items": max_items,
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "_SUCCESS").write_text("", encoding="utf-8")

    return {"key": ds.key, "status": "ok", "rows_indexed": n, "output_dir": str(out_dir), "index_path": str(index_path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None, help="Dataset key (optional). If omitted, run all doc/mixed/text.")
    ap.add_argument("--max-items", type=int, default=2000, help="Max indexed items per dataset (safety cap). Use 0 for no cap.")
    args = ap.parse_args()

    project_root = infer_project_root()
    manifest_path = project_root / "configs" / "datasets.yaml"

    registry = load_registry(manifest_path=manifest_path, project_root=project_root)

    def is_doc_like(d: Dataset) -> bool:
        t = (d.type or "").lower()
        return t in ("image", "mixed", "text", "shapefile")

    keys = [args.dataset] if args.dataset else [k for k, d in registry.items() if is_doc_like(d)]

    max_items = None if args.max_items == 0 else args.max_items

    results: List[Dict[str, Any]] = []
    for key in keys:
        if key not in registry:
            results.append({"key": key, "status": "error", "reason": "Unknown dataset key"})
            continue
        ds = registry[key]
        try:
            res = curate_documents_for_key(project_root, ds, max_items=max_items)
            results.append(res)
            print(f"[OK] {key} -> {res['output_dir']}")
        except Exception as e:
            results.append({"key": key, "status": "error", "reason": str(e)})
            print(f"[ERR] {key} -> {e}")

    reports_dir = project_root / "reports" / "curation_documents"
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
    print(f"\nWrote: {summary_path}")

    any_errors = any(r.get("status") != "ok" for r in results)
    return 1 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
