from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

# ---- Make imports work even if PYTHONPATH isn't set correctly ----
PROJECT_ROOT_FALLBACK = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FALLBACK) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FALLBACK))

from app.core.config import project_root, manifest_path  # noqa: E402
from app.services.dataset_registry import Dataset, load_registry  # noqa: E402

# Optional deps (we degrade gracefully if missing)
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover
    pq = None  # type: ignore


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:.2f} {u}"
        size /= 1024.0
    return f"{n} B"


def safe_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def file_info(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "is_dir": path.is_dir(),
        "size_bytes": st.st_size if path.is_file() else None,
        "size_human": human_bytes(st.st_size) if path.is_file() else None,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }


def dir_scan(root: Path, max_files_list: int = 30) -> Dict[str, Any]:
    """
    Scan a directory and summarize its contents without loading data.
    """
    files: List[Path] = []
    total_size = 0
    ext_counter: Counter[str] = Counter()

    for p in root.rglob("*"):
        if p.is_file():
            files.append(p)
            total_size += p.stat().st_size
            ext_counter[p.suffix.lower() or "(no_ext)"] += 1

    # Example files (relative)
    example_files = [str(p.relative_to(root)) for p in files[:max_files_list]]

    return {
        "exists": True,
        "is_dir": True,
        "file_count": len(files),
        "total_size_bytes": total_size,
        "total_size_human": human_bytes(total_size),
        "extensions": dict(ext_counter.most_common()),
        "example_files": example_files,
    }


def detect_delimiter(path: Path, sample_bytes: int = 64_000) -> str:
    """
    Try to detect delimiter for "CSV" files that might actually be pipe/tab delimited.
    Falls back to common delimiters.
    """
    raw = path.read_bytes()[:sample_bytes]
    text = raw.decode("utf-8", errors="replace")

    # Use csv.Sniffer first
    try:
        dialect = csv.Sniffer().sniff(text, delimiters=[",", "|", "\t", ";"])
        return dialect.delimiter
    except Exception:
        pass

    # Fallback: count occurrences in first non-empty line
    for line in text.splitlines():
        if line.strip():
            counts = {
                ",": line.count(","),
                "|": line.count("|"),
                "\t": line.count("\t"),
                ";": line.count(";"),
            }
            # choose delimiter with max count
            delim = max(counts, key=counts.get)
            return delim if counts[delim] > 0 else ","

    return ","



def profile_csv(
    path: Path,
    sample_rows: int,
    exact_row_count_max_bytes: int,
    chunksize: int,
) -> Dict[str, Any]:
    if pd is None:
        return {"error": "pandas is not installed; cannot profile CSV"}

    size_bytes = path.stat().st_size
    sep = detect_delimiter(path)

    out: Dict[str, Any] = {
        "format": "csv",
        "size_bytes": size_bytes,
        "size_human": human_bytes(size_bytes),
        "sample_rows_requested": sample_rows,
        "detected_delimiter": repr(sep),
    }

    read_kwargs = dict(
        sep=sep,
        engine="python",          # allows sep=None style sniffing behaviors and is more tolerant
        on_bad_lines="skip",      # don't die on malformed lines
        encoding="utf-8",
    )

    # ---- Sample read ----
    try:
        df = pd.read_csv(path, nrows=sample_rows, low_memory=False, **read_kwargs)
    except Exception as e:
        out["error"] = f"Sample read failed: {type(e).__name__}: {e}"
        out["hint"] = "File may not be a standard CSV (could be pipe/tab delimited or have metadata lines)."
        return out

    out["sample_rows_read"] = int(len(df))
    out["columns"] = list(df.columns)

    # ---- Column stats (from sample) ----
    col_stats: Dict[str, Any] = {}
    for c in df.columns:
        s = df[c]
        nulls = int(s.isna().sum())
        non_null = int(len(s) - nulls)
        unique = int(s.nunique(dropna=True))
        dtype = str(s.dtype)

        examples: List[str] = []
        try:
            examples = s.dropna().astype(str).head(5).tolist()
        except Exception:
            pass

        stats: Dict[str, Any] = {
            "dtype_sample": dtype,
            "null_count_sample": nulls,
            "null_pct_sample": (nulls / max(len(s), 1)) * 100.0,
            "non_null_count_sample": non_null,
            "unique_count_sample": unique,
            "unique_pct_sample": (unique / max(non_null, 1)) * 100.0,
            "example_values": examples,
        }

        # Numeric min/max (sample)
        try:
            if str(s.dtype).startswith(("int", "float")):
                stats["min_sample"] = float(s.min())
                stats["max_sample"] = float(s.max())
        except Exception:
            pass

        # Datetime parse attempt — only try if the column name looks date-like
        try:
            if str(s.dtype) == "object" and any(tok in str(c).lower() for tok in ["date", "dt", "time"]):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # silence "Could not infer format" warnings
                    parsed = pd.to_datetime(s, errors="coerce", utc=True)
                parsed_non_null = int(parsed.notna().sum())
                if parsed_non_null > 0:
                    stats["datetime_parseable_pct_sample"] = (parsed_non_null / max(len(s), 1)) * 100.0
        except Exception:
            pass

        col_stats[str(c)] = stats

    out["column_stats_sample"] = col_stats

    # ---- Candidate keys (heuristic) ----
    candidate_keys: List[str] = []
    try:
        for c in df.columns:
            s = df[c]
            if len(s) == 0:
                continue
            null_pct = (s.isna().sum() / len(s)) * 100.0
            non_null_s = s.dropna()
            if len(non_null_s) == 0:
                continue
            unique_pct = (non_null_s.nunique() / len(non_null_s)) * 100.0
            if null_pct <= 1.0 and unique_pct >= 99.0:
                candidate_keys.append(str(c))
    except Exception:
        pass
    out["candidate_keys_sample"] = candidate_keys

    # ---- Exact row count (only if not too big) ----
    if size_bytes <= exact_row_count_max_bytes:
        try:
            total_rows = 0
            for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False, **read_kwargs):
                total_rows += len(chunk)
            out["row_count_exact"] = int(total_rows)
        except Exception as e:
            out["row_count_exact_error"] = f"{type(e).__name__}: {e}"
    else:
        out["row_count_exact"] = None
        out["row_count_note"] = (
            f"Skipped exact row count because file > {human_bytes(exact_row_count_max_bytes)}. "
            "Increase --exact-rowcount-max-mb if you want."
        )

    return out



def profile_parquet(path: Path, sample_rows: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "format": "parquet",
        "size_bytes": path.stat().st_size,
        "size_human": human_bytes(path.stat().st_size),
        "sample_rows_requested": sample_rows,
    }

    if pq is None and pd is None:
        out["error"] = "Neither pyarrow nor pandas are available; cannot profile parquet"
        return out

    # If pyarrow is available, get metadata quickly
    if pq is not None:
        try:
            pf = pq.ParquetFile(path)
            out["parquet_num_row_groups"] = pf.num_row_groups
            out["parquet_schema"] = str(pf.schema)
        except Exception as e:
            out["parquet_metadata_error"] = str(e)

    # Sample read via pandas if possible
    if pd is not None:
        try:
            df = pd.read_parquet(path)
            if len(df) > sample_rows:
                df = df.head(sample_rows)
            out["sample_rows_read"] = int(len(df))
            out["columns"] = list(df.columns)
        except Exception as e:
            out["sample_read_error"] = str(e)

    return out


def profile_dataset(
    ds: Dataset,
    sample_rows: int,
    exact_row_count_max_bytes: int,
    chunksize: int,
) -> Dict[str, Any]:
    p = Path(ds.path)
    base: Dict[str, Any] = {
        "profile_version": 1,
        "generated_at_utc": utc_now_iso(),
        "dataset": asdict(ds),
        "exists": p.exists(),
        "path": ds.path,
        "type": ds.type,
    }

    if not p.exists():
        base["error"] = "Path does not exist"
        return base

    # Directory types: image / text / mixed / shapefile / ocr_json (often directories)
    if p.is_dir():
        base["storage"] = dir_scan(p)

        # Some extra hints by dataset type
        exts = base["storage"].get("extensions", {})
        if ds.type in {"image", "mixed"}:
            base["hints"] = {
                "image_files_estimate": sum(exts.get(e, 0) for e in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]),
                "json_files_estimate": exts.get(".json", 0),
                "txt_files_estimate": exts.get(".txt", 0),
            }
        if ds.type == "shapefile":
            base["hints"] = {
                "has_shp": exts.get(".shp", 0) > 0,
                "has_dbf": exts.get(".dbf", 0) > 0,
                "has_shx": exts.get(".shx", 0) > 0,
                "has_prj": exts.get(".prj", 0) > 0,
            }
        return base

    # File types: tabular/text/etc.
    base["storage"] = file_info(p)

    ext = p.suffix.lower()
    if ds.type == "tabular":
        if ext == ".csv":
            base["tabular_profile"] = profile_csv(
                p,
                sample_rows=sample_rows,
                exact_row_count_max_bytes=exact_row_count_max_bytes,
                chunksize=chunksize,
            )
        elif ext in {".parquet"}:
            base["tabular_profile"] = profile_parquet(p, sample_rows=sample_rows)
        elif ext in {".json"}:
            base["note"] = "JSON tabular profiling not implemented yet; treat as semi-structured."
        else:
            base["note"] = f"Unknown tabular extension: {ext}"
        return base

    if ds.type == "text":
        # For text files, don’t read the whole thing; sample head
        try:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                head_lines = []
                for _ in range(50):
                    line = f.readline()
                    if not line:
                        break
                    head_lines.append(line.rstrip("\n"))
            base["text_profile"] = {
                "head_lines": head_lines,
                "head_line_count": len(head_lines),
            }
        except Exception as e:
            base["text_profile_error"] = str(e)
        return base

    # default fallback
    base["note"] = f"No specialized profiler for dataset type='{ds.type}' and extension='{ext}'."
    return base


def render_markdown(profile: Dict[str, Any]) -> str:
    ds = profile.get("dataset", {})
    lines = []
    lines.append(f"# Dataset profile: `{ds.get('key', '')}`")
    lines.append("")
    lines.append(f"- **Name:** {ds.get('name')}")
    lines.append(f"- **Type:** {ds.get('type')}")
    lines.append(f"- **Path:** `{ds.get('path')}`")
    lines.append(f"- **Exists:** `{profile.get('exists')}`")
    lines.append(f"- **Generated (UTC):** {profile.get('generated_at_utc')}")
    lines.append("")

    if not profile.get("exists"):
        lines.append("## Error")
        lines.append(profile.get("error", "Unknown error"))
        return "\n".join(lines)

    storage = profile.get("storage", {})
    if storage.get("is_dir"):
        lines.append("## Storage summary (directory)")
        lines.append(f"- File count: **{storage.get('file_count')}**")
        lines.append(f"- Total size: **{storage.get('total_size_human')}**")
        lines.append("")
        lines.append("### Extensions (top)")
        for k, v in (storage.get("extensions") or {}).items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")
        lines.append("### Example files")
        for ex in storage.get("example_files", [])[:20]:
            lines.append(f"- `{ex}`")
        return "\n".join(lines)

    # File-based
    lines.append("## Storage summary (file)")
    lines.append(f"- Size: **{storage.get('size_human')}**")
    lines.append(f"- Modified (UTC): `{storage.get('mtime_utc')}`")
    lines.append("")

    tab = profile.get("tabular_profile")
    if tab:
        lines.append("## Tabular profile")
        if "error" in tab:
            lines.append(f"- Error: {tab['error']}")
            return "\n".join(lines)

        if tab.get("format"):
            lines.append(f"- Format: `{tab.get('format')}`")
        if tab.get("row_count_exact") is not None:
            lines.append(f"- Row count (exact): **{tab.get('row_count_exact')}**")
        if tab.get("row_count_note"):
            lines.append(f"- Row count note: {tab.get('row_count_note')}")
        if tab.get("sample_rows_read") is not None:
            lines.append(f"- Sample rows read: **{tab.get('sample_rows_read')}**")
        lines.append("")

        ckeys = tab.get("candidate_keys_sample", [])
        if ckeys:
            lines.append("### Candidate keys (heuristic from sample)")
            for c in ckeys:
                lines.append(f"- `{c}`")
            lines.append("")

        col_stats = tab.get("column_stats_sample", {})
        if col_stats:
            lines.append("### Columns (sample stats)")
            for col, st in list(col_stats.items())[:60]:
                lines.append(f"- `{col}` | dtype={st.get('dtype_sample')} | null%={st.get('null_pct_sample'):.2f} | unique%={st.get('unique_pct_sample'):.2f}")
        return "\n".join(lines)

    txt = profile.get("text_profile")
    if txt:
        lines.append("## Text profile")
        lines.append(f"- Head lines: {txt.get('head_line_count')}")
        lines.append("")
        lines.append("```")
        for l in txt.get("head_lines", [])[:50]:
            lines.append(l)
        lines.append("```")
        return "\n".join(lines)

    if profile.get("note"):
        lines.append("## Note")
        lines.append(str(profile["note"]))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 1: profile all datasets in configs/datasets.yaml")
    parser.add_argument("--out-dir", default="reports/profiles", help="Output directory for profile reports")
    parser.add_argument("--sample-rows", type=int, default=20000, help="Rows to sample for tabular datasets")
    parser.add_argument(
        "--exact-rowcount-max-mb",
        type=int,
        default=200,
        help="Compute exact row count only if CSV <= this many MB (default: 200)",
    )
    parser.add_argument("--chunksize", type=int, default=200000, help="CSV chunksize for exact row counting")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reg = load_registry(manifest_path=manifest_path, project_root=project_root)

    exact_row_count_max_bytes = int(args.exact_rowcount_max_mb * 1024 * 1024)

    summary: Dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "project_root": str(project_root),
        "manifest_path": str(manifest_path),
        "datasets_total": len(reg),
        "profiles": [],
    }

    for key, ds in reg.items():

        try:
            profile = profile_dataset(
                ds,
                sample_rows=args.sample_rows,
                exact_row_count_max_bytes=exact_row_count_max_bytes,
                chunksize=args.chunksize,
            )
        except Exception as e:
            profile = {
                "profile_version": 1,
                "generated_at_utc": utc_now_iso(),
                "dataset": asdict(ds),
                "exists": Path(ds.path).exists(),
                "error": f"Unhandled exception profiling dataset: {type(e).__name__}: {e}",
            }


        

        json_path = out_dir / f"{key}.json"
        md_path = out_dir / f"{key}.md"

        safe_write_json(json_path, profile)
        safe_write_text(md_path, render_markdown(profile))

        summary["profiles"].append(
            {
                "key": key,
                "exists": profile.get("exists"),
                "json": str(json_path),
                "md": str(md_path),
            }
        )

        print(f"[OK] {key} -> {json_path}")

    # Write run summary
    safe_write_json(out_dir / "_SUMMARY.json", summary)
    print(f"\nWrote summary: {out_dir / '_SUMMARY.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
