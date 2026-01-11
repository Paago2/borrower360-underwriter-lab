from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd
import yaml

from app.core.config import project_root, manifest_path
from app.services.dataset_registry import load_registry


def _load_contracts(contracts_path: Path) -> Dict[str, Dict[str, Any]]:
    if not contracts_path.exists():
        return {}
    data = yaml.safe_load(contracts_path.read_text(encoding="utf-8")) or {}
    return (data.get("contracts") or {})


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sniff_delimiter(sample_text: str) -> str:
    # Fallback-safe sniffer: try common delimiters if csv.Sniffer struggles.
    candidates = [",", "|", "\t", ";"]
    counts = {c: sample_text.count(c) for c in candidates}
    # choose the delimiter with max count; if all zero, default comma
    best = max(counts.items(), key=lambda kv: kv[1])[0]
    return best if counts[best] > 0 else ","


def _infer_header(sample_line: str) -> bool:
    # Simple heuristic: if it has letters, likely header.
    # (Fannie Mae sample rows are numeric/codes -> no header)
    return any(ch.isalpha() for ch in sample_line)


def _read_sample_text(path: Path, nbytes: int = 64_000, encoding: str = "utf-8") -> str:
    with path.open("rb") as f:
        raw = f.read(nbytes)
    try:
        return raw.decode(encoding, errors="replace")
    except Exception:
        return raw.decode("utf-8", errors="replace")


def _make_colnames(n: int) -> list[str]:
    # col_001, col_002, ...
    width = max(3, len(str(n)))
    return [f"col_{i:0{width}d}" for i in range(1, n + 1)]


def _read_csv_iter(
    path: Path,
    *,
    delimiter: str,
    header: bool,
    encoding: str,
    chunksize: int,
    sample_rows: Optional[int],
    columns: Optional[list[str]] = None,
) -> Iterable[pd.DataFrame]:

    # We keep everything as string initially (safe for large + messy data).
    # You can type-cast in Step 3 (standardization/features).
    read_kwargs: Dict[str, Any] = {
        "sep": delimiter,
        "encoding": encoding,
        "dtype": "string",
        "engine": "python",       # more tolerant for weird rows
        "on_bad_lines": "skip",
    }

    if header:
        read_kwargs["header"] = 0
    else:
        read_kwargs["header"] = None

    # If sampling, just read nrows and yield once.
    if sample_rows is not None:
        df = pd.read_csv(path, nrows=sample_rows, **read_kwargs)
        if not header:
            if columns and len(columns) == df.shape[1]:
                df.columns = columns
            else:
                df.columns = _make_colnames(df.shape[1])

        yield df
        return

    # Full read in chunks
    for chunk in pd.read_csv(path, chunksize=chunksize, **read_kwargs):
        if not header and chunk.shape[1] > 0:
            if columns and len(columns) == chunk.shape[1]:
                chunk.columns = columns
            else:
                chunk.columns = _make_colnames(chunk.shape[1])

        yield chunk


def curate_tabular_dataset(
    key: str,
    ds_path: Path,
    *,
    curated_root: Path,
    contracts: Dict[str, Dict[str, Any]],
    chunksize: int,
    sample_rows: Optional[int],
    force: bool,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    contract = contracts.get(key, {})
    encoding = contract.get("encoding", "utf-8")

    contract_columns = contract.get("columns")
    if contract_columns is not None and not isinstance(contract_columns, list):
        contract_columns = None


    if ds_path.is_dir():
        # Special case: datasets that are folders (e.g., parquet shards)
        # If folder contains parquet files, treat it as curated-by-copy (sample mode copies first N files).
        parquet_files = sorted(ds_path.glob("**/*.parquet"))
        if parquet_files:
            out_dir = curated_root / key
            _ensure_dir(out_dir)

            meta_path = out_dir / "_meta.json"
            done_flag = out_dir / "_SUCCESS"

            if done_flag.exists() and not force:
                return {
                    "key": key,
                    "status": "skipped",
                    "reason": "Already curated (found _SUCCESS). Use --force to rebuild.",
                    "output_dir": str(out_dir),
                }

            # Clean old
            if force and out_dir.exists():
                for p in out_dir.glob("*.parquet"):
                    p.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
                done_flag.unlink(missing_ok=True)

            # sample mode: copy first ~10 parquet shards (fast)
            max_files = 10 if sample_rows is not None else len(parquet_files)
            copied = 0
            for src in parquet_files[:max_files]:
                dst = out_dir / src.name
                dst.write_bytes(src.read_bytes())
                copied += 1

            meta = {
                "key": key,
                "source_path": str(ds_path),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "sample" if sample_rows is not None else "full",
                "parquet_files_found": len(parquet_files),
                "parquet_files_copied": copied,
                "output_dir": str(out_dir),
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            done_flag.write_text("ok\n", encoding="utf-8")

            return {"key": key, "status": "ok", **meta}

        return {"key": key, "status": "skipped", "reason": f"Directory has no parquet files: {ds_path}"}

    # file case
    if ds_path.suffix.lower() not in [".csv", ".txt", ".parquet"]:
        return {"key": key, "status": "skipped", "reason": f"Unsupported file type: {ds_path.name}"}

    # parquet file case
    if ds_path.suffix.lower() == ".parquet":
        out_dir = curated_root / key
        _ensure_dir(out_dir)
        dst = out_dir / ds_path.name
        if (out_dir / "_SUCCESS").exists() and not force:
            return {"key": key, "status": "skipped", "reason": "Already curated (found _SUCCESS)."}
        dst.write_bytes(ds_path.read_bytes())
        (out_dir / "_SUCCESS").write_text("ok\n", encoding="utf-8")
        return {"key": key, "status": "ok", "output_dir": str(out_dir)}


    # delimiter/header: contract wins, otherwise sniff
    sample_text = _read_sample_text(ds_path, encoding=encoding)
    first_line = sample_text.splitlines()[0] if sample_text.splitlines() else ""
    delimiter = contract.get("delimiter") or _sniff_delimiter(sample_text)
    header = contract.get("header")
    if header is None:
        header = _infer_header(first_line)

    out_dir = curated_root / key
    _ensure_dir(out_dir)

    # output files
    meta_path = out_dir / "_meta.json"
    done_flag = out_dir / "_SUCCESS"

    if done_flag.exists() and not force:
        return {
            "key": key,
            "status": "skipped",
            "reason": "Already curated (found _SUCCESS). Use --force to rebuild.",
            "output_dir": str(out_dir),
        }

    # Remove old outputs if force
    if force and out_dir.exists():
        for p in out_dir.glob("*.parquet"):
            p.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        done_flag.unlink(missing_ok=True)

    total_rows = 0
    part = 0

    for df in _read_csv_iter(
        ds_path,
        delimiter=delimiter,
        header=bool(header),
        encoding=encoding,
        chunksize=chunksize,
        sample_rows=sample_rows,
        columns=contract_columns,
    ):

        part += 1
        total_rows += len(df)
        out_file = out_dir / f"part-{part:05d}.parquet"
        df.to_parquet(out_file, index=False)

    meta = {
        "key": key,
        "source_path": str(ds_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "delimiter": delimiter,
        "header": bool(header),
        "encoding": encoding,
        "sample_rows": sample_rows,
        "chunksize": None if sample_rows is not None else chunksize,
        "parts_written": part,
        "rows_written": total_rows,
        "output_dir": str(out_dir),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    done_flag.write_text("ok\n", encoding="utf-8")

    return {"key": key, "status": "ok", **meta}


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 2: Curate tabular datasets to Parquet using registry + contracts.")
    parser.add_argument("--contracts", default="configs/contracts.yaml", help="Path to contracts yaml.")
    parser.add_argument("--dataset", default=None, help="Curate only one dataset key.")
    parser.add_argument("--full", action="store_true", help="Process full files (chunked). Default is sample mode.")
    parser.add_argument("--sample-rows", type=int, default=200_000, help="Rows to sample in non-full mode.")
    parser.add_argument("--chunksize", type=int, default=250_000, help="Chunk size for full mode.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing curated outputs.")
    args = parser.parse_args()

    contracts = _load_contracts(Path(args.contracts))
    reg = load_registry(manifest_path=manifest_path, project_root=project_root)

    curated_root = project_root / "data" / "01_curated"
    reports_dir = project_root / "reports" / "curation"
    _ensure_dir(curated_root)
    _ensure_dir(reports_dir)

    sample_rows = None if args.full else int(args.sample_rows)

    results: list[Dict[str, Any]] = []
    keys = [args.dataset] if args.dataset else list(reg.keys())

    for key in keys:
        ds = reg.get(key)
        if ds is None:
            results.append({"key": key, "status": "error", "reason": "Unknown dataset key"})
            continue

        if ds.type != "tabular":
            # Step 2 (this script) focuses only on tabular->parquet.
            results.append({"key": key, "status": "skipped", "reason": f"type={ds.type} (not tabular)"})
            continue

        p = Path(ds.path)
        if not p.exists():
            results.append({"key": key, "status": "error", "reason": f"Path not found: {p}"})
            continue

        try:
            res = curate_tabular_dataset(
                key,
                p,
                curated_root=curated_root,
                contracts=contracts,
                chunksize=int(args.chunksize),
                sample_rows=sample_rows,
                force=bool(args.force),
            )
            results.append(res)
            if res.get("status") == "ok":
                print(f"[OK] {key} -> {res.get('output_dir')}")
            else:
                print(f"[SKIP] {key} -> {res.get('reason')}")
        except Exception as e:
            results.append({"key": key, "status": "error", "reason": str(e)})
            print(f"[ERR] {key}: {e}")

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "manifest_path": str(manifest_path),
        "contracts_path": str(Path(args.contracts).resolve()),
        "mode": "full" if args.full else "sample",
        "sample_rows": sample_rows,
        "chunksize": None if sample_rows is not None else int(args.chunksize),
        "results": results,
    }
    (reports_dir / "_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote: {reports_dir / '_SUMMARY.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
