from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from app.services.dataset_writer import write_parquet_dataset

# Optional fuzzy matching (recommended later)
try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_project_root() -> Path:
    import os
    pr = os.environ.get("PROJECT_ROOT")
    if pr:
        return Path(pr).resolve()
    return Path(__file__).resolve().parents[1]


def load_all_parquet_parts(dir_path: Path, limit_parts: Optional[int] = None) -> pd.DataFrame:
    parts = sorted(dir_path.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts found under: {dir_path}")
    if limit_parts is not None:
        parts = parts[:limit_parts]
    dfs = [pd.read_parquet(p) for p in parts]
    return pd.concat(dfs, ignore_index=True)


def normalize_name(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --------- SAFE COLUMN HELPERS ---------

def col_or_na(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if present, else a same-length Series of <NA>."""
    if col in df.columns:
        return df[col].astype("string")
    return pd.Series(pd.NA, index=df.index, dtype="string")


def col_or_empty(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if present, else a same-length Series of empty strings."""
    if col in df.columns:
        return df[col].astype("string")
    return pd.Series("", index=df.index, dtype="string")


def is_valid_source_uid(s: pd.Series) -> pd.Series:
    """
    Enterprise QA gate: remove weird UID values (e.g., '␦') that poison watchlist_id.
    Keep only simple IDs: letters/numbers/_-:. (common across sources).
    """
    x = s.astype("string").fillna("")
    x = x.str.strip()
    bad_literals = {"", "nan", "none", "null", "␦"}
    ok_literal = ~x.str.lower().isin(bad_literals)
    ok_chars = x.str.match(r"^[A-Za-z0-9_\-:\.]+$")
    return ok_literal & ok_chars


def build_watchlist(project_root: Path, *, osd_parts: int = 5) -> pd.DataFrame:
    """
    Create a unified sanctions watchlist with a stable schema.
    """
    std_root = project_root / "data" / "02_standardized"

    # 1) sdn_ofac
    sdn = load_all_parquet_parts(std_root / "sdn_ofac")
    sdn_out = pd.DataFrame({
        "source_dataset": "sdn_ofac",
        "source_uid": col_or_na(sdn, "uid"),
        "name": col_or_na(sdn, "name"),
        "entity_type": col_or_empty(sdn, "type"),
        "program": pd.Series("", index=sdn.index, dtype="string"),
        "country": col_or_empty(sdn, "country"),
        "remarks": col_or_empty(sdn, "remarks"),
    })

    # 2) consolidated_sdn
    cons = load_all_parquet_parts(std_root / "consolidated_sdn")
    cons_out = pd.DataFrame({
        "source_dataset": "consolidated_sdn",
        "source_uid": col_or_na(cons, "uid"),
        "name": col_or_na(cons, "name"),
        "entity_type": col_or_empty(cons, "entity_type"),
        "program": col_or_empty(cons, "program"),
        "country": pd.Series("", index=cons.index, dtype="string"),
        "remarks": col_or_empty(cons, "remarks"),
    })

    # 3) open_sanctions_debarment (cap parts for safety)
    osd = load_all_parquet_parts(std_root / "open_sanctions_debarment", limit_parts=int(osd_parts))
    osd_out = pd.DataFrame({
        "source_dataset": "open_sanctions_debarment",
        "source_uid": col_or_na(osd, "id"),
        "name": col_or_na(osd, "name"),
        "entity_type": col_or_empty(osd, "schema"),
        "program": col_or_empty(osd, "dataset"),
        "country": col_or_empty(osd, "countries"),
        "remarks": col_or_empty(osd, "sanctions"),
    })

    watchlist = pd.concat([sdn_out, cons_out, osd_out], ignore_index=True)

    # Normalize + IDs
    watchlist["name_norm"] = watchlist["name"].map(normalize_name)
    watchlist["source_uid"] = watchlist["source_uid"].astype("string")

    # QA gates
    watchlist = watchlist[watchlist["name_norm"].str.len() > 0].copy()
    watchlist = watchlist[is_valid_source_uid(watchlist["source_uid"])].copy()

    watchlist["watchlist_id"] = (
        watchlist["source_dataset"].astype("string") + ":" + watchlist["source_uid"].astype("string")
    )

    watchlist = watchlist.reset_index(drop=True)
    return watchlist


def build_demo_applicants() -> pd.DataFrame:
    data = [
        {"applicant_id": "A-0001", "name": "Aerocaribbean Airlines", "country": "Cuba"},
        {"applicant_id": "A-0002", "name": "Banco Nacional de Cuba", "country": "Cuba"},
        {"applicant_id": "A-0003", "name": "Some Random Clean Entity LLC", "country": "US"},
        {"applicant_id": "A-0004", "name": "Myanmar Yatai International Holding Group Co., Ltd.", "country": "MM"},
        {"applicant_id": "A-0005", "name": "Haniya Ismail Abdul Salah", "country": ""},
    ]
    df = pd.DataFrame(data)
    df["name_norm"] = df["name"].map(normalize_name)
    return df


def exact_match_edges(applicants: pd.DataFrame, watchlist: pd.DataFrame) -> pd.DataFrame:
    wl = watchlist[
        ["watchlist_id", "name", "name_norm", "source_dataset", "source_uid", "entity_type", "program", "country", "remarks"]
    ].copy()

    merged = applicants.merge(wl, on="name_norm", how="left", suffixes=("", "_wl"))
    merged = merged[merged["watchlist_id"].notna()].copy()

    merged["match_method"] = "exact_name_norm"
    merged["match_score"] = 100

    return merged[
        [
            "applicant_id", "name", "country",
            "watchlist_id", "source_dataset", "source_uid",
            "entity_type", "program", "remarks",
            "match_method", "match_score",
        ]
    ]


def fuzzy_match_edges(applicants: pd.DataFrame, watchlist: pd.DataFrame, threshold: int = 92, top_k: int = 5) -> pd.DataFrame:
    if not HAS_RAPIDFUZZ:
        raise RuntimeError("rapidfuzz not installed. Install it: pip install rapidfuzz")

    wl_unique = watchlist.drop_duplicates("name_norm").copy()
    choices = wl_unique["name_norm"].tolist()
    wl_map = wl_unique.set_index("name_norm", drop=False)

    rows = []
    for _, a in applicants.iterrows():
        q = a.get("name_norm", "")
        if not q:
            continue

        hits = process.extract(q, choices, scorer=fuzz.token_sort_ratio, limit=top_k)
        for (hit_norm, score, _) in hits:
            if score < threshold:
                continue
            w = wl_map.loc[hit_norm]
            rows.append({
                "applicant_id": a["applicant_id"],
                "name": a["name"],
                "country": a.get("country", ""),
                "watchlist_id": w["watchlist_id"],
                "source_dataset": w["source_dataset"],
                "source_uid": w["source_uid"],
                "entity_type": w["entity_type"],
                "program": w["program"],
                "remarks": w["remarks"],
                "match_method": "fuzzy_token_sort_ratio",
                "match_score": int(score),
            })

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=92, help="Fuzzy match threshold (RapidFuzz)")
    ap.add_argument("--no-fuzzy", action="store_true", help="Skip fuzzy matching")
    ap.add_argument("--osd-parts", type=int, default=5, help="How many OpenSanctions parquet parts to load (safety cap)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they exist")
    args = ap.parse_args()

    project_root = infer_project_root()

    # Build watchlist
    watchlist = build_watchlist(project_root, osd_parts=int(args.osd_parts))

    # Build search index (unique normalized names)
    search_index = watchlist[["watchlist_id", "name_norm", "name", "source_dataset", "source_uid"]].copy()
    search_index = search_index.drop_duplicates(["watchlist_id"]).reset_index(drop=True)

    # Demo applicants
    applicants = build_demo_applicants()

    # Matches
    edges = exact_match_edges(applicants, watchlist)
    if not args.no_fuzzy:
        if HAS_RAPIDFUZZ:
            fuzzy_edges = fuzzy_match_edges(applicants, watchlist, threshold=args.threshold)
            if len(fuzzy_edges) > 0:
                edges = pd.concat([edges, fuzzy_edges], ignore_index=True)
        else:
            print("[WARN] rapidfuzz not installed; skipping fuzzy matching.")

    if len(edges) > 0:
        edges = edges.sort_values(["applicant_id", "watchlist_id", "match_score"], ascending=[True, True, False])
        edges = edges.drop_duplicates(["applicant_id", "watchlist_id"], keep="first").reset_index(drop=True)

    # ---- ENTERPRISE OUTPUTS (FOLDER DATASETS) ----
    wl_dir = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/03_entities",
        dataset_key="sanctions_watchlist",
        df=watchlist,
        meta={
            "producer_step": "step4_build_entities_and_sanctions",
            "generated_at_utc": utc_now_iso(),
            "counts": {"watchlist_rows": int(len(watchlist))},
            "inputs": [
                "data/02_standardized/sdn_ofac",
                "data/02_standardized/consolidated_sdn",
                f"data/02_standardized/open_sanctions_debarment (parts={int(args.osd_parts)})",
            ],
            "quality_gates": ["drop empty name_norm", "drop invalid source_uid"],
        },
        overwrite=bool(args.overwrite),
    )

    idx_dir = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/03_entities",
        dataset_key="sanctions_search_index",
        df=search_index,
        meta={
            "producer_step": "step4_build_entities_and_sanctions",
            "generated_at_utc": utc_now_iso(),
            "counts": {"rows": int(len(search_index))},
            "inputs": ["data/03_entities/sanctions_watchlist"],
        },
        overwrite=bool(args.overwrite),
    )

    demo_dir = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/03_entities",
        dataset_key="demo_applicants",
        df=applicants,
        meta={
            "producer_step": "step4_build_entities_and_sanctions",
            "generated_at_utc": utc_now_iso(),
            "counts": {"rows": int(len(applicants))},
        },
        overwrite=bool(args.overwrite),
    )

    edges_dir = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/03_entities",
        dataset_key="edges_sanctions_matches",
        df=edges,
        meta={
            "producer_step": "step4_build_entities_and_sanctions",
            "generated_at_utc": utc_now_iso(),
            "counts": {"edges_rows": int(len(edges))},
            "fuzzy": {"enabled": (not args.no_fuzzy), "rapidfuzz_installed": HAS_RAPIDFUZZ, "threshold": int(args.threshold)},
            "inputs": [
                "data/03_entities/demo_applicants",
                "data/03_entities/sanctions_watchlist",
            ],
        },
        overwrite=bool(args.overwrite),
    )

    print("\n[OK] Step 4 complete (enterprise datasets written)")
    print("watchlist rows:", len(watchlist))
    print("edges rows:", len(edges))
    print("output_dirs:")
    print(" -", wl_dir)
    print(" -", idx_dir)
    print(" -", demo_dir)
    print(" -", edges_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
