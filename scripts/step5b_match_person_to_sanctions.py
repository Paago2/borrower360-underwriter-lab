from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import pandas as pd

from app.services.dataset_loader import load_dataset_df
from app.services.dataset_writer import write_parquet_dataset

# Optional fuzzy matching
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


def normalize_name(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).strip().upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_count(s: str) -> int:
    if not s:
        return 0
    return len([t for t in s.split(" ") if t])


def _commonness_bucket(name_norm: str) -> str:
    """
    Very lightweight proxy for 'commonness' / ambiguity.
    Enterprise-friendly: deterministic + explainable, no external deps.
    """
    n = name_norm.strip()
    if not n:
        return "empty"
    tok = _token_count(n)
    if tok <= 1:
        return "single_token"
    # short overall names tend to be ambiguous
    if len(n) < 10:
        return "short_name"
    # many 2-token names are common (John Smith patterns)
    if tok == 2:
        return "two_token"
    return "normal"


def exact_edges(person: pd.DataFrame, watchlist: pd.DataFrame) -> pd.DataFrame:
    p = person[["person_id", "name", "name_norm", "source_dataset", "state", "zip"]].copy()
    wl = watchlist[[
        "watchlist_id", "name", "name_norm", "source_dataset", "source_uid",
        "entity_type", "program", "country", "remarks"
    ]].copy()

    m = p.merge(wl, on="name_norm", how="inner", suffixes=("_person", "_wl"))
    m["match_method"] = "exact_name_norm"
    m["match_score"] = 100
    m["name_similarity"] = 100
    m["is_fuzzy"] = False
    m["name_tokens_person"] = m["name_norm"].astype("string").map(lambda x: _token_count(str(x)))
    m["name_tokens_wl"] = m["name_norm"].astype("string").map(lambda x: _token_count(str(x)))
    m["name_commonness_bucket"] = m["name_norm"].astype("string").map(lambda x: _commonness_bucket(str(x)))

    out = m[[
        "person_id",
        "name_person", "state", "zip", "source_dataset_person",
        "watchlist_id", "source_dataset_wl", "source_uid",
        "entity_type", "program", "country", "remarks",
        "match_method", "match_score",
        "name_similarity", "is_fuzzy",
        "name_tokens_person", "name_tokens_wl",
        "name_commonness_bucket"
    ]].rename(columns={"name_person": "person_name"})

    return out


def _mk_edge_row(p: pd.Series, w: pd.Series, score: int, method: str) -> Dict:
    pn = str(p.get("name_norm", "") or "")
    wn = str(w.get("name_norm", "") or "")
    return {
        "person_id": p.get("person_id", ""),
        "person_name": p.get("name", ""),
        "state": p.get("state", ""),
        "zip": p.get("zip", ""),
        "source_dataset_person": p.get("source_dataset", ""),
        "watchlist_id": w.get("watchlist_id", ""),
        "source_dataset_wl": w.get("source_dataset", ""),
        "source_uid": w.get("source_uid", ""),
        "entity_type": w.get("entity_type", ""),
        "program": w.get("program", ""),
        "country": w.get("country", ""),
        "remarks": w.get("remarks", ""),
        "match_method": method,
        "match_score": int(score),
        "name_similarity": int(score),
        "is_fuzzy": True,
        "name_tokens_person": _token_count(pn),
        "name_tokens_wl": _token_count(wn),
        "name_commonness_bucket": _commonness_bucket(pn),
    }


def fuzzy_edges(
    person: pd.DataFrame,
    watchlist: pd.DataFrame,
    threshold: int = 90,
    top_k: int = 5,
    scorer: str = "token_sort_ratio",
) -> pd.DataFrame:
    if not HAS_RAPIDFUZZ:
        raise RuntimeError("rapidfuzz not installed. Run: pip install rapidfuzz")

    # pick scorer
    scorer_fn = fuzz.token_sort_ratio
    method = "fuzzy_token_sort_ratio"
    if scorer == "token_set_ratio":
        scorer_fn = fuzz.token_set_ratio
        method = "fuzzy_token_set_ratio"
    elif scorer == "partial_ratio":
        scorer_fn = fuzz.partial_ratio
        method = "fuzzy_partial_ratio"

    wl_unique = watchlist.dropna(subset=["name_norm"]).copy()
    wl_unique["name_norm"] = wl_unique["name_norm"].astype("string")
    wl_unique = wl_unique[wl_unique["name_norm"].str.len() > 0].copy()

    # Choices are name_norms; we keep a map to rows (can be non-unique)
    choices = wl_unique["name_norm"].tolist()
    wl_map = wl_unique.set_index("name_norm", drop=False)

    rows: List[Dict] = []

    for _, p in person.iterrows():
        q = str(p.get("name_norm", "") or "")
        if not q:
            continue

        hits: List[Tuple[str, float, int]] = process.extract(q, choices, scorer=scorer_fn, limit=int(top_k))
        for (hit_norm, score, _) in hits:
            score_i = int(score)
            if score_i < int(threshold):
                continue
            w = wl_map.loc[hit_norm]
            if isinstance(w, pd.DataFrame):
                for _, wr in w.iterrows():
                    rows.append(_mk_edge_row(p, wr, score_i, method))
            else:
                rows.append(_mk_edge_row(p, w, score_i, method))

    if not rows:
        return pd.DataFrame(columns=[
            "person_id", "person_name", "state", "zip", "source_dataset_person",
            "watchlist_id", "source_dataset_wl", "source_uid",
            "entity_type", "program", "country", "remarks",
            "match_method", "match_score",
            "name_similarity", "is_fuzzy",
            "name_tokens_person", "name_tokens_wl",
            "name_commonness_bucket"
        ])

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output dataset")
    ap.add_argument("--fuzzy", action="store_true", help="Enable fuzzy matching")
    ap.add_argument("--threshold", type=int, default=90, help="Fuzzy threshold (only used if --fuzzy)")
    ap.add_argument("--top-k", type=int, default=5, help="Top-k candidates per person (fuzzy)")
    ap.add_argument(
        "--scorer",
        type=str,
        default="token_sort_ratio",
        choices=["token_sort_ratio", "token_set_ratio", "partial_ratio"],
        help="RapidFuzz scorer",
    )
    ap.add_argument("--max-person-rows", type=int, default=None, help="Optional cap for fast runs")
    args = ap.parse_args()

    project_root = infer_project_root()

    # Load datasets via loader (enterprise folder datasets)
    person, person_loc = load_dataset_df(project_root, "entities_person", max_parts=None, max_rows=None)
    watchlist, watchlist_loc = load_dataset_df(project_root, "sanctions_watchlist", max_parts=None, max_rows=None)

    # Ensure name_norm exists
    if "name_norm" not in person.columns:
        person["name_norm"] = person["name"].map(normalize_name)

    if args.max_person_rows is not None:
        person = person.head(int(args.max_person_rows)).copy()

    edges = exact_edges(person, watchlist)

    if args.fuzzy:
        if not HAS_RAPIDFUZZ:
            raise RuntimeError("rapidfuzz not installed. Run: pip install rapidfuzz")
        fuzzy_df = fuzzy_edges(
            person,
            watchlist,
            threshold=int(args.threshold),
            top_k=int(args.top_k),
            scorer=str(args.scorer),
        )
        if len(fuzzy_df) > 0:
            edges = pd.concat([edges, fuzzy_df], ignore_index=True)

    # Dedup: keep best score per (person_id, watchlist_id)
    if len(edges) > 0:
        edges = edges.sort_values(["person_id", "watchlist_id", "match_score"], ascending=[True, True, False])
        edges = edges.drop_duplicates(["person_id", "watchlist_id"], keep="first").reset_index(drop=True)

    out_dir = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/03_entities",
        dataset_key="edges_person_sanctions",
        df=edges,
        meta={
            "producer_step": "step5b_match_person_to_sanctions",
            "inputs": [str(person_loc.dir_ref), str(watchlist_loc.dir_ref)],
            "counts": {
                "person_rows": int(len(person)),
                "watchlist_rows": int(len(watchlist)),
                "edges_rows": int(len(edges)),
            },
            "fuzzy": {
                "enabled": bool(args.fuzzy),
                "threshold": int(args.threshold),
                "top_k": int(args.top_k),
                "scorer": str(args.scorer),
                "rapidfuzz_installed": bool(HAS_RAPIDFUZZ),
            },
        },
        overwrite=bool(args.overwrite),
    )

    print("[OK] Step5B complete (enterprise dataset written)")
    print("edges rows:", len(edges))
    print("output_dir:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
