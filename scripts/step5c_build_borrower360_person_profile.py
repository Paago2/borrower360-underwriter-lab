from __future__ import annotations

import argparse
import hashlib
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from app.services.dataset_loader import load_dataset_df
from app.services.dataset_writer import write_parquet_dataset


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_project_root() -> Path:
    import os

    pr = os.environ.get("PROJECT_ROOT")
    if pr:
        return Path(pr).resolve()
    return Path(__file__).resolve().parents[1]


def _load_edges_prefer_clean(project_root: Path) -> tuple[pd.DataFrame, str]:
    """
    Enterprise rule:
      Prefer post-QA edges if available; fallback to raw edges.
    """
    try:
        edges_clean, _ = load_dataset_df(project_root, "edges_person_sanctions_clean", max_parts=None, max_rows=None)
        return edges_clean, "edges_person_sanctions_clean"
    except FileNotFoundError:
        edges_raw, _ = load_dataset_df(project_root, "edges_person_sanctions", max_parts=None, max_rows=None)
        return edges_raw, "edges_person_sanctions"


def _stable_int_seed(*parts: object) -> int:
    """
    Deterministic seed from input parts. Stable across runs and machines.
    """
    s = "|".join("" if p is None else str(p) for p in parts)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _is_blank(x: object) -> bool:
    if x is None:
        return True
    if pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "<na>"}


# Lightweight plausible ZIP ranges by state (demo-realistic).
# Not exhaustive. If a state isn't listed, we fallback to 10000-99999.
_STATE_ZIP_RANGES: dict[str, tuple[int, int]] = {
    "NJ": (7000, 8999),
    "CA": (90000, 96199),
    "PA": (15000, 19699),
    "HI": (96700, 96899),
    "WI": (53000, 54999),
    "NY": (10000, 14999),
    "TX": (75000, 79999),
    "FL": (32000, 34999),
    "IL": (60000, 62999),
    "GA": (30000, 31999),
    "MA": (1000, 2799),
    "WA": (98000, 99499),
    "VA": (20100, 24699),
    "MD": (20600, 21999),
    "NC": (27000, 28999),
    "SC": (29000, 29999),
    "OH": (43000, 45999),
    "MI": (48000, 49999),
    "AZ": (85000, 86599),
    "CO": (80000, 81699),
}


def synth_zip(person_id: int, state: str | None) -> str:
    st = (state or "").strip().upper()
    lo, hi = _STATE_ZIP_RANGES.get(st, (10000, 99999))
    rng = random.Random(_stable_int_seed("zip", person_id, st))
    return f"{rng.randint(lo, hi):05d}"


def synth_birth_date(person_id: int) -> str:
    """
    Returns ISO date string YYYY-MM-DD (stable per person_id).
    """
    start = date(1960, 1, 1)
    end = date(2002, 12, 31)
    span_days = (end - start).days
    rng = random.Random(_stable_int_seed("birth_date", person_id))
    d = start + timedelta(days=rng.randint(0, span_days))
    return d.isoformat()


def synth_ssn_last4(person_id: int) -> str:
    """
    Synthetic last-4. Not a real SSN. Stable per person_id.
    """
    rng = random.Random(_stable_int_seed("ssn_last4", person_id))
    return f"{rng.randint(0, 9999):04d}"


def synth_email_hash(person_id: int, name_norm: str | None) -> str:
    """
    Synthetic email hash (not an email). Stable and privacy-safe.
    """
    base = f"email|{person_id}|{(name_norm or '').strip().upper()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def synth_phone_hash(person_id: int) -> str:
    """
    Synthetic phone hash (not a phone number). Stable and privacy-safe.
    """
    base = f"phone|{person_id}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output dataset folder if exists")
    args = ap.parse_args()

    project_root = infer_project_root()

    # Load people
    person, _ = load_dataset_df(project_root, "entities_person", max_parts=None, max_rows=None)

    # Load edges (prefer clean)
    edges, edges_key_used = _load_edges_prefer_clean(project_root)

    # Ensure expected columns exist even if edges empty
    if edges is None or len(edges) == 0:
        edges = pd.DataFrame(columns=["person_id", "watchlist_id", "match_score", "source_dataset_wl"])

    # Aggregate sanctions signals per person
    if len(edges) > 0:
        # guard column presence
        if "match_score" not in edges.columns:
            edges["match_score"] = 0
        if "source_dataset_wl" not in edges.columns:
            edges["source_dataset_wl"] = ""

        agg = (
            edges.groupby("person_id", dropna=False)
            .agg(
                matched_watchlist_count=("watchlist_id", "nunique"),
                max_match_score=("match_score", "max"),
                matched_sources=("source_dataset_wl", lambda s: ",".join(sorted(set([x for x in s.dropna().astype(str) if x])))),
            )
            .reset_index()
        )
    else:
        agg = pd.DataFrame(
            {
                "person_id": person["person_id"],
                "matched_watchlist_count": 0,
                "max_match_score": 0,
                "matched_sources": "",
            }
        )

    profile_df = person.merge(agg, on="person_id", how="left")

    # --- Enterprise realism: deterministic corroboration signals ---
    # Ensure columns exist
    for col in ["birth_date", "zip", "state", "name_norm"]:
        if col not in profile_df.columns:
            profile_df[col] = ""

    # Fill missing birth_date / zip deterministically (stable)
    profile_df["birth_date"] = profile_df.apply(
        lambda r: synth_birth_date(int(r["person_id"])) if _is_blank(r.get("birth_date")) else str(r.get("birth_date")).strip(),
        axis=1,
    )
    profile_df["zip"] = profile_df.apply(
        lambda r: synth_zip(int(r["person_id"]), r.get("state")) if _is_blank(r.get("zip")) else str(r.get("zip")).strip(),
        axis=1,
    )

    # Add synthetic identity corroboration fields
    if "ssn_last4" not in profile_df.columns:
        profile_df["ssn_last4"] = profile_df["person_id"].apply(lambda pid: synth_ssn_last4(int(pid)))

    if "email_hash" not in profile_df.columns:
        profile_df["email_hash"] = profile_df.apply(
            lambda r: synth_email_hash(int(r["person_id"]), r.get("name_norm")),
            axis=1,
        )

    if "phone_hash" not in profile_df.columns:
        profile_df["phone_hash"] = profile_df["person_id"].apply(lambda pid: synth_phone_hash(int(pid)))

    # Enterprise-friendly flags
    profile_df["matched_watchlist_count"] = profile_df["matched_watchlist_count"].fillna(0).astype("int64")
    profile_df["max_match_score"] = profile_df["max_match_score"].fillna(0).astype("float64")
    profile_df["matched_sources"] = profile_df["matched_sources"].fillna("").astype("string")

    profile_df["sanctions_flag"] = profile_df["matched_watchlist_count"] > 0

    profile_df["risk_tier"] = pd.cut(
        profile_df["max_match_score"],
        bins=[-1, 0, 89, 94, 100],
        labels=["none", "low", "medium", "high"],
    ).astype("string")

    # Write as enterprise folder dataset
    out_dir = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/04_borrower360",
        dataset_key="borrower360_person_profile",
        df=profile_df,
        meta={
            "producer_step": "step5c_build_borrower360_person_profile",
            "generated_at_utc": utc_now_iso(),
            "inputs": [
                "data/03_entities/entities_person",
                f"data/03_entities/{edges_key_used}",
            ],
            "counts": {
                "person_rows": int(len(person)),
                "profile_rows": int(len(profile_df)),
                "edges_rows": int(len(edges)),
            },
            "edges_used": edges_key_used,
        },
        overwrite=bool(args.overwrite),
    )

    print("[OK] Step5C complete (enterprise dataset written)")
    print("edges_used:", edges_key_used)
    print("output_dir:", out_dir)
    print("profile rows:", len(profile_df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
