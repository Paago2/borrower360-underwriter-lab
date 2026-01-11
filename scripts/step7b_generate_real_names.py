from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from app.services.dataset_writer import write_parquet_dataset


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def infer_project_root() -> Path:
    import os
    pr = os.environ.get("PROJECT_ROOT")
    if pr:
        return Path(pr).resolve()
    return Path(__file__).resolve().parents[1]


FIRST = [
    "James","Mary","John","Patricia","Robert","Jennifer","Michael","Linda","William","Elizabeth",
    "David","Barbara","Richard","Susan","Joseph","Jessica","Thomas","Sarah","Charles","Karen",
]
LAST = [
    "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
    "Hernandez","Lopez","Gonzalez","Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Martin",
]
MIDDLE = ["A.","B.","C.","D.","E.","F.","G.","H.","J.","K."]


def make_name(rng: random.Random) -> str:
    # 70%: First Last, 30%: First M. Last
    if rng.random() < 0.7:
        return f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    return f"{rng.choice(FIRST)} {rng.choice(MIDDLE)} {rng.choice(LAST)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="How many person_id rows to generate names for")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    project_root = infer_project_root()

    rows = []
    for pid in range(1, int(args.n) + 1):
        rows.append({
            "person_id": str(pid),
            "real_name": make_name(rng),
            "generated_at_utc": utc_now_iso(),
            "generator": "synthetic_v1",
        })

    df = pd.DataFrame(rows)

    out_dir = write_parquet_dataset(
        project_root=project_root,
        tier_dir="data/03_entities",
        dataset_key="person_real_names",
        df=df,
        meta={
            "producer_step": "step7b_generate_real_names",
            "seed": int(args.seed),
            "counts": {"rows": int(len(df))},
        },
        overwrite=bool(args.overwrite),
    )

    print("[OK] Step7B-1 complete (real names generated)")
    print("output_dir:", out_dir)
    print("rows:", len(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
