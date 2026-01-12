from __future__ import annotations

from typing import List, Optional, Any, Dict

from fastapi import APIRouter, HTTPException, Query

from app.services.dataset_loader import load_dataset_df
from app.core.paths import project_root

root = project_root()

router = APIRouter(tags=["loader"])


@router.get("/load/{dataset_key}")
def load_dataset(
    dataset_key: str,
    max_parts: int = Query(1, ge=1, le=100),
    max_rows: int = Query(50, ge=1, le=10000),
    columns: Optional[List[str]] = Query(None),
) -> Dict[str, Any]:
    """
    Load a small sample from standardized (preferred) else curated parquet.
    Returns: location metadata + rows preview.
    """
    try:
        df, loc = load_dataset_df(
            project_root=project_root,
            dataset_key=dataset_key,
            columns=columns,
            max_parts=max_parts,
            max_rows=max_rows,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    # SAFE preview (avoid raw bytes dumps)
    preview = df.head(min(len(df), 5)).copy()

    if "image" in preview.columns:
        def _image_summary(x):
            try:
                b = x.get("bytes", b"") if isinstance(x, dict) else b""
                return {"bytes_len": len(b)}
            except Exception:
                return {"bytes_len": None}
        preview["image"] = preview["image"].apply(_image_summary)

    if "ground_truth" in preview.columns:
        preview["ground_truth"] = (
            preview["ground_truth"].astype("string").str.slice(0, 200).fillna("") + "..."
        )

    return {
        "dataset": dataset_key,
        "tier": loc.tier,
        "dir": str(loc.dir_path),
        "total_shards": len(loc.parquet_files),
        "shards_read": min(len(loc.parquet_files), max_parts),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": df.columns.tolist(),
        "preview": preview.to_dict(orient="records"),
    }
