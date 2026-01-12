from __future__ import annotations

from functools import lru_cache
from typing import List

from fastapi import APIRouter, HTTPException

from app.schemas.datasets import DatasetOut
from app.services.dataset_registry import load_registry, dataset_exists
from app.core.config import manifest_path
from app.core.paths import project_root

root = project_root()

router = APIRouter(tags=["datasets"])


@lru_cache(maxsize=1)
def _registry():
    # NOTE: cached for speed. Restart server to pick up edits to configs/datasets.yaml
    return load_registry(manifest_path=manifest_path, project_root=root)


@router.get("/datasets", response_model=List[DatasetOut])
def list_datasets() -> List[DatasetOut]:
    reg = _registry()
    out: List[DatasetOut] = []
    for key, ds in reg.items():
        out.append(
            DatasetOut(
                key=ds.key,
                name=ds.name,
                path=ds.path,
                type=ds.type,
                format=ds.format,
                size_estimate=ds.size_estimate,
                license_terms_notes=ds.license_terms_notes,
                exists=dataset_exists(ds),
            )
        )
    return out


@router.get("/datasets/{key}", response_model=DatasetOut)
def get_dataset(key: str) -> DatasetOut:
    reg = _registry()
    if key not in reg:
        raise HTTPException(status_code=404, detail=f"Unknown dataset key: {key}")
    ds = reg[key]
    return DatasetOut(
        key=ds.key,
        name=ds.name,
        path=ds.path,
        type=ds.type,
        format=ds.format,
        size_estimate=ds.size_estimate,
        license_terms_notes=ds.license_terms_notes,
        exists=dataset_exists(ds),
    )
