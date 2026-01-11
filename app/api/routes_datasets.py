from __future__ import annotations

from functools import lru_cache
from typing import List

from fastapi import APIRouter, HTTPException

from app.core.config import project_root, manifest_path
from app.schemas.datasets import DatasetOut
from app.services.dataset_registry import load_registry, dataset_exists

router = APIRouter(tags=["datasets"])


@lru_cache(maxsize=1)
def _registry():
    # NOTE: cached for speed. In case of edit configs/datasets.yaml while server is running,
    # must restart (or remove lru_cache for dev).
    return load_registry(manifest_path=manifest_path, project_root=project_root)


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
