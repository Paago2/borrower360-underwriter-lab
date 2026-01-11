from fastapi import APIRouter
from pathlib import Path

from app.core.config import project_root
from app.services.dataset_registry import load_registry, dataset_exists

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/datasets")
def list_datasets():
    root = project_root()
    manifest = root / "configs" / "datasets.yaml"
    registry = load_registry(manifest, root)

    items = []
    for k, ds in registry.items():
        items.append(
            {
                "key": k,
                "name": ds.name,
                "type": ds.type,
                "path": ds.path,
                "exists": dataset_exists(ds),
            }
        )
    return {"count": len(items), "datasets": items}
