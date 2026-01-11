from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

@dataclass(frozen=True)
class Dataset:
    key: str
    name: str
    path: str
    type: str
    format: str | None = None
    size_estimate: str | None = None
    license_terms_notes: str | None = None

def _substitute_vars(value: str, project_root: Path) -> str:
    return (
        value.replace("${PROJECT_ROOT}", str(project_root))
             .replace("${raw_root}", str(project_root / "data" / "00_raw"))
             .replace("${curated_root}", str(project_root / "data" / "01_curated"))
    )

def load_registry(manifest_path: Path, project_root: Path) -> Dict[str, Dataset]:
    data: Dict[str, Any] = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    datasets: Dict[str, Any] = data.get("datasets", {})
    out: Dict[str, Dataset] = {}

    for key, spec in datasets.items():
        raw_path = str(spec.get("path", ""))
        resolved_path = _substitute_vars(raw_path, project_root)
        out[key] = Dataset(
            key=key,
            name=str(spec.get("name", key)),
            path=resolved_path,
            type=str(spec.get("type", "unknown")),
            format=spec.get("format"),
            size_estimate=spec.get("size_estimate"),
            license_terms_notes=spec.get("license_terms_notes"),
        )
    return out

def dataset_exists(ds: Dataset) -> bool:
    p = Path(ds.path)
    return p.exists()
