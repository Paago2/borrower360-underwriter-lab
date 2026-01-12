from __future__ import annotations
from pathlib import Path
from app.core.paths import project_root

root = project_root()
manifest_path: Path = root / "configs" / "datasets.yaml"
