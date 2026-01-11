# app/core/config.py
from __future__ import annotations

import os
from pathlib import Path

def _resolve_project_root() -> Path:
    env = os.getenv("PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    # fallback: repo root assumed to be current working directory
    return Path.cwd().resolve()

project_root: Path = _resolve_project_root()
manifest_path: Path = project_root / "configs" / "datasets.yaml"

