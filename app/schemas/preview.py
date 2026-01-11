from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class PreviewOut(BaseModel):
    key: str
    type: str
    path: str
    format: Optional[str] = None
    limit: int

    # one of these will be populated depending on type
    rows: Optional[List[Dict[str, Any]]] = None       # tabular
    lines: Optional[List[str]] = None                 # text
    files: Optional[List[str]] = None                 # image dirs, etc.
    summary: Optional[Dict[str, Any]] = None          # json/ocr/shapefile summary
