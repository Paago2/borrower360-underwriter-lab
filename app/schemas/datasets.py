from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class DatasetOut(BaseModel):
    key: str
    name: str
    path: str
    type: str
    format: Optional[str] = None
    size_estimate: Optional[str] = None
    license_terms_notes: Optional[str] = None
    exists: bool
