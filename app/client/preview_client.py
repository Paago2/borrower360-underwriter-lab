from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import requests


@dataclass(frozen=True)
class PreviewClient:
    base_url: str
    api_key: str

    @property
    def headers(self) -> Dict[str, str]:
        return {"X-API-Key": self.api_key}

    def preview_dataset(
        self,
        dataset_key: str,
        *,
        max_rows: int = 50,
        columns: Optional[Sequence[str]] = None,
        max_parts: int = 1,
    ) -> Dict[str, Any]:
        params: List[Tuple[str, str]] = [("max_rows", str(max_rows)), ("max_parts", str(max_parts))]
        if columns:
            for c in columns:
                params.append(("columns", c))

        r = requests.get(f"{self.base_url}/preview/{dataset_key}", headers=self.headers, params=params)
        r.raise_for_status()
        return r.json()

    def get_record(
        self,
        dataset_key: str,
        record_id: str,
        *,
        include_annotation: bool = False,
        include_boxes: bool = False,
        include_entities: bool = False,
        include_qas: bool = False,
        include_ocr: bool = False,
        include_text: bool = False,
    ) -> Dict[str, Any]:
        rid = quote(record_id, safe="")
        params = {
            "include_annotation": str(include_annotation).lower(),
            "include_boxes": str(include_boxes).lower(),
            "include_entities": str(include_entities).lower(),
            "include_qas": str(include_qas).lower(),
            "include_ocr": str(include_ocr).lower(),
            "include_text": str(include_text).lower(),
        }
        r = requests.get(f"{self.base_url}/preview/{dataset_key}/records/{rid}", headers=self.headers, params=params)
        r.raise_for_status()
        return r.json()

    def get_image_bytes(self, dataset_key: str, record_id: str) -> bytes:
        rid = quote(record_id, safe="")
        r = requests.get(f"{self.base_url}/preview/{dataset_key}/records/{rid}/image", headers=self.headers)
        r.raise_for_status()
        return r.content


def from_env() -> PreviewClient:
    base_url = os.environ.get("BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    api_key = os.environ.get("API_KEY", "dev-preview-key")
    return PreviewClient(base_url=base_url, api_key=api_key)
