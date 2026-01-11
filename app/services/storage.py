# app/services/storage.py
from __future__ import annotations

import os
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

CHUNK_SIZE = 1024 * 1024  # 1MB


@dataclass(frozen=True)
class StorageSettings:
    backend: str  # "local" | "s3"
    project_root: Path

    # S3-only
    bucket: Optional[str] = None
    prefix: str = ""  # optional, e.g. "borrower360"

    # cache (for downloading s3 objects to local temp)
    cache_dir: Optional[Path] = None


def _norm_prefix(p: str) -> str:
    p = (p or "").strip().strip("/")
    return p


def _join_prefix(prefix: str, rel: str) -> str:
    prefix = _norm_prefix(prefix)
    rel = rel.lstrip("/")
    return f"{prefix}/{rel}" if prefix else rel


def get_storage_settings(project_root: Path) -> StorageSettings:
    backend = (os.environ.get("STORAGE_BACKEND") or "local").strip().lower()
    if backend not in {"local", "s3"}:
        backend = "local"

    if backend == "local":
        return StorageSettings(backend="local", project_root=project_root)

    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise RuntimeError("STORAGE_BACKEND=s3 but S3_BUCKET is not set")

    prefix = os.environ.get("S3_PREFIX") or ""
    cache_dir = project_root / ".cache" / "s3"
    cache_dir.mkdir(parents=True, exist_ok=True)

    return StorageSettings(
        backend="s3",
        project_root=project_root,
        bucket=bucket,
        prefix=prefix,
        cache_dir=cache_dir,
    )


class Storage:
    """
    Thin abstraction:
      - local: read from filesystem
      - s3: list + download to cache + optionally stream
    """

    def __init__(self, settings: StorageSettings):
        self.s = settings

        if self.s.backend == "s3":
            import boto3  # lazy import

            self._s3 = boto3.client("s3")
        else:
            self._s3 = None

    @property
    def backend(self) -> str:
        return self.s.backend

    def uri(self, rel_path: str) -> str:
        if self.s.backend == "local":
            return str((self.s.project_root / rel_path).resolve())
        key = _join_prefix(self.s.prefix or "", rel_path)
        return f"s3://{self.s.bucket}/{key}"

    # ----------------------
    # Listing / existence
    # ----------------------

    def exists_file(self, rel_path: str) -> bool:
        if self.s.backend == "local":
            p = (self.s.project_root / rel_path).resolve()
            return p.exists() and p.is_file()

        # s3
        key = _join_prefix(self.s.prefix or "", rel_path)
        try:
            self._s3.head_object(Bucket=self.s.bucket, Key=key)
            return True
        except Exception:
            return False

    def list_files(self, rel_dir: str, *, suffix: Optional[str] = None) -> List[str]:
        """
        Return RELATIVE paths under rel_dir (not recursive-filtered; but fine for dataset dirs).
        """
        rel_dir = rel_dir.strip("/")

        if self.s.backend == "local":
            base = (self.s.project_root / rel_dir).resolve()
            if not base.exists() or not base.is_dir():
                return []
            out: List[str] = []
            for p in sorted(base.iterdir()):
                if p.is_file():
                    if suffix and not p.name.endswith(suffix):
                        continue
                    out.append(str(p.relative_to(self.s.project_root)).replace("\\", "/"))
            return out

        # s3
        prefix = _join_prefix(self.s.prefix or "", rel_dir) + "/"
        out: List[str] = []
        token: Optional[str] = None

        while True:
            kwargs = {"Bucket": self.s.bucket, "Prefix": prefix, "MaxKeys": 1000}
            if token:
                kwargs["ContinuationToken"] = token

            resp = self._s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                # strip global prefix => relative
                if self.s.prefix:
                    gp = _norm_prefix(self.s.prefix) + "/"
                    if key.startswith(gp):
                        rel = key[len(gp) :]
                    else:
                        rel = key
                else:
                    rel = key

                # only direct children of rel_dir
                # (keeps behavior similar to local "iterdir")
                if not rel.startswith(rel_dir + "/"):
                    continue
                tail = rel[len(rel_dir) + 1 :]
                if "/" in tail:
                    continue  # skip nested files for now

                if suffix and not rel.endswith(suffix):
                    continue

                out.append(rel)

            if resp.get("IsTruncated"):
                token = resp.get("NextContinuationToken")
            else:
                break

        return sorted(out)

    # ----------------------
    # Local paths / reading
    # ----------------------

    def as_local_path(self, rel_path: str) -> Path:
        """
        local: return absolute path
        s3: download into cache and return cache path
        """
        if self.s.backend == "local":
            return (self.s.project_root / rel_path).resolve()

        # s3 download to cache
        assert self.s.cache_dir is not None
        key = _join_prefix(self.s.prefix or "", rel_path)

        cache_path = (self.s.cache_dir / self.s.bucket / key).resolve()
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.exists() and cache_path.is_file():
            return cache_path

        self._s3.download_file(self.s.bucket, key, str(cache_path))
        return cache_path

    def read_text(self, rel_path: str, *, encoding: str = "utf-8") -> str:
        p = self.as_local_path(rel_path)
        return p.read_text(encoding=encoding, errors="replace")

    def read_bytes(self, rel_path: str) -> bytes:
        p = self.as_local_path(rel_path)
        return p.read_bytes()

    def guess_media_type(self, rel_path: str) -> str:
        mt, _ = mimetypes.guess_type(rel_path)
        return mt or "application/octet-stream"

    def open_s3_stream(self, rel_path: str) -> Tuple[Iterator[bytes], str, str]:
        """
        S3-only: stream bytes without downloading full file.
        Returns: (iterator, media_type, filename)
        """
        if self.s.backend != "s3":
            raise RuntimeError("open_s3_stream called but backend is not s3")

        key = _join_prefix(self.s.prefix or "", rel_path)
        resp = self._s3.get_object(Bucket=self.s.bucket, Key=key)
        body = resp["Body"]

        media_type = resp.get("ContentType") or self.guess_media_type(rel_path)
        filename = Path(rel_path).name

        def gen() -> Iterator[bytes]:
            while True:
                chunk = body.read(CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk

        return gen(), media_type, filename
