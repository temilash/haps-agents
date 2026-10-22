"""Parquet caches tagged with a key, so a stale cache is rebuilt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

KEY_FIELD = b"cache_key"


def make_key(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def file_digest(path: Path) -> str | None:
    """Content hash for small files; size + mtime for big ones (rasters)."""
    if not path.exists():
        return None
    stat = path.stat()
    if stat.st_size > 10_000_000:
        return f"{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_keyed(df: pd.DataFrame, path: Path, key: str) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[KEY_FIELD] = key.encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(meta), path)


def read_keyed(path: Path, key: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    table = pq.read_table(path)
    if (table.schema.metadata or {}).get(KEY_FIELD, b"").decode() != key:
        return None
    return table.to_pandas()
