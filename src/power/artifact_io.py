"""Shared IO helpers for canonical Marshfield sandbox artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "Canonical sandbox artifact IO requires pyarrow. "
            "Install project dependencies first, then rerun this workflow."
        ) from exc
    return pa, pq


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_debug_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            out = {}
            for field in fields:
                value = row.get(field)
                out[field] = json.dumps(value, sort_keys=True) if isinstance(value, list) else value
            writer.writerow(out)


def write_parquet(path: Path, rows: list[dict[str, Any]], schema: Any) -> None:
    pa, pq = require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        field.name: [row.get(field.name) for row in rows]
        for field in schema
    }
    table = pa.table(columns, schema=schema)
    pq.write_table(table, path)


def read_parquet(path: Path) -> list[dict[str, Any]]:
    _, pq = require_pyarrow()
    return pq.read_table(path).to_pylist()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def maybe_sha256(path: Path) -> str | None:
    return sha256(path) if path.exists() else None


def short_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:8]


def count_by(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
