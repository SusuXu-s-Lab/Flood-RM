

# Artifact IO, parsing, stable IDs, and provenance

"""Shared Grid Dataset kernel: IO, value parsing, Stable Grid IDs, provenance.

The primitives every Grid Dataset Stage leans on live here so the stages never
reach sideways into each other for a float parser, a CSV reader, or a Stable
Grid ID. Kept dependency-light (stdlib only) so importing it is cheap.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

# Repo root, computed locally so artifact helpers stay cheap to import.
# src/power/artifacts.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Stable Grid ID namespace (CONTEXT.md: "<location>:*", e.g. "marshfield:*").
# Single Study Location for now; becomes location-config driven when the package
# stops hardcoding Marshfield as the default.
SANDBOX_ID = "marshfield"
PROTOCOL_VERSION = "v0.1"


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
    with path.open(newline="", encoding="utf-8") as fh:
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


# ---------------------------------------------------------------------------
# Value parsing — coerce Asset Registry cells without scattering try/except.
# ---------------------------------------------------------------------------

def parse_float(value: Any, default: float | None = None) -> float | None:
    """Coerce a registry cell to float, returning ``default`` when blank/invalid."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int | None = None) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return default
    return int(parsed)


def present(value: Any) -> bool:
    """False for None, NaN, empty string, or the literal text ``nan``."""
    if value is None:
        return False
    try:
        if isinstance(value, float) and math.isnan(value):
            return False
    except TypeError:
        pass
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def finite_float(value: Any) -> float | None:
    """``parse_float`` plus a ``math.isfinite`` guard (rejects inf/nan)."""
    out = parse_float(value)
    return out if out is not None and math.isfinite(out) else None


def finite_lon_lat(lon: float | None, lat: float | None) -> bool:
    """True when both coordinates are present and within valid WGS84 ranges."""
    return (
        lon is not None
        and lat is not None
        and -180.0 <= lon <= 180.0
        and -90.0 <= lat <= 90.0
    )


# ---------------------------------------------------------------------------
# Stable Grid IDs — deterministic, namespaced identifiers (CONTEXT.md).
# ---------------------------------------------------------------------------

def slug(value: str) -> str:
    """Normalize a label into a Stable Grid ID token."""
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "unknown"


def stable_token(*parts: object, max_len: int = 48) -> str:
    """Slug of the joined parts plus a short SHA1 suffix for collision safety."""
    raw = "_".join(str(part) for part in parts if part is not None)
    token = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{token[:max_len].strip('_')}_{digest}"


def stable_asset_id(source_table: str, source_name: str) -> str:
    return f"{SANDBOX_ID}:asset:{slug(source_table)}:{slug(source_name)}"


def stable_control_unit_id(feeder_id: str) -> str:
    return f"{SANDBOX_ID}:control_unit:feeder:{slug(feeder_id)}"


# ---------------------------------------------------------------------------
# Provenance — git state and validation-report accumulation.
# ---------------------------------------------------------------------------

def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=_REPO_ROOT, text=True).strip()
        except Exception:
            return ""

    status = run(["git", "status", "--short"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def validation_error(report: dict[str, Any], message: str) -> None:
    report["errors"].append(message)


# Power-grid filesystem anchors

"""Filesystem anchors for a Study Location's power-grid dataset."""


import os
from pathlib import Path
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if (_SOURCE_ROOT / "study_location.py").exists():
    sys.path = [entry for entry in sys.path if entry != str(_SOURCE_ROOT)]
    sys.path.insert(0, str(_SOURCE_ROOT))

from study_location import define_location

# src/power/artifacts.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]


def default_location_config() -> Path:
    configured = os.environ.get("FLOOD_RM_LOCATION_CONFIG")
    if configured:
        return Path(configured)
    return REPO_ROOT / "locations" / "marshfield" / "config.yaml"


def power_grid_root(config_path=None) -> Path:
    definition = define_location(config_path or default_location_config())
    raw = definition.grid.get("power_grid_root", "data/power_grid")
    path = Path(raw)
    if path.is_absolute():
        return path
    return definition.root / path


def power_grid_path(key: str, config_path=None, default=None) -> Path:
    definition = define_location(config_path or default_location_config())
    value = definition.grid.get(key, default)
    if value is None:
        raise KeyError(f"grid path is not configured: {key}")
    path = Path(value)
    if path.is_absolute():
        return path
    return definition.root / path


POWER_GRID = power_grid_root()
