"""Shared contracts for the artifact-first distribution-case workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class CasePaths:
    """Canonical artifact directories for one location workspace.

    The package treats artifacts, not in-memory notebook objects, as the API.
    These paths intentionally mirror HydroMT model-root behavior: every setup
    method reads and writes below one explicit root unless the caller overrides
    an input path.
    """

    root: Path
    power_grid: Path
    opendss: Path
    registry: Path
    augmented: Path
    onm: Path
    reports: Path
    figures: Path

    @classmethod
    def from_root(
        cls,
        root: str | os.PathLike[str],
        *,
        power_grid: str | os.PathLike[str] = "data/power_grid",
    ) -> "CasePaths":
        root_path = Path(root).resolve()
        pg = Path(power_grid)
        if not pg.is_absolute():
            pg = root_path / pg
        return cls(
            root=root_path,
            power_grid=pg,
            opendss=pg / "derived_opendss",
            registry=pg / "asset_registry",
            augmented=pg / "augmented",
            onm=pg / "onm_export",
            reports=root_path / "outputs" / "validation_audit",
            figures=pg / "figures",
        )


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_path(value: str | os.PathLike[str], *, base: Path | None = None) -> Path:
    path = Path(value)
    return path if path.is_absolute() or base is None else base / path


def present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def parse_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_int(value: Any, default: int | None = None) -> int | None:
    parsed = parse_float(value)
    return default if parsed is None else int(parsed)


def slug(value: Any, *, default: str = "unknown") -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return normalized or default


def safe_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_") or "unknown"


def stable_hash(payload: Any, *, n: int = 12) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:n]


def stable_seed(*parts: Any) -> int:
    token = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") & ((1 << 63) - 1)


def file_sha256(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return path


def read_csv(path: str | os.PathLike[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]], fields: Iterable[str]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
            count += 1
    return count


def read_table(path: str | os.PathLike[str]):
    import pandas as pd

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, keep_default_na=False)


def write_table(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]], *, schema: Any = None) -> Path:
    """Write CSV or Parquet from row dictionaries.

    Parquet support is optional and loaded lazily. Supplying a pyarrow schema is
    recommended for canonical artifacts, but not required for debug outputs.
    """

    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if path.suffix.lower() == ".parquet":
        if schema is None:
            pd.DataFrame(rows).to_parquet(path, index=False)
        else:
            import pyarrow as pa
            import pyarrow.parquet as pq

            columns = {field.name: [_clean_missing(row.get(field.name)) for row in rows] for field in schema}
            pq.write_table(pa.table(columns, schema=schema), path)
    else:
        pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _clean_missing(value: Any) -> Any:
    if value is None:
        return None
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def source_provenance(**items: Any) -> str:
    return json.dumps(items, sort_keys=True, default=str)


def git_info(repo_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    root = Path(repo_root or Path.cwd())

    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    status = run(["git", "status", "--short"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def manifest(
    *,
    stage: str,
    location_id: str,
    inputs: Mapping[str, str | os.PathLike[str]],
    outputs: Mapping[str, str | os.PathLike[str]],
    parameters: Mapping[str, Any] | None = None,
    repo_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    input_hashes = {
        name: {"path": str(path), "sha256": file_sha256(path) if Path(path).exists() else None}
        for name, path in sorted(inputs.items())
    }
    output_hashes = {
        name: {"path": str(path), "sha256": file_sha256(path) if Path(path).exists() else None}
        for name, path in sorted(outputs.items())
    }
    return {
        "run_id": f"{location_id}:run:{stage}:{stable_hash({'inputs': input_hashes, 'parameters': parameters or {}})}",
        "generated_at_utc": now_utc(),
        "location_id": location_id,
        "stage": stage,
        "parameters": dict(parameters or {}),
        "python": sys.version,
        "git": git_info(repo_root),
        "inputs": input_hashes,
        "outputs": output_hashes,
    }
