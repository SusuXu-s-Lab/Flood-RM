from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr


@dataclass(frozen=True)
class Artifact:
    """One auditable source/science artifact over a declared window."""

    source: str
    kind: str
    start: pd.Timestamp | str | None
    end: pd.Timestamp | str | None
    files: dict[str, Any]
    meta: dict[str, Any] | None = None
    status: str = "complete"

    def row(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "kind": self.kind,
            "status": self.status,
            "start": None if self.start is None else pd.Timestamp(self.start),
            "end": None if self.end is None else pd.Timestamp(self.end),
            "files": "; ".join(str(v) for v in self.files.values()),
            **(self.meta or {}),
        }


def resolve(paths: dict, value: Any, *, base: str = "location") -> Path | None:
    """Resolve a configured path without maintaining source-specific path helpers."""
    if value in (None, ""):
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    location_root = Path(paths.get("location_root") or paths.get("repo_root") or Path.cwd())
    repo_root = Path(paths.get("repo_root") or location_root)
    if p.parts and p.parts[0] in {"data", "02_flood", "01_grid", "locations"}:
        return location_root / p
    return (repo_root if base == "repo" else location_root) / p


def manifest_path(paths: dict, source: str, kind: str) -> Path:
    root = Path(paths.get("source_artifacts_root") or resolve(paths, "data/sources/source_artifacts"))
    return root / f"{source}_{kind}.json"


def write_artifact(paths: dict, artifact: Artifact) -> Path:
    root = Path(paths.get("repo_root") or paths.get("location_root") or Path.cwd())
    payload = {
        "study_location": paths.get("location_name"),
        "source": artifact.source,
        "kind": artifact.kind,
        "status": artifact.status,
        "start": None if artifact.start is None else pd.Timestamp(artifact.start).isoformat(),
        "end": None if artifact.end is None else pd.Timestamp(artifact.end).isoformat(),
        "artifacts": {k: _jsonable(v, root) for k, v in artifact.files.items()},
        "metadata": _jsonable(artifact.meta or {}, root),
    }
    path = manifest_path(paths, artifact.source, artifact.kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_artifact(paths: dict, source: str, kind: str) -> dict | None:
    path = manifest_path(paths, source, kind)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def covers(paths: dict, source: str, kind: str, start, end, *, allow_status=("complete",)) -> bool:
    manifest = read_artifact(paths, source, kind)
    if not manifest or manifest.get("status") not in set(allow_status):
        return False
    if manifest.get("metadata", {}).get("smoke") is True:
        return False
    a, b = manifest.get("start"), manifest.get("end")
    return bool(a and b and pd.Timestamp(a) <= pd.Timestamp(start) and pd.Timestamp(b) >= pd.Timestamp(end))


def nonempty(path: Any) -> bool:
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def csv_rows(path: Any) -> int:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return 0
    return int(len(pd.read_csv(p)))


def netcdf_covers(path: Any, start, end, *, time_names=("time", "valid_time")) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    with xr.open_dataset(p) as ds:
        name = next((n for n in time_names if n in ds.coords or n in ds.dims), None)
        if name is None or ds.sizes.get(name, 0) == 0:
            return False
        t = pd.to_datetime(ds[name].values)
    return pd.Timestamp(t.min()) <= pd.Timestamp(start) and pd.Timestamp(t.max()) >= pd.Timestamp(end)


def require_columns(frame: pd.DataFrame, columns: set[str] | list[str], label: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _jsonable(value: Any, root: Path):
    if isinstance(value, Path):
        try:
            return value.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return value.as_posix()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v, root) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v, root) for v in value]
    return value
