from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def location_path(location_root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(location_root) / path


def relative_to(path: str | Path, root: str | Path) -> str:
    path = Path(path)
    root = Path(root)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def event_catalog_path(config: dict[str, Any], location_root: str | Path, catalog_path=None) -> Path:
    if catalog_path is not None:
        return location_path(location_root, catalog_path)
    configured = (((config.get("event_catalog", {}) or {}).get("catalog", {}) or {}).get("probability_catalog"))
    if configured:
        return location_path(location_root, configured)
    return location_path(location_root, "data/event_catalog/catalog/probability_catalog.csv")
