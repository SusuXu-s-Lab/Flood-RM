"""Filesystem anchors for a Study Location's power-grid dataset."""

from __future__ import annotations

import os
from pathlib import Path
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if (_SOURCE_ROOT / "study_location.py").exists():
    sys.path = [entry for entry in sys.path if entry != str(_SOURCE_ROOT)]
    sys.path.insert(0, str(_SOURCE_ROOT))

from study_location import define_location

# Two levels up from src/power/ → repo root.
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
