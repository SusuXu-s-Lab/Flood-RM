"""Runtime adapter from Flood-RM Location Configuration to the clean power core."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paths import resolve_location_path
from study_location import LocationDefinition

from .core import CasePaths
from .model import DistributionCase


@dataclass(frozen=True)
class PowerRuntime:
    """Notebook-compatible power runtime projected from a Location Definition."""

    definition: LocationDefinition
    location_root: Path
    location_name: str
    config: dict[str, Any]
    grid_config: dict[str, Any]
    paths: CasePaths
    case: DistributionCase


def build_power_runtime(definition: LocationDefinition) -> PowerRuntime:
    """Build a distribution-grid runtime without re-reading location YAML."""

    config = deepcopy(definition.config)
    grid = deepcopy(definition.grid)
    root = definition.root
    power_grid = resolve_location_path(root, grid.get("power_grid_root", "data/power_grid"))
    paths = CasePaths(
        root=root,
        power_grid=power_grid,
        opendss=_grid_path(root, grid, "opendss_root", power_grid / "derived_opendss"),
        registry=_grid_path(root, grid, "asset_registry", power_grid / "asset_registry"),
        augmented=_grid_path(root, grid, "augmented_artifacts", power_grid / "augmented"),
        onm=_grid_path(root, grid, "onm_export", power_grid / "onm_export"),
        reports=resolve_location_path(root, grid.get("reports", "outputs/validation_audit")),
        figures=_grid_path(root, grid, "figures", power_grid / "figures"),
    )
    case = DistributionCase(paths=paths, location_id=definition.name, config=config)
    return PowerRuntime(
        definition=definition,
        location_root=root,
        location_name=definition.name,
        config=config,
        grid_config=grid,
        paths=paths,
        case=case,
    )


def _grid_path(root: Path, grid: dict[str, Any], key: str, default: str | Path) -> Path:
    return resolve_location_path(root, grid.get(key, default))
