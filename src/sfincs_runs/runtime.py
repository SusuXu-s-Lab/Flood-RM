"""Runtime adapter from Flood-RM Location Configuration to the clean SFINCS core."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from location_runtime import (
    apply_inland_runtime_defaults,
    build_sfincs_paths,
    location_path,
    repo_root_for_location,
    static_sources_with_defaults,
)
from study_location import LocationDefinition


@dataclass(frozen=True)
class SfincsRuntime:
    """Compatibility projection of the notebook-facing SFINCS runtime."""

    definition: LocationDefinition
    location_root: Path
    location_name: str
    repo_root: Path
    config: dict[str, Any]
    paths: dict[str, Path | str]
    static_dir: Path
    sfincs_root: Path
    base_model: Path
    design_outputs: Path
    events_dir: Path
    dep_dir: Path
    catalog_dir: Path
    raw_root: Path
    scenarios_root: Path
    storage_root: Path
    run_root: Path
    stats_root: Path
    wave_cfg: dict[str, Any]
    quadtree_cfg: dict[str, Any]
    snapwave_cfg: dict[str, Any]
    runup_cfg: dict[str, Any]
    hydrology_cfg: dict[str, Any]
    precip_cfg: dict[str, Any]
    infiltration_cfg: dict[str, Any]
    soil_cfg: dict[str, Any]


def build_sfincs_runtime(
    definition: LocationDefinition,
    *,
    wave: bool = False,
    create_base_model_dir: bool = True,
) -> SfincsRuntime:
    """Build the Stage 1 SFINCS runtime without re-reading location YAML."""

    config = apply_inland_runtime_defaults(deepcopy(definition.config))
    config["static_sources"] = static_sources_with_defaults(config)
    repo_root = repo_root_for_location(definition.root, definition.name, fallback="parent")
    paths = build_sfincs_paths(definition.root, definition.name, config, repo_root=repo_root)
    wave_cfg = config.get("coastal_wave_coupling") or {}
    quadtree_cfg = wave_cfg.get("quadtree") or {}
    snapwave_cfg = wave_cfg.get("snapwave") or {}
    runup_cfg = wave_cfg.get("runup_gauges") or {}
    hydrology_cfg = wave_cfg.get("hydrology") or {}
    precip_cfg = hydrology_cfg.get("precipitation") or {}
    infiltration_cfg = hydrology_cfg.get("infiltration") or {}
    soil_cfg = hydrology_cfg.get("soil_moisture") or {}

    base_model = paths["base_model_root"]
    if wave:
        base_model = location_path(
            definition.root,
            quadtree_cfg.get("base_model_root", "data/sfincs/base_quadtree_snapwave"),
            repo_root=repo_root,
            location_name=definition.name,
        )
    if create_base_model_dir:
        base_model.mkdir(parents=True, exist_ok=True)

    design_outputs = paths["design_outputs_root"]
    return SfincsRuntime(
        definition=definition,
        location_root=definition.root,
        location_name=definition.name,
        repo_root=repo_root,
        config=config,
        paths=paths,
        static_dir=paths["static_root"],
        sfincs_root=paths["outputs_root"],
        base_model=base_model,
        design_outputs=design_outputs,
        events_dir=design_outputs / "events",
        dep_dir=design_outputs / "dependence",
        catalog_dir=design_outputs / "catalog",
        raw_root=paths["raw_root"],
        scenarios_root=paths["scenarios_root"],
        storage_root=paths["storage_root"],
        run_root=paths["run_root"],
        stats_root=paths["stats_root"],
        wave_cfg=wave_cfg,
        quadtree_cfg=quadtree_cfg,
        snapwave_cfg=snapwave_cfg,
        runup_cfg=runup_cfg,
        hydrology_cfg=hydrology_cfg,
        precip_cfg=precip_cfg,
        infiltration_cfg=infiltration_cfg,
        soil_cfg=soil_cfg,
    )
