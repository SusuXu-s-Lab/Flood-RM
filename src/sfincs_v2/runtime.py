"""Runtime adapter from Flood-RM Location Configuration to the clean SFINCS core."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from study_location import LocationDefinition


@dataclass(frozen=True)
class SfincsRuntime:
    """Compatibility projection of the notebook-facing SFINCS runtime."""

    definition: LocationDefinition
    location_root: Path
    location_name: str
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

    config = _apply_inland_runtime_defaults(deepcopy(definition.config))
    paths = _build_paths(definition.root, definition.name, config)
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
        base_model = _location_path(
            definition.root,
            quadtree_cfg.get("base_model_root", "data/sfincs/base_quadtree_snapwave"),
        )
    if create_base_model_dir:
        base_model.mkdir(parents=True, exist_ok=True)

    design_outputs = paths["design_outputs_root"]
    return SfincsRuntime(
        definition=definition,
        location_root=definition.root,
        location_name=definition.name,
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


def _build_paths(root: Path, name: str, config: dict[str, Any]) -> dict[str, Path | str]:
    path_cfg = config.get("paths") or {}
    location_data_root = _location_path(root, path_cfg.get("data_root", "data"))
    paths = {
        "location_name": name,
        "location_root": root,
        "location_data_root": location_data_root,
        "location_config_path": root / "config.yaml",
        "project_config_path": root / "config.yaml",
        "root": location_data_root / "sfincs",
        "outputs_root": _location_path(root, path_cfg.get("sfincs_outputs_root", "data/sfincs")),
        "inputs_root": _location_path(root, path_cfg.get("static_inputs_root", "data/static")),
        "data_catalog": _location_path(root, path_cfg.get("data_catalog", "data/static/data_catalogue.yaml")),
        "static_root": _location_path(root, path_cfg.get("static_root", "data/static/processed")),
        "raw_root": _location_path(root, path_cfg.get("raw_root", "data/static/raw")),
        "observations_root": _location_path(root, path_cfg.get("observations_root", "data/sources")),
        "base_model_root": _location_path(root, path_cfg.get("base_model_root", "data/sfincs/base")),
        "scenarios_root": _location_path(root, path_cfg.get("scenarios_root", "data/sfincs/scenarios")),
        "storage_root": _location_path(root, path_cfg.get("storage_root", "data/sfincs/run_outputs")),
        "run_root": _location_path(root, path_cfg.get("run_root", "data/sfincs/run_stage")),
        "stats_root": _location_path(root, path_cfg.get("stats_root", "data/sfincs/stats")),
        "design_outputs_root": _location_path(root, path_cfg.get("design_outputs_root", "data/event_catalog")),
    }
    return paths


def _apply_inland_runtime_defaults(config: dict[str, Any]) -> dict[str, Any]:
    path_defaults = {
        "sfincs_outputs_root": "data/sfincs",
        "static_inputs_root": "data/static",
        "data_catalog": "data/static/data_catalogue.yaml",
        "static_root": "data/static/processed",
        "raw_root": "data/static/raw",
        "observations_root": "data/sources",
        "scenarios_root": "data/sfincs/scenarios",
        "storage_root": "data/sfincs/run_outputs",
        "run_root": "data/sfincs/run_stage",
        "stats_root": "data/sfincs/stats",
        "base_model_root": "data/sfincs/base",
        "design_outputs_root": "data/event_catalog",
    }
    path_config = config.setdefault("paths", {})
    for name, default in path_defaults.items():
        path_config.setdefault(name, default)

    evaluation = config.setdefault("evaluation", {})
    evaluation.setdefault("asset_source", "data/static/power_grid/smart_ds_compat/assets.parquet")
    evaluation.setdefault("output_root", "data/sfincs/evaluation")
    merge = evaluation.setdefault("multi_domain_merge", {})
    merge.setdefault("method", "max_depth_per_asset")
    merge.setdefault("retain_source_domain_id", True)
    merge.setdefault("write_overlap_diagnostics", True)

    sfincs_domain_set = config.get("sfincs_domain_set")
    if isinstance(sfincs_domain_set, dict):
        sfincs_domain_set.setdefault("domain_manifest", "data/sfincs/domains/domain_set.yaml")
        sfincs_domain_set.setdefault("event_catalog_scope", "shared_across_domain_set")
        sfincs_domain_set.setdefault("evaluation_merge", "max_depth_per_asset_with_source_domain")

    wflow = config.get("wflow")
    if isinstance(wflow, dict):
        wflow.setdefault("domain_set_manifest", "data/wflow/domain_set.yaml")
        domain_set = wflow.get("domain_set")
        if isinstance(domain_set, dict):
            domain_set.setdefault("event_catalog_scope", "shared_across_domain_set")
    return config


def _location_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
