"""Notebook-facing runtime helpers for SFINCS build, run, and evaluation stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sfincs_runs.config import load_runtime as _load_config_runtime


@dataclass(frozen=True)
class SfincsNotebookRuntime:
    location_root: Path
    location_name: str
    repo_root: Path
    config: dict
    paths: dict
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
    wave_cfg: dict
    quadtree_cfg: dict
    snapwave_cfg: dict
    runup_cfg: dict
    hydrology_cfg: dict
    precip_cfg: dict
    infiltration_cfg: dict
    soil_cfg: dict


def load_runtime(
    location_root,
    *,
    wave: bool = False,
    create_base_model_dir: bool = True,
) -> SfincsNotebookRuntime:
    """Load derived paths for SFINCS Location Workspace notebooks."""
    location_root = Path(location_root).resolve()
    config, paths = _load_config_runtime(location_root / "config.yaml")

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
        base_model = _resolve_location_path(
            location_root,
            quadtree_cfg.get("base_model_root", "data/sfincs/base_quadtree_snapwave"),
        )
    if create_base_model_dir:
        base_model.mkdir(parents=True, exist_ok=True)

    design_outputs = paths["design_outputs_root"]
    return SfincsNotebookRuntime(
        location_root=location_root,
        location_name=location_root.name,
        repo_root=location_root.parents[1],
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


def _resolve_location_path(location_root: Path, value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else location_root / path


# Compact notebook-facing workflow verbs.

from sfincs_runs.build_base import build_domains
from sfincs_runs.build_base import create_handoffs
from sfincs_runs.build_base import set_observations
from sfincs_runs.diagnostics import plot_standard_animation as plot_animation
from sfincs_runs.diagnostics import plot_standard_forcing as plot_forcing_standard
from sfincs_runs.diagnostics import plot_wave_forcing as plot_forcing_waves
from sfincs_runs.diagnostics import plot_forcing
from sfincs_runs.diagnostics import plot_diagnostics
from sfincs_runs.diagnostics import plot_animation as plot_inland_animation
from sfincs_runs.diagnostics import plot_standard_diagnostics
from sfincs_runs.diagnostics import plot_runup
from sfincs_runs.hydrology import validate_physics
from sfincs_runs.scenarios import audit_forcing
from sfincs_runs.scenarios import build_timeseries
from sfincs_runs.scenarios import init_hydrographs
from sfincs_runs.scenarios import plan_example
from sfincs_runs.scenarios import stage_scenarios
from sfincs_runs.scenarios.event_forcing import build_event
from sfincs_runs.scenarios.event_forcing import hydrology_inputs
from sfincs_runs.scenarios.event_forcing import run_model
from sfincs_runs.scenarios.event_forcing import stage_precip
from sfincs_runs.scenarios.event_forcing import stage_run
