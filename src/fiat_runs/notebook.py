"""Notebook-facing runtime helpers for FIAT risk and tide-gauge stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fiat_runs.config import fiat_paths, load_runtime


@dataclass(frozen=True)
class FiatNotebookRuntime:
    location_root: Path
    location_name: str
    repo_root: Path
    config: dict
    paths: dict
    catalog_csv: Path
    metadata_json: Path
    model_root: Path
    per_event_damage_csv: Path
    tide_gauge_root: Path
    tide_gauge_fig_root: Path


def load_fiat_notebook_runtime(location_root, *, create_tide_gauge_dirs: bool = False) -> FiatNotebookRuntime:
    """Load derived FIAT paths for a coastal Location Workspace.

    Set ``create_tide_gauge_dirs`` for the tide-gauge placement notebook; the
    FIAT risk notebook reads the same paths without creating figure folders.
    """
    location_root = Path(location_root).resolve()
    config, paths = load_runtime(location_root / "config.yaml")
    paths = fiat_paths(paths)
    tide_gauge_root = paths["location_data_root"] / "sfincs" / "tide_gauges"
    tide_gauge_fig_root = tide_gauge_root / "figures"
    if create_tide_gauge_dirs:
        tide_gauge_root.mkdir(parents=True, exist_ok=True)
        tide_gauge_fig_root.mkdir(parents=True, exist_ok=True)
    return FiatNotebookRuntime(
        location_root=location_root,
        location_name=location_root.name,
        repo_root=location_root.parents[1],
        config=config,
        paths=paths,
        catalog_csv=paths["design_outputs_root"] / "catalog" / "event_catalog.csv",
        metadata_json=paths["design_outputs_root"] / "catalog" / "catalog_risk_metadata.json",
        model_root=paths["fiat_model_root"],
        per_event_damage_csv=paths["fiat_risk_root"] / "per_event_damage.csv",
        tide_gauge_root=tide_gauge_root,
        tide_gauge_fig_root=tide_gauge_fig_root,
    )


# Compact notebook-facing workflow verbs.
load_runtime = load_fiat_notebook_runtime

from fiat_runs._env import fiat_env_available as env_ready
from fiat_runs.build_model import apply_dem_ground_elevation as apply_ground
from fiat_runs.build_model import build_fiat_model as build_model
from fiat_runs.build_model import fiat_model_is_built as model_ready
from fiat_runs.diagnostics import aggregate_building_risk as building_risk
from fiat_runs.diagnostics import damage_by_depth_band as damage_by_depth
from fiat_runs.diagnostics import damage_by_occupancy as damage_by_use
from fiat_runs.diagnostics import event_damage_summary as damage_summary
from fiat_runs.diagnostics import load_event_damage as event_damage
from fiat_runs.diagnostics import plot_building_risk as plot_risk
from fiat_runs.diagnostics import top_damaged_assets as top_assets
from fiat_runs.risk import damage_exceedance_curve as exceedance
from fiat_runs.risk_native import run_native_rp_risk as run_rp_risk
from fiat_runs.risk_native import select_rp_representatives as select_rp
from fiat_runs.run import run_fiat_event as run_event
from fiat_runs.validate import run_historical_validation as validate_history
from sfincs_runs.tide_gauges import candidate_event_response_table as response_table
from sfincs_runs.tide_gauges import candidate_points_from_building_risk as risk_candidates
from sfincs_runs.tide_gauges import candidate_points_from_runup_transects as runup_candidates
from sfincs_runs.tide_gauges import greedy_sensor_selection as select_sensors
from sfincs_runs.tide_gauges import load_runup_transects as load_transects
from sfincs_runs.tide_gauges import mark_selected_candidates as mark_selected
from sfincs_runs.tide_gauges import plot_candidate_damage_response as plot_response
from sfincs_runs.tide_gauges import plot_selected_sensor_network as plot_network
from sfincs_runs.tide_gauges import sample_sfincs_at_candidates as sample_candidates
from sfincs_runs.tide_gauges import score_sensor_candidates as score_candidates
