"""Notebook-facing runtime helpers for FIAT risk and tide-gauge stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fiat_runs.config import fiat_paths, load_runtime as _load_config_runtime


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


def load_runtime(location_root, *, create_tide_gauge_dirs: bool = False) -> FiatNotebookRuntime:
    """Load derived FIAT paths for a coastal Location Workspace.

    Set ``create_tide_gauge_dirs`` for the tide-gauge placement notebook; the
    FIAT risk notebook reads the same paths without creating figure folders.
    """
    location_root = Path(location_root).resolve()
    config, paths = _load_config_runtime(location_root / "config.yaml")
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

from fiat_runs._env import env_ready
from fiat_runs.build_model import apply_ground
from fiat_runs.build_model import build_model
from fiat_runs.build_model import model_ready
from fiat_runs.diagnostics import building_risk
from fiat_runs.diagnostics import damage_by_depth
from fiat_runs.diagnostics import damage_by_use
from fiat_runs.diagnostics import damage_summary
from fiat_runs.diagnostics import event_damage
from fiat_runs.diagnostics import plot_risk
from fiat_runs.diagnostics import top_assets
from fiat_runs.risk import exceedance
from fiat_runs.risk_native import run_rp_risk
from fiat_runs.risk_native import select_rp
from fiat_runs.run import run_event
from fiat_runs.validate import validate_history
from sfincs_runs.tide_gauges import response_table
from sfincs_runs.tide_gauges import risk_candidates
from sfincs_runs.tide_gauges import runup_candidates
from sfincs_runs.tide_gauges import select_sensors
from sfincs_runs.tide_gauges import load_transects
from sfincs_runs.tide_gauges import mark_selected
from sfincs_runs.tide_gauges import plot_response
from sfincs_runs.tide_gauges import plot_network
from sfincs_runs.tide_gauges import sample_candidates
from sfincs_runs.tide_gauges import score_candidates
