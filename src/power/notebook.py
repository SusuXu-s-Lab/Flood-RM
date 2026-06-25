"""Notebook-facing runtime helpers for the Grid Notebook Workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sfincs_runs.config import build_grid_paths, load_runtime


@dataclass(frozen=True)
class GridNotebookRuntime:
    location_root: Path
    location_name: str
    repo_root: Path
    config: dict
    paths: dict
    grid: dict


def load_grid_notebook_runtime(location_root, *, create_output_dirs: bool = True) -> GridNotebookRuntime:
    """Load Grid Dataset paths for a Location Workspace.

    When ``create_output_dirs`` is true, this helper creates the configured Grid
    Dataset output roots used throughout the Marshfield grid notebooks.
    """
    location_root = Path(location_root).resolve()
    config, paths = load_runtime(location_root / "config.yaml")
    grid = build_grid_paths(config)
    if create_output_dirs:
        for key in (
            "shift_cache",
            "opendss_root",
            "asset_registry",
            "augmented_artifacts",
            "onm_export",
            "figures",
        ):
            if key in grid:
                grid[key].mkdir(parents=True, exist_ok=True)
    return GridNotebookRuntime(
        location_root=location_root,
        location_name=location_root.name,
        repo_root=location_root.parents[1],
        config=config,
        paths=paths,
        grid=grid,
    )


# Compact notebook-facing workflow verbs.
load_runtime = load_grid_notebook_runtime

from power.audit.synthetic_validation import build_audit_summary as audit_summary
from power.audit.synthetic_validation import plot_validation_region_report_card as plot_audit
from power.audit.synthetic_validation import run_operational_validation as run_ops
from power.audit.synthetic_validation import run_statistical_validation as run_stats
from power.baseline_network.shift_equipment import build_shift_example_equipment_catalog as equipment_catalog
from power.baseline_network.source_inputs import fetch_building_parcels_in_geometry as fetch_parcels
from power.baseline_network.source_inputs import resolve_grid_source_anchors as source_anchors
from power.baseline_network.source_inputs import resolve_grid_source_area as source_area
from power.exports import build_control_sandbox_registry as control_registry
from power.exports import export_stage_a1 as export_base
from power.plotting import build_location_block_overview as block_overview
from power.plotting import build_ocean_bluff_block_detail as block_detail
from power.plotting import build_switch_line_overlay as plot_switches
from power.resilience import assemble_switch_artifact as write_switches
from power.resilience import build_layer_1_der_inventory as build_inventory
from power.resilience import build_location_load_profile_inputs as load_inputs
from power.resilience import build_ssap_components as switch_inputs
from power.resilience import build_switch_bounded_load_blocks as build_blocks
from power.resilience import derive_lateral_fuses as derive_fuses
from power.resilience import run_layer_2_reopt_sizing as size_der
from power.resilience import solve_ssap_per_feeder as solve_switches
