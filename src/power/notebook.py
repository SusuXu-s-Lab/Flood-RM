"""Notebook-facing runtime helpers for the Grid Notebook Workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sfincs_runs.config import build_grid_paths, load_runtime as _load_config_runtime


@dataclass(frozen=True)
class GridNotebookRuntime:
    location_root: Path
    location_name: str
    repo_root: Path
    config: dict
    paths: dict
    grid: dict


def load_runtime(location_root, *, create_output_dirs: bool = True) -> GridNotebookRuntime:
    """Load Grid Dataset paths for a Location Workspace.

    When ``create_output_dirs`` is true, this helper creates the configured Grid
    Dataset output roots used throughout the Marshfield grid notebooks.
    """
    location_root = Path(location_root).resolve()
    config, paths = _load_config_runtime(location_root / "config.yaml")
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

from power.audit.synthetic_validation import audit_summary
from power.audit.synthetic_validation import plot_audit
from power.audit.synthetic_validation import run_ops
from power.audit.synthetic_validation import run_stats
from power.baseline_network.shift_equipment import equipment_catalog
from power.baseline_network.source_inputs import fetch_parcels
from power.baseline_network.source_inputs import source_anchors
from power.baseline_network.source_inputs import source_area
from power.exports import control_registry
from power.exports import export_base
from power.plotting import block_overview
from power.plotting import block_detail
from power.plotting import plot_switches
from power.resilience import write_switches
from power.resilience import build_inventory
from power.resilience import load_inputs
from power.resilience import switch_inputs
from power.resilience import build_blocks
from power.resilience import derive_fuses
from power.resilience import size_der
from power.resilience import solve_switches
