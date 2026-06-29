"""Minimal HydroMT-SFINCS native event runtime.

The package intentionally keeps compatibility layers, plotting, notebooks, and legacy
handoff-placement algorithms out of the runtime core.  HydroMT-SFINCS components own
SFINCS files; Wflow supplies discharge at SFINCS-native ``src.name``; SnapWave event
forcing is staged against native SnapWave boundary points.
"""

from .schema import (
    EventManifest,
    NativeSourceConfig,
    RuntimePaths,
    SfincsRunResult,
    SnapWaveForcing,
)
from .runtime import load_config, paths_from_config
from .build import build_from_steps, apply_native_infiltration, apply_native_structures
from .sources import create_wflow_source_contract, read_source_contract, validate_source_contract
from .forcing import (
    read_wflow_discharge,
    stage_gridded_precipitation,
    stage_initial_condition,
    stage_inland_event_forcing,
    stage_water_level,
)
from .coastal import build_coastal_hydrograph_from_analog, stage_coastal_event_forcing
from .snapwave import era5_to_snapwave_forcing, stage_snapwave_event, write_snapwave_forcing_tables
from .solver import run_sfincs, run_prepared_events
from .pipeline import run_inland_event_pipeline
from .probability import annual_rate_table, poisson_exceedance_probability, catalog_depth_probability
from .audit import audit_run_folder

__all__ = [
    "EventManifest",
    "NativeSourceConfig",
    "RuntimePaths",
    "SfincsRunResult",
    "SnapWaveForcing",
    "load_config",
    "paths_from_config",
    "build_from_steps",
    "apply_native_infiltration",
    "apply_native_structures",
    "create_wflow_source_contract",
    "read_source_contract",
    "validate_source_contract",
    "read_wflow_discharge",
    "stage_gridded_precipitation",
    "stage_initial_condition",
    "stage_inland_event_forcing",
    "stage_water_level",
    "build_coastal_hydrograph_from_analog",
    "stage_coastal_event_forcing",
    "era5_to_snapwave_forcing",
    "stage_snapwave_event",
    "write_snapwave_forcing_tables",
    "run_sfincs",
    "run_prepared_events",
    "run_inland_event_pipeline",
    "annual_rate_table",
    "poisson_exceedance_probability",
    "catalog_depth_probability",
    "audit_run_folder",
]
