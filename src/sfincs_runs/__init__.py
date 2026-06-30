"""Shallow import root for the clean SFINCS core and runtime adapter."""

from .runtime import SfincsRuntime, build_sfincs_runtime
from .schema import (
    EventManifest,
    NativeSourceConfig,
    RuntimePaths,
    SfincsRunResult,
    SnapWaveForcing,
)
from design_events.probability import annual_rate_table, catalog_depth_probability, poisson_exceedance_probability
from .sources import create_wflow_source_contract, read_source_contract, validate_source_contract
from .forcing import (
    read_wflow_discharge,
    stage_gridded_precipitation,
)
from .coastal import build_coastal_hydrograph_from_analog, coastal_timeseries_from_catalog_row, stage_coastal_event_forcing
from .snapwave import (
    era5_to_snapwave_forcing,
    legacy_era5_spectra_to_snapwave_timeseries,
    stage_snapwave_event,
    write_snapwave_forcing_tables,
)
from .solver import run_sfincs, run_prepared_events, run_sfincs_process

__all__ = [
    "EventManifest",
    "NativeSourceConfig",
    "RuntimePaths",
    "SfincsRunResult",
    "SnapWaveForcing",
    "SfincsRuntime",
    "build_sfincs_runtime",
    "create_wflow_source_contract",
    "read_source_contract",
    "validate_source_contract",
    "read_wflow_discharge",
    "stage_gridded_precipitation",
    "build_coastal_hydrograph_from_analog",
    "coastal_timeseries_from_catalog_row",
    "stage_coastal_event_forcing",
    "era5_to_snapwave_forcing",
    "legacy_era5_spectra_to_snapwave_timeseries",
    "stage_snapwave_event",
    "write_snapwave_forcing_tables",
    "run_sfincs",
    "run_prepared_events",
    "run_sfincs_process",
    "annual_rate_table",
    "poisson_exceedance_probability",
    "catalog_depth_probability",
]
