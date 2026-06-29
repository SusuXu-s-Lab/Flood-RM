"""Load real driver records into aligned Series for paired-observation assembly.

The location-specific data step that feeds `build_paired_observations`: read the
collected source records (USGS discharge, AORC SST basin-mean storm depths, NWM
antecedent soil moisture, CORA water level) into per-driver pandas Series on a
DatetimeIndex, collapsing multi-point / multi-site / multi-layer records to one value
per timestamp. The downstream copula stages are source-agnostic, so all location
coupling lives here in the per-driver record specs.

A record spec is ``{"path", "time_column", "value_column", "aggregate"?, "group_column"?}``.
``aggregate`` (e.g. "mean", "max") collapses duplicate timestamps from gridded/multi-site
records; ``group_column`` optionally restricts to selected ids (e.g. a frequency-basis gage).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from design_events.records import (
    build_paired_observations,
    calibrate_threshold_for_rate,
    declustered_pot_peaks,
    load_driver_series,
    member_library_from_records,
)

# Source-record schemas (see locations/greensboro/data/sources/...). AORC SST stores per-storm
# basin-mean depths (sparse storm-event series), the rainfall conditioning record; NWM soil
# moisture and USGS discharge are multi-row per timestamp and aggregated to one value per time.
def _dependence(config):
    return (config.get("event_catalog", {}) or {}).get("dependence", {}) or {}


def record_specs_from_config(config):
    """Per-driver record specs declared in ``event_catalog.dependence.driver_records``.

    Location specifics (which file, which columns) live in each location's config, not in
    code, so the same notebook/pipeline is modular across locations.
    """
    specs = _dependence(config).get("driver_records")
    if not specs:
        raise ValueError("config event_catalog.dependence.driver_records is required for copula_joint")
    return specs


def cooccurrence_params_from_config(config):
    """Two-sided POT co-occurrence sampling parameters from config (with defaults).

    ``target_rate_per_year`` calibrates each conditioning threshold to a target exceedance
    rate instead of a fixed quantile; ``condition_on`` restricts conditioning
    to the extreme forcing drivers (an antecedent state is paired but never conditioned).
    """
    cooccurrence = _dependence(config).get("cooccurrence", {}) or {}
    params = {
        "threshold_quantiles": cooccurrence.get("threshold_quantile", 0.95),
        "decluster_window_hours": float(cooccurrence.get("decluster_window_hours", 120.0)),
        "pairing_window_hours": float(cooccurrence.get("pairing_window_hours", 72.0)),
    }
    if cooccurrence.get("target_rate_per_year") is not None:
        params["target_rate_per_year"] = float(cooccurrence["target_rate_per_year"])
    if cooccurrence.get("condition_on"):
        params["condition_on"] = list(cooccurrence["condition_on"])
    return params


def _resolve(path, location_root):
    path = Path(path)
    if path.is_absolute() or location_root is None:
        return path
    return Path(location_root) / path


def assemble_paired_observations(record_specs, driver_vector, *, location_root=None, sites=None, **paired_kwargs):
    """Load the records for ``driver_vector`` and build the two-sided POT co-occurrence sample."""
    specs = {driver: record_specs[driver] for driver in driver_vector}
    series = load_driver_series(specs, location_root=location_root, sites=sites)
    return build_paired_observations(series, driver_names=list(driver_vector), **paired_kwargs)


def assemble_paired_observations_from_config(config, *, location_root=None, sites=None, **overrides):
    """Config-driven paired-observation assembly — modular across locations.

    Reads the driver vector, record specs, and co-occurrence parameters from
    ``event_catalog.dependence`` so the calling notebook carries no location specifics.
    """
    driver_vector = list(_dependence(config).get("driver_vector") or [])
    if not driver_vector:
        raise ValueError("config event_catalog.dependence.driver_vector is required")
    params = {**cooccurrence_params_from_config(config), **overrides}
    return assemble_paired_observations(
        record_specs_from_config(config), driver_vector, location_root=location_root, sites=sites, **params
    )


def build_member_libraries(config, *, location_root=None):
    """Build the realization member library for each driver, per config — modular.

    ``event_catalog.dependence.member_libraries[driver].from`` selects ``member_table``
    (a curated forcing-member CSV with field pointers, e.g. AORC SST event windows) or
    ``records`` (built per-timestamp from the driver records via ``member_library_from_records``).
    """
    dependence = _dependence(config)
    driver_vector = list(dependence.get("driver_vector") or [])
    member_cfg = dependence.get("member_libraries", {}) or {}
    forcing = (config.get("event_catalog", {}) or {}).get("forcing_members", {}) or {}
    record_specs = dependence.get("driver_records", {}) or {}

    libraries = {}
    for driver in driver_vector:
        spec = member_cfg.get(driver, {}) or {}
        source = spec.get("from", "member_table")
        if source == "member_table":
            if driver not in forcing:
                raise ValueError(f"member_table source for {driver} needs event_catalog.forcing_members[{driver!r}]")
            libraries[driver] = pd.read_csv(_resolve(forcing[driver], location_root))
        elif source == "records":
            if driver not in record_specs:
                raise ValueError(f"records source for {driver} needs event_catalog.dependence.driver_records[{driver!r}]")
            record_spec = record_specs[driver]
            index_column = spec.get("index_column", f"{driver}_mean")
            member_file = str(_resolve(record_spec["path"], location_root))
            decluster_window = spec.get("decluster_window_hours")
            if decluster_window:
                # Event-based members (e.g. coastal water level): declustered POT peaks,
                # not one member per timestamp.
                series = load_driver_series({driver: record_spec}, location_root=location_root)[driver]
                if spec.get("target_rate_per_year") is not None:
                    _, peaks, _ = calibrate_threshold_for_rate(
                        series,
                        float(spec["target_rate_per_year"]),
                        min_separation_hours=float(decluster_window),
                    )
                else:
                    peaks = declustered_pot_peaks(
                        series, threshold_quantile=float(spec.get("threshold_quantile", 0.98)),
                        min_separation_hours=float(decluster_window),
                    )
                if driver == "coastal_water_level" and config.get("coastal_waves", False) and not peaks.empty:
                    # Same-analog wave coupling needs the coastal analog hour inside the ERA5 wave file.
                    collection_cfg = config.get("collection", {})
                    wave_cfg = collection_cfg.get("era5_waves", {})
                    wave_start_value = wave_cfg.get("start_date") or collection_cfg.get("start")
                    wave_end_value = wave_cfg.get("end_date") or collection_cfg.get("end")
                    if wave_start_value or wave_end_value:
                        wave_start = pd.Timestamp(wave_start_value) if wave_start_value else None
                        wave_end = pd.Timestamp(wave_end_value) if wave_end_value else None
                        if isinstance(wave_end_value, str) and len(wave_end_value) == 10:
                            wave_end = wave_end + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
                        half_window = float(
                            config.get("design_events", {}).get(
                                "tide_resolving_half_window_hours",
                                config.get("resilience_stress_training", {})
                                .get("compound_pairing", {})
                                .get("real_event_window_hours", 72.0),
                            )
                        )
                        peak_times = pd.to_datetime(peaks["time"], errors="coerce")
                        keep = peak_times.notna()
                        if wave_start is not None:
                            keep &= peak_times - pd.Timedelta(hours=half_window) >= wave_start
                        if wave_end is not None:
                            keep &= peak_times + pd.Timedelta(hours=half_window) <= wave_end
                        peaks = peaks.loc[keep].reset_index(drop=True)
                        if peaks.empty:
                            raise RuntimeError(
                                "no coastal water-level members overlap the configured ERA5 wave collection window"
                            )
                peak_times = pd.to_datetime(peaks["time"])
                libraries[driver] = pd.DataFrame({
                    "member_id": driver + "_" + peak_times.dt.strftime("%Y%m%dT%H%M%S"),
                    "member_file": member_file,
                    "time": peak_times.dt.strftime("%Y-%m-%dT%H:%M:%S"),
                    index_column: peaks["value"].to_numpy(dtype=float),
                })
            else:
                records = pd.read_csv(_resolve(record_spec["path"], location_root))
                libraries[driver] = member_library_from_records(
                    records,
                    value_column=record_spec["value_column"],
                    time_column=record_spec["time_column"],
                    index_column=index_column,
                    aggregate=spec.get("aggregate", record_spec.get("aggregate", "mean")),
                    id_prefix=driver,
                    member_file=member_file,
                )
        else:
            raise ValueError(f"unknown member_libraries source {source!r} for {driver}")
    return libraries
