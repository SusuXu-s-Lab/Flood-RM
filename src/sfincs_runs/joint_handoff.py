from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from design_events.catalog import write_event_catalog_audit
from sfincs_runs.coastal import coastal_timeseries_from_catalog_row as build_timeseries


def write_handoff(joint_catalog, components, *, config, paths):
    """Write copula-joint coastal realizations in the SFINCS scenario-builder contract."""
    catalog = _overlay_stress_training_fields(joint_catalog, paths).reset_index(drop=True)
    scenario = paths.get("scenario", {}) or {"name": "base", "slr_offset_m": 0.0}
    scenario_name = str(scenario.get("name", "base"))
    slr_offset_m = float(scenario.get("slr_offset_m", 0.0))
    window_hours = float(
        config.get("design_events", {}).get(
            "tide_resolving_half_window_hours",
            config.get("resilience_stress_training", {})
            .get("compound_pairing", {})
            .get("real_event_window_hours", 72.0),
        )
    )

    configured_drivers = [str(driver).strip() for driver in (config.get("event_drivers") or []) if str(driver).strip()]
    infiltration_method = _configured_infiltration_method(config)

    event_ids = catalog["event_id"].astype(str).to_list()
    series_by_event = {}
    summary_rows = []
    event_rows = []
    for _, row in catalog.iterrows():
        event_id = str(row["event_id"])
        forcing = build_timeseries(
            row,
            components,
            window_hours=window_hours,
            msl_offset_m=slr_offset_m,
        )
        h = forcing["h"].dropna().astype(float)
        start_hour = float(h.index.min())
        end_hour = float(h.index.max())
        analog_time = pd.Timestamp(_required(row, "coastal_water_level_member_time"))
        analog_id = _optional(
            row,
            "coastal_water_level_member_id",
            _optional(row, "coastal_water_level_template_member_id", event_id),
        )
        scale_factor = float(_optional(row, "coastal_water_level_scale_factor", 1.0))
        absolute_peak = float(h.max())
        target_peak = float(_optional(row, "coastal_water_level", absolute_peak))
        rainfall_time = _optional_timestamp(row, "rainfall_member_time")
        # Compound lag is the within-event offset of the rainfall peak relative to the coastal
        # peak (bounded by the pairing window), carried from the catalogue's
        # rainfall_peak_offset_hours / rainfall_pairing_lag_hours / rainfall_realization_lag_hours.
        # Do NOT difference the two
        # independently sampled analog calendar times (rainfall_member_time vs the coastal
        # analog_time) -- those span the historical record (decades) and are a provenance
        # artifact, never a physical compound lag.
        rainfall_start_offset_hours = _optional(row, "rainfall_start_offset_hours", pd.NA)
        rainfall_lag_hours = _compound_lag_hours(row)
        rainfall_end_offset_hours = _optional(row, "rainfall_end_offset_hours", pd.NA)
        snapwave_start = analog_time + pd.Timedelta(hours=start_hour)
        snapwave_end = analog_time + pd.Timedelta(hours=end_hour)

        # Soil moisture is the post-sampling antecedent state: when a row carries a paired soil
        # member, advertise it in event_drivers and switch infiltration on so the scenario
        # builder restages sfincs.seff instead of running with dry initial conditions.
        soil_present = not _is_missing(_optional(row, "soil_moisture_member_id", pd.NA))
        event_drivers_value = _event_drivers_value(row, configured_drivers, soil_present)
        infiltration_treatment_value = (
            infiltration_method if soil_present else _optional(row, "infiltration_treatment", "none")
        )

        series_by_event[event_id] = h
        summary_rows.append(
            {
                "event_id": event_id,
                "sample_rp_years": _optional(row, "sample_rp_years", pd.NA),
                "sampling_region": _optional(row, "sampling_region", pd.NA),
                "sampling_weight": _optional(row, "sampling_weight", pd.NA),
                "probability_weight": _optional(row, "probability_weight", pd.NA),
                "template_id": analog_id,
                "template_peak_m": _optional(row, "coastal_water_level_template_value", target_peak),
                "template_peak_time": _format_time(analog_time),
                "tail_morph_factor": scale_factor,
                "peak": target_peak,
                "volume": float(np.nansum(np.clip(h.to_numpy(dtype=float), 0.0, None))),
                "duration_above_50pct_peak": int(np.sum(h.to_numpy(dtype=float) >= 0.5 * absolute_peak)),
                "rise_time_to_peak": abs(start_hour),
                "fall_time_from_peak": end_hour,
                "asymmetry_ratio": end_hour / abs(start_hour) if start_hour else np.nan,
                "n_secondary_peaks": pd.NA,
                "valid_start_hour": start_hour,
                "valid_end_hour": end_hour,
                "scenario_name": scenario_name,
                "slr_offset_m": slr_offset_m,
                "absolute_peak_m": absolute_peak,
            }
        )
        event_rows.append(
            {
                "event_id": event_id,
                "study_location": str(paths.get("location_name") or config.get("project", {}).get("name")),
                "event_family": _optional(row, "event_family", "copula_joint_compound"),
                "scenario_name": scenario_name,
                "sample_rp_years": _optional(row, "sample_rp_years", pd.NA),
                "severity_band": _optional(row, "severity_band", pd.NA),
                "sampling_region": _optional(row, "sampling_region", pd.NA),
                "sampling_weight": _optional(row, "sampling_weight", pd.NA),
                "probability_weight": _optional(row, "probability_weight", pd.NA),
                "event_origin": _optional(row, "event_origin", pd.NA),
                "catalog_role": _optional(row, "catalog_role", pd.NA),
                "sampling_scheme": _optional(row, "sampling_scheme", pd.NA),
                "event_set": _optional(row, "event_set", pd.NA),
                "selection_role": _optional(row, "selection_role", pd.NA),
                "selection_reason": _optional(row, "selection_reason", pd.NA),
                "benchmark_return_period_years": _optional(row, "benchmark_return_period_years", pd.NA),
                "storm_type": _optional(row, "storm_type", pd.NA),
                "event_reference_time": _format_time(analog_time),
                "coastal_source": "cora",
                "coastal_member_file": str(paths["event_members_nc"]),
                "coastal_member_id": event_id,
                "coastal_template_peak_time": _format_time(analog_time),
                "coastal_peak_m": target_peak,
                "coastal_absolute_peak_m": absolute_peak,
                "coastal_valid_start_hour": start_hour,
                "coastal_valid_end_hour": end_hour,
                "coastal_start_offset_hours": start_hour,
                "coastal_peak_offset_hours": 0.0,
                "coastal_end_offset_hours": end_hour,
                "coastal_analog_id": analog_id,
                "coastal_analog_peak_time": _format_time(analog_time),
                "coastal_water_level_scale_factor": scale_factor,
                "snapwave_source": "era5" if config.get("coastal_waves", False) else pd.NA,
                "snapwave_member_file": str(paths.get("era5_waves_nc", "")) if config.get("coastal_waves", False) else pd.NA,
                "snapwave_member_id": analog_id if config.get("coastal_waves", False) else pd.NA,
                "snapwave_valid_start_time": _format_time(snapwave_start) if config.get("coastal_waves", False) else pd.NA,
                "snapwave_valid_end_time": _format_time(snapwave_end) if config.get("coastal_waves", False) else pd.NA,
                "snapwave_pairing_policy": "same_historical_analog" if config.get("coastal_waves", False) else pd.NA,
                "wave_start_offset_hours": start_hour,
                "wave_peak_offset_hours": 0.0,
                "wave_end_offset_hours": end_hour,
                "rainfall_source": _optional(row, "rainfall_source", "aorc_sst"),
                "rainfall_member_file": _optional(row, "rainfall_member_file", pd.NA),
                "rainfall_member_id": _optional(row, "rainfall_member_id", pd.NA),
                "rainfall_member_time": _format_optional_time(rainfall_time),
                "rainfall_peak_time": _optional(row, "rainfall_peak_time", pd.NA),
                "rainfall_peak_time_source": _optional(row, "rainfall_peak_time_source", pd.NA),
                "rainfall_peak_offset_from_start_hours": _optional(row, "rainfall_peak_offset_from_start_hours", pd.NA),
                "rainfall_duration_hours": _optional(row, "rainfall_duration_hours", pd.NA),
                "rainfall_metric_mm": _optional(row, "rainfall_metric_mm", _optional(row, "rainfall", pd.NA)),
                "rainfall_scale_factor": _optional(row, "rainfall_scale_factor", 1.0),
                "rainfall_pairing_policy": _optional(row, "rainfall_pairing_policy", "copula_joint_field_preserving_analog"),
                "rainfall_pairing_seed": _optional(row, "rainfall_pairing_seed", pd.NA),
                "rainfall_pairing_reference_time": _format_time(analog_time),
                "rainfall_pairing_lag_hours": rainfall_lag_hours,
                "rainfall_start_offset_hours": rainfall_start_offset_hours,
                "rainfall_peak_offset_hours": rainfall_lag_hours,
                "rainfall_end_offset_hours": rainfall_end_offset_hours,
                "streamflow_source": pd.NA,
                "streamflow_member_file": pd.NA,
                "streamflow_member_id": pd.NA,
                "streamflow_member_time": pd.NA,
                "streamflow_pairing_policy": pd.NA,
                "streamflow_pairing_seed": pd.NA,
                "soil_moisture_source": _optional(row, "soil_moisture_source", pd.NA),
                "soil_moisture_member_file": _optional(row, "soil_moisture_member_file", pd.NA),
                "soil_moisture_member_id": _optional(row, "soil_moisture_member_id", pd.NA),
                "soil_moisture_member_time": _optional(row, "soil_moisture_member_time", pd.NA),
                "soil_moisture_pairing_policy": _optional(row, "soil_moisture_pairing_policy", pd.NA),
                "soil_moisture_pairing_seed": _optional(row, "soil_moisture_pairing_seed", pd.NA),
                "soil_moisture_pairing_reference_time": _optional(row, "soil_moisture_pairing_reference_time", pd.NA),
                "soil_moisture_pairing_lag_hours": _optional(row, "soil_moisture_pairing_lag_hours", pd.NA),
                "forcing_pairing_policy": _optional(row, "forcing_pairing_policy", "copula_joint"),
                "event_drivers": event_drivers_value,
                "infiltration_treatment": infiltration_treatment_value,
                "compound_pairing_policy": _optional(row, "compound_pairing_policy", pd.NA),
                "compound_pairing_role": _optional(row, "compound_pairing_role", pd.NA),
                "scenario_timing_edge_case": _optional(row, "scenario_timing_edge_case", pd.NA),
                "historical_conditioned_on": _optional(row, "historical_conditioned_on", pd.NA),
                "historical_event_time": _optional(row, "historical_event_time", pd.NA),
            }
        )

    axis = np.array(sorted({int(hour) for series in series_by_event.values() for hour in series.index}), dtype=int)
    matrix = np.full((len(event_ids), len(axis)), np.nan, dtype=np.float32)
    for row_idx, event_id in enumerate(event_ids):
        values = series_by_event[event_id].reindex(axis).to_numpy(dtype=float)
        matrix[row_idx, :] = values.astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "water_level_total": (("event_id", "relative_hour"), matrix),
            "valid_mask": (("event_id", "relative_hour"), np.isfinite(matrix)),
        },
        coords={"event_id": event_ids, "relative_hour": axis},
        attrs={
            "source": "Copula-joint field-preserving coastal SFINCS handoff",
            "scenario_name": scenario_name,
            "slr_offset_m": slr_offset_m,
            "units": "m",
        },
    )
    ds["water_level_total"].attrs.update(
        long_name="tide-preserving total coastal water level from copula-joint analog realization",
        units="m",
    )

    event_members_nc = Path(paths["event_members_nc"])
    event_summary_csv = Path(paths["event_summary_csv"])
    event_catalog_csv = Path(paths["event_catalog_csv"])
    event_members_nc.parent.mkdir(parents=True, exist_ok=True)
    event_catalog_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_netcdf_replace(ds, event_members_nc)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(event_summary_csv, index=False)
    sfincs_catalog = pd.DataFrame(event_rows)
    _validate_compound_lag_contract(sfincs_catalog, window_hours=window_hours)
    sfincs_catalog.to_csv(event_catalog_csv, index=False)

    if paths.get("event_acceptance_json") is not None:
        Path(paths["event_acceptance_json"]).write_text(
            json.dumps({"passed": True, "event_count": int(len(sfincs_catalog)), "source": "copula_joint"}, indent=2) + "\n",
            encoding="utf-8",
        )
    if paths.get("lagtimes_csv") is not None:
        pd.Series({"h": 0.0}, name="lag_days").rename_axis("driver").to_csv(paths["lagtimes_csv"])

    # Require every configured event driver to be fully provenanced. Deriving this from
    # config.event_drivers (not just the rows present) is what makes the audit catch a soil
    # driver that silently dropped out of the handoff before scenarios are ever built.
    required_forcings = ["coastal", "rainfall"]
    if "soil_moisture" in configured_drivers:
        required_forcings.append("soil_moisture")
    if "streamflow" in configured_drivers:
        required_forcings.append("streamflow")
    audit = write_event_catalog_audit(
        sfincs_catalog,
        paths["event_catalog_audit_json"],
        required_forcings=required_forcings,
        wave_analog_policy="same_historical_analog" if config.get("coastal_waves", False) else "not_required",
    )
    if not audit["passed"]:
        raise RuntimeError(f"joint SFINCS handoff catalog failed audit: {audit['issues'][:5]}")
    return {"catalog": sfincs_catalog, "summary": summary, "dataset": ds, "audit": audit}


def _overlay_stress_training_fields(joint_catalog, paths):
    """Carry stress/training timing enrichment into the SFINCS scenario catalog."""
    catalog = joint_catalog.copy()
    stress_value = paths.get("resilience_stress_training_catalog_csv")
    if stress_value is None:
        return catalog
    stress_path = Path(stress_value)
    if not stress_path.exists():
        return catalog
    stress = pd.read_csv(stress_path)
    if "event_id" not in stress or stress.empty:
        return catalog

    overlay_cols = [c for c in stress.columns if c != "event_id"]
    merged = catalog.merge(stress[["event_id", *overlay_cols]], on="event_id", how="left", suffixes=("", "_stress"))
    for column in overlay_cols:
        stress_column = f"{column}_stress"
        if stress_column not in merged:
            continue
        if column in merged:
            merged[column] = merged[stress_column].combine_first(merged[column])
        else:
            merged[column] = merged[stress_column]
        merged = merged.drop(columns=[stress_column])
    return merged


def _compound_lag_hours(row):
    lag = _optional(row, "rainfall_peak_offset_hours", _optional(row, "rainfall_pairing_lag_hours", pd.NA))
    if not _is_missing(lag):
        return lag
    # Historical observed compound rows carry a true observed event time and may only have the
    # realization lag; synthetic copula-joint rows must get bounded stress/pairing lag instead.
    if str(_optional(row, "event_origin", "")).startswith("historical"):
        return _optional(row, "rainfall_realization_lag_hours", pd.NA)
    return pd.NA


def _validate_compound_lag_contract(catalog, *, window_hours):
    if "rainfall_member_id" not in catalog:
        return
    origin = catalog.get("event_origin", pd.Series("", index=catalog.index)).astype(str)
    policy = catalog.get("forcing_pairing_policy", pd.Series("", index=catalog.index)).astype(str)
    has_rain = ~catalog["rainfall_member_id"].map(_is_missing)
    synthetic_copula = origin.isin(["synthetic_body", "synthetic_tail"]) & policy.eq("copula_joint")
    required = catalog[has_rain & synthetic_copula]
    if required.empty:
        return
    timing_cols = [
        "rainfall_start_offset_hours",
        "rainfall_peak_offset_hours",
        "rainfall_end_offset_hours",
        "rainfall_pairing_lag_hours",
    ]
    missing_cols = [column for column in timing_cols if column not in required]
    if missing_cols:
        raise RuntimeError(
            "copula_joint SFINCS handoff requires full rainfall timing offsets; "
            f"missing columns: {missing_cols}"
        )

    timing = required[timing_cols].apply(pd.to_numeric, errors="coerce")
    missing = required[timing.isna().any(axis=1)]
    ordered = (
        (timing["rainfall_start_offset_hours"] <= timing["rainfall_peak_offset_hours"])
        & (timing["rainfall_peak_offset_hours"] <= timing["rainfall_end_offset_hours"])
    )
    inconsistent = required[~ordered]
    lag = timing["rainfall_pairing_lag_hours"]
    peak = timing["rainfall_peak_offset_hours"]
    unbounded = required[(lag.abs() > float(window_hours)) | (peak.abs() > float(window_hours))]
    if missing.empty and inconsistent.empty and unbounded.empty:
        return
    examples = []
    if not missing.empty:
        examples.extend(missing["event_id"].astype(str).head(5).tolist())
    if not inconsistent.empty:
        examples.extend(inconsistent["event_id"].astype(str).head(5).tolist())
    if not unbounded.empty:
        examples.extend(unbounded["event_id"].astype(str).head(5).tolist())
    raise RuntimeError(
        "copula_joint SFINCS handoff requires finite rainfall timing offsets and bounded peak lag "
        f"for synthetic rainfall events; examples: {examples[:5]}"
    )


def _required(row, column):
    value = row.get(column, pd.NA)
    if _is_missing(value):
        raise RuntimeError(f"joint catalog row is missing required SFINCS handoff field: {column}")
    return value


def _optional(row, column, default):
    value = row.get(column, pd.NA)
    return default if _is_missing(value) else value


def _configured_infiltration_method(config):
    """Resolve the SFINCS infiltration treatment that soil-conditioned events will run under."""
    hydrology = (config.get("coastal_wave_coupling") or {}).get("hydrology") or {}
    method = (hydrology.get("infiltration") or {}).get("method")
    if not method:
        method = (config.get("inland_coupling") or {}).get("infiltration", {}).get("method")
    if not method:
        method = config.get("infiltration", {}).get("treatment")
    return str(method) if method else "none"


def _event_drivers_value(row, configured_drivers, soil_present):
    """Advertise the row's drivers, adding soil_moisture once a soil member is actually paired."""
    existing = str(_optional(row, "event_drivers", "coastal_water_level, rainfall"))
    drivers = [token.strip() for token in existing.split(",") if token.strip()]
    if soil_present and "soil_moisture" in configured_drivers and "soil_moisture" not in drivers:
        drivers.append("soil_moisture")
    return ", ".join(drivers)


def _optional_timestamp(row, column):
    value = row.get(column, pd.NA)
    return None if _is_missing(value) else pd.Timestamp(value)


def _format_time(value):
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S")


def _format_optional_time(value):
    return pd.NA if value is None else _format_time(value)


def _is_missing(value):
    return value is None or bool(pd.isna(value)) or str(value).strip() == ""


def _write_netcdf_replace(dataset, path):
    # Notebook kernels often keep the previous NetCDF open; write a sibling file, then replace.
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.unlink(missing_ok=True)
        dataset.to_netcdf(tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
