"""Copula-joint + inland design-catalog builders (ADR-0021 single source of truth).

The keystone build stage relocated out of design_events.build_events.probability:
``build_joint_catalog`` (copula-joint coastal/compound) and ``build_inland_catalog``
(rainfall-driven Wflow-coupled), with the shared marginal fit and historical-tail builders.
Composes the v2 law/realization/timing/selection seams; imports no production design_events.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from design_events_v2.timing import (
    attach_empirical_rainfall_lags,
    attach_inland_rainfall_timing,
    enrich_rainfall_member_timing,
)
from design_events_v2.probability import (
    and_return_period,
    assign_severity_bands,
    check_stress_budget,
    default_severity_bands,
    select_catalog_indices,
)
from design_events_v2.realization import attach_field_preserving_realization
from design_events_v2.records import EmpiricalMarginal, HistoricalPeakMarginal
from design_events_v2.selection import attach_antecedent_soil_moisture
from design_events_v2.extreme_value import fit_best_distribution
from design_events_v2.workflow import fit_law, sample_catalog


# Per-driver defaults for mapping a member library onto the realization bridge.
default_realization_specs = {
    "rainfall": {"index_column": "mean_precip_mm", "time_column": "storm_start", "log_space": True},
    "streamflow": {"index_column": "peak_flow_cfs", "time_column": "event_time", "log_space": True},
    "soil_moisture": {"index_column": "soil_moisture_mean", "time_column": "time", "log_space": False},
    "coastal_water_level": {"index_column": "coastal_peak_m", "time_column": "time", "log_space": False},
}


@dataclass
class JointCatalogResult:
    catalog: pd.DataFrame
    budget_report: pd.DataFrame
    model: object
    population_report: pd.DataFrame = None  # per-storm-type fit summary when stratified (Fix 3)


def fit_index_marginal(values, *, event_rate, kind="pot", ev_type="pot", criterium="AIC"):
    """Fit a driver-role-aware marginal for one Driver Probability Index.

    ``kind="pot"`` fits an AIC-selected extreme-value tail (Exp/GPD) — for the conditioning
    extreme drivers (rainfall, water level, discharge). ``kind="empirical"`` fits a bounded
    empirical CDF — for antecedent/state drivers (e.g. soil saturation fraction), so the
    quantile function never extrapolates past the observed range.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        raise ValueError("need at least 3 finite values to fit a driver marginal")
    if kind == "empirical":
        return EmpiricalMarginal(v)
    if kind != "pot":
        raise ValueError(f"unknown marginal kind {kind!r}; use 'pot' or 'empirical'")
    params, dist_name = fit_best_distribution(v, ev_type, criterium=criterium)
    return HistoricalPeakMarginal(
        dist_name=dist_name,
        params=tuple(float(p) for p in params),
        extremes_rate=float(event_rate),
        method=ev_type,
        threshold_quantile=float("nan"),
        peak_count=int(v.size),
    )


def build_tail(
    paired_observations,
    model,
    config,
    paths,
    *,
    member_libraries,
    realization_specs=None,
    study_location=None,
    id_prefix="historical",
):
    """Append real observed joint-tail storms as reference scenarios, outside the 500 design budget."""
    dependence = (config.get("event_catalog", {}) or {}).get("dependence", {}) or {}
    driver_vector = list(dependence.get("driver_vector") or getattr(model, "driver_names", []))
    if not driver_vector:
        raise ValueError("driver_vector is required to build historical tail catalog")
    missing = [d for d in driver_vector if d not in paired_observations.columns]
    if missing:
        raise ValueError(f"paired_observations missing driver columns: {missing}")

    frame = paired_observations.copy().reset_index(drop=True)
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="coerce")
    frame = frame.dropna(subset=["event_time", *driver_vector]).copy()
    if frame.empty:
        return _empty_historical_tail_frame(driver_vector)

    physical = frame[driver_vector].to_numpy(dtype=float)
    if hasattr(model, "populations"):  # MixtureLaw
        freq, sample_rp = model.combined_return_period(physical)
        frame["and_joint_exceedance_prob"] = np.clip(freq / max(float(model.total_rate), 1e-12), 0.0, 1.0)
    else:  # JointLaw
        u = model.u(physical)
        survival = model.S_and(u)
        _, sample_rp = and_return_period(survival, model.rate)
        frame["and_joint_exceedance_prob"] = survival
        for j, driver in enumerate(driver_vector):
            frame[f"{driver}_u"] = u[:, j]

    severity_bands = config.get("sampling", {}).get("severity_bands")
    tail_return_period = float(dependence.get("historical_tail_min_return_period_years", 10.0))
    frame["sample_rp_years"] = sample_rp
    frame["severity_band"] = assign_severity_bands(sample_rp, severity_bands)
    frame = frame[np.isfinite(frame["sample_rp_years"]) & (frame["sample_rp_years"] >= tail_return_period)].copy()
    if frame.empty:
        return _empty_historical_tail_frame(driver_vector)

    decluster_hours = float(
        dependence.get("historical_tail_dedupe_hours")
        or dependence.get("cooccurrence", {}).get("decluster_window_hours", 120.0)
    )
    frame = _dedupe_observed_tail(frame, decluster_hours=decluster_hours)
    specs = {**default_realization_specs, **(realization_specs or {})}
    member_libraries = dict(member_libraries)
    if "rainfall" in member_libraries:
        member_libraries["rainfall"] = enrich_rainfall_member_timing(member_libraries["rainfall"])
    seed = int(dependence.get("copula_seed", 0))

    rows = []
    for _, row in frame.iterrows():
        event_time = pd.Timestamp(row["event_time"])
        out = {
            "event_id": f"{id_prefix}_{event_time.strftime('%Y%m%dT%H%M%S')}",
            "sample_rp_years": float(row["sample_rp_years"]),
            "and_joint_exceedance_prob": float(row["and_joint_exceedance_prob"]),
            "severity_band": row["severity_band"],
            "sampling_region": "tail",
            "sampling_weight": pd.NA,
            "probability_weight": pd.NA,
            "event_origin": "historical_tail",
            "catalog_role": "historical_reference",
            "sampling_scheme": "observed_historical_tail",
            "event_set": "historical_tail_reference",
            "selection_role": "historical_tail_reference",
            "selection_reason": "observed_joint_tail",
            "benchmark_return_period_years": pd.NA,
            "study_location": str(study_location or paths.get("location_name") or config.get("project", {}).get("name")),
            "event_family": "historical_compound_tail",
            "scenario_name": str(paths.get("scenario", {}).get("name", "base")),
            "forcing_pairing_policy": "historical_observed_compound",
            "event_drivers": ", ".join(driver_vector),
            "historical_conditioned_on": row.get("conditioned_on", pd.NA),
            "historical_event_time": event_time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if "storm_type" in row:
            out["storm_type"] = row["storm_type"]
        for driver in driver_vector:
            spec = specs.get(driver, {})
            driver_time = pd.to_datetime(row.get(f"{driver}_time", event_time), errors="coerce")
            if pd.isna(driver_time):
                driver_time = event_time
            lookup_time_column = "rainfall_peak_time" if driver == "rainfall" else spec.get("time_column")
            member = _nearest_observed_member(
                member_libraries[driver],
                value=row[driver],
                value_column=spec.get("index_column", "value"),
                time_column=lookup_time_column,
                reference_time=driver_time,
                window_hours=float(dependence.get("cooccurrence", {}).get("pairing_window_hours", 72.0)),
            )
            member_id = str(member["member_id"])
            template_value = float(member[spec.get("index_column", "value")])
            member_time = pd.Timestamp(member[spec.get("time_column", "time")])
            if driver == "coastal_water_level" and pd.notna(driver_time):
                member_time = pd.Timestamp(driver_time)
                member_id = f"{driver}_{member_time.strftime('%Y%m%dT%H%M%S')}"
                template_value = float(row[driver])
            out[driver] = float(row[driver])
            out[f"{driver}_u"] = row.get(f"{driver}_u", pd.NA)
            out[f"{driver}_template_member_id"] = member_id
            out[f"{driver}_template_value"] = template_value
            out[f"{driver}_scale_factor"] = 1.0
            out[f"{driver}_design_method"] = f"historical_observed_{driver}"
            out[f"{driver}_member_id"] = member_id
            out[f"{driver}_member_file"] = str(member.get("member_file", pd.NA))
            out[f"{driver}_member_time"] = member_time.strftime("%Y-%m-%dT%H:%M:%S")
            out[f"{driver}_realization_lag_hours"] = float((member_time - event_time) / pd.Timedelta(hours=1))
            out[f"{driver}_source"] = _member_source(member_libraries[driver], driver)
            out[f"{driver}_pairing_policy"] = "historical_observed_member"
            out[f"{driver}_pairing_seed"] = seed
            if driver == "rainfall":
                out["rainfall_peak_time"] = pd.Timestamp(driver_time).strftime("%Y-%m-%dT%H:%M:%S")
                out["rainfall_peak_time_source"] = "observed_paired_event"
                out["rainfall_peak_offset_from_start_hours"] = member.get("rainfall_peak_offset_from_start_hours", pd.NA)
                out["rainfall_duration_hours"] = member.get("duration_hours", pd.NA)
        if {"rainfall", "coastal_water_level"}.issubset(driver_vector):
            coastal_time = pd.to_datetime(row.get("coastal_water_level_time", event_time), errors="coerce")
            rainfall_peak = pd.to_datetime(row.get("rainfall_time", out.get("rainfall_peak_time")), errors="coerce")
            peak_from_start = pd.to_numeric(
                pd.Series([out.get("rainfall_peak_offset_from_start_hours", pd.NA)]), errors="coerce"
            ).iloc[0]
            duration = pd.to_numeric(pd.Series([out.get("rainfall_duration_hours", pd.NA)]), errors="coerce").iloc[0]
            if pd.notna(coastal_time) and pd.notna(rainfall_peak) and pd.notna(peak_from_start) and pd.notna(duration):
                lag = float((pd.Timestamp(rainfall_peak) - pd.Timestamp(coastal_time)) / pd.Timedelta(hours=1))
                out["rainfall_pairing_lag_hours"] = lag
                out["rainfall_peak_offset_hours"] = lag
                out["rainfall_start_offset_hours"] = lag - float(peak_from_start)
                out["rainfall_end_offset_hours"] = out["rainfall_start_offset_hours"] + float(duration)
                out["compound_pairing_policy"] = "historical_observed_peak_lag"
                out["compound_pairing_role"] = "historical_observed_lag"
                out["scenario_timing_edge_case"] = "historical-observed-lag"
        rows.append(out)

    catalog = pd.DataFrame(rows)
    events_root = config.get("wflow", {}).get("events_root", "data/wflow/events")
    scenarios_root = config.get("paths", {}).get("sfincs_scenarios_root", "data/sfincs/scenarios")
    catalog["wflow_event_dir"] = catalog["event_id"].map(lambda event_id: f"{events_root.rstrip('/')}/{event_id}")
    catalog["sfincs_scenario_dir"] = catalog["event_id"].map(lambda event_id: f"{scenarios_root.rstrip('/')}/{event_id}")
    catalog["infiltration_treatment"] = (
        config.get("inland_coupling", {}).get("infiltration", {}).get("method")
        or config.get("infiltration", {}).get("treatment", "none")
    )
    return catalog.sort_values("sample_rp_years").reset_index(drop=True)


def _empty_historical_tail_frame(driver_vector):
    columns = [
        "event_id", "sample_rp_years", "and_joint_exceedance_prob", "severity_band",
        "sampling_region", "event_origin", "catalog_role", "sampling_scheme",
    ]
    columns.extend(f"{driver}_{suffix}" for driver in driver_vector for suffix in ["member_id", "member_file", "member_time"])
    return pd.DataFrame(columns=columns)


def _dedupe_observed_tail(frame, *, decluster_hours):
    ordered = frame.sort_values("event_time").copy()
    groups, current = [], []
    separation = pd.Timedelta(hours=float(decluster_hours))
    for index, row in ordered.iterrows():
        event_time = pd.Timestamp(row["event_time"])
        if current and event_time - pd.Timestamp(ordered.loc[current[-1], "event_time"]) >= separation:
            groups.append(current)
            current = []
        current.append(index)
    if current:
        groups.append(current)
    keep = []
    for group in groups:
        winner = ordered.loc[group].sort_values(["sample_rp_years", "event_time"], ascending=[False, True]).index[0]
        keep.append(winner)
    return ordered.loc[keep].sort_values("event_time").reset_index(drop=True)


def _nearest_observed_member(members, *, value, value_column, time_column=None, reference_time=None, window_hours=None):
    members = members.copy()
    if value_column not in members:
        raise ValueError(f"member library missing value column {value_column!r}")
    if "member_id" not in members:
        raise ValueError("member library missing member_id column")
    members["_value"] = pd.to_numeric(members[value_column], errors="coerce")
    candidates = members.dropna(subset=["_value"]).copy()
    if candidates.empty:
        raise ValueError(f"member library has no finite {value_column!r} values")
    if time_column and time_column in candidates and reference_time is not None and window_hours is not None:
        candidates["_time"] = pd.to_datetime(candidates[time_column], errors="coerce")
        deltas = (candidates["_time"] - pd.Timestamp(reference_time)).abs() / pd.Timedelta(hours=1)
        in_window = candidates[deltas <= float(window_hours)].copy()
        if not in_window.empty:
            candidates = in_window
            deltas = deltas.loc[candidates.index]
        candidates["_time_distance"] = deltas.loc[candidates.index].to_numpy(dtype=float)
    else:
        candidates["_time_distance"] = 0.0
    candidates["_value_distance"] = (candidates["_value"] - float(value)).abs()
    return candidates.sort_values(["_value_distance", "_time_distance", "member_id"]).iloc[0]


def _member_source(members, driver):
    source = members.get("source")
    if source is not None and pd.Series(source).notna().any():
        return str(pd.Series(source).dropna().iloc[0])
    return str(driver)


def build_joint_catalog(
    config,
    paths,
    *,
    paired_observations,
    member_libraries,
    realization_specs=None,
    study_location=None,
    id_prefix="design",
):
    """Build the copula-joint, AND-labeled, field-realized design catalog.

    ``paired_observations`` is the two-sided POT co-occurrence sample (driver columns).
    ``member_libraries`` maps each driver to its observed member table (the field
    pointers). Statistical parameters and the driver vector come from
    ``config['event_catalog']['dependence']``.
    """
    dependence = (config.get("event_catalog", {}) or {}).get("dependence", {}) or {}
    driver_vector = list(dependence.get("driver_vector") or [])
    if not driver_vector:
        raise ValueError("config event_catalog.dependence.driver_vector is required for copula_joint")
    missing = [d for d in driver_vector if d not in paired_observations.columns]
    if missing:
        raise ValueError(f"paired_observations missing driver columns: {missing}")
    missing_libs = [d for d in driver_vector if d not in (member_libraries or {})]
    if missing_libs:
        raise ValueError(f"member_libraries missing drivers: {missing_libs}")

    # Rate for T = 1/(rate*S): the data-derived distinct-storm rate of the paired sample
    # (Fix 1), so the threshold and the return-period axis are self-consistent. Falls back
    # to the configured rate only when paired_observations carries no realized rate.
    realized_rate = float(paired_observations.attrs.get("base_event_rate_per_year", float("nan")))
    event_rate = realized_rate if np.isfinite(realized_rate) and realized_rate > 0 else float(dependence.get("event_rate_per_year", 5.0))
    seed = int(dependence.get("copula_seed", 0))
    n_catalog = int(config.get("events", {}).get("target_event_count", 2500))
    severity_bands = config.get("sampling", {}).get("severity_bands")
    band_fractions = dependence.get("catalog_band_fractions")
    pool_size = int(dependence.get("pool_size", 500_000))
    enforce_budget = bool(dependence.get("enforce_stress_budget", True))
    specs = {**default_realization_specs, **(realization_specs or {})}
    member_libraries = dict(member_libraries)
    if "rainfall" in member_libraries:
        member_libraries["rainfall"] = enrich_rainfall_member_timing(member_libraries["rainfall"])

    # Single composable v2 seam (ADR-0021): fit_law dispatches single JointLaw vs storm-type
    # MixtureLaw; sample_catalog produces the long AND-labeled, importance-weighted catalog.
    law = fit_law(paired_observations, driver_vector, event_rate, dependence, seed=seed)
    catalog, _u_selected, _x_selected = sample_catalog(law, config, dependence, seed=seed, id_prefix=id_prefix)
    model = law
    population_report = None
    budget_report = check_stress_budget(
        catalog,
        config.get("resilience_stress_training", {}) or {},
        severity_bands=severity_bands,
        raise_on_shortfall=enforce_budget,
    )

    for driver in driver_vector:
        spec = specs.get(driver, {})
        catalog = attach_field_preserving_realization(
            catalog,
            member_libraries[driver],
            driver=driver,
            target_column=driver,
            index_column=spec.get("index_column", "value"),
            member_id_column=spec.get("member_id_column", "member_id"),
            member_file_column=spec.get("member_file_column", "member_file"),
            time_column=spec.get("time_column"),
            design_method=f"copula_joint_scaled_{driver}_analog",
            log_space=bool(spec.get("log_space", True)),
            seed=seed,
        )
        source = member_libraries[driver].get("source")
        if source is not None and pd.Series(source).notna().any():
            catalog[f"{driver}_source"] = str(pd.Series(source).dropna().iloc[0])
        else:
            catalog[f"{driver}_source"] = str(driver)
        catalog[f"{driver}_pairing_policy"] = "copula_joint_field_preserving_analog"
        catalog[f"{driver}_pairing_seed"] = seed

    if {"rainfall", "coastal_water_level"}.issubset(driver_vector):
        timing_settings = (config.get("resilience_stress_training", {}) or {}).get("compound_pairing", {}) or {}
        catalog = attach_empirical_rainfall_lags(
            catalog,
            paired_observations,
            member_libraries["rainfall"],
            window_hours=float(timing_settings.get("real_event_window_hours", 72.0)),
            season_window_days=int(timing_settings.get("seasonal_window_days", timing_settings.get("window_days", 45))),
            min_storm_type_analogs=int(timing_settings.get("min_storm_type_lag_analogs", 5)),
            lag_pool_size=int(timing_settings.get("lag_pool_size", 25)),
            reuse_penalty_lambda=float(timing_settings.get("lag_reuse_penalty_lambda", 0.15)),
            seed=seed,
        )

    catalog["study_location"] = str(study_location or paths.get("location_name") or config.get("project", {}).get("name"))
    catalog["event_family"] = str(
        dependence.get("event_family")
        or config.get("event_catalog", {}).get("event_family")
        or "copula_joint_compound"
    )
    catalog["scenario_name"] = str(paths.get("scenario", {}).get("name", "base"))
    catalog["forcing_pairing_policy"] = "copula_joint"
    catalog["event_drivers"] = ", ".join(driver_vector)
    events_root = config.get("wflow", {}).get("events_root", "data/wflow/events")
    scenarios_root = config.get("paths", {}).get("sfincs_scenarios_root", "data/sfincs/scenarios")
    catalog["wflow_event_dir"] = catalog["event_id"].map(lambda event_id: f"{events_root.rstrip('/')}/{event_id}")
    catalog["sfincs_scenario_dir"] = catalog["event_id"].map(lambda event_id: f"{scenarios_root.rstrip('/')}/{event_id}")
    catalog["infiltration_treatment"] = (
        config.get("inland_coupling", {}).get("infiltration", {}).get("method")
        or config.get("infiltration", {}).get("treatment", "none")
    )
    return JointCatalogResult(catalog=catalog, budget_report=budget_report, model=model, population_report=population_report)


@dataclass
class InlandDesignCatalogResult:
    """Inland rainfall-driven design catalog plus its validation anchors."""

    catalog: pd.DataFrame
    rainfall_marginal: object
    streamflow_reference_pot: pd.Series | None
    budget_report: pd.DataFrame


def build_inland_historical_tail_catalog(
    config,
    paths,
    *,
    rainfall_members,
    rainfall_marginal,
    soil_moisture_members=None,
    id_prefix: str = "historical",
):
    """Build observed rainfall-tail reference events for inland Wflow/SFINCS runs.

    These rows are validation/reference events outside the stochastic design budget.
    They preserve the observed AORC SST rainfall window with scale factor 1.0, then
    reuse the same antecedent-soil and rainfall-peak timing contracts as the inland
    design catalog so downstream Wflow meteo staging and SFINCS packaging see the
    same columns for design and historical rows.
    """
    dependence = (config.get("event_catalog", {}) or {}).get("dependence", {}) or {}
    records_cfg = ((dependence.get("driver_records", {}) or {}).get("rainfall", {}) or {})
    members = rainfall_members.reset_index(drop=True).copy()
    value_column = _first_present(
        members, [records_cfg.get("value_column"), "mean_precip_mm", "mean", "precip_mm", "value"]
    )
    time_column = _first_present(
        members, [records_cfg.get("time_column"), "storm_start", "storm_date", "time"]
    )
    member_id_column = "member_id" if "member_id" in members else members.columns[0]
    if value_column is None:
        raise ValueError("rainfall_members has no recognizable rainfall value column")
    if time_column is None:
        raise ValueError("rainfall_members has no recognizable rainfall time column")
    if "member_file" not in members:
        raise ValueError("rainfall_members must include member_file for historical-tail handoff")

    rainfall = pd.to_numeric(members[value_column], errors="coerce")
    sample_rp = rainfall_marginal.return_period(rainfall.to_numpy(dtype=float))
    tail_return_period = float(dependence.get("historical_tail_min_return_period_years", 10.0))
    severity_bands = config.get("sampling", {}).get("severity_bands") or default_severity_bands()

    frame = members.copy()
    frame["rainfall_mm"] = rainfall
    frame["sample_rp_years"] = sample_rp
    frame = frame[np.isfinite(frame["sample_rp_years"]) & (frame["sample_rp_years"] >= tail_return_period)].copy()
    if frame.empty:
        return pd.DataFrame(columns=_inland_historical_tail_columns())

    frame["severity_band"] = assign_severity_bands(frame["sample_rp_years"], severity_bands)
    frame = frame.sort_values(["sample_rp_years", time_column], ascending=[False, True]).reset_index(drop=True)
    event_times = pd.to_datetime(
        frame["rainfall_peak_time"] if "rainfall_peak_time" in frame else frame[time_column],
        errors="coerce",
    ).fillna(pd.to_datetime(frame[time_column], errors="coerce"))

    catalog = pd.DataFrame(
        {
            "event_id": [
                f"{id_prefix}_{i + 1:04d}_{pd.Timestamp(t).strftime('%Y%m%dT%H%M%S')}"
                for i, t in enumerate(event_times)
            ],
            "rainfall_mm": frame["rainfall_mm"].to_numpy(dtype=float),
            "sample_rp_years": frame["sample_rp_years"].to_numpy(dtype=float),
            "severity_band": frame["severity_band"].to_numpy(dtype=object),
            "sampling_weight": pd.NA,
            "probability_weight": pd.NA,
            "event_origin": "historical_tail",
            "catalog_role": "historical_reference",
            "sampling_scheme": "observed_historical_tail",
            "event_set": "historical_tail_reference",
            "selection_role": "historical_tail_reference",
            "selection_reason": "observed_rainfall_tail",
            "scenario_name": "base",
            "rainfall_template_member_id": frame[member_id_column].astype(str).to_numpy(),
            "rainfall_template_value": frame["rainfall_mm"].to_numpy(dtype=float),
            "rainfall_scale_factor": 1.0,
            "rainfall_design_method": "historical_observed_rainfall",
            "rainfall_member_id": frame[member_id_column].astype(str).to_numpy(),
            "rainfall_member_file": frame["member_file"].astype(str).to_numpy(),
            "rainfall_member_time": pd.to_datetime(frame[time_column], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "rainfall_realization_lag_hours": 0.0,
            "rainfall_source": "aorc_sst",
            "forcing_pairing_policy": "historical_observed_rainfall",
            "event_drivers": "rainfall",
            "streamflow_design_role": "wflow_response",
            "study_location": str(paths.get("location_name") or config.get("project", {}).get("name") or ""),
        }
    )
    if soil_moisture_members is not None:
        catalog = attach_antecedent_soil_moisture(catalog, soil_moisture_members, config=config)
    catalog = attach_inland_rainfall_timing(
        catalog,
        members,
        member_id_column=member_id_column,
        start_time_column=time_column,
    )
    events_root = config.get("wflow", {}).get("events_root", "data/wflow/events")
    catalog["wflow_event_dir"] = catalog["event_id"].map(lambda e: f"{str(events_root).rstrip('/')}/{e}")
    catalog["infiltration_treatment"] = (
        config.get("inland_coupling", {}).get("infiltration", {}).get("method")
        or config.get("infiltration", {}).get("treatment", "none")
    )
    return catalog.reset_index(drop=True)


def fit_reference_streamflow_pot(
    streamflow_records,
    *,
    reference_gage,
    event_rate,
    pot_quantile: float = 0.95,
    return_periods=(2, 5, 10, 25, 50, 100, 200, 500),
):
    """Fit the in-domain streamflow POT at the Primary Reference Gage (validation only).

    Returns a frequency-curve ``Series`` (RP years -> discharge cfs) used to (a) validate that
    the rainfall-driven, Wflow-generated discharge ensemble reproduces observed streamflow
    frequency at the reference gage, and (b) inform the single calibration-constant K. It does
    NOT enter the design-event probability — discharge is the response, not a driver.
    """
    records = streamflow_records.copy()
    records["site_no"] = records["site_no"].astype(str).str.zfill(8)
    site = records[records["site_no"] == str(reference_gage).zfill(8)]
    values = pd.to_numeric(site.get("discharge_cfs"), errors="coerce").dropna()
    if values.empty:
        return None
    threshold = float(values.quantile(pot_quantile))
    peaks = values[values >= threshold].to_numpy(dtype=float)
    if peaks.size < 3:
        return None
    marginal = fit_index_marginal(peaks, event_rate=event_rate, kind="pot")
    curve = pd.Series(
        {int(rp): float(marginal.magnitude(rp)) for rp in return_periods},
        name=f"streamflow_pot_cfs_{reference_gage}",
    )
    curve.attrs["reference_gage"] = str(reference_gage)
    curve.attrs["pot_threshold_cfs"] = threshold
    curve.attrs["peak_count"] = int(peaks.size)
    return curve


def build_inland_catalog(
    config,
    paths,
    *,
    rainfall_members,
    soil_moisture_members=None,
    streamflow_records=None,
    reference_gage=None,
    id_prefix: str = "design",
):
    """Build the rainfall-driven, antecedent-conditioned inland design catalog.

    Rainfall is the single Driver Probability Index (POT/marginal); the catalog is
    tail-enriched across severity bands with importance weights; each event is realized to an
    observed AORC SST rainfall analog (field-preserving) and stamped with an antecedent
    soil-moisture state. Discharge is left to Wflow downstream. The streamflow POT at
    ``reference_gage`` is attached as a validation anchor in the result, not as a design column.
    """
    dependence = (config.get("event_catalog", {}) or {}).get("dependence", {}) or {}
    records_cfg = ((dependence.get("driver_records", {}) or {}).get("rainfall", {}) or {})

    members = rainfall_members.reset_index(drop=True).copy()
    value_column = _first_present(
        members, [records_cfg.get("value_column"), "mean_precip_mm", "mean", "precip_mm", "value"]
    )
    time_column = _first_present(
        members, [records_cfg.get("time_column"), "storm_start", "storm_date", "time"]
    )
    if value_column is None:
        raise ValueError("rainfall_members has no recognizable rainfall value column")
    rain_values = pd.to_numeric(members[value_column], errors="coerce").to_numpy(dtype=float)
    rain_values = rain_values[np.isfinite(rain_values)]
    if rain_values.size < 3:
        raise ValueError("need at least 3 finite rainfall member values to fit the inland marginal")

    event_rate = float(dependence.get("event_rate_per_year", 5.0))
    seed = int(dependence.get("copula_seed", 0))
    n_catalog = int(config.get("events", {}).get("target_event_count", 500))
    pool_size = int(dependence.get("pool_size", 100_000))
    band_fractions = (
        config.get("resilience_stress_training", {}).get("severity_band_fractions")
        or dependence.get("catalog_band_fractions")
    )
    severity_bands = config.get("sampling", {}).get("severity_bands") or default_severity_bands()
    # Persist the resolved bands back onto config so downstream notebook plots and
    # readiness tables read the exact bands this catalog was stratified on, instead of
    # KeyError'ing when no sampling block was authored in YAML. The coastal path does the
    # equivalent in configure_coastal_design_event_policy.
    config.setdefault("sampling", {})["severity_bands"] = severity_bands

    marginal = fit_index_marginal(rain_values, event_rate=event_rate, kind="pot")

    # Candidate pool: rainfall is the single driver, so RP = 1/(rate * (1 - F(rainfall))).
    rng = np.random.default_rng(seed)
    u = rng.uniform(1e-6, 1 - 1e-6, size=pool_size)
    pool_rainfall = np.asarray(marginal.ppf(u), dtype=float)
    pool_rp = 1.0 / np.clip(event_rate * (1.0 - u), 1e-12, None)
    pool_band = assign_severity_bands(pool_rp, severity_bands)

    # Tail-Enriched sampler shared with the coastal path (ADR-0021): corrected Sampling
    # Weight w_b = p_b/q_b and target-filling oversample of thin tail bands (replaces the
    # old inland-only _select_band_stratified with its inverted weight and replace=False
    # under-fill). probability_weight = normalized p_b/n_b is unchanged.
    band_names = list(band_fractions.keys()) if band_fractions else list(pd.unique(pool_band))
    selection = select_catalog_indices(pool_band.to_numpy(), band_names, n_catalog, band_fractions, rng)
    catalog = pd.DataFrame(
        {
            "rainfall_mm": pool_rainfall[selection.idx],
            "sample_rp_years": pool_rp[selection.idx],
            "severity_band": selection.band,
            "sampling_weight": selection.sampling_weight,
            "probability_weight": selection.probability_weight,
        }
    ).sort_values("sample_rp_years").reset_index(drop=True)
    catalog.insert(0, "event_id", [f"{id_prefix}_{i:04d}" for i in range(len(catalog))])
    catalog["catalog_role"] = "inland_design"
    catalog["scenario_name"] = "base"

    # Field-preserving rainfall realization (observed AORC SST analog + scale + lag).
    rainfall_member_id_column = "member_id" if "member_id" in members else members.columns[0]
    catalog = attach_field_preserving_realization(
        catalog,
        members,
        driver="rainfall",
        target_column="rainfall_mm",
        index_column=value_column,
        member_id_column=rainfall_member_id_column,
        member_file_column="member_file" if "member_file" in members else "member_file",
        time_column=time_column if time_column in members else None,
        design_method="rainfall_marginal_scaled_aorc_sst_analog",
        log_space=True,
        seed=seed,
    )
    catalog["rainfall_member_time"] = catalog.get("rainfall_member_time")
    catalog["rainfall_source"] = "aorc_sst"

    # Antecedent moisture: conditioning state, not a copula axis. Paired at storm onset
    # (rainfall_member_time) before the Event Reference Time is moved to the peak.
    if soil_moisture_members is not None:
        catalog = attach_antecedent_soil_moisture(catalog, soil_moisture_members, config=config)

    # Inland Event Reference Time = true rainfall peak (ADR-0019). rainfall_member_time stays
    # the storm onset (the AORC event-window lookup key); the rainfall peak centres the Wflow
    # forcing window and the storm-loading descriptors are attached for the stress set.
    catalog = attach_inland_rainfall_timing(
        catalog,
        members,
        member_id_column=rainfall_member_id_column,
        start_time_column=time_column if time_column in members else "storm_start",
    )

    catalog["forcing_pairing_policy"] = "rainfall_marginal_with_antecedent_moisture_wflow_response"
    catalog["event_drivers"] = "rainfall"
    catalog["streamflow_design_role"] = "wflow_response"  # not a design driver
    catalog["study_location"] = str(paths.get("location_name") or config.get("project", {}).get("name") or "")
    events_root = config.get("wflow", {}).get("events_root", "data/wflow/events")
    catalog["wflow_event_dir"] = catalog["event_id"].map(lambda e: f"{str(events_root).rstrip('/')}/{e}")
    catalog["infiltration_treatment"] = (
        config.get("inland_coupling", {}).get("infiltration", {}).get("method")
        or config.get("infiltration", {}).get("treatment", "none")
    )

    streamflow_reference_pot = None
    if streamflow_records is not None and reference_gage:
        streamflow_reference_pot = fit_reference_streamflow_pot(
            streamflow_records, reference_gage=reference_gage, event_rate=event_rate
        )
        if streamflow_reference_pot is not None:
            catalog["streamflow_reference_gage"] = str(reference_gage)

    budget_report = _band_budget_report(catalog, band_fractions)
    return InlandDesignCatalogResult(
        catalog=catalog,
        rainfall_marginal=marginal,
        streamflow_reference_pot=streamflow_reference_pot,
        budget_report=budget_report,
    )


def _first_present(frame, candidates):
    for name in candidates:
        if name and name in frame.columns:
            return str(name)
    return None


def _band_budget_report(catalog, band_fractions):
    counts = catalog["severity_band"].value_counts() if "severity_band" in catalog else pd.Series(dtype=int)
    rows = []
    for band in (band_fractions or {}):
        target = int(round(float(band_fractions[band]) * len(catalog)))
        got = int(counts.get(band, 0))
        rows.append({"severity_band": band, "target": target, "selected": got, "met": got >= target})
    return pd.DataFrame(rows)


def _inland_historical_tail_columns():
    return [
        "event_id",
        "rainfall_mm",
        "sample_rp_years",
        "severity_band",
        "sampling_weight",
        "probability_weight",
        "event_origin",
        "catalog_role",
        "sampling_scheme",
        "event_set",
        "selection_role",
        "selection_reason",
        "rainfall_member_id",
        "rainfall_member_file",
        "rainfall_member_time",
        "event_reference_time",
    ]


__all__ = [
    "JointCatalogResult", "InlandDesignCatalogResult",
    "default_realization_specs", "fit_index_marginal", "build_tail", "build_joint_catalog",
    "build_inland_catalog", "build_inland_historical_tail_catalog", "fit_reference_streamflow_pot",
]
