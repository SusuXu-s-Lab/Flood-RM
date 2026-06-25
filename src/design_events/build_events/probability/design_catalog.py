"""Copula-joint design-catalog build: the production wiring of the AND pipeline.

Ties the new stages into one catalog builder for the ``copula_joint`` Forcing Pairing
Policy, driven by a config-declared driver vector:

  paired observations -> per-driver marginals (AIC) -> vine fit -> selected
  AND-labeled design catalog -> stress-budget gate -> Field-Preserving Realization per driver.

It emits the standard weight/RP/severity columns plus ``<driver>_member_id`` /
``_member_file`` / ``_member_time`` / ``_scale_factor`` realization columns the SFINCS
staging and Wflow/SFINCS handoff already consume. The existing heuristic builder
(`build_inland_event_artifacts`) is retained as the documented sensitivity baseline;
this is the alternative production path selected by ``event_catalog.dependence.method``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from design_events.build_events.probability.dependence import (
    DriverDependenceModel,
    check_stress_budget,
    fit_driver_dependence,
    fit_storm_type_mixture,
    sample_mixture_catalog,
    sample_tail_enriched_catalog,
)
from design_events.build_events.selection import assign_severity_bands
from design_events.build_events.probability.exceedance import (
    and_return_period,
    and_survival_from_cdf,
    combined_return_period,
)
from design_events.build_events.probability.realization import attach_field_preserving_realization
from design_events.fit_history.extreme_value import fit_best_distribution
from design_events.fit_history.return_curve import EmpiricalMarginal, HistoricalPeakMarginal

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
    model: DriverDependenceModel
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


def build_historical_tail_catalog(
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
    if hasattr(model, "populations"):
        freq, sample_rp = combined_return_period(physical, model.populations)
        frame["and_joint_exceedance_prob"] = np.clip(freq / max(float(model.total_rate), 1e-12), 0.0, 1.0)
    else:
        u = np.column_stack(
            [np.asarray(m.cdf(physical[:, j]), dtype=float) for j, m in enumerate(model.marginals)]
        )
        survival = and_survival_from_cdf(np.clip(u, 1e-12, 1.0 - 1e-12), model.vine.cdf)
        _, sample_rp = and_return_period(survival, model.event_rate)
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
    seed = int(dependence.get("copula_seed", 42))

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
            member = _nearest_observed_member(
                member_libraries[driver],
                value=row[driver],
                value_column=spec.get("index_column", "value"),
                time_column=spec.get("time_column"),
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


def build_joint_design_catalog(
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
    seed = int(dependence.get("copula_seed", 42))
    n_catalog = int(config.get("events", {}).get("target_event_count", 2500))
    severity_bands = config.get("sampling", {}).get("severity_bands")
    band_fractions = dependence.get("catalog_band_fractions")
    pool_size = int(dependence.get("pool_size", 500_000))
    enforce_budget = bool(dependence.get("enforce_stress_budget", True))
    specs = {**default_realization_specs, **(realization_specs or {})}

    marginal_kinds = dependence.get("marginals", {}) or {}
    driver_kinds = {d: str((marginal_kinds.get(d, {}) or {}).get("kind", "pot")) for d in driver_vector}
    strat = dependence.get("storm_stratification", {}) or {}

    population_report = None
    if strat.get("enabled") and "storm_type" in paired_observations.columns:
        # Fix 3: fit a separate copula per storm-type population and combine their AEPs.
        model, population_report = fit_storm_type_mixture(
            paired_observations,
            driver_vector,
            marginal_kinds=driver_kinds,
            base_rate=event_rate,
            fit_marginal=lambda values, rate, kind: fit_index_marginal(values, event_rate=rate, kind=kind),
            min_population_events=int(strat.get("min_population_events", 20)),
            seed=seed,
        )
        catalog = sample_mixture_catalog(
            model,
            n_catalog,
            target_band_fractions=band_fractions,
            severity_bands=severity_bands,
            pool_size=pool_size,
            seed=seed,
            id_prefix=id_prefix,
        )
    else:
        observations = paired_observations[driver_vector].to_numpy(dtype=float)
        marginals = [
            fit_index_marginal(observations[:, j], event_rate=event_rate, kind=driver_kinds[driver])
            for j, driver in enumerate(driver_vector)
        ]
        model = fit_driver_dependence(observations, marginals, driver_vector, event_rate, seed=seed)
        catalog = sample_tail_enriched_catalog(
            model,
            n_catalog,
            target_band_fractions=band_fractions,
            severity_bands=severity_bands,
            pool_size=pool_size,
            seed=seed,
            id_prefix=id_prefix,
        )
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


build_joint_catalog = build_joint_design_catalog
build_tail = build_historical_tail_catalog
