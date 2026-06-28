"""Inland (Wflow-coupled) design-catalog builder.

Rainfall is the single stochastic design driver; antecedent soil moisture is a
conditioning attribute; discharge is the **Wflow response**, not a copula
dimension. The in-domain streamflow POT at the **Primary Reference Gage** is fit only as a
calibration/validation anchor (no per-event streamflow design target — the realized event
probability is carried by rainfall + antecedent wetness, and the flood frequency is
response-based after Wflow + SFINCS).

This module is intentionally isolated from the shared coastal copula path
(``design_catalog.py`` / ``dependence.py``) so the Marshfield coastal vine is untouched. It
reuses the shared marginal fit, severity banding, field-preserving realization attachment,
and antecedent-moisture conditioning, mirroring their call style.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from design_events.build_events.inland_timing import attach_inland_rainfall_timing
from design_events.build_events.probability.design_catalog import fit_index_marginal
from design_events.build_events.probability.realization import attach_field_preserving_realization
from design_events.build_events.selection import (
    assign_severity_bands,
    attach_antecedent_soil_moisture,
    default_severity_bands,
)


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

    catalog = _select_band_stratified(
        pool_rainfall, pool_rp, pool_band, n_catalog, band_fractions, rng, id_prefix=id_prefix
    )
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


def _select_band_stratified(pool_value, pool_rp, pool_band, n_catalog, band_fractions, rng, *, id_prefix):
    """Tail-enriched selection across severity bands with reconstructed probability weights.

    Mirrors the shared sampler's intent: enrich rare/extreme bands to the configured budget by
    count, while ``probability_weight`` reconstructs the true (body-dominated) mass so the
    Probability Catalog stays unbiased.
    """
    pool = pd.DataFrame({"rainfall_mm": pool_value, "sample_rp_years": pool_rp})
    if pool_band is not None:
        pool["severity_band"] = np.asarray(pool_band, dtype=object)
    else:
        pool["severity_band"] = "all"
    band_names = list(band_fractions.keys()) if band_fractions else list(pd.unique(pool["severity_band"]))

    picks = []
    for band in band_names:
        frac = float(band_fractions[band]) if band_fractions else (1.0 / len(band_names))
        target = max(int(round(frac * n_catalog)), 0)
        band_pool = pool[pool["severity_band"] == band]
        if band_pool.empty or target == 0:
            continue
        take = min(target, len(band_pool))
        idx = rng.choice(band_pool.index.to_numpy(), size=take, replace=False)
        chosen = band_pool.loc[idx].copy()
        true_band_mass = len(band_pool) / len(pool)
        chosen["sampling_weight"] = float(take) / max(true_band_mass * n_catalog, 1e-9)
        chosen["probability_weight"] = (true_band_mass / take) if take else 0.0
        picks.append(chosen)

    catalog = pd.concat(picks, ignore_index=True) if picks else pool.head(n_catalog).copy()
    total_pw = catalog["probability_weight"].sum()
    if total_pw > 0:
        catalog["probability_weight"] = catalog["probability_weight"] / total_pw
    catalog = catalog.sort_values("sample_rp_years").reset_index(drop=True)
    catalog.insert(0, "event_id", [f"{id_prefix}_{i:04d}" for i in range(len(catalog))])
    return catalog


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
