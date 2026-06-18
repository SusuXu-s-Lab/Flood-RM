from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


default_benchmark_return_period_years = [10, 50, 100, 500]
default_stress_training_severity_fractions = {
    "mild": 0.05,
    "common": 0.28,
    "significant": 0.28,
    "rare": 0.12,
    "extreme": 0.27,
}


def select_resilience_stress_training_set(
    catalog,
    rainfall_members=None,
    soil_moisture_members=None,
    config=None,
):
    settings = (config or {}).get("resilience_stress_training", {})
    benchmarks = settings.get("benchmark_return_period_years", default_benchmark_return_period_years)
    target_count = int(settings.get("target_event_count", min(500, len(catalog))))
    max_mild_fraction = float(settings.get("max_mild_fraction", 0.20))
    rainfall_fraction = float(settings.get("rainfall_heavy_fraction", 0.10))
    wet_soil_fraction = float(settings.get("wet_soil_fraction", 0.10))
    wave_fraction = float(settings.get("wave_overtopping_fraction", 0.10))
    severity_fractions = settings.get(
        "severity_band_fractions",
        default_stress_training_severity_fractions,
    )
    compound_pairing = settings.get("compound_pairing", {})

    frame = catalog.copy().reset_index(drop=True)
    frame["sample_rp_years"] = pd.to_numeric(frame["sample_rp_years"], errors="coerce")
    frame["_selection_score"] = 0.0
    frame["_selection_reason"] = [[] for _ in range(len(frame))]
    frame["_benchmark_rp"] = [[] for _ in range(len(frame))]

    _mark_benchmark_events(frame, benchmarks)
    _mark_tail_driver_events(frame)
    _mark_rainfall_heavy_events(frame, rainfall_members, rainfall_fraction)
    _mark_wet_soil_events(frame, soil_moisture_members, wet_soil_fraction)
    _mark_wave_sensitive_events(frame, wave_fraction)
    _mark_response_threshold_events(frame)

    candidates = frame.copy()

    candidates["_is_mild"] = candidates.get("severity_band", "").astype(str).eq("mild")
    candidates["_rp_sort"] = candidates["sample_rp_years"].fillna(-np.inf)
    candidates = candidates.sort_values(
        ["_selection_score", "_rp_sort", "event_id"],
        ascending=[False, False, True],
    )

    selected = _apply_budget(candidates, target_count, max_mild_fraction, severity_fractions)
    selected = selected.sort_values("sample_rp_years", na_position="last").reset_index(drop=True)
    selected["event_set"] = "resilience_stress_training"
    selected["selection_role"] = "resilience_stress_training"
    selected["_selection_reason"] = selected["_selection_reason"].map(
        lambda values: values if values else ["probability_catalog_representative"]
    )
    selected["selection_reason"] = selected["_selection_reason"].map(lambda values: ";".join(sorted(set(values))))
    selected["benchmark_return_period_years"] = selected["_benchmark_rp"].map(_format_benchmarks)
    if compound_pairing.get("enabled", False):
        selected = apply_compound_stress_pairing(
            selected,
            rainfall_members=rainfall_members,
            soil_moisture_members=soil_moisture_members,
            settings=compound_pairing,
        )
    return selected.drop(
        columns=[
            "_selection_score",
            "_selection_reason",
            "_benchmark_rp",
            "_is_mild",
            "_rp_sort",
        ],
        errors="ignore",
    )


def apply_compound_stress_pairing(
    catalog,
    *,
    rainfall_members=None,
    soil_moisture_members=None,
    settings=None,
):
    """Re-pair stress-set hydrology for operationally severe compound cases."""
    rainfall = _rainfall_member_table(rainfall_members)
    if rainfall is None or rainfall.empty:
        return catalog.copy()

    settings = dict(settings or {})
    out = catalog.copy().reset_index(drop=True)
    seed = int(settings.get("seed", 42))
    rng = np.random.default_rng(seed)
    season_window_days = int(settings.get("seasonal_window_days", settings.get("window_days", 45)))
    real_count = int(settings.get("real_event_count", 8))
    real_window_hours = float(settings.get("real_event_window_hours", 72))
    lead_hours = float(settings.get("soil_moisture_lead_time_hours", 24))
    reuse_penalty = float(settings.get("reuse_penalty", 0.15))

    out["compound_pairing_policy"] = "operationally_severe_plausible_dependence"
    out["compound_pairing_role"] = pd.NA
    out["scenario_timing_edge_case"] = pd.NA
    out["event_reference_time"] = _event_reference_times(out).dt.strftime("%Y-%m-%dT%H:%M:%S")
    for column in [
        "rainfall_source",
        "rainfall_member_file",
        "rainfall_member_id",
        "rainfall_member_time",
        "rainfall_pairing_policy",
        "rainfall_pairing_reference_time",
        "soil_moisture_source",
        "soil_moisture_member_file",
        "soil_moisture_member_id",
        "soil_moisture_member_time",
        "soil_moisture_pairing_policy",
        "soil_moisture_pairing_reference_time",
    ]:
        if column in out:
            out[column] = out[column].astype("object")

    role_by_index = _compound_roles(out, rainfall, real_count, real_window_hours, rng, settings)
    reuse_counts: dict[str, int] = {}
    for index, role in role_by_index.items():
        reference = pd.Timestamp(out.at[index, "event_reference_time"])
        member = _select_compound_rainfall_member(
            rainfall,
            reference,
            role,
            season_window_days=season_window_days,
            real_window_hours=real_window_hours,
            reuse_counts=reuse_counts,
            reuse_penalty=reuse_penalty,
        )
        if member is None:
            continue
        _assign_rainfall_member(out, index, member, reference, role, season_window_days, seed)

    soil = _soil_member_table(soil_moisture_members)
    if soil is not None and not soil.empty:
        for index, role in role_by_index.items():
            if pd.isna(out.at[index, "rainfall_member_time"]):
                continue
            rainfall_time = pd.Timestamp(out.at[index, "rainfall_member_time"])
            soil_member = _select_soil_member(
                soil,
                rainfall_time,
                lead_hours=lead_hours,
                season_window_days=season_window_days,
                wet=role == "wet_soil_high_rainfall",
            )
            if soil_member is not None:
                _assign_soil_member(
                    out,
                    index,
                    soil_member,
                    rainfall_time,
                    lead_hours,
                    wet=role == "wet_soil_high_rainfall",
                    seed=seed,
                )

    return out


def attach_antecedent_soil_moisture(catalog, soil_moisture_members, *, config=None):
    """Stamp an antecedent soil-moisture state onto every design event (ADR-0011).

    Soil moisture is not a copula axis; it is the wetness the basin carries into the storm. For
    each row with a paired rainfall time, pick the nearest soil observation ``lead_hours`` before
    that forcing and write the ``soil_moisture_*`` provenance the SFINCS handoff and seff staging
    expect. This is the design-catalog counterpart to the soil pairing inside the stress set, so
    the simulated events (synthetic design + historical tail) all run with conditioned moisture.
    """
    out = catalog.copy().reset_index(drop=True)
    soil = _soil_member_table(soil_moisture_members)
    if soil is None or soil.empty or "rainfall_member_time" not in out:
        return out

    settings = (config or {}).get("resilience_stress_training", {}).get("compound_pairing", {})
    seed = int(settings.get("seed", 42))
    lead_hours = float(settings.get("soil_moisture_lead_time_hours", 24))
    season_window_days = int(settings.get("seasonal_window_days", settings.get("window_days", 45)))

    for column in [
        "soil_moisture_source",
        "soil_moisture_member_file",
        "soil_moisture_member_id",
        "soil_moisture_member_time",
        "soil_moisture_pairing_policy",
        "soil_moisture_pairing_reference_time",
    ]:
        if column in out:
            out[column] = out[column].astype("object")

    rainfall_times = pd.to_datetime(out["rainfall_member_time"], errors="coerce")
    for index in out.index:
        rainfall_time = rainfall_times.iloc[index]
        if pd.isna(rainfall_time):
            continue
        soil_member = _select_soil_member(
            soil,
            rainfall_time,
            lead_hours=lead_hours,
            season_window_days=season_window_days,
            wet=False,
        )
        if soil_member is not None:
            _assign_soil_member(out, index, soil_member, rainfall_time, lead_hours, wet=False, seed=seed)
    return out


def write_resilience_stress_training_artifacts(config, paths):
    catalog = pd.read_csv(paths["event_catalog_csv"])
    rainfall_members = _read_optional_csv(paths.get("aorc_sst_rainfall_members_csv"))
    soil_moisture_members = _read_optional_csv(paths.get("nwm_soil_moisture_csv"))
    selected = select_resilience_stress_training_set(
        catalog,
        rainfall_members=rainfall_members,
        soil_moisture_members=soil_moisture_members,
        config=config,
    )
    output = Path(paths["resilience_stress_training_catalog_csv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    return selected


def _read_optional_csv(path):
    if path is None or not Path(path).exists():
        return None
    return pd.read_csv(path)


def _add_reason(frame, indices, reason, score):
    if len(indices) == 0:
        return
    for index in indices:
        frame.at[index, "_selection_reason"].append(reason)
    frame.loc[indices, "_selection_score"] += float(score)


def _mark_benchmark_events(frame, benchmarks):
    valid = frame["sample_rp_years"].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return
    driver = _driver_reason_suffix(frame)
    log_rp = np.log(valid)
    for benchmark in benchmarks:
        benchmark = float(benchmark)
        index = (log_rp - np.log(benchmark)).abs().idxmin()
        frame.at[index, "_selection_reason"].append(f"nearest_{_annual_chance_label(benchmark)}_{driver}")
        frame.at[index, "_selection_score"] += 100.0
        frame.at[index, "_benchmark_rp"].append(int(benchmark) if benchmark.is_integer() else benchmark)


def _annual_chance_label(return_period_years):
    aep = 100.0 / float(return_period_years)
    text = f"{aep:g}".replace(".", "p")
    return f"{text}pct_annual_chance"


def _mark_tail_driver_events(frame):
    severity = frame.get("severity_band", pd.Series([""] * len(frame))).astype(str)
    sampling_region = frame.get("sampling_region", pd.Series([""] * len(frame))).astype(str)
    indices = frame.index[
        severity.isin(["significant", "rare", "extreme", "beyond_design"])
        | sampling_region.eq("tail")
    ]
    _add_reason(frame, indices, f"tail_or_benchmark_{_driver_reason_suffix(frame)}", 35.0)


def _driver_reason_suffix(frame):
    columns = set(frame.columns)
    if columns & {"basis_site_no", "peak_flow_cfs", "streamflow_member_id", "streamflow_source"}:
        return "streamflow_driver"
    if columns & {"coastal_peak_m", "coastal_member_id", "coastal_analog_id", "coastal_source"}:
        return "coastal_driver"
    return "event_driver"


def _mark_rainfall_heavy_events(frame, rainfall_members, fraction):
    members = _rainfall_member_metrics(rainfall_members)
    if members is None:
        return
    merged = frame[["rainfall_member_id"]].merge(members, left_on="rainfall_member_id", right_on="member_id", how="left")
    metric = pd.to_numeric(merged["rainfall_metric"], errors="coerce")
    frame["rainfall_metric"] = metric.to_numpy()
    indices = _top_fraction_indices(metric, fraction)
    _add_reason(frame, indices, "rainfall_heavy_sst_member", 30.0)


def _rainfall_member_metrics(rainfall_members):
    rainfall = _rainfall_member_table(rainfall_members)
    if rainfall is None or rainfall.empty:
        if rainfall_members is None or "member_id" not in rainfall_members:
            return None
        value_column = _first_existing_column(
            rainfall_members,
            [
                "mean_precip_mm",
                "max_precip_mm",
                "total_precip_mm",
                "rainfall_depth_mm",
                "mean_precip_in",
                "max_precip_in",
                "total_precip_in",
                "rainfall_depth_in",
            ],
        )
        if value_column is None:
            return None
        return rainfall_members[["member_id", value_column]].rename(columns={value_column: "rainfall_metric"})
    return rainfall[["member_id", "rainfall_metric"]]


def _mark_wet_soil_events(frame, soil_moisture_members, fraction):
    # Soil moisture is now an Antecedent Moisture State attached after sampling, not a copula
    # realization column (ADR-0011), so this wet-soil marking no-ops until that state is wired.
    if "soil_moisture_member_id" not in frame.columns:
        return
    members = _soil_member_metrics(soil_moisture_members)
    if members is None:
        return
    merged = frame[["soil_moisture_member_id"]].merge(
        members,
        left_on="soil_moisture_member_id",
        right_on="member_id",
        how="left",
    )
    metric = pd.to_numeric(merged["soil_moisture_mean"], errors="coerce")
    frame["soil_moisture_mean"] = metric.to_numpy()
    indices = _top_fraction_indices(metric, fraction)
    _add_reason(frame, indices, "wet_antecedent_soil_state", 25.0)


def _soil_member_metrics(soil_moisture_members):
    soil = _soil_member_table(soil_moisture_members)
    if soil is None or soil.empty:
        if (
            soil_moisture_members is not None
            and {"member_id", "soil_moisture_mean"}.issubset(soil_moisture_members.columns)
        ):
            return soil_moisture_members[["member_id", "soil_moisture_mean"]]
        return None
    return soil[["member_id", "soil_moisture_mean"]]


def _rainfall_member_table(rainfall_members):
    if rainfall_members is None or len(rainfall_members) == 0 or "member_id" not in rainfall_members:
        return None
    members = rainfall_members.copy()
    value_column = _first_existing_column(
        members,
        [
            "mean_precip_mm",
            "max_precip_mm",
            "total_precip_mm",
            "rainfall_depth_mm",
            # Backward-compatible legacy names. AORC APCP values are mm, not inches.
            "mean_precip_in",
            "max_precip_in",
            "total_precip_in",
            "rainfall_depth_in",
        ],
    )
    if value_column is None:
        return None
    time_column = _first_existing_column(members, ["storm_start", "member_time", "storm_date", "time"])
    if time_column is None:
        return None
    out = members.copy()
    out["member_time"] = pd.to_datetime(out[time_column], errors="coerce")
    out["rainfall_metric"] = pd.to_numeric(out[value_column], errors="coerce")
    out = out.dropna(subset=["member_time", "rainfall_metric"]).copy()
    if "source" not in out:
        out["source"] = "aorc_sst"
    if "member_file" not in out:
        out["member_file"] = pd.NA
    if "duration_hours" not in out:
        out["duration_hours"] = 72
    out["duration_hours"] = pd.to_numeric(out["duration_hours"], errors="coerce").fillna(72.0)
    return out


def _soil_member_table(soil_moisture_members):
    if soil_moisture_members is None or len(soil_moisture_members) == 0:
        return None
    members = soil_moisture_members.copy()
    value_column = "SOILSAT_TOP" if "SOILSAT_TOP" in members else "SOIL_M"
    if {"time", value_column}.issubset(members.columns):
        grouped = members.groupby("time", as_index=False).agg(soil_moisture_mean=(value_column, "mean"))
        times = pd.to_datetime(grouped["time"], errors="coerce")
        grouped["member_id"] = "soil_moisture_" + times.dt.strftime("%Y%m%dT%H%M%S")
        grouped["member_time"] = times
        grouped["source"] = "nwm"
        grouped["member_file"] = getattr(soil_moisture_members, "attrs", {}).get("source_file", pd.NA)
        return grouped.dropna(subset=["member_time"])[
            ["member_id", "member_time", "soil_moisture_mean", "source", "member_file"]
        ]
    if {"member_id", "soil_moisture_mean"}.issubset(members.columns):
        out = members.copy()
        time_column = _first_existing_column(out, ["member_time", "time", "storm_start"])
        out["member_time"] = pd.to_datetime(out[time_column], errors="coerce") if time_column else pd.NaT
        if "source" not in out:
            out["source"] = "nwm"
        if "member_file" not in out:
            out["member_file"] = pd.NA
        return out.dropna(subset=["member_time"])[
            ["member_id", "member_time", "soil_moisture_mean", "source", "member_file"]
        ]
    return None


def _first_existing_column(frame, candidates):
    return next((column for column in candidates if column in frame), None)


def _event_reference_times(frame):
    for column in ["event_reference_time", "coastal_template_peak_time", "coastal_analog_peak_time", "rainfall_member_time"]:
        if column in frame:
            times = pd.to_datetime(frame[column], errors="coerce")
            if times.notna().any():
                return times.fillna(times.dropna().iloc[0])
    return pd.Series(pd.Timestamp("2000-01-01T00:00:00"), index=frame.index)


def _compound_roles(frame, rainfall, real_count, real_window_hours, rng, settings):
    roles = {}
    for index in _historical_compound_indices(frame, rainfall, real_count, real_window_hours):
        roles[index] = "historical_coastal_rainfall_pair"

    remaining = [index for index in frame.index if index not in roles]
    if not remaining:
        return roles
    fractions = settings.get(
        "role_fractions",
        {
            "high_rainfall_cooccurrence": 0.40,
            "rainfall_before_coastal": 0.25,
            "rainfall_after_coastal": 0.25,
            "wet_soil_high_rainfall": 0.10,
        },
    )
    role_names = [name for name, fraction in fractions.items() if float(fraction) > 0]
    weights = np.array([float(fractions[name]) for name in role_names], dtype=float)
    weights = weights / weights.sum() if weights.sum() else np.full(len(role_names), 1 / len(role_names))
    order = frame.loc[remaining].copy()
    order["_rp"] = pd.to_numeric(order.get("sample_rp_years"), errors="coerce").fillna(-np.inf)
    order = order.sort_values(["_rp", "event_id"], ascending=[False, True])
    role_sequence = rng.choice(role_names, size=len(order), replace=True, p=weights)
    for index, role in zip(order.index, role_sequence):
        roles[index] = str(role)
    for role, index in zip(role_names, order.index[: len(role_names)]):
        roles[index] = role
    return roles


def _historical_compound_indices(frame, rainfall, real_count, real_window_hours):
    if real_count <= 0:
        return []
    references = _event_reference_times(frame)
    scored = []
    for index, reference in references.items():
        deltas = (rainfall["member_time"] - reference).abs() / pd.Timedelta(hours=1)
        close = rainfall[deltas <= real_window_hours]
        if close.empty:
            continue
        rp = pd.to_numeric(pd.Series([frame.at[index, "sample_rp_years"]]), errors="coerce").iloc[0]
        scored.append((float(rp) if pd.notna(rp) else 0.0, float(close["rainfall_metric"].max()), index))
    scored.sort(reverse=True)
    return [index for _, _, index in scored[:real_count]]


def _select_compound_rainfall_member(
    rainfall,
    reference,
    role,
    *,
    season_window_days,
    real_window_hours,
    reuse_counts,
    reuse_penalty,
):
    if role == "historical_coastal_rainfall_pair":
        deltas = (rainfall["member_time"] - reference).abs() / pd.Timedelta(hours=1)
        candidates = rainfall[deltas <= real_window_hours].copy()
        if not candidates.empty:
            candidates["_time_score"] = -deltas.loc[candidates.index].to_numpy(dtype=float)
        else:
            candidates = _seasonal_candidates(rainfall, reference, season_window_days)
    else:
        candidates = _seasonal_candidates(rainfall, reference, season_window_days)
    if candidates.empty:
        return None
    scored = candidates.copy()
    scored["_reuse"] = scored["member_id"].astype(str).map(reuse_counts).fillna(0).astype(float)
    scored["_score"] = scored["rainfall_metric"].astype(float) - reuse_penalty * scored["_reuse"]
    if "_time_score" in scored:
        scored["_score"] = scored["_score"] + scored["_time_score"] / max(real_window_hours, 1.0)
    selected = scored.sort_values(["_score", "rainfall_metric", "member_id"], ascending=[False, False, True]).iloc[0]
    member_id = str(selected["member_id"])
    reuse_counts[member_id] = reuse_counts.get(member_id, 0) + 1
    return selected


def _seasonal_candidates(members, reference, window_days):
    member_doy = members["member_time"].dt.dayofyear.to_numpy(dtype=float)
    distances = _day_of_year_distance(member_doy, pd.Timestamp(reference).dayofyear)
    return members[distances <= float(window_days)].copy()


def _day_of_year_distance(a, b):
    diff = np.abs(np.asarray(a, dtype=float) - float(b))
    return np.minimum(diff, 366.0 - diff)


def _assign_rainfall_member(out, index, member, reference, role, season_window_days, seed):
    duration = float(member.get("duration_hours", 72.0))
    start_offset, peak_offset, end_offset, edge_case = _rainfall_offsets(role, duration, member, reference)
    member_time = pd.Timestamp(member["member_time"])
    out.at[index, "rainfall_source"] = member.get("source", "aorc_sst")
    out.at[index, "rainfall_member_file"] = member.get("member_file", pd.NA)
    out.at[index, "rainfall_member_id"] = str(member["member_id"])
    out.at[index, "rainfall_member_time"] = member_time.strftime("%Y-%m-%dT%H:%M:%S")
    out.at[index, "rainfall_pairing_policy"] = "compound_stress_operational"
    out.at[index, "rainfall_pairing_seed"] = seed
    out.at[index, "rainfall_pairing_window_days"] = season_window_days
    out.at[index, "rainfall_pairing_reference_time"] = pd.Timestamp(reference).strftime("%Y-%m-%dT%H:%M:%S")
    out.at[index, "rainfall_pairing_lag_hours"] = peak_offset
    out.at[index, "rainfall_start_offset_hours"] = start_offset
    out.at[index, "rainfall_peak_offset_hours"] = peak_offset
    out.at[index, "rainfall_end_offset_hours"] = end_offset
    out.at[index, "compound_pairing_role"] = role
    out.at[index, "scenario_timing_edge_case"] = edge_case
    out.at[index, "rainfall_metric_mm"] = float(member["rainfall_metric"])


def _rainfall_offsets(role, duration, member, reference):
    half = duration / 2.0
    if role == "rainfall_before_coastal":
        return -duration, -half, 0.0, "rainfall-before-coastal"
    if role == "rainfall_after_coastal":
        return 0.0, half, duration, "rainfall-after-coastal"
    if role == "historical_coastal_rainfall_pair":
        start = (pd.Timestamp(member["member_time"]) - pd.Timestamp(reference)) / pd.Timedelta(hours=1)
        return float(start), float(start + half), float(start + duration), "historical-compound-pair"
    if role == "wet_soil_high_rainfall":
        return -half, 0.0, half, "wet-soil-high-rain"
    return -half, 0.0, half, "rainfall-coincident"


def _select_soil_member(soil, rainfall_time, *, lead_hours, season_window_days, wet):
    target = pd.Timestamp(rainfall_time) - pd.Timedelta(hours=float(lead_hours))
    if wet:
        candidates = _seasonal_candidates(soil, target, season_window_days)
        if candidates.empty:
            candidates = soil.copy()
        return candidates.sort_values(["soil_moisture_mean", "member_id"], ascending=[False, True]).iloc[0]
    before = soil[soil["member_time"] <= target].copy()
    candidates = before if not before.empty else soil.copy()
    deltas = (candidates["member_time"] - target).abs()
    return candidates.loc[deltas.idxmin()]


def _assign_soil_member(out, index, member, rainfall_time, lead_hours, wet, seed):
    member_time = pd.Timestamp(member["member_time"])
    out.at[index, "soil_moisture_source"] = member.get("source", "nwm")
    out.at[index, "soil_moisture_member_file"] = member.get("member_file", pd.NA)
    out.at[index, "soil_moisture_member_id"] = str(member["member_id"])
    out.at[index, "soil_moisture_member_time"] = member_time.strftime("%Y-%m-%dT%H:%M:%S")
    out.at[index, "soil_moisture_pairing_policy"] = "wet_soil_stress" if wet else "antecedent_to_forcing"
    # The validator requires pairing-seed provenance for every non-coastal forcing; reuse the
    # stress-pairing seed so soil reproduces the same draw as the rainfall it is conditioned on.
    out.at[index, "soil_moisture_pairing_seed"] = seed
    out.at[index, "soil_moisture_pairing_reference_time"] = pd.Timestamp(rainfall_time).strftime("%Y-%m-%dT%H:%M:%S")
    out.at[index, "soil_moisture_pairing_lag_hours"] = lead_hours
    out.at[index, "soil_moisture_metric"] = float(member["soil_moisture_mean"])


def _mark_wave_sensitive_events(frame, fraction):
    for column in ["snapwave_hs_max_m", "wave_height_m", "hs_m", "significant_wave_height_m"]:
        if column in frame:
            indices = _top_fraction_indices(pd.to_numeric(frame[column], errors="coerce"), fraction)
            _add_reason(frame, indices, "wave_overtopping_sensitive_analog", 20.0)
            return


def _mark_response_threshold_events(frame):
    for column in ["near_islanding_threshold", "first_wet_grid_asset", "first_wet_critical_load"]:
        if column in frame:
            mask = frame[column].astype(bool)
            _add_reason(frame, frame.index[mask], column, 45.0)


def _top_fraction_indices(metric, fraction):
    metric = pd.to_numeric(metric, errors="coerce")
    valid = metric.dropna()
    if valid.empty:
        return []
    count = max(1, int(np.ceil(len(valid) * float(fraction))))
    return valid.nlargest(count).index.tolist()


def _apply_budget(candidates, target_count, max_mild_fraction, severity_fractions=None):
    target_count = min(max(int(target_count), 1), len(candidates))
    max_mild = int(np.floor(target_count * max_mild_fraction))
    max_mild = max(1, max_mild) if candidates["_is_mild"].any() else 0
    severity_targets = _severity_target_counts(
        candidates,
        target_count,
        max_mild=max_mild,
        severity_fractions=severity_fractions,
    )

    selected_indices = []
    mild_count = 0
    mandatory = candidates[candidates["_benchmark_rp"].map(bool)]
    for index, row in mandatory.iterrows():
        if index in selected_indices:
            continue
        selected_indices.append(index)
        mild_count += int(bool(row["_is_mild"]))

    if severity_targets:
        for band in severity_targets:
            selected_in_band = int(
                candidates.loc[selected_indices, "severity_band"].astype(str).eq(str(band)).sum()
            ) if selected_indices else 0
            needed = max(int(severity_targets[band]) - selected_in_band, 0)
            if needed <= 0:
                continue
            band_candidates = candidates[candidates.get("severity_band", pd.Series(index=candidates.index)).astype(str).eq(str(band))]
            for index, row in band_candidates.iterrows():
                if len(selected_indices) >= target_count or needed <= 0:
                    break
                if index in selected_indices:
                    continue
                is_mild = bool(row["_is_mild"])
                if is_mild and mild_count >= max_mild:
                    continue
                selected_indices.append(index)
                mild_count += int(is_mild)
                needed -= 1
    else:
        for index, row in candidates.iterrows():
            if len(selected_indices) >= target_count:
                break
            if index in selected_indices:
                continue
            is_mild = bool(row["_is_mild"])
            if is_mild and mild_count >= max_mild:
                continue
            selected_indices.append(index)
            mild_count += int(is_mild)

    if len(selected_indices) < target_count:
        for index, _ in candidates.iterrows():
            if len(selected_indices) >= target_count:
                break
            if index not in selected_indices:
                is_mild = bool(candidates.at[index, "_is_mild"]) if "_is_mild" in candidates else False
                if is_mild and mild_count >= max_mild:
                    continue
                selected_indices.append(index)
                mild_count += int(is_mild)
    if len(selected_indices) < target_count:
        for index, _ in candidates.iterrows():
            if len(selected_indices) >= target_count:
                break
            if index not in selected_indices:
                selected_indices.append(index)
    return candidates.loc[selected_indices]


def _severity_target_counts(candidates, target_count, *, max_mild, severity_fractions=None):
    if not severity_fractions or "severity_band" not in candidates:
        return {}
    fractions = dict(severity_fractions)
    bands = [band for band in fractions if band in set(candidates["severity_band"].astype(str))]
    if not bands:
        return {}
    total_fraction = sum(float(fractions[band]) for band in bands)
    if total_fraction <= 0:
        return {}
    raw = {
        band: target_count * float(fractions[band]) / total_fraction
        for band in bands
    }
    targets = {band: int(np.floor(raw[band])) for band in bands}
    remainder = target_count - sum(targets.values())
    for band in sorted(bands, key=lambda value: (raw[value] - targets[value], raw[value]), reverse=True):
        if remainder <= 0:
            break
        targets[band] += 1
        remainder -= 1
    if "mild" in targets:
        targets["mild"] = min(targets["mild"], max_mild)
    available = candidates["severity_band"].astype(str).value_counts()
    deficit = 0
    for band in list(targets):
        capped = min(targets[band], int(available.get(band, 0)))
        deficit += targets[band] - capped
        targets[band] = capped
    while deficit > 0:
        progressed = False
        for band in sorted(bands, key=lambda value: float(fractions[value]), reverse=True):
            if band == "mild" and targets.get(band, 0) >= max_mild:
                continue
            if targets.get(band, 0) >= int(available.get(band, 0)):
                continue
            targets[band] = targets.get(band, 0) + 1
            deficit -= 1
            progressed = True
            if deficit <= 0:
                break
        if not progressed:
            break
    return {band: targets[band] for band in bands if targets.get(band, 0) > 0}


def _format_benchmarks(values):
    if not values:
        return pd.NA
    out = []
    for value in sorted(set(values)):
        out.append(str(int(value)) if float(value).is_integer() else str(value))
    return ";".join(out)
