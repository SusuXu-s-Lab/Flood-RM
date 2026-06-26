from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def enrich_rainfall_member_timing(members, *, precip_variable="APCP_surface"):
    """Carry true rainfall peak timing alongside the storm-window anchor.

    ``rainfall_member_time`` remains the storm-window start downstream; the peak
    timestamp is separate provenance used for compound RF-vs-coastal lag offsets.
    Legacy member tables without a peak timestamp are enriched from stored event
    windows when possible and clearly marked when midpoint inference is the only
    available fallback.
    """
    if members is None or len(members) == 0:
        return members
    out = members.copy()
    start_column = _first_existing_column(out, ["storm_start", "member_time", "storm_date", "time"])
    if start_column is None:
        return out
    out["_storm_start_time"] = pd.to_datetime(out[start_column], errors="coerce")
    duration = _numeric_catalog_series(out, "duration_hours", 72.0).fillna(72.0)

    peak_column = _first_existing_column(out, ["rainfall_peak_time", "peak_time", "max_time"])
    if peak_column is not None:
        out["rainfall_peak_time"] = pd.to_datetime(out[peak_column], errors="coerce")
        out["rainfall_peak_time_source"] = out.get("rainfall_peak_time_source", "member_table")
    else:
        peak_times, peak_values, sources = [], [], []
        for _, row in out.iterrows():
            peak_time, peak_value, source = _derive_peak_time_from_event_window(
                row,
                precip_variable=precip_variable,
            )
            if peak_time is None:
                start = row.get("_storm_start_time")
                hours = duration.loc[row.name] / 2.0
                peak_time = pd.Timestamp(start) + pd.Timedelta(hours=float(hours)) if pd.notna(start) else pd.NaT
                peak_value = np.nan
                source = "legacy_midpoint_inferred"
            peak_times.append(peak_time)
            peak_values.append(peak_value)
            sources.append(source)
        out["rainfall_peak_time"] = pd.to_datetime(peak_times, errors="coerce")
        out["rainfall_peak_mm_per_hour"] = peak_values
        out["rainfall_peak_time_source"] = sources

    offset_hours = (out["rainfall_peak_time"] - out["_storm_start_time"]) / pd.Timedelta(hours=1)
    out["rainfall_peak_offset_from_start_hours"] = pd.to_numeric(offset_hours, errors="coerce")
    out["duration_hours"] = duration
    return out.drop(columns=["_storm_start_time"], errors="ignore")


def observed_compound_lag_pool(paired_observations):
    """Observed peak-RF minus peak-coastal lag pool with conditioning covariates."""
    if paired_observations is None or len(paired_observations) == 0:
        return pd.DataFrame()
    frame = pd.DataFrame(paired_observations).copy()
    required = {"rainfall_time", "coastal_water_level_time", "rainfall", "coastal_water_level"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame["rainfall_time"] = pd.to_datetime(frame["rainfall_time"], errors="coerce")
    frame["coastal_water_level_time"] = pd.to_datetime(frame["coastal_water_level_time"], errors="coerce")
    frame["rainfall"] = pd.to_numeric(frame["rainfall"], errors="coerce")
    frame["coastal_water_level"] = pd.to_numeric(frame["coastal_water_level"], errors="coerce")
    frame["observed_lag_hours"] = (
        (frame["rainfall_time"] - frame["coastal_water_level_time"]) / pd.Timedelta(hours=1)
    )
    columns = [
        "event_time",
        "conditioned_on",
        "storm_type",
        "rainfall",
        "coastal_water_level",
        "rainfall_time",
        "coastal_water_level_time",
        "observed_lag_hours",
    ]
    present = [column for column in columns if column in frame.columns]
    out = frame[present].dropna(subset=["rainfall", "coastal_water_level", "observed_lag_hours"]).copy()
    if "event_time" in out:
        out["event_time"] = pd.to_datetime(out["event_time"], errors="coerce")
    return out.reset_index(drop=True)


def attach_empirical_rainfall_lags(
    catalog,
    paired_observations,
    rainfall_members,
    *,
    window_hours=72.0,
    season_window_days=45,
    min_storm_type_analogs=5,
    lag_pool_size=25,
    reuse_penalty_lambda=0.15,
    seed=0,
):
    """Attach conditional empirical RF-vs-coastal lags to catalogue rows.

    Lag is inherited from an observed co-occurrence analogue selected by weighted
    kNN over peak rainfall, peak NTR, season, and storm type where supported. This
    mirrors the field-preserving analogue idea used for rainfall/NTR shapes: keep
    timing empirical, but avoid collapsing many synthetic rows onto one nearest
    historical event.
    """
    if catalog is None or len(catalog) == 0:
        return catalog
    out = catalog.copy()
    if not {"rainfall", "coastal_water_level"}.issubset(out.columns):
        return out

    rainfall_members = enrich_rainfall_member_timing(rainfall_members)
    _attach_selected_rainfall_peak_metadata(out, rainfall_members)
    if "rainfall_peak_time_source" in out:
        legacy = out["rainfall_peak_time_source"].astype(str).eq("legacy_midpoint_inferred")
        if legacy.any():
            examples = out.loc[legacy, "rainfall_member_id"].astype(str).head(5).tolist()
            raise RuntimeError(
                "Empirical compound lagging requires true rainfall peak timestamps for selected "
                f"rainfall members; legacy midpoint inference found for {examples}. Regenerate "
                "the rainfall member/event-window source artifacts before building the normal catalogue."
            )

    pool = observed_compound_lag_pool(paired_observations)
    if pool.empty:
        out["compound_pairing_policy"] = out.get("compound_pairing_policy", pd.NA)
        out["scenario_timing_edge_case"] = out.get("scenario_timing_edge_case", "missing-observed-lag-pool")
        return out

    rng = np.random.default_rng(int(seed))
    analog_indices, lags = [], []
    analog_distances, analog_weights = [], []
    analog_candidate_counts, analog_strata = [], []
    analog_reuse_before = []
    usage = pd.Series(0, index=pool.index, dtype=int)
    for _, row in out.iterrows():
        analog, diagnostics = weighted_observed_lag_analog(
            row,
            pool,
            usage,
            rng=rng,
            season_window_days=season_window_days,
            min_storm_type_analogs=min_storm_type_analogs,
            lag_pool_size=lag_pool_size,
            reuse_penalty_lambda=reuse_penalty_lambda,
        )
        analog_indices.append(int(analog.name))
        lags.append(float(np.clip(analog["observed_lag_hours"], -float(window_hours), float(window_hours))))
        analog_distances.append(diagnostics["distance"])
        analog_weights.append(diagnostics["weight"])
        analog_candidate_counts.append(diagnostics["candidate_count"])
        analog_strata.append(diagnostics["stratum"])
        analog_reuse_before.append(diagnostics["reuse_count_before"])
        usage.loc[analog.name] += 1

    lag = pd.Series(lags, index=out.index, dtype=float)
    peak_from_start = _numeric_catalog_series(out, "rainfall_peak_offset_from_start_hours", 36.0).fillna(36.0)
    duration = _numeric_catalog_series(out, "rainfall_duration_hours", np.nan)
    if duration.isna().all():
        duration = _numeric_catalog_series(out, "duration_hours", 72.0)
    duration = duration.fillna(72.0)
    out["rainfall_pairing_lag_hours"] = lag
    out["rainfall_peak_offset_hours"] = lag
    out["rainfall_start_offset_hours"] = lag - peak_from_start
    out["rainfall_end_offset_hours"] = out["rainfall_start_offset_hours"] + duration
    if "rainfall" in out:
        out["rainfall_metric_mm"] = pd.to_numeric(out["rainfall"], errors="coerce")
    out["compound_pairing_policy"] = "conditional_empirical_weighted_knn_lag"
    out["compound_pairing_role"] = "empirical_analog_lag"
    out["scenario_timing_edge_case"] = "observed-analog-lag"
    out["empirical_lag_analog_index"] = analog_indices
    out["empirical_lag_analog_distance"] = analog_distances
    out["empirical_lag_analog_draw_weight"] = analog_weights
    out["empirical_lag_analog_candidate_count"] = analog_candidate_counts
    out["empirical_lag_analog_stratum"] = analog_strata
    out["empirical_lag_analog_reuse_count_before"] = analog_reuse_before
    out["empirical_lag_analog_event_time"] = [
        _format_optional_time(pool.loc[index].get("event_time", pd.NaT)) for index in analog_indices
    ]
    out["empirical_lag_analog_storm_type"] = [
        pool.loc[index].get("storm_type", pd.NA) if "storm_type" in pool else pd.NA for index in analog_indices
    ]
    return out


def weighted_observed_lag_analog(
    row,
    pool,
    usage,
    *,
    rng,
    season_window_days=45,
    min_storm_type_analogs=5,
    lag_pool_size=25,
    reuse_penalty_lambda=0.15,
):
    candidates = pool.copy()
    row_storm_type = row.get("storm_type", pd.NA)
    stratum = "all"
    if "storm_type" in candidates and pd.notna(row_storm_type):
        same_type = candidates[candidates["storm_type"].astype(str).eq(str(row_storm_type))]
        if len(same_type) >= int(min_storm_type_analogs):
            candidates = same_type.copy()
            stratum = str(row_storm_type)

    target_rainfall = float(pd.to_numeric(pd.Series([row.get("rainfall")]), errors="coerce").iloc[0])
    target_coastal = float(pd.to_numeric(pd.Series([row.get("coastal_water_level")]), errors="coerce").iloc[0])
    rain_values = np.log1p(pd.to_numeric(candidates["rainfall"], errors="coerce").to_numpy(dtype=float))
    coast_values = pd.to_numeric(candidates["coastal_water_level"], errors="coerce").to_numpy(dtype=float)
    rain_scale = _finite_scale(rain_values)
    coast_scale = _finite_scale(coast_values)
    distance_sq = ((np.log1p(max(target_rainfall, 0.0)) - rain_values) / rain_scale) ** 2
    distance_sq += ((target_coastal - coast_values) / coast_scale) ** 2

    target_time = _target_season_time(row)
    if target_time is not None and "event_time" in candidates:
        event_time = pd.to_datetime(candidates["event_time"], errors="coerce")
        valid = event_time.notna()
        if valid.any():
            day_distance = _day_of_year_distance(
                event_time.dt.dayofyear.to_numpy(dtype=float),
                pd.Timestamp(target_time).dayofyear,
            )
            season_scale = max(float(season_window_days), 1.0)
            distance_sq = distance_sq + np.where(valid.to_numpy(), 0.2 * (day_distance / season_scale) ** 2, 0.0)

    scored = candidates.copy()
    scored["_distance"] = np.sqrt(np.maximum(distance_sq, 0.0))
    scored = scored[np.isfinite(scored["_distance"])].copy()
    if scored.empty:
        scored = candidates.copy()
        scored["_distance"] = 0.0

    k = max(1, min(int(lag_pool_size), len(scored)))
    local = scored.sort_values("_distance").head(k).copy()
    distance = local["_distance"].to_numpy(dtype=float)
    distance_weights = 1.0 / (distance + 1e-6)
    reuse_counts = usage.reindex(local.index).fillna(0).to_numpy(dtype=float)
    reuse_weights = np.exp(-float(reuse_penalty_lambda) * reuse_counts)
    weights = distance_weights * reuse_weights
    if not np.isfinite(weights).any() or weights.sum() <= 0.0:
        weights = np.ones(len(local), dtype=float)
    weights = weights / weights.sum()
    draw_position = int(rng.choice(len(local), p=weights))
    analog = local.iloc[draw_position]
    return analog, {
        "distance": float(distance[draw_position]),
        "weight": float(weights[draw_position]),
        "candidate_count": int(len(local)),
        "stratum": stratum,
        "reuse_count_before": int(reuse_counts[draw_position]),
    }


def _attach_selected_rainfall_peak_metadata(catalog, rainfall_members):
    if rainfall_members is None or len(rainfall_members) == 0 or "rainfall_member_id" not in catalog:
        return
    if "member_id" not in rainfall_members:
        return
    keyed = rainfall_members.drop_duplicates("member_id").set_index("member_id")
    member_ids = catalog["rainfall_member_id"].astype(str)
    for source_column, target_column in [
        ("rainfall_peak_time", "rainfall_peak_time"),
        ("rainfall_peak_time_source", "rainfall_peak_time_source"),
        ("rainfall_peak_offset_from_start_hours", "rainfall_peak_offset_from_start_hours"),
        ("duration_hours", "rainfall_duration_hours"),
    ]:
        if source_column in keyed:
            catalog[target_column] = member_ids.map(keyed[source_column])
    if "rainfall_peak_time" in catalog:
        catalog["rainfall_peak_time"] = pd.to_datetime(catalog["rainfall_peak_time"], errors="coerce").dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )


def _derive_peak_time_from_event_window(row, *, precip_variable):
    path = _event_window_path(row)
    if path is None or not path.exists():
        return None, np.nan, "missing_event_window"
    try:
        import xarray as xr

        with xr.open_dataset(path) as ds:
            variable = precip_variable if precip_variable in ds else next(
                (name for name, da in ds.data_vars.items() if "time" in da.dims),
                None,
            )
            if variable is None:
                return None, np.nan, "event_window_missing_time_variable"
            da = ds[variable]
            spatial_dims = [dim for dim in da.dims if dim != "time"]
            series = da.mean(dim=spatial_dims, skipna=True) if spatial_dims else da
            if series.size == 0 or bool(series.isnull().all()):
                return None, np.nan, "event_window_empty_precip"
            index = int(series.argmax(dim="time").item())
            return pd.Timestamp(series["time"].isel(time=index).values), float(series.isel(time=index).values), "event_window_hourly_peak"
    except Exception:
        return None, np.nan, "event_window_peak_read_failed"


def _event_window_path(row):
    member_id = row.get("member_id")
    start = pd.to_datetime(row.get("storm_start", row.get("member_time", row.get("storm_date"))), errors="coerce")
    member_file = row.get("member_file")
    if pd.isna(member_id) or pd.isna(start) or pd.isna(member_file):
        return None
    return Path(str(member_file)).parent / "event_windows" / f"{member_id}_{pd.Timestamp(start):%Y%m%dT%H}.nc"


def _target_season_time(row):
    for column in ["coastal_water_level_member_time", "coastal_template_peak_time", "event_reference_time", "rainfall_member_time"]:
        value = row.get(column, pd.NaT)
        time = pd.to_datetime(value, errors="coerce")
        if pd.notna(time):
            return pd.Timestamp(time)
    return None


def _finite_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    scale = float(np.std(values)) if values.size > 1 else 1.0
    return scale if scale > 1e-9 else 1.0


def _numeric_catalog_series(frame, column, default):
    if column in frame:
        values = frame[column]
    else:
        values = pd.Series([default] * len(frame), index=frame.index)
    return pd.to_numeric(values, errors="coerce")


def _day_of_year_distance(a, b):
    diff = np.abs(np.asarray(a, dtype=float) - float(b))
    return np.minimum(diff, 366.0 - diff)


def _first_existing_column(frame, candidates):
    return next((column for column in candidates if column in frame), None)


def _format_optional_time(value):
    time = pd.to_datetime(value, errors="coerce")
    return pd.Timestamp(time).strftime("%Y-%m-%dT%H:%M:%S") if pd.notna(time) else pd.NA
