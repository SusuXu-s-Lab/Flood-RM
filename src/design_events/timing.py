"""Timing descriptors for the reference bundle.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

loading_pattern_edges = (1.0 / 3.0, 2.0 / 3.0)
loading_pattern_labels = ("front_loaded", "center_loaded", "back_loaded")


def storm_loading_pattern(peak_offset_hours, duration_hours):
    """Normalized rainfall peak position and front/center/back-loaded label."""

    offset = pd.to_numeric(pd.Series(peak_offset_hours), errors="coerce").to_numpy(dtype=float)
    duration = pd.to_numeric(pd.Series(duration_hours), errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        position = np.clip(offset / np.where(duration > 0, duration, np.nan), 0.0, 1.0)
    lower, upper = loading_pattern_edges
    label = np.full(position.shape, "", dtype=object)
    finite = np.isfinite(position)
    label[finite & (position < lower)] = loading_pattern_labels[0]
    label[finite & (position >= lower) & (position < upper)] = loading_pattern_labels[1]
    label[finite & (position >= upper)] = loading_pattern_labels[2]
    label[~finite] = "unresolved"
    return position, label


def _enrich_rainfall_member_timing_reviewer(members):
    """Reviewer-bundle rainfall peak timing (no event-window read).

    Kept internal for ``attach_timing`` / ``build_reference_bundle`` so the reviewer
    bundle stays byte-identical. The public ``enrich_rainfall_member_timing`` is the
    production version (with event-window peak reading). Member tables with true
    ``rainfall_peak_time`` keep that provenance; legacy tables fall back to the
    storm-window midpoint and are explicitly marked ``legacy_midpoint_inferred``.
    """

    if members is None or len(members) == 0:
        return members
    out = pd.DataFrame(members).copy()
    start_column = _first_existing_column(out, ["storm_start", "member_time", "storm_date", "time"])
    if start_column is None:
        return out

    start = pd.to_datetime(out[start_column], errors="coerce")
    duration = _numeric_series(out, "duration_hours", 72.0).fillna(72.0)
    peak_column = _first_existing_column(out, ["rainfall_peak_time", "peak_time", "max_time"])
    if peak_column is not None:
        peak = pd.to_datetime(out[peak_column], errors="coerce")
        source = out.get("rainfall_peak_time_source", pd.Series("member_table", index=out.index)).astype(object)
    else:
        peak = pd.Series(pd.NaT, index=out.index)
        source = pd.Series("legacy_midpoint_inferred", index=out.index, dtype=object)

    missing = peak.isna()
    if missing.any():
        peak = peak.where(~missing, start + pd.to_timedelta((duration / 2.0).fillna(36.0), unit="h"))
        source = source.where(~missing, "legacy_midpoint_inferred")

    offset = (peak - start) / pd.Timedelta(hours=1)
    out["rainfall_peak_time"] = peak
    out["rainfall_peak_time_source"] = source.to_numpy()
    out["rainfall_peak_offset_from_start_hours"] = pd.to_numeric(offset, errors="coerce")
    out["duration_hours"] = duration
    return out


def observed_compound_lag_pool(paired_observations):
    """Observed peak-rainfall minus peak-coastal lag pool.

    Matches the production column contract when production names are supplied, and also
    accepts v2's ``coastal_ntr`` aliases.
    """

    if paired_observations is None or len(paired_observations) == 0:
        return pd.DataFrame()
    frame = pd.DataFrame(paired_observations).copy()
    frame = _coastal_aliases(frame)
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


def attach_timing(events, drivers, paired, members, config, seed=0):
    """Attach timing descriptors to v2 ``events`` and long ``drivers`` tables."""

    out_events = pd.DataFrame(events).copy()
    out_drivers = pd.DataFrame(drivers).copy()
    cfg = dict(config or {})
    member_tables = dict(members or {})
    audit = {"inland": {}, "compound_lag": {}, "seasonality": {}}

    if "rainfall" in member_tables and not out_drivers.empty:
        rainfall_members = _enrich_rainfall_member_timing_reviewer(member_tables["rainfall"])
        out_events, out_drivers, rainfall_audit = _attach_rainfall_timing(
            out_events,
            out_drivers,
            rainfall_members,
            event_reference_time_policy=_event_reference_time_policy(cfg),
        )
        audit["inland"].update(rainfall_audit)
        audit["seasonality"].update(_seasonality_summary(rainfall_members))

    if _has_coastal_compound(out_drivers) and paired is not None:
        out_events, out_drivers, lag_audit = _attach_compound_lag(out_events, out_drivers, paired, seed=seed)
        audit["compound_lag"].update(lag_audit)

    return out_events, out_drivers, audit


def _attach_rainfall_timing(events, drivers, rainfall_members, *, event_reference_time_policy):
    rainfall = drivers[drivers["driver"].astype(str).eq("rainfall")].copy()
    if rainfall.empty or rainfall_members is None or "member_id" not in rainfall_members:
        return events, drivers, {"event_reference_time_policy": "none", "rainfall_member_count": 0}

    keyed = rainfall_members.drop_duplicates("member_id").set_index(rainfall_members["member_id"].astype(str))
    member_ids = rainfall["member_id"].astype(str)
    start_column = _first_existing_column(keyed, ["storm_start", "time", "member_time", "storm_date"])
    start = pd.to_datetime(keyed[start_column].reindex(member_ids).to_numpy(), errors="coerce") if start_column else pd.Series(pd.NaT, index=rainfall.index)
    peak = pd.to_datetime(keyed["rainfall_peak_time"].reindex(member_ids).to_numpy(), errors="coerce")
    duration = pd.to_numeric(keyed["duration_hours"].reindex(member_ids).to_numpy(), errors="coerce")
    source = keyed.get("rainfall_peak_time_source", pd.Series("member_table", index=keyed.index)).reindex(member_ids)

    missing_peak = pd.Series(peak).isna()
    if missing_peak.any():
        warnings.warn(
            f"{int(missing_peak.sum())} realized rainfall members lack rainfall_peak_time; using midpoint timing.",
            RuntimeWarning,
            stacklevel=2,
        )

    offset = (pd.Series(peak).reset_index(drop=True) - pd.Series(start).reset_index(drop=True)) / pd.Timedelta(hours=1)
    position, label = storm_loading_pattern(offset, duration)

    enriched = rainfall.copy()
    enriched["rainfall_member_time"] = pd.Series(start).dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy()
    enriched["rainfall_peak_time"] = pd.Series(peak).dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy()
    enriched["rainfall_peak_time_source"] = source.fillna("member_table").to_numpy()
    enriched["rainfall_peak_offset_hours"] = pd.to_numeric(offset, errors="coerce").to_numpy()
    enriched["rainfall_start_offset_hours"] = -pd.to_numeric(offset, errors="coerce").to_numpy()
    enriched["storm_loading_position"] = position
    enriched["storm_loading_pattern"] = label

    drivers_out = drivers.copy()
    for column in [
        "rainfall_member_time",
        "rainfall_peak_time",
        "rainfall_peak_time_source",
        "rainfall_peak_offset_hours",
        "rainfall_start_offset_hours",
        "storm_loading_position",
        "storm_loading_pattern",
    ]:
        drivers_out.loc[enriched.index, column] = enriched[column].to_numpy()

    event_timing = enriched.drop_duplicates("event_id").set_index("event_id")
    events_out = events.copy()
    event_ids = events_out["event_id"].astype(str)
    for column in [
        "rainfall_peak_time",
        "rainfall_peak_time_source",
        "rainfall_peak_offset_hours",
        "rainfall_start_offset_hours",
        "storm_loading_position",
        "storm_loading_pattern",
    ]:
        events_out[column] = event_ids.map(event_timing[column])
    if event_reference_time_policy == "rainfall_peak_time":
        events_out["event_reference_time"] = events_out["rainfall_peak_time"]

    audit = {
        "event_reference_time_policy": event_reference_time_policy,
        "rainfall_member_count": int(len(enriched)),
        "storm_loading_pattern_counts": enriched["storm_loading_pattern"].value_counts(dropna=False).to_dict(),
    }
    return events_out, drivers_out, audit


def _attach_compound_lag(events, drivers, paired, *, seed):
    pool = observed_compound_lag_pool(paired)
    if pool.empty:
        return events, drivers, {"observed_lag_pool_count": 0}

    events_out = events.copy()
    drivers_out = drivers.copy()
    rng = np.random.default_rng(int(seed))
    lags = rng.choice(pool["observed_lag_hours"].to_numpy(dtype=float), size=len(events_out), replace=True)
    events_out["rainfall_pairing_lag_hours"] = lags
    events_out["compound_pairing_policy"] = "empirical_observed_lag_pool"

    rainfall_mask = drivers_out["driver"].astype(str).eq("rainfall")
    lag_by_event = pd.Series(lags, index=events_out["event_id"].astype(str))
    drivers_out.loc[rainfall_mask, "lag_hours"] = drivers_out.loc[rainfall_mask, "event_id"].astype(str).map(lag_by_event).to_numpy()
    drivers_out.loc[rainfall_mask, "time_policy"] = "empirical_observed_lag_pool"

    return events_out, drivers_out, {
        "observed_lag_pool_count": int(len(pool)),
        "lag_hours_min": float(pool["observed_lag_hours"].min()),
        "lag_hours_max": float(pool["observed_lag_hours"].max()),
        "selection_policy": "empirical_observed_lag_pool",
    }


def _coastal_aliases(frame):
    out = frame.copy()
    if "coastal_water_level" not in out and "coastal_ntr" in out:
        out["coastal_water_level"] = out["coastal_ntr"]
    if "coastal_water_level_time" not in out and "coastal_ntr_time" in out:
        out["coastal_water_level_time"] = out["coastal_ntr_time"]
    return out


def _event_reference_time_policy(config):
    family = str(config.get("event_family", ""))
    policy = str(config.get("reference_time_policy", "") or "")
    if policy:
        return policy
    if "inland" in family:
        return "rainfall_peak_time"
    return "none"


def _has_coastal_compound(drivers):
    names = set(drivers.get("driver", pd.Series(dtype=str)).astype(str))
    return "rainfall" in names and bool({"coastal_ntr", "coastal_water_level"} & names)


def _seasonality_summary(members):
    if members is None or "rainfall_peak_time" not in members:
        return {}
    peak = pd.to_datetime(members["rainfall_peak_time"], errors="coerce").dropna()
    if peak.empty:
        return {}
    return {
        "peak_month_counts": peak.dt.month.value_counts().sort_index().astype(int).to_dict(),
        "peak_hour_counts": peak.dt.hour.value_counts().sort_index().astype(int).to_dict(),
    }


def _numeric_series(frame, column, default):
    if column in frame:
        values = frame[column]
    else:
        values = pd.Series(default, index=frame.index)
    return pd.to_numeric(values, errors="coerce")


def _first_existing_column(frame, candidates):
    return next((column for column in candidates if column in frame), None)


# --------------------------------------------------------------------------------------
# Production wide-catalog timing (moved from the legacy compound and inland timing
# builders). These annotate the
# wide production catalog; the reviewer ``attach_timing`` above stays on the long tables.
# --------------------------------------------------------------------------------------


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


def attach_inland_rainfall_timing(
    catalog,
    members,
    *,
    member_id_column="member_id",
    start_time_column="storm_start",
    peak_time_column="rainfall_peak_time",
    duration_column="duration_hours",
):
    """Anchor the Event Reference Time on the true rainfall peak and attach descriptors.

    Joins each catalog row's realized ``rainfall_member_id`` back to the member table to
    recover the storm-window start, true peak time, and duration, then writes the inland
    Event Timing Descriptors (event_reference_time on the rainfall peak, peak/start
    offsets, storm loading pattern). Members without a collected ``rainfall_peak_time``
    fall back to the window midpoint and are flagged + warned.
    """
    out = catalog.copy()
    members = members.reset_index(drop=True)
    if member_id_column not in members:
        raise ValueError(f"members missing member id column {member_id_column!r}")
    if "rainfall_member_id" not in out:
        raise ValueError("catalog has no rainfall_member_id; run the rainfall realization first")

    lookup = members.set_index(members[member_id_column].astype(str))
    selected_ids = out["rainfall_member_id"].astype(str)

    start = pd.to_datetime(lookup.get(start_time_column).reindex(selected_ids).to_numpy(), errors="coerce")
    if peak_time_column in lookup:
        peak = pd.to_datetime(lookup[peak_time_column].reindex(selected_ids).to_numpy(), errors="coerce")
    else:
        peak = pd.Series(pd.NaT, index=range(len(out)))
    duration = pd.to_numeric(
        lookup.get(duration_column).reindex(selected_ids).to_numpy() if duration_column in lookup else np.nan,
        errors="coerce",
    )
    start = pd.Series(pd.to_datetime(start), index=out.index)
    peak = pd.Series(pd.to_datetime(peak), index=out.index)
    duration = pd.Series(np.asarray(duration, dtype=float), index=out.index)

    missing_peak = peak.isna()
    if missing_peak.any():
        midpoint_offset = (duration.where(duration > 0, np.nan) / 2.0).fillna(36.0)
        inferred = start + pd.to_timedelta(midpoint_offset.where(missing_peak, 0.0), unit="h")
        peak = peak.where(~missing_peak, inferred)
        warnings.warn(
            f"{int(missing_peak.sum())} realized rainfall members lack a collected "
            "rainfall_peak_time; inferring the storm-window midpoint and flagging "
            "rainfall_peak_time_source='inferred_midpoint'. Re-run AORC SST collection "
            "(02) to populate true peak timing.",
            RuntimeWarning,
            stacklevel=2,
        )

    source = pd.Series("storm_stats_hourly_peak", index=out.index)
    if peak_time_column in lookup:
        member_source = lookup.get("rainfall_peak_time_source")
        if member_source is not None:
            source = pd.Series(
                member_source.reindex(selected_ids).to_numpy(), index=out.index
            ).astype(object)
    source = source.where(~missing_peak, "inferred_midpoint")

    peak_offset = (peak - start) / pd.Timedelta(hours=1)
    position, label = storm_loading_pattern(peak_offset, duration)

    out["rainfall_member_time"] = start.dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["rainfall_peak_time"] = peak.dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["rainfall_peak_time_source"] = source.to_numpy()
    out["rainfall_peak_offset_hours"] = pd.to_numeric(peak_offset, errors="coerce")
    out["rainfall_start_offset_hours"] = -pd.to_numeric(peak_offset, errors="coerce")
    out["event_reference_time"] = peak.dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["storm_loading_position"] = position
    out["storm_loading_pattern"] = label
    return out


def observed_basin_lag(
    rainfall_members,
    streamflow_records,
    reference_gage,
    *,
    soil_moisture=None,
    peak_time_column="rainfall_peak_time",
    max_lag_hours=168.0,
    antecedent_window_hours=24.0,
    soil_value_column="SOILSAT_TOP",
):
    """Observed catchment basin lag at the Primary Reference Gage for each member's storm.

    For every rainfall member with a true ``rainfall_peak_time``, find the observed
    discharge peak at ``reference_gage`` within ``[peak, peak + max_lag_hours]`` and record
    ``basin_lag_hours = observed_peak_time - rainfall_peak_time``, the antecedent soil
    moisture at storm onset, and the season. This is the *observed reference* for the Wflow
    Readiness peak-timing check and the observed side of the Soil-Moisture Modulation
    Diagnostic — never a design driver.

    Returns a DataFrame (one row per member with a resolvable observed peak); empty when the
    reference gage has no usable record.
    """
    records = _normalize_discharge_records(streamflow_records)
    gage = str(reference_gage).zfill(8)
    gage_records = records[records["site_no"] == gage].set_index("time")["discharge_cfs"].sort_index()
    members = rainfall_members.copy()
    if peak_time_column not in members:
        return pd.DataFrame()
    members[peak_time_column] = pd.to_datetime(members[peak_time_column], errors="coerce")

    soil_series = None
    if soil_moisture is not None and soil_value_column in soil_moisture:
        soil_series = (
            soil_moisture.assign(time=pd.to_datetime(soil_moisture["time"], errors="coerce"))
            .dropna(subset=["time"])
            .set_index("time")[soil_value_column]
            .sort_index()
        )

    rows = []
    for _, member in members.iterrows():
        peak_time = member[peak_time_column]
        if pd.isna(peak_time) or gage_records.empty:
            continue
        window = gage_records.loc[peak_time : peak_time + pd.Timedelta(hours=float(max_lag_hours))]
        if window.empty:
            continue
        obs_peak_time = window.idxmax()
        lag_hours = (obs_peak_time - peak_time) / pd.Timedelta(hours=1)
        if not (0.0 <= lag_hours <= float(max_lag_hours)):
            continue
        antecedent_soil = np.nan
        if soil_series is not None:
            start = pd.to_datetime(member.get("storm_start"), errors="coerce")
            anchor = start if pd.notna(start) else peak_time
            prior = soil_series.loc[anchor - pd.Timedelta(hours=float(antecedent_window_hours)) : anchor]
            if not prior.empty:
                antecedent_soil = float(prior.iloc[-1])
        rows.append(
            {
                "member_id": member.get("member_id"),
                "rainfall_peak_time": peak_time,
                "observed_discharge_peak_time": obs_peak_time,
                "basin_lag_hours": float(lag_hours),
                "observed_peak_discharge_cfs": float(window.max()),
                "antecedent_soil_moisture": antecedent_soil,
                "month": int(peak_time.month),
                "season": _season(peak_time.month),
            }
        )
    return pd.DataFrame(rows)


def timing_seasonality(rainfall_members, *, peak_time_column="rainfall_peak_time"):
    """Peak month/hour distribution + a convective-vs-frontal season split for the members."""
    members = rainfall_members.copy()
    if peak_time_column not in members:
        return pd.DataFrame()
    peak = pd.to_datetime(members[peak_time_column], errors="coerce").dropna()
    if peak.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "member_id": members.loc[peak.index, "member_id"] if "member_id" in members else peak.index,
            "rainfall_peak_time": peak.to_numpy(),
            "month": peak.dt.month.to_numpy(),
            "hour": peak.dt.hour.to_numpy(),
            "season": [_season(m) for m in peak.dt.month],
        }
    )


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


def _format_optional_time(value):
    time = pd.to_datetime(value, errors="coerce")
    return pd.Timestamp(time).strftime("%Y-%m-%dT%H:%M:%S") if pd.notna(time) else pd.NA


def _normalize_discharge_records(streamflow_records):
    frame = streamflow_records.rename(
        columns={
            "datetime": "time",
            "dateTime": "time",
            "value": "discharge_cfs",
            "flow_cfs": "discharge_cfs",
            "00060": "discharge_cfs",
        }
    ).copy()
    required = {"site_no", "time", "discharge_cfs"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("streamflow records missing columns for basin-lag: " + ", ".join(sorted(missing)))
    frame["site_no"] = frame["site_no"].astype(str).str.zfill(8)
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame["discharge_cfs"] = pd.to_numeric(frame["discharge_cfs"], errors="coerce")
    return frame.dropna(subset=["time", "discharge_cfs"]).sort_values("time")


def _season(month):
    if month in (6, 7, 8, 9):
        return "warm_convective"
    if month in (12, 1, 2, 3):
        return "cool_frontal"
    return "transition"


__all__ = [
    "attach_timing",
    "enrich_rainfall_member_timing",
    "observed_compound_lag_pool",
    "storm_loading_pattern",
    "attach_empirical_rainfall_lags",
    "weighted_observed_lag_analog",
    "attach_inland_rainfall_timing",
    "observed_basin_lag",
    "timing_seasonality",
]
