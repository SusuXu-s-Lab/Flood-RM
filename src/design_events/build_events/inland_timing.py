"""Inland (Wflow-coupled) rainfall timing descriptors and observed diagnostics.

- ``storm_loading_pattern`` / ``attach_inland_rainfall_timing`` — where the storm peaks
  inside its accumulation window (front/center/back-loaded), plus the Event Reference
  Time anchored on the true rainfall peak and the offset descriptors the Wflow handoff
  reconstructs the storm window from.
- ``observed_basin_lag`` — observed USGS peak minus rainfall peak at the Primary Reference
  Gage for each member's historical storm, joined with antecedent soil moisture and season.
- ``timing_seasonality`` — peak month/hour distribution (convective vs frontal/tropical).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Front / center / back-loaded split of the normalized peak position (Huff terciles;
# Huff 1967, Australian Rainfall & Runoff temporal patterns).
loading_pattern_edges = (1.0 / 3.0, 2.0 / 3.0)
loading_pattern_labels = ("front_loaded", "center_loaded", "back_loaded")


def storm_loading_pattern(peak_offset_hours, duration_hours):
    """Normalized peak position in [0, 1] and a front/center/back-loaded label.

    ``peak_offset_hours`` is ``rainfall_peak_time - storm_start``; ``duration_hours`` is the
    storm accumulation window. Returns ``(position, label)`` as numpy arrays.
    """
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
    recover the storm-window start, true peak time, and duration, then writes:

    - ``event_reference_time`` = ``rainfall_peak_time`` (centres the Wflow forcing window)
    - ``rainfall_member_time`` left = storm onset (the AORC event-window ``.nc`` lookup key)
    - ``rainfall_peak_offset_hours`` = peak - storm_start (Event Timing Descriptor)
    - ``rainfall_start_offset_hours`` = -peak_offset (so the handoff reconstructs storm_start)
    - ``rainfall_peak_time_source`` provenance; ``inferred_midpoint`` when a member lacks a peak
    - ``storm_loading_position`` / ``storm_loading_pattern``

    Members without a collected ``rainfall_peak_time`` fall back to the window midpoint and
    are flagged + warned, mirroring the legacy-midpoint guard in ``compound_timing.py``.
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


def _season(month):
    if month in (6, 7, 8, 9):
        return "warm_convective"
    if month in (12, 1, 2, 3):
        return "cool_frontal"
    return "transition"
