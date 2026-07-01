"""USGS streamgage member construction for inland external-boundary events."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from event_streamflow import (
    streamflow_members_path as _streamflow_members_path,
    streamflow_records_path as _streamflow_records_path,
)
from paths import location_root_from_paths


def build_usgs_streamflow_event_members(config, paths, *, streamflow_records=None):
    """Build streamflow event-member rows from reviewed USGS discharge records."""
    records = load_usgs_streamflow_records(config, paths, streamflow_records=streamflow_records)
    candidates = streamflow_pot_candidate_peaks(records, config)
    members = decluster_streamflow_network_events(candidates, config)
    write_streamflow_event_members(config, paths, members)
    return members


def load_usgs_streamflow_records(config, paths, *, streamflow_records=None):
    """Load reviewed USGS discharge records for visible POT diagnostics."""
    return _load_streamflow_records(config, paths, streamflow_records)


def streamflow_pot_candidate_peaks(records, config):
    """Find per-site discharge POT candidate peaks using the configured threshold."""
    return _site_peak_candidates(records, config)


def decluster_streamflow_network_events(candidates, config):
    """Collapse nearby per-site POT peaks into coherent network event members."""
    return _decluster_network_candidates(candidates, config)


def write_streamflow_event_members(config, paths, members):
    """Write the streamflow member table after the visible POT/decluster stages."""
    output_path = streamflow_event_members_path(config, paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    members.to_csv(output_path, index=False)
    return output_path


def streamflow_event_members_path(config, paths):
    return _streamflow_members_path(config, _location_root(paths))


def streamflow_records_path(config, paths):
    return _streamflow_records_path(config, _location_root(paths))


def _load_streamflow_records(config, paths, streamflow_records):
    if streamflow_records is not None:
        records = streamflow_records.copy()
    else:
        input_path = _streamflow_records_path(config, _location_root(paths))
        records = pd.read_csv(input_path, dtype={"site_no": str})
    records = records.rename(
        columns={
            "datetime": "time",
            "dateTime": "time",
            "value": "discharge_cfs",
            "flow_cfs": "discharge_cfs",
            "00060": "discharge_cfs",
        }
    )
    required = {"site_no", "time", "discharge_cfs"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError("USGS streamflow records are missing required columns: " + ", ".join(sorted(missing)))
    records["site_no"] = records["site_no"].astype(str)
    records["time"] = pd.to_datetime(records["time"], errors="coerce")
    records["discharge_cfs"] = pd.to_numeric(records["discharge_cfs"], errors="coerce")
    records = records.dropna(subset=["site_no", "time", "discharge_cfs"]).sort_values(["site_no", "time"])
    if records.empty:
        raise ValueError("USGS streamflow records contain no valid discharge observations")
    return records


def _site_peak_candidates(records, config):
    settings = config.get("extremes", {}).get("pot", {})
    qthresh = float(settings.get("threshold_quantile", 0.98))
    min_distance_hours = int(settings.get("min_peak_distance_hours", 72))
    rows = []
    for site_no, group in records.groupby("site_no", sort=True):
        series = group.sort_values("time").set_index("time")["discharge_cfs"].dropna()
        if series.empty:
            continue
        spacing_hours = _median_spacing_hours(series.index)
        distance_steps = max(1, int(round(min_distance_hours / spacing_hours)))
        peak_indices, _ = find_peaks(series.to_numpy(dtype=float), distance=distance_steps)
        if len(peak_indices) == 0:
            continue
        peaks = series.iloc[peak_indices]
        threshold = float(series.quantile(qthresh))
        peaks = peaks[peaks >= threshold].sort_values(ascending=False)
        for time, value in peaks.items():
            rows.append(
                {
                    "site_no": str(site_no),
                    "event_time": pd.Timestamp(time),
                    "peak_flow_cfs": float(value),
                    "site_threshold_cfs": threshold,
                }
            )
    if not rows:
        raise ValueError("no USGS POT peaks found above the configured threshold")
    return pd.DataFrame(rows).sort_values("peak_flow_cfs", ascending=False).reset_index(drop=True)


def _decluster_network_candidates(candidates, config):
    window_hours = int(config.get("event_catalog", {}).get("streamflow", {}).get("network_decluster_hours", 72))
    if window_hours <= 0:
        window_hours = int(config.get("extremes", {}).get("pot", {}).get("min_peak_distance_hours", 72))
    remaining = candidates.copy()
    events = []
    while not remaining.empty:
        dominant = remaining.iloc[0]
        center = pd.Timestamp(dominant["event_time"])
        in_window = remaining["event_time"].sub(center).abs() <= pd.Timedelta(hours=window_hours)
        contributors = remaining.loc[in_window]
        events.append(_event_row(dominant, contributors))
        remaining = remaining.loc[~in_window].sort_values("peak_flow_cfs", ascending=False).reset_index(drop=True)
    members = pd.DataFrame(events).sort_values("peak_flow_cfs", ascending=False).reset_index(drop=True)
    ranks = np.arange(1, len(members) + 1, dtype=float)
    members["sample_rp_years"] = (len(members) + 1.0) / ranks
    members["sampling_weight"] = 1.0
    members["probability_weight"] = 1.0 / float(len(members))
    tail_count = max(1, int(np.ceil(len(members) * 0.2)))
    members["sampling_region"] = "body"
    members.loc[: tail_count - 1, "sampling_region"] = "tail"
    members["event_time"] = pd.to_datetime(members["event_time"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    members["member_id"] = members["site_no"] + "_" + pd.to_datetime(members["event_time"]).dt.strftime("%Y%m%dT%H%M%S")
    members["event_id"] = "usgs_" + members["member_id"]
    members["source"] = "usgs"
    return members[
        [
            "event_id",
            "member_id",
            "site_no",
            "event_time",
            "peak_flow_cfs",
            "site_threshold_cfs",
            "sample_rp_years",
            "sampling_region",
            "sampling_weight",
            "probability_weight",
            "source",
            "network_site_count",
            "contributing_site_nos",
        ]
    ]


def _event_row(dominant, contributors):
    site_nos = sorted(set(contributors["site_no"].astype(str)))
    return {
        "site_no": str(dominant["site_no"]),
        "event_time": pd.Timestamp(dominant["event_time"]),
        "peak_flow_cfs": float(dominant["peak_flow_cfs"]),
        "site_threshold_cfs": float(dominant["site_threshold_cfs"]),
        "network_site_count": int(len(site_nos)),
        "contributing_site_nos": ",".join(site_nos),
    }


def _median_spacing_hours(index):
    if len(index) < 2:
        return 1.0
    diffs = pd.Series(index).sort_values().diff().dropna()
    hours = diffs.dt.total_seconds().median() / 3600.0
    return max(float(hours), 1.0)


def _location_root(paths):
    return location_root_from_paths(paths)


__all__ = [
    "build_usgs_streamflow_event_members",
    "decluster_streamflow_network_events",
    "load_usgs_streamflow_records",
    "streamflow_event_members_path",
    "streamflow_pot_candidate_peaks",
    "streamflow_records_path",
    "write_streamflow_event_members",
]
