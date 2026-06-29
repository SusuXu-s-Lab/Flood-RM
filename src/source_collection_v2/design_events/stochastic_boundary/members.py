from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def event_members(rainfall, *, waterlevel=None, soil=None, waves=None, tracks=None, output_csv=None, tc_radius_km=500, tc_hours=72) -> pd.DataFrame:
    """Assemble B_i = (P_i, eta_i, W_i, M_i, Q_i, G pointers) as a CSV-friendly table."""
    out = pd.DataFrame(rainfall).copy()
    if out.empty:
        return out
    out["storm_start"] = pd.to_datetime(out["storm_start"])
    out["storm_end"] = pd.to_datetime(out["storm_end"])
    out["event_time"] = out["storm_start"] + (out["storm_end"] - out["storm_start"]) / 2
    if waterlevel is not None and not pd.DataFrame(waterlevel).empty:
        out = _nearest_time(out, pd.DataFrame(waterlevel), "waterlevel", value_columns=["value"])
    if soil is not None and not pd.DataFrame(soil).empty:
        soil_cols = [c for c in ["SOILSAT_TOP", "SOIL_M"] if c in pd.DataFrame(soil).columns]
        out = _nearest_time(out, pd.DataFrame(soil), "soil", value_columns=soil_cols)
    if tracks is not None and not pd.DataFrame(tracks).empty:
        out["storm_regime"] = classify_tc(out, pd.DataFrame(tracks), radius_km=tc_radius_km, hours=tc_hours)
    else:
        out["storm_regime"] = "unclassified"
    if waves is not None:
        out["wave_source"] = str(waves)
    if output_csv is not None:
        output_csv = Path(output_csv); output_csv.parent.mkdir(parents=True, exist_ok=True); out.to_csv(output_csv, index=False)
    return out


def classify_tc(events: pd.DataFrame, tracks: pd.DataFrame, *, radius_km=500, hours=72) -> list[str]:
    """Label rainfall events as TC when a HURDAT2 point is close in time and space."""
    events = events.copy()
    tracks = tracks.copy()
    tracks["time"] = pd.to_datetime(tracks["time"])
    labels = []
    lon_col = next((c for c in ["target_footprint_center_lon", "historical_footprint_center_lon", "centroid_lon"] if c in events), None)
    lat_col = next((c for c in ["target_footprint_center_lat", "historical_footprint_center_lat", "centroid_lat"] if c in events), None)
    for _, event in events.iterrows():
        if lon_col is None or lat_col is None or pd.isna(event.get(lon_col)) or pd.isna(event.get(lat_col)):
            labels.append("unclassified"); continue
        t0, t1 = pd.Timestamp(event["storm_start"]) - pd.Timedelta(hours=hours), pd.Timestamp(event["storm_end"]) + pd.Timedelta(hours=hours)
        nearby = tracks.loc[(tracks["time"] >= t0) & (tracks["time"] <= t1)]
        if nearby.empty:
            labels.append("nonTC"); continue
        d = _haversine(float(event[lon_col]), float(event[lat_col]), nearby["lon"].to_numpy(float), nearby["lat"].to_numpy(float))
        labels.append("TC" if np.nanmin(d) <= float(radius_km) else "nonTC")
    return labels


def empirical_measure(members: pd.DataFrame, weight_column: str | None = None) -> pd.DataFrame:
    """Return member weights for \u005chat{P}_n = sum_i w_i delta_{B_i}."""
    out = members[["member_id"]].copy()
    if weight_column and weight_column in members:
        w = pd.to_numeric(members[weight_column], errors="coerce").fillna(0)
        out["weight"] = w / w.sum()
    else:
        out["weight"] = 1.0 / max(len(out), 1)
    return out


def _nearest_time(left: pd.DataFrame, right: pd.DataFrame, prefix: str, *, value_columns: list[str]) -> pd.DataFrame:
    r = right.copy()
    r["time"] = pd.to_datetime(r["time"])
    cols = ["time", *value_columns]
    merged = pd.merge_asof(left.sort_values("event_time"), r[cols].sort_values("time"), left_on="event_time", right_on="time", direction="nearest")
    return merged.rename(columns={"time": f"{prefix}_time", **{c: f"{prefix}_{c}" for c in value_columns}})


def _haversine(lon0, lat0, lon, lat):
    r = 6371.0
    lon0, lat0, lon, lat = map(np.radians, [lon0, lat0, lon, lat])
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_boundary_condition_members(*args, **kwargs):
    """Alias for event_members; kept for notebook readability."""
    return event_members(*args, **kwargs)
