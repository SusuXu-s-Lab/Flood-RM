"""Storm-type classification of historical compound events from HURDAT2
"""

from __future__ import annotations

import numpy as np
import pandas as pd

earth_radius_km = 6371.0


def _haversine_km(lon0, lat0, lon, lat):
    lon0, lat0, lon, lat = map(np.radians, (lon0, lat0, np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)))
    d = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2
    return 2 * earth_radius_km * np.arcsin(np.sqrt(d))


def classify_storm_type(
    event_times,
    tracks,
    *,
    centroid_lonlat,
    radius_km=350.0,
    days_before=2,
    days_after=1,
    cool_season_months=(10, 11, 12, 1, 2, 3, 4),
):
    """Per-event population label: ``tc`` / ``nor_easter`` / ``other_non_tropical`` / ``unresolved``."""
    lon0, lat0 = float(centroid_lonlat[0]), float(centroid_lonlat[1])
    tr = tracks.copy()
    tr["time"] = pd.to_datetime(tr["time"], errors="coerce")
    tr = tr.dropna(subset=["time", "lat", "lon"]).sort_values("time")
    # track points that pass within range of the study coast
    near = tr[_haversine_km(lon0, lat0, tr["lon"], tr["lat"]) <= float(radius_km)]
    near_times = near["time"].to_numpy()
    cover_start, cover_end = tr["time"].min(), tr["time"].max()
    before, after = pd.Timedelta(days=days_before), pd.Timedelta(days=days_after)
    cool = set(int(m) for m in cool_season_months)

    times = pd.to_datetime(pd.Series(list(event_times)), errors="coerce")
    labels = []
    for t in times:
        if pd.isna(t) or t < cover_start or t > cover_end:
            labels.append("unresolved")
        elif ((near_times >= np.datetime64(t - before)) & (near_times <= np.datetime64(t + after))).any():
            labels.append("tc")
        elif t.month in cool:
            labels.append("nor_easter")
        else:
            labels.append("other_non_tropical")
    index = event_times.index if isinstance(event_times, pd.Series) else pd.RangeIndex(len(labels))
    return pd.Series(labels, index=index, name="storm_type")


def classify_from_config(event_times, config, *, tracks_path):
    """Classify using the ``event_catalog.dependence.storm_stratification`` config block."""
    strat = (config.get("event_catalog", {}) or {}).get("dependence", {}).get("storm_stratification", {}) or {}
    tracks = pd.read_csv(tracks_path)
    return classify_storm_type(
        event_times,
        tracks,
        centroid_lonlat=strat["centroid"],
        radius_km=float(strat.get("radius_km", 350.0)),
        days_before=int(strat.get("days_before", 2)),
        days_after=int(strat.get("days_after", 1)),
        cool_season_months=tuple(strat.get("cool_season_months", (10, 11, 12, 1, 2, 3, 4))),
    )


__all__ = ["classify_storm_type", "classify_from_config"]
