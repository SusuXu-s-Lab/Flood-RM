"""Collect the NHC HURDAT2 Atlantic tropical-cyclone track database (ADR-0011, Fix 3).

HURDAT2 is a single public text file: a 3-field header line per storm
(``AL092021, IDA, 40``) followed by that many 6-hourly track rows
(``20210826, 1200, , TS, 19.1N, 81.3W, 35, ...``). We parse it to one row per track
point ``[storm_id, name, time, lat, lon, status, wind_kt]`` so the storm-type classifier
can label each historical compound event TC vs non-TC by proximity to a passing track.
"""

from __future__ import annotations

import re

import pandas as pd
import requests

default_hurdat2_url = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"
_header = re.compile(r"^[A-Z]{2}\d{6}$")


def _signed(value):
    # "19.1N"/"81.3W" -> signed decimal degrees (S/W negative).
    value = value.strip()
    sign = -1.0 if value[-1] in "SW" else 1.0
    return sign * float(value[:-1])


def parse_hurdat2(text):
    """Parse raw HURDAT2 text into a track-point DataFrame."""
    storm_id = name = None
    rows = []
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) >= 3 and _header.match(fields[0]):
            storm_id, name = fields[0], fields[1]
            continue
        if storm_id is None or len(fields) < 7:
            continue
        rows.append({
            "storm_id": storm_id,
            "name": name,
            "time": pd.to_datetime(fields[0] + fields[1], format="%Y%m%d%H%M", errors="coerce"),
            "status": fields[3],
            "lat": _signed(fields[4]),
            "lon": _signed(fields[5]),
            "wind_kt": pd.to_numeric(fields[6], errors="coerce"),
        })
    return pd.DataFrame(rows).dropna(subset=["time"]).reset_index(drop=True)


def collect_hurdat2(settings, *, skip_existing=False, smoke=False):
    """Download + parse HURDAT2 to the location's track CSV (collector contract)."""
    paths = settings["paths"]
    spec = settings.get("hurdat2", {}) or {}
    out_csv = paths["hurdat2_tracks_csv"]
    if skip_existing and out_csv.exists() and out_csv.stat().st_size > 0:
        tracks = pd.read_csv(out_csv)
        return {"hurdat2_tracks_csv": out_csv, "track_points": len(tracks), "reused": True}
    url = spec.get("url", default_hurdat2_url)
    response = requests.get(url, timeout=int(spec.get("request_timeout_seconds", 60)))
    response.raise_for_status()
    tracks = parse_hurdat2(response.text)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(out_csv, index=False)
    return {
        "hurdat2_tracks_csv": out_csv,
        "track_points": len(tracks),
        "storms": int(tracks["storm_id"].nunique()),
        "reused": False,
    }
