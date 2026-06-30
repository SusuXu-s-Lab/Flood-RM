"""Collect the NHC HURDAT2 Atlantic tropical-cyclone track database.

HURDAT2 is a single public text file: a 3-field header line per storm
(``AL092021, IDA, 40``) followed by that many 6-hourly track rows
(``20210826, 1200, , TS, 19.1N, 81.3W, 35, ...``). We parse it to one row per track
point ``[storm_id, name, time, lat, lon, status, wind_kt]`` so the storm-type classifier
can label each historical compound event TC vs non-TC by proximity to a passing track.
"""

from __future__ import annotations

import pandas as pd
import requests

from source_collection_v2.hurdat2 import URL as _V2_HURDAT2_URL
from source_collection_v2.hurdat2 import parse as _parse_hurdat2

default_hurdat2_url = _V2_HURDAT2_URL


def parse_hurdat2(text):
    """Parse raw HURDAT2 text into a track-point DataFrame."""
    return _parse_hurdat2(text)


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
