"""Collect the NHC HURDAT2 Atlantic tropical-cyclone track database."""

from __future__ import annotations

import pandas as pd
import requests

URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt"
default_hurdat2_url = URL


def _coord(value: str) -> float:
    value = value.strip()
    if not value:
        return float("nan")
    sign = -1.0 if value[-1].upper() in {"S", "W"} else 1.0
    return sign * float(value[:-1])


def parse(text: str) -> pd.DataFrame:
    rows = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    i = 0
    while i < len(lines):
        header = [part.strip() for part in lines[i].split(",")]
        i += 1
        if len(header) < 3:
            continue
        storm_id, name = header[0], header[1]
        try:
            count = int(header[2])
        except ValueError:
            continue
        for raw in lines[i : i + count]:
            parts = [part.strip() for part in raw.split(",")]
            if len(parts) < 7:
                continue
            time = pd.to_datetime(parts[0] + parts[1].zfill(4), format="%Y%m%d%H%M", errors="coerce")
            if pd.isna(time):
                continue
            rows.append(
                {
                    "storm_id": storm_id,
                    "name": name,
                    "time": time,
                    "status": parts[3],
                    "lat": _coord(parts[4]),
                    "lon": _coord(parts[5]),
                    "wind_kt": pd.to_numeric(parts[6], errors="coerce"),
                }
            )
        i += count
    return pd.DataFrame(rows).reset_index(drop=True)


def parse_hurdat2(text):
    """Parse raw HURDAT2 text into a track-point DataFrame."""
    return parse(text)


def collect_hurdat2(settings, *, skip_existing=False, smoke=False):
    """Download + parse HURDAT2 to the location's track CSV."""
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
