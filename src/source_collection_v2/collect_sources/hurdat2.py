from __future__ import annotations

import re

import pandas as pd
import requests

from design_events.stochastic_boundary.audit import Artifact, nonempty, resolve, write_artifact

URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt"
HEADER = re.compile(r"^[A-Z]{2}\d{6}$")


def parse(text: str) -> pd.DataFrame:
    storm_id = name = None
    rows = []
    for line in text.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 3 and HEADER.match(f[0]):
            storm_id, name = f[0], f[1]
        elif storm_id and len(f) >= 7:
            rows.append({"storm_id": storm_id, "name": name, "time": pd.to_datetime(f[0] + f[1], format="%Y%m%d%H%M", errors="coerce"), "status": f[3], "lat": _signed(f[4]), "lon": _signed(f[5]), "wind_kt": pd.to_numeric(f[6], errors="coerce")})
    return pd.DataFrame(rows).dropna(subset=["time"])


def collect(settings: dict, *, skip_existing=False) -> Artifact:
    paths, spec = settings["paths"], settings["spec"]
    out = paths.get("hurdat2_tracks_csv") or resolve(paths, spec.get("output", "data/sources/hurdat2/tracks.csv"))
    if skip_existing and nonempty(out):
        return Artifact("hurdat2", "tracks", None, None, {"tracks_csv": out}, {"reused": True, "rows": len(pd.read_csv(out))})
    response = requests.get(spec.get("url", URL), timeout=int(spec.get("request_timeout_seconds", 60)))
    response.raise_for_status()
    tracks = parse(response.text)
    out.parent.mkdir(parents=True, exist_ok=True); tracks.to_csv(out, index=False)
    artifact = Artifact("hurdat2", "tracks", tracks.time.min(), tracks.time.max(), {"tracks_csv": out}, {"track_points": len(tracks), "storms": int(tracks.storm_id.nunique())})
    write_artifact(paths, artifact)
    return artifact


def _signed(value: str) -> float:
    value = value.strip()
    return (-1 if value[-1] in "SW" else 1) * float(value[:-1])
