from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from pyproj import Transformer

from design_events.stochastic_boundary.audit import Artifact, covers, nonempty, resolve, write_artifact


def boundary_center(paths: dict, spec: dict) -> tuple[float, float]:
    if spec.get("lon") is not None and spec.get("lat") is not None:
        return float(spec["lon"]), float(spec["lat"])
    boundary = Path(spec.get("boundary_file") or paths["sfincs_boundary_file"])
    xy = np.loadtxt(boundary, ndmin=2)[:, :2]
    crs = spec.get("boundary_points_crs", "EPSG:26919")
    return tuple(map(float, Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xy[:, 0].mean(), xy[:, 1].mean())))


def collect(settings: dict, *, skip_existing=False) -> Artifact:
    paths, spec = settings["paths"], settings["spec"]
    start, end = pd.Timestamp(settings["start"]), pd.Timestamp(settings["end"])
    out = Path(paths.get("waterlevel_csv") or resolve(paths, spec.get("output", "data/sources/cora/boundary_water_level.csv")))
    if skip_existing and nonempty(out) and covers(paths, "cora", "boundary_water_level", start, end):
        return Artifact("cora", "boundary_water_level", start, end, {"waterlevel_csv": out}, {"reused": True, "rows": len(pd.read_csv(out))})
    lon, lat = boundary_center(paths, spec)
    node = _nearest_wet_node(spec, pd.Timestamp(start), lon, lat)
    dates = pd.date_range(start.floor("D"), end.floor("D"), freq="D")
    workers = max(1, int(spec.get("parallel_workers", 4)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(lambda d: _day_frame(spec, d, node), dates))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.dropna().drop_duplicates("time").sort_values("time")
    frame = frame[(frame.time >= start) & (frame.time <= end)]
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    artifact = Artifact("cora", "boundary_water_level", start, end, {"waterlevel_csv": out}, {"rows": len(frame), "node": node, "lon": lon, "lat": lat, "units": spec.get("units", "m"), "datum": spec.get("datum", "MSL")})
    write_artifact(paths, artifact)
    return artifact


def _url(spec: dict, date: pd.Timestamp) -> str:
    return f"https://{spec['s3_bucket']}.s3.amazonaws.com/{spec['s3_key_pattern'].format(date=date.to_pydatetime())}"


def _open_day(spec: dict, date: pd.Timestamp) -> xr.Dataset:
    response = requests.get(_url(spec, date), timeout=int(spec.get("request_timeout_seconds", 60)))
    response.raise_for_status()
    return xr.open_dataset(io.BytesIO(response.content), engine=spec.get("engine", "h5netcdf"))


def _nearest_wet_node(spec: dict, date: pd.Timestamp, lon: float, lat: float) -> int:
    with _open_day(spec, date) as ds:
        dist2 = (ds["lon"].values - lon) ** 2 + (ds["lat"].values - lat) ** 2
        order = np.argsort(dist2)[: int(spec.get("nearest_k", 20))]
        sample = ds[spec.get("variable", "zeta")].isel(nodes=order.tolist()).load().values
        wet = ~np.isnan(sample).all(axis=1)
    return int(order[int(np.argmax(wet))])


def _day_frame(spec: dict, date: pd.Timestamp, node: int) -> pd.DataFrame:
    with _open_day(spec, date) as ds:
        return pd.DataFrame({"time": pd.to_datetime(ds["time"].values), "value": ds[spec.get("variable", "zeta")].isel(nodes=node).values.astype(float)})
