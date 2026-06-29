from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from .build import open_model
from .io import config_set, model_root_path, parse_sfincs_inp, read_json, write_json
from .schema import SnapWaveForcing

ERA5_SNAPWAVE_MAP = {
    "bhs": "swh",   # significant wave height [m]
    "btp": "pp1d",  # peak wave period [s]
    "bwd": "mwd",   # mean wave direction [deg]
    "bds": "wdw",   # directional spread [deg or rad depending dataset]
}


def unwrap_direction_degrees(values) -> pd.Series:
    series = pd.Series(values)
    radians = np.deg2rad(series.astype(float).to_numpy())
    return pd.Series(np.rad2deg(np.unwrap(radians)), index=series.index)


def _snapwave_component_gdf(sf) -> gpd.GeoDataFrame | None:
    component = getattr(sf, "snapwave_boundary_conditions", None)
    for attr in ("gdf", "data"):
        value = getattr(component, attr, None)
        if isinstance(value, gpd.GeoDataFrame) and not value.empty:
            return value.copy()
    return None


def snapwave_boundary_points(sf) -> gpd.GeoDataFrame:
    """Return native SFINCS/SnapWave boundary points.

    Prefer the HydroMT-SFINCS component data.  If the model was already written,
    fall back to ``snapwave.bnd`` as an event-time reader only.
    """
    gdf = _snapwave_component_gdf(sf)
    if gdf is not None:
        if "name" not in gdf:
            gdf["name"] = [f"{i:04d}" for i in range(1, len(gdf) + 1)]
        return gdf

    root = model_root_path(sf)
    bnd = root / "snapwave.bnd"
    if not bnd.exists():
        raise FileNotFoundError(
            "No native SnapWave boundary points found. Build the base with HydroMT-SFINCS snapwave components first."
        )
    rows = []
    for index, line in enumerate(bnd.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        parts = line.split()
        if len(parts) < 2:
            continue
        rows.append({"name": f"{index:04d}", "x": float(parts[0]), "y": float(parts[1])})
    if not rows:
        raise RuntimeError(f"No points in {bnd}")

    epsg_text = parse_sfincs_inp(root / "sfincs.inp").get("epsg")
    crs = f"EPSG:{epsg_text}" if epsg_text else getattr(sf, "crs", None)
    gdf = gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy([r["x"] for r in rows], [r["y"] for r in rows]), crs=crs)
    return gdf


def _time_coord(ds: xr.Dataset) -> str:
    for candidate in ("valid_time", "time"):
        if candidate in ds.coords or candidate in ds.dims:
            return candidate
    raise KeyError("ERA5 wave dataset has no valid_time/time coordinate")


def _nearest_finite_series(ds: xr.Dataset, variable: str, *, longitude: float, latitude: float, time_name: str) -> np.ndarray:
    sample = ds[variable].sel(longitude=longitude, latitude=latitude, method="nearest")
    values = np.asarray(sample.values, dtype=float)
    if np.isfinite(values).any():
        return values

    da = ds[variable].transpose(time_name, "latitude", "longitude")
    cube = np.asarray(da.values, dtype=float)
    finite_cells = np.isfinite(cube).any(axis=0)
    if not finite_cells.any():
        return values
    lon_grid, lat_grid = np.meshgrid(da["longitude"].values.astype(float), da["latitude"].values.astype(float))
    distance = (lon_grid - longitude) ** 2 + (lat_grid - latitude) ** 2
    distance = np.where(finite_cells, distance, np.inf)
    iy, ix = np.unravel_index(int(np.nanargmin(distance)), distance.shape)
    return cube[:, iy, ix]


def era5_to_snapwave_forcing(
    era5: str | Path | xr.Dataset,
    points: gpd.GeoDataFrame,
    *,
    start,
    stop,
    variables: dict[str, str] | None = None,
) -> SnapWaveForcing:
    """Sample ERA5 wave variables at native SnapWave boundary points."""
    variables = variables or ERA5_SNAPWAVE_MAP
    opened = xr.open_dataset(era5) if isinstance(era5, (str, Path)) else era5
    close = isinstance(era5, (str, Path))
    try:
        time_name = _time_coord(opened)
        window = opened.sel({time_name: slice(pd.Timestamp(start), pd.Timestamp(stop))})
        if window.sizes.get(time_name, 0) == 0:
            raise RuntimeError(f"No ERA5 wave records in {start}..{stop}")
        points_wgs84 = points.set_crs("EPSG:4326") if points.crs is None else points.to_crs("EPSG:4326")
        frames: dict[str, pd.DataFrame] = {}
        for snap_key, era5_name in variables.items():
            if era5_name not in window:
                raise KeyError(f"{era5_name!r} not found in ERA5 wave dataset")
            per_point = {}
            for _, point in points_wgs84.iterrows():
                name = str(point.get("name", f"{len(per_point) + 1:04d}"))
                per_point[name] = _nearest_finite_series(
                    window,
                    era5_name,
                    longitude=float(point.geometry.x),
                    latitude=float(point.geometry.y),
                    time_name=time_name,
                )
            frames[snap_key] = pd.DataFrame(per_point, index=pd.DatetimeIndex(window[time_name].values, name="time"))
        frames["bds"] = _directional_spread_degrees(frames["bds"])
        return SnapWaveForcing(**frames)  # type: ignore[arg-type]
    finally:
        if close:
            opened.close()


def _directional_spread_degrees(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size and float(np.nanmax(np.abs(finite))) <= (2 * np.pi + 1e-6):
        return frame.apply(np.rad2deg)
    return frame


def write_snapwave_forcing_tables(
    run_root: str | Path,
    forcing: SnapWaveForcing,
    *,
    reference_time=None,
    start_seconds: float = 0.0,
) -> dict[str, str]:
    """Write ``snapwave.bhs/btp/bwd/bds`` tables.

    HydroMT-SFINCS is still used for SnapWave mask and boundary geometry.  These
    four event-varying tables remain a thin SFINCS table writer until the
    installed HydroMT-SFINCS component exposes arbitrary multi-point wave
    time-series input.
    """
    run_root = Path(run_root)
    written: dict[str, str] = {}
    for key, frame in forcing.frames().items():
        frame = frame.copy().astype(float)
        if frame.empty:
            raise ValueError(f"SnapWave {key} table is empty")
        times = pd.DatetimeIndex(pd.to_datetime(frame.index), name="time")
        ref = pd.Timestamp(reference_time) if reference_time is not None else times[0]
        seconds = ((times - ref) / pd.Timedelta(seconds=1)).to_numpy(dtype=float) + float(start_seconds)
        values = np.column_stack([seconds, frame.to_numpy(dtype=float)])
        path = run_root / f"snapwave.{key}"
        path.write_text("\n".join(" ".join(f"{value:.3f}" for value in row) for row in values) + "\n", encoding="utf-8")
        written[f"snapwave_{key}file"] = path.name
    return written


def _driver_start_seconds(run_root: Path, driver: str) -> float:
    manifest = read_json(run_root / "forcing_manifest.json")
    run_start = manifest.get("run_start")
    reference = manifest.get("event_reference_time")
    if not run_start or not reference:
        return 0.0
    for window in manifest.get("driver_windows", []):
        if window.get("driver") != driver:
            continue
        start = pd.Timestamp(reference) + pd.Timedelta(hours=float(window["start_offset_hours"]))
        return float((start - pd.Timestamp(run_start)) / pd.Timedelta(seconds=1))
    return 0.0


def stage_snapwave_event(
    run_root: str | Path,
    *,
    era5_wave_nc: str | Path,
    start,
    stop,
    reference_time=None,
    variables: dict[str, str] | None = None,
    update_manifest: bool = True,
) -> dict[str, str]:
    """Stage SnapWave event forcing on an existing SFINCS/SnapWave run folder."""
    run_root = Path(run_root)
    sf = open_model(run_root, mode="r+", read=True)
    points = snapwave_boundary_points(sf)
    forcing = era5_to_snapwave_forcing(era5_wave_nc, points, start=start, stop=stop, variables=variables)
    start_seconds = _driver_start_seconds(run_root, "wave")
    written = write_snapwave_forcing_tables(run_root, forcing, reference_time=reference_time or start, start_seconds=start_seconds)

    for key in ("bhs", "btp", "bwd", "bds"):
        config_set(sf, f"{key}file", f"snapwave.{key}")
    sf.write()

    if update_manifest:
        manifest = read_json(run_root / "forcing_manifest.json")
        manifest.update(
            {
                "snapwave": True,
                "snapwave_member_file": str(era5_wave_nc),
                "snapwave_valid_start_time": pd.Timestamp(start).strftime("%Y-%m-%dT%H:%M:%S"),
                "snapwave_valid_end_time": pd.Timestamp(stop).strftime("%Y-%m-%dT%H:%M:%S"),
                "snapwave_boundary_point_count": int(len(points)),
                **written,
            }
        )
        write_json(run_root / "forcing_manifest.json", manifest)
    return written
