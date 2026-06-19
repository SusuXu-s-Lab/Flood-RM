from __future__ import annotations
import io
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import numpy as np
import pandas as pd
import requests
import xarray as xr
from pyproj import Transformer

from design_events.utils import iter_progress
from design_events.utils import write_source_artifact


# prevent reruns if data already exists
def _has_data(path):
    return path.exists() and path.stat().st_size > 0

# clean dataframe
def _clean_frame(frame):
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["time", "value"])
    frame = frame.drop_duplicates(subset=["time"]).sort_values("time")
    return frame.reset_index(drop=True)


def _read_existing_frame(path):
    return _clean_frame(pd.read_csv(path, parse_dates=["time"]))


def _covers_window(frame, start, end):
    if frame.empty:
        return False
    return frame["time"].min() <= pd.Timestamp(start) and frame["time"].max() >= pd.Timestamp(end)


# prepare so that cora data is averaged for deployment along SFINCS boyndary line
def boundary_center(paths, crs):
    # average boundary points, then convert to lon/lat for Cora.
    points = []
    for line in paths["sfincs_boundary_file"].read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            points.append([float(parts[0]), float(parts[1])])
    points = np.asarray(points, dtype=float)
    x = float(points[:, 0].mean())
    y = float(points[:, 1].mean())
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
    return float(lon), float(lat)

# AWS bucket
def cora_url(cora, date):
    key = cora["s3_key_pattern"].format(date=date.to_pydatetime())
    return f"https://{cora['s3_bucket']}.s3.amazonaws.com/{key}"

def daily_cache_path(paths, cora, date):
    if not cora.get("raw_cache_enabled", False):
        return None
    cache_name = cora.get("raw_cache_dirname", "cora_daily_nc")
    return paths["cache_root"] / cache_name / f"{date:%Y}" / f"cora_{date:%Y%m%d}.nc"

def download_day(paths, cora, date):
    # Reuse daily NetCDF files when present; otherwise pull directly from NOAA S3.
    cache_path = daily_cache_path(paths, cora, date)
    if cache_path is not None and cache_path.exists():
        return cache_path.read_bytes()
    response = requests.get(cora_url(cora, date), timeout=int(cora.get("request_timeout_seconds", 60)))
    response.raise_for_status()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(response.content)
    return response.content

@contextmanager
def open_cora_dataset(payload):
    try:
        with xr.open_dataset(io.BytesIO(payload), engine="h5netcdf") as ds:
            yield ds
    except ModuleNotFoundError:
        tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
        try:
            tmp.write(payload)
            tmp.close()
            with xr.open_dataset(tmp.name, engine="netcdf4") as ds:
                yield ds
        finally:
            try:
                os.unlink(tmp.name)
            except FileNotFoundError:
                pass

def read_day(paths, cora, date, node):
    payload = download_day(paths, cora, date)
    with open_cora_dataset(payload) as ds:
        series = ds[cora.get("variable", "zeta")].isel(nodes=node).load()
        return pd.DataFrame({
            "time": pd.to_datetime(ds["time"].values),
            "value": series.values.astype(float),
        })


def read_days_with_progress(paths, cora, dates, node, workers):
    # Track completed remote daily files so long production pulls expose ETA.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(read_day, paths, cora, date, node): date
            for date in dates
        }
        completed = iter_progress(
            as_completed(futures),
            total=len(futures),
            desc="CORA daily files",
            unit="day",
            dynamic_ncols=True,
        )
        frames = []
        for future in completed:
            date = futures[future]
            if hasattr(completed, "set_postfix_str"):
                completed.set_postfix_str(date.strftime("%Y-%m-%d"), refresh=False)
            frames.append(future.result())
    return frames


def snap_node(paths, cora, date, lon, lat):
    # Use the nearest wet Cora node to the Sfincs boundary center.
    payload = download_day(paths, cora, date)
    with open_cora_dataset(payload) as ds:
        lon_all = ds["lon"].values
        lat_all = ds["lat"].values
        dist2 = (lon_all - lon) ** 2 + (lat_all - lat) ** 2
        order = np.argsort(dist2)[:int(cora.get("nearest_k", 20))]
        sample = ds[cora.get("variable", "zeta")].isel(nodes=order.tolist()).load().values
    wet = ~np.isnan(sample).all(axis=1)
    node = int(order[int(np.argmax(wet))])
    print(f"cora node {node}: lon={lon_all[node]:.4f}, lat={lat_all[node]:.4f}")
    return node

def collect_cora(settings, skip_existing=False, smoke=False):
    # Main Cora pull: date range -> daily files -> single boundary-node CSV.
    paths = settings["paths"]
    cora = settings["cora"]
    output_csv = paths["waterlevel_csv"]
    # formatting for compatibility
    start = pd.Timestamp("2018-01-01") if smoke else settings["start"]
    end = pd.Timestamp("2018-01-31") if smoke else settings["end"]
    reuse_existing = (skip_existing or cora.get("reuse_existing", False)) and not smoke
    if reuse_existing and _has_data(output_csv):
        frame = _read_existing_frame(output_csv)
        if _covers_window(frame, start, end):
            print(
                "CORA water level: reusing "
                f"{output_csv} ({len(frame):,} hourly rows, "
                f"{frame['time'].min()} to {frame['time'].max()})"
            )
            _write_manifest(paths, cora, start, end, output_csv)
            return frame
        if cora.get("reuse_existing", False):
            raise ValueError(
                f"configured CORA output does not cover {start} to {end}: {output_csv} "
                f"covers {frame['time'].min()} to {frame['time'].max()}"
            )
        print(f"CORA water level: existing file does not cover {start} to {end}; redownloading")
    dates = pd.date_range(start, end, freq="D")
    lon, lat = boundary_center(paths, cora.get("boundary_points_crs", "EPSG:26919"))
    node = snap_node(paths, cora, dates[0], lon, lat)
    workers = int(cora.get("parallel_workers", 8))
    print(f"fetching {len(dates):,} Cora days with {workers} workers")
    frames = read_days_with_progress(paths, cora, dates, node, workers)
    frame = _clean_frame(pd.concat(frames, ignore_index=True))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    _write_manifest(paths, cora, start, end, output_csv)
    print(f"wrote {output_csv} ({len(frame):,} rows)")
    return frame


def _write_manifest(paths, cora, start, end, output_csv):
    write_source_artifact(
        paths,
        source="cora",
        kind="boundary_water_level",
        start=start,
        end=end,
        artifacts={"waterlevel_csv": output_csv},
        metadata={
            "datum": cora.get("datum", "MSL"),
            "units": cora.get("units", "m"),
            "variable": cora.get("variable", "zeta"),
            "s3_bucket": cora.get("s3_bucket"),
            "s3_key_pattern": cora.get("s3_key_pattern"),
        },
    )
