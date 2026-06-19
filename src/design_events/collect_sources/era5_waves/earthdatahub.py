"""Fetch ERA5 ocean-wave variables from Earth Data Hub Zarr."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import xarray as xr

from design_events.utils import progress_bar


earthdatahub_era5_ocean_zarr = (
    "https://api.earthdatahub.destine.eu/era5/"
    "reanalysis-era5-single-levels-ocean-v0.zarr"
)

earthdatahub_wave_variables = ["swh", "pp1d", "mwd", "wdw"]


def normalize_earthdatahub_zarr_url(url=None):
    if not url:
        return earthdatahub_era5_ocean_zarr
    url = str(url).rstrip("/")
    if url.endswith(".zarr"):
        return url
    if url == "https://api.earthdatahub.destine.eu":
        return earthdatahub_era5_ocean_zarr
    return f"{url}/era5/reanalysis-era5-single-levels-ocean-v0.zarr"


def read_earthdatahub_auth(path):
    auth = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        label, separator, value = line.partition(":")
        if not separator:
            continue
        label = label.strip().lower().replace("-", "_")
        value = value.strip()
        if label in {"url", "base_url", "zarr_url"}:
            auth["url"] = value
        if label in {"key", "token", "api_key"}:
            auth["token"] = value
    return auth


def earthdatahub_store_url(token=None, base_url=earthdatahub_era5_ocean_zarr):
    base_url = normalize_earthdatahub_zarr_url(base_url)
    token = token or os.environ.get("EARTHDATAHUB_TOKEN")
    if not token:
        return base_url
    return base_url.replace("https://", f"https://edh:{token}@")


def earthdatahub_storage_options(token=None):
    if token or os.environ.get("EARTHDATAHUB_TOKEN"):
        return None
    return {"client_kwargs": {"trust_env": True}}


def _coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(f"ERA5 wave Zarr missing coordinate: one of {candidates}")


def _slice_axis(values, lower, upper):
    first = float(values[0])
    last = float(values[-1])
    if first <= last:
        return slice(lower, upper)
    return slice(upper, lower)


def _longitude_bounds(values, west, east):
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum >= 0 and west < 0:
        west = west % 360
        east = east % 360
    if west > east:
        raise ValueError("ERA5 wave longitude selection crosses the dateline")
    return west, east


def subset_earthdatahub_waves(ds, bbox_wgs84, time_window, variables=None):
    west, south, east, north = bbox_wgs84
    start, end = pd.Timestamp(time_window[0]), pd.Timestamp(time_window[1])
    selected_variables = list(variables or earthdatahub_wave_variables)

    time_name = _coord_name(ds, ["time", "valid_time"])
    lat_name = _coord_name(ds, ["latitude", "lat"])
    lon_name = _coord_name(ds, ["longitude", "lon"])

    missing = [name for name in selected_variables if name not in ds.data_vars]
    if missing:
        raise ValueError(f"Earth Data Hub ERA5 wave Zarr missing variables: {missing}")

    west, east = _longitude_bounds(ds[lon_name].values, west, east)
    return ds[selected_variables].sel(
        {
            time_name: slice(start, end),
            lat_name: _slice_axis(ds[lat_name].values, south, north),
            lon_name: _slice_axis(ds[lon_name].values, west, east),
        }
    )


def open_earthdatahub_waves(url=None, token=None):
    return xr.open_dataset(
        earthdatahub_store_url(token=token, base_url=url or earthdatahub_era5_ocean_zarr),
        chunks={},
        engine="zarr",
        storage_options=earthdatahub_storage_options(token=token),
    )


def chunk_time_windows(start, end, chunk_months=12):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if chunk_months is None or int(chunk_months) <= 0:
        return [(start, end)]
    windows = []
    current = start
    while current <= end:
        chunk_end = min(current + pd.DateOffset(months=int(chunk_months)) - pd.Timedelta(hours=1), end)
        windows.append((current, chunk_end))
        current = chunk_end + pd.Timedelta(hours=1)
    return windows


def _time_name(ds):
    return _coord_name(ds, ["time", "valid_time"])


def _write_first_chunk(path, chunk, time_name):
    chunk.to_netcdf(path, unlimited_dims=[time_name])


def _append_chunk(path, chunk, time_name):
    from netCDF4 import Dataset, date2num

    loaded = chunk.load()
    append_count = int(loaded.sizes.get(time_name, 0))
    if append_count == 0:
        return
    with Dataset(path, "a") as target:
        start_index = len(target.dimensions[time_name])
        time_var = target.variables[time_name]
        calendar = getattr(time_var, "calendar", "standard")
        encoded_times = date2num(
            pd.to_datetime(loaded[time_name].values).to_pydatetime(),
            units=time_var.units,
            calendar=calendar,
        )
        time_var[start_index : start_index + append_count] = encoded_times
        for name in loaded.data_vars:
            target_var = target.variables[name]
            time_axis = target_var.dimensions.index(time_name)
            key = [slice(None)] * target_var.ndim
            key[time_axis] = slice(start_index, start_index + append_count)
            target_var[tuple(key)] = loaded[name].values


def _write_wave_chunks(ds, bbox_wgs84, time_window, output_path, variables, chunk_months):
    start, end = pd.Timestamp(time_window[0]), pd.Timestamp(time_window[1])
    windows = chunk_time_windows(start, end, chunk_months=chunk_months)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    if temp_path.exists():
        temp_path.unlink()
    wrote_first = False
    with progress_bar(total=len(windows), desc="Earth Data Hub wave chunks", unit="chunk", dynamic_ncols=True) as progress:
        for index, window in enumerate(windows, start=1):
            progress.set_postfix_str(f"{window[0]:%Y-%m-%d} to {window[1]:%Y-%m-%d}", refresh=False)
            chunk = subset_earthdatahub_waves(ds, bbox_wgs84, window, variables)
            time_name = _time_name(chunk)
            if not wrote_first:
                _write_first_chunk(temp_path, chunk, time_name)
                wrote_first = True
            else:
                _append_chunk(temp_path, chunk, time_name)
            progress.update()
    if not wrote_first:
        raise ValueError("Earth Data Hub wave subset is empty for requested bbox/time window")
    temp_path.replace(output_path)


def fetch_era5_waves_from_earthdatahub(
    bbox_wgs84,
    time_window,
    output_path,
    variables=None,
    force=False,
    url=None,
    token=None,
    auth_path=None,
    opener=None,
    chunk_months=12,
):
    output_path = Path(output_path)
    if output_path.exists() and not force:
        print(f"skip: {output_path} already exists (use --force to overwrite)")
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    open_dataset = opener or open_earthdatahub_waves
    if auth_path is not None:
        auth = read_earthdatahub_auth(auth_path)
        url = url or auth.get("url")
        token = token or auth.get("token")

    with progress_bar(total=3, desc="Earth Data Hub waves", unit="stage", dynamic_ncols=True) as progress:
        progress.set_postfix_str("open zarr", refresh=False)
        with open_dataset(url=url, token=token) as ds:
            progress.update()
            windows = chunk_time_windows(time_window[0], time_window[1], chunk_months=chunk_months)
            progress.set_postfix_str(f"subset/write {len(windows)} chunks", refresh=False)
            _write_wave_chunks(ds, bbox_wgs84, time_window, output_path, variables, chunk_months)
            progress.update()
            progress.set_postfix_str(f"wrote {output_path.name}", refresh=False)
            progress.update()
    return output_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Fetch ERA5 ocean waves from Earth Data Hub.")
    parser.add_argument("--bbox", required=True, nargs=4, type=float, metavar=("W", "S", "E", "N"))
    parser.add_argument("--start", required=True, type=pd.Timestamp)
    parser.add_argument("--stop", required=True, type=pd.Timestamp)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--url", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--auth-path", default=None, type=Path)
    parser.add_argument("--chunk-months", default=12, type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    fetch_era5_waves_from_earthdatahub(
        bbox_wgs84=tuple(args.bbox),
        time_window=(args.start, args.stop),
        output_path=args.out,
        force=args.force,
        url=args.url,
        token=args.token,
        auth_path=args.auth_path,
        chunk_months=args.chunk_months,
    )


if __name__ == "__main__":
    main()
