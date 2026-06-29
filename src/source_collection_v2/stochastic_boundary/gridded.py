from __future__ import annotations

import pandas as pd
import xarray as xr


def coord(ds: xr.Dataset | xr.DataArray, names: tuple[str, ...] | list[str]) -> str:
    for name in names:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(f"dataset lacks coordinate/dimension from {list(names)}")


def open_zarr(url: str, *, chunks=None, consolidated=True) -> xr.Dataset:
    kwargs = {"chunks": chunks or {}, "consolidated": consolidated}
    if str(url).startswith("s3://"):
        kwargs["storage_options"] = {"anon": True}
    return xr.open_dataset(url, engine="zarr", **kwargs)


def subset(ds: xr.Dataset, *, variables=None, bbox=None, start=None, end=None) -> xr.Dataset:
    """Native xarray time/bbox subset for latitude/longitude grids."""
    out = ds[list(variables)] if variables is not None else ds
    indexers = {}
    if start is not None or end is not None:
        t = coord(out, ("time", "valid_time"))
        indexers[t] = slice(None if start is None else pd.Timestamp(start), None if end is None else pd.Timestamp(end))
    if bbox is not None:
        west, south, east, north = map(float, bbox)
        y = coord(out, ("latitude", "lat", "y"))
        x = coord(out, ("longitude", "lon", "x"))
        lon = out[x].values
        if float(lon.min()) >= 0 and west < 0:
            west, east = west % 360, east % 360
        if west > east:
            raise ValueError("longitude bbox crosses the dateline; split the request explicitly")
        indexers[y] = _axis_slice(out[y].values, south, north)
        indexers[x] = _axis_slice(lon, west, east)
    return out.sel(indexers)


def to_yx(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    for src, dst in {"latitude": "y", "lat": "y", "longitude": "x", "lon": "x"}.items():
        if src in da.dims:
            rename[src] = dst
    return da.rename(rename).sortby("y").sortby("x")


def _axis_slice(values, lower, upper):
    return slice(lower, upper) if float(values[0]) <= float(values[-1]) else slice(upper, lower)
