from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr


def find_aorc_event_window(event_windows_dir, *, member_id=None, storm_start=None):
    """Find one local AORC storm-window NetCDF by rank and/or start hour."""
    root = Path(event_windows_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    patterns = []
    if member_id:
        match = re.search(r"rank(\d{4})", str(member_id))
        if match is None:
            raise ValueError(f"Could not parse rank#### from member_id={member_id!r}")
        patterns.append(f"*rank{match.group(1)}_*.nc")
    if storm_start is not None:
        patterns.append(f"*_{pd.Timestamp(storm_start):%Y%m%dT%H}.nc")
    patterns = patterns or ["*.nc"]

    matches = [set(root.glob(pattern)) for pattern in patterns]
    candidates = sorted(set.intersection(*matches) if len(matches) > 1 else matches[0])
    if len(candidates) != 1:
        sample = ", ".join(path.name for path in candidates[:8]) or "no matches"
        raise RuntimeError(f"AORC event-window lookup expected 1 match; got {sample}")
    return candidates[0]


def prepare_aorc_precip_for_sfincs(
    source_nc: str | Path,
    output_nc: str | Path,
    *,
    t_start,
    t_stop,
    variable: str = "APCP_surface",
    freq: str = "1h",
    window_alignment: str = "start",
    precip_start=None,
    scale_factor: float = 1.0,
) -> Path:
    """Write HydroMT-ready interval precipitation ``precip(time, y, x)``.

    The field-preserving event realization is

    ``P_e(x,y,t) = K_e * P_analog(x,y,t)``.

    The output remains cumulative interval precipitation in mm when
    ``cumulative_input=True`` is later passed to HydroMT-SFINCS.
    """
    scale = float(scale_factor)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale_factor must be finite and positive, got {scale_factor!r}")
    source_nc = Path(source_nc)
    output_nc = Path(output_nc)
    if not source_nc.exists():
        raise FileNotFoundError(source_nc)

    full_time = pd.date_range(pd.Timestamp(t_start), pd.Timestamp(t_stop), freq=freq)
    with xr.open_dataset(source_nc) as ds:
        if variable not in ds:
            raise KeyError(f"{variable!r} not found in {source_nc}")
        da = ds[variable]
        rename = {old: new for old, new in {"latitude": "y", "longitude": "x"}.items() if old in da.dims}
        da = da.rename(rename).rename("precip")
        missing = {"time", "y", "x"}.difference(da.dims)
        if missing:
            raise ValueError(f"precipitation is missing dimensions: {sorted(missing)}")
        if window_alignment == "wettest":
            da = _wettest_window(da, len(full_time))
        elif window_alignment != "start":
            raise ValueError("window_alignment must be 'start' or 'wettest'")

        start_time = pd.Timestamp(precip_start) if precip_start is not None else pd.Timestamp(t_start)
        da = da.assign_coords(time=pd.date_range(start_time, periods=da.sizes["time"], freq=freq))
        precip = (da.sortby("y").sortby("x").reindex(time=full_time, fill_value=0.0) * scale).astype("float32")
        precip.attrs.update(units="mm", crs="EPSG:4326", applied_scale_factor=scale)
        output_nc.parent.mkdir(parents=True, exist_ok=True)
        out = precip.to_dataset()
        out.attrs.update(crs="EPSG:4326", source=str(source_nc))
        out.to_netcdf(output_nc)
    return output_nc


def _wettest_window(da, steps: int):
    if steps <= 0:
        raise ValueError("steps must be positive")
    if da.sizes["time"] <= steps:
        return da
    spatial_dims = [dim for dim in da.dims if dim != "time"]
    totals = da.sum(dim=spatial_dims, skipna=True).to_numpy()
    start = int(np.nanargmax(np.convolve(totals, np.ones(steps), mode="valid")))
    return da.isel(time=slice(start, start + steps))
