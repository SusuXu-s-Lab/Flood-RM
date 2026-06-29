from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HSG_CODE = {"A": 1, "B": 2, "C": 3, "D": 4}


def cn_recovery_seff(smax, soil_saturation, *, units: str = "fraction"):
    """Initial effective storage for SFINCS CN-with-recovery.

    ``seff = clip(theta, 0, 1) * smax`` where ``theta`` is top-layer soil
    saturation as a fraction or percent.
    """
    if units not in {"fraction", "percent"}:
        raise ValueError("units must be 'fraction' or 'percent'")
    frac = soil_saturation / 100 if units == "percent" else soil_saturation
    out = (frac.clip(0, 1) * smax).rename("seff")
    out.attrs.update(units="m", long_name="initial effective soil moisture storage")
    return out


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
    """Write HydroMT-SFINCS-ready interval precipitation ``precip(time, y, x)``.

    The field-preserving event realization is

    ``P_e(x,y,t) = K_e * P_analog(x,y,t)``.

    The output remains cumulative interval precipitation in mm when
    ``cumulative_input=True`` is later passed to ``sf.precipitation.create``.
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


def stage_seff_from_fraction(run_root: str | Path, fraction: float, *, smax_name: str = "sfincs.smax") -> dict[str, float | str]:
    """Write event-specific ``sfincs.seff`` from native CN-recovery ``smax``.

    HydroMT-SFINCS owns the static ``smax`` and ``ks`` files.  This event helper
    only updates the mutable antecedent storage file.
    """
    run_root = Path(run_root)
    smax_path = run_root / smax_name
    if not smax_path.exists():
        raise FileNotFoundError(smax_path)
    value = float(np.clip(fraction, 0.0, 1.0))
    smax = np.fromfile(smax_path, dtype="<f4")
    seff = (smax * value).astype("<f4")
    out = run_root / "sfincs.seff"
    seff.tofile(out)
    return {"sefffile": out.name, "initial_soil_moisture_fraction": value}


def condition_ksat_raster(source, output, *, scale_factor: float = 1.0, max_mmhr: float | None = None):
    """Scale and optionally cap a Ksat raster while preserving georeferencing."""
    import rioxarray as rxr

    source = Path(source)
    output = Path(output)
    if not source.exists():
        raise FileNotFoundError(source)
    da = rxr.open_rasterio(source, masked=True).squeeze(drop=True).astype("float32")
    valid = da.notnull()
    scaled = da * float(scale_factor)
    capped = valid & (scaled > float(max_mmhr)) if max_mmhr is not None else xr.zeros_like(valid, dtype=bool)
    conditioned = scaled.where(valid)
    if max_mmhr is not None:
        conditioned = conditioned.clip(max=float(max_mmhr))
    output.parent.mkdir(parents=True, exist_ok=True)
    conditioned = conditioned.astype("float32").rio.write_nodata(np.nan)
    conditioned.rio.to_raster(output, compress="deflate")
    valid_pixels = int(valid.sum().item())
    capped_pixels = int(capped.sum().item())
    return {
        "ksat": str(output),
        "source_ksat": str(source),
        "scale_factor": float(scale_factor),
        "max_mmhr": None if max_mmhr is None else float(max_mmhr),
        "valid_pixels": valid_pixels,
        "capped_fraction": capped_pixels / max(valid_pixels, 1),
    }
