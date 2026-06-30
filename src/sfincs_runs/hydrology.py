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


def ssurgo_infiltration_fields(attributes, *, top_depth_cm=40.0, drainage_condition="undrained", ksat_units="um/s"):
    """Collapse SSURGO horizons to one HSG and harmonic-mean Ksat per map unit."""
    df = _read_table(attributes)
    required = {"mukey", "hydgrp", "ksat_r", "hzdept_r", "hzdepb_r"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"SSURGO attributes missing columns: {sorted(missing)}")
    df = df.copy()
    df["mukey"] = df["mukey"].astype(str)
    for column in ["ksat_r", "hzdept_r", "hzdepb_r"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    hsg = df.groupby("mukey", sort=True)["hydgrp"].agg(lambda s: _select_hsg(s, drainage_condition))
    ksat = _top_depth_harmonic_ksat(df, top_depth_cm=top_depth_cm) * _ksat_factor(ksat_units)
    out = pd.DataFrame({"hsg": hsg}).join(ksat.rename("ksat_mmhr"), how="inner")
    out["hsg_code"] = out["hsg"].map(HSG_CODE)
    out = out.dropna(subset=["hsg_code", "ksat_mmhr"]).reset_index()
    out["hsg_code"] = out["hsg_code"].astype("uint8")
    out["ksat_mmhr"] = out["ksat_mmhr"].astype("float32")
    return out[["mukey", "hsg", "hsg_code", "ksat_mmhr"]]


def write_ssurgo_infiltration_rasters(
    soil_polygons,
    attributes,
    template_raster,
    *,
    hsg_out,
    ksat_out,
    land_domain=None,
    top_depth_cm=40.0,
    drainage_condition="undrained",
    ksat_units="um/s",
    all_touched=False,
):
    """Rasterize SSURGO-derived HSG and Ksat onto a template grid."""
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(template_raster) as template:
        if template.crs is None:
            raise ValueError(f"template_raster has no CRS: {template_raster}")
        out_shape = (template.height, template.width)
        transform = template.transform
        crs = template.crs
        base_profile = template.profile.copy()
    if crs is None:
        raise ValueError(f"template_raster has no CRS: {template_raster}")
    soils = _read_geodataframe(soil_polygons)
    if "mukey" not in soils.columns:
        raise ValueError("soil_polygons must contain a 'mukey' column")
    soils = _match_crs(soils, crs)
    fields = ssurgo_infiltration_fields(
        attributes,
        top_depth_cm=top_depth_cm,
        drainage_condition=drainage_condition,
        ksat_units=ksat_units,
    )
    gdf = (
        soils.assign(mukey=soils["mukey"].astype(str))
        .merge(fields, on="mukey", how="inner")
        .dropna(subset=["hsg_code", "ksat_mmhr"])
    )
    if land_domain is not None:
        land = _match_crs(_read_geodataframe(land_domain), crs)
        gdf = gdf.clip(land)

    hsg_shapes = ((geom, int(value)) for geom, value in zip(gdf.geometry, gdf["hsg_code"]) if geom is not None)
    ksat_shapes = ((geom, float(value)) for geom, value in zip(gdf.geometry, gdf["ksat_mmhr"]) if geom is not None)
    hsg = rasterize(
        hsg_shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )
    ksat = rasterize(
        ksat_shapes,
        out_shape=out_shape,
        transform=transform,
        fill=np.nan,
        dtype="float32",
        all_touched=all_touched,
    )

    hsg_out, ksat_out = Path(hsg_out), Path(ksat_out)
    hsg_out.parent.mkdir(parents=True, exist_ok=True)
    ksat_out.parent.mkdir(parents=True, exist_ok=True)
    _write_single_band_raster(hsg_out, hsg, base_profile, dtype="uint8", nodata=0)
    _write_single_band_raster(ksat_out, ksat, base_profile, dtype="float32", nodata=np.nan)
    return {
        "hsg": str(hsg_out),
        "ksat": str(ksat_out),
        "mapunits": int(len(fields)),
        "rasterized_polygons": int(len(gdf)),
        "hsg_pixels": int((hsg != 0).sum().item()),
        "ksat_pixels": int(np.isfinite(ksat).sum().item()),
    }


def _write_single_band_raster(path, values, template_profile, *, dtype, nodata):
    import rasterio

    profile = {
        **template_profile,
        "driver": "GTiff",
        "count": 1,
        "dtype": dtype,
        "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values.astype(dtype), 1)


def _read_table(data):
    return pd.read_csv(data) if isinstance(data, (str, Path)) else data.copy()


def _read_geodataframe(data):
    import geopandas as gpd

    return gpd.read_file(data) if isinstance(data, (str, Path)) else data.copy()


def _match_crs(gdf, crs):
    return gdf.set_crs(crs) if gdf.crs is None else gdf.to_crs(crs)


def _select_hsg(values, drainage_condition):
    for raw in values.dropna():
        parts = [part.strip().upper()[:1] for part in str(raw).split("/") if part.strip()]
        parts = [part for part in parts if part in HSG_CODE]
        if not parts:
            continue
        if drainage_condition == "drained":
            return parts[0]
        if drainage_condition == "undrained":
            return parts[-1]
        if drainage_condition == "conservative":
            return max(parts, key=HSG_CODE.__getitem__)
        raise ValueError("drainage_condition must be 'drained', 'undrained', or 'conservative'")
    return None


def _top_depth_harmonic_ksat(df, *, top_depth_cm):
    horizons = df.dropna(subset=["mukey", "ksat_r", "hzdept_r", "hzdepb_r"]).copy()
    horizons = horizons.loc[horizons["ksat_r"] > 0]
    top = horizons["hzdept_r"].clip(0, top_depth_cm)
    bottom = horizons["hzdepb_r"].clip(0, top_depth_cm)
    horizons["thickness"] = (bottom - top).clip(lower=0)
    horizons = horizons.loc[horizons["thickness"] > 0]
    horizons["thickness_over_ksat"] = horizons["thickness"] / horizons["ksat_r"]
    sums = horizons.groupby("mukey").agg(thickness=("thickness", "sum"), resistance=("thickness_over_ksat", "sum"))
    return sums["thickness"] / sums["resistance"]


def _ksat_factor(units):
    if units == "um/s":
        return 3.6
    if units == "mm/hr":
        return 1.0
    raise ValueError("ksat_units must be 'um/s' or 'mm/hr'")
