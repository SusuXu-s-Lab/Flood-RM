from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr


def compute_cn_recovery_seff(
    smax: xr.DataArray,
    soilsat_top: xr.DataArray,
    *,
    input_units: str = "fraction",
) -> xr.DataArray:
    """Compute CN-with-recovery initial storage from NWM top saturation."""
    if input_units not in {"fraction", "percent"}:
        raise ValueError("input_units must be 'fraction' or 'percent'")
    fraction = soilsat_top / 100.0 if input_units == "percent" else soilsat_top
    seff = (fraction.clip(min=0.0, max=1.0) * smax).rename("seff")
    seff.attrs.update({"units": "m", "long_name": "initial effective soil moisture storage"})
    return seff


def aggregate_ssurgo_infiltration_fields(
    attributes,
    *,
    top_depth_cm: float = 40.0,
    drainage_condition: str = "undrained",
    ksat_units: str = "um/s",
) -> pd.DataFrame:
    """Collapse SSURGO horizon rows to one HSG/Ksat record per map unit."""
    frame = pd.read_csv(attributes) if isinstance(attributes, (str, Path)) else attributes.copy()
    required = {"mukey", "hydgrp", "ksat_r", "hzdept_r", "hzdepb_r"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"SSURGO attributes missing required columns: {sorted(missing)}")

    frame["mukey"] = frame["mukey"].astype(str)
    frame["ksat_r"] = pd.to_numeric(frame["ksat_r"], errors="coerce")
    frame["hzdept_r"] = pd.to_numeric(frame["hzdept_r"], errors="coerce")
    frame["hzdepb_r"] = pd.to_numeric(frame["hzdepb_r"], errors="coerce")
    rows = []
    for mukey, group in frame.groupby("mukey", sort=True):
        hsg = _select_hsg(group["hydgrp"], drainage_condition=drainage_condition)
        ksat = _top_depth_harmonic_ksat(group, top_depth_cm=top_depth_cm)
        rows.append(
            {
                "mukey": mukey,
                "hsg": hsg,
                "hsg_code": _hsg_code(hsg),
                "ksat_mmhr": _ksat_to_mmhr(ksat, units=ksat_units),
                "source_rows": int(len(group)),
            }
        )
    out = pd.DataFrame(rows)
    return out.dropna(subset=["hsg_code", "ksat_mmhr"]).reset_index(drop=True)


def write_ssurgo_infiltration_rasters(
    soil_polygons,
    attributes,
    template_raster,
    *,
    hsg_out,
    ksat_out,
    land_domain=None,
    top_depth_cm: float = 40.0,
    drainage_condition: str = "undrained",
    ksat_units: str = "um/s",
    all_touched: bool = False,
) -> dict:
    """Rasterize SSURGO HSG and Ksat fields to a template raster grid."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    soils = gpd.read_file(soil_polygons) if isinstance(soil_polygons, (str, Path)) else soil_polygons.copy()
    if "mukey" not in soils:
        raise ValueError("SSURGO polygons must contain a mukey column")
    fields = aggregate_ssurgo_infiltration_fields(
        attributes,
        top_depth_cm=top_depth_cm,
        drainage_condition=drainage_condition,
        ksat_units=ksat_units,
    )
    merged = soils.copy()
    merged["mukey"] = merged["mukey"].astype(str)
    merged = merged.merge(fields, on="mukey", how="left")
    merged = merged.dropna(subset=["hsg_code", "ksat_mmhr"])

    template_raster = Path(template_raster)
    hsg_out = Path(hsg_out)
    ksat_out = Path(ksat_out)
    hsg_out.parent.mkdir(parents=True, exist_ok=True)
    ksat_out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(template_raster) as src:
        profile = src.profile.copy()
        shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs

    if merged.crs is None:
        merged = merged.set_crs(crs)
    elif crs is not None and merged.crs != crs:
        merged = merged.to_crs(crs)

    land_mask = None
    if land_domain is not None:
        land = gpd.read_file(land_domain) if isinstance(land_domain, (str, Path)) else land_domain.copy()
        if land.crs is None:
            land = land.set_crs(crs)
        elif crs is not None and land.crs != crs:
            land = land.to_crs(crs)
        land_geom = land.union_all()
        if not land_geom.is_empty:
            merged = merged.copy()
            merged["geometry"] = merged.geometry.intersection(land_geom)
            merged = merged[~merged.geometry.is_empty].copy()
        land_mask = rasterize(
            ((geom, 1) for geom in land.geometry if geom is not None and not geom.is_empty),
            out_shape=shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched,
        ).astype(bool)

    hsg = rasterize(
        ((geom, int(value)) for geom, value in zip(merged.geometry, merged["hsg_code"]) if geom is not None and not geom.is_empty),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )
    ksat = rasterize(
        ((geom, float(value)) for geom, value in zip(merged.geometry, merged["ksat_mmhr"]) if geom is not None and not geom.is_empty),
        out_shape=shape,
        transform=transform,
        fill=np.nan,
        dtype="float32",
        all_touched=all_touched,
    )
    if land_mask is not None:
        hsg = np.where(land_mask, hsg, 0).astype("uint8")
        ksat = np.where(land_mask, ksat, np.nan).astype("float32")

    hsg_profile = profile.copy()
    hsg_profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    ksat_profile = profile.copy()
    ksat_profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(hsg_out, "w", **hsg_profile) as dst:
        dst.write(hsg, 1)
    with rasterio.open(ksat_out, "w", **ksat_profile) as dst:
        dst.write(ksat.astype("float32"), 1)

    return {
        "hsg": str(hsg_out),
        "ksat": str(ksat_out),
        "mapunits": int(len(fields)),
        "rasterized_polygons": int(len(merged)),
        "hsg_pixels": int(np.count_nonzero(hsg)),
        "ksat_pixels": int(np.count_nonzero(np.isfinite(ksat))),
        "land_pixels": int(np.count_nonzero(land_mask)) if land_mask is not None else None,
    }


def prepare_ksat_raster_for_cn_recovery(
    source,
    output,
    *,
    scale_factor: float = 1.0,
    max_mmhr: float | None = None,
) -> dict:
    """Write an effective Ksat raster for CN-with-recovery infiltration."""
    import rasterio

    source = Path(source)
    output = Path(output)
    if not source.exists():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as src:
        profile = src.profile.copy()
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        valid = np.isfinite(arr)
        if nodata is not None and np.isfinite(nodata):
            valid &= arr != nodata
        conditioned = arr.copy()
        conditioned[valid] = conditioned[valid] * float(scale_factor)
        capped_fraction = 0.0
        if max_mmhr is not None:
            capped = valid & (conditioned > float(max_mmhr))
            capped_fraction = float(np.count_nonzero(capped) / max(np.count_nonzero(valid), 1))
            conditioned[capped] = float(max_mmhr)
        conditioned[~valid] = np.nan
        profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(conditioned.astype("float32"), 1)

    return {
        "ksat": str(output),
        "source_ksat": str(source),
        "scale_factor": float(scale_factor),
        "max_mmhr": None if max_mmhr is None else float(max_mmhr),
        "valid_pixels": int(np.count_nonzero(valid)),
        "capped_fraction": capped_fraction,
    }


def _select_hsg(values, *, drainage_condition):
    cleaned = [str(value).strip().upper() for value in values if pd.notna(value) and str(value).strip()]
    if not cleaned:
        return None
    value = cleaned[0]
    if "/" not in value:
        return value[0]
    parts = [part.strip() for part in value.split("/") if part.strip()]
    if drainage_condition == "drained":
        return parts[0]
    if drainage_condition == "undrained":
        return parts[-1]
    if drainage_condition == "conservative":
        return max(parts, key=_hsg_code)
    raise ValueError("drainage_condition must be 'undrained', 'drained', or 'conservative'")


def _hsg_code(value):
    return {"A": 1, "B": 2, "C": 3, "D": 4}.get(str(value).strip().upper())


def _top_depth_harmonic_ksat(group, *, top_depth_cm):
    valid = group.dropna(subset=["ksat_r", "hzdept_r", "hzdepb_r"]).copy()
    valid = valid[valid["ksat_r"] > 0]
    if valid.empty:
        return np.nan
    top = valid["hzdept_r"].clip(lower=0, upper=top_depth_cm)
    bottom = valid["hzdepb_r"].clip(lower=0, upper=top_depth_cm)
    thickness = (bottom - top).clip(lower=0)
    valid = valid.assign(_thickness=thickness)
    valid = valid[valid["_thickness"] > 0]
    if valid.empty:
        return np.nan
    return float(valid["_thickness"].sum() / (valid["_thickness"] / valid["ksat_r"]).sum())


def _ksat_to_mmhr(value, *, units):
    if pd.isna(value):
        return np.nan
    if units == "um/s":
        return float(value) * 3.6
    if units == "mm/hr":
        return float(value)
    raise ValueError("ksat_units must be 'um/s' or 'mm/hr'")


def validate_infiltration_config(
    infiltration_cfg: dict | None,
    *,
    event_drivers: list[str] | tuple[str, ...] | set[str],
) -> None:
    """Validate that rain-on-grid runs have explicit infiltration inputs."""
    infiltration_cfg = infiltration_cfg or {}
    if not bool(infiltration_cfg.get("enabled", True)):
        return

    hydrologic_drivers = set(event_drivers or []) & {"rainfall", "soil_moisture"}
    if not hydrologic_drivers:
        return

    method = str(infiltration_cfg.get("method", "cn_with_recovery")).lower()
    if method == "cn_with_recovery":
        missing = [
            key
            for key in ("hsg", "ksat", "effective")
            if infiltration_cfg.get(key) in (None, "")
        ]
        if missing:
            raise RuntimeError(
                "Rainfall/soil-moisture drivers require CN-with-recovery inputs. "
                f"Missing coastal_wave_coupling.hydrology.infiltration keys: {missing}. "
                "Add HSG and Ksat rasters plus an event/soil-moisture-derived "
                "effective soil-retention fraction before running pluvial SFINCS. "
                "The SSURGO mapunit polygons currently fetched in 01_region_setup.ipynb "
                "are geometry only; they do not contain HSG/Ksat attributes."
            )
    elif method == "cn":
        if infiltration_cfg.get("cn") in (None, ""):
            raise RuntimeError(
                "CN infiltration requires coastal_wave_coupling.hydrology.infiltration.cn "
                "pointing to a gridded Curve Number raster with cn/cn_avg variables."
            )
    elif method == "constant_lulc":
        if infiltration_cfg.get("qinf_reclass_table") in (None, ""):
            raise RuntimeError(
                "constant_lulc infiltration requires a qinf reclass table that maps "
                "impervious/developed LULC classes to near-zero infiltration."
            )
    else:
        raise ValueError(f"Unsupported infiltration method: {method}")


def setup_hydromt_infiltration(sf, config, paths, *, datadir=None):
    hydrology_cfg = (config.get("coastal_wave_coupling") or {}).get("hydrology") or {}
    infiltration_cfg = hydrology_cfg.get("infiltration") or {}
    hydrology_enabled = bool(infiltration_cfg.get("enabled", True))
    hydrologic_drivers = (
        set(config.get("event_drivers") or []) & {"rainfall", "soil_moisture"}
        if hydrology_enabled
        else set()
    )
    if not hydrologic_drivers:
        return {
            "enabled": hydrology_enabled,
            "drivers": sorted(hydrologic_drivers),
            "method": str(infiltration_cfg.get("method", "cn_with_recovery")).lower(),
            "written": False,
        }

    validate_infiltration_config(
        infiltration_cfg,
        event_drivers=config.get("event_drivers") or [],
    )
    method = str(infiltration_cfg.get("method", "cn_with_recovery")).lower()
    component = getattr(sf, "quadtree_infiltration", None) or getattr(sf, "infiltration", None)
    if component is None:
        raise RuntimeError("HydroMT-SFINCS model has no infiltration component")

    if method == "cn_with_recovery":
        reclass_table = infiltration_cfg.get("reclass_table")
        if reclass_table in (None, "") and datadir is not None:
            reclass_table = str(Path(datadir) / "lulc" / "esa_worldcover_HSG.csv")
        ksat_source = infiltration_cfg["ksat"]
        ksat_conditioning = None
        if infiltration_cfg.get("ksat_effective") not in (None, ""):
            ksat_conditioning = prepare_ksat_raster_for_cn_recovery(
                _resolve_location_path(paths, infiltration_cfg["ksat"]),
                _resolve_location_path(paths, infiltration_cfg["ksat_effective"]),
                scale_factor=float(infiltration_cfg.get("ksat_scale_factor", 1.0)),
                max_mmhr=(
                    None
                    if infiltration_cfg.get("ksat_max_mmhr") in (None, "")
                    else float(infiltration_cfg.get("ksat_max_mmhr"))
                ),
            )
            ksat_source = ksat_conditioning["ksat"]
        factor_ksat = infiltration_cfg.get("factor_ksat")
        if factor_ksat in (None, ""):
            # HydroMT-SFINCS' quadtree CN-recovery path multiplies Ksat by 3.6
            # internally. A conditioned raster from this module is already mm/hr.
            factor_ksat = 1.0 / 3.6 if ksat_conditioning else 1.0
        component.create_cn_with_recovery(
            lulc=infiltration_cfg.get("lulc", "worldcover"),
            hsg=_register_raster_source(sf, paths, "hsg", infiltration_cfg["hsg"]),
            ksat=_register_raster_source(sf, paths, "ksat", ksat_source),
            reclass_table=reclass_table,
            effective=float(infiltration_cfg["effective"]),
            factor_ksat=float(factor_ksat),
            block_size=int(infiltration_cfg.get("block_size", 2000)),
        )
    elif method == "cn":
        component.create_cn(
            cn=_register_raster_source(sf, paths, "curve_number", infiltration_cfg["cn"]),
            antecedent_moisture=infiltration_cfg.get("antecedent_moisture", "avg"),
        )
    elif method == "constant_lulc":
        component.create_constant(
            lulc=infiltration_cfg.get("lulc", "worldcover"),
            reclass_table=str(_resolve_location_path(paths, infiltration_cfg["qinf_reclass_table"])),
        )
    else:
        raise ValueError(f"Unsupported infiltration method: {method}")

    return {
        "enabled": hydrology_enabled,
        "drivers": sorted(hydrologic_drivers),
        "method": method,
        "written": True,
        "ksat_conditioning": ksat_conditioning if method == "cn_with_recovery" else None,
    }


def _resolve_location_path(paths, value):
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else Path(paths["location_root"]) / path


def _register_raster_source(sf, paths, name, value):
    path = _resolve_location_path(paths, value)
    if path is None:
        return name
    if not path.exists():
        raise FileNotFoundError(f"{name} raster not found: {path}")
    sf.data_catalog.from_dict(
        {
            name: {
                "uri": str(path),
                "data_type": "RasterDataset",
                "driver": {"name": "rasterio"},
            }
        }
    )
    return name


def find_aorc_event_window(
    event_windows_dir: str | Path,
    *,
    member_id: str | None = None,
    storm_start: str | pd.Timestamp | None = None,
) -> Path:
    """Find a local AORC storm-window NetCDF for a rainfall member."""
    root = Path(event_windows_dir)
    if not root.exists():
        raise FileNotFoundError(f"AORC event-window directory not found: {root}")

    patterns: list[str] = []
    if member_id:
        match = re.search(r"rank(\d{4})", str(member_id))
        if not match:
            raise ValueError(f"Could not parse rank#### from member_id={member_id!r}")
        patterns.append(f"*rank{match.group(1)}_*.nc")
    if storm_start is not None:
        token = pd.Timestamp(storm_start).strftime("%Y%m%dT%H")
        patterns.append(f"*_{token}.nc")
    if not patterns:
        patterns.append("*.nc")

    matches_by_pattern = [set(root.glob(pattern)) for pattern in patterns]
    if len(matches_by_pattern) > 1:
        unique = sorted(set.intersection(*matches_by_pattern))
    else:
        unique = sorted(matches_by_pattern[0])
    if not unique:
        raise FileNotFoundError(
            f"No AORC event-window NetCDF matched {patterns} in {root}"
        )
    if len(unique) > 1:
        raise RuntimeError(
            "AORC event-window lookup is ambiguous: "
            + ", ".join(path.name for path in unique[:8])
        )
    return unique[0]


def prepare_aorc_precip_for_sfincs(
    source_nc: str | Path,
    output_nc: str | Path,
    *,
    t_start: str | pd.Timestamp,
    t_stop: str | pd.Timestamp,
    variable: str = "APCP_surface",
    freq: str = "1h",
    align_start_to_run: bool = False,
    window_alignment: str = "start",
    precip_start: str | pd.Timestamp | None = None,
) -> Path:
    """Normalize AORC storm-window precipitation for HydroMT-SFINCS.

    AORC APCP is equivalent to millimeters of water over the reporting
    interval. HydroMT-SFINCS expects a variable named ``precip`` on
    ``time/y/x`` coordinates and converts interval totals to mm/hr when
    called with ``cumulative_input=True``.
    """
    source_nc = Path(source_nc)
    output_nc = Path(output_nc)
    if not source_nc.exists():
        raise FileNotFoundError(source_nc)

    ds = xr.open_dataset(source_nc)
    if variable not in ds:
        raise KeyError(f"{variable!r} not found in {source_nc}")

    da = ds[variable]
    rename: dict[str, str] = {}
    if "latitude" in da.dims:
        rename["latitude"] = "y"
    if "longitude" in da.dims:
        rename["longitude"] = "x"
    da = da.rename(rename).rename("precip")
    missing_dims = {"time", "y", "x"} - set(da.dims)
    if missing_dims:
        raise ValueError(f"Precipitation is missing required dims: {missing_dims}")

    full_time = pd.date_range(pd.Timestamp(t_start), pd.Timestamp(t_stop), freq=freq)
    if window_alignment not in {"start", "wettest"}:
        raise ValueError("window_alignment must be 'start' or 'wettest'")
    if window_alignment == "wettest" and da.sizes["time"] >= len(full_time):
        da = _select_wettest_precip_window(da, len(full_time))

    if align_start_to_run:
        start_time = pd.Timestamp(t_start) if precip_start is None else pd.Timestamp(precip_start)
        da = da.assign_coords(
            time=pd.date_range(start_time, periods=da.sizes["time"], freq=freq)
        )

    da = da.sortby("y").sortby("x").reindex(time=full_time, fill_value=0.0)
    da = da.astype("float32")
    da.attrs.update({"units": "mm", "crs": "EPSG:4326"})

    output_nc.parent.mkdir(parents=True, exist_ok=True)
    out = da.to_dataset()
    out.attrs.update({"crs": "EPSG:4326", "source": str(source_nc)})
    out.to_netcdf(output_nc)
    return output_nc


def _select_wettest_precip_window(da: xr.DataArray, window_steps: int) -> xr.DataArray:
    if window_steps <= 0:
        raise ValueError("window_steps must be positive")
    spatial_dims = [dim for dim in da.dims if dim != "time"]
    totals = da.sum(dim=spatial_dims, skipna=True).to_numpy()
    if len(totals) == window_steps:
        return da
    window_sums = np.convolve(totals, np.ones(window_steps), mode="valid")
    start = int(np.nanargmax(window_sums))
    return da.isel(time=slice(start, start + window_steps))


def summarize_soil_moisture(
    source_csv: str | Path,
    *,
    at_time: str | pd.Timestamp,
    lookback_hours: float = 24.0,
) -> dict[str, float]:
    """Return a scalar NWM soil-moisture summary around an event start."""
    source_csv = Path(source_csv)
    if not source_csv.exists():
        raise FileNotFoundError(source_csv)
    at_time = pd.Timestamp(at_time)
    start = at_time - pd.Timedelta(hours=float(lookback_hours))

    df = pd.read_csv(source_csv, parse_dates=["time"])
    window = df[(df["time"] >= start) & (df["time"] <= at_time)].copy()
    if window.empty:
        raise RuntimeError(
            f"No NWM soil-moisture rows in {lookback_hours:g}h before {at_time}"
        )
    value_column = "SOILSAT_TOP" if "SOILSAT_TOP" in window else "SOIL_M"
    values = window[value_column].astype(float)
    return {
        "soil_moisture_variable": value_column,
        "mean_soil_moisture": float(values.mean()),
        "min_soil_moisture": float(values.min()),
        "max_soil_moisture": float(values.max()),
        "row_count": float(len(window)),
    }
