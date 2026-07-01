from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def qa_status(report: pd.DataFrame) -> str:
    if report.empty:
        return "unknown"
    if (report["status"] == "not_available").any():
        return "unknown"
    if (report["status"] == "failed").any():
        return "failed"
    if (report["status"] == "review_required").any():
        return "review_required"
    return "passed"


def open_wflow_model(model_root: Path, catalog_path: Path, *, model_cls=None, mode: str):
    if model_cls is None:
        os.environ.pop("DEBUG", None)
        from hydromt_wflow import WflowSbmModel

        model_cls = WflowSbmModel
    return model_cls(root=str(model_root), mode=mode, data_libs=[str(catalog_path)])


def validate_staticmaps(
    model_root,
    *,
    river_upa_km2: float | None = None,
    require_variable_river_geometry: bool = True,
    max_land_slope: float = 10.0,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """QA Wflow staticmaps for nodata, river geometry, river mask, and slope issues."""
    staticmaps_path = Path(model_root) / "staticmaps.nc"
    if not staticmaps_path.exists():
        raise FileNotFoundError(staticmaps_path)
    rows: list[dict] = []
    with xr.open_dataset(staticmaps_path, mask_and_scale=False) as ds:
        _append_wflow_nodata_checks(rows, ds)
        _append_wflow_river_geometry_checks(
            rows,
            ds,
            require_variable_river_geometry=require_variable_river_geometry,
        )
        _append_wflow_slope_checks(rows, ds, max_land_slope=max_land_slope)
        _append_wflow_river_mask_checks(rows, ds, river_upa_km2=river_upa_km2)
    report = pd.DataFrame(rows)
    failed = report[report["status"].isin(["failed", "review_required"])] if not report.empty else report
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{row.check}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Wflow staticmap QA failed for {staticmaps_path}: {details}")
    return report


def _append_wflow_nodata_checks(rows: list[dict], ds: xr.Dataset) -> None:
    if {"subcatchment", "local_drain_direction"} - set(ds.data_vars):
        rows.append({"check": "nodata", "status": "failed", "message": "missing subcatchment or local_drain_direction"})
        return
    subcatchment = np.asarray(ds["subcatchment"].values)
    ldd = np.asarray(ds["local_drain_direction"].values)
    intmin = np.iinfo(np.int32).min
    bad_intmin = int(np.count_nonzero(subcatchment == intmin))
    active_missing_ldd = int(np.count_nonzero((subcatchment != 0) & (ldd == 255)))
    status = "passed" if bad_intmin == 0 and active_missing_ldd == 0 else "failed"
    rows.append(
        {
            "check": "nodata",
            "status": status,
            "message": f"intmin_subcatchment={bad_intmin}; active_missing_ldd={active_missing_ldd}",
        }
    )


def _append_wflow_river_geometry_checks(
    rows: list[dict],
    ds: xr.Dataset,
    *,
    require_variable_river_geometry: bool,
) -> None:
    active = wflow_active_river_cells(ds)
    for name in ("river_width", "river_depth"):
        if name not in ds:
            rows.append({"check": name, "status": "failed", "message": f"missing {name}"})
            continue
        values = np.asarray(ds[name].values, dtype=float)
        valid = values[active & np.isfinite(values) & (values > 0)]
        unique = int(len(np.unique(np.round(valid, 4)))) if valid.size else 0
        status = "passed"
        if valid.size == 0:
            status = "failed"
        elif require_variable_river_geometry and unique <= 1:
            status = "review_required"
        rows.append(
            {
                "check": name,
                "status": status,
                "message": f"valid_cells={int(valid.size)}; unique_values={unique}",
            }
        )


def _append_wflow_slope_checks(rows: list[dict], ds: xr.Dataset, *, max_land_slope: float) -> None:
    active_land = wflow_active_land_cells(ds)
    active_river = wflow_active_river_cells(ds)
    for name in ("land_slope", "river_slope"):
        if name not in ds:
            continue
        data_array = ds[name]
        values = np.asarray(data_array.values, dtype=float)
        active = active_land if name == "land_slope" else active_river
        missing_active = int(np.count_nonzero(active & static_missing_mask(data_array)))
        finite = values[np.isfinite(values)]
        vmax = float(np.nanmax(finite)) if finite.size else np.nan
        status = "passed"
        if missing_active:
            status = "failed"
        elif name == "land_slope" and finite.size and vmax > float(max_land_slope):
            status = "review_required"
        rows.append(
            {
                "check": name,
                "status": status,
                "message": f"missing_active_cells={missing_active}; max={vmax:g}",
            }
        )


def _append_wflow_river_mask_checks(rows: list[dict], ds: xr.Dataset, *, river_upa_km2: float | None) -> None:
    if "river_mask" not in ds:
        rows.append({"check": "river_mask", "status": "failed", "message": "missing river_mask"})
        return
    active = wflow_active_river_cells(ds)
    count = int(np.count_nonzero(active))
    message = f"active_river_cells={count}"
    if river_upa_km2 is not None and "meta_upstream_area" in ds:
        uparea = np.asarray(ds["meta_upstream_area"].values, dtype=float)
        below = int(np.count_nonzero(active & np.isfinite(uparea) & (uparea < float(river_upa_km2))))
        status = "passed" if below == 0 else "review_required"
        message += f"; below_river_upa_threshold={below}"
    else:
        status = "passed" if count > 0 else "failed"
    rows.append({"check": "river_mask", "status": status, "message": message})


def wflow_active_river_cells(ds: xr.Dataset) -> np.ndarray:
    if "river_mask" not in ds:
        first = next(iter(ds.data_vars.values()))
        return np.zeros(first.shape, dtype=bool)
    return np.asarray(ds["river_mask"].values) > 0


def wflow_active_land_cells(ds: xr.Dataset) -> np.ndarray:
    if {"subcatchment", "local_drain_direction"} - set(ds.data_vars):
        first = next(iter(ds.data_vars.values()))
        return np.zeros(first.shape, dtype=bool)
    subcatchment = ds["subcatchment"]
    ldd = ds["local_drain_direction"]
    sub = np.asarray(subcatchment.values)
    ldd_values = np.asarray(ldd.values)
    return (sub != 0) & ~static_missing_mask(subcatchment) & (ldd_values > 0) & ~static_missing_mask(ldd)


def static_missing_mask(data_array: xr.DataArray) -> np.ndarray:
    values = np.asarray(data_array.values)
    if np.issubdtype(values.dtype, np.number):
        missing = ~np.isfinite(values.astype(float, copy=False))
    else:
        missing = np.zeros(values.shape, dtype=bool)
    fill_value = data_array.attrs.get("_FillValue")
    if fill_value is None:
        return missing
    try:
        fill = np.asarray(fill_value).item()
    except ValueError:
        return missing
    try:
        if np.isnan(fill):
            return missing | np.isnan(values.astype(float, copy=False))
    except TypeError:
        pass
    return missing | (values == fill)


def staticmap_yx_dims(data_array: xr.DataArray) -> tuple[str, str]:
    dims = tuple(data_array.dims)
    if len(dims) < 2:
        raise ValueError(f"{data_array.name or 'staticmap'} must have at least two dimensions")
    y_dim = next((name for name in ("y", "lat", "latitude") if name in dims), dims[-2])
    x_dim = next((name for name in ("x", "lon", "longitude") if name in dims), dims[-1])
    return y_dim, x_dim


def active_wflow_river_mask(ds: xr.Dataset) -> np.ndarray:
    active = np.asarray(ds["river_mask"].values) > 0
    if "subcatchment" not in ds:
        return active
    subcatchment = ds["subcatchment"]
    sub_values = np.asarray(subcatchment.values)
    fill_value = subcatchment.attrs.get("_FillValue", 0)
    active = active & (sub_values != fill_value) & (sub_values != np.iinfo(np.int32).min)
    if np.issubdtype(sub_values.dtype, np.floating):
        active = active & np.isfinite(sub_values)
    return active


def staticmaps_crs(ds: xr.Dataset, *, x_dim: str, y_dim: str):
    spatial_ref = ds.coords.get("spatial_ref")
    if spatial_ref is None:
        spatial_ref = ds.get("spatial_ref")
    if spatial_ref is not None:
        crs_wkt = spatial_ref.attrs.get("crs_wkt") or spatial_ref.attrs.get("spatial_ref")
        epsg = spatial_ref.attrs.get("epsg_code") or spatial_ref.attrs.get("epsg")
        try:
            from pyproj import CRS

            if crs_wkt:
                return CRS.from_wkt(crs_wkt)
            if epsg:
                return CRS.from_user_input(epsg)
        except Exception:
            pass
    try:
        xs = np.asarray(ds.coords[x_dim].values, dtype=float)
        ys = np.asarray(ds.coords[y_dim].values, dtype=float)
    except Exception:
        return None
    if np.nanmin(xs) >= -180 and np.nanmax(xs) <= 180 and np.nanmin(ys) >= -90 and np.nanmax(ys) <= 90:
        return "EPSG:4326"
    return None


def valid_static_values(data_array: xr.DataArray) -> np.ndarray:
    values = np.asarray(data_array.values)
    valid = np.isfinite(values) if np.issubdtype(values.dtype, np.floating) else np.ones(values.shape, dtype=bool)
    fill_value = data_array.attrs.get("_FillValue")
    if fill_value is None:
        return valid
    if isinstance(fill_value, float) and np.isnan(fill_value):
        return valid & np.isfinite(values)
    return valid & (values != fill_value)


def write_netcdf_atomically(ds: xr.Dataset, output_path: Path, *, encoding: dict | None = None) -> None:
    output_path = Path(output_path)
    temp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    temp_path.unlink(missing_ok=True)
    ds.to_netcdf(temp_path, encoding=encoding or {})
    temp_path.replace(output_path)
