from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage
import xarray as xr

from paths import location_root_from_paths, resolve_location_path
from wflow_runs.staticmaps_qa import (
    active_wflow_river_mask,
    static_missing_mask,
    staticmap_yx_dims,
    staticmaps_crs,
    valid_static_values,
    wflow_active_land_cells,
    write_netcdf_atomically,
)


def ensure_wflow_hydrography_basemap_nodata(config, paths) -> Path | None:
    """Repair stale local HydroMT-Wflow hydrography support-map nodata metadata."""
    import hydromt  # noqa: F401

    location_root = location_root_from_paths(paths)
    collection = config.get("collection", {}).get("national_hydrography", {})
    hydrography_path = resolve_location_path(
        location_root,
        collection.get("hydromt_basemap", "data/wflow/hydrography/us_hydrography_basemap.nc"),
    )
    if not hydrography_path.exists():
        return None

    stale = False
    ds = xr.open_dataset(hydrography_path)
    try:
        for name in ("strord", "basins"):
            if name in ds and ds[name].raster.nodata is None:
                stale = True
                break
    finally:
        ds.close()
    if not stale:
        return hydrography_path

    raw = xr.open_dataset(hydrography_path, mask_and_scale=False).load()
    raw.close()
    encoding = {}
    for name, dtype, fill in (
        ("flwdir", "uint8", None),
        ("strord", "int16", np.int16(0)),
        ("basins", "int32", np.int32(0)),
        ("rivmsk_review", "uint8", np.uint8(0)),
    ):
        if name not in raw:
            continue
        raw[name].attrs.pop("_FillValue", None)
        options = {"dtype": dtype}
        if fill is not None:
            options["_FillValue"] = fill
        encoding[name] = options
    write_netcdf_atomically(raw, hydrography_path, encoding=encoding)
    return hydrography_path


def normalize_wflow_staticmaps_nodata(model_root) -> Path | None:
    """Normalize HydroMT-Wflow nodata artifacts to Wflow's active-cell contract.

    HydroMT-Wflow can write integer-minimum values in ``subcatchment`` even when
    the declared fill value is 0. Wflow treats ``subcatchment != 0`` as active,
    so those cells must be written as 0 in the base model artifact.

    Some clipped/reprojected builds also leave tiny active-cell holes in ``land_slope``
    along the model edge. Fill those holes from the nearest valid active slope cell; do
    not invent values outside the active Wflow domain.
    """
    staticmaps_path = Path(model_root) / "staticmaps.nc"
    if not staticmaps_path.exists():
        return None

    ds = xr.open_dataset(staticmaps_path, mask_and_scale=False)
    try:
        if "subcatchment" not in ds:
            return staticmaps_path
        subcatchment = ds["subcatchment"]
        if not np.issubdtype(subcatchment.dtype, np.number):
            return staticmaps_path
        values = np.asarray(subcatchment.values)
        sentinel = np.iinfo(np.int32).min
        bad = values == sentinel
        if np.issubdtype(values.dtype, np.floating):
            bad = bad | ~np.isfinite(values)
        slope_needs_repair = _active_land_slope_nodata_mask(ds).any()
        needs_repair = (
            bool(bad.any())
            or subcatchment.dtype != np.dtype("int32")
            or subcatchment.attrs.get("_FillValue") != 0
            or bool(slope_needs_repair)
        )
        if not needs_repair:
            return staticmaps_path
    finally:
        ds.close()

    raw = xr.open_dataset(staticmaps_path, mask_and_scale=False).load()
    raw.close()
    values = np.asarray(raw["subcatchment"].values)
    bad = values == sentinel
    if np.issubdtype(values.dtype, np.floating):
        bad = bad | ~np.isfinite(values)
    repaired = np.where(bad, 0, values).astype("int32", copy=False)
    raw["subcatchment"].values = repaired
    raw["subcatchment"].attrs.pop("_FillValue", None)
    _fill_active_land_slope_nodata(raw)
    encoding = _staticmaps_nodata_encoding(raw)
    for variable in encoding:
        if variable in raw:
            raw[variable].attrs.pop("_FillValue", None)
    write_netcdf_atomically(
        raw,
        staticmaps_path,
        encoding=encoding,
    )
    return staticmaps_path


def _active_land_slope_nodata_mask(ds: xr.Dataset) -> np.ndarray:
    if "land_slope" not in ds:
        first = next(iter(ds.data_vars.values()))
        return np.zeros(first.shape, dtype=bool)
    return wflow_active_land_cells(ds) & static_missing_mask(ds["land_slope"])


def _fill_active_land_slope_nodata(ds: xr.Dataset) -> bool:
    if "land_slope" not in ds:
        return False
    missing_active = _active_land_slope_nodata_mask(ds)
    if not bool(missing_active.any()):
        return False
    slope = ds["land_slope"]
    values = np.asarray(slope.values, dtype=float)
    valid = wflow_active_land_cells(ds) & ~static_missing_mask(slope) & np.isfinite(values)
    if not bool(valid.any()):
        return False
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    filled = values.copy()
    filled[missing_active] = values[tuple(index[missing_active] for index in nearest)]
    slope.values = filled.astype(slope.dtype, copy=False)
    return True


def _staticmaps_nodata_encoding(ds: xr.Dataset) -> dict:
    encoding = {"subcatchment": {"dtype": "int32", "_FillValue": np.int32(0)}}
    if "land_slope" in ds:
        fill_value = ds["land_slope"].attrs.get("_FillValue", np.float32(np.nan))
        encoding["land_slope"] = {"dtype": "float32", "_FillValue": np.float32(fill_value)}
    return encoding


def repair_wflow_river_width(model_root, *, min_width_m: float = 30.0) -> Path | None:
    """Add a minimal Wflow river-width map/config entry to legacy generated models.

    HydroMT-Wflow writes ``river_width`` when ``setup_rivers.river_geom_fn`` is set. Some
    older generated bases predate that recipe contract, so keep them runnable by filling
    river cells with the recipe's conservative ``min_rivwth`` fallback.
    """
    model_root = Path(model_root)
    staticmaps_path = model_root / "staticmaps.nc"
    toml_path = model_root / "wflow_sbm.toml"
    repaired: Path | None = None

    if staticmaps_path.exists():
        ds = xr.open_dataset(staticmaps_path, mask_and_scale=False)
        try:
            needs_staticmap = "river_width" not in ds and "river_mask" in ds
        finally:
            ds.close()
        if needs_staticmap:
            raw = xr.open_dataset(staticmaps_path, mask_and_scale=False).load()
            raw.close()
            river_mask = raw["river_mask"]
            fill_value = np.float32(-9999.0)
            values = np.where(
                np.asarray(river_mask.values) > 0,
                np.float32(min_width_m),
                fill_value,
            ).astype("float32", copy=False)
            raw["river_width"] = xr.DataArray(
                values,
                dims=river_mask.dims,
                coords=river_mask.coords,
                attrs={"units": "m"},
            )
            raw["river_width"].attrs.pop("_FillValue", None)
            write_netcdf_atomically(
                raw,
                staticmaps_path,
                encoding={"river_width": {"dtype": "float32", "_FillValue": fill_value}},
            )
            repaired = staticmaps_path

    if toml_path.exists() and _ensure_wflow_static_toml_mapping(toml_path, "river__width", "river_width"):
        repaired = repaired or toml_path
    return repaired


def repair_wflow_canopy_parameters(model_root) -> Path | None:
    """Add non-cyclic Wflow canopy parameters missing from legacy HydroMT outputs."""
    model_root = Path(model_root)
    staticmaps_path = model_root / "staticmaps.nc"
    toml_path = model_root / "wflow_sbm.toml"
    repaired: Path | None = None

    if staticmaps_path.exists():
        ds = xr.open_dataset(staticmaps_path, mask_and_scale=False)
        try:
            has_inputs = {"vegetation_kext", "vegetation_leaf_storage", "vegetation_wood_storage"} <= set(ds.data_vars)
            needs_gap = "vegetation_canopy_gap_fraction" not in ds
            needs_storage = "vegetation_water_storage_capacity" not in ds
        finally:
            ds.close()
        if has_inputs and (needs_gap or needs_storage):
            raw = xr.open_dataset(staticmaps_path, mask_and_scale=False).load()
            raw.close()
            kext = raw["vegetation_kext"]
            fill_value = np.float32(-999.0)
            valid = valid_static_values(kext)
            if needs_gap:
                gap = np.full(kext.shape, fill_value, dtype="float32")
                gap[valid] = np.exp(-np.clip(np.asarray(kext.values, dtype="float32")[valid], 0.0, None))
                raw["vegetation_canopy_gap_fraction"] = xr.DataArray(
                    gap,
                    dims=kext.dims,
                    coords=kext.coords,
                    attrs={"units": "-"},
                )
                raw["vegetation_canopy_gap_fraction"].attrs.pop("_FillValue", None)
            if needs_storage:
                leaf = np.asarray(raw["vegetation_leaf_storage"].values, dtype="float32")
                wood = np.asarray(raw["vegetation_wood_storage"].values, dtype="float32")
                storage = np.full(kext.shape, fill_value, dtype="float32")
                storage[valid] = np.maximum(leaf[valid] + wood[valid], 0.0)
                raw["vegetation_water_storage_capacity"] = xr.DataArray(
                    storage,
                    dims=kext.dims,
                    coords=kext.coords,
                    attrs={"units": "mm"},
                )
                raw["vegetation_water_storage_capacity"].attrs.pop("_FillValue", None)
            encoding = {}
            if needs_gap:
                encoding["vegetation_canopy_gap_fraction"] = {"dtype": "float32", "_FillValue": fill_value}
            if needs_storage:
                encoding["vegetation_water_storage_capacity"] = {"dtype": "float32", "_FillValue": fill_value}
            write_netcdf_atomically(raw, staticmaps_path, encoding=encoding)
            repaired = staticmaps_path

    if toml_path.exists():
        if _ensure_wflow_static_toml_mapping(
            toml_path,
            "vegetation_canopy__gap_fraction",
            "vegetation_canopy_gap_fraction",
        ):
            repaired = repaired or toml_path
        if _ensure_wflow_static_toml_mapping(
            toml_path,
            "vegetation_water__storage_capacity",
            "vegetation_water_storage_capacity",
        ):
            repaired = repaired or toml_path
    return repaired


def repair_wflow_gauge_map(model_root, *, basename: str = "sfincs") -> Path | None:
    """Ensure every handoff gauge is represented on an active Wflow river cell.

    SFINCS-native inflow points are intentionally placed on the SFINCS boundary, which
    can fall just off the rasterized Wflow river cells. Wflow only writes a Q column for
    gauges present in the integer gauge map, so snap the stored handoff points to nearest
    active river cells while keeping the staticgeoms identity table unchanged.
    """
    model_root = Path(model_root)
    staticmaps_path = model_root / "staticmaps.nc"
    gauges_path = model_root / "staticgeoms" / f"gauges_{basename}.geojson"
    gauge_var = f"gauges_{basename}"
    if not staticmaps_path.exists() or not gauges_path.exists():
        return None

    gauges = gpd.read_file(gauges_path)
    if gauges.empty:
        return None
    if "index" not in gauges:
        gauges = gauges.copy()
        gauges["index"] = np.arange(1, len(gauges) + 1, dtype=np.int32)
    expected = {
        int(value)
        for value in pd.to_numeric(gauges["index"], errors="coerce").dropna().astype(int)
        if int(value) > 0
    }
    if not expected:
        return None

    ds = xr.open_dataset(staticmaps_path, mask_and_scale=False)
    try:
        if "river_mask" not in ds:
            return None
        river_mask = ds["river_mask"]
        y_dim, x_dim = staticmap_yx_dims(river_mask)
        existing = set()
        existing_on_river = False
        if gauge_var in ds:
            values = np.asarray(ds[gauge_var].values)
            existing = {int(value) for value in np.unique(values) if int(value) > 0}
            active = active_wflow_river_mask(ds)
            existing_on_river = all(bool(np.any((values == idx) & active)) for idx in expected)
        if expected <= existing and existing_on_river:
            return staticmaps_path
    finally:
        ds.close()

    raw = xr.open_dataset(staticmaps_path, mask_and_scale=False).load()
    raw.close()
    river_mask = raw["river_mask"]
    y_dim, x_dim = staticmap_yx_dims(river_mask)
    active = active_wflow_river_mask(raw)
    active_rows, active_cols = np.where(active)
    if active_rows.size == 0:
        return None

    xs = np.asarray(raw.coords[x_dim].values, dtype=float)
    ys = np.asarray(raw.coords[y_dim].values, dtype=float)
    active_x = xs[active_cols]
    active_y = ys[active_rows]

    target_crs = staticmaps_crs(raw, x_dim=x_dim, y_dim=y_dim)
    if target_crs is not None and gauges.crs is not None:
        gauges = gauges.to_crs(target_crs)

    values = np.zeros(river_mask.shape, dtype=np.int32)
    used: set[tuple[int, int]] = set()
    for _, row in gauges.iterrows():
        try:
            gauge_index = int(row["index"])
        except (TypeError, ValueError):
            continue
        if gauge_index <= 0 or row.geometry is None or row.geometry.is_empty:
            continue
        dx = active_x - float(row.geometry.x)
        dy = active_y - float(row.geometry.y)
        order = np.argsort((dx * dx) + (dy * dy))
        for active_pos in order:
            cell = (int(active_rows[active_pos]), int(active_cols[active_pos]))
            if cell not in used:
                used.add(cell)
                values[cell] = np.int32(gauge_index)
                break

    raw[gauge_var] = xr.DataArray(
        values,
        dims=river_mask.dims,
        coords=river_mask.coords,
        attrs={"long_name": f"{basename} gauge locations"},
    )
    raw[gauge_var].attrs.pop("_FillValue", None)
    write_netcdf_atomically(
        raw,
        staticmaps_path,
        encoding={gauge_var: {"dtype": "int32", "_FillValue": np.int32(0)}},
    )
    return staticmaps_path


def _ensure_wflow_static_toml_mapping(toml_path: Path, standard_name: str, variable_name: str) -> bool:
    """Ensure ``[input.static]`` maps a Wflow standard name to a staticmap variable."""
    lines = toml_path.read_text(encoding="utf-8").splitlines(keepends=True)
    assignment = f'{standard_name} = "{variable_name}"\n'
    if any(line.strip().startswith(f"{standard_name} =") for line in lines):
        return False
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    static_header = None
    for index, line in enumerate(lines):
        if line.strip() == "[input.static]":
            static_header = index
            break
    if static_header is None:
        if lines and lines[-1].strip():
            lines.append("\n")
        lines.extend(["[input.static]\n", assignment])
        toml_path.write_text("".join(lines), encoding="utf-8")
        return True

    section_end = len(lines)
    for index in range(static_header + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section_end = index
            break

    insert_at = static_header + 1
    for index in range(static_header + 1, section_end):
        stripped = lines[index].strip()
        if stripped.startswith(("river__length =", "river__slope =")):
            insert_at = index + 1

    lines.insert(insert_at, assignment)
    toml_path.write_text("".join(lines), encoding="utf-8")
    return True
