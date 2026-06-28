from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from design_events.collect_sources.aorc_event_meteo import aorc_wflow_temp_pet_variables, prepare_aorc_temp_pet_for_wflow
from design_events.collect_sources.source_artifacts import (
    read_source_artifact,
    source_artifact_covers,
    source_artifact_path,
    write_source_artifact,
)
from tqdm.auto import tqdm as iter_progress


DEFAULT_AORC_ZARR_YEAR_PATTERN = "s3://noaa-nws-aorc-v1-1-1km/{year}.zarr"


def _repo_path(paths, value):
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {"data", "02_flood", "01_grid"}:
        return paths["location_root"] / path
    return paths["repo_root"] / path


def _open_aorc_year(year, spec):
    return xr.open_dataset(
        spec.get("zarr_year_pattern", DEFAULT_AORC_ZARR_YEAR_PATTERN).format(year=int(year)),
        engine="zarr",
        chunks=spec.get("chunks", {}),
        consolidated=spec.get("consolidated", True),
    )


def _coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(f"AORC dataset missing coordinate: one of {candidates}")


def _slice_axis(values, lower, upper):
    first = float(values[0])
    last = float(values[-1])
    if first <= last:
        return slice(lower, upper)
    return slice(upper, lower)


def _longitude_bounds(values, west, east):
    minimum = float(values.min())
    if minimum >= 0 and west < 0:
        west = west % 360
        east = east % 360
    if west > east:
        raise ValueError("AORC SST longitude selection crosses the dateline")
    return west, east


def _normalize_end(value):
    timestamp = pd.Timestamp(value)
    if timestamp == timestamp.floor("D"):
        return timestamp + pd.Timedelta(hours=23)
    return timestamp


def _bbox_from_spec(paths, spec):
    if spec.get("bbox_wgs84") is not None:
        return tuple(float(value) for value in spec["bbox_wgs84"])
    region = spec.get("transposition_region", {})
    geometry_file = region.get("geometry_file")
    if not geometry_file:
        raise ValueError("aorc_sst requires bbox_wgs84 or transposition_region.geometry_file")
    import geopandas as gpd

    geometry = gpd.read_file(_repo_path(paths, geometry_file)).to_crs("EPSG:4326")
    return tuple(float(value) for value in geometry.total_bounds)


def _transposition_region(paths, spec):
    region = spec.get("transposition_region", {})
    geometry_file = region.get("geometry_file")
    if not geometry_file:
        return None
    import geopandas as gpd

    geometry = gpd.read_file(_repo_path(paths, geometry_file)).to_crs("EPSG:4326")
    if geometry.empty:
        return None
    return geometry


def _study_footprint(paths, config, spec):
    geometry_files = [
        spec.get("watershed_geometry_file"),
        spec.get("study_area_geometry_file"),
        spec.get("grid_footprint"),
        config.get("smart_ds_evaluation_footprint", {}).get("output"),
        config.get("grid_footprint", {}).get("source"),
        "data/static/aoi/evaluation_footprint.geojson",
        "data/static/aoi/study_area.geojson",
    ]
    geometry_file = None
    for candidate in geometry_files:
        if candidate and _repo_path(paths, candidate).exists():
            geometry_file = candidate
            break
    if geometry_file is None:
        return None
    import geopandas as gpd

    footprint = gpd.read_file(_repo_path(paths, geometry_file)).to_crs("EPSG:4326")
    if footprint.empty:
        return None
    return footprint


def _point_mask(geometry, lon_values, lat_values):
    from shapely.geometry import Point

    mask = np.zeros((len(lat_values), len(lon_values)), dtype=bool)
    for row, lat in enumerate(lat_values):
        for col, lon in enumerate(lon_values):
            mask[row, col] = geometry.covers(Point(float(lon), float(lat)))
    return mask


def _moving_footprint_plan(precip, paths, config, spec):
    footprint = _study_footprint(paths, config, spec)
    region = _transposition_region(paths, spec)
    if footprint is None or region is None:
        return None
    from shapely.affinity import translate
    from shapely.geometry import Point

    lat_name = _coord_name(precip.to_dataset(name="precip"), ["latitude", "lat"])
    lon_name = _coord_name(precip.to_dataset(name="precip"), ["longitude", "lon"])
    lat_values = np.asarray(precip[lat_name].values, dtype=float)
    lon_values = np.asarray(precip[lon_name].values, dtype=float)
    footprint_geom = footprint.geometry.union_all()
    region_geom = region.geometry.union_all()
    center = footprint_geom.centroid
    center_row = int(np.abs(lat_values - center.y).argmin())
    center_col = int(np.abs(lon_values - center.x).argmin())
    footprint_mask = _point_mask(footprint_geom, lon_values, lat_values)
    footprint_cells = np.argwhere(footprint_mask)
    if footprint_cells.size == 0:
        footprint_cells = np.array([[center_row, center_col]], dtype=int)
    offsets = footprint_cells - np.array([[center_row, center_col]])
    candidate_mask = _point_mask(region_geom, lon_values, lat_values)
    candidate_cells = np.argwhere(candidate_mask)
    stride = max(1, int(spec.get("transposition_stride_cells", 1)))
    row_indices = []
    col_indices = []
    target_lons = []
    target_lats = []
    valid_center_mask = np.zeros((len(lat_values), len(lon_values)), dtype=bool)
    for row, col in candidate_cells:
        if (row - center_row) % stride != 0 or (col - center_col) % stride != 0:
            continue
        rows = row + offsets[:, 0]
        cols = col + offsets[:, 1]
        if (
            rows.min() < 0
            or cols.min() < 0
            or rows.max() >= len(lat_values)
            or cols.max() >= len(lon_values)
        ):
            continue
        target_lon = float(lon_values[col])
        target_lat = float(lat_values[row])
        shifted = translate(
            footprint_geom,
            xoff=target_lon - center.x,
            yoff=target_lat - center.y,
        )
        if not region_geom.covers(shifted):
            continue
        row_indices.append(rows)
        col_indices.append(cols)
        valid_center_mask[row, col] = True
        target_lons.append(target_lon)
        target_lats.append(target_lat)
    if not row_indices:
        return None
    return {
        "rows": np.vstack(row_indices),
        "cols": np.vstack(col_indices),
        "offsets": offsets,
        "valid_center_mask": valid_center_mask,
        "stride": stride,
        "lon_values": lon_values,
        "lat_values": lat_values,
        "target_footprint_center_lon": float(center.x),
        "target_footprint_center_lat": float(center.y),
        "target_lons": np.asarray(target_lons, dtype=float),
        "target_lats": np.asarray(target_lats, dtype=float),
    }


def _subset_precip(ds, spec, bbox_wgs84, start, end):
    variable = spec.get("variable", "APCP_surface")
    variables = _event_window_variables(ds, spec, variable)
    time_name = _coord_name(ds, ["time", "valid_time"])
    lat_name = _coord_name(ds, ["latitude", "lat", "y"])
    lon_name = _coord_name(ds, ["longitude", "lon", "x"])
    west, south, east, north = bbox_wgs84
    west, east = _longitude_bounds(ds[lon_name].values, west, east)
    subset = ds[variables].sel(
        {
            time_name: slice(start, end),
            lat_name: _slice_axis(ds[lat_name].values, south, north),
            lon_name: _slice_axis(ds[lon_name].values, west, east),
        }
    )
    if time_name != "time":
        subset = subset.rename({time_name: "time"})
    return subset


def _event_window_variables(ds, spec, precip_variable):
    variables = [precip_variable]
    meteo_cfg = spec.get("event_meteo") or {}
    if bool(meteo_cfg.get("enabled", False)):
        for candidates in aorc_wflow_temp_pet_variables({"collection": {"aorc_sst": spec}}).values():
            for candidate in candidates:
                if candidate in ds and candidate not in variables:
                    variables.append(candidate)
                    break
    missing = [name for name in variables if name not in ds]
    if missing:
        raise KeyError(f"AORC dataset missing configured event-window variables: {missing}")
    return variables


def _spatial_stats(precip, duration_hours, check_every_n_hours=1, *, paths=None, config=None, spec=None):
    window = int(duration_hours)
    stride = max(1, int(check_every_n_hours))
    # Materialize the year's precip once. The per-window rainfall-peak-time lookup
    # re-slices this array ~once per checked window; left lazy (zarr/dask) each slice
    # re-fetches from S3, turning one bulk read into thousands of tiny remote reads.
    precip = precip.load()
    rolled = precip.rolling(time=window, min_periods=window).sum()
    rolled = rolled.isel(time=slice(window - 1, None, stride))
    moving_plan = None
    if paths is not None and config is not None and spec is not None:
        moving_plan = _moving_footprint_plan(precip, paths, config, spec)
    if moving_plan is not None:
        records = []
        rolled_values = np.asarray(rolled.values, dtype=float)
        times = pd.to_datetime(rolled["time"].values)
        rows = moving_plan["rows"]
        cols = moving_plan["cols"]
        for time_index, storm_end in enumerate(times):
            field = rolled_values[time_index]
            values = field[rows, cols]
            finite = np.isfinite(values)
            counts = finite.sum(axis=1)
            means = np.full(len(values), np.nan, dtype=float)
            valid = counts > 0
            means[valid] = np.nansum(values[valid], axis=1) / counts[valid]
            if np.isnan(means).all():
                continue
            best = int(np.nanargmax(means))
            best_values = values[best]
            storm_start = pd.Timestamp(storm_end) - pd.to_timedelta(duration_hours - 1, unit="h")
            peak_time, peak_value = _rainfall_window_peak_time(
                precip,
                storm_start,
                pd.Timestamp(storm_end),
                rows=rows[best],
                cols=cols[best],
            )
            records.append(
                {
                    "storm_end": pd.Timestamp(storm_end),
                    "mean": float(means[best]),
                    "max": float(np.nanmax(best_values)),
                    "min": float(np.nanmin(best_values)),
                    "rainfall_peak_time": peak_time,
                    "rainfall_peak_mm_per_hour": peak_value,
                    "x": float(moving_plan["target_lons"][best]),
                    "y": float(moving_plan["target_lats"][best]),
                    "historical_footprint_center_lon": float(moving_plan["target_lons"][best]),
                    "historical_footprint_center_lat": float(moving_plan["target_lats"][best]),
                    "target_footprint_center_lon": float(moving_plan["target_footprint_center_lon"]),
                    "target_footprint_center_lat": float(moving_plan["target_footprint_center_lat"]),
                    "potential_method": "moving_footprint_max_mean",
                }
            )
        frame = pd.DataFrame(records).dropna()
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "storm_date",
                    "mean",
                    "max",
                    "min",
                    "rainfall_peak_time",
                    "rainfall_peak_mm_per_hour",
                    "x",
                    "y",
                    "historical_footprint_center_lon",
                    "historical_footprint_center_lat",
                    "target_footprint_center_lon",
                    "target_footprint_center_lat",
                    "potential_method",
                ]
            )
        frame["storm_date"] = frame["storm_end"] - pd.to_timedelta(duration_hours - 1, unit="h")
        return frame[
            [
                "storm_date",
                "mean",
                "max",
                "min",
                "rainfall_peak_time",
                "rainfall_peak_mm_per_hour",
                "x",
                "y",
                "historical_footprint_center_lon",
                "historical_footprint_center_lat",
                "target_footprint_center_lon",
                "target_footprint_center_lat",
                "potential_method",
            ]
        ]
    mean_depth = rolled.mean(dim=[dim for dim in rolled.dims if dim != "time"])
    max_depth = rolled.max(dim=[dim for dim in rolled.dims if dim != "time"])
    min_depth = rolled.min(dim=[dim for dim in rolled.dims if dim != "time"])
    frame = pd.DataFrame(
        {
            "storm_end": pd.to_datetime(mean_depth["time"].values),
            "mean": mean_depth.values,
            "max": max_depth.values,
            "min": min_depth.values,
        }
    ).dropna()
    frame["storm_date"] = frame["storm_end"] - pd.to_timedelta(duration_hours - 1, unit="h")
    peak_times, peak_values = [], []
    hourly_mean = precip.mean(dim=[dim for dim in precip.dims if dim != "time"], skipna=True)
    for _, row in frame.iterrows():
        peak_time, peak_value = _rainfall_window_peak_time(
            hourly_mean,
            pd.Timestamp(row["storm_date"]),
            pd.Timestamp(row["storm_end"]),
        )
        peak_times.append(peak_time)
        peak_values.append(peak_value)
    frame["rainfall_peak_time"] = peak_times
    frame["rainfall_peak_mm_per_hour"] = peak_values
    frame["potential_method"] = "region_mean"
    return frame[["storm_date", "mean", "max", "min", "rainfall_peak_time", "rainfall_peak_mm_per_hour", "potential_method"]]


def _rainfall_window_peak_time(precip, start, end, *, rows=None, cols=None):
    window = precip.sel(time=slice(pd.Timestamp(start), pd.Timestamp(end)))
    if window.sizes.get("time", 0) == 0:
        return pd.NaT, np.nan
    if rows is not None and cols is not None:
        values = np.asarray(window.values, dtype=float)
        hourly = np.nanmean(values[:, rows, cols], axis=1)
        times = pd.to_datetime(window["time"].values)
    else:
        series = window
        if "time" in series.dims and len(series.dims) > 1:
            series = series.mean(dim=[dim for dim in series.dims if dim != "time"], skipna=True)
        hourly = np.asarray(series.values, dtype=float)
        times = pd.to_datetime(series["time"].values)
    if not np.isfinite(hourly).any():
        return pd.NaT, np.nan
    index = int(np.nanargmax(hourly))
    return pd.Timestamp(times[index]), float(hourly[index])


def _centroid_from_precip_window(precip):
    field = precip.sum(dim="time")
    lat_name = _coord_name(field.to_dataset(name="precip"), ["latitude", "lat"])
    lon_name = _coord_name(field.to_dataset(name="precip"), ["longitude", "lon"])
    lons, lats = xr.broadcast(field[lon_name], field[lat_name])
    total = float(field.sum(skipna=True))
    if total <= 0:
        return pd.NA, pd.NA
    return (
        float((field * lons).sum(skipna=True) / total),
        float((field * lats).sum(skipna=True) / total),
    )


def _decluster_top_events(candidates, *, min_threshold, decluster_hours, top_n=None):
    """Keep every independent storm above ``min_threshold`` (threshold-driven POT).

    The member count is data-driven by the threshold and declustering window.
    ``top_n`` is an optional safety cap only — left ``None`` it keeps all exceedances.
    """
    candidates = candidates[candidates["mean"] >= float(min_threshold)].copy()
    candidates = candidates.sort_values("mean", ascending=False)
    selected = []
    selected_times = []
    for _, row in candidates.iterrows():
        storm_time = pd.Timestamp(row["storm_date"])
        if any(abs((storm_time - existing).total_seconds()) < decluster_hours * 3600 for existing in selected_times):
            continue
        selected.append(row)
        selected_times.append(storm_time)
        if top_n is not None and len(selected) >= int(top_n):
            break
    if not selected:
        return pd.DataFrame(columns=[*candidates.columns, "por_rank", "annual_rank"])
    ranked = pd.DataFrame(selected).reset_index(drop=True)
    ranked["por_rank"] = range(1, len(ranked) + 1)
    ranked["annual_rank"] = ranked.groupby(ranked["storm_date"].dt.year)["mean"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    return ranked


def _ranked_storms_are_sst_equivalent(path):
    if not Path(path).exists():
        return False
    try:
        ranked = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    required = {
        "potential_method",
        "rainfall_peak_time",
        "historical_footprint_center_lon",
        "historical_footprint_center_lat",
        "target_footprint_center_lon",
        "target_footprint_center_lat",
        "transposition_offset_lon",
        "transposition_offset_lat",
    }
    return required.issubset(ranked.columns) and not ranked.empty and set(ranked["potential_method"].dropna()) == {
        "moving_footprint_max_mean"
    }


def _yearly_stats_dir(collection_dir):
    return collection_dir / "yearly-stats"


def _yearly_stats_csv(collection_dir, year):
    return _yearly_stats_dir(collection_dir) / f"storm-stats-{int(year)}.csv"


def _stats_checkpoint_is_current(path):
    if not Path(path).exists():
        return False
    try:
        frame = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    moving = frame.get("potential_method", pd.Series(dtype=object)).eq("moving_footprint_max_mean").any()
    if moving:
        required = {
            "rainfall_peak_time",
            "historical_footprint_center_lon",
            "historical_footprint_center_lat",
            "target_footprint_center_lon",
            "target_footprint_center_lat",
        }
        return required.issubset(frame.columns)
    return (
        "potential_method" in frame.columns
        and "rainfall_peak_time" in frame.columns
        and not frame["potential_method"].dropna().empty
    )


def _collection_dir(paths, duration_hours):
    return paths["aorc_sst_root"] / paths["location_name"] / f"{int(duration_hours)}hr-events"


def _event_id(location_name, duration_hours, row):
    return f"rainfall_{location_name}_{int(duration_hours)}h_rank{int(row['por_rank']):04d}"


def _event_window_subset(year_datasets, opener, spec, bbox_wgs84, start, end):
    subsets = []
    for year in range(start.year, end.year + 1):
        if year not in year_datasets:
            year_datasets[year] = opener(year, spec)
        else:
            try:
                year_datasets.move_to_end(year)
            except AttributeError:
                pass
        ds = year_datasets[year]
        year_start = max(start, pd.Timestamp(f"{year}-01-01"))
        year_end = min(end, pd.Timestamp(f"{year}-12-31 23:00:00"))
        subsets.append(_subset_precip(ds, spec, bbox_wgs84, year_start, year_end))
    if len(subsets) == 1:
        return subsets[0]
    return xr.concat(subsets, dim="time")


def _trim_year_dataset_cache(year_datasets, max_open_year_datasets: int) -> None:
    while len(year_datasets) > max_open_year_datasets:
        _, ds = year_datasets.popitem(last=False)
        ds.close()


def _write_rainfall_members(paths, spec, ranked, source_csv, duration_hours):
    output_csv = Path(paths["aorc_sst_rainfall_members_csv"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ranked = _ensure_transposition_targets(paths, spec, ranked)
    transposition_id = spec.get("transposition_region", {}).get("id", pd.NA)
    historical_lon = ranked["historical_centroid_lon"] if "historical_centroid_lon" in ranked else pd.NA
    historical_lat = ranked["historical_centroid_lat"] if "historical_centroid_lat" in ranked else pd.NA
    transposed_lon = ranked["transposed_centroid_lon"] if "transposed_centroid_lon" in ranked else pd.NA
    transposed_lat = ranked["transposed_centroid_lat"] if "transposed_centroid_lat" in ranked else pd.NA
    offset_lon = ranked["transposition_offset_lon"] if "transposition_offset_lon" in ranked else pd.NA
    offset_lat = ranked["transposition_offset_lat"] if "transposition_offset_lat" in ranked else pd.NA
    historical_footprint_lon = (
        ranked["historical_footprint_center_lon"] if "historical_footprint_center_lon" in ranked else pd.NA
    )
    historical_footprint_lat = (
        ranked["historical_footprint_center_lat"] if "historical_footprint_center_lat" in ranked else pd.NA
    )
    target_footprint_lon = ranked["target_footprint_center_lon"] if "target_footprint_center_lon" in ranked else pd.NA
    target_footprint_lat = ranked["target_footprint_center_lat"] if "target_footprint_center_lat" in ranked else pd.NA
    members = pd.DataFrame(
        {
            "member_id": [
                _event_id(paths["location_name"], duration_hours, row)
                for _, row in ranked.iterrows()
            ],
            "source": "aorc_sst",
            "member_file": source_csv.as_posix(),
            "storm_start": pd.to_datetime(ranked["storm_date"]).dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "storm_end": (
                pd.to_datetime(ranked["storm_date"]) + pd.to_timedelta(duration_hours, unit="h")
            ).dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "rainfall_peak_time": (
                pd.to_datetime(ranked["rainfall_peak_time"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
                if "rainfall_peak_time" in ranked else pd.NA
            ),
            "rainfall_peak_mm_per_hour": ranked["rainfall_peak_mm_per_hour"] if "rainfall_peak_mm_per_hour" in ranked else pd.NA,
            "rainfall_peak_time_source": "storm_stats_hourly_peak" if "rainfall_peak_time" in ranked else "missing",
            "duration_hours": int(duration_hours),
            "rank": ranked["por_rank"],
            "annual_rank": ranked["annual_rank"],
            "mean_precip_mm": ranked["mean"],
            "max_precip_mm": ranked["max"],
            "min_precip_mm": ranked["min"],
            "precip_units": "mm",
            "potential_method": ranked["potential_method"] if "potential_method" in ranked else pd.NA,
            "centroid_lon": ranked["x"] if "x" in ranked else pd.NA,
            "centroid_lat": ranked["y"] if "y" in ranked else pd.NA,
            "historical_centroid_lon": historical_lon,
            "historical_centroid_lat": historical_lat,
            "transposed_centroid_lon": transposed_lon,
            "transposed_centroid_lat": transposed_lat,
            "transposition_offset_lon": offset_lon,
            "transposition_offset_lat": offset_lat,
            "historical_footprint_center_lon": historical_footprint_lon,
            "historical_footprint_center_lat": historical_footprint_lat,
            "target_footprint_center_lon": target_footprint_lon,
            "target_footprint_center_lat": target_footprint_lat,
            "transposition_region_id": transposition_id,
        }
    )
    members.to_csv(output_csv, index=False)
    return members


def collect_warmup(config: dict, paths: dict, *, force: bool = False, opener=None) -> dict:
    """Collect a shared AORC warmup forcing baseline for native HydroMT-Wflow.

    The output is event-agnostic: ``data/wflow/warmup/<baseline_id>/precip.nc``
    and ``temp_pet.nc`` can seed multiple event replays through the same reviewed
    antecedent state.
    """
    spec = ((config.get("collection", {}) or {}).get("aorc_sst", {}) or {}).copy()
    if not spec:
        raise KeyError("collection.aorc_sst is required for AORC Wflow warmup collection")
    spec["event_meteo"] = {**(spec.get("event_meteo") or {}), "enabled": True}
    settings = ((config.get("wflow", {}) or {}).get("dynamic_handoff", {}) or {})
    warmup_days = float(settings.get("warmup_days", 90))
    baseline_id = str(settings.get("baseline_id", f"baseline_{int(warmup_days)}d"))
    reference_time = settings.get("baseline_reference_time")
    if reference_time in (None, ""):
        raise ValueError("wflow.dynamic_handoff.baseline_reference_time is required for shared warmup collection")
    start = pd.Timestamp(reference_time) - pd.Timedelta(days=warmup_days)
    end = pd.Timestamp(reference_time) - pd.Timedelta(hours=1)
    root = Path(settings.get("baseline_root", f"data/wflow/warmup/{baseline_id}"))
    if not root.is_absolute():
        root = Path(paths["location_root"]) / root
    precip_nc = root / "precip.nc"
    source_nc = root / "aorc_warmup_source.nc"
    temp_pet_nc = root / "temp_pet.nc"
    provenance_json = root / "aorc_warmup_provenance.json"
    if (
        not force
        and precip_nc.exists()
        and temp_pet_nc.exists()
        and _warmup_file_covers(precip_nc, "precip", start, end)
        and _warmup_file_covers(temp_pet_nc, "temp", start, end)
    ):
        return {
            "status": "reused",
            "baseline_id": baseline_id,
            "warmup_start": start.isoformat(),
            "warmup_end": end.isoformat(),
            "precip_nc": str(precip_nc),
            "temp_pet_nc": str(temp_pet_nc),
        }

    opener = opener or _open_aorc_year
    bbox_wgs84 = _bbox_from_spec(paths, spec)
    source = _event_window_subset({}, opener, spec, bbox_wgs84, start, end).sortby("time")
    variable = spec.get("variable", "APCP_surface")
    if variable not in source:
        raise KeyError(f"AORC warmup source lacks precipitation variable {variable!r}")
    precip = _aorc_precip_to_wflow(source[variable])
    root.mkdir(parents=True, exist_ok=True)
    source.to_netcdf(source_nc)
    precip.to_dataset(name="precip").to_netcdf(precip_nc)
    temp_pet_provenance = prepare_aorc_temp_pet_for_wflow(
        source_nc,
        temp_pet_nc,
        t_start=start,
        t_stop=end,
        precip_template=precip_nc,
        variable_candidates=aorc_wflow_temp_pet_variables(config),
        provenance_path=root / "temp_pet_provenance.json",
    )
    provenance = {
        "status": "collected",
        "baseline_id": baseline_id,
        "reference_time": pd.Timestamp(reference_time).isoformat(),
        "warmup_days": warmup_days,
        "warmup_start": start.isoformat(),
        "warmup_end": end.isoformat(),
        "source": "AORC",
        "source_nc": str(source_nc),
        "precip_nc": str(precip_nc),
        "temp_pet_nc": str(temp_pet_nc),
        "temp_pet_provenance": temp_pet_provenance,
        "hydromt_wflow_contract": "setup_precip_forcing + setup_temp_pet_forcing",
    }
    provenance_json.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def _aorc_precip_to_wflow(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "y"
    if "longitude" in da.dims:
        rename["longitude"] = "x"
    if "lat" in da.dims:
        rename["lat"] = "y"
    if "lon" in da.dims:
        rename["lon"] = "x"
    out = da.rename(rename).rename("precip").astype("float32")
    out.attrs.update(units="mm", source_units=da.attrs.get("units", "unknown"))
    return out.sortby("y").sortby("x")


def _warmup_file_covers(path: Path, variable: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    try:
        with xr.open_dataset(path) as ds:
            if variable not in ds or "time" not in ds[variable].dims:
                return False
            tmin = pd.Timestamp(ds["time"].min().values)
            tmax = pd.Timestamp(ds["time"].max().values)
    except Exception:
        return False
    return tmin <= start and tmax >= end


def _compute_selected_event_windows(
    opener,
    ranked,
    spec,
    bbox_wgs84,
    output_dir,
    duration_hours,
    location_name,
    *,
    skip_existing=True,
):
    if ranked.empty:
        ranked = ranked.copy()
        ranked["x"] = pd.Series(dtype="Float64")
        ranked["y"] = pd.Series(dtype="Float64")
        return [], ranked
    write_event_windows = spec.get("write_event_windows", True)
    if write_event_windows:
        output_dir.mkdir(parents=True, exist_ok=True)
    elif (
        {"historical_centroid_lon", "historical_centroid_lat"}.issubset(ranked.columns)
        and ranked["historical_centroid_lon"].notna().all()
        and ranked["historical_centroid_lat"].notna().all()
    ):
        return [], ranked.copy()
    variable = spec.get("variable", "APCP_surface")
    written = []
    year_datasets = OrderedDict()
    max_open_year_datasets = max(1, int(spec.get("max_open_year_datasets", 4)))
    centroid_lons = []
    centroid_lats = []
    try:
        for _, row in iter_progress(
            list(ranked.iterrows()),
            total=len(ranked),
            desc="AORC selected storm windows",
            unit="storm",
            dynamic_ncols=True,
        ):
            start = pd.Timestamp(row["storm_date"])
            end = start + pd.to_timedelta(duration_hours - 1, unit="h")
            path = None
            requires_transposition = _row_requires_field_transposition(row)
            if write_event_windows:
                event_id = _event_id(location_name, duration_hours, row)
                path = output_dir / f"{event_id}_{start:%Y%m%dT%H}.nc"
                if skip_existing and _event_window_file_has_required_variables(
                    path,
                    spec,
                    require_transposition=requires_transposition,
                ):
                    centroid_lons.append(
                        row.get("historical_centroid_lon", row.get("x", pd.NA))
                    )
                    centroid_lats.append(
                        row.get("historical_centroid_lat", row.get("y", pd.NA))
                    )
                    written.append(path)
                    continue
                if skip_existing and _promote_complete_event_window_tmp(
                    path,
                    spec,
                    require_transposition=requires_transposition,
                ):
                    centroid_lons.append(
                        row.get("historical_centroid_lon", row.get("x", pd.NA))
                    )
                    centroid_lats.append(
                        row.get("historical_centroid_lat", row.get("y", pd.NA))
                    )
                    written.append(path)
                    continue
            subset = _event_window_subset(year_datasets, opener, spec, bbox_wgs84, start, end)
            centroid_lon, centroid_lat = _centroid_from_precip_window(subset[variable])
            centroid_lons.append(centroid_lon)
            centroid_lats.append(centroid_lat)
            if write_event_windows:
                subset = _apply_field_transposition(subset, row, historical_centroid=(centroid_lon, centroid_lat))
                _write_event_window_netcdf(subset, path)
                written.append(path)
            _trim_year_dataset_cache(year_datasets, max_open_year_datasets)
    finally:
        for ds in year_datasets.values():
            ds.close()
    ranked = ranked.copy()
    ranked["historical_centroid_lon"] = centroid_lons
    ranked["historical_centroid_lat"] = centroid_lats
    if "x" not in ranked:
        ranked["x"] = centroid_lons
    if "y" not in ranked:
        ranked["y"] = centroid_lats
    return written, ranked


def _event_window_tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _promote_complete_event_window_tmp(
    path: Path,
    spec: dict,
    *,
    require_transposition: bool = False,
) -> bool:
    tmp_path = _event_window_tmp_path(Path(path))
    if not _event_window_file_has_required_variables(
        tmp_path,
        spec,
        require_transposition=require_transposition,
    ):
        return False
    tmp_path.replace(path)
    return True


def _write_event_window_netcdf(ds: xr.Dataset, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _event_window_tmp_path(path)
    safe = _strip_netcdf_endian_encoding(ds)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="endian-ness of dtype and endian kwarg do not match.*",
            category=UserWarning,
            module="xarray.backends.netCDF4_",
        )
        safe.to_netcdf(tmp_path, engine="netcdf4")
    tmp_path.replace(path)


def _strip_netcdf_endian_encoding(ds: xr.Dataset) -> xr.Dataset:
    out = ds.copy()
    for name in out.variables:
        encoding = dict(out[name].encoding)
        if "endian" in encoding:
            encoding.pop("endian", None)
            out[name].encoding = encoding
    return out


def _event_window_file_has_required_variables(
    path: Path,
    spec: dict,
    *,
    require_transposition: bool = False,
    require_transposition_metadata: bool = False,
) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    required_targets = aorc_wflow_temp_pet_variables({"collection": {"aorc_sst": spec}})
    precip_variable = spec.get("variable", "APCP_surface")
    try:
        with xr.open_dataset(path) as ds:
            if precip_variable not in ds:
                return False
            if (
                require_transposition_metadata
                and ds.attrs.get("aorc_sst_field_transposition") not in {"applied", "none"}
            ):
                return False
            if require_transposition and ds.attrs.get("aorc_sst_field_transposition") != "applied":
                return False
            meteo_cfg = spec.get("event_meteo") or {}
            if not bool(meteo_cfg.get("enabled", False)):
                return True
            for candidates in required_targets.values():
                if not any(candidate in ds for candidate in candidates):
                    return False
    except Exception:
        return False
    return True


def _row_requires_field_transposition(row) -> bool:
    offset_lon = _numeric_row_value(row, "transposition_offset_lon")
    offset_lat = _numeric_row_value(row, "transposition_offset_lat")
    if np.isfinite(offset_lon) or np.isfinite(offset_lat):
        return bool(abs(offset_lon) > 0.0 or abs(offset_lat) > 0.0)
    historical_lon = _numeric_row_value(row, "historical_footprint_center_lon")
    historical_lat = _numeric_row_value(row, "historical_footprint_center_lat")
    target_lon = _numeric_row_value(row, "target_footprint_center_lon")
    target_lat = _numeric_row_value(row, "target_footprint_center_lat")
    return bool(
        np.isfinite(historical_lon)
        and np.isfinite(historical_lat)
        and np.isfinite(target_lon)
        and np.isfinite(target_lat)
        and (abs(target_lon - historical_lon) > 0.0 or abs(target_lat - historical_lat) > 0.0)
    )


def _numeric_row_value(row, column, default=np.nan):
    try:
        value = row.get(column, default)
    except AttributeError:
        value = default
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else np.nan


def _apply_field_transposition(ds: xr.Dataset, row, *, historical_centroid=None) -> xr.Dataset:
    offset_lon = _numeric_row_value(row, "transposition_offset_lon")
    offset_lat = _numeric_row_value(row, "transposition_offset_lat")
    if not np.isfinite(offset_lon):
        historical_lon = _numeric_row_value(row, "historical_footprint_center_lon")
        target_lon = _numeric_row_value(row, "target_footprint_center_lon")
        offset_lon = target_lon - historical_lon if np.isfinite(historical_lon) and np.isfinite(target_lon) else 0.0
    if not np.isfinite(offset_lat):
        historical_lat = _numeric_row_value(row, "historical_footprint_center_lat")
        target_lat = _numeric_row_value(row, "target_footprint_center_lat")
        offset_lat = target_lat - historical_lat if np.isfinite(historical_lat) and np.isfinite(target_lat) else 0.0
    if not (np.isfinite(offset_lon) and np.isfinite(offset_lat)):
        offset_lon, offset_lat = 0.0, 0.0

    out = ds.copy()
    lat_name = _coord_name(out, ["latitude", "lat", "y"])
    lon_name = _coord_name(out, ["longitude", "lon", "x"])
    if abs(offset_lon) > 0.0:
        out = out.assign_coords({lon_name: out[lon_name] + offset_lon})
    if abs(offset_lat) > 0.0:
        out = out.assign_coords({lat_name: out[lat_name] + offset_lat})

    attrs = {
        "aorc_sst_field_transposition": "applied" if (abs(offset_lon) > 0.0 or abs(offset_lat) > 0.0) else "none",
        "transposition_offset_lon": float(offset_lon),
        "transposition_offset_lat": float(offset_lat),
        "transposition_method": "coordinate_shift_to_study_footprint",
    }
    if historical_centroid is not None:
        hist_lon, hist_lat = historical_centroid
        if pd.notna(hist_lon):
            attrs["historical_centroid_lon"] = float(hist_lon)
            attrs["transposed_centroid_lon"] = float(hist_lon) + float(offset_lon)
        if pd.notna(hist_lat):
            attrs["historical_centroid_lat"] = float(hist_lat)
            attrs["transposed_centroid_lat"] = float(hist_lat) + float(offset_lat)
    for column in [
        "historical_footprint_center_lon",
        "historical_footprint_center_lat",
        "target_footprint_center_lon",
        "target_footprint_center_lat",
        "potential_method",
    ]:
        try:
            value = row.get(column)
        except AttributeError:
            value = None
        if pd.notna(value):
            attrs[column] = str(value) if column == "potential_method" else float(value)
    out.attrs.update(attrs)
    return out


def _ensure_transposition_targets(paths, spec, ranked):
    ranked = ranked.copy()
    if "historical_centroid_lon" not in ranked:
        ranked["historical_centroid_lon"] = ranked["x"] if "x" in ranked else pd.NA
    if "historical_centroid_lat" not in ranked:
        ranked["historical_centroid_lat"] = ranked["y"] if "y" in ranked else pd.NA
    if {"historical_footprint_center_lon", "target_footprint_center_lon"}.issubset(ranked.columns):
        ranked["transposition_offset_lon"] = (
            pd.to_numeric(ranked["target_footprint_center_lon"], errors="coerce")
            - pd.to_numeric(ranked["historical_footprint_center_lon"], errors="coerce")
        )
    elif "transposition_offset_lon" not in ranked:
        ranked["transposition_offset_lon"] = 0.0
    if {"historical_footprint_center_lat", "target_footprint_center_lat"}.issubset(ranked.columns):
        ranked["transposition_offset_lat"] = (
            pd.to_numeric(ranked["target_footprint_center_lat"], errors="coerce")
            - pd.to_numeric(ranked["historical_footprint_center_lat"], errors="coerce")
        )
    elif "transposition_offset_lat" not in ranked:
        ranked["transposition_offset_lat"] = 0.0
    ranked["transposed_centroid_lon"] = (
        pd.to_numeric(ranked["historical_centroid_lon"], errors="coerce")
        + pd.to_numeric(ranked["transposition_offset_lon"], errors="coerce").fillna(0.0)
    )
    ranked["transposed_centroid_lat"] = (
        pd.to_numeric(ranked["historical_centroid_lat"], errors="coerce")
        + pd.to_numeric(ranked["transposition_offset_lat"], errors="coerce").fillna(0.0)
    )
    return ranked


def _aorc_sst_artifact_covers_settings(
    paths,
    start,
    end,
    *,
    duration_hours,
    check_every_n_hours,
    min_threshold,
    decluster_hours,
    top_n,
):
    if not source_artifact_covers(paths, "aorc_sst", "rainfall_catalog", start, end):
        return False
    manifest = read_source_artifact(paths, "aorc_sst", "rainfall_catalog") or {}
    metadata = manifest.get("metadata", {})
    return (
        _metadata_int(metadata, "duration_hours") == int(duration_hours)
        and _metadata_int(metadata, "check_every_n_hours") == int(check_every_n_hours)
        and _metadata_float(metadata, "min_precip_threshold") == float(min_threshold)
        and _metadata_int(metadata, "decluster_hours") == int(decluster_hours)
        and _metadata_optional_int(metadata, "top_n_events_safety_cap", fallback_key="top_n_events") == top_n
    )


def _metadata_int(metadata, key):
    value = metadata.get(key)
    return int(value) if value is not None else None


def _metadata_float(metadata, key):
    value = metadata.get(key)
    return float(value) if value is not None else None


def _metadata_optional_int(metadata, key, *, fallback_key=None):
    value = metadata.get(key)
    if value is None and fallback_key is not None:
        value = metadata.get(fallback_key)
    return int(value) if value is not None else None


def collect_aorc_sst(settings, skip_existing=False, opener=None):
    paths = settings["paths"]
    spec = settings.get("aorc_sst", {})
    opener = opener or _open_aorc_year
    start = pd.Timestamp(settings["start"])
    end = _normalize_end(settings["end"])
    duration_hours = int(spec.get("storm_duration_hours", spec.get("storms", {}).get("storm_duration_hours", 72)))
    check_every_n_hours = int(spec.get("check_every_n_hours", spec.get("storms", {}).get("check_every_n_hours", 1)))
    top_n_setting = spec.get("top_n_events", spec.get("storms", {}).get("top_n_events"))
    top_n = int(top_n_setting) if top_n_setting is not None else None
    min_threshold = float(spec.get("min_precip_threshold", spec.get("storms", {}).get("min_precip_threshold", 2.5)))
    decluster_hours = int(spec.get("decluster_hours", duration_hours))
    bbox_wgs84 = _bbox_from_spec(paths, spec)
    collection_dir = _collection_dir(paths, duration_hours)
    event_window_dir = collection_dir / "event_windows"
    ranked_csv = collection_dir / "ranked-storms.csv"
    stats_csv = collection_dir / "storm-stats.csv"
    yearly_stats_dir = _yearly_stats_dir(collection_dir)
    defer_event_windows = bool(spec.get("defer_event_windows", False))
    if (
        skip_existing
        and ranked_csv.exists()
        and stats_csv.exists()
        and _ranked_storms_are_sst_equivalent(ranked_csv)
        and _event_windows_include_required_meteo(
            event_window_dir,
            spec,
            require_transposition_metadata=True,
        )
        and _aorc_sst_artifact_covers_settings(
            paths,
            start,
            end,
            duration_hours=duration_hours,
            check_every_n_hours=check_every_n_hours,
            min_threshold=min_threshold,
            decluster_hours=decluster_hours,
            top_n=top_n,
        )
    ):
        print(f"AORC SST: reusing complete rainfall catalog {ranked_csv}")
        ranked = pd.read_csv(ranked_csv, parse_dates=["storm_date"])
        members = _write_rainfall_members(paths, spec, ranked, ranked_csv, duration_hours)
        return {
            "ranked_rows": int(len(ranked)),
            "storm_stats_rows": int(len(pd.read_csv(stats_csv))),
            "rainfall_member_rows": int(len(members)),
            "event_window_count": len(list(event_window_dir.glob("*.nc"))),
            "ranked_storms_csv": ranked_csv,
            "storm_stats_csv": stats_csv,
        }

    years = list(range(start.year, end.year + 1))
    candidates = []
    for year in iter_progress(years, total=len(years), desc="AORC SST years", unit="year", dynamic_ncols=True):
        yearly_csv = _yearly_stats_csv(collection_dir, year)
        if skip_existing and _stats_checkpoint_is_current(yearly_csv):
            candidates.append(pd.read_csv(yearly_csv, parse_dates=["storm_date"]))
            continue
        ds = opener(year, spec)
        try:
            year_start = max(start, pd.Timestamp(f"{year}-01-01"))
            year_end = min(end, pd.Timestamp(f"{year}-12-31 23:00:00"))
            subset = _subset_precip(ds, spec, bbox_wgs84, year_start, year_end)
            variable = spec.get("variable", "APCP_surface")
            yearly_stats = _spatial_stats(
                subset[variable],
                duration_hours,
                check_every_n_hours,
                paths=paths,
                config=settings.get("config", {}),
                spec=spec,
            )
            yearly_stats_dir.mkdir(parents=True, exist_ok=True)
            yearly_stats.to_csv(yearly_csv, index=False)
            candidates.append(yearly_stats)
        finally:
            ds.close()

    stats = pd.concat(candidates, ignore_index=True).sort_values("storm_date")
    ranked = _decluster_top_events(
        stats,
        min_threshold=min_threshold,
        decluster_hours=decluster_hours,
        top_n=top_n,
    )
    collection_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(stats_csv, index=False)
    if defer_event_windows:
        event_windows = []
    else:
        event_windows, ranked = _compute_selected_event_windows(
            opener,
            ranked,
            spec,
            bbox_wgs84,
            event_window_dir,
            duration_hours,
            paths["location_name"],
            skip_existing=skip_existing,
        )
    ranked = _ensure_transposition_targets(paths, spec, ranked)
    ranked.to_csv(ranked_csv, index=False)
    members = _write_rainfall_members(paths, spec, ranked, ranked_csv, duration_hours)
    artifact_json = write_source_artifact(
        paths,
        source="aorc_sst",
        kind="rainfall_catalog",
        start=start,
        end=end,
        artifacts={
            "storm_stats_csv": stats_csv,
            "ranked_storms_csv": ranked_csv,
            "rainfall_members_csv": paths["aorc_sst_rainfall_members_csv"],
            "event_windows_dir": event_window_dir,
        },
        metadata={
            "backend": "direct_aorc_sst",
            "bbox_wgs84": list(bbox_wgs84),
            "duration_hours": duration_hours,
            "check_every_n_hours": check_every_n_hours,
            "selection": "threshold_driven_pot",
            "min_precip_threshold": min_threshold,
            "decluster_hours": decluster_hours,
            "top_n_events_safety_cap": top_n,
            "potential_method": "moving_footprint_max_mean",
            "event_windows_deferred": defer_event_windows,
        },
        status="pending_event_windows" if defer_event_windows else "complete",
    )
    return {
        "ranked_rows": int(len(ranked)),
        "storm_stats_rows": int(len(stats)),
        "rainfall_member_rows": int(len(members)),
        "event_window_count": len(event_windows),
        "ranked_storms_csv": ranked_csv,
        "storm_stats_csv": stats_csv,
        "source_artifact_json": artifact_json,
    }


def collect_aorc_sst_event_windows(settings, skip_existing=True, opener=None):
    paths = settings["paths"]
    spec = {**settings.get("aorc_sst", {}), "write_event_windows": True}
    opener = opener or _open_aorc_year
    start = pd.Timestamp(settings["start"])
    end = _normalize_end(settings["end"])
    duration_hours = int(spec.get("storm_duration_hours", spec.get("storms", {}).get("storm_duration_hours", 72)))
    check_every_n_hours = int(spec.get("check_every_n_hours", spec.get("storms", {}).get("check_every_n_hours", 1)))
    top_n_setting = spec.get("top_n_events", spec.get("storms", {}).get("top_n_events"))
    top_n = int(top_n_setting) if top_n_setting is not None else None
    min_threshold = float(spec.get("min_precip_threshold", spec.get("storms", {}).get("min_precip_threshold", 2.5)))
    decluster_hours = int(spec.get("decluster_hours", duration_hours))
    bbox_wgs84 = _bbox_from_spec(paths, spec)
    collection_dir = _collection_dir(paths, duration_hours)
    event_window_dir = collection_dir / "event_windows"
    ranked_csv = collection_dir / "ranked-storms.csv"
    stats_csv = collection_dir / "storm-stats.csv"
    if not ranked_csv.exists():
        raise FileNotFoundError(f"AORC SST ranked storm catalog is missing: {ranked_csv}")
    ranked = pd.read_csv(ranked_csv, parse_dates=["storm_date"])
    event_windows, ranked = _compute_selected_event_windows(
        opener,
        ranked,
        spec,
        bbox_wgs84,
        event_window_dir,
        duration_hours,
        paths["location_name"],
        skip_existing=skip_existing,
    )
    ranked = _ensure_transposition_targets(paths, spec, ranked)
    ranked.to_csv(ranked_csv, index=False)
    members = _write_rainfall_members(paths, spec, ranked, ranked_csv, duration_hours)
    artifact_json = write_source_artifact(
        paths,
        source="aorc_sst",
        kind="rainfall_catalog",
        start=start,
        end=end,
        artifacts={
            "storm_stats_csv": stats_csv,
            "ranked_storms_csv": ranked_csv,
            "rainfall_members_csv": paths["aorc_sst_rainfall_members_csv"],
            "event_windows_dir": event_window_dir,
        },
        metadata={
            "backend": "direct_aorc_sst",
            "bbox_wgs84": list(bbox_wgs84),
            "duration_hours": duration_hours,
            "check_every_n_hours": check_every_n_hours,
            "selection": "threshold_driven_pot",
            "min_precip_threshold": min_threshold,
            "decluster_hours": decluster_hours,
            "top_n_events_safety_cap": top_n,
            "potential_method": "moving_footprint_max_mean",
            "event_windows_deferred": False,
        },
    )
    return {
        "ranked_rows": int(len(ranked)),
        "rainfall_member_rows": int(len(members)),
        "event_window_count": len(event_windows),
        "ranked_storms_csv": ranked_csv,
        "event_windows_dir": event_window_dir,
        "source_artifact_json": artifact_json,
    }


def _event_windows_include_required_meteo(
    event_window_dir: Path,
    spec: dict,
    *,
    require_transposition_metadata: bool = False,
) -> bool:
    meteo_cfg = spec.get("event_meteo") or {}
    if not bool(meteo_cfg.get("enabled", False)) and not require_transposition_metadata:
        return True
    files = sorted(Path(event_window_dir).glob("*.nc"))
    if not files:
        return False
    for path in files:
        if not _event_window_file_has_required_variables(
            path,
            spec,
            require_transposition_metadata=require_transposition_metadata,
        ):
            return False
    return True


# ── Transposition-domain homogeneity diagnostic ───────────────────────────────
# SST assumes storms are exchangeable across the transposition domain. This
# diagnostic samples the 72h storm maxima at the target footprint and at nearby
# candidate cells across the domain and summarizes how similar they are, so the
# homogeneity assumption can be reviewed before production collection (see
# docs/design_events/methodology/marshfield_sst_transposition_region.md).


def summarize_homogeneity(samples):
    target = samples.loc[samples["sample_role"] == "target"].iloc[0]
    candidates = samples.loc[samples["sample_role"] != "target"].copy()
    ratios = candidates["max_72h_mm"] / float(target["max_72h_mm"])
    months = sorted(int(value) for value in candidates["max_month"].dropna().unique())
    summary = {
        "target_max_72h_mm": float(target["max_72h_mm"]),
        "candidate_count": int(len(candidates)),
        "min_ratio_to_target": float(ratios.min()) if len(ratios) else None,
        "median_ratio_to_target": float(ratios.median()) if len(ratios) else None,
        "max_ratio_to_target": float(ratios.max()) if len(ratios) else None,
        "months_observed": months,
    }
    summary["review_required"] = bool(
        summary["max_ratio_to_target"] is not None
        and (
            summary["max_ratio_to_target"] > 2.0
            or summary["min_ratio_to_target"] < 0.5
            or len(months) > 4
        )
    )
    return summary


def _sample_points(counties_path, target_geometry_path, max_candidates=48):
    import geopandas as gpd

    counties = gpd.read_file(counties_path).to_crs(5070)
    target = gpd.read_file(target_geometry_path).to_crs(5070).geometry.union_all().centroid
    counties = counties.assign(distance_km=counties.geometry.centroid.distance(target) / 1000.0)
    selected = counties.sort_values("distance_km").head(max_candidates).copy()
    rows = [
        {"sample_id": "target", "sample_role": "target", "distance_km": 0.0, "geometry": target}
    ]
    for _, row in selected.iterrows():
        rows.append(
            {
                "sample_id": f"{row.get('STATEFP')}_{row.get('COUNTYFP')}_{row.get('NAME')}",
                "sample_role": "candidate",
                "distance_km": float(row["distance_km"]),
                "geometry": row.geometry.centroid,
            }
        )
    return gpd.GeoDataFrame(rows, crs=5070).to_crs(4326)


def _aorc_72h_max(points, year, variable="APCP_surface", open_zarr=None):
    open_zarr = open_zarr or xr.open_zarr
    url = f"s3://noaa-nws-aorc-v1-1-1km/{int(year)}.zarr"
    ds = open_zarr(url, storage_options={"anon": True}, chunks={})
    try:
        point_index = pd.Index(points["sample_id"], name="sample_id")
        lat = xr.DataArray(points.geometry.y.to_numpy(), dims="sample_id", coords={"sample_id": point_index})
        lon = xr.DataArray(points.geometry.x.to_numpy(), dims="sample_id", coords={"sample_id": point_index})
        series = ds[variable].sel(latitude=lat, longitude=lon, method="nearest").load()
        rolling = series.rolling(time=72, min_periods=72).sum()
        peak = rolling.max("time").to_series().rename("max_72h_mm").reset_index()
        peak_time = rolling.idxmax("time").to_series().rename("max_time").reset_index()
        out = peak.merge(peak_time, on="sample_id")
        out["max_month"] = pd.to_datetime(out["max_time"]).dt.month
        return out
    finally:
        ds.close()


def run_aorc_homogeneity_diagnostic(
    *,
    counties_path,
    target_geometry_path,
    output_dir,
    year=2018,
    max_candidates=48,
    open_zarr=None,
):
    """Sample AORC 72h maxima across the transposition domain and summarize
    homogeneity vs the target footprint. Writes samples, summary, and the
    candidate points for review."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points = _sample_points(counties_path, target_geometry_path, max_candidates=max_candidates)
    peaks = _aorc_72h_max(points, year=year, open_zarr=open_zarr)
    samples = points.drop(columns="geometry").merge(peaks, on="sample_id")
    summary = summarize_homogeneity(samples)
    summary["year"] = int(year)
    summary["sample_count"] = int(len(samples))
    samples.to_csv(output_dir / f"aorc_homogeneity_samples_{year}.csv", index=False)
    (output_dir / f"aorc_homogeneity_summary_{year}.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    points.to_file(output_dir / "aorc_homogeneity_sample_points.geojson", driver="GeoJSON")
    return summary
