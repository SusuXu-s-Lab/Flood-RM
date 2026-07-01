from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from paths import resolve_location_path
from wflow_runs.domain import configured_or_manifest_submodels
from wflow_runs.staticmaps_qa import staticmap_yx_dims, staticmaps_crs


# Mirrors HydroMT-Wflow WflowSbmModel.setup_reservoirs_no_control output. Austin
# intentionally uses no-control reservoir maps; simple-control operations require
# reviewed release/operation data and are not part of the current handoff path.
REQUIRED_RESERVOIR_STATICMAPS = (
    "reservoir_area_id",
    "reservoir_outlet_id",
    "reservoir_initial_depth",
    "meta_reservoir_mean_outflow",
    "reservoir_b",
    "reservoir_e",
    "reservoir_rating_curve",
    "reservoir_storage_curve",
)
IMPORTANT_RESERVOIR_NAMES = ("Lake Travis", "Lake Austin", "Lady Bird Lake")


def wflow_reservoirs_enabled(config: dict) -> bool:
    return bool(
        ((config.get("collection", {}) or {}).get("national_hydrography", {}) or {})
        .get("reservoirs", {})
        .get("enabled", False)
    )


def assert_wflow_reservoir_staticmaps_current(config: dict, model_root: Path, submodel_id: str) -> None:
    if not wflow_reservoirs_enabled(config):
        return
    report = validate_wflow_reservoir_staticmaps(model_root, required=True, raise_on_error=False)
    failed = report[report["status"].isin(["failed", "review_required"])]
    if failed.empty:
        return
    details = "; ".join(f"{row.check}: {row.message}" for row in failed.itertuples())
    raise RuntimeError(
        f"Wflow base {submodel_id!r} is stale for enabled reservoirs: {details}. "
        "Rebuild the Wflow base with setup_reservoirs_no_control active."
    )


def validate_wflow_reservoir_staticmaps(
    model_root,
    *,
    required: bool = True,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """QA native HydroMT-Wflow reservoir maps in ``staticmaps.nc``."""
    staticmaps_path = Path(model_root) / "staticmaps.nc"
    if not staticmaps_path.exists():
        raise FileNotFoundError(staticmaps_path)
    rows: list[dict] = []
    with xr.open_dataset(staticmaps_path, mask_and_scale=False) as ds:
        missing = [name for name in REQUIRED_RESERVOIR_STATICMAPS if name not in ds]
        if missing:
            status = "failed" if required else "not_available"
            rows.append(
                {
                    "check": "reservoir_staticmaps",
                    "status": status,
                    "message": f"missing={missing}",
                }
            )
            report = pd.DataFrame(rows)
            if raise_on_error and status == "failed":
                raise RuntimeError(f"Wflow reservoir staticmap QA failed for {staticmaps_path}: missing={missing}")
            return report
        _append_wflow_reservoir_id_checks(rows, ds)
        _append_wflow_reservoir_parameter_checks(rows, ds)
    report = pd.DataFrame(rows)
    failed = report[report["status"].isin(["failed", "review_required"])] if not report.empty else report
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{row.check}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Wflow reservoir staticmap QA failed for {staticmaps_path}: {details}")
    return report


def write_wflow_reservoir_readiness(
    config: dict,
    location_root,
    *,
    submodel_id: str | None = None,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    """Write reservoir staticmap/outlet QA for configured Wflow submodels."""
    location_root = Path(location_root)
    base_root = resolve_location_path(location_root, config.get("wflow", {}).get("base_model_root", "data/wflow/base"))
    readiness_root = resolve_location_path(location_root, config.get("wflow", {}).get("readiness_root", "data/wflow/readiness"))
    reservoir_cfg = ((config.get("collection", {}) or {}).get("national_hydrography", {}) or {}).get("reservoirs", {}) or {}
    reservoirs_path = reservoir_cfg.get("output", "data/sources/national_hydrography/nhdplus_hr_wflow_reservoirs.gpkg")
    reservoirs_path = resolve_location_path(location_root, reservoirs_path)
    rows = []
    submodels = configured_or_manifest_submodels(config, location_root)
    if submodel_id is not None:
        submodels = [submodel for submodel in submodels if str(submodel.get("wflow_submodel_id")) == str(submodel_id)]
    for submodel in submodels:
        current_id = str(submodel["wflow_submodel_id"])
        model_root = base_root / current_id
        static_report = validate_wflow_reservoir_staticmaps(
            model_root,
            required=wflow_reservoirs_enabled(config),
            raise_on_error=False,
        )
        static_report.insert(0, "submodel_id", current_id)
        rows.append(static_report)
        outlet_report = validate_wflow_reservoir_outlets(
            model_root,
            reservoirs_path=reservoirs_path,
            raise_on_error=False,
        )
        outlet_report.insert(0, "submodel_id", current_id)
        rows.append(outlet_report)
    report = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        [{"submodel_id": "<none>", "check": "reservoir_submodels", "status": "failed", "message": "no Wflow submodels found"}]
    )
    readiness_root.mkdir(parents=True, exist_ok=True)
    report.to_csv(readiness_root / "wflow_reservoir_readiness.csv", index=False)
    (readiness_root / "wflow_reservoir_readiness.json").write_text(
        json.dumps(report.to_dict(orient="records"), indent=2, default=str),
        encoding="utf-8",
    )
    failed = report[report["status"].isin(["failed", "review_required"])] if not report.empty else report
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{row.submodel_id}:{row.check}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Wflow reservoir readiness failed: {details}")
    return report


def validate_wflow_reservoir_outlets(
    model_root,
    *,
    reservoirs_path=None,
    important_names: tuple[str, ...] = IMPORTANT_RESERVOIR_NAMES,
    max_river_distance_m: float = 1000.0,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """Check that Wflow reservoir outlets exist and align with river cells."""
    model_root = Path(model_root)
    staticmaps_path = model_root / "staticmaps.nc"
    if not staticmaps_path.exists():
        raise FileNotFoundError(staticmaps_path)
    rows: list[dict] = []
    with xr.open_dataset(staticmaps_path, mask_and_scale=False) as ds:
        if "reservoir_outlet_id" not in ds or "reservoir_area_id" not in ds:
            rows.append({"check": "reservoir_outlets", "status": "failed", "message": "missing reservoir outlet or area maps"})
            report = pd.DataFrame(rows)
            if raise_on_error:
                raise RuntimeError(f"Wflow reservoir outlet QA failed for {staticmaps_path}: missing reservoir maps")
            return report
        outlet = np.asarray(ds["reservoir_outlet_id"].values)
        area = np.asarray(ds["reservoir_area_id"].values)
        outlet_ids = {int(value) for value in np.unique(outlet) if np.isfinite(value) and value > 0}
        area_ids = {int(value) for value in np.unique(area) if np.isfinite(value) and value > 0}
        missing_outlets = sorted(area_ids - outlet_ids)
        outlet_points = _reservoir_outlet_points(ds)
    status = "passed" if outlet_ids and not missing_outlets else "failed"
    rows.append(
        {
            "check": "reservoir_outlets",
            "status": status,
            "message": f"reservoir_ids={len(area_ids)}; outlet_ids={len(outlet_ids)}; missing_outlet_ids={missing_outlets}",
        }
    )
    if not outlet_points.empty:
        _append_wflow_reservoir_river_distance_checks(
            rows,
            model_root,
            outlet_points,
            max_river_distance_m=max_river_distance_m,
        )
    if reservoirs_path is not None and Path(reservoirs_path).exists():
        _append_wflow_reservoir_source_checks(
            rows,
            reservoirs_path=Path(reservoirs_path),
            area_ids=area_ids,
            outlet_ids=outlet_ids,
            important_names=important_names,
        )
    report = pd.DataFrame(rows)
    failed = report[report["status"].isin(["failed", "review_required"])] if not report.empty else report
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{row.check}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Wflow reservoir outlet QA failed for {staticmaps_path}: {details}")
    return report


def _append_wflow_reservoir_id_checks(rows: list[dict], ds: xr.Dataset) -> None:
    area = np.asarray(ds["reservoir_area_id"].values)
    outlet = np.asarray(ds["reservoir_outlet_id"].values)
    area_ids = sorted(int(value) for value in np.unique(area) if np.isfinite(value) and value > 0)
    outlet_ids = sorted(int(value) for value in np.unique(outlet) if np.isfinite(value) and value > 0)
    missing_outlets = sorted(set(area_ids) - set(outlet_ids))
    rows.append(
        {
            "check": "reservoir_area_id",
            "status": "passed" if area_ids else "failed",
            "message": f"reservoir_ids={area_ids}; area_cells={int(np.count_nonzero(area > 0))}",
        }
    )
    rows.append(
        {
            "check": "reservoir_outlet_id",
            "status": "passed" if outlet_ids and not missing_outlets else "failed",
            "message": f"outlet_ids={outlet_ids}; outlet_cells={int(np.count_nonzero(outlet > 0))}; missing_outlet_ids={missing_outlets}",
        }
    )


def _append_wflow_reservoir_parameter_checks(rows: list[dict], ds: xr.Dataset) -> None:
    area_mask = np.asarray(ds["reservoir_area_id"].values) > 0
    parameter_names = (
        "reservoir_initial_depth",
        "meta_reservoir_mean_outflow",
        "reservoir_b",
        "reservoir_e",
        "reservoir_rating_curve",
        "reservoir_storage_curve",
    )
    for name in parameter_names:
        values = np.asarray(ds[name].values, dtype=float)
        mask = area_mask if values.shape == area_mask.shape else np.isfinite(values)
        valid = values[mask & np.isfinite(values)]
        positive_required = name not in {"reservoir_rating_curve", "reservoir_storage_curve"}
        if positive_required:
            valid = valid[valid > 0]
        status = "passed" if valid.size else "failed"
        vmin = float(np.nanmin(valid)) if valid.size else np.nan
        vmax = float(np.nanmax(valid)) if valid.size else np.nan
        rows.append(
            {
                "check": name,
                "status": status,
                "message": f"valid_cells={int(valid.size)}; min={vmin:g}; max={vmax:g}",
            }
        )


def _reservoir_outlet_points(ds: xr.Dataset) -> gpd.GeoDataFrame:
    outlet = ds["reservoir_outlet_id"]
    values = np.asarray(outlet.values)
    if values.ndim < 2:
        return gpd.GeoDataFrame({"reservoir_id": []}, geometry=[], crs=None)
    y_dim, x_dim = staticmap_yx_dims(outlet)
    ys = np.asarray(ds.coords[y_dim].values, dtype=float)
    xs = np.asarray(ds.coords[x_dim].values, dtype=float)
    crs = staticmaps_crs(ds, x_dim=x_dim, y_dim=y_dim)
    records = []
    for reservoir_id in sorted(int(value) for value in np.unique(values) if np.isfinite(value) and value > 0):
        positions = np.argwhere(values == reservoir_id)
        if positions.size == 0:
            continue
        row, col = positions[0][-2], positions[0][-1]
        records.append(
            {
                "reservoir_id": reservoir_id,
                "geometry": Point(float(xs[col]), float(ys[row])),
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def _append_wflow_reservoir_river_distance_checks(
    rows: list[dict],
    model_root: Path,
    outlet_points: gpd.GeoDataFrame,
    *,
    max_river_distance_m: float,
) -> None:
    rivers_path = model_root / "staticgeoms" / "rivers.geojson"
    if not rivers_path.exists():
        rows.append({"check": "reservoir_outlet_river_distance", "status": "review_required", "message": f"missing {rivers_path}"})
        return
    rivers = gpd.read_file(rivers_path)
    if rivers.empty:
        rows.append({"check": "reservoir_outlet_river_distance", "status": "failed", "message": "staticgeoms/rivers.geojson is empty"})
        return
    if outlet_points.crs is None:
        rows.append({"check": "reservoir_outlet_river_distance", "status": "review_required", "message": "reservoir outlet points have no CRS"})
        return
    if rivers.crs is None:
        rivers = rivers.set_crs(outlet_points.crs)
    rivers = rivers.to_crs(outlet_points.crs)
    points_m, rivers_m = _project_distance_geometries(outlet_points, rivers)
    river_union = rivers_m.geometry.union_all()
    distances = points_m.geometry.distance(river_union)
    max_distance = float(distances.max()) if len(distances) else np.nan
    too_far = points_m.loc[distances > float(max_river_distance_m), "reservoir_id"].astype(int).tolist()
    status = "passed" if not too_far else "review_required"
    rows.append(
        {
            "check": "reservoir_outlet_river_distance",
            "status": status,
            "message": f"max_distance_m={max_distance:g}; too_far_ids={too_far}",
        }
    )


def _append_wflow_reservoir_source_checks(
    rows: list[dict],
    *,
    reservoirs_path: Path,
    area_ids: set[int],
    outlet_ids: set[int],
    important_names: tuple[str, ...],
) -> None:
    reservoirs = gpd.read_file(reservoirs_path)
    if reservoirs.empty:
        rows.append({"check": "reservoir_source", "status": "failed", "message": f"empty source {reservoirs_path}"})
        return
    if "waterbody_id" not in reservoirs:
        rows.append({"check": "reservoir_source", "status": "failed", "message": "source missing waterbody_id"})
        return
    source_ids = set(pd.to_numeric(reservoirs["waterbody_id"], errors="coerce").dropna().astype(int))
    missing_area = sorted(source_ids - area_ids)
    missing_outlet = sorted(source_ids - outlet_ids)
    status = "passed" if not missing_area and not missing_outlet else "review_required"
    rows.append(
        {
            "check": "reservoir_source_ids",
            "status": status,
            "message": f"source_ids={sorted(source_ids)}; missing_area_ids={missing_area}; missing_outlet_ids={missing_outlet}",
        }
    )
    if "waterbody_name" not in reservoirs:
        return
    names = reservoirs["waterbody_name"].fillna("").astype(str)
    present = set(names)
    missing_important = [name for name in important_names if name not in present]
    important_ids = sorted(
        int(value)
        for value in pd.to_numeric(reservoirs.loc[names.isin(important_names), "waterbody_id"], errors="coerce").dropna()
    )
    important_missing_outlets = sorted(set(important_ids) - outlet_ids)
    status = "passed" if not missing_important and not important_missing_outlets else "review_required"
    rows.append(
        {
            "check": "important_reservoir_outlets",
            "status": status,
            "message": f"names={list(important_names)}; source_ids={important_ids}; missing_names={missing_important}; missing_outlet_ids={important_missing_outlets}",
        }
    )


def _project_distance_geometries(points: gpd.GeoDataFrame, lines: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    crs = points.crs
    if crs is not None and getattr(crs, "is_projected", False):
        return points, lines
    try:
        target_crs = points.estimate_utm_crs()
    except Exception:
        target_crs = None
    if target_crs is None:
        target_crs = "EPSG:3857"
    return points.to_crs(target_crs), lines.to_crs(target_crs)
