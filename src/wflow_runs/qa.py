from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import xarray as xr


def discharge_source_ids(discharge_nc: str | Path) -> list[str]:
    with xr.open_dataset(discharge_nc) as ds:
        if "name" in ds:
            return [str(v) for v in ds["name"].values.tolist()]
        if "index" in ds:
            return [str(v) for v in ds["index"].values.tolist()]
    return []


def discharge_frame(discharge_nc: str | Path) -> pd.DataFrame:
    with xr.open_dataset(discharge_nc) as ds:
        if "discharge" not in ds:
            raise ValueError(f"{discharge_nc} lacks discharge")
        da = ds["discharge"]
        frame = da.transpose("time", "index").to_pandas() if {"time", "index"} <= set(da.dims) else da.to_pandas()
        if "name" in ds and len(ds["name"]) == frame.shape[1]:
            frame.columns = [str(v) for v in ds["name"].values]
    frame.index = pd.DatetimeIndex(frame.index)
    return frame.astype(float)


def total_peak(discharge_nc: str | Path) -> float:
    with xr.open_dataset(discharge_nc) as ds:
        da = ds["discharge"]
        if "index" in da.dims:
            da = da.sum("index", skipna=True)
        return float(da.max(skipna=True))


def event_discharge_path(
    row: Mapping[str, Any],
    *,
    location_root: str | Path,
    events_root: str | Path,
    event_id: str,
    discharge_filename: str = "sfincs_discharge.nc",
) -> Path:
    """Resolve the Wflow-to-SFINCS discharge artifact for one Event Catalog row."""
    location_root = Path(location_root)
    events_root = _resolve_path(location_root, events_root)
    for column in ("wflow_discharge_forcing", "sfincs_discharge_forcing"):
        value = row.get(column)
        if value not in (None, "") and not _is_missing_value(value):
            return _resolve_path(location_root, value)
    value = row.get("wflow_event_dir")
    if value not in (None, "") and not _is_missing_value(value):
        return _resolve_path(location_root, value) / discharge_filename
    return events_root / str(event_id) / discharge_filename


def event_peak_discharge_table(
    catalog,
    *,
    location_root: str | Path,
    events_root: str | Path | None = None,
    discharge_filename: str = "sfincs_discharge.nc",
) -> pd.DataFrame:
    """Attach Wflow handoff peak discharge metrics to Event Catalog rows."""
    location_root = Path(location_root)
    events_root = _resolve_path(location_root, events_root or "data/wflow/events")
    frame = catalog.copy()
    if "event_id" not in frame:
        raise ValueError("catalog must include an event_id column")

    rows = []
    for _, row in frame.iterrows():
        event_id = str(row["event_id"])
        discharge_nc = event_discharge_path(
            row,
            location_root=location_root,
            events_root=events_root,
            event_id=event_id,
            discharge_filename=discharge_filename,
        )
        metrics = {
            "event_id": event_id,
            "wflow_discharge_nc": str(discharge_nc),
            "wflow_discharge_exists": discharge_nc.exists(),
            "peak_discharge_m3s": np.nan,
            "max_source_peak_discharge_m3s": np.nan,
            "handoff_source_count": np.nan,
        }
        if discharge_nc.exists():
            with xr.open_dataset(discharge_nc) as ds:
                if "discharge" in ds and "time" in ds["discharge"].dims:
                    discharge = ds["discharge"]
                    source_dims = tuple(dim for dim in discharge.dims if dim != "time")
                    total = discharge.sum(dim=source_dims, skipna=True) if source_dims else discharge
                    metrics["peak_discharge_m3s"] = float(total.max(skipna=True))
                    if source_dims:
                        source_peak = discharge.max(dim="time", skipna=True)
                        metrics["max_source_peak_discharge_m3s"] = float(source_peak.max(skipna=True))
                        metrics["handoff_source_count"] = int(source_peak.size)
                    else:
                        metrics["max_source_peak_discharge_m3s"] = metrics["peak_discharge_m3s"]
                        metrics["handoff_source_count"] = 1
        rows.append(metrics)

    return frame.merge(pd.DataFrame(rows), on="event_id", how="left")


def _resolve_path(root: Path, value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _is_missing_value(value) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def validate_event_boundary(
    discharge_nc: str | Path,
    *,
    expected_source_ids: set[str] | None = None,
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    zero_rain_discharge_nc: str | Path | None = None,
    max_zero_peak_fraction: float | None = None,
    max_shape_correlation: float = 0.9999,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """Validate a SFINCS discharge boundary generated from Wflow."""
    rows: list[dict[str, Any]] = []
    peak = total_peak(discharge_nc)
    rows.append({"check": "event_peak", "status": "passed" if peak > 0 else "failed", "message": f"peak_m3s={peak:g}"})
    if expected_source_ids is not None:
        found = set(discharge_source_ids(discharge_nc))
        rows.append(_source_id_row(expected_source_ids, found))
    if window is not None:
        rows.append(_time_window_row(discharge_nc, window))
    if zero_rain_discharge_nc is not None:
        zero_peak = total_peak(zero_rain_discharge_nc)
        fraction = zero_peak / peak if peak > 0 else np.inf
        threshold = "" if max_zero_peak_fraction is None else f"; diagnostic_threshold={float(max_zero_peak_fraction):g}"
        rows.append(
            {
                "check": "zero_rain_peak_fraction",
                "status": "diagnostic",
                "message": f"zero_peak_m3s={zero_peak:g}; fraction={fraction:g}{threshold}",
            }
        )
    rows.append(_shape_diversity_row(discharge_nc, max_correlation=max_shape_correlation))
    report = pd.DataFrame(rows)
    failed = report[report["status"].isin(["failed", "review_required"])] if not report.empty else report
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{r.check}: {r.message}" for r in failed.itertuples())
        raise RuntimeError(f"Wflow event boundary QA failed: {details}")
    return report


def validate_handoff_gauge_locations(
    source_locations,
    gauge_locations,
    *,
    model_crs: str | None = None,
    max_distance_m: float = 100.0,
    reservoir_boundary_max_distance_m: float | None = None,
    submodel_id: str | None = None,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """Check SFINCS source points against HydroMT-snapped Wflow gauges."""
    import geopandas as gpd

    sources = _read_geodataframe(source_locations, gpd)
    gauges = _read_geodataframe(gauge_locations, gpd)
    if submodel_id is not None and "wflow_submodel_id" in sources:
        sources = sources[sources["wflow_submodel_id"].astype(str).eq(str(submodel_id))].copy()
    row = {"check": "wflow_gauge_source_distance", "status": "passed", "message": ""}
    if submodel_id is not None:
        row["submodel_id"] = str(submodel_id)
    if sources.empty:
        row.update(status="skipped", message="no SFINCS handoff sources for submodel")
        return _handoff_gauge_location_report(row, raise_on_error=raise_on_error)
    if gauges.empty:
        row.update(status="failed", message="empty Wflow gauges_sfincs layer")
        return _handoff_gauge_location_report(row, raise_on_error=raise_on_error)
    if "sfincs_handoff_id" not in sources:
        row.update(status="failed", message="SFINCS source layer lacks sfincs_handoff_id")
        return _handoff_gauge_location_report(row, raise_on_error=raise_on_error)
    if "sfincs_handoff_id" not in gauges and "name" not in gauges:
        row.update(status="failed", message="Wflow gauges layer lacks sfincs_handoff_id/name")
        return _handoff_gauge_location_report(row, raise_on_error=raise_on_error)
    sources = _normalize_geodataframe_crs(sources, model_crs)
    gauges = _normalize_geodataframe_crs(gauges, str(sources.crs) if sources.crs is not None else model_crs)
    sources, gauges = _project_distance_geometries(sources, gauges)
    source_ids = set(sources["sfincs_handoff_id"].astype(str))
    gauge_col = "sfincs_handoff_id" if "sfincs_handoff_id" in gauges else "name"
    gauge_ids = set(gauges[gauge_col].astype(str))
    missing = sorted(source_ids - gauge_ids)
    stale = sorted(gauge_ids - source_ids)
    too_far: dict[str, float] = {}
    max_seen = np.nan
    for _, src in sources.iterrows():
        hid = str(src["sfincs_handoff_id"])
        matched = gauges[gauges[gauge_col].astype(str).eq(hid)]
        if matched.empty:
            continue
        distance = float(matched.geometry.distance(src.geometry).min())
        max_seen = distance if not np.isfinite(max_seen) else max(max_seen, distance)
        limit = _handoff_gauge_distance_limit(
            src,
            max_distance_m=max_distance_m,
            reservoir_boundary_max_distance_m=reservoir_boundary_max_distance_m,
        )
        if distance > limit:
            too_far[hid] = distance
    too_far_preview = ", ".join(f"{hid}={distance:.1f} m" for hid, distance in list(sorted(too_far.items()))[:8])
    if len(too_far) > 8:
        too_far_preview += f", +{len(too_far) - 8} more"
    row.update(
        status="passed" if not missing and not stale and not too_far else "failed",
        message=(
            f"max_distance_m={max_seen:.3f}; limit_m={float(max_distance_m):g}; "
            f"reservoir_limit_m={reservoir_boundary_max_distance_m if reservoir_boundary_max_distance_m is not None else 'none'}; "
            f"too_far={too_far_preview or 'none'}; missing={missing or 'none'}; stale={stale or 'none'}"
        ),
    )
    return _handoff_gauge_location_report(row, raise_on_error=raise_on_error)


def _handoff_gauge_distance_limit(source, *, max_distance_m: float, reservoir_boundary_max_distance_m: float | None) -> float:
    if reservoir_boundary_max_distance_m is None:
        return float(max_distance_m)
    placement = str(source.get("handoff_placement", "") or "").lower()
    if placement == "sfincs_native_reservoir_boundary_inflow":
        return float(reservoir_boundary_max_distance_m)
    return float(max_distance_m)


def _handoff_gauge_location_report(row: dict[str, Any], *, raise_on_error: bool) -> pd.DataFrame:
    report = pd.DataFrame([row])
    failed = report[report["status"].isin(["failed", "review_required"])]
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{item.check}: {item.message}" for item in failed.itertuples())
        raise RuntimeError(f"Wflow handoff gauge location QA failed: {details}")
    return report


def _read_geodataframe(value, gpd):
    if hasattr(value, "geometry") and hasattr(value, "crs"):
        return value.copy()
    return gpd.read_file(value)


def _normalize_geodataframe_crs(gdf, model_crs: str | None):
    if model_crs is not None:
        if gdf.crs is None:
            return gdf.set_crs(model_crs)
        return gdf.to_crs(model_crs)
    return gdf


def _project_distance_geometries(points, targets):
    crs = points.crs
    if crs is not None and getattr(crs, "is_projected", False):
        return points, targets
    try:
        target_crs = points.estimate_utm_crs()
    except Exception:
        target_crs = None
    if target_crs is None:
        target_crs = "EPSG:3857"
    return points.to_crs(target_crs), targets.to_crs(target_crs)


def write_acceptance(path: str | Path, *, event, discharge_nc: str | Path, qa_report: pd.DataFrame, amplification: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> Path:
    accepted = bool(not qa_report["status"].isin(["failed", "review_required"]).any())
    payload = {
        "event_id": str(event.event_id if hasattr(event, "event_id") else event),
        "status": "accepted" if accepted else "failed",
        "method": "wflow_dynamic_boundary_condition",
        "discharge_nc": str(discharge_nc),
        "probability": _event_probability(event),
        "event": event.to_dict() if hasattr(event, "to_dict") else {},
        "amplification": amplification or {"K": 1.0, "status": "not_applied"},
        "checks": qa_report.to_dict(orient="records"),
        "metadata": metadata or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_dynamic_handoff_acceptance(path, *, event_id: str, discharge_nc, qa_report: pd.DataFrame, metadata: dict | None = None) -> Path:
    """Write the legacy notebook dynamic-handoff acceptance JSON schema."""
    path = Path(path)
    accepted = bool(not qa_report["status"].isin(["failed", "review_required"]).any())
    payload = {
        "event_id": str(event_id),
        "status": "accepted" if accepted else "failed",
        "discharge_source": "wflow_dynamic",
        "discharge_nc": str(discharge_nc),
        "checks": qa_report.to_dict(orient="records"),
        "metadata": metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_acceptance(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _event_probability(event) -> dict[str, Any]:
    value = getattr(event, "probability", None)
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return {}


def _source_id_row(expected: set[str], found: set[str]) -> dict[str, Any]:
    missing = sorted(expected - found)
    stale = sorted(found - expected)
    return {"check": "source_ids", "status": "passed" if not missing and not stale else "failed", "message": f"missing={missing or 'none'}; stale={stale or 'none'}"}


def _time_window_row(discharge_nc: str | Path, window: tuple[pd.Timestamp, pd.Timestamp]) -> dict[str, Any]:
    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    with xr.open_dataset(discharge_nc) as ds:
        if "time" not in ds:
            return {"check": "time_coverage", "status": "failed", "message": "missing time coordinate"}
        tmin = pd.Timestamp(ds["time"].min().values)
        tmax = pd.Timestamp(ds["time"].max().values)
    ok = tmin <= start and tmax >= end
    return {"check": "time_coverage", "status": "passed" if ok else "failed", "message": f"actual={tmin.isoformat()}..{tmax.isoformat()}; expected={start.isoformat()}..{end.isoformat()}"}


def _shape_diversity_row(discharge_nc: str | Path, *, max_correlation: float) -> dict[str, Any]:
    frame = discharge_frame(discharge_nc)
    if frame.shape[1] < 2:
        return {
            "check": "source_hydrograph_shape_diversity",
            "status": "passed",
            "message": "sources<2; diversity check skipped",
        }
    values = frame.to_numpy(dtype=float).T
    names = [str(name) for name in frame.columns]
    max_seen = -np.inf
    duplicate_pairs: list[str] = []
    for i in range(values.shape[0]):
        left = _normalized_shape(values[i])
        for j in range(i + 1, values.shape[0]):
            right = _normalized_shape(values[j])
            if np.allclose(left, right, rtol=1e-6, atol=1e-8):
                corr = 1.0
            else:
                corr = float(np.corrcoef(left, right)[0, 1])
            max_seen = max(max_seen, corr)
            if corr >= float(max_correlation):
                duplicate_pairs.append(f"{names[i]}~{names[j]}:{corr:.6f}")
    pairs = ", ".join(duplicate_pairs[:5]) if duplicate_pairs else "none"
    if len(duplicate_pairs) > 5:
        pairs += f", +{len(duplicate_pairs) - 5} more"
    return {
        "check": "source_hydrograph_shape_diversity",
        "status": "failed" if duplicate_pairs else "passed",
        "message": f"sources={values.shape[0]}; max_corr={max_seen:.6f}; duplicate_shape_pairs={pairs}",
    }


def _normalized_shape(values: np.ndarray) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    if clean.size == 0:
        return clean
    fill = np.nanmedian(clean) if np.isfinite(clean).any() else 0.0
    clean = np.nan_to_num(clean, nan=float(fill), posinf=float(fill), neginf=float(fill))
    span = float(clean.max() - clean.min())
    if span <= 0.0:
        return np.zeros_like(clean, dtype=float)
    return (clean - float(clean.min())) / span


def _report(rows: list[dict[str, Any]], raise_on_error: bool) -> pd.DataFrame:
    report = pd.DataFrame(rows)
    failed = report[report["status"].isin(["failed", "review_required"])] if not report.empty else report
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{r.check}: {r.message}" for r in failed.itertuples())
        raise RuntimeError(f"Boundary QA failed: {details}")
    return report
