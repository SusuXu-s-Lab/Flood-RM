from __future__ import annotations

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


def validate_event_boundary(
    discharge_nc: str | Path,
    *,
    expected_source_ids: set[str] | None = None,
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    zero_rain_discharge_nc: str | Path | None = None,
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
    rows.append(_shape_diversity_row(discharge_nc, max_correlation=max_shape_correlation))
    if zero_rain_discharge_nc is not None and Path(zero_rain_discharge_nc).exists():
        zero_peak = total_peak(zero_rain_discharge_nc)
        fraction = zero_peak / peak if peak > 0 else np.inf
        rows.append({"check": "zero_rain_peak_fraction", "status": "diagnostic", "message": f"zero_peak_m3s={zero_peak:g}; fraction={fraction:g}"})
    report = pd.DataFrame(rows)
    failed = report[report["status"].isin(["failed", "review_required"])] if not report.empty else report
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{r.check}: {r.message}" for r in failed.itertuples())
        raise RuntimeError(f"Wflow event boundary QA failed: {details}")
    return report


def validate_handoff_gauge_locations(source_locations, gauge_locations, *, model_crs: str | None = None, max_distance_m: float = 100.0, raise_on_error: bool = True) -> pd.DataFrame:
    """Check SFINCS source points against HydroMT-snapped Wflow gauges."""
    import geopandas as gpd

    sources = gpd.read_file(source_locations) if isinstance(source_locations, (str, Path)) else source_locations.copy()
    gauges = gpd.read_file(gauge_locations) if isinstance(gauge_locations, (str, Path)) else gauge_locations.copy()
    row = {"check": "wflow_gauge_source_distance", "status": "passed", "message": ""}
    if sources.empty or gauges.empty:
        row.update(status="failed", message="empty sources or gauges")
        return _report([row], raise_on_error)
    if model_crs:
        sources = sources.to_crs(model_crs) if sources.crs else sources.set_crs(model_crs)
        gauges = gauges.to_crs(model_crs) if gauges.crs else gauges.set_crs(model_crs)
    if not getattr(sources.crs, "is_projected", False):
        target = sources.estimate_utm_crs() or "EPSG:3857"
        sources = sources.to_crs(target)
        gauges = gauges.to_crs(target)
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
        if distance > float(max_distance_m):
            too_far[hid] = distance
    row.update(
        status="passed" if not missing and not stale and not too_far else "failed",
        message=f"max_distance_m={max_seen:g}; missing={missing or 'none'}; stale={stale or 'none'}; too_far={too_far or 'none'}",
    )
    return _report([row], raise_on_error)


def write_acceptance(path: str | Path, *, event, discharge_nc: str | Path, qa_report: pd.DataFrame, amplification: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> Path:
    accepted = bool(not qa_report["status"].isin(["failed", "review_required"]).any())
    payload = {
        "event_id": str(event.event_id if hasattr(event, "event_id") else event),
        "status": "accepted" if accepted else "failed",
        "method": "wflow_dynamic_boundary_condition",
        "discharge_nc": str(discharge_nc),
        "probability": event.probability.to_dict() if hasattr(event, "probability") else {},
        "event": event.to_dict() if hasattr(event, "to_dict") else {},
        "amplification": amplification or {"K": 1.0, "status": "not_applied"},
        "checks": qa_report.to_dict(orient="records"),
        "metadata": metadata or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def read_acceptance(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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
        return {"check": "source_hydrograph_shape_diversity", "status": "passed", "message": "sources<2"}
    corr = frame.corr().abs()
    duplicates: list[str] = []
    max_seen = 0.0
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1:]:
            value = float(corr.loc[left, right])
            max_seen = max(max_seen, value)
            if value >= float(max_correlation):
                duplicates.append(f"{left}~{right}:{value:.6f}")
    return {"check": "source_hydrograph_shape_diversity", "status": "failed" if duplicates else "passed", "message": f"max_corr={max_seen:.6f}; duplicate_pairs={duplicates or 'none'}"}


def _report(rows: list[dict[str, Any]], raise_on_error: bool) -> pd.DataFrame:
    report = pd.DataFrame(rows)
    failed = report[report["status"].isin(["failed", "review_required"])] if not report.empty else report
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{r.check}: {r.message}" for r in failed.itertuples())
        raise RuntimeError(f"Boundary QA failed: {details}")
    return report
