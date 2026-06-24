from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def discharge_source_ids(discharge_nc) -> list[str]:
    with xr.open_dataset(discharge_nc) as ds:
        if "name" in ds:
            return [str(value) for value in ds["name"].values.tolist()]
        return [str(value) for value in ds.get("index", []).values.tolist()]


CFS_TO_CMS = 0.028316846592


def validate_baseflow_against_observed(
    config: dict,
    location_root,
    *,
    zero_rain_discharge_nc,
    streamflow_records_csv=None,
    min_baseflow_fraction: float = 0.25,
) -> pd.DataFrame:
    """ADR-0016 Wflow Readiness: confirm warm-state spin-up left non-dry channels.

    Baseflow in Wflow is established by warm hydrologic states (hydromt-wflow
    ``setup_cold_states`` + warmup spin-up; ``model.states``), not by injected forcing. The
    zero-rain control already isolates that baseflow (precip zeroed), so this compares the
    control's per-handoff baseflow to the observed low-flow at the Primary Reference Gage,
    transferred to each crossing by the drainage-area (``uparea``) ratio — a standard low-flow
    regionalization. A near-zero simulated baseflow (flatlined / dry river) fails the gate so
    spin-up can be extended, rather than silently handing SFINCS a dry inflow.
    """
    import geopandas as gpd

    location_root = Path(location_root)
    cfg = (((config.get("inland_coupling", {}) or {}).get("baseflow", {})) or {})
    rows: list[dict] = []
    if not cfg.get("enabled", True):
        return pd.DataFrame([{"check": "baseflow", "status": "disabled", "message": "inland_coupling.baseflow.enabled is false"}])

    reference_gage = str(
        cfg.get("reference_gage")
        or (((config.get("inland_coupling", {}) or {}).get("amplification", {}) or {}).get("primary_reference_gage"))
        or ""
    )
    statistic = str(cfg.get("reference_statistic", "median")).lower()
    if not reference_gage:
        return pd.DataFrame([{"check": "baseflow", "status": "skipped", "message": "no reference_gage configured"}])

    records_csv = Path(streamflow_records_csv) if streamflow_records_csv else (
        location_root / "data/sources/usgs_streamgages/streamflow_records.csv"
    )
    if not records_csv.exists():
        return pd.DataFrame([{"check": "baseflow", "status": "skipped", "message": f"missing {records_csv}"}])
    records = pd.read_csv(records_csv, dtype={"site_no": str})
    site = records[records["site_no"].astype(str).str.zfill(8) == reference_gage.zfill(8)]
    q = pd.to_numeric(site.get("discharge_cfs"), errors="coerce").dropna()
    if q.empty:
        return pd.DataFrame([{"check": "baseflow", "status": "skipped", "message": f"no records for {reference_gage}"}])
    observed_baseflow_cfs = {
        "median": float(q.median()),
        "annual_mean": float(q.mean()),
        "q90": float(q.quantile(0.10)),
    }.get(statistic, float(q.median()))
    observed_baseflow_cms = observed_baseflow_cfs * CFS_TO_CMS

    uparea_by_handoff, reference_uparea = _handoff_upareas(location_root, reference_gage, gpd)
    if not uparea_by_handoff or not reference_uparea:
        return pd.DataFrame([{"check": "baseflow", "status": "skipped", "message": "could not resolve handoff/reference uparea"}])

    with xr.open_dataset(zero_rain_discharge_nc) as opened:
        ds = opened.load()
    names = [str(v) for v in ds["name"].values.tolist()] if "name" in ds else [str(v) for v in ds["index"].values.tolist()]
    sim = np.asarray(ds["discharge"].transpose("index", "time").values, dtype=float)
    for i, handoff_id in enumerate(names):
        upa = uparea_by_handoff.get(handoff_id)
        if not upa:
            continue
        expected_cms = observed_baseflow_cms * (float(upa) / float(reference_uparea))
        simulated_cms = float(np.nanmin(sim[i])) if sim[i].size else 0.0
        ok = simulated_cms >= float(min_baseflow_fraction) * expected_cms
        rows.append(
            {
                "check": "baseflow",
                "sfincs_handoff_id": handoff_id,
                "observed_baseflow_cms": round(expected_cms, 4),
                "simulated_baseflow_cms": round(simulated_cms, 4),
                "status": "passed" if ok else "failed",
                "message": (
                    f"{statistic} ref={observed_baseflow_cfs:.1f} cfs; uparea_ratio={float(upa)/float(reference_uparea):.3f}; "
                    f"min_fraction={min_baseflow_fraction}"
                ),
            }
        )
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        [{"check": "baseflow", "status": "skipped", "message": "no handoff matched uparea map"}]
    )


def _handoff_upareas(location_root: Path, reference_gage: str, gpd):
    """Map sfincs_handoff_id -> Wflow uparea, and the reference gage's uparea."""
    uparea_by_handoff: dict[str, float] = {}
    for path in sorted(location_root.glob("data/sfincs/domains/*/base/gis/wflow_handoff_sources.geojson")):
        gdf = gpd.read_file(path)
        if "sfincs_handoff_id" not in gdf or "uparea" not in gdf:
            continue
        for _, r in gdf.iterrows():
            uparea_by_handoff[str(r["sfincs_handoff_id"])] = float(r["uparea"])
    reference_uparea = None
    for path in sorted(location_root.glob("data/wflow/domain_set_gauges/*_observation_gauges.geojson")):
        gdf = gpd.read_file(path)
        if "site_no" not in gdf or "uparea" not in gdf:
            continue
        match = gdf[gdf["site_no"].astype(str).str.zfill(8) == reference_gage.zfill(8)]
        if not match.empty:
            reference_uparea = float(match.iloc[0]["uparea"])
            break
    return uparea_by_handoff, reference_uparea


def validate_dynamic_handoff(
    event_discharge_nc,
    *,
    zero_rain_discharge_nc=None,
    expected_source_ids: set[str] | None = None,
    max_zero_peak_fraction: float | None = None,
    max_source_shape_correlation: float = 0.9999,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    event_peak = _total_peak(event_discharge_nc)
    rows.append({"check": "event_peak", "status": "passed" if event_peak > 0 else "failed", "message": f"peak_m3s={event_peak:g}"})
    if expected_source_ids is not None:
        found = set(discharge_source_ids(event_discharge_nc))
        missing = sorted(expected_source_ids - found)
        stale = sorted(found - expected_source_ids)
        status = "passed" if not missing and not stale else "failed"
        rows.append({"check": "source_ids", "status": status, "message": f"missing={missing or 'none'}; stale={stale or 'none'}"})
    if zero_rain_discharge_nc is not None:
        zero_peak = _total_peak(zero_rain_discharge_nc)
        fraction = zero_peak / event_peak if event_peak > 0 else np.inf
        threshold = "" if max_zero_peak_fraction is None else f"; diagnostic_threshold={float(max_zero_peak_fraction):g}"
        rows.append(
            {
                "check": "zero_rain_peak_fraction",
                "status": "diagnostic",
                "message": f"zero_peak_m3s={zero_peak:g}; fraction={fraction:g}{threshold}",
            }
        )
    rows.append(_source_hydrograph_shape_diversity(event_discharge_nc, max_correlation=max_source_shape_correlation))
    report = pd.DataFrame(rows)
    failed = report[report["status"].isin(["failed", "review_required"])]
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{row.check}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Dynamic Wflow handoff QA failed: {details}")
    return report


def write_dynamic_handoff_acceptance(path, *, event_id: str, discharge_nc, qa_report: pd.DataFrame, metadata: dict | None = None) -> Path:
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


def read_dynamic_handoff_acceptance(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _total_peak(discharge_nc) -> float:
    with xr.open_dataset(discharge_nc) as ds:
        if "discharge" not in ds:
            raise ValueError(f"{discharge_nc} lacks discharge variable")
        da = ds["discharge"]
        if "index" in da.dims:
            da = da.sum("index")
        return float(da.max(skipna=True))


def _source_hydrograph_shape_diversity(discharge_nc, *, max_correlation: float) -> dict:
    with xr.open_dataset(discharge_nc) as ds:
        if "discharge" not in ds:
            raise ValueError(f"{discharge_nc} lacks discharge variable")
        da = ds["discharge"]
        if "index" not in da.dims or da.sizes.get("index", 0) < 2:
            return {
                "check": "source_hydrograph_shape_diversity",
                "status": "passed",
                "message": "sources<2; diversity check skipped",
            }
        values = da.transpose("index", "time").values.astype(float)
        names = discharge_source_ids(discharge_nc)

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
    status = "failed" if duplicate_pairs else "passed"
    pairs = ", ".join(duplicate_pairs[:5]) if duplicate_pairs else "none"
    if len(duplicate_pairs) > 5:
        pairs += f", +{len(duplicate_pairs) - 5} more"
    return {
        "check": "source_hydrograph_shape_diversity",
        "status": status,
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
