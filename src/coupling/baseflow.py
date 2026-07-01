from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

CFS_TO_CMS = 0.028316846592


def validate_baseflow_against_observed(
    config: dict,
    location_root,
    *,
    zero_rain_discharge_nc,
    streamflow_records_csv=None,
    min_baseflow_fraction: float = 0.25,
) -> pd.DataFrame:
    """Confirm warm-state spin-up left non-dry channels."""
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
        for _, row in gdf.iterrows():
            uparea_by_handoff[str(row["sfincs_handoff_id"])] = float(row["uparea"])
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
