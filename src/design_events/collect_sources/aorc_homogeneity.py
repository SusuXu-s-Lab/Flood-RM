from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


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
    counties = gpd.read_file(counties_path).to_crs(5070)
    target = gpd.read_file(target_geometry_path).to_crs(5070).geometry.union_all().centroid
    counties = counties.assign(distance_km=counties.geometry.centroid.distance(target) / 1000.0)
    selected = counties.sort_values("distance_km").head(max_candidates).copy()
    rows = [
        {
            "sample_id": "target",
            "sample_role": "target",
            "distance_km": 0.0,
            "geometry": target,
        }
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
    import xarray as xr

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
