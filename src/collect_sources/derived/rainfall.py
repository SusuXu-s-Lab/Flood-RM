from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage
import xarray as xr

from collect_sources.audit import Artifact, covers, resolve, write_artifact
from collect_sources.derived.gridded import coord, subset, to_yx


def bbox_from_spec(paths: dict, spec: dict) -> tuple[float, float, float, float]:
    if spec.get("bbox_wgs84") is not None:
        return tuple(float(x) for x in spec["bbox_wgs84"])
    value = (spec.get("transposition_region") or {}).get("geometry_file") or spec.get("geometry_file")
    if not value:
        raise ValueError("AORC rainfall requires bbox_wgs84 or transposition_region.geometry_file")
    return tuple(float(x) for x in gpd.read_file(resolve(paths, value)).to_crs("EPSG:4326").total_bounds)


def transposition_plan(precip: xr.DataArray, footprint_gdf, region_gdf, *, stride_cells=1) -> dict | None:
    """Grid plan for moving the study footprint through an exchangeable region."""
    y, x = coord(precip, ("latitude", "lat", "y")), coord(precip, ("longitude", "lon", "x"))
    lat, lon = np.asarray(precip[y].values, float), np.asarray(precip[x].values, float)
    footprint = footprint_gdf.to_crs("EPSG:4326").geometry.union_all()
    region = region_gdf.to_crs("EPSG:4326").geometry.union_all()
    center = footprint.centroid
    row0, col0 = int(np.abs(lat - center.y).argmin()), int(np.abs(lon - center.x).argmin())
    footprint_cells = np.argwhere(_mask(footprint, lon, lat))
    if len(footprint_cells) == 0:
        footprint_cells = np.array([[row0, col0]])
    offsets = footprint_cells - np.array([[row0, col0]])
    candidate_cells = np.argwhere(_mask(region, lon, lat))[:: max(1, int(stride_cells))]
    rows, cols, center_rows, center_cols, center_lons, center_lats = [], [], [], [], [], []
    for r, c in candidate_cells:
        rr, cc = r + offsets[:, 0], c + offsets[:, 1]
        if rr.min() < 0 or cc.min() < 0 or rr.max() >= len(lat) or cc.max() >= len(lon):
            continue
        rows.append(rr); cols.append(cc); center_rows.append(r); center_cols.append(c); center_lons.append(float(lon[c])); center_lats.append(float(lat[r]))
    if not rows:
        return None
    return {
        "rows": np.vstack(rows), "cols": np.vstack(cols),
        "center_rows": np.asarray(center_rows), "center_cols": np.asarray(center_cols),
        "center_lons": np.asarray(center_lons), "center_lats": np.asarray(center_lats),
        "target_lon": float(center.x), "target_lat": float(center.y),
        "kernel": _kernel(offsets),
    }


def storm_potential(precip: xr.DataArray, *, duration_hours: int, stride_hours=1, plan: dict | None = None) -> pd.DataFrame:
    """Compute R_d(t,c): duration rainfall depth over either region mean or moving footprint."""
    da = to_yx(precip).load()
    rolled = da.rolling(time=int(duration_hours), min_periods=int(duration_hours)).sum().isel(time=slice(int(duration_hours) - 1, None, int(stride_hours)))
    times = pd.to_datetime(rolled.time.values)
    if plan is None:
        mean = rolled.mean([d for d in rolled.dims if d != "time"], skipna=True).values
        mx = rolled.max([d for d in rolled.dims if d != "time"], skipna=True).values
        mn = rolled.min([d for d in rolled.dims if d != "time"], skipna=True).values
        out = pd.DataFrame({"storm_end": times, "mean": mean, "max": mx, "min": mn, "potential_method": "region_mean"})
    else:
        rows = []
        kernel = plan["kernel"]
        for t, field in zip(times, np.asarray(rolled.values, float)):
            finite = np.isfinite(field)
            sums = ndimage.correlate(np.where(finite, field, 0.0), kernel, mode="constant", cval=0.0)
            counts = ndimage.correlate(finite.astype(float), kernel, mode="constant", cval=0.0)
            means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
            values = means[plan["center_rows"], plan["center_cols"]]
            if np.isnan(values).all():
                continue
            i = int(np.nanargmax(values))
            cell_values = field[plan["rows"][i], plan["cols"][i]]
            rows.append({
                "storm_end": t, "mean": float(values[i]), "max": float(np.nanmax(cell_values)), "min": float(np.nanmin(cell_values)),
                "historical_footprint_center_lon": float(plan["center_lons"][i]),
                "historical_footprint_center_lat": float(plan["center_lats"][i]),
                "target_footprint_center_lon": float(plan["target_lon"]),
                "target_footprint_center_lat": float(plan["target_lat"]),
                "potential_method": "moving_footprint_max_mean",
            })
        out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["storm_date"] = pd.to_datetime(out["storm_end"]) - pd.to_timedelta(int(duration_hours) - 1, unit="h")
    return out.sort_values("storm_date").reset_index(drop=True)


def decluster_pot(candidates: pd.DataFrame, *, threshold: float, tau_hours: int, top_n: int | None = None) -> pd.DataFrame:
    """Threshold-driven POT: keep independent events with R_d >= u."""
    selected, selected_times = [], []
    for _, row in candidates.loc[candidates["mean"] >= float(threshold)].sort_values("mean", ascending=False).iterrows():
        t = pd.Timestamp(row["storm_date"])
        if any(abs(t - s) < pd.Timedelta(hours=int(tau_hours)) for s in selected_times):
            continue
        selected.append(row); selected_times.append(t)
        if top_n is not None and len(selected) >= int(top_n):
            break
    ranked = pd.DataFrame(selected).reset_index(drop=True)
    if ranked.empty:
        return ranked
    ranked["por_rank"] = np.arange(1, len(ranked) + 1)
    ranked["annual_rank"] = ranked.groupby(pd.to_datetime(ranked["storm_date"]).dt.year)["mean"].rank(method="first", ascending=False).astype(int)
    return ranked


def rainfall_members(ranked: pd.DataFrame, *, duration_hours: int, location_name="study", transposition_region_id=None, output_csv=None) -> pd.DataFrame:
    out = ranked.copy()
    if out.empty:
        members = pd.DataFrame(columns=["member_id", "source", "storm_start", "storm_end", "duration_hours", "rank", "mean_precip_mm"])
    else:
        out["transposition_offset_lon"] = pd.to_numeric(out.get("target_footprint_center_lon"), errors="coerce") - pd.to_numeric(out.get("historical_footprint_center_lon"), errors="coerce")
        out["transposition_offset_lat"] = pd.to_numeric(out.get("target_footprint_center_lat"), errors="coerce") - pd.to_numeric(out.get("historical_footprint_center_lat"), errors="coerce")
        members = pd.DataFrame({
            "member_id": [f"rainfall_{location_name}_{int(duration_hours)}h_rank{int(r.por_rank):04d}" for r in out.itertuples()],
            "source": "aorc_sst",
            "storm_start": pd.to_datetime(out["storm_date"]).dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "storm_end": (pd.to_datetime(out["storm_date"]) + pd.to_timedelta(int(duration_hours) - 1, unit="h")).dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration_hours": int(duration_hours),
            "rank": out["por_rank"], "annual_rank": out["annual_rank"],
            "mean_precip_mm": out["mean"], "max_precip_mm": out["max"], "min_precip_mm": out["min"],
            "potential_method": out.get("potential_method", pd.NA),
            "historical_footprint_center_lon": out.get("historical_footprint_center_lon", pd.NA),
            "historical_footprint_center_lat": out.get("historical_footprint_center_lat", pd.NA),
            "target_footprint_center_lon": out.get("target_footprint_center_lon", pd.NA),
            "target_footprint_center_lat": out.get("target_footprint_center_lat", pd.NA),
            "transposition_offset_lon": out["transposition_offset_lon"].fillna(0),
            "transposition_offset_lat": out["transposition_offset_lat"].fillna(0),
            "transposition_region_id": transposition_region_id,
        })
    if output_csv is not None:
        output_csv = Path(output_csv); output_csv.parent.mkdir(parents=True, exist_ok=True); members.to_csv(output_csv, index=False)
    return members


def transpose(ds: xr.Dataset, member: pd.Series | dict) -> xr.Dataset:
    out = ds.copy()
    dx = float(pd.to_numeric(pd.Series([member.get("transposition_offset_lon", 0)]), errors="coerce").fillna(0).iloc[0])
    dy = float(pd.to_numeric(pd.Series([member.get("transposition_offset_lat", 0)]), errors="coerce").fillna(0).iloc[0])
    for name in ("longitude", "lon", "x"):
        if name in out.coords:
            out = out.assign_coords({name: out[name] + dx})
            break
    for name in ("latitude", "lat", "y"):
        if name in out.coords:
            out = out.assign_coords({name: out[name] + dy})
            break
    out.attrs.update(transposition_offset_lon=dx, transposition_offset_lat=dy, transposition_method="coordinate_shift_to_study_footprint")
    return out


def build_aorc_sst(settings: dict, *, skip_existing=True, opener=None) -> Artifact:
    """Science step: AORC hourly source -> POT/SST rainfall catalog."""
    from collect_sources.aorc import open_year

    paths, spec = settings["paths"], settings["spec"]
    start, end = pd.Timestamp(settings["start"]), pd.Timestamp(settings["end"])
    d = int(spec.get("storm_duration_hours", spec.get("storms", {}).get("storm_duration_hours", 72)))
    stride = int(spec.get("check_every_n_hours", spec.get("storms", {}).get("check_every_n_hours", 1)))
    threshold = float(spec.get("min_precip_threshold", spec.get("storms", {}).get("min_precip_threshold", 2.5)))
    tau = int(spec.get("decluster_hours", d))
    top_n = spec.get("top_n_events", spec.get("storms", {}).get("top_n_events"))
    top_n = None if top_n is None else int(top_n)
    root = resolve(paths, spec.get("catalog_dir", f"data/sources/aorc_sst/{int(d)}hr-events"))
    stats_csv, ranked_csv = root / "storm-stats.csv", root / "ranked-storms.csv"
    members_csv = Path(paths.get("aorc_sst_rainfall_members_csv") or root / "rainfall_members.csv")
    if skip_existing and stats_csv.exists() and ranked_csv.exists() and covers(paths, "aorc_sst", "rainfall_catalog", start, end):
        ranked = pd.read_csv(ranked_csv, parse_dates=["storm_date"])
        rainfall_members(ranked, duration_hours=d, location_name=paths.get("location_name", "study"), output_csv=members_csv)
        return Artifact("aorc_sst", "rainfall_catalog", start, end, {"storm_stats_csv": stats_csv, "ranked_storms_csv": ranked_csv, "rainfall_members_csv": members_csv}, {"reused": True})
    bbox = bbox_from_spec(paths, spec)
    plan = None
    fp = spec.get("watershed_geometry_file") or spec.get("study_area_geometry_file")
    rg = (spec.get("transposition_region") or {}).get("geometry_file")
    if fp and rg and resolve(paths, fp) and resolve(paths, rg):
        footprint, region = gpd.read_file(resolve(paths, fp)), gpd.read_file(resolve(paths, rg))
    else:
        footprint = region = None
    frames = []
    opener = opener or open_year
    for year in range(start.year, end.year + 1):
        with opener(year, spec) as ds:
            ds = subset(ds, variables=[spec.get("variable", "APCP_surface")], bbox=bbox, start=max(start, pd.Timestamp(year=year, month=1, day=1)), end=min(end, pd.Timestamp(year=year, month=12, day=31, hour=23)))
            precip = ds[spec.get("variable", "APCP_surface")]
            if plan is None and footprint is not None and region is not None:
                plan = transposition_plan(precip, footprint, region, stride_cells=spec.get("transposition_stride_cells", 1))
            frames.append(storm_potential(precip, duration_hours=d, stride_hours=stride, plan=plan))
    stats = pd.concat(frames, ignore_index=True).sort_values("storm_date") if frames else pd.DataFrame()
    ranked = decluster_pot(stats, threshold=threshold, tau_hours=tau, top_n=top_n)
    root.mkdir(parents=True, exist_ok=True)
    stats.to_csv(stats_csv, index=False); ranked.to_csv(ranked_csv, index=False)
    members = rainfall_members(ranked, duration_hours=d, location_name=paths.get("location_name", "study"), transposition_region_id=(spec.get("transposition_region") or {}).get("id"), output_csv=members_csv)
    artifact = Artifact("aorc_sst", "rainfall_catalog", start, end, {"storm_stats_csv": stats_csv, "ranked_storms_csv": ranked_csv, "rainfall_members_csv": members_csv}, {"selection": "threshold_driven_pot", "duration_hours": d, "threshold_mm": threshold, "decluster_hours": tau, "top_n_safety_cap": top_n, "ranked_rows": int(len(ranked)), "member_rows": int(len(members))})
    write_artifact(paths, artifact)
    return artifact


def _mask(geometry, lon, lat) -> np.ndarray:
    from shapely.geometry import Point
    return np.array([[geometry.covers(Point(float(x), float(y))) for x in lon] for y in lat], dtype=bool)


def _kernel(offsets: np.ndarray) -> np.ndarray:
    r, c = offsets[:, 0], offsets[:, 1]
    rr, cc = int(max(abs(r.min()), abs(r.max()))), int(max(abs(c.min()), abs(c.max())))
    k = np.zeros((2 * rr + 1, 2 * cc + 1), dtype=float)
    k[r + rr, c + cc] = 1.0
    return k


def collect(settings: dict, *, skip_existing=True):
    """Workflow adapter for the AORC SST science product."""
    return build_aorc_sst(settings, skip_existing=skip_existing)
