"""Diagnostics and review plots for Delft-FIAT event damages."""

from __future__ import annotations

from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm

M_TO_FT = 3.28084
DEFAULT_DEPTH_BINS_FT = [-999.0, -5.0, -2.0, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 999.0]
DEFAULT_DEPTH_LEVELS_FT = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0]


def event_damage(
    risk_root,
    event_id: str,
    *,
    scenario: str = "base",
    exposure_csv=None,
) -> gpd.GeoDataFrame:
    """Load one FIAT event output and optionally enrich it with exposure attributes."""
    gpkg = Path(risk_root) / scenario / event_id / "spatial.gpkg"
    gdf = gpd.read_file(gpkg)
    gdf["total_damage"] = pd.to_numeric(gdf.get("total_damage"), errors="coerce").fillna(0.0)
    if "inun_depth" in gdf:
        gdf["inun_depth"] = pd.to_numeric(gdf["inun_depth"], errors="coerce")
    if exposure_csv is not None and "object_id" in gdf:
        exposure = pd.read_csv(exposure_csv)
        keep = [
            c
            for c in (
                "object_id",
                "primary_object_type",
                "secondary_object_type",
                "max_damage_structure",
                "max_damage_content",
                "ground_elevtn",
                "ground_flht",
            )
            if c in exposure
        ]
        gdf = gdf.merge(exposure[keep], on="object_id", how="left", suffixes=("", "_ex"))
        for col in ("max_damage_structure", "max_damage_content", "ground_elevtn", "ground_flht"):
            if col in gdf:
                gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
    return gdf


def damage_summary(gdf: gpd.GeoDataFrame) -> dict:
    """Small audit record for one FIAT event damage table."""
    damaged = gdf[gdf["total_damage"] > 0].copy()
    total = float(damaged["total_damage"].sum())
    max_cols = [c for c in ("max_damage_structure", "max_damage_content") if c in damaged]
    max_total = damaged[max_cols].sum(axis=1) if max_cols else pd.Series(np.nan, index=damaged.index)
    loss_ratio = damaged["total_damage"] / max_total.replace(0, np.nan)
    low_depth_damage = (
        float(damaged.loc[damaged["inun_depth"] <= 0, "total_damage"].sum())
        if "inun_depth" in damaged
        else np.nan
    )
    return {
        "total_damage": total,
        "n_assets_damaged": int(len(damaged)),
        "median_damage": float(damaged["total_damage"].median()) if len(damaged) else 0.0,
        "p95_damage": float(damaged["total_damage"].quantile(0.95)) if len(damaged) else 0.0,
        "max_damage": float(damaged["total_damage"].max()) if len(damaged) else 0.0,
        "top10_damage_share_pct": float(100 * damaged["total_damage"].nlargest(10).sum() / total) if total else 0.0,
        "median_inun_depth_ft": float(damaged["inun_depth"].median()) if "inun_depth" in damaged and len(damaged) else np.nan,
        "p95_inun_depth_ft": float(damaged["inun_depth"].quantile(0.95)) if "inun_depth" in damaged and len(damaged) else np.nan,
        "low_depth_damage": low_depth_damage,
        "low_depth_damage_pct": float(100 * low_depth_damage / total) if total else 0.0,
        "median_loss_ratio": float(loss_ratio.median()) if len(damaged) else np.nan,
        "p95_loss_ratio": float(loss_ratio.quantile(0.95)) if len(damaged) else np.nan,
    }


def damage_by_depth(gdf: gpd.GeoDataFrame, bins=DEFAULT_DEPTH_BINS_FT) -> pd.DataFrame:
    """Damage grouped by FIAT reported inundation-depth bands."""
    damaged = gdf[gdf["total_damage"] > 0].copy()
    damaged["depth_band_ft"] = pd.cut(damaged["inun_depth"], bins=bins, right=True)
    out = (
        damaged.groupby("depth_band_ft", observed=False)
        .agg(n_assets=("total_damage", "size"), damage=("total_damage", "sum"))
        .reset_index()
    )
    total = float(out["damage"].sum())
    out["damage_pct"] = np.where(total > 0, 100 * out["damage"] / total, 0.0)
    return out


def damage_by_use(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Damage grouped by enriched NSI primary occupancy class."""
    damaged = gdf[gdf["total_damage"] > 0].copy()
    if "primary_object_type" not in damaged:
        return pd.DataFrame(columns=["primary_object_type", "n_assets", "damage", "damage_pct"])
    out = (
        damaged.groupby("primary_object_type", dropna=False)
        .agg(n_assets=("total_damage", "size"), damage=("total_damage", "sum"))
        .sort_values("damage", ascending=False)
        .reset_index()
    )
    total = float(out["damage"].sum())
    out["damage_pct"] = np.where(total > 0, 100 * out["damage"] / total, 0.0)
    return out


def top_assets(gdf: gpd.GeoDataFrame, n: int = 15) -> pd.DataFrame:
    """Top damaged assets with the columns most useful for audit."""
    cols = [
        "object_id",
        "total_damage",
        "inun_depth",
        "ground_elevtn",
        "ground_flht",
        "max_damage_structure",
        "max_damage_content",
        "primary_object_type",
        "secondary_object_type",
    ]
    keep = [c for c in cols if c in gdf]
    return gdf[gdf["total_damage"] > 0].sort_values("total_damage", ascending=False)[keep].head(n).reset_index(drop=True)


def nonzero_negative_depth_curves(vulnerability_csv) -> pd.DataFrame:
    """List vulnerability functions that allow nonzero damage at depth <= 0 ft."""
    curves = pd.read_csv(vulnerability_csv, comment="#")
    depth_col = curves.columns[0]
    curves[depth_col] = pd.to_numeric(curves[depth_col], errors="coerce")
    neg = curves[curves[depth_col] <= 0]
    rows = []
    for col in curves.columns[1:]:
        vals = pd.to_numeric(neg[col], errors="coerce").fillna(0.0)
        if vals.max() > 0:
            rows.append({"damage_function": col, "max_ratio_at_depth_le_0": float(vals.max())})
    return pd.DataFrame(rows).sort_values("max_ratio_at_depth_le_0", ascending=False).reset_index(drop=True)


def masked_sfincs_depth(
    map_path,
    *,
    huthresh_m: float = 0.1,
    land_min_elev_m: float | None = -0.5,
) -> dict:
    """Masked SFINCS flood depth in feet for review plots.

    The FIAT engine uses datum water-level rasters; this returns depth for human-readable
    maps. ``land_min_elev_m`` hides open-water bathymetry so offshore propagation does not
    dominate landward flood visuals.
    """
    with xr.open_dataset(map_path) as ds:
        zsmax = ds["zsmax"].max("timemax") if "zsmax" in ds else ds["zs"].max("time")
        zs0 = ds["zs"].isel(time=0)
        depth_m = zsmax - ds["zb"]
        flooded = (depth_m > huthresh_m) & ((zsmax - zs0) > huthresh_m)
        if land_min_elev_m is not None:
            flooded = flooded & (ds["zb"] >= land_min_elev_m)
        return {
            "x": np.asarray(ds["x"].values, dtype=float),
            "y": np.asarray(ds["y"].values, dtype=float),
            "depth_ft": np.asarray((depth_m.where(flooded) * M_TO_FT).values, dtype=float),
        }


def add_review_basemap(ax, *, crs="EPSG:32619", style: str = "dark") -> None:
    """Add a high-contrast basemap when tile access is available."""
    providers = {
        "dark": cx.providers.CartoDB.DarkMatter,
        "satellite": cx.providers.Esri.WorldImagery,
        "osm": cx.providers.OpenStreetMap.HOT,
    }
    cx.add_basemap(ax, crs=crs, source=providers.get(style, providers["dark"]), attribution_size=7)


def plot_flood_depth(
    map_path,
    *,
    ax=None,
    title: str | None = None,
    basemap_style: str = "dark",
    huthresh_m: float = 0.1,
    land_min_elev_m: float | None = -0.5,
    depth_levels_ft=DEFAULT_DEPTH_LEVELS_FT,
):
    """Plot land-focused masked flood depth over a dark/satellite basemap."""
    data = masked_sfincs_depth(map_path, huthresh_m=huthresh_m, land_min_elev_m=land_min_elev_m)
    x, y, depth = data["x"], data["y"], np.ma.masked_invalid(data["depth_ft"])
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    ax.set_ylim(float(np.nanmin(y)), float(np.nanmax(y)))
    try:
        add_review_basemap(ax, style=basemap_style)
    except Exception as exc:
        ax.text(0.01, 0.01, f"Basemap unavailable: {exc}", transform=ax.transAxes, fontsize=8)
    cmap = plt.get_cmap("Blues", len(depth_levels_ft))
    norm = BoundaryNorm(depth_levels_ft, cmap.N, extend="max")
    mesh = ax.pcolormesh(x, y, depth, shading="auto", cmap=cmap, norm=norm, alpha=0.86, zorder=2)
    ax.set_title(title or "Masked SFINCS flood depth")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax, mesh


def plot_damaged_assets(
    gdf: gpd.GeoDataFrame,
    *,
    ax=None,
    basemap_style: str = "dark",
    title: str | None = None,
):
    """Plot damaged FIAT assets with size/color scaled by total damage."""
    damaged = gdf[gdf["total_damage"] > 0].to_crs("EPSG:32619").copy()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    if len(damaged):
        minx, miny, maxx, maxy = damaged.total_bounds
        pad = max(maxx - minx, maxy - miny, 1000.0) * 0.20
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
    try:
        add_review_basemap(ax, style=basemap_style)
    except Exception as exc:
        ax.text(0.01, 0.01, f"Basemap unavailable: {exc}", transform=ax.transAxes, fontsize=8)
    if len(damaged):
        sizes = 18 + 90 * np.sqrt(damaged["total_damage"] / damaged["total_damage"].max())
        damaged.plot(
            ax=ax,
            column="total_damage",
            cmap="inferno",
            markersize=sizes,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.3,
            legend=True,
            legend_kwds={"label": "Total damage ($)"},
            zorder=3,
        )
    ax.set_title(title or "Damaged FIAT assets")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def load_exposure_buildings(model_root) -> gpd.GeoDataFrame:
    """Load HydroMT-FIAT NSI building points and enrich them with exposure attributes."""
    model_root = Path(model_root)
    buildings = gpd.read_file(model_root / "exposure" / "buildings.gpkg")
    exposure = pd.read_csv(model_root / "exposure" / "exposure.csv")
    if "object_id" in buildings and "object_id" in exposure:
        buildings["object_id"] = buildings["object_id"].astype(str)
        exposure["object_id"] = exposure["object_id"].astype(str)
        buildings = buildings.merge(exposure, on="object_id", how="left", suffixes=("", "_exposure"))
    return buildings


def exposure_summary(buildings: gpd.GeoDataFrame) -> pd.Series:
    """Human-readable FIAT exposure summary for notebook review."""
    max_structure = _numeric_column(buildings, "max_damage_structure")
    max_content = _numeric_column(buildings, "max_damage_content")
    ground = _numeric_column(buildings, "ground_elevtn")
    return pd.Series(
        {
            "n_building_assets": int(len(buildings)),
            "crs": str(buildings.crs),
            "occupancy_classes": int(buildings.get("primary_object_type", pd.Series(dtype=object)).nunique()),
            "max_potential_structure_damage": float(max_structure.sum(skipna=True)),
            "max_potential_content_damage": float(max_content.sum(skipna=True)),
            "median_ground_elev_ft": float(ground.median(skipna=True)),
            "grounded_assets": int(ground.notna().sum()),
        },
        name="fiat_building_exposure",
    )


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def building_risk_frames(
    event_assets: list[gpd.GeoDataFrame],
    outcomes: pd.DataFrame,
    *,
    exposure: pd.DataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Aggregate per-event FIAT asset damage into annualized building risk."""
    if not event_assets:
        return gpd.GeoDataFrame()
    outcome_cols = ["event_id", "design_scenario", "probability_weight", "annual_rate"]
    rates = outcomes[[c for c in outcome_cols if c in outcomes]].drop_duplicates()
    frames = []
    for gdf in event_assets:
        if "event_id" not in gdf or "design_scenario" not in gdf:
            raise ValueError("each event asset frame must include event_id and design_scenario")
        frame = gdf[["object_id", "event_id", "design_scenario", "total_damage", "inun_depth", "geometry"]].copy()
        frame["object_id"] = frame["object_id"].astype(str)
        frame["total_damage"] = pd.to_numeric(frame["total_damage"], errors="coerce").fillna(0.0)
        frame["inun_depth"] = pd.to_numeric(frame["inun_depth"], errors="coerce")
        frames.append(frame)
    all_assets = pd.concat(frames, ignore_index=True).merge(rates, on=["event_id", "design_scenario"], how="left")
    all_assets["annual_rate"] = pd.to_numeric(all_assets["annual_rate"], errors="coerce").fillna(0.0)
    all_assets["probability_weight"] = pd.to_numeric(all_assets["probability_weight"], errors="coerce").fillna(0.0)
    all_assets["damage_positive_rate"] = np.where(all_assets["total_damage"] > 0, all_assets["annual_rate"], 0.0)
    all_assets["annual_damage_component"] = all_assets["annual_rate"] * all_assets["total_damage"]
    all_assets["depth_weight_component"] = all_assets["probability_weight"] * all_assets["inun_depth"].fillna(0.0)
    all_assets["depth_weight"] = np.where(all_assets["inun_depth"].notna(), all_assets["probability_weight"], 0.0)

    grouped = (
        all_assets.groupby("object_id", as_index=False)
        .agg(
            annual_damage=("annual_damage_component", "sum"),
            damage_positive_rate=("damage_positive_rate", "sum"),
            event_damage_sum=("total_damage", "sum"),
            max_event_damage=("total_damage", "max"),
            max_inun_depth_ft=("inun_depth", "max"),
            weighted_depth_sum=("depth_weight_component", "sum"),
            depth_weight=("depth_weight", "sum"),
            events_reviewed=("event_id", "nunique"),
        )
    )
    grouped["damage_aep"] = 1.0 - np.exp(-grouped["damage_positive_rate"].to_numpy(dtype=float))
    grouped["weighted_mean_inun_depth_ft"] = np.divide(
        grouped["weighted_depth_sum"],
        grouped["depth_weight"],
        out=np.zeros(len(grouped), dtype=float),
        where=grouped["depth_weight"].to_numpy(dtype=float) > 0,
    )
    geom = gpd.GeoDataFrame(all_assets[["object_id", "geometry"]].drop_duplicates("object_id"), geometry="geometry", crs=event_assets[0].crs)
    out = geom.merge(grouped, on="object_id", how="left")
    if exposure is not None:
        exp = exposure.copy()
        exp["object_id"] = exp["object_id"].astype(str)
        keep = [
            c
            for c in [
                "object_id",
                "primary_object_type",
                "secondary_object_type",
                "max_damage_structure",
                "max_damage_content",
                "ground_elevtn",
                "ground_flht",
                "aggregation_label:Census Blockgroup",
            ]
            if c in exp
        ]
        out = out.merge(exp[keep].drop_duplicates("object_id"), on="object_id", how="left")
    return out


def building_risk(
    risk_root,
    outcomes: pd.DataFrame,
    *,
    scenario: str = "base",
    exposure_csv=None,
) -> gpd.GeoDataFrame:
    """Load per-event FIAT outputs and aggregate building-level annualized risk."""
    risk_root = Path(risk_root)
    event_assets = []
    sub = outcomes[outcomes["design_scenario"] == scenario].copy()
    for row in sub.itertuples(index=False):
        gpkg = risk_root / row.design_scenario / row.event_id / "spatial.gpkg"
        if not gpkg.exists():
            continue
        gdf = gpd.read_file(gpkg)
        gdf["event_id"] = row.event_id
        gdf["design_scenario"] = row.design_scenario
        gdf["total_damage"] = pd.to_numeric(gdf.get("total_damage"), errors="coerce").fillna(0.0)
        if "inun_depth" not in gdf:
            gdf["inun_depth"] = np.nan
        event_assets.append(gdf)
    exposure = pd.read_csv(exposure_csv) if exposure_csv is not None and Path(exposure_csv).exists() else None
    return building_risk_frames(event_assets, outcomes, exposure=exposure)


def top_neighborhoods(
    building_risk: gpd.GeoDataFrame,
    *,
    label_col: str = "aggregation_label:Census Blockgroup",
    n: int = 4,
) -> pd.DataFrame:
    """Top affected Census Blockgroups used as the default neighborhood proxy."""
    if label_col not in building_risk:
        return pd.DataFrame(columns=[label_col, "annual_damage", "damaged_buildings", "max_damage_aep"])
    frame = building_risk.copy()
    frame[label_col] = frame[label_col].fillna("unassigned").astype(str)
    out = (
        frame.groupby(label_col, dropna=False)
        .agg(
            annual_damage=("annual_damage", "sum"),
            damaged_buildings=("damage_aep", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) > 0).sum())),
            max_damage_aep=("damage_aep", "max"),
            max_event_damage=("max_event_damage", "max"),
        )
        .sort_values(["annual_damage", "damaged_buildings"], ascending=False)
        .head(int(n))
        .reset_index()
    )
    return out


def plot_building_exposure(buildings: gpd.GeoDataFrame, *, ax=None, basemap_style: str = "osm", title: str | None = None):
    """Map NSI building exposure points from the HydroMT-FIAT model."""
    gdf = buildings.to_crs("EPSG:32619")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    if len(gdf):
        minx, miny, maxx, maxy = gdf.total_bounds
        pad = max(maxx - minx, maxy - miny, 1000.0) * 0.06
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
    try:
        add_review_basemap(ax, style=basemap_style)
    except Exception as exc:
        ax.text(0.01, 0.01, f"Basemap unavailable: {exc}", transform=ax.transAxes, fontsize=8)
    if len(gdf):
        gdf.plot(ax=ax, markersize=2, color="#2b8cbe", alpha=0.45, linewidth=0, zorder=3)
    ax.set_title(title or "HydroMT-FIAT NSI building exposure")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_risk(
    building_risk: gpd.GeoDataFrame,
    *,
    metric: str = "annual_damage",
    ax=None,
    basemap_style: str = "dark",
    title: str | None = None,
    neighborhood=None,
    neighborhood_col: str = "aggregation_label:Census Blockgroup",
):
    """Map annualized FIAT building risk with optional Census Blockgroup zoom."""
    gdf = building_risk.copy()
    if neighborhood is not None and neighborhood_col in gdf:
        gdf = gdf[gdf[neighborhood_col].astype(str) == str(neighborhood)]
    gdf = gdf.to_crs("EPSG:32619")
    positive = gdf[pd.to_numeric(gdf.get(metric), errors="coerce").fillna(0.0) > 0].copy()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    bounds_source = positive if len(positive) else gdf
    if len(bounds_source):
        minx, miny, maxx, maxy = bounds_source.total_bounds
        pad = max(maxx - minx, maxy - miny, 600.0) * 0.25
        ax.set_xlim(minx - pad, maxx + pad)
        ax.set_ylim(miny - pad, maxy + pad)
    try:
        add_review_basemap(ax, style=basemap_style)
    except Exception as exc:
        ax.text(0.01, 0.01, f"Basemap unavailable: {exc}", transform=ax.transAxes, fontsize=8)
    if len(gdf):
        gdf.plot(ax=ax, markersize=1.5, color="white", alpha=0.12, linewidth=0, zorder=2)
    if len(positive):
        values = pd.to_numeric(positive[metric], errors="coerce").fillna(0.0)
        sizes = 12 + 85 * np.sqrt(values / values.max()) if values.max() > 0 else 12
        positive.plot(
            ax=ax,
            column=metric,
            cmap="inferno",
            markersize=sizes,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.25,
            legend=True,
            legend_kwds={"label": metric.replace("_", " ")},
            zorder=4,
        )
    ax.set_title(title or f"Building risk: {metric}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax
