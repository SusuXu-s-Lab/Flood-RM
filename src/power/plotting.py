import json
import sys
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.lines import Line2D
from scipy.cluster.vq import kmeans2
from shapely.geometry import LineString, MultiPoint, box
from shapely.ops import unary_union

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if (_SOURCE_ROOT / "paths.py").exists():
    sys.path = [entry for entry in sys.path if entry != str(_SOURCE_ROOT)]
    sys.path.insert(0, str(_SOURCE_ROOT))

from paths import default_location_config_path, find_repo_root
from study_location import define_location


repo_root = find_repo_root(Path(__file__).resolve())

sectionalizing_switch_color = "#f97316"
tie_switch_color = "#dc2626"
ocean_bluff_bbox = {"min_lon": -70.666, "max_lon": -70.635, "min_lat": 42.078, "max_lat": 42.105}
webster_substation_bus = "marshfield_shift_synthetic_region_044__66050127"
block_alpha = 0.22


def power_grid_root():
    definition = define_location(default_location_config_path(repo_root))
    value = Path(definition.grid.get("power_grid_root", "data/power_grid"))
    return value if value.is_absolute() else definition.root / value


def _default_registry_dir():
    return power_grid_root() / "asset_registry"


def _default_smart_ds_compat_dir():
    return power_grid_root() / "augmented"


def _default_figure_path():
    return power_grid_root() / "figures" / "switch_line_overlay.png"


def save_review_figure(fig, output_path, *, dpi, pad=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(**({"pad": pad} if pad is not None else {}))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _line_handle(**kwargs):
    return Line2D([0], [0], **kwargs)


def _patch_handle(**kwargs):
    return mpatches.Patch(**kwargs)


def _read_table(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _optional_table(path):
    return _read_table(path) if Path(path).exists() else pd.DataFrame()


def _first_coord_pair(df, candidates):
    for x, y in candidates:
        if x in df and y in df:
            return x, y
    lon_cols = [c for c in df.columns if "lon" in c.lower()]
    lat_cols = [c for c in df.columns if "lat" in c.lower()]
    return (lon_cols[0], lat_cols[0]) if lon_cols and lat_cols else (None, None)


def _point_gdf(df, x="lon", y="lat", crs=4326):
    if df.empty or x not in df or y not in df:
        return gpd.GeoDataFrame(df.copy(), geometry=[], crs=crs)
    rows = df.dropna(subset=[x, y]).copy()
    return gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy(rows[x], rows[y]), crs=crs)


def _line_gdf(df):
    required = ["from_lon", "from_lat", "to_lon", "to_lat"]
    if df.empty or any(c not in df for c in required):
        return gpd.GeoDataFrame(df.copy(), geometry=[], crs=4326)
    rows = df.dropna(subset=required).copy()
    if "has_buscoords" in rows:
        rows = rows[rows["has_buscoords"].astype(str).str.lower().eq("true")]
    rows["geometry"] = [
        LineString([(r.from_lon, r.from_lat), (r.to_lon, r.to_lat)])
        for r in rows.itertuples()
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


def _asset_registry(registry_dir, compat_dir):
    r, c = Path(registry_dir), Path(compat_dir)
    return {
        "buses": _optional_table(r / "buses.csv"),
        "lines": _optional_table(r / "lines.csv"),
        "load_buses": _optional_table(r / "load_buses.csv"),
        "transformers": _optional_table(r / "transformers.csv"),
        "sources": _optional_table(r / "sources.csv"),
        "switches": _optional_table(c / "controllable_switches.parquet"),
        "blocks": _optional_table(c / "switch_bounded_load_blocks.parquet"),
        "facilities": _optional_table(c / "critical_facilities.parquet"),
        "assignments": _optional_table(c / "critical_load_assignments.parquet"),
    }


def _web_mercator_bbox(bbox_dict):
    b = bbox_dict
    return gpd.GeoSeries([box(b["min_lon"], b["min_lat"], b["max_lon"], b["max_lat"])], crs=4326).to_crs(3857).iloc[0]


def _add_basemap(ax):
    try:
        ctx.add_basemap(ax, crs=3857, source=ctx.providers.CartoDB.PositronNoLabels)
        return True
    except Exception:
        return False


def _clip(gdf, geom):
    return gdf.clip(geom) if not gdf.empty else gdf


def _block_hulls(blocks, buses_3857):
    if blocks.empty or buses_3857.empty or "buses_json" not in blocks:
        return gpd.GeoDataFrame(geometry=[], crs=3857)
    bus_geom = buses_3857.dropna(subset=["bus"]).set_index("bus").geometry
    records = []
    for i, row in enumerate(blocks.sort_values([c for c in ["feeder_id", "block_id"] if c in blocks]).itertuples()):
        names = [b for b in json.loads(row.buses_json) if b in bus_geom.index]
        if len(names) < 3:
            continue
        geom = MultiPoint(list(bus_geom.loc[names])).convex_hull.envelope
        if geom.is_empty:
            continue
        records.append(
            {
                "block_id": getattr(row, "block_id", f"block_{i:04d}"),
                "feeder_id": getattr(row, "feeder_id", ""),
                "bus_count": int(getattr(row, "bus_count", len(names))),
                "load_kw": float(getattr(row, "load_kw", 0.0)),
                "color_id": i,
                "bus_names": names,
                "geometry": geom,
            }
        )
    if not records:
        return gpd.GeoDataFrame(geometry=[], crs=3857)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=3857)


def _switch_lines(switches, buses_3857, lines_3857):
    if switches.empty:
        return gpd.GeoDataFrame(geometry=[], crs=3857), 0
    bus_geom = buses_3857.dropna(subset=["bus"]).set_index("bus").geometry
    line_key = "line_name" if "line_name" in lines_3857 else "line"
    line_geom = lines_3857.dropna(subset=[line_key]).set_index(line_key).geometry if line_key in lines_3857 else {}
    records, zero_length = [], 0
    for row in switches.itertuples():
        role = str(getattr(row, "switch_role", ""))
        geom = None
        if role == "sectionalizing" and bool(getattr(row, "opens_existing_line", False)):
            geom = line_geom.get(str(getattr(row, "associated_line_name", "")), None)
        elif role == "tie":
            left = bus_geom.get(str(getattr(row, "from_bus", "")), None)
            right = bus_geom.get(str(getattr(row, "to_bus", "")), None)
            if left is not None and right is not None:
                if left.equals(right):
                    zero_length += 1
                    geom = LineString([(left.x - 20, left.y - 20), (left.x + 20, left.y + 20)])
                else:
                    geom = LineString([left, right])
        if geom is not None and not geom.is_empty:
            records.append({**row._asdict(), "geometry": geom})
    return gpd.GeoDataFrame(records, geometry="geometry", crs=3857), zero_length


def _facility_connectors(facilities, assignments, buses):
    if facilities.empty or assignments.empty or "matched_bus" not in assignments:
        return gpd.GeoDataFrame(geometry=[], crs=3857)
    bus_geom = buses.dropna(subset=["bus"]).set_index("bus").geometry
    rows = assignments.merge(facilities.drop(columns="geometry"), on="facility_id", how="inner")
    lines = []
    for row in rows.itertuples():
        bus = bus_geom.get(str(row.matched_bus), None)
        fac = facilities.loc[facilities["facility_id"].astype(str).eq(str(row.facility_id)), "geometry"]
        if bus is not None and not fac.empty:
            lines.append({"facility_id": row.facility_id, "geometry": LineString([fac.iloc[0], bus])})
    return gpd.GeoDataFrame(lines, geometry="geometry", crs=3857)


def _plot_gdf(gdf, ax, **kwargs):
    if not gdf.empty:
        gdf.plot(ax=ax, **kwargs)


def plot_switches(*, registry_dir=None, smart_ds_compat_dir=None, output_path=None):
    registry_dir = Path(registry_dir or _default_registry_dir())
    smart_ds_compat_dir = Path(smart_ds_compat_dir or _default_smart_ds_compat_dir())
    output_path = Path(output_path or _default_figure_path())
    buses = _point_gdf(pd.read_csv(Path(registry_dir) / "buses.csv")).to_crs(3857)
    lines = _line_gdf(pd.read_csv(Path(registry_dir) / "lines.csv")).to_crs(3857)
    switches = _point_gdf(pd.read_parquet(Path(smart_ds_compat_dir) / "controllable_switches.parquet")).to_crs(3857)
    switch_lines, zero_length = _switch_lines(switches, buses, lines)
    role_counts = switches["switch_role"].value_counts() if "switch_role" in switches else pd.Series(dtype=int)

    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_gdf(lines, ax, color="#486b7d", linewidth=0.35, alpha=0.40)
    if not switch_lines.empty:
        switch_lines.plot(
            ax=ax,
            color=switch_lines["switch_role"].map({"sectionalizing": sectionalizing_switch_color, "tie": tie_switch_color}),
            linewidth=switch_lines["switch_role"].map({"sectionalizing": 1.2, "tie": 1.6}).fillna(1.2),
            alpha=0.95,
        )
    _plot_gdf(buses, ax, color="#1f2933", markersize=0.45, alpha=0.28)
    ax.set_title(
        "Grid Dataset: SFO-style switch line overlay\n"
        f"{len(lines):,} plotted lines, {len(buses):,} buses, {len(switches):,} controllable switches"
    )
    ax.axis("off")
    ax.legend(
        handles=[
            _line_handle(color="#486b7d", lw=1.4, label="Line"),
            _line_handle(color=sectionalizing_switch_color, lw=1.8, label="Sectionalizing switch (NC)"),
            _line_handle(color=tie_switch_color, lw=1.8, label="Tie switch (NO)"),
            _line_handle(marker="o", color="none", markerfacecolor="#1f2933", markersize=4, label="Bus"),
        ],
        loc="lower left",
        frameon=False,
    )
    save_review_figure(fig, output_path, dpi=160)
    return {
        "output_path": str(output_path),
        "line_count": len(lines),
        "bus_count": len(buses),
        "switch_count": len(switches),
        "sectionalizing_switch_count": int(role_counts.get("sectionalizing", 0)),
        "tie_switch_count": int(role_counts.get("tie", 0)),
        "sectionalizing_switch_segments_plotted": int((switch_lines["switch_role"] == "sectionalizing").sum()) if not switch_lines.empty else 0,
        "tie_switch_segments_plotted": int((switch_lines["switch_role"] == "tie").sum()) if not switch_lines.empty else 0,
        "zero_length_tie_ticks": zero_length,
    }


def block_overview(*, registry_dir, smart_ds_compat_dir, output_path):
    a = _asset_registry(registry_dir, smart_ds_compat_dir)
    buses = _point_gdf(a["buses"]).to_crs(3857)
    lines = _line_gdf(a["lines"]).to_crs(3857)
    switches = _point_gdf(a["switches"]).to_crs(3857)
    facilities = _point_gdf(a["facilities"]).to_crs(3857)
    blocks = _block_hulls(a["blocks"], buses)
    switch_lines, zero_length = _switch_lines(switches, buses, lines)
    connectors = _facility_connectors(facilities, a["assignments"], buses)

    fig, ax = plt.subplots(figsize=(11, 11))
    _plot_gdf(blocks, ax, column="color_id", cmap="tab20", alpha=0.25, edgecolor="#374151", linewidth=0.5)
    _plot_gdf(lines, ax, color="#4b5563", linewidth=0.35, alpha=0.45)
    if not switch_lines.empty:
        switch_lines.plot(
            ax=ax,
            color=switch_lines["switch_role"].map({"sectionalizing": sectionalizing_switch_color, "tie": tie_switch_color}),
            linewidth=1.2,
            alpha=0.95,
        )
    _plot_gdf(connectors, ax, color="#7c3aed", linewidth=0.7, alpha=0.6, linestyle=":")
    _plot_gdf(facilities, ax, color="#7c3aed", marker="*", markersize=52, edgecolor="white", linewidth=0.45)
    ax.set_title(f"Grid Dataset: {len(blocks):.1f} switch-bounded blocks after opening Controllable Switches", fontsize=13)
    ax.axis("off")
    ax.legend(handles=_overview_legend(), loc="lower left", frameon=False)
    save_review_figure(fig, output_path, dpi=180)
    block_sizes = blocks["bus_count"] if "bus_count" in blocks else pd.Series(dtype=float)
    return {
        "output_path": str(output_path),
        "switch_bounded_blocks": int(len(blocks)),
        "opened_existing_lines_for_blocks": int(a["switches"].get("opens_existing_line", pd.Series(dtype=bool)).fillna(False).sum()) if not a["switches"].empty else 0,
        "median_block_bus_count": float(block_sizes.median()) if not block_sizes.empty else 0.0,
        "max_block_bus_count": int(block_sizes.max()) if not block_sizes.empty else 0,
        "display_block_hulls": len(blocks),
        "block_line_segments": len(lines),
        "sectionalizing_switch_segments": int((switch_lines["switch_role"] == "sectionalizing").sum()) if not switch_lines.empty else 0,
        "tie_switch_segments": int((switch_lines["switch_role"] == "tie").sum()) if not switch_lines.empty else 0,
        "zero_length_tie_ticks": zero_length,
        "critical_facility_segments": len(connectors),
    }


def block_detail(*, registry_dir, smart_ds_compat_dir, output_path, bbox=None, add_basemap=True):
    a = _asset_registry(registry_dir, smart_ds_compat_dir)
    bbox = bbox or ocean_bluff_bbox
    clip_box = _web_mercator_bbox(bbox)
    buses = _point_gdf(a["buses"]).to_crs(3857)
    loads = _point_gdf(a["load_buses"]).to_crs(3857)
    transformers = _point_gdf(a["transformers"], "location_lon", "location_lat").to_crs(3857)
    sources = _point_gdf(a["sources"]).to_crs(3857)
    switches = _point_gdf(a["switches"]).to_crs(3857)
    facilities = _point_gdf(a["facilities"]).to_crs(3857)
    lines = _line_gdf(a["lines"]).to_crs(3857)
    blocks = _clip(_block_hulls(a["blocks"], buses), clip_box)
    if blocks.empty:
        raise RuntimeError(f"No switch-bounded load blocks intersect {bbox}.")

    selected_buses = set(np.concatenate(blocks["bus_names"].to_numpy()))
    local_buses = buses[buses["bus"].isin(selected_buses)]
    local_loads = loads[loads["bus"].isin(selected_buses)] if "bus" in loads else _clip(loads, clip_box)
    local_lines = lines[lines["from_bus"].isin(selected_buses) & lines["to_bus"].isin(selected_buses)] if {"from_bus", "to_bus"} <= set(lines.columns) else _clip(lines, clip_box)
    local_sources = sources[sources["bus"].isin(selected_buses)] if "bus" in sources else _clip(sources, clip_box)
    local_transformers = _clip(transformers, clip_box)
    local_switches = _clip(switches, clip_box)
    local_facilities = _clip(facilities, clip_box)
    connectors = _clip(_facility_connectors(local_facilities, a["assignments"], local_buses), clip_box)
    switch_lines, _zero = _switch_lines(local_switches, local_buses, local_lines)
    plot_extent = blocks.geometry.unary_union.envelope.buffer(120)
    minx, miny, maxx, maxy = plot_extent.bounds

    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set_facecolor("#eef2f3")
    basemap_added = _add_basemap(ax) if add_basemap else False
    _plot_gdf(blocks, ax, column="color_id", cmap="tab20", alpha=block_alpha, edgecolor="k", linewidth=0.8)
    _plot_gdf(local_lines, ax, color="#4b5563", linewidth=0.58, alpha=0.70)
    if not switch_lines.empty:
        switch_lines.plot(
            ax=ax,
            color=switch_lines["switch_role"].map({"sectionalizing": sectionalizing_switch_color, "tie": tie_switch_color}).fillna("#ef4444"),
            linewidth=1.9,
            alpha=0.96,
        )
    _plot_gdf(local_buses, ax, color="#111827", markersize=1.0, alpha=0.36)
    _plot_gdf(local_loads, ax, color="#2563eb", markersize=5.0, alpha=0.78)
    _plot_gdf(local_transformers, ax, color="#ca8a04", marker="D", markersize=22, edgecolor="white")
    if not local_switches.empty:
        local_switches.plot(ax=ax, color=local_switches["switch_role"].map({"sectionalizing": sectionalizing_switch_color, "tie": tie_switch_color}).fillna("#ef4444"), marker="s", markersize=18, edgecolor="white")
    _plot_gdf(connectors, ax, color="#7c3aed", linewidth=0.85, alpha=0.62, linestyle=":")
    _plot_gdf(local_facilities, ax, color="#7c3aed", marker="*", markersize=86, edgecolor="white")
    _plot_gdf(local_sources[~local_sources.get("bus", pd.Series(dtype=str)).eq(webster_substation_bus)], ax, color="#111827", marker="^", markersize=72, edgecolor="white")
    _plot_gdf(local_sources[local_sources.get("bus", pd.Series(dtype=str)).eq(webster_substation_bus)], ax, color="#f59e0b", marker="^", markersize=135, edgecolor="#111827")
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.axis("off")
    ax.set_title(
        "Ocean Bluff and Brant Rock\n"
        f"{len(blocks):,} blocks | {len(local_lines):,} lines\n"
        f"{len(local_buses):,} buses | {len(local_loads):,} demand buses\n"
        f"{len(local_transformers):,} transformers | {len(local_switches):,} switch points | {len(local_facilities):,} critical facilities",
        fontsize=10.5,
    )
    legend = ax.legend(handles=_detail_legend(), loc="upper right", frameon=True, fontsize=8.8)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.88)
    if basemap_added:
        ax.text(0.01, 0.01, "Basemap: OpenStreetMap contributors, CARTO no-label tiles", transform=ax.transAxes, fontsize=6.5, color="#6b7280")
    save_review_figure(fig, output_path, dpi=240, pad=0.3)
    return {
        "output_path": str(output_path),
        "selected_block_bus_members": len(selected_buses),
        "local_lines": len(local_lines),
        "local_buses": len(local_buses),
        "local_demand_buses": len(local_loads),
        "local_transformers": len(local_transformers),
        "local_sources": len(local_sources),
        "local_switch_markers": len(local_switches),
        "local_sectionalizing_switch_segments": int((switch_lines["switch_role"] == "sectionalizing").sum()) if not switch_lines.empty else 0,
        "local_tie_switch_segments": int((switch_lines["switch_role"] == "tie").sum()) if not switch_lines.empty else 0,
        "visible_block_hulls": len(blocks),
        "critical_facilities_visible": len(local_facilities),
        "critical_facility_proxy_segments": len(connectors),
        "basemap_added": basemap_added,
    }


def _overview_legend():
    return [
        _patch_handle(facecolor="#9ca3af", edgecolor="#374151", alpha=0.35, label="Switch-bounded block"),
        _line_handle(color=sectionalizing_switch_color, lw=1.8, label="Sectionalizing switch (NC)"),
        _line_handle(color=tie_switch_color, lw=1.8, label="Tie switch (NO)"),
        _line_handle(color="#7c3aed", lw=1.0, ls=":", label="Facility-to-load-bus proxy"),
        _line_handle(marker="*", color="none", markerfacecolor="#7c3aed", markeredgecolor="white", markersize=8, label="Critical facility"),
    ]


def _detail_legend():
    return [
        _patch_handle(facecolor="#9ca3af", edgecolor="#374151", alpha=block_alpha, label="Switch-bounded block hull"),
        _line_handle(color="#4b5563", lw=1.6, label="Distribution line"),
        _line_handle(marker="o", color="none", markerfacecolor="#111827", alpha=0.55, markersize=4, label="Bus"),
        _line_handle(marker="o", color="none", markerfacecolor="#2563eb", markersize=5, label="Demand bus"),
        _line_handle(marker="D", color="none", markerfacecolor="#ca8a04", markeredgecolor="white", markersize=7, label="Transformer"),
        _line_handle(color=sectionalizing_switch_color, lw=2.0, label="Sectionalizing switch (NC)"),
        _line_handle(color=tie_switch_color, lw=2.2, label="Tie switch (NO)"),
        _line_handle(marker="s", color="none", markerfacecolor="#ef4444", markeredgecolor="white", markersize=6, label="Switch point"),
        _line_handle(marker="^", color="none", markerfacecolor="#111827", markeredgecolor="white", markersize=9, label="Source"),
        _line_handle(color="#7c3aed", lw=1.0, ls=":", label="Facility-to-load-bus proxy"),
        _line_handle(marker="*", color="none", markerfacecolor="#7c3aed", markeredgecolor="white", markersize=11, label="Critical facility"),
        _line_handle(marker="^", color="none", markerfacecolor="#f59e0b", markeredgecolor="#111827", markersize=11, label="Eversource Webster Substation"),
    ]


def iter_buscoords(smart_ds_year_root):
    for p in Path(smart_ds_year_root).rglob("Buscoords.dss"):
        for line in p.read_text(errors="ignore").splitlines():
            if line.strip() and not line.startswith(("!", "//")):
                parts = line.split()
                if len(parts) >= 3:
                    yield float(parts[1]), float(parts[2])


def iter_asset_registry_points(asset_registry_dir):
    for p in sorted(Path(asset_registry_dir).glob("*.csv")):
        df = pd.read_csv(p)
        for x, y in [("lon", "lat"), ("location_lon", "location_lat"), ("from_lon", "from_lat"), ("to_lon", "to_lat")]:
            if x in df and y in df:
                yield from df[[x, y]].dropna().astype(float).itertuples(index=False, name=None)


def get_asset_points(registry_dir):
    return list(iter_asset_registry_points(registry_dir))


def concave_power_extent(points, *, alpha_ratio):
    pts = list(points)
    if not (0 < alpha_ratio <= 1):
        raise ValueError(f"alpha_ratio must be in (0, 1]; got {alpha_ratio!r}")
    if len(pts) < 3:
        raise ValueError(f"need at least 3 points to form a hull; got {len(pts)}")
    return MultiPoint(pts).convex_hull if alpha_ratio == 1.0 else gpd.GeoSeries([MultiPoint(pts)], crs=4326).concave_hull(ratio=alpha_ratio).iloc[0]


def write_power_extent(pts, region_id, out_path, alpha=0.05, source="asset coordinates"):
    hull = concave_power_extent(pts, alpha_ratio=alpha)
    gdf = gpd.GeoDataFrame(
        {"region_id": [region_id], "n_assets": [len(pts)], "alpha_ratio": [alpha], "source": [source]},
        geometry=[hull],
        crs=4326,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    return {
        "region_id": region_id,
        "n_assets": len(pts),
        "n_buses": len(pts),
        "alpha_ratio": alpha,
        "convex_hull_area": float(MultiPoint(pts).convex_hull.area),
        "concave_hull_area": float(hull.area),
        "output_path": str(out_path),
    }


def write_smart_ds_power_extent(*, region_id, smart_ds_year_root, output_path, alpha_ratio):
    points = list(iter_buscoords(smart_ds_year_root))
    if not points:
        raise FileNotFoundError(f"no Buscoords.dss rows found under {smart_ds_year_root}")
    return write_power_extent(points, region_id, output_path, alpha_ratio, source="smart_ds Buscoords.dss")


def write_marshfield_power_extent(*, asset_registry_dir, output_path, alpha_ratio):
    points = get_asset_points(asset_registry_dir)
    if not points:
        raise FileNotFoundError(f"no asset coordinates found under {asset_registry_dir}")
    return write_power_extent(points, "marshfield", output_path, alpha_ratio, source="marshfield asset_registry")


def _domain_record(domain_id, geom_utm, crs, n_assets):
    geom = gpd.GeoSeries([geom_utm], crs=crs).to_crs(4326).iloc[0]
    minx, miny, maxx, maxy = geom_utm.bounds
    return {
        "domain_id": domain_id,
        "polygon": geom,
        "n_assets_inside": int(n_assets),
        "n_assets": int(n_assets),
        "aabb_km_x": float((maxx - minx) / 1000),
        "aabb_km_y": float((maxy - miny) / 1000),
    }


def _points_gdf(points):
    pts = list(points)
    if len(pts) < 1:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(*zip(*pts)), crs=4326)


def cluster_buses_kmeans(points, *, region_id, k, aabb_buffer_km=1.0, seed=0):
    gdf = _points_gdf(points)
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if len(gdf) < k:
        raise ValueError(f"k={k} requested but only {len(gdf)} points provided")
    utm = gdf.to_crs(gdf.estimate_utm_crs())
    xy = np.column_stack((utm.geometry.x, utm.geometry.y))
    labels = np.zeros(len(utm), dtype=int) if k == 1 else kmeans2(xy, k, seed=seed, minit="++")[1]
    domains = []
    for i in range(k):
        cluster = utm[labels == i]
        if cluster.empty:
            continue
        domains.append(_domain_record(f"{region_id}:k{i}", cluster.geometry.unary_union.envelope.buffer(aabb_buffer_km * 1000), utm.crs, len(cluster)))
    return sorted(domains, key=lambda d: -d["n_assets_inside"])


def cluster_marshfield_by_feeder(asset_registry_dir, *, region_id="marshfield", min_n_assets=100, aabb_buffer_km=1.0):
    frames = [pd.read_csv(f) for f in Path(asset_registry_dir).glob("*.csv")]
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["feeder_id", "lon", "lat"])
    if df.empty:
        return []
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326).to_crs(gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326).estimate_utm_crs())
    domains = []
    for feeder, group in gdf.groupby("feeder_id"):
        if len(group) >= min_n_assets:
            domains.append(_domain_record(f"{region_id}:{feeder}", group.geometry.unary_union.envelope.buffer(aabb_buffer_km * 1000), gdf.crs, len(group)))
    return sorted(domains, key=lambda d: -d["n_assets_inside"])


def cluster_smart_ds_by_subregion(smart_ds_year_root, *, region_id, min_n_buses=1000, aabb_buffer_km=1.0):
    domains = []
    for subregion in sorted(p for p in Path(smart_ds_year_root).iterdir() if p.is_dir()):
        pts = list(iter_buscoords(subregion))
        if len(pts) < min_n_buses:
            continue
        gdf = _points_gdf(pts)
        utm = gdf.to_crs(gdf.estimate_utm_crs())
        domains.append(_domain_record(f"{region_id}:{subregion.name}", utm.geometry.unary_union.envelope.buffer(aabb_buffer_km * 1000), utm.crs, len(utm)))
    return sorted(domains, key=lambda d: -d["n_assets_inside"])


def cluster_to_sfincs_domains(points, *, region_id, alpha_split=0.02, min_component_area_km2=5.0, aabb_buffer_km=1.0):
    gdf = _points_gdf(points)
    if len(gdf) < 3:
        raise ValueError(f"need at least 3 points to cluster; got {len(gdf)}")
    hull = gpd.GeoSeries([MultiPoint(list(gdf.geometry))], crs=4326).concave_hull(ratio=alpha_split).iloc[0]
    components = list(hull.geoms) if hull.geom_type == "MultiPolygon" else [hull]
    domains = []
    for i, component in enumerate(sorted(components, key=lambda g: -g.area)):
        comp = gpd.GeoDataFrame(geometry=[component], crs=4326).to_crs(gdf.estimate_utm_crs())
        if comp.area.iloc[0] / 1e6 < min_component_area_km2:
            continue
        points_inside = int(gdf.geometry.within(component).sum())
        domains.append(_domain_record(f"{region_id}:{i}", comp.geometry.iloc[0].envelope.buffer(aabb_buffer_km * 1000), comp.crs, points_inside))
    return sorted(domains, key=lambda d: -d["n_assets_inside"])


def _domain_value(domain, key):
    return domain[key] if isinstance(domain, dict) else getattr(domain, key)


def _merge_domains(domains, domain_id=None, total_assets=None):
    geom = unary_union([_domain_value(d, "polygon") for d in domains]).envelope
    n_assets = sum(_domain_value(d, "n_assets_inside") for d in domains) if total_assets is None else total_assets
    if domain_id is None:
        ids = sorted(_domain_value(d, "domain_id") for d in domains)
        domain_id = f"{ids[0]} + {len(ids) - 1} more" if len(ids) > 1 else ids[0]
    center_lat = geom.centroid.y
    minx, miny, maxx, maxy = geom.bounds
    return {
        "domain_id": domain_id,
        "polygon": geom,
        "n_assets_inside": int(n_assets),
        "n_assets": int(n_assets),
        "aabb_km_x": float((maxx - minx) * 111.0 * np.cos(np.radians(center_lat))),
        "aabb_km_y": float((maxy - miny) * 111.0),
    }


def merge_overlapping_aabbs(domains, *, min_intersection_km2=10.0, max_anchor_aabb_km=80.0):
    if not domains:
        return []
    remaining = list(domains)
    merged = []
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        changed = True
        while changed:
            changed = False
            keep = []
            for candidate in remaining:
                if max_anchor_aabb_km is not None and (
                    _domain_value(candidate, "aabb_km_x") > max_anchor_aabb_km or _domain_value(candidate, "aabb_km_y") > max_anchor_aabb_km
                ):
                    keep.append(candidate)
                    continue
                if any(_domain_value(candidate, "polygon").intersection(_domain_value(d, "polygon")).area * 12321 >= min_intersection_km2 for d in group):
                    group.append(candidate)
                    changed = True
                else:
                    keep.append(candidate)
            remaining = keep
        merged.append(_merge_domains(group))
    return sorted(merged, key=lambda d: -d["n_assets_inside"])


def merge_by_centroid_proximity(domains, *, eps_km=20.0, min_assets_per_cluster=50_000):
    if not domains:
        return []
    gdf = gpd.GeoDataFrame(domains, geometry=[d["polygon"] if isinstance(d, dict) else d.polygon for d in domains], crs=4326)
    utm = gdf.to_crs(gdf.estimate_utm_crs())
    xy = np.column_stack((utm.centroid.x, utm.centroid.y))
    parent = np.arange(len(utm))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            if np.linalg.norm(xy[i] - xy[j]) < eps_km * 1000:
                parent[root(i)] = root(j)
    out = []
    for group_id in sorted(set(root(i) for i in range(len(parent)))):
        members = [domains[i] for i in range(len(parent)) if root(i) == group_id]
        total = sum(_domain_value(m, "n_assets_inside") for m in members)
        if total >= min_assets_per_cluster:
            out.append(_merge_domains(members, total_assets=total))
    return sorted(out, key=lambda d: -d["n_assets_inside"])


def write_sfincs_domains(*, region_id, points, output_path, alpha_split=0.02, min_component_area_km2=5.0, aabb_buffer_km=1.0):
    domains = cluster_to_sfincs_domains(
        points,
        region_id=region_id,
        alpha_split=alpha_split,
        min_component_area_km2=min_component_area_km2,
        aabb_buffer_km=aabb_buffer_km,
    )
    gdf = gpd.GeoDataFrame(
        [
            {
                "domain_id": d["domain_id"],
                "n_assets_inside": d["n_assets_inside"],
                "aabb_km_x": round(d["aabb_km_x"], 1),
                "aabb_km_y": round(d["aabb_km_y"], 1),
                "source": f"cluster-derived AABB at alpha_split={alpha_split}",
                "geometry": d["polygon"],
            }
            for d in domains
        ],
        crs=4326,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")
    return domains
