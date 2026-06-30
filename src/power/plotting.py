import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import contextily as ctx
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from shapely.geometry import LineString, MultiPoint, box

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
