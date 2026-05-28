"""Block-review figures for switch-bounded load blocks."""

from __future__ import annotations

import io
import json
import math
import urllib.request
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image
from shapely.geometry import LineString, MultiPoint, Point, box
from shapely.ops import unary_union


OCEAN_BLUFF_BBOX = {
    "min_lon": -70.666,
    "max_lon": -70.635,
    "min_lat": 42.078,
    "max_lat": 42.105,
}
WEBSTER_SUBSTATION_BUS = "marshfield_shift_synthetic_region_044__66050127"
CARTO_LIGHT_NOLABELS_URL = "https://a.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png"
BLOCK_ALPHA = 0.20
BASEMAP_ZOOM = 15
BLOCK_PALETTE = (
    "#2563eb",
    "#16a34a",
    "#dc2626",
    "#9333ea",
    "#ca8a04",
    "#0891b2",
    "#ea580c",
    "#0f766e",
    "#c026d3",
    "#65a30d",
    "#4f46e5",
    "#d97706",
    "#0284c7",
    "#15803d",
    "#be185d",
    "#7c2d12",
    "#4338ca",
    "#84cc16",
    "#0e7490",
    "#a21caf",
    "#b45309",
    "#0369a1",
    "#166534",
    "#e11d48",
)


def build_location_block_overview(
    *,
    registry_dir: Path,
    smart_ds_compat_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Render the DFT-style full-network block hull and switch overlay."""

    artifacts = _load_block_plot_artifacts(registry_dir, smart_ds_compat_dir)
    buses = artifacts["buses"]
    lines = artifacts["lines"]
    transformers = artifacts["transformers"]
    switches = artifacts["switches"]
    facilities = artifacts["critical_facilities"]
    assignments = artifacts["critical_load_assignments"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    line_gdf = _line_gdf_dft(lines)
    bus_gdf = _point_gdf(buses)
    switch_gdf = _point_gdf(switches)
    facility_gdf = _point_gdf(facilities)
    block_line_gdf, block_root_by_bus = _derive_switch_bounded_block_lines(
        line_gdf,
        lines,
        buses,
        switches,
        transformers,
    )
    block_plot_lines = block_line_gdf.to_crs(3857)
    all_plot_lines = line_gdf.to_crs(3857)
    all_plot_buses = bus_gdf.to_crs(3857)
    plot_switches = switch_gdf.to_crs(3857) if not switch_gdf.empty else switch_gdf
    plot_critical = facility_gdf.to_crs(3857) if not facility_gdf.empty else facility_gdf

    projected_buses_by_bus = {
        row.bus: (row.geometry.x, row.geometry.y)
        for row in all_plot_buses.dropna(subset=["bus"]).itertuples(index=False)
    }
    coords_by_root: dict[str, list[tuple[float, float]]] = {}
    for bus_name, root in block_root_by_bus.items():
        pt = projected_buses_by_bus.get(bus_name)
        if pt is not None:
            coords_by_root.setdefault(root, []).append(pt)
    roots_sorted = sorted(coords_by_root, key=lambda root: (-len(coords_by_root[root]), root))

    fig, ax = plt.subplots(figsize=(11, 11))
    block_hull_records = []
    for root in roots_sorted:
        coords = coords_by_root.get(root, [])
        if len(coords) < 6:
            continue
        hull = _block_hull_polygon(coords)
        if hull is not None:
            block_hull_records.append({"root": root, "hull": hull, "area": hull.area})

    block_color_by_root = _assign_contrasting_block_colors(block_hull_records, BLOCK_PALETTE)
    block_plot_lines["block_color"] = block_plot_lines["block_root"].map(block_color_by_root)
    missing = block_plot_lines["block_color"].isna()
    block_plot_lines.loc[missing, "block_color"] = block_plot_lines.loc[missing, "block_color_index"].map(
        lambda idx: BLOCK_PALETTE[int(idx) % len(BLOCK_PALETTE)]
    )
    for record in block_hull_records:
        record["color"] = block_color_by_root.get(record["root"], BLOCK_PALETTE[0])

    covered_hulls = []
    for record in sorted(block_hull_records, key=lambda item: (item["area"], item["root"])):
        hull = record["hull"]
        display_hull = hull.difference(unary_union(covered_hulls)) if covered_hulls else hull
        display_hull = _polygonal_geometry(display_hull)
        if display_hull is None or display_hull.is_empty:
            covered_hulls.append(hull)
            continue
        display_series = gpd.GeoSeries([display_hull], crs="EPSG:3857")
        display_series.plot(ax=ax, color=record["color"], alpha=0.14, linewidth=0, zorder=1)
        display_series.boundary.plot(ax=ax, color=record["color"], alpha=0.55, linewidth=0.7, zorder=2)
        covered_hulls.append(hull)

    if not block_plot_lines.empty:
        block_plot_lines.plot(ax=ax, color=block_plot_lines["block_color"], linewidth=0.45, alpha=0.7, zorder=5)

    sectionalizing, ties, zero_length_tie_ticks = _sfo_style_switch_line_layers(plot_switches, all_plot_lines, all_plot_buses)
    if sectionalizing:
        ax.add_collection(LineCollection(sectionalizing, colors="#f97316", linewidths=1.2, alpha=0.95, zorder=6))
    if ties:
        ax.add_collection(LineCollection(ties, colors="#dc2626", linewidths=1.6, alpha=0.95, zorder=7))
    facility_segments = _critical_connector_segments_3857(plot_critical, assignments, all_plot_buses)
    if facility_segments:
        ax.add_collection(LineCollection(facility_segments, colors="#7c3aed", linewidths=0.72, alpha=0.58, linestyles=":", zorder=7))
    if not plot_critical.empty:
        plot_critical.plot(ax=ax, color="#7c3aed", marker="*", markersize=52, alpha=0.96, edgecolor="white", linewidth=0.45, zorder=8)

    ax.set_title(
        f"Grid Dataset: {len(set(block_root_by_bus.values())):.1f} switch-bounded blocks "
        "after opening Controllable Switches",
        fontsize=13,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.legend(handles=_overview_legend(), loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    block_sizes = pd.Series(block_root_by_bus).value_counts()
    return {
        "output_path": str(output_path),
        "switch_bounded_blocks": int(len(block_sizes)),
        "opened_existing_lines_for_blocks": int(switches["opens_existing_line"].fillna(False).sum()) if "opens_existing_line" in switches else 0,
        "median_block_bus_count": float(block_sizes.median()) if not block_sizes.empty else 0.0,
        "max_block_bus_count": int(block_sizes.max()) if not block_sizes.empty else 0,
        "display_block_hulls": len(block_hull_records),
        "block_line_segments": len(block_plot_lines),
        "sectionalizing_switch_segments": len(sectionalizing),
        "tie_switch_segments": len(ties),
        "zero_length_tie_ticks": zero_length_tie_ticks,
        "critical_facility_segments": len(facility_segments),
    }


def build_ocean_bluff_block_detail(
    *,
    registry_dir: Path,
    smart_ds_compat_dir: Path,
    output_path: Path,
    bbox: dict[str, float] | None = None,
    add_basemap: bool = True,
) -> dict[str, Any]:
    """Render the Ocean Bluff / Brant Rock block detail figure."""

    artifacts = _load_block_plot_artifacts(registry_dir, smart_ds_compat_dir)
    buses = artifacts["buses"]
    lines = artifacts["lines"]
    load_buses = artifacts["load_buses"]
    transformers = artifacts["transformers"]
    sources = artifacts["sources"]
    switches = artifacts["switches"]
    blocks = artifacts["blocks"]
    facilities = artifacts["critical_facilities"]
    assignments = artifacts["critical_load_assignments"]
    bbox = bbox or OCEAN_BLUFF_BBOX
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bbox_4326 = box(bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"])
    selection_bbox = gpd.GeoSeries([bbox_4326], crs="EPSG:4326").to_crs(3857).iloc[0]
    bus_gdf = _point_gdf(buses).to_crs(3857)
    load_bus_gdf = _point_gdf(load_buses).to_crs(3857)
    transformer_gdf = _point_gdf(transformers, lon_col="location_lon", lat_col="location_lat").to_crs(3857)
    source_gdf = _point_gdf(sources).to_crs(3857)
    facility_gdf = _point_gdf(facilities).to_crs(3857)
    switch_gdf = _point_gdf(switches).to_crs(3857)
    line_gdf = _line_gdf(lines).to_crs(3857)

    bus_point_by_name = bus_gdf.dropna(subset=["bus"]).set_index("bus").geometry.to_dict()
    block_records = []
    for index, row in blocks.sort_values(["feeder_id", "block_id"]).reset_index(drop=True).iterrows():
        block_buses = [bus for bus in json.loads(row.buses_json) if bus in bus_point_by_name]
        hull = _block_hull_polygon([bus_point_by_name[bus].coords[0] for bus in block_buses])
        if hull is None or not hull.intersects(selection_bbox):
            continue
        block_records.append(
            {
                "block_id": row.block_id,
                "feeder_id": row.feeder_id,
                "bus_count": int(row.bus_count),
                "load_kw": float(row.load_kw),
                "color": BLOCK_PALETTE[index % len(BLOCK_PALETTE)],
                "raw_geometry": hull,
                "buses": block_buses,
                "area": hull.area,
            }
        )
    if not block_records:
        raise RuntimeError(f"No switch-bounded load blocks intersect {bbox}.")

    selected_block_buses = {bus for record in block_records for bus in record["buses"]}
    _assign_visible_block_geometries(block_records)
    block_gdf = gpd.GeoDataFrame(
        [record for record in block_records if "geometry" in record],
        geometry="geometry",
        crs="EPSG:3857",
    )
    block_union = unary_union([record["raw_geometry"] for record in block_records])
    plot_extent = block_union.envelope

    selected_line_names = set(
        lines.loc[
            lines["from_bus"].isin(selected_block_buses) & lines["to_bus"].isin(selected_block_buses),
            "line_name",
        ].astype(str)
    )
    local_lines = line_gdf[line_gdf["line_name"].astype(str).isin(selected_line_names)].copy()
    local_buses = bus_gdf[bus_gdf["bus"].isin(selected_block_buses)].copy()
    local_demand_buses = load_bus_gdf[load_bus_gdf["bus"].isin(selected_block_buses)].copy()
    local_sources = source_gdf[source_gdf["bus"].isin(selected_block_buses)].copy()
    transformer_mask = (
        transformers["location_bus"].isin(selected_block_buses)
        | transformers["primary_bus"].isin(selected_block_buses)
        | transformers["winding_buses"].apply(lambda value: _any_csv_token_in_set(value, selected_block_buses))
    )
    local_transformers = transformer_gdf[transformer_mask.to_numpy()].copy()
    switch_mask = (
        switches["associated_line_name"].astype(str).isin(selected_line_names)
        | (switches["from_bus"].isin(selected_block_buses) & switches["to_bus"].isin(selected_block_buses))
    )
    local_switches = switch_gdf[switch_mask.to_numpy()].copy()
    sectionalizing, ties, switch_markers = _switch_segments_3857(local_switches, local_lines, local_buses, plot_extent)
    webster_sources = local_sources[local_sources["bus"].eq(WEBSTER_SUBSTATION_BUS)].copy()
    generic_sources = local_sources[~local_sources["bus"].eq(WEBSTER_SUBSTATION_BUS)].copy()
    local_facilities = _local_critical_facilities(facility_gdf, assignments, local_buses, plot_extent, selected_block_buses)
    facility_segments = _critical_connector_segments_3857(local_facilities, assignments, local_buses)

    minx, miny, maxx, maxy = plot_extent.bounds
    pad_x = (maxx - minx) * 0.075
    pad_y = (maxy - miny) * 0.075
    plot_bounds = (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)
    fig, ax = plt.subplots(figsize=_figure_size_for_bounds(plot_bounds))
    ax.set_facecolor("#eef2f3")
    basemap_added = _add_minimal_basemap(ax, plot_bounds) if add_basemap else False
    for row in block_gdf.itertuples(index=False):
        gpd.GeoSeries([row.geometry], crs="EPSG:3857").plot(
            ax=ax,
            facecolor=row.color,
            edgecolor=row.color,
            alpha=BLOCK_ALPHA,
            linewidth=0.9,
            zorder=1,
        )
    if not local_lines.empty:
        local_lines.plot(ax=ax, color="#4b5563", linewidth=0.58, alpha=0.70, zorder=3)
    if sectionalizing:
        ax.add_collection(LineCollection(sectionalizing, colors="#f97316", linewidths=2.0, alpha=0.96, zorder=5))
    if ties:
        ax.add_collection(LineCollection(ties, colors="#dc2626", linewidths=2.2, alpha=0.98, zorder=6))
    _plot_points(ax, local_buses, "#111827", "o", 1.0, 7, alpha=0.36)
    _plot_points(ax, local_demand_buses, "#2563eb", "o", 5.0, 8, alpha=0.78)
    _plot_points(ax, local_transformers, "#ca8a04", "D", 22, 10)
    _plot_points(ax, switch_markers, "#ef4444", "s", 18, 11)
    if facility_segments:
        ax.add_collection(LineCollection(facility_segments, colors="#7c3aed", linewidths=0.85, alpha=0.62, linestyles=":", zorder=12))
    _plot_points(ax, local_facilities, "#7c3aed", "*", 86, 13)
    _annotate_facilities(ax, local_facilities)
    _plot_points(ax, generic_sources, "#111827", "^", 72, 14)
    _plot_points(ax, webster_sources, "#f59e0b", "^", 135, 15, edgecolor="#111827")

    ax.set_xlim(plot_bounds[0], plot_bounds[2])
    ax.set_ylim(plot_bounds[1], plot_bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(
        "Ocean Bluff and Brant Rock\n"
        f"{len(block_gdf):,} blocks | {len(local_lines):,} lines\n"
        f"{len(local_buses):,} buses | {len(local_demand_buses):,} demand buses\n"
        f"{len(local_transformers):,} transformers | {len(switch_markers):,} switch points | "
        f"{len(local_facilities):,} critical facilities",
        fontsize=10.5,
    )
    legend = ax.legend(handles=_detail_legend(), loc="upper right", frameon=True, fontsize=8.8, ncol=1)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.88)
    legend.get_frame().set_edgecolor("#e5e7eb")
    if basemap_added:
        ax.text(
            0.01,
            0.01,
            "Basemap: OpenStreetMap contributors, CARTO no-label tiles",
            transform=ax.transAxes,
            fontsize=6.5,
            color="#6b7280",
            ha="left",
            va="bottom",
            zorder=20,
        )
    fig.tight_layout(pad=0.3)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
    return {
        "output_path": str(output_path),
        "selected_block_bus_members": len(selected_block_buses),
        "local_lines": len(local_lines),
        "local_buses": len(local_buses),
        "local_demand_buses": len(local_demand_buses),
        "local_transformers": len(local_transformers),
        "local_sources": len(local_sources),
        "local_switch_markers": len(switch_markers),
        "local_sectionalizing_switch_segments": len(sectionalizing),
        "local_tie_switch_segments": len(ties),
        "visible_block_hulls": len(block_gdf),
        "critical_facilities_visible": len(local_facilities),
        "critical_facility_proxy_segments": len(facility_segments),
        "basemap_added": basemap_added,
    }


def _load_block_plot_artifacts(registry_dir: Path, smart_ds_compat_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "buses": pd.read_csv(registry_dir / "buses.csv"),
        "lines": pd.read_csv(registry_dir / "lines.csv"),
        "load_buses": pd.read_csv(registry_dir / "load_buses.csv"),
        "transformers": pd.read_csv(registry_dir / "transformers.csv"),
        "sources": pd.read_csv(registry_dir / "sources.csv"),
        "switches": pd.read_parquet(smart_ds_compat_dir / "controllable_switches.parquet"),
        "blocks": pd.read_parquet(smart_ds_compat_dir / "switch_bounded_load_blocks.parquet"),
        "critical_facilities": _optional_parquet(smart_ds_compat_dir / "critical_facilities.parquet"),
        "critical_load_assignments": _optional_parquet(smart_ds_compat_dir / "critical_load_assignments.parquet"),
    }


def _optional_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _point_gdf(rows: pd.DataFrame, *, lon_col: str = "lon", lat_col: str = "lat") -> gpd.GeoDataFrame:
    if rows.empty or lon_col not in rows.columns or lat_col not in rows.columns:
        return gpd.GeoDataFrame(rows.copy(), geometry=[], crs="EPSG:4326")
    rows = rows.dropna(subset=[lon_col, lat_col]).copy()
    return gpd.GeoDataFrame(rows, geometry=[Point(xy) for xy in zip(rows[lon_col], rows[lat_col])], crs="EPSG:4326")


def _line_gdf(rows: pd.DataFrame) -> gpd.GeoDataFrame:
    rows = rows[rows["has_buscoords"].astype(str).str.lower().eq("true")].dropna(
        subset=["from_lon", "from_lat", "to_lon", "to_lat"]
    ).copy()
    rows["geometry"] = [
        LineString([(row.from_lon, row.from_lat), (row.to_lon, row.to_lat)])
        for row in rows.itertuples(index=False)
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _line_gdf_dft(rows: pd.DataFrame) -> gpd.GeoDataFrame:
    rows = rows[rows["has_buscoords"].astype(str).str.lower().eq("true")].dropna(
        subset=["from_lon", "from_lat", "to_lon", "to_lat"]
    ).copy()
    rows["geometry"] = [
        LineString([(row.from_lon, row.from_lat), (row.to_lon, row.to_lat)])
        for row in rows.itertuples(index=False)
    ]
    rows["line"] = rows["line_name"].astype(str)
    rows["bus1"] = rows["from_bus"].astype(str)
    rows["bus2"] = rows["to_bus"].astype(str)
    rows["enabled"] = rows.get("enabled", True)
    if not isinstance(rows["enabled"], pd.Series):
        rows["enabled"] = True
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _derive_switch_bounded_block_lines(
    line_gdf: gpd.GeoDataFrame,
    lines: pd.DataFrame,
    buses: pd.DataFrame,
    switches: pd.DataFrame,
    transformers: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, dict[str, str]]:
    parent = {str(bus): str(bus) for bus in buses["bus"].dropna().unique()}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    opened_existing_lines = set(
        switches.loc[switches["opens_existing_line"].fillna(False), "associated_line_name"].dropna()
    )
    active_lines = lines[
        lines["line_class"].fillna("line").eq("line")
        & ~lines["line_name"].isin(opened_existing_lines)
    ].copy()

    for edge in active_lines.itertuples(index=False):
        union(str(edge.from_bus), str(edge.to_bus))

    if not transformers.empty and "winding_buses" in transformers.columns:
        for row in transformers.itertuples(index=False):
            windings = [bus.strip() for bus in str(row.winding_buses).split(",") if bus.strip()]
            unique = []
            seen = set()
            for bus in windings:
                if bus in seen:
                    continue
                seen.add(bus)
                unique.append(bus)
            if len(unique) >= 2:
                hub = unique[0]
                for other in unique[1:]:
                    union(hub, other)

    block_root_by_bus = {str(bus): find(str(bus)) for bus in buses["bus"].dropna().unique()}
    block_sizes = pd.Series(block_root_by_bus).value_counts()
    root_to_block_id = {
        root: f"block_{ordinal:04d}"
        for ordinal, root in enumerate(block_sizes.index, start=1)
    }
    root_to_color_index = {
        root: ordinal - 1
        for ordinal, root in enumerate(block_sizes.index, start=1)
    }

    block_lines = line_gdf[line_gdf["enabled"] & line_gdf["line_class"].fillna("line").eq("line")].copy()
    block_lines = block_lines[~block_lines["line"].isin(opened_existing_lines)].copy()
    block_lines["block_root"] = block_lines["bus1"].map(block_root_by_bus)
    block_lines["block_id"] = block_lines["block_root"].map(root_to_block_id)
    block_lines["block_color_index"] = block_lines["block_root"].map(root_to_color_index).fillna(0).astype(int)
    block_lines["block_bus_count"] = block_lines["block_root"].map(block_sizes).fillna(1).astype(int)
    return block_lines, block_root_by_bus


def _bus_xy(buses: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        str(row.bus): (float(row.lon), float(row.lat))
        for row in buses.dropna(subset=["lon", "lat"]).itertuples(index=False)
    }


def _line_segments(lines: pd.DataFrame) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [
        ((float(row.from_lon), float(row.from_lat)), (float(row.to_lon), float(row.to_lat)))
        for row in lines.dropna(subset=["from_lon", "from_lat", "to_lon", "to_lat"]).itertuples(index=False)
    ]


def _block_hulls(blocks: pd.DataFrame, bus_xy: dict[str, tuple[float, float]]) -> list:
    hulls = []
    for row in blocks.itertuples(index=False):
        coords = [bus_xy[bus] for bus in json.loads(row.buses_json) if bus in bus_xy]
        hull = _block_hull_polygon(coords)
        if hull is not None:
            hulls.append(hull)
    return hulls


def _block_hull_polygon(coords):
    if len(coords) < 3:
        return None
    hull = MultiPoint(coords).convex_hull
    return hull if not hull.is_empty and hull.geom_type == "Polygon" else None


def _switch_line_segments(switches: pd.DataFrame, lines: pd.DataFrame, bus_xy: dict[str, tuple[float, float]]):
    line_xy = {
        str(row.line_name): ((float(row.from_lon), float(row.from_lat)), (float(row.to_lon), float(row.to_lat)))
        for row in lines.dropna(subset=["from_lon", "from_lat", "to_lon", "to_lat"]).itertuples(index=False)
    }
    sectionalizing = []
    ties = []
    for row in switches.itertuples(index=False):
        if row.switch_role == "sectionalizing" and bool(row.opens_existing_line):
            segment = line_xy.get(str(row.associated_line_name))
            if segment is not None:
                sectionalizing.append(segment)
        elif row.switch_role == "tie":
            left = bus_xy.get(str(row.from_bus))
            right = bus_xy.get(str(row.to_bus))
            if left is not None and right is not None:
                ties.append((left, right))
    return sectionalizing, ties


def _critical_facility_segments(facilities: pd.DataFrame, assignments: pd.DataFrame, bus_xy: dict[str, tuple[float, float]]):
    if facilities.empty or assignments.empty:
        return []
    assignment_by_facility = assignments.dropna(subset=["facility_id"]).set_index("facility_id")
    segments = []
    for row in facilities.dropna(subset=["lon", "lat"]).itertuples(index=False):
        if row.facility_id not in assignment_by_facility.index:
            continue
        assignment = assignment_by_facility.loc[row.facility_id]
        if isinstance(assignment, pd.DataFrame):
            assignment = assignment.iloc[0]
        bus_point = bus_xy.get(str(assignment.get("matched_bus", "")))
        if bus_point is not None:
            segments.append(((float(row.lon), float(row.lat)), bus_point))
    return segments


def _assign_visible_block_geometries(block_records: list[dict[str, Any]]) -> None:
    covered = []
    for record in sorted(block_records, key=lambda item: (item["area"], item["block_id"])):
        geometry = record["raw_geometry"].difference(unary_union(covered)) if covered else record["raw_geometry"]
        geometry = _polygonal_geometry(geometry)
        if geometry is not None and not geometry.is_empty:
            record["geometry"] = geometry
        covered.append(record["raw_geometry"])


def _polygonal_geometry(geometry):
    if geometry.is_empty:
        return None
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    if hasattr(geometry, "geoms"):
        polygons = [part for part in geometry.geoms if part.geom_type in {"Polygon", "MultiPolygon"}]
        return unary_union(polygons) if polygons else None
    return None


def _block_color_distance(left: str, right: str) -> float:
    left_rgb = mcolors.to_rgb(left)
    right_rgb = mcolors.to_rgb(right)
    return sum((left_rgb[channel] - right_rgb[channel]) ** 2 for channel in range(3)) ** 0.5


def _assign_contrasting_block_colors(
    block_hull_records: list[dict],
    palette: tuple[str, ...],
    *,
    adjacency_distance_m: float = 24.0,
) -> dict[str, str]:
    if not block_hull_records:
        return {}
    adjacency = {record["root"]: set() for record in block_hull_records}
    buffered = {
        record["root"]: record["hull"].buffer(adjacency_distance_m)
        for record in block_hull_records
    }
    for left_index, left_record in enumerate(block_hull_records):
        left_root = left_record["root"]
        for right_record in block_hull_records[left_index + 1:]:
            right_root = right_record["root"]
            if buffered[left_root].intersects(buffered[right_root]):
                adjacency[left_root].add(right_root)
                adjacency[right_root].add(left_root)

    assigned: dict[str, str] = {}
    color_use_count = {color: 0 for color in palette}
    ordering = sorted(
        block_hull_records,
        key=lambda record: (-len(adjacency[record["root"]]), -record["area"], record["root"]),
    )
    for record in ordering:
        root = record["root"]
        neighbor_colors = [
            assigned[neighbor]
            for neighbor in adjacency[root]
            if neighbor in assigned
        ]

        def score(color: str):
            if not neighbor_colors:
                return (1.0, 1.0, -color_use_count[color])
            min_neighbor_distance = min(_block_color_distance(color, other) for other in neighbor_colors)
            exact_match_penalty = 0.0 if color in neighbor_colors else 1.0
            return (exact_match_penalty, min_neighbor_distance, -color_use_count[color])

        assigned[root] = max(palette, key=score)
        color_use_count[assigned[root]] += 1
    return assigned


def _any_csv_token_in_set(value, names: set[str]) -> bool:
    if pd.isna(value):
        return False
    return any(token.strip() in names for token in str(value).split(","))


def _switch_segments_3857(switches_3857, lines_3857, buses_3857, bbox_3857):
    bus_xy = buses_3857.dropna(subset=["bus"]).set_index("bus").geometry.to_dict()
    line_xy = {str(row.line_name): row.geometry for row in lines_3857.dropna(subset=["line_name"]).itertuples(index=False)}
    sectionalizing = []
    ties = []
    marker_rows = []
    for row in switches_3857.itertuples(index=False):
        segment = None
        if row.switch_role == "sectionalizing" and bool(row.opens_existing_line):
            segment = line_xy.get(str(row.associated_line_name))
        elif row.switch_role == "tie":
            left = bus_xy.get(str(row.from_bus))
            right = bus_xy.get(str(row.to_bus))
            if left is not None and right is not None:
                segment = LineString([left, right])
        if segment is not None and segment.intersects(bbox_3857):
            target = sectionalizing if row.switch_role == "sectionalizing" else ties
            target.extend(_clipped_line_segments(segment.intersection(bbox_3857)))
        if row.geometry.within(bbox_3857):
            marker_rows.append(row._asdict())
    if marker_rows:
        return sectionalizing, ties, gpd.GeoDataFrame(marker_rows, geometry="geometry", crs="EPSG:3857")
    return sectionalizing, ties, gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")


def _sfo_style_switch_line_layers(switches_3857, lines_3857, buses_3857):
    bus_xy = buses_3857.dropna(subset=["bus"]).set_index("bus").geometry.to_dict()
    line_xy = {str(row.line): row.geometry for row in lines_3857.dropna(subset=["line"]).itertuples(index=False)}
    sectionalizing = []
    ties = []
    zero_length_tie_ticks = 0
    for row in switches_3857.itertuples(index=False):
        if row.switch_role == "sectionalizing" and bool(row.opens_existing_line):
            segment = line_xy.get(str(row.associated_line_name))
            if segment is not None:
                sectionalizing.extend(_clipped_line_segments(segment))
        elif row.switch_role == "tie":
            left = bus_xy.get(str(row.from_bus))
            right = bus_xy.get(str(row.to_bus))
            if left is None or right is None:
                continue
            if left.equals(right):
                zero_length_tie_ticks += 1
                tick = _tiny_tick_3857(left)
                ties.append(tick)
            else:
                ties.append([(left.x, left.y), (right.x, right.y)])
    return sectionalizing, ties, zero_length_tie_ticks


def _tiny_tick_3857(point: Point):
    delta = 20.0
    return [(point.x - delta, point.y - delta), (point.x + delta, point.y + delta)]


def _clipped_line_segments(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        coords = list(geometry.coords)
        return [coords[:2]] if len(coords) >= 2 else []
    if hasattr(geometry, "geoms"):
        segments = []
        for part in geometry.geoms:
            segments.extend(_clipped_line_segments(part))
        return segments
    return []


def _local_critical_facilities(facility_gdf, assignments, local_buses, plot_extent, selected_block_buses):
    if facility_gdf.empty:
        return facility_gdf
    visible_ids = set()
    if not assignments.empty and "matched_bus" in assignments.columns:
        visible_ids.update(
            assignments.loc[assignments["matched_bus"].isin(selected_block_buses), "facility_id"].astype(str)
        )
    visible_ids.update(facility_gdf[facility_gdf.geometry.within(plot_extent)]["facility_id"].astype(str))
    return facility_gdf[facility_gdf["facility_id"].astype(str).isin(visible_ids)].copy()


def _critical_connector_segments_3857(facilities_3857, assignments: pd.DataFrame, buses_3857):
    if facilities_3857.empty or assignments.empty:
        return []
    bus_xy = buses_3857.dropna(subset=["bus"]).set_index("bus").geometry.to_dict()
    assignment_by_facility = assignments.dropna(subset=["facility_id"]).set_index("facility_id")
    segments = []
    for row in facilities_3857.itertuples(index=False):
        if row.facility_id not in assignment_by_facility.index:
            continue
        assignment = assignment_by_facility.loc[row.facility_id]
        if isinstance(assignment, pd.DataFrame):
            assignment = assignment.iloc[0]
        bus_point = bus_xy.get(str(assignment.get("matched_bus", "")))
        if bus_point is not None:
            segments.append([(row.geometry.x, row.geometry.y), (bus_point.x, bus_point.y)])
    return segments


def _figure_size_for_bounds(bounds, *, height: float = 10.5):
    minx, miny, maxx, maxy = bounds
    aspect = max((maxy - miny) / max(maxx - minx, 1.0), 0.1)
    return height / aspect, height


def _add_minimal_basemap(ax, bounds, *, zoom: int = BASEMAP_ZOOM, alpha: float = 0.78) -> bool:
    x0, x1, y0, y1, tile_size_m, origin = _web_mercator_tile_range(bounds, zoom)
    cols = x1 - x0 + 1
    rows = y1 - y0 + 1
    if cols <= 0 or rows <= 0 or cols * rows > 64:
        return False
    canvas = Image.new("RGB", (cols * 256, rows * 256), "white")
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "flood-rm-grid-plot/1.0")]
    try:
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                url = CARTO_LIGHT_NOLABELS_URL.format(z=zoom, x=x, y=y)
                with opener.open(url, timeout=8) as response:
                    tile = Image.open(io.BytesIO(response.read())).convert("RGB")
                canvas.paste(tile, ((x - x0) * 256, (y - y0) * 256))
    except Exception:
        return False
    extent = (
        x0 * tile_size_m - origin,
        (x1 + 1) * tile_size_m - origin,
        origin - (y1 + 1) * tile_size_m,
        origin - y0 * tile_size_m,
    )
    ax.imshow(np.asarray(canvas), extent=extent, origin="upper", alpha=alpha, zorder=0)
    return True


def _web_mercator_tile_range(bounds, zoom: int):
    minx, miny, maxx, maxy = bounds
    origin = 20037508.342789244
    tile_size_m = (2 * origin) / (2**zoom)
    max_tile = (2**zoom) - 1
    x0 = max(0, min(max_tile, math.floor((minx + origin) / tile_size_m)))
    x1 = max(0, min(max_tile, math.floor((maxx + origin) / tile_size_m)))
    y0 = max(0, min(max_tile, math.floor((origin - maxy) / tile_size_m)))
    y1 = max(0, min(max_tile, math.floor((origin - miny) / tile_size_m)))
    return x0, x1, y0, y1, tile_size_m, origin


def _plot_points(ax, rows: gpd.GeoDataFrame, color: str, marker: str, size: float, zorder: int, *, alpha: float = 0.96, edgecolor: str = "white") -> None:
    if rows.empty:
        return
    rows.plot(ax=ax, color=color, marker=marker, markersize=size, alpha=alpha, edgecolor=edgecolor, linewidth=0.35, zorder=zorder)


def _annotate_facilities(ax, facilities: gpd.GeoDataFrame) -> None:
    offsets = {
        "Marshfield Wastewater Treatment Plant": (12, 10, "left"),
        "Fire Station #1": (-12, 12, "right"),
        "Fire Station #2": (-12, -14, "right"),
    }
    for row in facilities.itertuples(index=False):
        dx, dy, ha = offsets.get(row.facility_name, (8, 8, "left"))
        ax.annotate(
            row.facility_name,
            xy=(row.geometry.x, row.geometry.y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.2,
            color="#4c1d95",
            ha=ha,
            va="center",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "#ddd6fe", "alpha": 0.84},
            arrowprops={"arrowstyle": "-", "color": "#7c3aed", "alpha": 0.50, "lw": 0.6, "shrinkA": 1, "shrinkB": 2},
            zorder=14,
        )


def _overview_legend():
    return [
        mpatches.Patch(facecolor="#9ca3af", edgecolor="#374151", alpha=0.35, label="Switch-bounded block (convex hull)"),
        Line2D([0], [0], color="#f97316", lw=1.8, label="Sectionalizing switch (NC)"),
        Line2D([0], [0], color="#dc2626", lw=1.8, label="Tie switch (NO)"),
        Line2D([0], [0], color="#7c3aed", lw=1.0, ls=":", label="Facility-to-load-bus proxy"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#7c3aed", markeredgecolor="white", markersize=8, label="Critical facility"),
    ]


def _detail_legend():
    return [
        mpatches.Patch(facecolor="#9ca3af", edgecolor="#374151", alpha=BLOCK_ALPHA, label="Switch-bounded block hull"),
        Line2D([0], [0], color="#4b5563", lw=1.6, label="Distribution line"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#111827", alpha=0.55, markersize=4, label="Bus"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2563eb", markersize=5, label="Demand bus"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#ca8a04", markeredgecolor="white", markersize=7, label="Transformer"),
        Line2D([0], [0], color="#f97316", lw=2.0, label="Sectionalizing switch (NC)"),
        Line2D([0], [0], color="#dc2626", lw=2.2, label="Tie switch (NO)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#ef4444", markeredgecolor="white", markersize=6, label="Switch point"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#111827", markeredgecolor="white", markersize=9, label="Source"),
        Line2D([0], [0], color="#7c3aed", lw=1.0, ls=":", label="Facility-to-load-bus proxy"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#7c3aed", markeredgecolor="white", markersize=11, label="Critical facility"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#f59e0b", markeredgecolor="#111827", markersize=11, label="Eversource Webster Substation"),
    ]
