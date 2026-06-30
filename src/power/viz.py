from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _point_gdf(df: pd.DataFrame, x: str = "lon", y: str = "lat"):
    import geopandas as gpd

    if df.empty or x not in df or y not in df:
        return gpd.GeoDataFrame(df.copy(), geometry=[], crs=4326)
    rows = df.dropna(subset=[x, y]).copy()
    return gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy(rows[x], rows[y]), crs=4326)


def _line_gdf(lines: pd.DataFrame):
    import geopandas as gpd
    from shapely.geometry import LineString

    required = ["from_lon", "from_lat", "to_lon", "to_lat"]
    if lines.empty or any(c not in lines for c in required):
        return gpd.GeoDataFrame(lines.copy(), geometry=[], crs=4326)
    rows = lines.dropna(subset=required).copy()
    if "has_buscoords" in rows:
        rows = rows[rows["has_buscoords"].astype(str).str.lower().eq("true")]
    rows["geometry"] = [LineString([(r.from_lon, r.from_lat), (r.to_lon, r.to_lat)]) for r in rows.itertuples()]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)


def plot_switches(*, registry_dir: str | Path, augmented_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Draw lines, buses, and controllable switches for stakeholder review."""

    import matplotlib.pyplot as plt

    registry = Path(registry_dir)
    augmented = Path(augmented_dir)
    output = Path(output_path)
    buses = _point_gdf(pd.read_csv(registry / "buses.csv")).to_crs(3857)
    lines = _line_gdf(pd.read_csv(registry / "lines.csv")).to_crs(3857)
    switches = _point_gdf(pd.read_parquet(augmented / "controllable_switches.parquet")).to_crs(3857)
    fig, ax = plt.subplots(figsize=(10, 10))
    if not lines.empty:
        lines.plot(ax=ax, linewidth=0.35, alpha=0.45)
    if not buses.empty:
        buses.plot(ax=ax, markersize=0.5, alpha=0.30)
    if not switches.empty:
        switches.plot(ax=ax, markersize=10, alpha=0.90)
    ax.set_title(f"Distribution switch overlay: {len(lines):,} lines, {len(buses):,} buses, {len(switches):,} switches")
    ax.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {"output_path": str(output), "line_count": len(lines), "bus_count": len(buses), "switch_count": len(switches)}


def block_overview(*, registry_dir: str | Path, augmented_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Draw switch-bounded load-block envelopes from block bus membership."""

    import geopandas as gpd
    import matplotlib.pyplot as plt
    from shapely.geometry import MultiPoint

    registry = Path(registry_dir)
    augmented = Path(augmented_dir)
    output = Path(output_path)
    buses = _point_gdf(pd.read_csv(registry / "buses.csv")).to_crs(3857)
    lines = _line_gdf(pd.read_csv(registry / "lines.csv")).to_crs(3857)
    blocks = pd.read_parquet(augmented / "switch_bounded_load_blocks.parquet")
    bus_geom = buses.dropna(subset=["bus"]).set_index("bus").geometry if not buses.empty else {}
    records = []
    for row in blocks.itertuples(index=False):
        names = [name for name in json.loads(row.buses_json) if name in bus_geom.index]
        if len(names) >= 3:
            records.append({"block_id": row.block_id, "bus_count": len(names), "geometry": MultiPoint(list(bus_geom.loc[names])).convex_hull.envelope})
    block_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=3857) if records else gpd.GeoDataFrame(geometry=[], crs=3857)
    fig, ax = plt.subplots(figsize=(11, 11))
    if not block_gdf.empty:
        block_gdf.plot(ax=ax, alpha=0.25, edgecolor="black")
    if not lines.empty:
        lines.plot(ax=ax, linewidth=0.35, alpha=0.45)
    ax.set_title(f"Switch-bounded blocks: {len(blocks):,} artifact rows, {len(block_gdf):,} plotted hulls")
    ax.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {"output_path": str(output), "switch_bounded_blocks": int(len(blocks)), "display_block_hulls": int(len(block_gdf))}
