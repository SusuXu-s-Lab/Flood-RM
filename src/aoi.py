import json
import pandas as pd
import geopandas as gpd
import shapely
from pathlib import Path
from shapely.geometry import MultiPoint

def _iter_buscoords(d):
    for p in Path(d).rglob("Buscoords.dss"):
        for line in p.read_text(errors="ignore").splitlines():
            if line.strip() and not line.startswith(("!", "//")):
                parts = line.split()
                if len(parts) >= 3: yield (float(parts[1]), float(parts[2]))

def _iter_asset_csvs(d):
    for p in Path(d).glob("*.csv"):
        df = pd.read_csv(p)
        cols = [c for c in df.columns if 'lon' in c.lower() or 'lat' in c.lower()]
        if len(cols) >= 2: yield from df[cols[:2]].dropna().values

def build_study_area(config, repo_root):
    name = config["project"]["name"]
    aoi = config.get("aoi", {})
    fmt = aoi.get("source_format", "asset_registry")
    src = Path(repo_root) / "locations" / name / aoi.get("source", "")
    out = Path(repo_root) / "locations" / name / aoi.get("output", "data/static/aoi/study_area.geojson")

    if fmt == "asset_registry":
        pts = list(_iter_asset_csvs(src))
        geom = shapely.concave_hull(MultiPoint(pts), ratio=float(aoi.get("alpha_ratio", 0.3)))
    elif fmt == "smart_ds_buscoords":
        pts = list(_iter_buscoords(src))
        geom = shapely.concave_hull(MultiPoint(pts), ratio=float(aoi.get("alpha_ratio", 0.05)))
    else:
        geom = gpd.read_file(src).geometry.unary_union
        pts = []

    out.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").to_file(out, driver="GeoJSON")

    meta = {
        "location_name": name, "source_format": fmt, "n_points": len(pts), "bounds": list(geom.bounds),
        "subregion_count": len(geom.geoms) if geom.geom_type == "MultiPolygon" else 1
    }
    Path(aoi.get("metadata_output", out.with_suffix('.json'))).write_text(json.dumps(meta, indent=2))
    return meta

def study_area_bbox(config, repo_root, buffer_degrees=0.0):
    name = config["project"]["name"]
    src = config.get("grid_footprint", {}).get("source") or config.get("aoi", {}).get("output", "data/static/aoi/study_area.geojson")
    bounds = gpd.read_file(Path(repo_root) / "locations" / name / src).total_bounds
    b = float(buffer_degrees)
    return (bounds[0]-b, bounds[1]-b, bounds[2]+b, bounds[3]+b)