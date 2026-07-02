from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from collect_sources.audit import resolve


def derive_soilsat_top(frame: pd.DataFrame, *, layer_dim="soil_layers_stag", top_layers=(0, 1)) -> pd.DataFrame:
    """Derive a bounded top-layer saturation index from NWM SOIL_M."""
    if "SOILSAT_TOP" in frame or "SOIL_M" not in frame:
        return frame
    out = frame.copy()
    if layer_dim not in out:
        out["SOILSAT_TOP"] = pd.to_numeric(out["SOIL_M"], errors="coerce").clip(0, 1)
        return out
    group = [c for c in out.columns if c not in {layer_dim, "SOIL_M"}]
    top = out[out[layer_dim].isin([int(x) for x in top_layers])]
    sat = top.groupby(group, dropna=False)["SOIL_M"].mean().clip(0, 1).reset_index(name="SOILSAT_TOP")
    return out.merge(sat, on=group, how="left")


def nwm_points(spec: dict, paths: dict) -> list[dict]:
    """Explicit points, reviewed point file, or simple footprint-derived points."""
    if spec.get("points"):
        return [dict(p) for p in spec["points"]]
    if spec.get("points_file"):
        path = resolve(paths, spec["points_file"])
        if path and path.exists():
            gdf = gpd.read_file(path).to_crs(spec.get("crs", "EPSG:4326"))
            return [{"id": str(r.get("id", i)), spec.get("x", "x"): float(r.geometry.x), spec.get("y", "y"): float(r.geometry.y)} for i, r in gdf.iterrows()]
    footprint = next((resolve(paths, p) for p in [spec.get("points_source"), "data/static/aoi/evaluation_footprint.geojson", "data/static/aoi/study_area.geojson"] if p and resolve(paths, p) and resolve(paths, p).exists()), None)
    if footprint is None:
        return []
    gdf = gpd.read_file(footprint)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    geom = gdf.to_crs(spec.get("crs", "EPSG:4326")).geometry.union_all()
    minx, miny, maxx, maxy = geom.bounds
    samples = {
        "center": geom.centroid,
        "southwest": Point(minx, miny), "southeast": Point(maxx, miny),
        "northwest": Point(minx, maxy), "northeast": Point(maxx, maxy),
    }
    return [{"id": k, spec.get("x", "x"): float(pt.x), spec.get("y", "y"): float(pt.y)} for k, pt in samples.items()]
