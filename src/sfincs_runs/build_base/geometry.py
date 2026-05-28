from pathlib import Path
import geopandas as gpd
from shapely.geometry import LineString, Polygon, box

def inward_buffer(gdf, d):
    # Buffer inward/outward and drop empty geometries.
    out = gdf.copy()
    out["geometry"] = out.geometry.buffer(float(d))
    return out[~out.geometry.is_empty & out.geometry.notna()].copy()

def edge_strip(gdf, edges, w):
    # Clip simple N/S/E/W strips to the model domain.
    w = abs(float(w))
    if gdf.empty or not edges or w == 0:
        return gdf.iloc[0:0].copy()
    x, y, X, Y = gdf.total_bounds
    lookup = {
        "west": (x, y, x + w, Y),
        "east": (X - w, y, X, Y),
        "south": (x, y, X, y + w),
        "north": (x, Y - w, X, Y),
    }
    strips = gpd.GeoDataFrame(geometry=[box(*lookup[str(e).lower()]) for e in edges if str(e).lower() in lookup], crs=gdf.crs)
    return gpd.overlay(strips, gdf[["geometry"]], how="intersection") if len(strips) else gdf.iloc[0:0].copy()

def coast_strip(gdf, coast, w):
    # Outer domain strip minus coastline/land polygons.
    if gdf.empty or coast.empty or float(w) == 0:
        return gdf.iloc[0:0].copy()
    domain = gdf.unary_union
    land = coast.to_crs(gdf.crs).unary_union
    ocean = domain.difference(domain.buffer(-abs(float(w)))).intersection(domain.difference(land))
    return gpd.GeoDataFrame(geometry=[ocean], crs=gdf.crs) if not ocean.is_empty else gdf.iloc[0:0].copy()

def save_gdf(gdf, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")
    return path

def write_geom(coords, crs, path, geom_type=Polygon, buf=0):
    geom = geom_type(coords)
    if buf:
        geom = geom.buffer(float(buf))
    return save_gdf(gpd.GeoDataFrame(geometry=[geom], crs=crs), path)

def write_line_buffer(coords, crs, path, buf):
    return write_geom(coords, crs, path, geom_type=LineString, buf=buf)