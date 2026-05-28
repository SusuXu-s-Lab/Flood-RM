"""Parcel fetching helpers for the SHIFT Baseline Network workflow."""

from __future__ import annotations

import pandas as pd
import osmnx as ox
from geopandas import GeoDataFrame
from osmnx._errors import InsufficientResponseError
from shapely.geometry import MultiPolygon
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shift.parcel import parcels_from_geodataframe


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if hasattr(geometry, "convex_hull") and isinstance(geometry.convex_hull, Polygon):
        return [geometry.convex_hull]
    raise TypeError(f"parcel fetch geometry must be Polygon or MultiPolygon, got {geometry.geom_type!r}")


def fetch_building_parcels_in_geometry(geometry: BaseGeometry):
    """Fetch OSM building parcels while preserving Shapely lon/lat coordinate order.

    SHIFT's ``get_parcels_in_polygon`` reverses Shapely polygon coordinates before
    calling OSMnx. For CRS84/EPSG:4326 polygons that are already ``(lon, lat)``,
    this queries the wrong hemisphere. This adapter calls OSMnx directly and
    then uses SHIFT's public GeoDataFrame conversion.
    """
    frames = []
    tags = {"building": True}
    for polygon in _polygon_parts(geometry):
        try:
            frame = ox.features_from_polygon(polygon, tags)
        except InsufficientResponseError:
            continue
        if len(frame) > 0:
            frames.append(frame)

    if not frames:
        return []
    combined = GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    return parcels_from_geodataframe(combined)
