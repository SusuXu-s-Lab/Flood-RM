"""Compatibility skin for Baseline Network source-input adapters."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
from geopandas import GeoDataFrame
from osmnx._errors import InsufficientResponseError
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shift.parcel import parcels_from_geodataframe

from power_v2.baseline import GridSourceArea
from power_v2.baseline import SourceAnchorReviewRequired
from power_v2.baseline import source_anchors
from power_v2.baseline import source_area as _source_area


def source_area(
    config: dict[str, Any],
    *,
    geocode_place: Callable[[str], Any] | None = None,
) -> GridSourceArea:
    """Resolve the notebook source area through the clean baseline core."""

    source_area_config = ((config.get("grid") or {}).get("source_area") or {})
    for patch in source_area_config.get("extra_areas") or []:
        if "bbox" not in patch:
            raise ValueError(f"source area patch needs bbox: {patch!r}")
    return _source_area(config, geocode_place=geocode_place)


def fetch_parcels(geometry: BaseGeometry):
    """Fetch OSM building parcels while preserving Shapely lon/lat order."""

    import osmnx as ox

    frames = []
    for polygon in _polygon_parts(geometry):
        try:
            frame = ox.features_from_polygon(polygon, {"building": True})
        except InsufficientResponseError:
            continue
        if len(frame) > 0:
            frames.append(frame)

    if not frames:
        return []
    combined = GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    return parcels_from_geodataframe(combined)


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if hasattr(geometry, "convex_hull") and isinstance(geometry.convex_hull, Polygon):
        return [geometry.convex_hull]
    raise TypeError(
        "parcel fetch geometry must be Polygon or MultiPolygon, "
        f"got {geometry.geom_type!r}"
    )
