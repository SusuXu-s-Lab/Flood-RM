"""Baseline Network source-input adapters.

This module owns the geospatial inputs that feed SHIFT generation: the Study
Location source area, reviewed source anchors, and OSM building parcels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import osmnx as ox
from geopandas import GeoDataFrame
from osmnx._errors import InsufficientResponseError
from shapely.geometry import MultiPolygon, Point, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, unary_union
from shift.parcel import parcels_from_geodataframe


@dataclass(frozen=True)
class GridSourceArea:
    place_name: str
    geometry: BaseGeometry
    patch_names: tuple[str, ...]


@dataclass(frozen=True)
class SourceAnchorReviewRequired(RuntimeError):
    candidate_path: Path
    reviewed_path: Path

    def __str__(self) -> str:
        return (
            "review grid source anchors before building the Baseline Network: "
            f"copy reviewed candidates from {self.candidate_path} to {self.reviewed_path}"
        )


def resolve_grid_source_area(
    config: dict[str, Any],
    *,
    geocode_place: Callable[[str], Any] | None = None,
) -> GridSourceArea:
    grid = config.get("grid") or {}
    source_area = grid.get("source_area") or {}
    if source_area.get("source") != "osmnx_place":
        raise ValueError(f"unsupported grid.source_area.source: {source_area.get('source')!r}")

    place_name = _resolve_config_reference(
        config,
        source_area.get("place_name", "project.place_name"),
    )
    geocode = geocode_place or ox.geocode_to_gdf
    boundary_geometry = geocode(place_name).geometry.union_all()

    patches = []
    patch_names = []
    for patch in source_area.get("extra_areas") or []:
        if "bbox" not in patch:
            raise ValueError(f"source area patch needs bbox: {patch!r}")
        patches.append(box(*[float(value) for value in patch["bbox"]]))
        patch_names.append(str(patch.get("name", "unnamed_patch")))

    return GridSourceArea(
        place_name=place_name,
        geometry=unary_union([boundary_geometry, *patches]),
        patch_names=tuple(patch_names),
    )


def resolve_grid_source_anchors(
    config: dict[str, Any],
    *,
    location_root: Path,
    source_area_geometry: BaseGeometry,
    fetch_candidates: Callable[[BaseGeometry], list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    source_anchors = (config.get("grid") or {}).get("source_anchors") or {}
    reviewed_path = _location_path(location_root, source_anchors.get("reviewed_override"))
    candidate_path = _location_path(location_root, source_anchors.get("candidate_output"))

    if reviewed_path.exists():
        return _read_anchor_geojson(reviewed_path)

    fetch = fetch_candidates or _fetch_osm_power_substations
    candidates = _snap_anchors_to_source_area(fetch(source_area_geometry), source_area_geometry)
    _write_anchor_geojson(candidate_path, candidates)
    if source_anchors.get("accept_unreviewed_source_anchors"):
        return candidates
    raise SourceAnchorReviewRequired(candidate_path=candidate_path, reviewed_path=reviewed_path)


def fetch_building_parcels_in_geometry(geometry: BaseGeometry):
    """Fetch OSM building parcels while preserving Shapely lon/lat order."""
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


def _resolve_config_reference(config: dict[str, Any], value: str) -> str:
    if value == "project.place_name":
        place_name = str(config.get("project", {}).get("place_name", "")).strip()
        if not place_name:
            raise ValueError("project.place_name is required for grid source ingestion")
        return place_name
    return str(value)


def _location_path(location_root: Path, value: str | None) -> Path:
    if not value:
        raise ValueError("source anchor path is required")
    path = Path(value)
    return path if path.is_absolute() else Path(location_root) / path


def _snap_anchors_to_source_area(
    anchors: list[dict[str, Any]],
    source_area_geometry: BaseGeometry,
) -> list[dict[str, Any]]:
    snapped = []
    for index, row in enumerate(anchors, start=1):
        point = Point(float(row["lon"]), float(row["lat"]))
        if not (source_area_geometry.contains(point) or source_area_geometry.touches(point)):
            point = nearest_points(source_area_geometry, point)[0]
        snapped.append(
            {
                **row,
                "name": str(row.get("name") or f"source_anchor_{index}"),
                "substation_id": str(row.get("substation_id") or f"source_anchor_{index}"),
                "lon": float(point.x),
                "lat": float(point.y),
            }
        )
    return snapped


def _write_anchor_geojson(path: Path, anchors: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = [
        {
            "type": "Feature",
            "geometry": mapping(Point(float(row["lon"]), float(row["lat"]))),
            "properties": {key: value for key, value in row.items() if key not in {"lon", "lat"}},
        }
        for row in anchors
    ]
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_anchor_geojson(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchors = []
    for index, feature in enumerate(payload.get("features", []), start=1):
        point = shape(feature["geometry"])
        props = dict(feature.get("properties") or {})
        anchors.append(
            {
                **props,
                "name": str(props.get("name") or f"source_anchor_{index}"),
                "substation_id": str(props.get("substation_id") or f"source_anchor_{index}"),
                "lon": float(point.x),
                "lat": float(point.y),
            }
        )
    return anchors


def _fetch_osm_power_substations(source_area_geometry: BaseGeometry) -> list[dict[str, Any]]:
    anchors = []
    frame = ox.features_from_polygon(source_area_geometry, {"power": "substation"})
    for index, row in enumerate(frame.itertuples(), start=1):
        geometry = row.geometry
        point = geometry if geometry.geom_type == "Point" else geometry.representative_point()
        osm_id = getattr(row, "osmid", None) or getattr(row, "osm_id", None) or index
        anchors.append(
            {
                "name": str(getattr(row, "name", "") or f"OSM substation {osm_id}"),
                "substation_id": f"osm:{osm_id}",
                "lon": float(point.x),
                "lat": float(point.y),
                "source": "osm_power_substations",
            }
        )
    return anchors


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


source_area = resolve_grid_source_area
source_anchors = resolve_grid_source_anchors
fetch_parcels = fetch_building_parcels_in_geometry
