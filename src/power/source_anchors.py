"""Resolve reviewed grid source anchors for SHIFT generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from shapely.geometry import Point, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points


@dataclass(frozen=True)
class SourceAnchorReviewRequired(RuntimeError):
    candidate_path: Path
    reviewed_path: Path

    def __str__(self) -> str:
        return (
            "review grid source anchors before building the Baseline Network: "
            f"copy reviewed candidates from {self.candidate_path} to {self.reviewed_path}"
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


def _location_path(location_root: Path, value: str | None) -> Path:
    if not value:
        raise ValueError("source anchor path is required")
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(location_root) / path


def _snap_anchors_to_source_area(
    anchors: list[dict[str, Any]],
    source_area_geometry: BaseGeometry,
) -> list[dict[str, Any]]:
    snapped = []
    for index, row in enumerate(anchors, start=1):
        lon = float(row["lon"])
        lat = float(row["lat"])
        point = Point(lon, lat)
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
    features = []
    for row in anchors:
        properties = {key: value for key, value in row.items() if key not in {"lon", "lat"}}
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(Point(float(row["lon"]), float(row["lat"]))),
                "properties": properties,
            }
        )
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
    import osmnx as ox

    frame = ox.features_from_polygon(source_area_geometry, {"power": "substation"})
    anchors = []
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
