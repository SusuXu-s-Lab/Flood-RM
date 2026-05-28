"""Resolve Study Location source geometry for grid generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


@dataclass(frozen=True)
class GridSourceArea:
    place_name: str
    geometry: BaseGeometry
    patch_names: tuple[str, ...]


def resolve_grid_source_area(
    config: dict[str, Any],
    *,
    geocode_place: Callable[[str], Any] | None = None,
) -> GridSourceArea:
    grid = config.get("grid") or {}
    source_area = grid.get("source_area") or {}
    if source_area.get("source") != "osmnx_place":
        raise ValueError(f"unsupported grid.source_area.source: {source_area.get('source')!r}")

    place_name = _resolve_config_reference(config, source_area.get("place_name", "project.place_name"))
    geocode = geocode_place or _geocode_place
    boundary = geocode(place_name)
    boundary_geometry = boundary.geometry.union_all()

    patch_geometries = []
    patch_names = []
    for patch in source_area.get("extra_areas") or []:
        if "bbox" not in patch:
            raise ValueError(f"source area patch needs bbox: {patch!r}")
        patch_geometries.append(box(*[float(value) for value in patch["bbox"]]))
        patch_names.append(str(patch.get("name", "unnamed_patch")))

    return GridSourceArea(
        place_name=place_name,
        geometry=unary_union([boundary_geometry, *patch_geometries]),
        patch_names=tuple(patch_names),
    )


def _resolve_config_reference(config: dict[str, Any], value: str) -> str:
    if value == "project.place_name":
        place_name = str(config.get("project", {}).get("place_name", "")).strip()
        if not place_name:
            raise ValueError("project.place_name is required for grid source ingestion")
        return place_name
    return str(value)


def _geocode_place(place_name: str):
    import osmnx as ox

    return ox.geocode_to_gdf(place_name)
