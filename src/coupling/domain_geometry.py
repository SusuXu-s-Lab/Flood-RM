"""Coverage and stream-boundary geometry helpers for Wflow-SFINCS coupling."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point
from shapely.ops import unary_union

from paths import resolve_location_path


def coverage_domain_id(box_row, project_name: str, index: int, count: int) -> str:
    """Return the SFINCS domain id for one coverage box."""
    subregion_id = box_row.get("subregion_id") if hasattr(box_row, "get") else None
    if subregion_id is not None and not pd.isna(subregion_id) and str(subregion_id).strip():
        suffix = "".join(char.lower() if char.isalnum() else "_" for char in str(subregion_id)).strip("_")
        if suffix.startswith(f"{project_name}_"):
            return suffix
        return f"{project_name}_{suffix}"
    if count == 1:
        return f"{project_name}_main"
    if count == 2:
        return f"{project_name}_{'west' if index == 0 else 'east'}"
    return f"{project_name}_{index + 1:02d}"


def coverage_box_crossings(
    boxes: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    *,
    project_name: str,
    min_uparea_km2: float,
    uparea_col: str = "uparea",
) -> gpd.GeoDataFrame:
    """Return every inflow crossing across all coverage boxes with stable ids."""
    boxes = boxes.reset_index(drop=True)
    count = len(boxes)
    rows = []
    for index, box_row in boxes.iterrows():
        domain_id = coverage_domain_id(box_row, project_name, int(index), count)
        crossings = stream_boundary_inflow_crossings(
            rivers, box_row.geometry, min_uparea_km2=min_uparea_km2, uparea_col=uparea_col
        )
        for rank, (_, crossing) in enumerate(crossings.iterrows(), start=1):
            rows.append(
                {
                    "sfincs_domain_id": domain_id,
                    "sfincs_handoff_id": f"{domain_id}_inflow_{rank:02d}",
                    "uparea_km2": float(crossing["uparea_km2"]),
                    "geometry": crossing.geometry,
                }
            )
    return gpd.GeoDataFrame(
        rows,
        columns=["sfincs_domain_id", "sfincs_handoff_id", "uparea_km2", "geometry"],
        geometry="geometry",
        crs=getattr(rivers, "crs", None),
    )


def select_encompassing_huc(coverage_geom, huc_loader, *, levels=(8, 6, 4), allow_union=True) -> dict:
    """Return the WBD HUC domain that contains ``coverage_geom``."""
    for level in levels:
        hucs = huc_loader(level)
        if hucs is None or len(hucs) == 0:
            continue
        containing = hucs[hucs.geometry.covers(coverage_geom)]
        if not containing.empty:
            areas = containing.geometry.to_crs("EPSG:5070").area if containing.crs else containing.geometry.area
            smallest = containing.loc[areas.idxmin()]
            huc_id = str(smallest.get("huc_id", ""))
            huc_ids = [part for part in huc_id.split("_") if part]
            kind = str(smallest.get("huc_kind", ""))
            if kind not in {"single", "union"}:
                kind = "union" if len(huc_ids) > 1 else "single"
            return {
                "level": int(level),
                "kind": kind,
                "huc_id": huc_id,
                "huc_ids": huc_ids or [huc_id],
                "geometry": smallest.geometry,
            }

    if allow_union:
        for level in levels:
            hucs = huc_loader(level)
            if hucs is None or len(hucs) == 0:
                continue
            intersecting = hucs[hucs.geometry.intersects(coverage_geom)]
            if intersecting.empty:
                continue
            union = intersecting.geometry.union_all()
            if union.covers(coverage_geom):
                huc_ids = [str(value) for value in intersecting.get("huc_id", [])]
                return {
                    "level": int(level),
                    "kind": "union",
                    "huc_id": "_".join(huc_ids),
                    "huc_ids": huc_ids,
                    "geometry": union,
                }

    raise ValueError(
        "No WBD HUC (single or union) at levels "
        f"{tuple(levels)} contains the coverage area; widen the levels (e.g. add 2)."
    )


def sfincs_coverage_boxes(config, location_root: Path) -> tuple[Path, gpd.GeoDataFrame]:
    """Return the SFINCS hydraulic coverage boxes that Wflow must enclose."""
    domain_set = config.get("sfincs_domain_set", {})
    source_value = domain_set.get(
        "source",
        config.get("static_sources", {}).get("bbox", {}).get("output", "data/static/aoi/bbox.geojson"),
    )
    coverage_path = resolve_location_path(location_root, source_value)
    if not coverage_path.exists():
        return coverage_path, gpd.GeoDataFrame(columns=["subregion_id", "geometry"], geometry="geometry", crs="EPSG:4326")

    source = gpd.read_file(coverage_path).to_crs("EPSG:4326")
    records = _coverage_component_records(source)
    if domain_set.get("allow_multiple_domains") is False and records:
        records = [{"source_subregion_id": None, "geometry": unary_union([record["geometry"] for record in records])}]

    project_name = str(config.get("project", {}).get("name", location_root.name))
    region_geometry = str(domain_set.get("region_geometry", "component")).lower()
    rows = []
    count = len(records)
    for index, record in enumerate(records):
        domain_id = coverage_domain_id({"subregion_id": record.get("source_subregion_id")}, project_name, index, count)
        geometry = record["geometry"]
        if region_geometry in {"bbox", "bounding_box", "envelope"}:
            geometry = geometry.envelope
        rows.append(
            {
                "name": "sfincs_coverage_bbox",
                "subregion_id": domain_id,
                "source_subregion_id": record.get("source_subregion_id"),
                "component_index": index,
                "geometry": geometry,
            }
        )
    include_domain_ids = _included_sfincs_domain_ids(domain_set)
    if include_domain_ids:
        rows = [row for row in rows if row["subregion_id"] in include_domain_ids]
    return coverage_path, gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def stream_boundary_inflow_crossings(
    rivers: gpd.GeoDataFrame,
    coverage_bbox,
    *,
    min_uparea_km2: float,
    uparea_col: str = "uparea",
) -> gpd.GeoDataFrame:
    """Return stream/coverage-box boundary crossings ranked by drainage area."""
    boundary = coverage_bbox.boundary
    rows = []
    for river_index, row in rivers.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        uparea = float(row[uparea_col])
        if uparea < float(min_uparea_km2):
            continue
        if not _flows_into(geometry, coverage_bbox):
            continue
        for point in _intersection_points(geometry.intersection(boundary)):
            rows.append({"river_index": int(river_index), "uparea_km2": uparea, "geometry": point})

    crossings = gpd.GeoDataFrame(
        rows,
        columns=["river_index", "uparea_km2", "geometry"],
        geometry="geometry",
        crs=rivers.crs,
    )
    return crossings.sort_values("uparea_km2", ascending=False).reset_index(drop=True)


def load_crossing_rivers(config, location_root: Path) -> gpd.GeoDataFrame:
    """Load stream geometry with a normalized ``uparea`` km2 column for crossing planning."""
    rivers_path = resolve_location_path(
        location_root,
        config.get("collection", {})
        .get("national_hydrography", {})
        .get("river_geometry", "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"),
    )
    if not rivers_path.exists():
        raise FileNotFoundError(f"Stream network for crossings is missing: {rivers_path}")
    rivers = gpd.read_file(rivers_path).to_crs("EPSG:4326")
    if rivers.empty:
        raise FileNotFoundError(f"Stream network for crossings has no features: {rivers_path}")
    uparea_col = _find_column(rivers, ("uparea", "uparea_km2", "totdasqkm"))
    if uparea_col is None:
        raise FileNotFoundError(f"Stream network has no drainage-area column (uparea/TotDASqKm): {rivers_path}")
    if uparea_col != "uparea":
        rivers = rivers.copy()
        rivers["uparea"] = pd.to_numeric(rivers[uparea_col], errors="coerce")
    return rivers


def coverage_domain_geometries(boxes: gpd.GeoDataFrame, project_name: str) -> dict[str, object]:
    count = len(boxes)
    return {
        coverage_domain_id(box_row, project_name, int(index), count): box_row.geometry
        for index, box_row in boxes.reset_index(drop=True).iterrows()
        if box_row.geometry is not None and not box_row.geometry.is_empty
    }


def subbasin_submodels_from_crossings(
    crossings: gpd.GeoDataFrame,
    *,
    project_name: str,
    sfincs_domain_id: str,
) -> list[dict]:
    """Turn ranked inflow crossings into Wflow subbasin submodel descriptors."""
    submodels = []
    for rank, (_, crossing) in enumerate(crossings.iterrows(), start=1):
        submodel_id = f"{sfincs_domain_id}_inflow_{rank:02d}"
        outlet_xy = [float(crossing.geometry.x), float(crossing.geometry.y)]
        submodels.append(
            {
                "wflow_submodel_id": submodel_id,
                "region_kind": "subbasin",
                "region": {"subbasin": outlet_xy, "uparea": float(crossing["uparea_km2"])},
                "outlet_region": {"subbasin": outlet_xy},
                "subbasin_geometry": None,
                "sfincs_domain_ids": [sfincs_domain_id],
                "sfincs_handoff_ids": [submodel_id],
                "gauge_site_nos": [],
                "frequency_basis": [],
                "role_counts": {},
            }
        )
    return submodels


def _coverage_component_records(source: gpd.GeoDataFrame) -> list[dict]:
    if source.empty:
        return []
    if "subregion_id" in source.columns:
        records = []
        for _, row in source.iterrows():
            geometry = row.geometry
            if geometry is None or geometry.is_empty:
                continue
            subregion_id = row.get("subregion_id")
            subregion_id = None if pd.isna(subregion_id) else str(subregion_id)
            polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
            records.extend(
                {"source_subregion_id": subregion_id, "geometry": polygon}
                for polygon in polygons
                if polygon is not None and not polygon.is_empty
            )
        return sorted(
            records,
            key=lambda record: (
                str(record["source_subregion_id"] or ""),
                record["geometry"].centroid.x,
                record["geometry"].centroid.y,
            ),
        )

    geometry = source.geometry.union_all()
    if geometry.is_empty:
        return []
    polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    return [
        {"source_subregion_id": None, "geometry": polygon}
        for polygon in sorted(polygons, key=lambda geom: (geom.centroid.x, geom.centroid.y))
    ]


def _included_sfincs_domain_ids(domain_set) -> set[str]:
    values = domain_set.get("include_domain_ids", ())
    return {str(value).strip() for value in values if str(value).strip()}


def _find_column(frame: gpd.GeoDataFrame, candidates: tuple[str, ...]) -> str | None:
    columns = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        column = columns.get(str(candidate).lower())
        if column is not None:
            return column
    return None


def _flows_into(river: LineString, coverage_bbox) -> bool:
    downstream_end = Point(river.coords[-1])
    return coverage_bbox.covers(downstream_end)


def _intersection_points(geometry) -> list[Point]:
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, Point):
        return [geometry]
    if isinstance(geometry, MultiPoint):
        return list(geometry.geoms)
    if isinstance(geometry, LineString):
        return [geometry.interpolate(0.5, normalized=True)]
    if isinstance(geometry, GeometryCollection) or hasattr(geometry, "geoms"):
        points = []
        for part in geometry.geoms:
            points.extend(_intersection_points(part))
        return points
    return []
