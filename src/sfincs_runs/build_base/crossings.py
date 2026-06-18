"""Derive Wflow outlets and SFINCS discharge source points from where the
stream network crosses each SFINCS coverage box, with no USGS-gage dependency.

A stream that crosses the coverage-box boundary carrying meaningful drainage
area is simultaneously (a) a SFINCS inflow boundary source point and (b) the
outlet of a Wflow subbasin that must be delineated upstream of the crossing.
Drainage area (``uparea``) ranks crossings and screens out trickles.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point


def coverage_domain_id(box_row, project_name: str, index: int, count: int) -> str:
    """Return the SFINCS domain id for one coverage box.

    The bbox carries the canonical ``subregion_id`` (e.g. ``greensboro_main``); fall back
    to a positional id only when it is absent so wflow and SFINCS agree on domain ids.
    """
    subregion_id = box_row.get("subregion_id") if hasattr(box_row, "get") else None
    if subregion_id is not None and not pd.isna(subregion_id) and str(subregion_id).strip():
        return str(subregion_id)
    if count == 1:
        return f"{project_name}_main"
    return f"{project_name}_{index + 1:02d}"


def coverage_box_crossings(
    boxes: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    *,
    project_name: str,
    min_uparea_km2: float,
    uparea_col: str = "uparea",
) -> gpd.GeoDataFrame:
    """Return every inflow crossing across all coverage boxes with stable ids.

    This is the single source of truth for crossing locations: SFINCS reads it for
    discharge source points, and Wflow reads it for the gauges it reports ``river_q`` at.
    Each crossing id is namespaced by its SFINCS domain (``<domain>_inflow_NN``).
    """
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
    """Return the WBD HUC domain that contains ``coverage_geom``.

    ``huc_loader(level)`` returns candidate HUC polygons (with a ``huc_id`` column).
    Preference order: the smallest *single* HUC that covers the geometry (finest level
    first). If none exists at any level -- e.g. a coverage box straddling a basin divide --
    and ``allow_union`` is set, fall back to the union of the HUCs the geometry intersects,
    at the finest level whose union covers it. Returns ``{level, kind, huc_id, huc_ids,
    geometry}`` where ``kind`` is ``"single"`` or ``"union"``.
    """
    # Pass 1: prefer a single HUC, coarsening only as needed.
    for level in levels:
        hucs = huc_loader(level)
        if hucs is None or len(hucs) == 0:
            continue
        containing = hucs[hucs.geometry.covers(coverage_geom)]
        if not containing.empty:
            # Equal-area (CONUS Albers) so "smallest" is a true area comparison, not degrees.
            areas = containing.geometry.to_crs("EPSG:5070").area if containing.crs else containing.geometry.area
            smallest = containing.loc[areas.idxmin()]
            huc_id = str(smallest.get("huc_id", ""))
            huc_ids = [part for part in huc_id.split("_") if part]
            kind = str(smallest.get("huc_kind", ""))
            if kind not in {"single", "union"}:
                kind = "union" if len(huc_ids) > 1 else "single"
            return {"level": int(level), "kind": kind, "huc_id": huc_id, "huc_ids": huc_ids or [huc_id], "geometry": smallest.geometry}

    # Pass 2: union of intersecting HUCs (a box on a divide needs HUCs from both basins).
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
                return {"level": int(level), "kind": "union", "huc_id": "_".join(huc_ids), "huc_ids": huc_ids, "geometry": union}

    raise ValueError(
        "No WBD HUC (single or union) at levels "
        f"{tuple(levels)} contains the coverage area; widen the levels (e.g. add 2)."
    )


def stream_boundary_inflow_crossings(
    rivers: gpd.GeoDataFrame,
    coverage_bbox,
    *,
    min_uparea_km2: float,
    uparea_col: str = "uparea",
) -> gpd.GeoDataFrame:
    """Return stream/coverage-box boundary crossings ranked by drainage area.

    Each row is the point where one river reach meets the box perimeter, carrying
    that reach's upstream drainage area. Crossings below ``min_uparea_km2`` are
    dropped so only hydrologically meaningful inflows become coupling points.
    """
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

    crossings = gpd.GeoDataFrame(rows, columns=["river_index", "uparea_km2", "geometry"], geometry="geometry", crs=rivers.crs)
    return crossings.sort_values("uparea_km2", ascending=False).reset_index(drop=True)


def subbasin_submodels_from_crossings(
    crossings: gpd.GeoDataFrame,
    *,
    project_name: str,
    sfincs_domain_id: str,
) -> list[dict]:
    """Turn ranked inflow crossings into Wflow subbasin submodel descriptors.

    One submodel per crossing: HydroMT delineates the contributing subbasin upstream
    of the crossing (region ``{"subbasin": [x, y], "uparea": km2}``) and the same point
    is the SFINCS discharge source it feeds. No USGS gage is involved, so gauge fields
    are empty. Shape mirrors the reviewed-gage planner so manifest/build consumers are
    unchanged.
    """
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


def _flows_into(river: LineString, coverage_bbox) -> bool:
    """True when the reach drains into the box (downstream end inside it).

    Flowlines are digitised upstream->downstream, so the last vertex is the
    downstream end. An inflow ends inside the box; the trunk that leaves the box
    (the SFINCS outflow) ends outside it.
    """
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
        # A boundary-coincident reach contributes its midpoint as the crossing.
        return [geometry.interpolate(0.5, normalized=True)]
    if isinstance(geometry, GeometryCollection) or hasattr(geometry, "geoms"):
        points = []
        for part in geometry.geoms:
            points.extend(_intersection_points(part))
        return points
    return []
