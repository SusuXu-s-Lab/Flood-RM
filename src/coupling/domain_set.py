"""Wflow-SFINCS Domain Set geometry helpers."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point
from shapely.ops import unary_union

from paths import location_root_from_paths, relative_to_or_absolute, resolve_location_path
from coupling.handoff_sources import read_stream_boundary_handoff_location_artifacts


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


def plan_wflow_domain_set_from_stream_boundary_crossings(config, paths):
    """Plan Wflow Submodels from where streams cross each SFINCS coverage box."""
    from wflow_runs.types import WflowDomainSetPlan

    location_root = location_root_from_paths(paths)
    domain_set = config.get("wflow", {}).get("domain_set", {})
    crossings_cfg = domain_set.get("crossings", {})
    project_name = str(config.get("project", {}).get("name", location_root.name))
    min_uparea_km2 = float(crossings_cfg.get("min_uparea_km2", 5.0))

    coverage_path, boxes = sfincs_coverage_boxes(config, location_root)
    if not coverage_path.exists():
        return WflowDomainSetPlan(
            coverage_path,
            "missing_coverage_bbox",
            0,
            0,
            0,
            (),
            (f"SFINCS coverage source is missing: {coverage_path}",),
        )
    if boxes.empty:
        return WflowDomainSetPlan(
            coverage_path,
            "missing_coverage_bbox",
            0,
            0,
            0,
            (),
            (f"SFINCS coverage source has no polygons: {coverage_path}",),
        )
    try:
        rivers = load_crossing_rivers(config, location_root)
    except FileNotFoundError as exc:
        return WflowDomainSetPlan(coverage_path, "missing_river_geometry", 0, 0, 0, (), (str(exc),))

    submodels = []
    issues = []
    for index, row in boxes.iterrows():
        domain_id = coverage_domain_id(row, project_name, int(index), len(boxes))
        crossings = stream_boundary_inflow_crossings(rivers, row.geometry, min_uparea_km2=min_uparea_km2)
        if crossings.empty:
            issues.append(f"{domain_id}: no stream/coverage-box inflow crossings above {min_uparea_km2} km2")
            continue
        submodels.extend(
            subbasin_submodels_from_crossings(
                crossings,
                project_name=project_name,
                sfincs_domain_id=domain_id,
            )
        )

    status = "ready" if submodels and not issues else "review_required"
    return WflowDomainSetPlan(
        reviewed_network=coverage_path,
        status=status,
        gage_count=0,
        submodel_count=len(submodels),
        handoff_count=len(submodels),
        submodels=tuple(submodels),
        issues=tuple(issues),
    )


def plan_wflow_domain_set_from_boundary_handoff_watersheds(config, paths):
    """Plan one Wflow watershed per SFINCS domain from boundary inflow points."""
    from wflow_runs.types import WflowDomainSetPlan

    location_root = location_root_from_paths(paths)
    domain_set = config.get("wflow", {}).get("domain_set", {})
    crossings_cfg = domain_set.get("crossings", {})
    project_name = str(config.get("project", {}).get("name", location_root.name))
    min_uparea_km2 = float(crossings_cfg.get("min_uparea_km2", 5.0))

    coverage_path, boxes = sfincs_coverage_boxes(config, location_root)
    if not coverage_path.exists():
        return WflowDomainSetPlan(
            coverage_path,
            "missing_coverage_bbox",
            0,
            0,
            0,
            (),
            (f"SFINCS coverage source is missing: {coverage_path}",),
        )
    if boxes.empty:
        return WflowDomainSetPlan(
            coverage_path,
            "missing_coverage_bbox",
            0,
            0,
            0,
            (),
            (f"SFINCS coverage source has no polygons: {coverage_path}",),
        )

    network_path = resolve_location_path(
        location_root,
        config.get("wflow", {})
        .get("streamgage_network", {})
        .get("reviewed_network", "data/sources/usgs_streamgages/streamgage_network.geojson"),
    )
    accepted_gages = accepted_streamgages_frame(network_path)
    ignore_source_artifacts = bool(
        domain_set.get("ignore_sfincs_handoff_artifacts")
        or crossings_cfg.get("ignore_sfincs_handoff_artifacts")
    )
    artifact_submodels = (
        []
        if ignore_source_artifacts
        else boundary_handoff_submodels_from_source_artifacts(
            config,
            location_root,
            boxes,
            accepted_gages=accepted_gages,
            project_name=project_name,
            min_uparea_km2=min_uparea_km2,
        )
    )
    if artifact_submodels:
        return WflowDomainSetPlan(
            reviewed_network=coverage_path,
            status="ready",
            gage_count=len(accepted_gages),
            submodel_count=len(artifact_submodels),
            handoff_count=sum(len(submodel["handoff_points"]) for submodel in artifact_submodels),
            submodels=tuple(artifact_submodels),
            issues=(),
        )

    try:
        rivers = load_crossing_rivers(config, location_root)
    except FileNotFoundError as exc:
        return WflowDomainSetPlan(coverage_path, "missing_river_geometry", 0, 0, 0, (), (str(exc),))

    crossings = coverage_box_crossings(
        boxes,
        rivers,
        project_name=project_name,
        min_uparea_km2=min_uparea_km2,
    )
    if crossings.empty:
        return WflowDomainSetPlan(
            coverage_path,
            "review_required",
            0,
            0,
            0,
            (),
            (f"no stream/coverage-box inflow crossings above {min_uparea_km2} km2",),
        )

    submodels = []
    gages_by_domain = accepted_streamgages_by_sfincs_domain(accepted_gages, boxes, project_name)
    domain_geometries = coverage_domain_geometries(boxes, project_name)
    for domain_id, group in crossings.groupby("sfincs_domain_id", sort=True):
        group = group.sort_values("uparea_km2", ascending=False).reset_index(drop=True)
        handoff_points = [
            {
                "sfincs_handoff_id": str(row["sfincs_handoff_id"]),
                "sfincs_domain_id": str(domain_id),
                "lon": float(row.geometry.x),
                "lat": float(row.geometry.y),
                "uparea_km2": float(row["uparea_km2"]),
            }
            for _, row in group.iterrows()
        ]
        domain_gages = gages_by_domain.get(str(domain_id), [])
        handoff_region = hydromt_boundary_handoff_subbasin_region(
            handoff_points,
            min_uparea_km2=min_uparea_km2,
        )
        reviewed_region, subbasin_geometry, watershed_source = reviewed_wflow_watershed_region(
            config,
            location_root,
            str(domain_id),
            domain_geometries.get(str(domain_id)),
        )
        region = reviewed_region or handoff_region
        submodels.append(
            {
                "wflow_submodel_id": str(domain_id),
                "region_kind": "geom" if reviewed_region else "subbasin",
                "region": region,
                "outlet_region": handoff_region,
                "subbasin_geometry": subbasin_geometry,
                "sfincs_domain_ids": [str(domain_id)],
                "sfincs_handoff_ids": [point["sfincs_handoff_id"] for point in handoff_points],
                "handoff_points": handoff_points,
                "gauge_site_nos": _sorted_values(gage.get("site_no") for gage in domain_gages),
                "frequency_basis": _sorted_values(gage.get("frequency_basis") for gage in domain_gages),
                "role_counts": _role_counts(domain_gages),
                "watershed_source": watershed_source or "hydromt_primary_boundary_handoff_subbasin",
            }
        )

    return WflowDomainSetPlan(
        reviewed_network=coverage_path,
        status="ready",
        gage_count=len(accepted_gages),
        submodel_count=len(submodels),
        handoff_count=sum(len(submodel["handoff_points"]) for submodel in submodels),
        submodels=tuple(submodels),
        issues=(),
    )


def boundary_handoff_submodels_from_source_artifacts(
    config: dict,
    location_root: Path,
    boxes: gpd.GeoDataFrame,
    *,
    accepted_gages: list[dict],
    project_name: str,
    min_uparea_km2: float | None,
) -> list[dict]:
    count = len(boxes)
    domain_ids = {
        coverage_domain_id(box_row, project_name, int(index), count)
        for index, box_row in boxes.reset_index(drop=True).iterrows()
    }
    points_by_domain = source_artifact_handoff_points_by_domain(
        config,
        location_root,
        boxes,
        project_name=project_name,
    )
    if not points_by_domain or set(points_by_domain) != domain_ids:
        return []

    submodels = []
    gages_by_domain = accepted_streamgages_by_sfincs_domain(accepted_gages, boxes, project_name)
    domain_geometries = coverage_domain_geometries(boxes, project_name)
    for domain_id in sorted(points_by_domain):
        handoff_points = points_by_domain[domain_id]
        domain_gages = gages_by_domain.get(str(domain_id), [])
        handoff_region = hydromt_boundary_handoff_subbasin_region(
            handoff_points,
            min_uparea_km2=min_uparea_km2,
        )
        reviewed_region, subbasin_geometry, watershed_source = reviewed_wflow_watershed_region(
            config,
            location_root,
            domain_id,
            domain_geometries.get(str(domain_id)),
        )
        region = reviewed_region or handoff_region
        submodels.append(
            {
                "wflow_submodel_id": str(domain_id),
                "region_kind": "geom" if reviewed_region else "subbasin",
                "region": region,
                "outlet_region": handoff_region,
                "subbasin_geometry": subbasin_geometry,
                "sfincs_domain_ids": [str(domain_id)],
                "sfincs_handoff_ids": [point["sfincs_handoff_id"] for point in handoff_points],
                "handoff_points": handoff_points,
                "gauge_site_nos": _sorted_values(gage.get("site_no") for gage in domain_gages),
                "frequency_basis": _sorted_values(gage.get("frequency_basis") for gage in domain_gages),
                "role_counts": _role_counts(domain_gages),
                "watershed_source": watershed_source or "hydromt_primary_sfincs_handoff_subbasin",
            }
        )
    return submodels


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


def write_wflow_crossing_gauge_locations(config, paths, submodel: dict, *, output=None) -> dict:
    """Write HydroMT-Wflow gauges for crossing-derived Wflow-SFINCS handoff points."""
    location_root = location_root_from_paths(paths)
    points = submodel.get("handoff_points")
    if points:
        records = [
            {
                "sfincs_handoff_id": str(point["sfincs_handoff_id"]),
                "sfincs_domain_id": str(point.get("sfincs_domain_id", "")),
                "lon": float(point["lon"]),
                "lat": float(point["lat"]),
                "uparea": float(point["uparea_km2"]) if point.get("uparea_km2") is not None else np.nan,
            }
            for point in points
        ]
    else:
        region = submodel.get("region", {}) or {}
        outlet_xy = (submodel.get("outlet_region") or region).get("subbasin") or region.get("subbasin")
        if not outlet_xy:
            raise ValueError(f"Wflow Submodel {submodel.get('wflow_submodel_id')} has no crossing outlet for a gauge")
        handoff_id = next(
            (str(value) for value in submodel.get("sfincs_handoff_ids", ()) if value),
            str(submodel.get("wflow_submodel_id")),
        )
        uparea = region.get("uparea")
        records = [
            {
                "sfincs_handoff_id": handoff_id,
                "sfincs_domain_id": next((str(value) for value in submodel.get("sfincs_domain_ids", ()) if value), ""),
                "lon": float(outlet_xy[0]),
                "lat": float(outlet_xy[1]),
                "uparea": float(uparea) if uparea is not None else np.nan,
            }
        ]

    gauges = gpd.GeoDataFrame(
        {
            "index": range(1, len(records) + 1),
            "name": [record["sfincs_handoff_id"] for record in records],
            "uparea": [record["uparea"] for record in records],
            "sfincs_handoff_id": [record["sfincs_handoff_id"] for record in records],
            "wflow_submodel_id": [str(submodel.get("wflow_submodel_id"))] * len(records),
            "sfincs_domain_id": [record["sfincs_domain_id"] for record in records],
            "gauge_location_source": ["sfincs_stream_boundary_intersection"] * len(records),
        },
        geometry=[Point(record["lon"], record["lat"]) for record in records],
        crs="EPSG:4326",
    )
    out_path = resolve_location_path(
        location_root,
        output
        or Path(config.get("wflow", {}).get("gauges", {}).get("root", "data/wflow/domain_set_gauges"))
        / f"{submodel['wflow_submodel_id']}_sfincs_gauges.geojson",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gauges.to_file(out_path, driver="GeoJSON")
    outlet_source = str(config.get("wflow", {}).get("domain_set", {}).get("outlet_source", "reviewed_streamgages"))
    snap_uparea = bool(gauges["uparea"].notna().all()) and outlet_source not in {
        "boundary_handoff_watershed",
        "stream_boundary_watershed",
        "sfincs_boundary_watershed",
    }
    return {
        "gauges_fn": out_path,
        "gauge_count": int(len(gauges)),
        "snap_to_river": True,
        "snap_uparea": snap_uparea,
        "wflow_submodel_id": str(submodel["wflow_submodel_id"]),
        "sfincs_handoff_ids": tuple(record["sfincs_handoff_id"] for record in records),
    }


def accepted_streamgages_frame(network_path: Path) -> gpd.GeoDataFrame:
    if not network_path.exists():
        return gpd.GeoDataFrame(columns=["site_no", "geometry"], geometry="geometry", crs="EPSG:4326")
    gages = _accepted_streamgages(network_path)
    if not gages:
        return gpd.GeoDataFrame(columns=["site_no", "geometry"], geometry="geometry", crs="EPSG:4326")
    geometry = [Point(float(gage["longitude"]), float(gage["latitude"])) for gage in gages]
    return gpd.GeoDataFrame(gages, geometry=geometry, crs="EPSG:4326")


def accepted_streamgages_by_sfincs_domain(
    accepted_gages,
    boxes: gpd.GeoDataFrame,
    project_name: str,
) -> dict[str, list[dict]]:
    by_domain: dict[str, list[dict]] = {}
    if hasattr(accepted_gages, "to_dict"):
        gage_records = accepted_gages.to_dict("records")
    else:
        gage_records = list(accepted_gages)
    domain_ids = {
        coverage_domain_id(box_row, project_name, int(index), len(boxes))
        for index, box_row in boxes.reset_index(drop=True).iterrows()
    }
    for gage in gage_records:
        domain_id = str(gage.get("sfincs_domain_id") or "")
        if domain_id in domain_ids:
            by_domain.setdefault(domain_id, []).append(gage)
    if by_domain:
        return by_domain

    boxes_wgs = boxes.to_crs("EPSG:4326") if boxes.crs is not None else boxes
    for gage in gage_records:
        try:
            point = Point(float(gage["longitude"]), float(gage["latitude"]))
        except Exception:
            continue
        for index, box_row in boxes_wgs.reset_index(drop=True).iterrows():
            if box_row.geometry is not None and box_row.geometry.intersects(point):
                domain_id = coverage_domain_id(box_row, project_name, int(index), len(boxes_wgs))
                by_domain.setdefault(domain_id, []).append(gage)
    return by_domain


def source_artifact_handoff_points_by_domain(
    config: dict,
    location_root: Path,
    boxes: gpd.GeoDataFrame,
    *,
    project_name: str,
) -> dict[str, list[dict]]:
    locations = read_stream_boundary_handoff_location_artifacts(
        config,
        location_root,
        location_path=resolve_location_path,
    )
    if locations is None or locations.empty:
        return {}

    count = len(boxes)
    domain_ids = {
        coverage_domain_id(box_row, project_name, int(index), count)
        for index, box_row in boxes.reset_index(drop=True).iterrows()
    }
    locations = locations[locations["sfincs_domain_id"].astype(str).isin(domain_ids)].copy()
    if locations.empty:
        return {}

    locations = locations.to_crs("EPSG:4326")
    uparea_col = "uparea" if "uparea" in locations else "uparea_km2" if "uparea_km2" in locations else None
    if uparea_col is None:
        locations["_uparea_km2"] = float("nan")
        uparea_col = "_uparea_km2"
    points_by_domain: dict[str, list[dict]] = {}
    for domain_id, group in locations.groupby("sfincs_domain_id", sort=True):
        group = group.copy()
        group["_uparea_sort"] = pd.to_numeric(group[uparea_col], errors="coerce").fillna(-1.0)
        group = group.sort_values(["_uparea_sort", "sfincs_handoff_id"], ascending=[False, True]).reset_index(drop=True)
        points_by_domain[str(domain_id)] = [
            {
                "sfincs_handoff_id": str(row["sfincs_handoff_id"]),
                "sfincs_domain_id": str(domain_id),
                "lon": float(row.geometry.x),
                "lat": float(row.geometry.y),
                "uparea_km2": float(row["_uparea_sort"]) if float(row["_uparea_sort"]) >= 0 else float("nan"),
            }
            for _, row in group.iterrows()
        ]
    return points_by_domain


def hydromt_boundary_handoff_subbasin_region(
    handoff_points: list[dict],
    *,
    min_uparea_km2: float | None,
) -> dict:
    """Return the fallback HydroMT subbasin region for one SFINCS domain."""
    if not handoff_points:
        raise ValueError("Boundary-handoff Wflow region requires at least one handoff point")
    point = _primary_handoff_subbasin_point(handoff_points)
    region = {"subbasin": [float(point["lon"]), float(point["lat"])]}
    if min_uparea_km2 is not None and float(min_uparea_km2) > 0:
        region["uparea"] = float(min_uparea_km2)
    return region


def coverage_domain_geometries(boxes: gpd.GeoDataFrame, project_name: str) -> dict[str, object]:
    count = len(boxes)
    return {
        coverage_domain_id(box_row, project_name, int(index), count): box_row.geometry
        for index, box_row in boxes.reset_index(drop=True).iterrows()
        if box_row.geometry is not None and not box_row.geometry.is_empty
    }


def reviewed_wflow_watershed_region(
    config: dict,
    location_root: Path,
    domain_id: str,
    domain_geometry,
) -> tuple[dict | None, str | None, str | None]:
    """Return the reviewed full-watershed HydroMT ``geom`` region for a SFINCS domain."""
    if domain_geometry is None or domain_geometry.is_empty:
        return None, None, None

    wflow_extent = (config.get("static_sources", {}) or {}).get("wflow_collection_extent", {}) or {}
    source_path = resolve_location_path(
        location_root,
        wflow_extent.get("watersheds", "data/static/aoi/wflow_nhdplus_watersheds.geojson"),
    )
    if not source_path.exists():
        return None, None, None

    watersheds = gpd.read_file(source_path)
    if watersheds.empty:
        return None, None, None
    if watersheds.crs is None:
        watersheds = watersheds.set_crs("EPSG:4326")
    watersheds = watersheds.to_crs("EPSG:4326")
    selected = _select_reviewed_watershed_rows(watersheds, str(domain_id), domain_geometry)
    if selected.empty:
        return None, None, None

    domain_frame = gpd.GeoDataFrame(geometry=[domain_geometry], crs="EPSG:4326")
    domain_equal_area = domain_frame.to_crs("EPSG:5070").geometry.iloc[0]
    selected_equal_area = selected.to_crs("EPSG:5070").geometry.union_all()
    if not selected_equal_area.covers(domain_equal_area):
        return None, None, None

    output_root = resolve_location_path(
        location_root,
        (config.get("wflow", {}) or {})
        .get("domain_set", {})
        .get("reviewed_watershed_root", "data/wflow/domain_set_watersheds"),
    )
    output_path = output_root / f"{domain_id}.geojson"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {
            "wflow_submodel_id": [str(domain_id)],
            "sfincs_domain_id": [str(domain_id)],
            "source": [relative_to_or_absolute(source_path, location_root)],
        },
        geometry=[selected.geometry.union_all()],
        crs="EPSG:4326",
    ).to_file(output_path, driver="GeoJSON")
    relative_output = relative_to_or_absolute(output_path, location_root)
    return {"geom": output_path.as_posix()}, relative_output, "reviewed_wflow_watershed_geometry"


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


def _accepted_streamgages(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for feature in payload.get("features", []):
        properties = dict(feature.get("properties", {}))
        if str(properties.get("status", "")).lower() != "active":
            continue
        if str(properties.get("review_status", "")).lower() not in {"accepted", "accepted_with_warning"}:
            continue
        coordinates = (feature.get("geometry") or {}).get("coordinates") or []
        if len(coordinates) < 2:
            continue
        properties["site_no"] = str(properties.get("site_no", ""))
        properties["longitude"] = coordinates[0]
        properties["latitude"] = coordinates[1]
        rows.append(properties)
    return rows


def _select_reviewed_watershed_rows(watersheds: gpd.GeoDataFrame, domain_id: str, domain_geometry) -> gpd.GeoDataFrame:
    for column in ("wflow_submodel_id", "sfincs_domain_id", "subregion_id"):
        if column in watersheds.columns:
            selected = watersheds[watersheds[column].astype(str).eq(str(domain_id))].copy()
            if not selected.empty:
                return selected

    domain_frame = gpd.GeoDataFrame(geometry=[domain_geometry], crs="EPSG:4326")
    domain_equal_area = domain_frame.to_crs("EPSG:5070").geometry.iloc[0]
    candidates = watersheds[watersheds.intersects(domain_geometry)].copy()
    if candidates.empty and len(watersheds) == 1:
        candidates = watersheds.copy()
    if candidates.empty:
        return candidates

    candidates_equal_area = candidates.to_crs("EPSG:5070")
    covering = candidates_equal_area[candidates_equal_area.covers(domain_equal_area)].copy()
    if covering.empty:
        return gpd.GeoDataFrame(columns=watersheds.columns, geometry="geometry", crs=watersheds.crs)
    areas = covering.geometry.area
    return candidates.loc[[areas.idxmin()]].copy()


def _primary_handoff_subbasin_point(handoff_points: list[dict]) -> dict:
    def sort_key(point: dict):
        uparea = point.get("uparea_km2")
        try:
            uparea_value = float(uparea)
        except (TypeError, ValueError):
            uparea_value = float("nan")
        if pd.isna(uparea_value):
            uparea_value = -1.0
        return (-uparea_value, str(point.get("sfincs_handoff_id") or ""))

    return sorted(handoff_points, key=sort_key)[0]


def _sorted_values(values) -> tuple:
    return tuple(sorted({str(value) for value in values if value not in {None, ""}}))


def _role_counts(gages: list[dict]) -> dict:
    counts = {}
    for gage in gages:
        roles = gage.get("roles") or []
        if isinstance(roles, str):
            roles = [role.strip() for role in roles.split(",")]
        for role in roles:
            if role:
                counts[str(role)] = counts.get(str(role), 0) + 1
    return {role: counts[role] for role in sorted(counts)}


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


__all__ = [
    "coverage_box_crossings",
    "coverage_domain_id",
    "coverage_domain_geometries",
    "load_crossing_rivers",
    "plan_wflow_domain_set_from_boundary_handoff_watersheds",
    "plan_wflow_domain_set_from_stream_boundary_crossings",
    "source_artifact_handoff_points_by_domain",
    "select_encompassing_huc",
    "sfincs_coverage_boxes",
    "stream_boundary_inflow_crossings",
    "subbasin_submodels_from_crossings",
    "write_wflow_crossing_gauge_locations",
]
