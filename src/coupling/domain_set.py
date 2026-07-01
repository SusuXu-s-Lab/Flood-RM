"""Wflow-SFINCS Domain Set geometry helpers."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from paths import location_root_from_paths, relative_to_or_absolute, resolve_location_path
from coupling.domain_geometry import (
    coverage_box_crossings,
    coverage_domain_geometries,
    coverage_domain_id,
    load_crossing_rivers,
    sfincs_coverage_boxes,
    stream_boundary_inflow_crossings,
    subbasin_submodels_from_crossings,
)
from coupling.handoff_sources import read_stream_boundary_handoff_location_artifacts
from coupling.reviewed_gages import (
    accepted_streamgages_by_sfincs_domain,
    accepted_streamgages_frame,
    role_counts,
    sorted_values,
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
                "gauge_site_nos": sorted_values(gage.get("site_no") for gage in domain_gages),
                "frequency_basis": sorted_values(gage.get("frequency_basis") for gage in domain_gages),
                "role_counts": role_counts(domain_gages),
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
                "gauge_site_nos": sorted_values(gage.get("site_no") for gage in domain_gages),
                "frequency_basis": sorted_values(gage.get("frequency_basis") for gage in domain_gages),
                "role_counts": role_counts(domain_gages),
                "watershed_source": watershed_source or "hydromt_primary_sfincs_handoff_subbasin",
            }
        )
    return submodels


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




__all__ = [
    "plan_wflow_domain_set_from_boundary_handoff_watersheds",
    "plan_wflow_domain_set_from_stream_boundary_crossings",
    "source_artifact_handoff_points_by_domain",
]
