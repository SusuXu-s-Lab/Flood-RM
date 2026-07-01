"""Wflow-SFINCS Domain Set planning."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from paths import location_root_from_paths, relative_to_or_absolute, resolve_location_path
from coupling.domain_set import (
    coverage_domain_id,
    load_crossing_rivers,
    plan_wflow_domain_set_from_boundary_handoff_watersheds,
    plan_wflow_domain_set_from_stream_boundary_crossings,
    source_artifact_handoff_points_by_domain,
    sfincs_coverage_boxes,
)
import wflow_runs.hydromt_recipe as hydromt_recipe
import wflow_runs.types as wflow_types


REVIEWED_STREAMGAGE_SCHEMA = (
    "site_no site_name status drainage_area_sqmi period_start period_end record_years "
    "completeness_score roles frequency_basis wflow_submodel_id sfincs_domain_id "
    "sfincs_handoff_id review_status review_notes"
).split()
NULLABLE_REVIEWED_STREAMGAGE_FIELDS = {"sfincs_handoff_id", "review_notes"}


def plan_wflow_domain_set_from_streamgages(config, paths) -> wflow_types.WflowDomainSetPlan:
    """Plan Wflow submodels from the reviewed USGS streamgage network."""
    location_root = location_root_from_paths(paths)
    wflow = config.get("wflow", {})
    network_path = resolve_location_path(
        location_root,
        wflow.get("streamgage_network", {}).get(
            "reviewed_network",
            "data/sources/usgs_streamgages/streamgage_network.geojson",
        ),
    )
    if not network_path.exists():
        return wflow_types.WflowDomainSetPlan(
            reviewed_network=network_path,
            status="missing_reviewed_network",
            gage_count=0,
            submodel_count=0,
            handoff_count=0,
            submodels=(),
            issues=(f"Reviewed Streamgage Network Artifact is missing: {network_path}",),
        )

    gages = _accepted_streamgages(network_path)
    issues = _schema_issues(gages)
    domain_set = wflow.get("domain_set", {})
    if domain_set.get("allow_multiple_submodels") is False:
        handoff_gages = [gage for gage in gages if gage.get("sfincs_handoff_id")]
        if not handoff_gages:
            issues.append("Accepted streamgages include no SFINCS handoff gages")
        region = _configured_wflow_region(config, location_root)
        if region is None:
            issues.append("Single Wflow domain requires setup_basemaps.region in the Wflow build config")
            region = {}
        submodel_id = str(
            domain_set.get("single_submodel_id")
            or f"{config.get('project', {}).get('name', location_root.name)}_main"
        )
        submodels = []
        if handoff_gages and region:
            submodels.append(
                {
                    "wflow_submodel_id": submodel_id,
                    "region_kind": str(next(iter(region))),
                    "region": region,
                    "outlet_region": region,
                    "subbasin_geometry": None,
                    "sfincs_domain_ids": _sorted_values(gage.get("sfincs_domain_id") for gage in handoff_gages),
                    "sfincs_handoff_ids": _sorted_values(gage.get("sfincs_handoff_id") for gage in handoff_gages),
                    "gauge_site_nos": _sorted_values(gage.get("site_no") for gage in gages),
                    "frequency_basis": _sorted_values(gage.get("frequency_basis") for gage in gages),
                    "role_counts": _role_counts(gages),
                }
            )
        status = "ready" if submodels and not issues else "review_required"
        return wflow_types.WflowDomainSetPlan(
            reviewed_network=network_path,
            status=status,
            gage_count=len(gages),
            submodel_count=len(submodels),
            handoff_count=len(handoff_gages),
            submodels=tuple(submodels),
            issues=tuple(issues),
        )

    submodels = []
    handoff_submodel_ids = sorted(
        {
            gage["wflow_submodel_id"]
            for gage in gages
            if gage.get("wflow_submodel_id") and gage.get("sfincs_handoff_id")
        }
    )
    for submodel_id in handoff_submodel_ids:
        group = [gage for gage in gages if gage.get("wflow_submodel_id") == submodel_id]
        handoff_gages = [gage for gage in group if gage.get("sfincs_handoff_id")]
        outlet = sorted(handoff_gages, key=lambda gage: str(gage["site_no"]))[0]
        outlet_xy = [float(outlet["longitude"]), float(outlet["latitude"])]
        outlet_region = {"subbasin": outlet_xy}
        submodels.append(
            {
                "wflow_submodel_id": submodel_id,
                "region_kind": "subbasin",
                "region": _hydromt_subbasin_region(outlet_xy, outlet),
                "outlet_region": outlet_region,
                "subbasin_geometry": None,
                "sfincs_domain_ids": _sorted_values(gage.get("sfincs_domain_id") for gage in group),
                "sfincs_handoff_ids": _sorted_values(gage.get("sfincs_handoff_id") for gage in handoff_gages),
                "gauge_site_nos": _sorted_values(gage.get("site_no") for gage in group),
                "frequency_basis": _sorted_values(gage.get("frequency_basis") for gage in group),
                "role_counts": _role_counts(group),
            }
        )

    missing_submodel = [gage["site_no"] for gage in gages if not gage.get("wflow_submodel_id")]
    if missing_submodel:
        issues.append("Accepted streamgages missing wflow_submodel_id: " + ", ".join(sorted(missing_submodel)))

    status = "ready" if submodels and not issues else "review_required"
    return wflow_types.WflowDomainSetPlan(
        reviewed_network=network_path,
        status=status,
        gage_count=len(gages),
        submodel_count=len(submodels),
        handoff_count=sum(len(submodel["sfincs_handoff_ids"]) for submodel in submodels),
        submodels=tuple(submodels),
        issues=tuple(issues),
    )


def plan_wflow_domain_set(config, paths) -> wflow_types.WflowDomainSetPlan:
    """Plan the Wflow Domain Set from the configured outlet source."""
    outlet_source = str(config.get("wflow", {}).get("domain_set", {}).get("outlet_source", "reviewed_streamgages"))
    if outlet_source in {"boundary_handoff_watershed", "stream_boundary_watershed", "sfincs_boundary_watershed"}:
        return plan_wflow_domain_set_from_boundary_handoff_watersheds(config, paths)
    if outlet_source == "encompassing_huc":
        return plan_wflow_domain_set_from_encompassing_huc(config, paths)
    if outlet_source == "stream_boundary_crossings":
        return plan_wflow_domain_set_from_stream_boundary_crossings(config, paths)
    return plan_wflow_domain_set_from_streamgages(config, paths)


def plan_wflow_domain_set_from_encompassing_huc(config, paths) -> wflow_types.WflowDomainSetPlan:
    """Plan one Wflow basin per SFINCS coverage box from the smallest encapsulating WBD HUC."""
    from coupling.domain_set import coverage_box_crossings, select_encompassing_huc

    location_root = location_root_from_paths(paths)
    domain_set = config.get("wflow", {}).get("domain_set", {})
    crossings_cfg = domain_set.get("crossings", {})
    huc_cfg = domain_set.get("huc", {})
    project_name = str(config.get("project", {}).get("name", location_root.name))
    min_uparea_km2 = float(crossings_cfg.get("min_uparea_km2", 5.0))
    levels = tuple(int(level) for level in huc_cfg.get("levels", (8, 6, 4)))
    allow_union = bool(huc_cfg.get("allow_union", True))

    coverage_path, boxes = sfincs_coverage_boxes(config, location_root)
    if not coverage_path.exists():
        return wflow_types.WflowDomainSetPlan(
            coverage_path,
            "missing_coverage_bbox",
            0,
            0,
            0,
            (),
            (f"SFINCS coverage source is missing: {coverage_path}",),
        )
    if boxes.empty:
        return wflow_types.WflowDomainSetPlan(
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
        return wflow_types.WflowDomainSetPlan(coverage_path, "missing_river_geometry", 0, 0, 0, (), (str(exc),))

    network_path = resolve_location_path(
        location_root,
        config.get("wflow", {})
        .get("streamgage_network", {})
        .get("reviewed_network", "data/sources/usgs_streamgages/streamgage_network.geojson"),
    )
    accepted_gages = _accepted_streamgages_frame(network_path)
    crossings = coverage_box_crossings(boxes, rivers, project_name=project_name, min_uparea_km2=min_uparea_km2)
    artifact_handoff_points = source_artifact_handoff_points_by_domain(
        config,
        location_root,
        boxes,
        project_name=project_name,
    )
    if crossings.empty and not artifact_handoff_points:
        return wflow_types.WflowDomainSetPlan(
            coverage_path,
            "review_required",
            0,
            0,
            0,
            (),
            (f"no stream/coverage-box inflow crossings above {min_uparea_km2} km2",),
        )

    huc_root = resolve_location_path(location_root, huc_cfg.get("root", "data/wflow/domain_huc"))
    huc_root.mkdir(parents=True, exist_ok=True)
    submodels = []
    issues = []
    combined_rows = []
    count = len(boxes)
    for index, box_row in boxes.iterrows():
        domain_id = coverage_domain_id(box_row, project_name, int(index), count)
        box_crossings = crossings[crossings["sfincs_domain_id"] == domain_id]
        if box_crossings.empty and domain_id not in artifact_handoff_points:
            issues.append(f"{domain_id}: no inflow crossings above {min_uparea_km2} km2")
            continue
        try:
            selected = select_encompassing_huc(
                box_row.geometry,
                _wbd_huc_loader(config, location_root, box_row.geometry),
                levels=levels,
                allow_union=allow_union,
            )
        except ValueError as exc:
            issues.append(f"{domain_id}: {exc}")
            continue

        huc_path = huc_root / f"{domain_id}.geojson"
        gpd.GeoDataFrame(
            {
                "sfincs_domain_id": [domain_id],
                "huc_id": ["_".join(selected["huc_ids"])],
                "huc_level": [selected["level"]],
                "huc_kind": [selected["kind"]],
            },
            geometry=[selected["geometry"]],
            crs="EPSG:4326",
        ).to_file(huc_path, driver="GeoJSON")
        huc_gages = _streamgages_in_geometry(accepted_gages, selected["geometry"])

        handoff_points = artifact_handoff_points.get(domain_id) or [
            {
                "sfincs_handoff_id": str(row["sfincs_handoff_id"]),
                "sfincs_domain_id": domain_id,
                "lon": float(row.geometry.x),
                "lat": float(row.geometry.y),
                "uparea_km2": float(row["uparea_km2"]),
            }
            for _, row in box_crossings.iterrows()
        ]
        region = {"geom": str(huc_path)}
        submodels.append(
            {
                "wflow_submodel_id": domain_id,
                "region_kind": "geom",
                "region": region,
                "outlet_region": region,
                "subbasin_geometry": relative_to_or_absolute(huc_path, location_root),
                "sfincs_domain_ids": [domain_id],
                "sfincs_handoff_ids": [point["sfincs_handoff_id"] for point in handoff_points],
                "handoff_points": handoff_points,
                "gauge_site_nos": _sorted_values(gage.get("site_no") for gage in huc_gages),
                "frequency_basis": _sorted_values(gage.get("frequency_basis") for gage in huc_gages),
                "role_counts": _role_counts(huc_gages),
                "huc_id": "_".join(selected["huc_ids"]),
                "huc_ids": selected["huc_ids"],
                "huc_level": int(selected["level"]),
                "huc_kind": selected["kind"],
            }
        )
        combined_rows.append(
            {
                "wflow_submodel_id": domain_id,
                "sfincs_domain_id": domain_id,
                "huc_id": "_".join(selected["huc_ids"]),
                "huc_level": int(selected["level"]),
                "huc_kind": selected["kind"],
                "handoff_count": int(len(handoff_points)),
                "geometry": selected["geometry"],
            }
        )

    if not submodels:
        return wflow_types.WflowDomainSetPlan(
            coverage_path,
            "review_required",
            0,
            0,
            0,
            (),
            tuple(issues) or ("no Wflow HUC domains could be formed",),
        )

    combined_path = resolve_location_path(location_root, huc_cfg.get("output", "data/wflow/wflow_domain_huc.geojson"))
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(combined_rows, geometry="geometry", crs="EPSG:4326").to_file(combined_path, driver="GeoJSON")

    status = "ready" if not issues else "review_required"
    handoff_count = sum(len(submodel["handoff_points"]) for submodel in submodels)
    return wflow_types.WflowDomainSetPlan(
        coverage_path,
        status,
        len(accepted_gages),
        len(submodels),
        handoff_count,
        tuple(submodels),
        tuple(issues),
    )


def _wbd_huc_loader(config, location_root: Path, coverage_union):
    """Return a level->GeoDataFrame loader over the WBD service for the coverage area."""
    from collect_sources.national_hydrography import WBD_MAPSERVER, fetch_wbd_huc

    huc_cfg = config.get("wflow", {}).get("domain_set", {}).get("huc", {})
    service_url = str(huc_cfg.get("service_url", WBD_MAPSERVER))
    pad = float(huc_cfg.get("query_pad_degrees", 0.1))
    minx, miny, maxx, maxy = coverage_union.bounds
    bbox = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    def loader(level):
        cached = _cached_wbd_hucs(config, location_root, coverage_union, int(level))
        if cached is not None:
            return cached
        return fetch_wbd_huc(bbox, huc_level=level, service_url=service_url)

    return loader


def _cached_wbd_hucs(config, location_root: Path, coverage_union, level: int) -> gpd.GeoDataFrame | None:
    huc_cfg = config.get("wflow", {}).get("domain_set", {}).get("huc", {})
    root = huc_cfg.get("root")
    if not root:
        return None
    huc_root = resolve_location_path(location_root, root)
    if not huc_root.exists():
        return None

    frames = []
    for path in sorted(huc_root.glob("*.geojson")):
        with pd.option_context("mode.chained_assignment", None):
            frame = gpd.read_file(path).to_crs("EPSG:4326")
        if frame.empty or "huc_level" not in frame:
            continue
        frame = frame[pd.to_numeric(frame["huc_level"], errors="coerce") == int(level)].copy()
        if frame.empty:
            continue
        if "huc_id" not in frame:
            frame["huc_id"] = path.stem
        frame["huc_id"] = frame["huc_id"].astype(str)
        frame = frame[frame["huc_id"].str.fullmatch(rf"\d{{{int(level)}}}(?:_\d{{{int(level)}}})*")].copy()
        if frame.empty:
            continue
        columns = ["huc_id", "huc_level", "geometry"]
        if "huc_kind" in frame.columns:
            columns.insert(2, "huc_kind")
        frames.append(frame[columns])
    if not frames:
        return None

    cached = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326")
    cached = cached[cached.geometry.intersects(coverage_union)].copy()
    if cached.empty or not cached.geometry.union_all().covers(coverage_union):
        return None
    return cached


def _accepted_streamgages_frame(network_path: Path) -> gpd.GeoDataFrame:
    if not network_path.exists():
        return gpd.GeoDataFrame(columns=["site_no", "geometry"], geometry="geometry", crs="EPSG:4326")
    gages = _accepted_streamgages(network_path)
    if not gages:
        return gpd.GeoDataFrame(columns=["site_no", "geometry"], geometry="geometry", crs="EPSG:4326")
    geometry = [Point(float(gage["longitude"]), float(gage["latitude"])) for gage in gages]
    return gpd.GeoDataFrame(gages, geometry=geometry, crs="EPSG:4326")


def _streamgages_in_geometry(gages: gpd.GeoDataFrame, geometry) -> list[dict]:
    if gages.empty:
        return []
    selected = gages[gages.geometry.map(lambda point: bool(geometry.covers(point)))].copy()
    return selected.drop(columns=["geometry"]).to_dict("records")


def _configured_wflow_region(config, location_root: Path) -> dict | None:
    build_config = resolve_location_path(
        location_root,
        config.get("wflow", {}).get("build_config", "wflow_build.yml"),
    )
    build_config = hydromt_recipe.ensure_model_recipe_file(config, "wflow_build", build_config)
    if not build_config.exists():
        return None
    workflow = hydromt_recipe.read_workflow(build_config)
    for step in workflow.get("steps", []):
        if not isinstance(step, dict):
            continue
        basemaps = step.get("setup_basemaps")
        if not isinstance(basemaps, dict):
            continue
        region = basemaps.get("region")
        if isinstance(region, dict) and region:
            return deepcopy(region)
    return None


def _hydromt_subbasin_region(outlet_xy: list[float], outlet: dict) -> dict:
    region = {"subbasin": list(outlet_xy)}
    uparea = _reviewed_drainage_area_km2(outlet)
    if uparea is not None:
        region["uparea"] = uparea
    return region


def _reviewed_drainage_area_km2(gage: dict) -> float | None:
    drainage_area_sqmi = pd.to_numeric(gage.get("drainage_area_sqmi"), errors="coerce")
    if pd.isna(drainage_area_sqmi):
        return None
    return _drainage_area_km2(drainage_area_sqmi)


def _drainage_area_km2(drainage_area_sqmi) -> float | None:
    if pd.isna(drainage_area_sqmi):
        return None
    return float(drainage_area_sqmi) * 2.589988110336


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


def _schema_issues(gages: list[dict]) -> list[str]:
    issues = []
    for gage in gages:
        missing = [field for field in REVIEWED_STREAMGAGE_SCHEMA if _missing(gage, field)]
        if missing:
            issues.append(
                f"{gage.get('site_no', '<unknown>')} missing reviewed schema fields: "
                + ", ".join(missing)
            )
    return issues


def _missing(gage: dict, field: str) -> bool:
    if field not in gage:
        return True
    if field in NULLABLE_REVIEWED_STREAMGAGE_FIELDS:
        return False
    value = gage.get(field)
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    return False
