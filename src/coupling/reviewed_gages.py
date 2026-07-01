"""Reviewed streamgage helpers for Wflow-SFINCS Domain Set planning."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point

from coupling.domain_geometry import coverage_domain_id


def accepted_streamgages(path: Path) -> list[dict]:
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


def accepted_streamgages_frame(network_path: Path) -> gpd.GeoDataFrame:
    if not network_path.exists():
        return gpd.GeoDataFrame(columns=["site_no", "geometry"], geometry="geometry", crs="EPSG:4326")
    gages = accepted_streamgages(network_path)
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


def sorted_values(values) -> tuple:
    return tuple(sorted({str(value) for value in values if value not in {None, ""}}))


def role_counts(gages: list[dict]) -> dict:
    counts = {}
    for gage in gages:
        roles = gage.get("roles") or []
        if isinstance(roles, str):
            roles = [role.strip() for role in roles.split(",")]
        for role in roles:
            if role:
                counts[str(role)] = counts.get(str(role), 0) + 1
    return {role: counts[role] for role in sorted(counts)}
