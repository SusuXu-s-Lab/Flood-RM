from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import Point

from paths import location_root_from_paths, resolve_location_path
from coupling.domain_set import write_wflow_crossing_gauge_locations
from coupling.handoff_sources import (
    read_stream_boundary_handoff_location_artifacts,
    read_stream_boundary_handoff_locations,
)


def write_wflow_sfincs_gauge_locations(config, paths, submodel: dict, *, output=None) -> dict:
    """Write HydroMT-Wflow gauges aligned to SFINCS discharge handoff points."""
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
        raise FileNotFoundError(network_path)
    configured_handoff_ids = {str(value) for value in submodel.get("sfincs_handoff_ids", ()) if value}
    if not configured_handoff_ids:
        raise ValueError(f"Wflow Submodel {submodel.get('wflow_submodel_id')} has no SFINCS handoff IDs")

    boundary_locations = sfincs_boundary_handoff_locations(
        config,
        location_root,
        configured_handoff_ids,
        submodel_id=str(submodel.get("wflow_submodel_id", "")),
    )
    handoff_ids = (
        set(boundary_locations["sfincs_handoff_id"].astype(str))
        if boundary_locations is not None
        else configured_handoff_ids
    )
    gages = boundary_locations if boundary_locations is not None else gpd.GeoDataFrame(_accepted_streamgages(network_path))
    if gages.empty:
        raise ValueError(f"Reviewed Streamgage Network has no accepted active gages: {network_path}")
    gages = gages[gages["sfincs_handoff_id"].astype(str).isin(handoff_ids)].copy()
    missing = sorted(handoff_ids - set(gages["sfincs_handoff_id"].astype(str)))
    if missing:
        raise ValueError("SFINCS handoff IDs missing from accepted active reviewed gages: " + ", ".join(missing))

    if isinstance(gages, gpd.GeoDataFrame) and gages.geometry.name in gages:
        gauges = gages.copy()
    else:
        geometry = [Point(float(row["longitude"]), float(row["latitude"])) for _, row in gages.iterrows()]
        gauges = gpd.GeoDataFrame(gages, geometry=geometry, crs="EPSG:4326")
    gauges = gauges.sort_values(["sfincs_handoff_id", "site_no"]).reset_index(drop=True)
    if "index" in gauges.columns:
        gauges = gauges.drop(columns=["index"])
    gauges.insert(0, "index", range(1, len(gauges) + 1))
    gauges["name"] = gauges["sfincs_handoff_id"].astype(str)
    use_boundary_locations = boundary_locations is not None
    if "drainage_area_sqmi" in gauges:
        gauges["uparea"] = pd.to_numeric(gauges["drainage_area_sqmi"], errors="coerce").map(_drainage_area_km2)
    elif "uparea" in gauges:
        gauges["uparea"] = pd.to_numeric(gauges["uparea"], errors="coerce")
    else:
        gauges["uparea"] = np.nan
    if not use_boundary_locations and gauges["uparea"].isna().any():
        missing_sites = ", ".join(gauges.loc[gauges["uparea"].isna(), "site_no"].astype(str))
        raise ValueError(f"Reviewed SFINCS handoff gages are missing drainage_area_sqmi for snap_uparea: {missing_sites}")
    if use_boundary_locations:
        if "gauge_location_source" not in gauges:
            gauges["gauge_location_source"] = "sfincs_stream_boundary_intersection"
        else:
            missing_source = gauges["gauge_location_source"].isna() | gauges["gauge_location_source"].astype(str).str.strip().eq("")
            gauges.loc[missing_source, "gauge_location_source"] = "sfincs_stream_boundary_intersection"

    out_path = resolve_location_path(
        location_root,
        output
        or Path(wflow.get("gauges", {}).get("root", "data/wflow/domain_set_gauges"))
        / f"{submodel['wflow_submodel_id']}_sfincs_gauges.geojson",
    )
    keep = [
        "index",
        "name",
        "uparea",
        "site_no",
        "sfincs_handoff_id",
        "wflow_submodel_id",
        "sfincs_domain_id",
        "gauge_location_source",
        "handoff_placement",
        "handoff_location_review_status",
        "stream_boundary_river_source",
        "geometry",
    ]
    keep = [column for column in keep if column in gauges.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gauges[keep].to_file(out_path, driver="GeoJSON")
    return {
        "gauges_fn": out_path,
        "gauge_count": int(len(gauges)),
        "snap_to_river": True,
        "snap_uparea": False if use_boundary_locations else not gauges["uparea"].isna().any(),
        "wflow_submodel_id": str(submodel["wflow_submodel_id"]),
        "sfincs_handoff_ids": tuple(gauges["sfincs_handoff_id"].astype(str)),
    }


def sfincs_boundary_handoff_locations(
    config,
    location_root: Path,
    handoff_ids: set[str],
    *,
    submodel_id: str | None = None,
) -> gpd.GeoDataFrame | None:
    if submodel_id:
        locations = read_stream_boundary_handoff_location_artifacts(
            config,
            location_root,
            location_path=resolve_location_path,
        )
        if locations is not None and "wflow_submodel_id" in locations:
            locations = locations[locations["wflow_submodel_id"].astype(str) == str(submodel_id)].copy()
            if not locations.empty:
                locations = locations[locations["sfincs_handoff_id"].astype(str).isin(handoff_ids)].copy()
                missing = sorted(handoff_ids - set(locations["sfincs_handoff_id"].astype(str)))
                if missing:
                    raise ValueError(
                        "SFINCS boundary handoff source artifacts are missing IDs needed by Wflow: "
                        + ", ".join(missing)
                    )
                return locations

    locations = read_stream_boundary_handoff_locations(
        config,
        location_root,
        handoff_ids,
        location_path=resolve_location_path,
    )
    if locations is not None:
        return locations
    return None


def write_wflow_observation_gauge_locations(config, paths, submodel: dict, *, output=None) -> dict:
    """Write all reviewed streamgages in the submodel as Wflow observation gauges."""
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
        raise FileNotFoundError(network_path)

    gages = gpd.GeoDataFrame(_accepted_streamgages(network_path))
    if gages.empty:
        raise ValueError(f"Reviewed Streamgage Network has no accepted active gages: {network_path}")
    submodel_sites = {str(value) for value in submodel.get("gauge_site_nos", ()) if value}
    if submodel_sites:
        gages = gages[gages["site_no"].astype(str).isin(submodel_sites)].copy()
    if gages.empty:
        raise ValueError(
            f"Wflow Submodel {submodel.get('wflow_submodel_id')} has no reviewed observation gages"
        )

    geometry = [Point(float(row["longitude"]), float(row["latitude"])) for _, row in gages.iterrows()]
    gauges = gpd.GeoDataFrame(gages, geometry=geometry, crs="EPSG:4326")
    if "wflow_submodel_id" in gauges:
        gauges["reviewed_wflow_submodel_id"] = gauges["wflow_submodel_id"]
    if "sfincs_domain_id" in gauges:
        gauges["reviewed_sfincs_domain_id"] = gauges["sfincs_domain_id"]
    gauges["wflow_submodel_id"] = str(submodel["wflow_submodel_id"])
    handoff = gauges.get("sfincs_handoff_id")
    gauges["role"] = [
        "sfincs_source" if value not in (None, "") and not pd.isna(value) else "observation"
        for value in (handoff if handoff is not None else [None] * len(gauges))
    ]
    gauges = gauges.sort_values(["role", "site_no"]).reset_index(drop=True)
    gauges.insert(0, "index", range(1, len(gauges) + 1))
    gauges["name"] = gauges["site_no"].astype(str)
    gauges["uparea"] = pd.to_numeric(gauges["drainage_area_sqmi"], errors="coerce").map(_drainage_area_km2)
    snap_uparea = bool(gauges["uparea"].notna().all())

    out_path = resolve_location_path(
        location_root,
        output
        or Path(wflow.get("gauges", {}).get("root", "data/wflow/domain_set_gauges"))
        / f"{submodel['wflow_submodel_id']}_observation_gauges.geojson",
    )
    keep = [
        "index",
        "name",
        "uparea",
        "site_no",
        "role",
        "sfincs_handoff_id",
        "wflow_submodel_id",
        "reviewed_wflow_submodel_id",
        "sfincs_domain_id",
        "reviewed_sfincs_domain_id",
        "geometry",
    ]
    keep = [column for column in keep if column in gauges.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gauges[keep].to_file(out_path, driver="GeoJSON")
    return {
        "gauges_fn": out_path,
        "gauge_count": int(len(gauges)),
        "snap_uparea": snap_uparea,
        "wflow_submodel_id": str(submodel["wflow_submodel_id"]),
        "site_nos": tuple(gauges["site_no"].astype(str)),
    }


def write_wflow_sfincs_handoff_gauge_locations(config, paths, submodel: dict) -> dict:
    """Write Wflow gauges at the active SFINCS handoff geometry for this submodel."""
    domain_set = config.get("wflow", {}).get("domain_set", {})
    outlet_source = str(domain_set.get("outlet_source", "reviewed_streamgages"))
    if domain_set.get("ignore_sfincs_handoff_artifacts"):
        return write_wflow_crossing_gauge_locations(config, paths, submodel)
    if outlet_source in {"stream_boundary_crossings", "boundary_handoff_watershed", "stream_boundary_watershed", "sfincs_boundary_watershed"}:
        location_root = location_root_from_paths(paths)
        handoff_ids = {str(value) for value in submodel.get("sfincs_handoff_ids", ()) if value}
        if sfincs_boundary_handoff_locations(
            config,
            location_root,
            handoff_ids,
            submodel_id=str(submodel.get("wflow_submodel_id", "")),
        ) is not None:
            return write_wflow_sfincs_gauge_locations(config, paths, submodel)
        return write_wflow_crossing_gauge_locations(config, paths, submodel)
    if outlet_source == "encompassing_huc":
        location_root = location_root_from_paths(paths)
        handoff_ids = {str(value) for value in submodel.get("sfincs_handoff_ids", ()) if value}
        if handoff_ids and sfincs_boundary_handoff_locations(
            config,
            location_root,
            handoff_ids,
            submodel_id=str(submodel.get("wflow_submodel_id", "")),
        ) is not None:
            return write_wflow_sfincs_gauge_locations(config, paths, submodel)
        return write_wflow_crossing_gauge_locations(config, paths, submodel)
    return write_wflow_sfincs_gauge_locations(config, paths, submodel)


def sfincs_gauge_layer_matches(
    model_root: Path,
    submodel: dict,
    *,
    config: dict | None = None,
    paths: dict | None = None,
) -> bool:
    expected = {str(value) for value in submodel.get("sfincs_handoff_ids", ()) if value}
    if not expected:
        return True
    gauges_path = model_root / "staticgeoms" / "gauges_sfincs.geojson"
    if not gauges_path.exists():
        return False
    if config is not None and paths is not None:
        for source_path in sfincs_handoff_source_paths(config, paths, submodel):
            if source_path.stat().st_mtime > gauges_path.stat().st_mtime:
                return False
    try:
        gauges = gpd.read_file(gauges_path)
    except Exception:
        return False
    if "sfincs_handoff_id" in gauges:
        actual = set(gauges["sfincs_handoff_id"].dropna().astype(str))
    elif "name" in gauges:
        actual = set(gauges["name"].dropna().astype(str))
    else:
        return False
    return actual == expected


def sfincs_handoff_source_paths(config: dict, paths: dict, submodel: dict) -> list[Path]:
    """Return current SFINCS handoff source files feeding a Wflow submodel."""
    location_root = location_root_from_paths(paths)
    expected_domains = {str(value) for value in submodel.get("sfincs_domain_ids", ()) if value}
    expected_handoffs = {str(value) for value in submodel.get("sfincs_handoff_ids", ()) if value}
    manifest_value = (
        (config.get("sfincs_domain_set", {}) or {}).get("domain_manifest")
        or "data/sfincs/domains/domain_set.yaml"
    )
    manifest_path = resolve_location_path(location_root, manifest_value)
    candidates: list[Path] = []
    if manifest_path.exists():
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        for domain in payload.get("domains", []) or []:
            domain_id = str(domain.get("sfincs_domain_id", ""))
            handoff_ids = {str(value) for value in domain.get("handoff_source_ids", ()) if value}
            if expected_domains and domain_id not in expected_domains and not (expected_handoffs & handoff_ids):
                continue
            base_root = domain.get("base_model_root")
            if base_root:
                candidates.append(resolve_location_path(location_root, base_root) / "gis/wflow_handoff_sources.geojson")
    if not candidates:
        candidates = sorted(location_root.glob("data/sfincs/domains/*/base/gis/wflow_handoff_sources.geojson"))
    return [path for path in candidates if path.exists()]


def observation_gauge_layer_matches(model_root: Path, submodel: dict, config: dict) -> bool:
    submodel_sites = {str(value).zfill(8) for value in submodel.get("gauge_site_nos", ()) if value}
    reference_sites = configured_reference_gage_site_nos(config) & submodel_sites
    expected = reference_sites or submodel_sites
    if not expected:
        return True
    gauges_path = model_root / "staticgeoms" / "gauges_usgs.geojson"
    if not gauges_path.exists():
        return False
    try:
        gauges = gpd.read_file(gauges_path)
    except Exception:
        return False
    if "site_no" in gauges:
        actual = {str(value).zfill(8) for value in gauges["site_no"].dropna().astype(str)}
    elif "name" in gauges:
        actual = {str(value).zfill(8) for value in gauges["name"].dropna().astype(str)}
    else:
        return False
    return expected <= actual


def configured_reference_gage_site_nos(config: dict) -> set[str]:
    inland = config.get("inland_coupling", {}) or {}
    candidates = [
        ((inland.get("amplification", {}) or {}).get("primary_reference_gage")),
        ((inland.get("baseflow", {}) or {}).get("reference_gage")),
    ]
    return {
        str(value).strip().zfill(8)
        for value in candidates
        if value is not None and not pd.isna(value) and str(value).strip()
    }


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


def _drainage_area_km2(drainage_area_sqmi) -> float | None:
    if pd.isna(drainage_area_sqmi):
        return None
    return float(drainage_area_sqmi) * 2.589988110336
