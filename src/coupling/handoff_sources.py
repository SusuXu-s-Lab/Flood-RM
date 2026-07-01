from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

RESERVOIR_BOUNDARY_HANDOFF_MODES = {
    "sfincs_native_reservoir_boundary_inflow",
}

STREAM_BOUNDARY_HANDOFF_MODES = {
    "stream_boundary_intersection",
    "sfincs_stream_boundary",
    "boundary_stream_intersection",
    "sfincs_native_river_inflow",
} | RESERVOIR_BOUNDARY_HANDOFF_MODES
LEGACY_BOUNDARY_HANDOFF_MODES = {"sfincs_domain_boundary", "domain_boundary", "boundary"}


def handoff_location_mode(config: dict) -> str:
    return str(
        config.get("inland_coupling", {})
        .get("discharge_forcing", {})
        .get("handoff_location", "reviewed_gage")
    ).lower()


def uses_stream_boundary_handoff(config: dict) -> bool:
    return handoff_location_mode(config) in STREAM_BOUNDARY_HANDOFF_MODES


def _accepted_stream_boundary_artifact_placements(config: dict) -> set[str]:
    mode = handoff_location_mode(config)
    if mode == "sfincs_native_river_inflow":
        return {"sfincs_native_river_inflow"} | RESERVOIR_BOUNDARY_HANDOFF_MODES
    if mode in RESERVOIR_BOUNDARY_HANDOFF_MODES:
        return {mode}
    if mode in {"stream_boundary_intersection", "sfincs_stream_boundary", "boundary_stream_intersection"}:
        return {"stream_boundary_intersection", "sfincs_stream_boundary", "boundary_stream_intersection"}
    return {mode} if mode in STREAM_BOUNDARY_HANDOFF_MODES else set()


def crossing_handoff_sources(domain_plan) -> gpd.GeoDataFrame:
    """Return SFINCS discharge source points from a Wflow Domain Set Plan."""
    rows = []
    for submodel in domain_plan.submodels:
        points = submodel.get("handoff_points")
        if points:
            for point in points:
                rows.append(
                    {
                        "site_no": str(point["sfincs_handoff_id"]),
                        "sfincs_handoff_id": str(point["sfincs_handoff_id"]),
                        "wflow_submodel_id": str(submodel["wflow_submodel_id"]),
                        "sfincs_domain_id": str(point.get("sfincs_domain_id", "")),
                        "geometry": Point(float(point["lon"]), float(point["lat"])),
                    }
                )
            continue

        outlet_xy = (submodel.get("outlet_region") or submodel["region"])["subbasin"]
        handoff_id = next(
            (str(value) for value in submodel.get("sfincs_handoff_ids", ()) if value),
            str(submodel["wflow_submodel_id"]),
        )
        rows.append(
            {
                "site_no": handoff_id,
                "sfincs_handoff_id": handoff_id,
                "wflow_submodel_id": str(submodel["wflow_submodel_id"]),
                "sfincs_domain_id": next(
                    (str(value) for value in submodel.get("sfincs_domain_ids", ()) if value),
                    "",
                ),
                "geometry": Point(float(outlet_xy[0]), float(outlet_xy[1])),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def crossing_handoff_sources_from_wflow_domain_plan(
    config: dict,
    location_root: Path,
    *,
    plan_wflow_domain_set=None,
) -> gpd.GeoDataFrame:
    """Return crossing-derived SFINCS source points from the Wflow Domain Set plan."""
    if plan_wflow_domain_set is None:
        from coupling.wflow_domain_set import plan_wflow_domain_set

    plan = plan_wflow_domain_set(config, {"location_root": location_root})
    if plan.status != "ready":
        raise ValueError(f"Crossing-derived Wflow domain plan is not ready: {plan.status}: {plan.issues}")
    return crossing_handoff_sources(plan)


def _candidate_handoff_source_paths(config: dict, location_root: Path, location_path) -> list[Path]:
    paths = []
    manifest_value = config.get("sfincs_domain_set", {}).get(
        "domain_manifest",
        "data/sfincs/domains/domain_set.yaml",
    )
    manifest_path = location_path(location_root, manifest_value)
    if manifest_path.exists():
        import yaml

        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        for domain in manifest.get("domains", []):
            base_root = location_path(location_root, domain.get("base_model_root", ""))
            paths.append(base_root / "gis/wflow_handoff_sources.geojson")

    domains_root = location_path(
        location_root,
        config.get("sfincs_domain_set", {}).get("domains_root", "data/sfincs/domains"),
    )
    if domains_root.exists():
        paths.extend(sorted(domains_root.glob("*/base/gis/wflow_handoff_sources.geojson")))
    paths.append(location_root / "data/sfincs/base/gis/wflow_handoff_sources.geojson")
    return paths


def read_stream_boundary_handoff_locations(
    config: dict,
    location_root: Path,
    handoff_ids: set[str],
    *,
    location_path,
) -> gpd.GeoDataFrame | None:
    if not uses_stream_boundary_handoff(config):
        return None

    locations = read_stream_boundary_handoff_location_artifacts(
        config,
        location_root,
        location_path=location_path,
    )
    if locations is None:
        return None

    locations = locations[locations["sfincs_handoff_id"].astype(str).isin(handoff_ids)].copy()
    if locations.empty:
        return None
    missing = sorted(handoff_ids - set(locations["sfincs_handoff_id"].astype(str)))
    if missing:
        raise ValueError(
            "SFINCS boundary handoff source artifacts are missing IDs needed by Wflow: "
            + ", ".join(missing)
        )
    return locations


def read_stream_boundary_handoff_location_artifacts(
    config: dict,
    location_root: Path,
    *,
    location_path,
) -> gpd.GeoDataFrame | None:
    """Read all generated SFINCS stream-boundary handoff source artifacts."""
    if not uses_stream_boundary_handoff(config):
        return None
    frames = []
    seen = set()
    for path in _candidate_handoff_source_paths(config, location_root, location_path):
        path_key = path.resolve()
        if path_key in seen or not path.exists():
            continue
        seen.add(path_key)
        frame = gpd.read_file(path)
        if frame.empty or "sfincs_handoff_id" not in frame:
            continue
        if "handoff_placement" in frame:
            accepted_placements = _accepted_stream_boundary_artifact_placements(config)
            frame = frame[
                frame["handoff_placement"].fillna("").astype(str).str.lower().isin(accepted_placements)
            ].copy()
        else:
            frame = frame.iloc[[]].copy()
        frames.append(frame)

    if not frames:
        return None

    locations = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs)
    if locations.empty:
        return None
    locations = locations.drop_duplicates(subset=["sfincs_domain_id", "sfincs_handoff_id"]).copy()
    return locations
