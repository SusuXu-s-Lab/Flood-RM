from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy import ndimage
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point
from shapely.ops import nearest_points, unary_union

from sfincs_runs.build_base.infiltration import setup_hydromt_infiltration, validate_physics
from wflow_runs.handoff_locations import (
    LEGACY_BOUNDARY_HANDOFF_MODES,
    RESERVOIR_BOUNDARY_HANDOFF_MODES,
    STREAM_BOUNDARY_HANDOFF_MODES,
    crossing_handoff_sources,
    handoff_location_mode,
    uses_stream_boundary_handoff,
)

DEFAULT_OUTFLOW_ELEVATION_QUANTILE = 0.05
HYDROMT_SFINCS_GEOM_STYLE = {
    "rivers": dict(linestyle="-", linewidth=1.0, color="darkblue"),
    "rivers_inflow": dict(linestyle=":", linewidth=1.0, color="darkblue"),
    "rivers_outflow": dict(linestyle=":", linewidth=1.0, color="darkgreen"),
}
_GENERATED_NOTICE = (
    "# GENERATED FILE — do not edit. Overwritten when {source} runs.\n"
    "# Source of truth is the location config and the code that produces this file.\n"
)


@dataclass(frozen=True)
class InlandSfincsBasePlan:
    base_model_root: Path
    region: Path
    dem: Path
    landcover: Path
    hsg: Path
    ksat: Path
    handoff_manifest: Path
    model_crs: str
    grid_resolution_m: float
    ready_to_build: bool
    missing_inputs: tuple[Path, ...]
    built: bool

    def summary_rows(self):
        return [
            {"item": "base_model_root", "value": self.base_model_root.as_posix(), "ready": self.built},
            {"item": "region", "value": self.region.as_posix(), "ready": self.region.exists()},
            {"item": "dem", "value": self.dem.as_posix(), "ready": self.dem.exists()},
            {"item": "landcover", "value": self.landcover.as_posix(), "ready": self.landcover.exists()},
            {"item": "hsg", "value": self.hsg.as_posix(), "ready": self.hsg.exists()},
            {"item": "ksat", "value": self.ksat.as_posix(), "ready": self.ksat.exists()},
            {"item": "handoff_manifest", "value": self.handoff_manifest.as_posix(), "ready": self.handoff_manifest.exists()},
        ]


@dataclass(frozen=True)
class InlandSfincsDomainSetPlan:
    status: str
    manifest: Path
    domains: tuple[dict, ...]
    domain_count: int
    handoff_count: int
    issues: tuple[str, ...]

    def summary_rows(self):
        if not self.domains:
            return [{"status": self.status, "issue": issue} for issue in self.issues]
        return [
            {
                "sfincs_domain_id": domain["sfincs_domain_id"],
                "region": domain["region"].as_posix(),
                "base_model_root": domain["base_model_root"].as_posix(),
                "exposure_area_km2": domain["exposure_area_km2"],
                "handoff_source_ids": ", ".join(domain["handoff_source_ids"]),
                "wflow_submodel_ids": ", ".join(domain["wflow_submodel_ids"]),
            }
            for domain in self.domains
        ]


def meaningful_model_files(path):
    path = Path(path)
    if not path.exists():
        return []
    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.name != ".gitkeep" and not file.name.endswith("~")
    )


def is_built_sfincs_base(path):
    path = Path(path)
    files = {file.name for file in meaningful_model_files(path)}
    if "sfincs.inp" not in files:
        return False
    return bool(files & {"sfincs.dep", "sfincs.msk", "sfincs.subgrid", "subgrid.nc", "sfincs.ind"})


def sfincs_grid_resolution_matches(path, expected_resolution_m: float, *, tolerance: float = 1.0e-6) -> bool:
    """Return whether an existing SFINCS input file uses the expected regular-grid spacing."""
    path = Path(path)
    inp_path = path if path.name == "sfincs.inp" else path / "sfincs.inp"
    if not inp_path.exists():
        return False

    values = {}
    for line in inp_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip().lower()
        if key not in {"dx", "dy"}:
            continue
        try:
            values[key] = float(value.strip().split()[0])
        except (IndexError, ValueError):
            return False

    expected = float(expected_resolution_m)
    return all(abs(values.get(axis, np.nan) - expected) <= tolerance for axis in ("dx", "dy"))


def is_built_wflow_base(path):
    path = Path(path)
    files = {file.name for file in meaningful_model_files(path)}
    return bool(files & {"wflow_sbm.toml", "staticmaps.nc", "staticgeoms.nc"})


def plan_inland_sfincs_base(config, paths):
    location_root = _location_root(paths)
    sfincs_cfg = config.get("sfincs", {})
    static_sources = config.get("static_sources", {})
    infiltration = config.get("inland_coupling", {}).get("infiltration", {})
    handoff_value = (
        config.get("inland_coupling", {})
        .get("discharge_forcing", {})
        .get("handoff_manifest")
        or config.get("wflow", {})
        .get("handoff", {})
        .get("manifest", "data/wflow/domain_set_handoff.yaml")
    )
    base_model_root = _location_path(location_root, config.get("paths", {}).get("base_model_root", "data/sfincs/base"))
    plan = InlandSfincsBasePlan(
        base_model_root=base_model_root,
        region=_location_path(location_root, config.get("grid_footprint", {}).get("source", "data/static/aoi/study_area.geojson")),
        dem=_location_path(location_root, static_sources.get("terrain", {}).get("output", "data/static/processed/dem_region_setup.tif")),
        landcover=_location_path(
            location_root,
            static_sources.get("landcover", {}).get("output", "data/static/processed/landcover_region_setup.tif"),
        ),
        hsg=_location_path(location_root, infiltration.get("hsg", static_sources.get("ssurgo", {}).get("hsg_output", "data/static/soils/hsg.tif"))),
        ksat=_location_path(
            location_root,
            infiltration.get("ksat", static_sources.get("ssurgo", {}).get("ksat_output", "data/static/soils/ksat_mmhr.tif")),
        ),
        handoff_manifest=_location_path(location_root, handoff_value),
        model_crs=str(sfincs_cfg.get("model_crs", config.get("project", {}).get("model_crs", "EPSG:4326"))),
        grid_resolution_m=float(sfincs_cfg.get("grid_resolution_m", sfincs_cfg.get("resolution_m", 30))),
        ready_to_build=False,
        missing_inputs=(),
        built=is_built_sfincs_base(base_model_root),
    )
    required = (plan.region, plan.dem, plan.landcover, plan.hsg, plan.ksat, plan.handoff_manifest)
    missing = tuple(path for path in required if not path.exists())
    return InlandSfincsBasePlan(
        base_model_root=plan.base_model_root,
        region=plan.region,
        dem=plan.dem,
        landcover=plan.landcover,
        hsg=plan.hsg,
        ksat=plan.ksat,
        handoff_manifest=plan.handoff_manifest,
        model_crs=plan.model_crs,
        grid_resolution_m=plan.grid_resolution_m,
        ready_to_build=not missing,
        missing_inputs=missing,
        built=plan.built,
    )


def plan_inland_sfincs_domain_set(config, paths):
    """Plan SFINCS hydraulic domains from disconnected SMART-DS exposure regions."""
    location_root = _location_root(paths)
    domain_set = config.get("sfincs_domain_set", {})
    manifest = _location_path(location_root, domain_set.get("domain_manifest", "data/sfincs/domains/domain_set.yaml"))
    exposure_path = _location_path(
        location_root,
        domain_set.get("source", config.get("grid_footprint", {}).get("source", "data/static/aoi/study_area.geojson")),
    )
    model_crs = str(config.get("sfincs", {}).get("model_crs", config.get("project", {}).get("model_crs", "EPSG:4326")))
    issues = []
    if not exposure_path.exists():
        issues.append(f"missing exposure footprint: {exposure_path}")
        return InlandSfincsDomainSetPlan("missing_inputs", manifest, (), 0, 0, tuple(issues))

    exposure_source = gpd.read_file(exposure_path)
    if exposure_source.crs is None:
        exposure_source = exposure_source.set_crs("EPSG:4326")
    exposure = exposure_source.to_crs(model_crs)
    component_records = _exposure_component_records(exposure)
    source_component_records = _exposure_component_records(exposure_source)
    if len(source_component_records) == len(component_records):
        for record, source_record in zip(component_records, source_component_records):
            record["source_geometry"] = source_record["geometry"]
            record["source_crs"] = exposure_source.crs
    if not component_records:
        issues.append(f"exposure footprint has no polygon components: {exposure_path}")
        return InlandSfincsDomainSetPlan("missing_inputs", manifest, (), 0, 0, tuple(issues))
    if domain_set.get("allow_multiple_domains") is False:
        source_geometries = [
            record.get("source_geometry")
            for record in component_records
            if record.get("source_geometry") is not None
        ]
        component_records = [
            {
                "geometry": unary_union([record["geometry"] for record in component_records]),
                "source_geometry": unary_union(source_geometries) if source_geometries else None,
                "source_crs": exposure_source.crs,
                "subregion_id": None,
                "component_index": 0,
            }
        ]
    else:
        component_records = [
            {**record, "component_index": index}
            for index, record in enumerate(component_records)
        ]

    project_name = str(config.get("project", {}).get("name", location_root.name))
    domain_count = len(component_records)
    planned_records = [
        {
            **record,
            "sfincs_domain_id": _auto_domain_id(
                project_name,
                index,
                domain_count,
                subregion_id=record.get("subregion_id"),
            ),
        }
        for index, record in enumerate(component_records)
    ]
    include_domain_ids = _included_domain_ids(domain_set)
    if include_domain_ids:
        found_domain_ids = {record["sfincs_domain_id"] for record in planned_records}
        planned_records = [
            record
            for record in planned_records
            if record["sfincs_domain_id"] in include_domain_ids
        ]
        missing_domain_ids = sorted(include_domain_ids - found_domain_ids)
        if missing_domain_ids:
            issues.append("configured SFINCS domains were not found: " + ", ".join(missing_domain_ids))
    if not planned_records:
        issues.append("no SFINCS exposure components remain after include_domain_ids filtering")
        return InlandSfincsDomainSetPlan("missing_inputs", manifest, (), 0, 0, tuple(issues))

    component_records = planned_records
    components = [record["geometry"] for record in component_records]

    try:
        handoff = _accepted_handoff_gages(config, location_root).to_crs(model_crs)
    except (FileNotFoundError, ValueError) as exc:
        issues.append(str(exc))
        handoff = gpd.GeoDataFrame(
            columns=["site_no", "sfincs_handoff_id", "wflow_submodel_id", "geometry"],
            geometry="geometry",
            crs=model_crs,
        )
    domains_root = _location_path(location_root, domain_set.get("domains_root", "data/sfincs/domains"))
    domain_ids = [record["sfincs_domain_id"] for record in component_records]
    assignments = _assign_handoffs_to_components(handoff, components, domain_ids=domain_ids)
    corridor_buffer_m = float(domain_set.get("handoff_corridor_buffer_m", 300.0))
    region_geometry = str(domain_set.get("region_geometry", "component")).lower()
    use_stream_boundary_handoffs = uses_stream_boundary_handoff(config)
    domains = []
    for index, record in enumerate(component_records):
        component = record["geometry"]
        domain_id = record["sfincs_domain_id"]
        assigned = assignments.get(index, handoff.iloc[[]])
        # Stream-boundary handoffs keep the SFINCS coverage box tied to the
        # SMART-DS footprint; reviewed gages define Wflow outlets, not SFINCS area.
        geometry = (
            component
            if use_stream_boundary_handoffs
            else _domain_geometry_with_handoffs(component, assigned, corridor_buffer_m)
        )
        if region_geometry in {"bbox", "bounding_box", "envelope"}:
            source_geometry = record.get("source_geometry")
            source_crs = record.get("source_crs")
            if source_geometry is not None and source_crs is not None:
                geometry = gpd.GeoSeries([source_geometry.envelope], crs=source_crs).to_crs(model_crs).iloc[0]
            else:
                geometry = geometry.envelope
        domain_root = domains_root / domain_id
        handoff_source_ids = sorted(assigned["sfincs_handoff_id"].astype(str).tolist()) if not assigned.empty else []
        wflow_submodel_ids = _sfincs_domain_wflow_submodel_ids(config, location_root, assigned)
        domains.append(
            {
                "sfincs_domain_id": domain_id,
                "region": domain_root / "region.geojson",
                "base_model_root": domain_root / "base",
                "scenarios_root": domain_root / "scenarios",
                "exposure_component_index": record["component_index"],
                "exposure_subregion_id": record.get("subregion_id"),
                "exposure_area_km2": float(component.area / 1.0e6),
                "handoff_source_ids": handoff_source_ids,
                "wflow_submodel_ids": wflow_submodel_ids,
                "geometry": geometry,
            }
        )
    if handoff.empty:
        issues.append("no accepted Wflow-SFINCS handoff gages")
    status = "ready" if not issues else "needs_review"
    return InlandSfincsDomainSetPlan(status, manifest, tuple(domains), len(domains), int(len(handoff)), tuple(issues))


def write_inland_sfincs_domain_set_manifest(plan, config, paths):
    """Write the planned SFINCS domain geometries and domain-set manifest."""
    location_root = _location_root(paths)
    manifest = _location_path(location_root, plan.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    model_crs = str(config.get("sfincs", {}).get("model_crs", config.get("project", {}).get("model_crs", "EPSG:4326")))
    domain_rows = []
    for domain in plan.domains:
        region = _location_path(location_root, domain["region"])
        region.parent.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame(
            {
                "sfincs_domain_id": [domain["sfincs_domain_id"]],
                "exposure_component_index": [domain["exposure_component_index"]],
                "exposure_subregion_id": [domain.get("exposure_subregion_id")],
                "handoff_source_ids": [", ".join(domain["handoff_source_ids"])],
                "wflow_submodel_ids": [", ".join(domain["wflow_submodel_ids"])],
            },
            geometry=[domain["geometry"]],
            crs=model_crs,
        ).to_crs("EPSG:4326").to_file(region, driver="GeoJSON")
        domain_rows.append(
            {
                "sfincs_domain_id": domain["sfincs_domain_id"],
                "region": region.as_posix(),
                "base_model_root": _location_path(location_root, domain["base_model_root"]).as_posix(),
                "scenarios_root": _location_path(location_root, domain["scenarios_root"]).as_posix(),
                "exposure_component_index": int(domain["exposure_component_index"]),
                "exposure_subregion_id": domain.get("exposure_subregion_id"),
                "exposure_area_km2": float(domain["exposure_area_km2"]),
                "handoff_source_ids": list(domain["handoff_source_ids"]),
                "wflow_submodel_ids": list(domain["wflow_submodel_ids"]),
            }
        )
    payload = {
        "status": plan.status,
        "event_catalog_scope": config.get("sfincs_domain_set", {}).get("event_catalog_scope", "shared_across_domain_set"),
        "evaluation_merge": config.get("sfincs_domain_set", {}).get("evaluation_merge", "max_depth_per_asset_with_source_domain"),
        "domain_count": plan.domain_count,
        "handoff_count": plan.handoff_count,
        "issues": list(plan.issues),
        "domains": domain_rows,
    }
    manifest.write_text(
        _GENERATED_NOTICE.format(source="the SFINCS domain-set build")
        + yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return manifest


def write_inland_sfincs_handoff_locations(
    config,
    paths,
    *,
    output=None,
    handoff_source_ids=None,
    domain_region=None,
    sfincs_domain_id=None,
):
    """Write SFINCS source-point locations from reviewed gages or stream crossings."""
    location_root = _location_root(paths)
    out_path = _location_path(
        location_root,
        output or "data/sfincs/base/gis/wflow_handoff_sources.geojson",
    )
    handoff = _accepted_handoff_gages(config, location_root)
    if handoff_source_ids is not None:
        wanted = {str(value) for value in handoff_source_ids}
        handoff = handoff[handoff["sfincs_handoff_id"].astype(str).isin(wanted)].copy()
        missing = sorted(wanted - set(handoff["sfincs_handoff_id"].astype(str)))
        if missing:
            raise ValueError("Requested handoff source ids are not accepted handoff sources: " + ", ".join(missing))
    handoff = handoff.to_crs(config.get("sfincs", {}).get("model_crs", config.get("project", {}).get("model_crs", "EPSG:4326")))
    handoff = _snap_handoff_locations_to_domain_boundary(
        handoff,
        config,
        paths,
        domain_region=domain_region,
    )
    if sfincs_domain_id is not None:
        handoff["sfincs_domain_id"] = str(sfincs_domain_id)
    return _write_inland_sfincs_handoff_artifacts(
        handoff,
        config,
        paths,
        out_path,
        domain_region=domain_region,
    )


def write_inland_sfincs_handoff_locations_from_wflow_rivers(
    config,
    paths,
    *,
    output=None,
    domain_region,
    sfincs_domain_id,
    wflow_submodel_id,
    min_streamorder=None,
):
    """Write SFINCS handoff sources from built Wflow river/SFINCS-boundary intersections."""
    if domain_region in (None, ""):
        raise ValueError("domain_region is required when deriving handoffs from built Wflow rivers")

    location_root = _location_root(paths)
    out_path = _location_path(
        location_root,
        output or "data/sfincs/base/gis/wflow_handoff_sources.geojson",
    )
    model_crs = config.get("sfincs", {}).get(
        "model_crs",
        config.get("project", {}).get("model_crs", "EPSG:4326"),
    )
    seed = gpd.GeoDataFrame(
        {"wflow_submodel_id": [str(wflow_submodel_id)]},
        geometry=[Point(0.0, 0.0)],
        crs="EPSG:4326",
    )
    rivers = _wflow_native_river_lines(config, location_root, seed)
    if rivers.empty:
        raise FileNotFoundError(
            "Built Wflow native rivers are required before deriving SFINCS handoff sources: "
            f"{_location_path(location_root, config.get('wflow', {}).get('base_model_root', 'data/wflow/base'))}"
            f"/{wflow_submodel_id}/staticgeoms/rivers.geojson"
        )

    region_path = _location_path(location_root, domain_region)
    if not region_path.exists():
        raise FileNotFoundError(region_path)
    region = gpd.read_file(region_path)
    if region.crs is None:
        region = region.set_crs(model_crs)
    rivers = rivers.to_crs(region.crs)
    domain_geometry = region.geometry.union_all()
    boundary = domain_geometry.boundary

    streamorder_column = _streamorder_column(rivers)
    if streamorder_column is None:
        raise ValueError("Built Wflow river geometry has no stream-order column (strord/streamorder)")
    explicit_min_streamorder = _explicit_min_handoff_streamorder(config, min_streamorder)
    minimum_order = _configured_min_handoff_streamorder(config, min_streamorder)
    min_uparea_km2 = _configured_min_handoff_uparea(config)
    staticmaps_path = _wflow_staticmaps_path(config, location_root, str(wflow_submodel_id))
    staticmaps = xr.open_dataset(staticmaps_path) if staticmaps_path.exists() else None
    stream_order = pd.to_numeric(rivers[streamorder_column], errors="coerce")
    if staticmaps is None:
        selected_rivers = rivers.loc[stream_order >= minimum_order].copy()
    elif explicit_min_streamorder is None:
        selected_rivers = rivers.copy()
    else:
        selected_rivers = rivers.loc[stream_order >= explicit_min_streamorder].copy()
    selected_rivers["stream_order"] = stream_order.loc[selected_rivers.index].astype(float)

    rows = []
    try:
        for river_index, river in selected_rivers.iterrows():
            geometry = river.geometry
            if geometry is None or geometry.is_empty:
                continue
            crossing = _first_directed_domain_inflow_crossing(geometry, domain_geometry)
            if crossing is None:
                continue
            point, measure = crossing
            uparea_km2 = _sample_wflow_staticmap_value(
                staticmaps,
                "meta_upstream_area",
                point,
                region.crs,
            )
            if staticmaps is not None and uparea_km2 is not None and uparea_km2 < min_uparea_km2:
                continue
            sampled_stream_order = _sample_wflow_staticmap_value(
                staticmaps,
                "meta_streamorder",
                point,
                region.crs,
            )
            effective_stream_order = sampled_stream_order if sampled_stream_order is not None else river["stream_order"]
            if explicit_min_streamorder is not None and float(effective_stream_order) < float(explicit_min_streamorder):
                continue

            handoff_id = f"{sfincs_domain_id}_inflow_{len(rows) + 1:02d}"
            rows.append(
                {
                    "site_no": handoff_id,
                    "sfincs_handoff_id": handoff_id,
                    "wflow_submodel_id": str(wflow_submodel_id),
                    "sfincs_domain_id": str(sfincs_domain_id),
                    "source_gage_x": float(point.x),
                    "source_gage_y": float(point.y),
                    "handoff_placement": "stream_boundary_intersection",
                    "handoff_location_review_status": "hydromt_wflow_native_stream_boundary_intersection",
                    "stream_boundary_river_index": int(river_index),
                    "stream_boundary_river_id": _river_identifier(river),
                    "stream_boundary_river_source": str(river.get("river_geometry_source", "hydromt_wflow_setup_rivers")),
                    "stream_boundary_river_source_path": str(river.get("river_geometry_source_path", "")),
                    "stream_boundary_measure_m": float(measure),
                    "stream_gage_measure_m": float(measure),
                    "stream_snap_distance_m": 0.0,
                    "stream_boundary_candidate_count": int(len(selected_rivers)),
                    "stream_boundary_upstream_candidate_count": int(len(rows) + 1),
                    "stream_order": _handoff_numeric_value(effective_stream_order),
                    "uparea_km2": None if uparea_km2 is None else float(uparea_km2),
                    "geometry": point,
                }
            )
    finally:
        if staticmaps is not None:
            staticmaps.close()
    if not rows:
        raise ValueError(
            f"No built Wflow river/SFINCS-boundary inflow intersections at stream order >= {minimum_order}"
        )

    handoff = gpd.GeoDataFrame(rows, geometry="geometry", crs=region.crs).to_crs(model_crs)
    handoff = _add_reservoir_boundary_handoff_sources(
        handoff,
        config,
        paths,
        model=None,
        sfincs_domain_id=str(sfincs_domain_id),
        wflow_submodel_id=str(wflow_submodel_id),
        model_crs=model_crs,
        domain_region=domain_region,
    )
    return _write_inland_sfincs_handoff_artifacts(
        handoff,
        config,
        paths,
        out_path,
        domain_region=domain_region,
    )


def _write_inland_sfincs_handoff_artifacts(
    handoff: gpd.GeoDataFrame,
    config,
    paths,
    out_path: Path,
    *,
    domain_region=None,
):
    handoff = handoff.sort_values(["sfincs_domain_id", "sfincs_handoff_id", "site_no"]).reset_index(drop=True)
    handoff = _deduplicate_handoff_source_locations(
        handoff,
        tolerance_m=_handoff_source_dedup_tolerance_m(config),
    )
    if "index" in handoff.columns:
        handoff = handoff.drop(columns=["index"])
    handoff.insert(0, "index", range(1, len(handoff) + 1))
    handoff["name"] = handoff["sfincs_handoff_id"].astype(str)
    keep = [
        "index",
        "name",
        "site_no",
        "sfincs_handoff_id",
        "wflow_submodel_id",
        "sfincs_domain_id",
        "source_gage_x",
        "source_gage_y",
        "handoff_placement",
        "handoff_location_review_status",
        "stream_boundary_river_index",
        "stream_boundary_river_id",
        "stream_boundary_river_source",
        "stream_boundary_river_source_path",
        "stream_boundary_measure_m",
        "stream_gage_measure_m",
        "stream_snap_distance_m",
        "stream_boundary_candidate_count",
        "stream_boundary_upstream_candidate_count",
        "stream_order",
        "uparea_km2",
        "geometry",
    ]
    keep = [column for column in keep if column in handoff.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handoff[keep].to_file(out_path, driver="GeoJSON")
    rivers = wflow_handoff_rivers_inflow_geoms(
        config,
        paths,
        handoff,
        domain_region=domain_region,
        model_crs=handoff.crs,
    )
    _write_sfincs_rivers_inflow_geoms(rivers, out_path.parent / "rivers_inflow.geojson")
    return {
        "handoff_locations": out_path,
        "rivers_inflow": out_path.parent / "rivers_inflow.geojson",
        "source_point_count": int(len(handoff)),
        "rivers_inflow_count": int(len(rivers)),
        "sfincs_domain_count": int(handoff["sfincs_domain_id"].nunique()),
        "wflow_submodel_count": int(handoff["wflow_submodel_id"].nunique()),
    }


def _configured_min_handoff_streamorder(config, value) -> int:
    if value is not None:
        return int(value)
    crossings = ((config.get("wflow", {}) or {}).get("domain_set", {}) or {}).get("crossings", {}) or {}
    forcing = ((config.get("inland_coupling", {}) or {}).get("discharge_forcing", {}) or {})
    return int(
        crossings.get(
            "min_streamorder",
            forcing.get("min_handoff_streamorder", 2),
        )
    )


def _explicit_min_handoff_streamorder(config, value) -> int | None:
    if value is not None:
        return int(value)
    crossings = ((config.get("wflow", {}) or {}).get("domain_set", {}) or {}).get("crossings", {}) or {}
    forcing = ((config.get("inland_coupling", {}) or {}).get("discharge_forcing", {}) or {})
    if "min_streamorder" in crossings:
        return int(crossings["min_streamorder"])
    if "min_handoff_streamorder" in forcing:
        return int(forcing["min_handoff_streamorder"])
    return None


def _configured_min_handoff_uparea(config) -> float:
    crossings = ((config.get("wflow", {}) or {}).get("domain_set", {}) or {}).get("crossings", {}) or {}
    forcing = ((config.get("inland_coupling", {}) or {}).get("discharge_forcing", {}) or {})
    return float(
        crossings.get(
            "min_uparea_km2",
            forcing.get("min_handoff_uparea_km2", forcing.get("river_upa_km2", 5.0)),
        )
    )


def _streamorder_column(rivers: gpd.GeoDataFrame) -> str | None:
    for column in ("strord", "streamorder", "stream_order", "meta_streamorder"):
        if column in rivers.columns:
            return column
    return None


def _wflow_staticmaps_path(config, location_root: Path, submodel_id: str) -> Path:
    base_root = _location_path(
        location_root,
        config.get("wflow", {}).get("base_model_root", "data/wflow/base"),
    )
    submodel_path = base_root / str(submodel_id) / "staticmaps.nc"
    if submodel_path.exists():
        return submodel_path
    return base_root / "staticmaps.nc"


def _first_directed_domain_inflow_crossing(river: LineString, domain_geometry) -> tuple[Point, float] | None:
    """Return the first external-to-internal SFINCS boundary crossing for a directed reach."""
    if river is None or river.is_empty or not hasattr(river, "coords"):
        return None
    coords = list(river.coords)
    if not coords or domain_geometry.covers(Point(coords[0])):
        return None

    crossings = []
    for point in _intersection_points(river.intersection(domain_geometry.boundary)):
        measure = float(river.project(point))
        before_inside, after_inside = _domain_state_around_measure(river, domain_geometry, measure)
        if before_inside is False and after_inside is True:
            crossings.append((measure, point))
    if not crossings:
        return None
    measure, point = sorted(crossings, key=lambda item: item[0])[0]
    return point, measure


def _domain_state_around_measure(river: LineString, domain_geometry, measure: float) -> tuple[bool, bool]:
    epsilon = min(120.0, max(float(river.length) * 0.01, 1.0))
    before = river.interpolate(max(0.0, float(measure) - epsilon))
    after = river.interpolate(min(float(river.length), float(measure) + epsilon))
    return bool(domain_geometry.covers(before)), bool(domain_geometry.covers(after))


def _sample_wflow_staticmap_value(staticmaps, variable: str, point: Point, point_crs) -> float | None:
    if staticmaps is None or variable not in staticmaps or point is None or point.is_empty:
        return None
    x_name, y_name = _staticmap_xy_names(staticmaps)
    if x_name is None or y_name is None:
        return None
    sample_point = _point_in_staticmap_crs(point, point_crs, staticmaps, x_name=x_name, y_name=y_name)
    value = staticmaps[variable].sel({x_name: float(sample_point.x), y_name: float(sample_point.y)}, method="nearest").values
    try:
        number = float(np.asarray(value).item())
    except Exception:
        return None
    if not np.isfinite(number):
        return None
    return number


def _staticmap_xy_names(staticmaps) -> tuple[str | None, str | None]:
    x_name = next((name for name in ("longitude", "lon", "x") if name in staticmaps.coords or name in staticmaps.dims), None)
    y_name = next((name for name in ("latitude", "lat", "y") if name in staticmaps.coords or name in staticmaps.dims), None)
    return x_name, y_name


def _point_in_staticmap_crs(point: Point, point_crs, staticmaps, *, x_name: str, y_name: str) -> Point:
    static_crs = _staticmap_crs(staticmaps, x_name=x_name, y_name=y_name)
    if static_crs is None or point_crs is None or str(static_crs) == str(point_crs):
        return point
    return gpd.GeoSeries([point], crs=point_crs).to_crs(static_crs).iloc[0]


def _staticmap_crs(staticmaps, *, x_name: str, y_name: str):
    raster = getattr(staticmaps, "raster", None)
    crs = getattr(raster, "crs", None)
    if crs is not None:
        return crs
    rio = getattr(staticmaps, "rio", None)
    crs = getattr(rio, "crs", None)
    if crs is not None:
        return crs
    if x_name in {"longitude", "lon"} and y_name in {"latitude", "lat"}:
        return "EPSG:4326"
    return None


def _handoff_numeric_value(value):
    number = float(value)
    return int(number) if number.is_integer() else number


def _handoff_source_dedup_tolerance_m(config) -> float:
    forcing = (config.get("inland_coupling", {}) or {}).get("discharge_forcing", {}) or {}
    if "handoff_dedup_distance_m" in forcing:
        return float(forcing["handoff_dedup_distance_m"])
    grid_resolution = float(config.get("sfincs", {}).get("grid_resolution_m", 60.0))
    return max(grid_resolution * 0.5, 1.0)


def _deduplicate_handoff_source_locations(
    handoff: gpd.GeoDataFrame,
    *,
    tolerance_m: float = 0.0,
) -> gpd.GeoDataFrame:
    """Collapse multiple handoff IDs that resolve to the same SFINCS source point."""
    if handoff.empty:
        return handoff
    if tolerance_m > 0:
        metric = handoff
        if metric.crs is None:
            metric = metric.set_crs("EPSG:4326")
        if metric.crs is not None and metric.crs.is_geographic:
            metric = metric.to_crs("EPSG:3857")
        keep = []
        kept_geometries = []
        for index, geometry in metric.geometry.items():
            if geometry is None or geometry.is_empty:
                keep.append(index)
                continue
            if any(float(geometry.distance(existing)) <= float(tolerance_m) for existing in kept_geometries):
                continue
            keep.append(index)
            kept_geometries.append(geometry)
        return handoff.loc[keep].reset_index(drop=True)
    source_keys = []
    for geometry in handoff.geometry:
        if geometry is None or geometry.is_empty:
            source_keys.append(None)
        elif geometry.geom_type == "Point":
            source_keys.append((round(float(geometry.x), 3), round(float(geometry.y), 3)))
        else:
            source_keys.append(geometry.wkb_hex)
    deduped = handoff.copy()
    deduped["_source_location_key"] = source_keys
    deduped = deduped.drop_duplicates("_source_location_key", keep="first")
    return deduped.drop(columns=["_source_location_key"]).reset_index(drop=True)


def create_handoffs(
    model,
    config,
    paths,
    *,
    output=None,
    sfincs_domain_id,
    wflow_submodel_id=None,
    hydrography=None,
    river_upa=None,
    river_len=None,
    buffer=None,
    river_width=0,
    first_index=1,
    handoff_source_ids=None,
    domain_region=None,
) -> gpd.GeoDataFrame:
    """Create SFINCS source points with HydroMT-SFINCS native river inflow tooling.

    The documented HydroMT coupling order is: SFINCS places discharge ``src`` points where
    river centerlines enter the SFINCS domain, then HydroMT-Wflow uses those ``src`` points
    as Wflow gauges. This helper wraps ``SfincsRivers.create_river_inflow`` and writes the
    generated locations to the project handoff artifact consumed by the Wflow build.
    """
    location_root = _location_root(paths)
    forcing = config.get("inland_coupling", {}).get("discharge_forcing", {})
    crossings = config.get("wflow", {}).get("domain_set", {}).get("crossings", {})
    hydrography = hydrography or forcing.get("hydrography", "us_hydrography_basemap")
    river_upa = float(river_upa if river_upa is not None else forcing.get("river_upa_km2", crossings.get("min_uparea_km2", 5.0)))
    river_len = float(river_len if river_len is not None else forcing.get("river_len_m", 500.0))
    buffer = float(buffer if buffer is not None else forcing.get("river_inflow_buffer_m", 200.0))
    max_source_points = forcing.get("max_source_points")
    out_path = _location_path(location_root, output or "data/sfincs/base/gis/wflow_handoff_sources.geojson")

    if not hasattr(model, "rivers"):
        raise ValueError("HydroMT-SFINCS model has no rivers component for native river inflow setup")
    _register_sfincs_hydrography_source(model, config, paths, hydrography)
    model.rivers.create_river_inflow(
        hydrography=hydrography,
        buffer=buffer,
        river_upa=river_upa,
        river_len=river_len,
        river_width=river_width,
        keep_rivers_geom=True,
        merge=False,
        first_index=int(first_index),
        src_type="inflow",
    )

    src = model.discharge_points.gdf
    if src is None or src.empty:
        raise ValueError(
            "HydroMT-SFINCS native river inflow created no discharge source points; "
            "review the SFINCS domain, hydrography source, river_upa, and buffer settings."
        )
    src = src.copy()
    model_crs = getattr(model, "crs", None) or config.get("sfincs", {}).get(
        "model_crs",
        config.get("project", {}).get("model_crs", "EPSG:4326"),
    )
    if src.crs is None:
        src = src.set_crs(model_crs)
    else:
        src = src.to_crs(model_crs)
    src = src.reset_index(drop=True)
    if max_source_points is not None:
        max_source_points = int(max_source_points)
        if max_source_points < 1:
            raise ValueError("inland_coupling.discharge_forcing.max_source_points must be positive when set")
        if len(src) > max_source_points:
            sort_columns = ["uparea"] if "uparea" in src.columns else []
            src = (
                src.sort_values(sort_columns, ascending=False)
                if sort_columns
                else src
            )
            src = src.head(max_source_points).reset_index(drop=True)
    if "index" in src.columns:
        src = src.drop(columns=["index"])
    src.insert(0, "index", range(int(first_index), int(first_index) + len(src)))

    sfincs_domain_id = str(sfincs_domain_id)
    if wflow_submodel_id is None:
        wflow_submodel_id = _single_domain_wflow_submodel_id(config, location_root, sfincs_domain_id)
    wflow_submodel_id = "" if wflow_submodel_id is None else str(wflow_submodel_id)
    ids = [f"{sfincs_domain_id}_inflow_{idx:02d}" for idx in range(1, len(src) + 1)]
    src["name"] = ids
    src["site_no"] = ids
    src["sfincs_handoff_id"] = ids
    if handoff_source_ids is not None and handoff_location_mode(config) != "sfincs_native_river_inflow":
        wanted = {str(value) for value in handoff_source_ids}
        src = src[src["sfincs_handoff_id"].astype(str).isin(wanted)].copy()
        missing = sorted(wanted - set(src["sfincs_handoff_id"].astype(str)))
        if missing:
            raise ValueError("Native SFINCS river inflow did not create requested handoff IDs: " + ", ".join(missing))
    src["wflow_submodel_id"] = wflow_submodel_id
    src["sfincs_domain_id"] = sfincs_domain_id
    src["gauge_location_source"] = "sfincs_native_river_inflow"
    src["handoff_placement"] = "sfincs_native_river_inflow"
    src["handoff_location_review_status"] = "hydromt_sfincs_native_river_inflow"
    src["stream_boundary_river_source"] = str(hydrography)
    src["river_upa_km2"] = river_upa
    src["river_len_m"] = river_len
    src["river_inflow_buffer_m"] = buffer
    src = _add_reservoir_boundary_handoff_sources(
        src,
        config,
        paths,
        model=model,
        sfincs_domain_id=sfincs_domain_id,
        wflow_submodel_id=wflow_submodel_id,
        model_crs=model_crs,
        domain_region=domain_region,
    )
    src = src.reset_index(drop=True)
    src["index"] = range(int(first_index), int(first_index) + len(src))
    src["name"] = src["sfincs_handoff_id"].astype(str)
    src["site_no"] = src["sfincs_handoff_id"].astype(str)

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
        "source_reservoir_name",
        "source_reservoir_id",
        "reservoir_boundary_intersection_length_m",
        "river_upa_km2",
        "river_len_m",
        "river_inflow_buffer_m",
        "geometry",
    ]
    keep = [column for column in keep if column in src.columns]
    src = gpd.GeoDataFrame(src[keep], geometry="geometry", crs=model_crs)
    model.discharge_points.set_locations(src, merge=False)
    rivers = sfincs_rivers_inflow_geoms(model)
    if not rivers.empty:
        _write_sfincs_rivers_inflow_geoms(rivers, out_path.parent / "rivers_inflow.geojson")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src.to_file(out_path, driver="GeoJSON")
    return src


def _register_sfincs_hydrography_source(model, config, paths, source_name) -> None:
    """Register the collected local HydroMT hydrography basemap for native SFINCS rivers."""
    if not isinstance(source_name, str) or Path(source_name).suffix:
        return
    location_root = _location_root(paths)
    hydrography_path = _location_path(
        location_root,
        config.get("collection", {})
        .get("national_hydrography", {})
        .get("hydromt_basemap", "data/wflow/hydrography/us_hydrography_basemap.nc"),
    )
    if not hydrography_path.exists():
        return
    model.data_catalog.from_dict(
        {
            source_name: {
                "uri": str(hydrography_path),
                "data_type": "RasterDataset",
                "driver": {"name": "raster_xarray"},
                "metadata": {"category": "hydrography"},
            }
        }
    )


def _add_reservoir_boundary_handoff_sources(
    src: gpd.GeoDataFrame,
    config,
    paths,
    *,
    model,
    sfincs_domain_id: str,
    wflow_submodel_id: str,
    model_crs,
    domain_region=None,
) -> gpd.GeoDataFrame:
    """Append SFINCS sources where configured reservoir polygons cross the SFINCS boundary."""
    if not _reservoir_boundary_handoffs_enabled(config):
        return src
    region = _resolve_handoff_region(model, paths, domain_region=domain_region, model_crs=model_crs)
    if region.empty:
        return src
    reservoirs = _resolve_reservoir_plot_layer(None, model_crs=model_crs, config=config, paths=paths)
    if reservoirs.empty:
        return src

    region = region.to_crs(model_crs)
    reservoirs = reservoirs.to_crs(model_crs)
    boundary = region.geometry.union_all().boundary
    existing = src.to_crs(model_crs) if src.crs is not None else src.set_crs(model_crs)
    existing_ids = set(existing["sfincs_handoff_id"].astype(str)) if "sfincs_handoff_id" in existing else set()
    tolerance = _reservoir_boundary_existing_source_tolerance_m(config)
    placement = sorted(RESERVOIR_BOUNDARY_HANDOFF_MODES)[0]

    rows = []
    for _, reservoir in reservoirs.iterrows():
        geometry = reservoir.geometry
        if geometry is None or geometry.is_empty or not geometry.intersects(boundary):
            continue
        intersection = geometry.intersection(boundary)
        if intersection is None or intersection.is_empty:
            continue
        for point in _intersection_points(intersection):
            if not existing.empty and float(existing.geometry.distance(point).min()) <= tolerance:
                continue
            handoff_id = _reservoir_boundary_handoff_id(
                str(sfincs_domain_id),
                reservoir,
                existing_ids | {str(row["sfincs_handoff_id"]) for row in rows},
            )
            rows.append(
                {
                    "index": np.nan,
                    "name": handoff_id,
                    "uparea": np.nan,
                    "site_no": handoff_id,
                    "sfincs_handoff_id": handoff_id,
                    "wflow_submodel_id": str(wflow_submodel_id),
                    "sfincs_domain_id": str(sfincs_domain_id),
                    "gauge_location_source": placement,
                    "handoff_placement": placement,
                    "handoff_location_review_status": f"hydromt_{placement}",
                    "stream_boundary_river_source": "reservoir_boundary_intersection",
                    "source_reservoir_name": _reservoir_name(reservoir),
                    "source_reservoir_id": _reservoir_identifier(reservoir),
                    "reservoir_boundary_intersection_length_m": float(intersection.length),
                    "river_upa_km2": np.nan,
                    "river_len_m": np.nan,
                    "river_inflow_buffer_m": np.nan,
                    "geometry": point,
                }
            )
    if not rows:
        return src
    reservoir_src = gpd.GeoDataFrame(rows, geometry="geometry", crs=model_crs)
    return gpd.GeoDataFrame(pd.concat([src, reservoir_src], ignore_index=True), geometry="geometry", crs=model_crs)


def _reservoir_boundary_handoffs_enabled(config) -> bool:
    forcing = (config.get("inland_coupling", {}) or {}).get("discharge_forcing", {}) or {}
    cfg = forcing.get("reservoir_boundary_handoffs", {}) or {}
    return bool(cfg.get("enabled", False))


def _reservoir_boundary_existing_source_tolerance_m(config) -> float:
    forcing = (config.get("inland_coupling", {}) or {}).get("discharge_forcing", {}) or {}
    cfg = forcing.get("reservoir_boundary_handoffs", {}) or {}
    if "max_existing_source_distance_m" in cfg:
        return float(cfg["max_existing_source_distance_m"])
    grid_resolution = float(config.get("sfincs", {}).get("grid_resolution_m", 60.0))
    return max(grid_resolution * 2.0, grid_resolution)


def _resolve_handoff_region(model, paths, *, domain_region=None, model_crs=None) -> gpd.GeoDataFrame:
    region = getattr(model, "region", None)
    if region is not None and not region.empty:
        return region.to_crs(model_crs) if region.crs is not None and model_crs is not None else region
    if domain_region in (None, ""):
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=model_crs)
    location_root = _location_root(paths)
    region_path = _location_path(location_root, domain_region)
    if not region_path.exists():
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=model_crs)
    region = gpd.read_file(region_path)
    if region.crs is None and model_crs is not None:
        return region.set_crs(model_crs)
    return region.to_crs(model_crs) if model_crs is not None else region


def _reservoir_boundary_handoff_id(sfincs_domain_id: str, reservoir, existing_ids: set[str]) -> str:
    reservoir_name = _reservoir_name(reservoir).strip().lower().replace(" ", "_")
    reservoir_name = "".join(ch for ch in reservoir_name if ch.isalnum() or ch == "_").strip("_")
    stem = f"{sfincs_domain_id}_reservoir"
    if reservoir_name:
        stem = f"{stem}_{reservoir_name}"
    candidate = stem
    suffix = 1
    while candidate in existing_ids:
        suffix += 1
        candidate = f"{stem}_{suffix:02d}"
    return candidate


def _reservoir_name(row) -> str:
    for column in ("waterbody_name", "GNIS_Name", "gnis_name", "name", "Name", "res_name", "RES_NAME"):
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column]).strip()
    return ""


def _reservoir_identifier(row) -> str:
    for column in ("waterbody_id", "source_nhdplusid", "NHDPlusID", "nhdplusid", "featureid", "FeatureID"):
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column]).strip()
    return ""


def build_inland_sfincs_base(config, paths, *, model_cls=None, force=False):
    plan = plan_inland_sfincs_base(config, paths)
    return _build_inland_sfincs_base_plan(config, paths, plan, model_cls=model_cls, force=force)


def build_domains(config, paths, *, model_cls=None, force=False):
    """Build all SFINCS base models listed in the inland domain-set manifest."""
    location_root = _location_root(paths)
    manifest_path = _location_path(
        location_root,
        config.get("sfincs_domain_set", {}).get("domain_manifest", "data/sfincs/domains/domain_set.yaml"),
    )
    if not manifest_path.exists():
        plan = plan_inland_sfincs_domain_set(config, paths)
        manifest_path = write_inland_sfincs_domain_set_manifest(plan, config, paths)
        if plan.status != "ready":
            raise RuntimeError(f"SFINCS domain set is not ready: {plan.issues}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    rows = []
    for domain in manifest.get("domains", []):
        domain_plan = _domain_base_plan(config, paths, domain)
        summary = _build_inland_sfincs_base_plan(config, paths, domain_plan, model_cls=model_cls, force=force)
        handoff_path = domain_plan.base_model_root / "gis/wflow_handoff_sources.geojson"
        if handoff_location_mode(config) == "sfincs_native_river_inflow":
            if model_cls is None:
                os.environ.pop("DEBUG", None)
                from hydromt_sfincs import SfincsModel

                domain_model_cls = SfincsModel
            else:
                domain_model_cls = model_cls
            model = domain_model_cls(root=str(domain_plan.base_model_root), mode="r+", write_gis=True)
            model.read()
            src = create_handoffs(
                model,
                config,
                paths,
                output=handoff_path,
                sfincs_domain_id=domain["sfincs_domain_id"],
                wflow_submodel_id=next(iter(domain.get("wflow_submodel_ids", [])), None),
                handoff_source_ids=domain.get("handoff_source_ids", ()),
                domain_region=domain_plan.region,
            )
            model.write()
            rivers = sfincs_rivers_inflow_geoms(model)
            handoff_summary = {
                "handoff_locations": handoff_path,
                "rivers_inflow": handoff_path.parent / "rivers_inflow.geojson",
                "source_point_count": int(len(src)),
                "rivers_inflow_count": int(len(rivers)),
            }
        else:
            handoff_summary = write_inland_sfincs_handoff_locations(
                config,
                paths,
                output=handoff_path,
                domain_region=domain_plan.region,
                sfincs_domain_id=domain["sfincs_domain_id"],
            )
        rows.append(
            {
                "sfincs_domain_id": domain["sfincs_domain_id"],
                "status": summary["status"],
                "base_model_root": str(summary["base_model_root"]),
                "built": bool(summary["built"]),
                "region": domain_plan.region.as_posix(),
                "handoff_locations": str(handoff_summary["handoff_locations"]),
                "source_point_count": int(handoff_summary["source_point_count"]),
            }
        )
    if not rows:
        raise ValueError(f"SFINCS domain-set manifest has no domains: {manifest_path}")
    return pd.DataFrame(rows)


def sfincs_rivers_inflow_geoms(
    model,
    *,
    root=None,
    config=None,
    paths=None,
    handoff_sources=None,
    domain_region=None,
) -> gpd.GeoDataFrame:
    """Return HydroMT-SFINCS native ``rivers_inflow`` linework for plotting."""
    model_crs = getattr(model, "crs", None)
    component = getattr(model, "components", {}).get("rivers")
    if component is not None:
        data = getattr(component, "data", None)
        if data is not None and not data.empty:
            rivers = data.copy()
            if rivers.crs is None and model_crs is not None:
                rivers = rivers.set_crs(model_crs)
            elif model_crs is not None:
                rivers = rivers.to_crs(model_crs)
            return rivers

    model_root = _sfincs_model_root_path(model, root=root)
    path = model_root / "gis/rivers_inflow.geojson"
    if path.exists():
        rivers = gpd.read_file(path)
    elif config is not None and paths is not None and handoff_sources is not None:
        handoff = (
            gpd.read_file(handoff_sources)
            if not isinstance(handoff_sources, gpd.GeoDataFrame)
            else handoff_sources.copy()
        )
        rivers = wflow_handoff_rivers_inflow_geoms(
            config,
            paths,
            handoff,
            domain_region=domain_region,
            model_crs=model_crs,
        )
    else:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=model_crs)
    if rivers.crs is None and model_crs is not None:
        rivers = rivers.set_crs(model_crs)
    elif model_crs is not None:
        rivers = rivers.to_crs(model_crs)
    valid = [geometry is not None and not geometry.is_empty for geometry in rivers.geometry]
    return rivers.loc[valid].reset_index(drop=True)


def _sfincs_model_root_path(model, *, root=None) -> Path:
    raw_root = root if root is not None else getattr(model, "root", None)
    if hasattr(raw_root, "path"):
        raw_root = raw_root.path
    if raw_root in (None, ""):
        return Path(".")
    return Path(raw_root)


def wflow_handoff_rivers_inflow_geoms(
    config,
    paths,
    handoff: gpd.GeoDataFrame,
    *,
    domain_region=None,
    model_crs=None,
) -> gpd.GeoDataFrame:
    """Return Wflow-native river linework clipped to the SFINCS handoff domain."""
    if handoff.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=model_crs or handoff.crs)

    location_root = _location_root(paths)
    rivers = _wflow_native_river_lines(config, location_root, handoff)
    if rivers.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=model_crs or handoff.crs)

    if model_crs is not None:
        rivers = rivers.to_crs(model_crs)
    if domain_region not in (None, ""):
        region_path = _location_path(location_root, domain_region)
        if region_path.exists():
            region = gpd.read_file(region_path)
            if region.crs is None:
                region = region.set_crs(rivers.crs)
            else:
                region = region.to_crs(rivers.crs)
            try:
                rivers = gpd.clip(rivers, region)
            except Exception:
                region_geometry = region.union_all() if hasattr(region, "union_all") else region.unary_union
                rivers = rivers.loc[rivers.intersects(region_geometry)].copy()
    valid = [geometry is not None and not geometry.is_empty for geometry in rivers.geometry]
    return rivers.loc[valid].reset_index(drop=True)


def _write_sfincs_rivers_inflow_geoms(rivers: gpd.GeoDataFrame, output: Path) -> None:
    if rivers.empty:
        if output.exists():
            output.unlink()
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    rivers.to_file(output, driver="GeoJSON")


def set_observations(model, gages) -> gpd.GeoDataFrame:
    """Set reviewed gages as HydroMT-SFINCS observation points for plotting."""
    gdf = gpd.read_file(gages) if not isinstance(gages, gpd.GeoDataFrame) else gages.copy()
    if gdf.empty:
        return gdf
    model_crs = getattr(model, "crs", None)
    if model_crs is None:
        raise ValueError("HydroMT-SFINCS model CRS is required before setting observation points")
    if gdf.crs is None:
        gdf = gdf.set_crs(model_crs)
    else:
        gdf = gdf.to_crs(model_crs)

    valid = [geometry is not None and not geometry.is_empty and geometry.geom_type == "Point" for geometry in gdf.geometry]
    gdf = gdf.loc[valid].copy()
    if gdf.empty:
        return gdf

    if "name" not in gdf:
        for column in ("site_no", "sfincs_handoff_id", "site_name", "station_nm"):
            if column in gdf and gdf[column].notna().any():
                gdf["name"] = gdf[column].fillna("").astype(str)
                break
        else:
            gdf["name"] = [str(index + 1) for index in range(len(gdf))]
    gdf["name"] = gdf["name"].fillna("").astype(str)
    empty_names = gdf["name"].str.strip().eq("")
    if empty_names.any():
        gdf.loc[empty_names, "name"] = [str(index + 1) for index in range(int(empty_names.sum()))]

    region = getattr(model, "region", None)
    if region is not None and not region.empty:
        if region.crs is not None and region.crs != gdf.crs:
            region = region.to_crs(gdf.crs)
        region_geometry = region.union_all() if hasattr(region, "union_all") else region.unary_union
        gdf = gdf.loc[gdf.within(region_geometry)].copy()
    if gdf.empty:
        return gdf

    model.observation_points.set(gdf, merge=False)
    return gdf


def plot_sfincs_handoff_basemap(
    model,
    *,
    handoff_sources,
    rivers: gpd.GeoDataFrame | None = None,
    observations: gpd.GeoDataFrame | None = None,
    reservoirs=None,
    config=None,
    paths=None,
    domain_region=None,
    figsize=(8, 6),
    legend_outside: bool = True,
    plot_reservoirs: bool = False,
):
    """Plot a HydroMT-SFINCS basemap with visible Wflow handoff QA overlays."""
    from matplotlib.lines import Line2D

    src = gpd.read_file(handoff_sources) if not isinstance(handoff_sources, gpd.GeoDataFrame) else handoff_sources.copy()
    if src.crs is None:
        src = src.set_crs(model.crs)
    src = src.to_crs(model.crs)
    rivers, river_layer_label = _resolve_handoff_plot_rivers(
        model,
        rivers,
        config=config,
        paths=paths,
        handoff_sources=src,
        domain_region=domain_region,
    )
    reservoir_layer = (
        _resolve_reservoir_plot_layer(
            reservoirs,
            model_crs=model.crs,
            config=config,
            paths=paths,
        )
        if plot_reservoirs
        else gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=model.crs)
    )

    _ensure_dep_nodata_for_native_plot(model)
    fig, ax = model.plot_basemap(
        figsize=figsize,
        plot_bounds=True,
        plot_geoms=True,
        geom_names=["src", "obs", "rivers"],
        geom_kwargs={
            "src": dict(marker=">", markersize=55, facecolor="white", edgecolor="crimson", linewidth=1.2),
            "obs": dict(marker="o", facecolor="none", edgecolor="black", markersize=25),
            "rivers": dict(color="royalblue", linewidth=0.0, alpha=0.0),
        },
    )

    visible_rivers = _visible_sfincs_rivers(model, rivers)
    visible_reservoirs = _visible_sfincs_polygons(model, reservoir_layer)
    river_style = _sfincs_river_plot_style(river_layer_label, alpha=0.9)
    if not visible_reservoirs.empty:
        visible_reservoirs.boundary.plot(
            ax=ax,
            color="black",
            linewidth=0.7,
            alpha=0.75,
            zorder=7,
            label="reservoirs",
        )
    if not visible_rivers.empty:
        visible_rivers.plot(
            ax=ax,
            zorder=8,
            label=river_layer_label,
            **river_style,
        )
    if observations is not None and not observations.empty:
        observations.to_crs(model.crs).plot(
            ax=ax,
            marker="o",
            facecolor="none",
            edgecolor="black",
            markersize=25,
            zorder=9,
            label="reviewed USGS gage",
        )
    if not src.empty:
        src.plot(
            ax=ax,
            marker=">",
            facecolor="white",
            edgecolor="crimson",
            linewidth=1.2,
            markersize=55,
            zorder=10,
            label="src",
        )
        for collection in ax.collections[-1:]:
            collection.set_clip_on(False)

    base_handles, base_labels = ax.get_legend_handles_labels()
    handles: list = []
    labels: list[str] = []
    skip = {"src", "obs", "rivers", "rivers_inflow", "rivers_outflow", "Wflow rivers", "reviewed USGS gage", "reservoirs"}
    for handle, label in zip(base_handles, base_labels):
        if not label or label == "_nolegend_" or label in skip or label in labels:
            continue
        handles.append(handle)
        labels.append(label)
    if not visible_rivers.empty:
        handles.append(Line2D([0], [0], **river_style))
        labels.append(river_layer_label)
    if not visible_reservoirs.empty:
        handles.append(Line2D([0], [0], color="black", linewidth=0.7, alpha=0.75))
        labels.append("reservoirs")
    if observations is not None and not observations.empty:
        handles.append(
            Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none", markeredgecolor="black", markersize=5)
        )
        labels.append("reviewed USGS gage")
    if not src.empty:
        handles.append(
            Line2D([0], [0], marker=">", linestyle="None", markerfacecolor="white", markeredgecolor="crimson", markersize=8)
        )
        labels.append("src")
    if handles:
        if legend_outside:
            ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
            fig.subplots_adjust(bottom=0.2)
        else:
            ax.legend(handles, labels, loc="best")
    qa = {
        "discharge_sources_src": int(len(src)),
        "unique_discharge_source_locations": int(len(src.geometry.apply(lambda geom: (round(geom.x, 2), round(geom.y, 2))).drop_duplicates()))
        if not src.empty
        else 0,
        "visible_wflow_native_river_features": int(len(visible_rivers)),
        "river_layer": river_layer_label,
        "reviewed_usgs_gages_visible": int(0 if observations is None else len(observations)),
        "reservoir_features_visible": int(len(visible_reservoirs)),
    }
    return fig, ax, qa


def _resolve_handoff_plot_rivers(
    model,
    rivers: gpd.GeoDataFrame | None,
    *,
    config=None,
    paths=None,
    handoff_sources=None,
    domain_region=None,
) -> tuple[gpd.GeoDataFrame, str]:
    if config is not None and paths is not None and handoff_sources is not None:
        try:
            wflow_rivers = wflow_handoff_rivers_inflow_geoms(
                config,
                paths,
                handoff_sources,
                domain_region=domain_region,
                model_crs=getattr(model, "crs", None),
            )
        except Exception:
            wflow_rivers = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=getattr(model, "crs", None))
        if not wflow_rivers.empty:
            return wflow_rivers, "rivers_inflow"
        if uses_stream_boundary_handoff(config):
            return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=getattr(model, "crs", None)), "rivers_inflow"
        if rivers is not None and not rivers.empty:
            return rivers, "rivers_inflow"
    if rivers is not None and not rivers.empty:
        return rivers, "rivers_inflow"
    return sfincs_rivers_inflow_geoms(
        model,
        config=config,
        paths=paths,
        handoff_sources=handoff_sources,
        domain_region=domain_region,
    ), "rivers_inflow"


def _sfincs_river_plot_style(layer: str, **overrides) -> dict:
    style = HYDROMT_SFINCS_GEOM_STYLE.get(layer, HYDROMT_SFINCS_GEOM_STYLE["rivers_inflow"]).copy()
    style.update(overrides)
    return style


def _ensure_dep_nodata_for_native_plot(model) -> None:
    """Keep HydroMT-SFINCS native plotting on its mask_nodata path for dep."""
    grid = getattr(getattr(model, "grid", None), "data", None)
    if grid is None or "dep" not in grid or "mask" not in grid:
        return

    dep = grid["dep"]
    mask = grid["mask"]
    fill_value = _finite_dep_fill_value(dep.attrs.get("_FillValue"))
    if fill_value is None:
        fill_value = _infer_dep_fill_value(dep, mask)
    if fill_value is None:
        return

    # HydroMT-SFINCS derives dep color limits before applying mask > 0, so the
    # inactive fill value must be registered with the raster accessor.
    _set_dep_nodata(dep, float(fill_value))
    elevation = getattr(getattr(model, "elevation", None), "data", None)
    if elevation is not None and "dep" in elevation:
        _set_dep_nodata(elevation["dep"], float(fill_value))


def _infer_dep_fill_value(dep, mask) -> float | None:
    dep_values = np.asarray(dep.values)
    inactive = np.asarray(mask.values) == 0
    if not inactive.any():
        return None
    inactive_values = dep_values[inactive & np.isfinite(dep_values)]
    if inactive_values.size == 0:
        return None
    fill_values, counts = np.unique(inactive_values, return_counts=True)
    fill_value = float(fill_values[int(np.argmax(counts))])
    if fill_value >= -100.0:
        return None
    return fill_value


def _finite_dep_fill_value(value) -> float | None:
    if value is None:
        return None
    try:
        fill_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(fill_value):
        return None
    return fill_value


def _set_dep_nodata(dep, fill_value: float) -> None:
    raster = getattr(dep, "raster", None)
    if raster is not None and hasattr(raster, "set_nodata"):
        raster.set_nodata(fill_value)
    else:
        dep.attrs["_FillValue"] = fill_value


def _resolve_reservoir_plot_layer(reservoirs, *, model_crs, config=None, paths=None) -> gpd.GeoDataFrame:
    layer = None
    if hasattr(reservoirs, "geometry"):
        layer = reservoirs.copy()
    elif reservoirs not in (None, ""):
        location_root = _location_root(paths) if paths is not None else Path(".")
        layer = gpd.read_file(_location_path(location_root, reservoirs))
    elif config is not None and paths is not None:
        reservoir_cfg = (
            ((config.get("collection", {}) or {}).get("national_hydrography", {}) or {})
            .get("reservoirs", {})
            or {}
        )
        if reservoir_cfg.get("enabled", False):
            location_root = _location_root(paths)
            reservoirs_path = reservoir_cfg.get(
                "output",
                "data/sources/national_hydrography/nhdplus_hr_wflow_reservoirs.gpkg",
            )
            reservoirs_path = _location_path(location_root, reservoirs_path)
            if reservoirs_path.exists():
                layer = gpd.read_file(reservoirs_path)

    if layer is None or layer.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=model_crs)
    if layer.crs is None and model_crs is not None:
        layer = layer.set_crs(model_crs)
    elif model_crs is not None:
        layer = layer.to_crs(model_crs)
    valid = [geometry is not None and not geometry.is_empty for geometry in layer.geometry]
    return layer.loc[valid].reset_index(drop=True)


def _visible_sfincs_polygons(model, polygons: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    if polygons is None or polygons.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=getattr(model, "crs", None))
    polygons = polygons.to_crs(model.crs)
    region = getattr(model, "region", None)
    if region is None or region.empty:
        return polygons
    region = region.to_crs(polygons.crs) if region.crs is not None else region.set_crs(polygons.crs)
    try:
        clipped = gpd.clip(polygons, region)
    except Exception:
        region_geometry = region.union_all() if hasattr(region, "union_all") else region.unary_union
        clipped = polygons.loc[polygons.intersects(region_geometry)].copy()
    valid = [geometry is not None and not geometry.is_empty for geometry in clipped.geometry]
    return clipped.loc[valid].reset_index(drop=True)


def _visible_sfincs_rivers(model, rivers: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame:
    if rivers is None or rivers.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=getattr(model, "crs", None))
    rivers = rivers.to_crs(model.crs)
    region = getattr(model, "region", None)
    if region is None or region.empty:
        return rivers
    region = region.to_crs(rivers.crs) if region.crs is not None else region.set_crs(rivers.crs)
    try:
        clipped = gpd.clip(rivers, region)
    except Exception:
        region_geometry = region.union_all() if hasattr(region, "union_all") else region.unary_union
        clipped = rivers.loc[rivers.intersects(region_geometry)].copy()
    valid = [geometry is not None and not geometry.is_empty for geometry in clipped.geometry]
    return clipped.loc[valid].reset_index(drop=True)


def outflow_zmax_from_active_mask(mask, dep, *, quantile=DEFAULT_OUTFLOW_ELEVATION_QUANTILE) -> float:
    """Return the elevation threshold marking the lowest active-domain perimeter cells.

    A watershed drains at the topographic low point(s) of the domain perimeter, so the
    outflow boundary is placed where perimeter-cell elevations fall at or below ``quantile``
    of the perimeter elevation distribution.
    """
    mask = np.asarray(mask)
    dep = np.asarray(dep, dtype="float64")
    active = mask > 0
    if not active.any():
        raise ValueError("Active mask is empty; create the active mask before the outflow boundary")
    interior = ndimage.binary_erosion(active, structure=np.ones((3, 3), dtype=bool), border_value=0)
    edge = active & ~interior
    edge_dep = dep[edge]
    edge_dep = edge_dep[np.isfinite(edge_dep)]
    if edge_dep.size == 0:
        raise ValueError("Active-domain perimeter has no valid elevations for the outflow boundary")
    return float(np.quantile(edge_dep, float(quantile)))


def add_inland_outflow_boundary(model, *, quantile=DEFAULT_OUTFLOW_ELEVATION_QUANTILE) -> dict:
    """Mark the watershed drainage outlet as a SFINCS outflow (sink) boundary.

    Without an outflow boundary every active cell is closed (mask=1) and water cannot leave
    the domain. This marks the lowest perimeter cells -- where the main channel exits
    following topography -- as outflow cells (mask=3) so the model drains at the bottom of
    the watershed. Must be called after the active mask and elevation are created.
    """
    grid = model.grid.data
    zmax = outflow_zmax_from_active_mask(grid["mask"].values, grid["dep"].values, quantile=quantile)
    model.mask.create_boundary(btype="outflow", zmax=zmax, reset_bounds=True)
    return {"outflow_zmax_m": zmax}


def _outflow_elevation_quantile(config) -> float:
    return float(
        config.get("sfincs", {}).get(
            "outflow_boundary_elevation_quantile", DEFAULT_OUTFLOW_ELEVATION_QUANTILE
        )
    )


def _build_inland_sfincs_base_plan(config, paths, plan: InlandSfincsBasePlan, *, model_cls=None, force=False):
    if plan.built and not force:
        try:
            validate_physics(plan.base_model_root, config)
        except RuntimeError as exc:
            if not _should_rebuild_stale_native_physics(config, exc):
                raise
            summary = _build_inland_sfincs_base_plan(
                config,
                paths,
                plan,
                model_cls=model_cls,
                force=True,
            )
            return {
                **summary,
                "status": "rebuilt_stale_native_physics",
                "rebuild_reason": str(exc),
            }
        return {"status": "reused", "base_model_root": plan.base_model_root, "built": True}
    if plan.missing_inputs:
        raise FileNotFoundError("Missing SFINCS base inputs: " + ", ".join(path.as_posix() for path in plan.missing_inputs))
    if model_cls is None:
        os.environ.pop("DEBUG", None)
        from hydromt_sfincs import SfincsModel

        model_cls = SfincsModel
    if force and plan.base_model_root.exists():
        shutil.rmtree(plan.base_model_root)
    plan.base_model_root.mkdir(parents=True, exist_ok=True)

    model = model_cls(root=str(plan.base_model_root), mode="w+", write_gis=True)
    model.data_catalog.from_dict(
        {
            "dem_region": _raster_source(plan.dem, crs=plan.model_crs),
            "landcover_region": _raster_source(plan.landcover, crs=plan.model_crs),
            "hydrologic_soil_group": _raster_source(plan.hsg, crs=plan.model_crs),
            "saturated_conductivity": _raster_source(plan.ksat, crs=plan.model_crs),
        }
    )
    model.grid.create_from_region(
        region={"geom": str(plan.region)},
        res=plan.grid_resolution_m,
        rotated=False,
        crs=plan.model_crs,
    )
    model.elevation.create(elevation_list=[{"elevation": "dem_region"}], buffer_cells=1)
    model.mask.create_active(include_polygon=str(plan.region), reset_mask=True)
    add_inland_outflow_boundary(model, quantile=_outflow_elevation_quantile(config))
    esa_mapping = _esa_worldcover_mapping()
    if hasattr(model, "roughness"):
        model.roughness.create(
            roughness_list=[{"lulc": "landcover_region", "reclass_table": str(esa_mapping)}],
            manning_land=0.04,
            manning_sea=0.02,
            rgh_lev_land=0,
        )
    if hasattr(model, "subgrid"):
        model.subgrid.create(
            elevation_list=[{"elevation": "dem_region"}],
            roughness_list=[{"lulc": "landcover_region", "reclass_table": str(esa_mapping)}],
            nr_subgrid_pixels=int(config.get("sfincs", {}).get("nr_subgrid_pixels", 6)),
            write_dep_tif=True,
            write_man_tif=True,
        )
    setup_hydromt_infiltration(model, config, paths, datadir=_hydromt_sfincs_datadir())
    epsg = _epsg_code(plan.model_crs)
    if epsg is not None:
        model.config.update({"epsg": epsg})
    model.config.update({"storevel": 1})
    model.write()
    validate_physics(plan.base_model_root, config)
    return {
        "status": "built",
        "base_model_root": plan.base_model_root,
        "built": is_built_sfincs_base(plan.base_model_root),
    }


def _should_rebuild_stale_native_physics(config: dict, exc: RuntimeError) -> bool:
    """Return true when an existing base predates required native SFINCS physics."""
    drivers = set(config.get("event_drivers") or [])
    infiltration_cfg = (
        config.get("inland_coupling", {})
        .get("infiltration", {})
        or {}
    )
    rain_on_grid = bool(drivers & {"rainfall", "soil_moisture"})
    infiltration_required = bool(infiltration_cfg.get("enabled", True))
    text = str(exc)
    return bool(
        rain_on_grid
        and infiltration_required
        and (
            "lacks active native infiltration" in text
            or "spatial roughness" in text
            or "subgrid file has no roughness" in text
        )
    )


def _snap_handoff_locations_to_domain_boundary(handoff, config, paths, *, domain_region=None):
    location_mode = (
        config.get("inland_coupling", {})
        .get("discharge_forcing", {})
        .get("handoff_location", "reviewed_gage")
    )
    location_mode = str(location_mode).lower()
    if location_mode in STREAM_BOUNDARY_HANDOFF_MODES:
        return _place_handoff_locations_at_stream_boundary_intersections(
            handoff,
            config,
            paths,
            domain_region=domain_region,
        )
    if location_mode not in LEGACY_BOUNDARY_HANDOFF_MODES:
        return handoff
    if domain_region in (None, "") or handoff.empty:
        return handoff

    location_root = _location_root(paths)
    region_path = _location_path(location_root, domain_region)
    if not region_path.exists():
        raise FileNotFoundError(region_path)
    region = gpd.read_file(region_path).to_crs(handoff.crs)
    boundary = region.geometry.union_all().boundary
    snapped = handoff.copy()
    snapped["geometry"] = [
        nearest_points(boundary, point)[0] if point is not None and not point.is_empty else point
        for point in snapped.geometry
    ]
    return snapped


def _place_handoff_locations_at_stream_boundary_intersections(handoff, config, paths, *, domain_region=None):
    if domain_region in (None, "") or handoff.empty:
        return handoff

    location_root = _location_root(paths)
    region_path = _location_path(location_root, domain_region)
    if not region_path.exists():
        raise FileNotFoundError(region_path)
    region = gpd.read_file(region_path).to_crs(handoff.crs)
    rivers = _resolve_boundary_handoff_rivers(config, paths, handoff).to_crs(handoff.crs)
    boundary = region.geometry.union_all().boundary
    candidates = _stream_boundary_intersection_candidates(rivers, boundary)
    if candidates.empty:
        raise ValueError(
            f"No stream/SFINCS-boundary intersections found for {region_path}; "
            "review the coverage box or enlarge the hydrologic domain."
        )

    snapped = handoff.copy()
    snapped["source_gage_x"] = snapped.geometry.x
    snapped["source_gage_y"] = snapped.geometry.y
    for index, row in snapped.iterrows():
        selected = _select_upstream_stream_boundary_intersection(row.geometry, rivers, candidates, row)
        for key, value in selected.items():
            if key == "geometry":
                snapped.at[index, "geometry"] = value
            else:
                snapped.at[index, key] = value
    return snapped


def _resolve_boundary_handoff_rivers(config, paths, handoff: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return river lines for SFINCS boundary handoffs, preferring Wflow-native geometry."""
    location_root = _location_root(paths)
    wflow_rivers = _wflow_native_river_lines(config, location_root, handoff)
    if not wflow_rivers.empty:
        return wflow_rivers

    rivers_path = _location_path(
        location_root,
        config.get("inland_coupling", {})
        .get("discharge_forcing", {})
        .get(
            "fallback_river_geometry",
            config.get("collection", {})
            .get("national_hydrography", {})
            .get("river_geometry", "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"),
        ),
    )
    if not rivers_path.exists():
        raise FileNotFoundError(
            f"No HydroMT-Wflow native rivers found under the Wflow base model root, "
            f"and fallback river geometry is missing: {rivers_path}"
        )
    rivers = gpd.read_file(rivers_path)
    if rivers.empty:
        raise ValueError(f"Fallback river geometry has no features: {rivers_path}")
    rivers = rivers.copy()
    rivers["river_geometry_source"] = "review_hydrography_fallback"
    rivers["river_geometry_source_path"] = str(rivers_path)
    return _prepare_boundary_rivers(rivers)


def _wflow_native_river_lines(config, location_root: Path, handoff: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    base_root = _location_path(
        location_root,
        config.get("wflow", {}).get("base_model_root", "data/wflow/base"),
    )
    submodel_ids = sorted(
        {
            str(value)
            for value in handoff.get("wflow_submodel_id", pd.Series([], dtype=object)).dropna()
            if str(value).strip()
        }
    )
    paths = []
    for submodel_id in submodel_ids:
        paths.append((submodel_id, base_root / submodel_id / "staticgeoms/rivers.geojson"))
    paths.append(("", base_root / "staticgeoms/rivers.geojson"))

    frames = []
    for submodel_id, path in paths:
        if not path.exists():
            continue
        rivers = gpd.read_file(path)
        if rivers.empty:
            continue
        valid = [geometry is not None and not geometry.is_empty for geometry in rivers.geometry]
        rivers = rivers.loc[valid].copy()
        if rivers.empty:
            continue
        if submodel_id:
            rivers["wflow_submodel_id"] = submodel_id
        rivers["river_geometry_source"] = "hydromt_wflow_setup_rivers"
        rivers["river_geometry_source_path"] = str(path)
        frames.append(rivers)
    if not frames:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=handoff.crs)
    return _prepare_boundary_rivers(gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=frames[0].crs))


def _prepare_boundary_rivers(rivers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rivers = rivers.copy()
    if rivers.crs is None:
        rivers = rivers.set_crs("EPSG:4326")
    valid = [geometry is not None and not geometry.is_empty for geometry in rivers.geometry]
    rivers = rivers.loc[valid].reset_index(drop=True)
    rivers["_boundary_river_uid"] = range(len(rivers))
    return rivers


def _stream_boundary_intersection_candidates(rivers: gpd.GeoDataFrame, boundary) -> gpd.GeoDataFrame:
    rows = []
    for river_index, row in rivers.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        intersection = geometry.intersection(boundary)
        for point in _intersection_points(intersection):
            rows.append(
                {
                    "river_index": int(river_index),
                    "_boundary_river_uid": int(row.get("_boundary_river_uid", river_index)),
                    "river_id": _river_identifier(row),
                    "river_geometry_source": row.get("river_geometry_source", ""),
                    "river_geometry_source_path": row.get("river_geometry_source_path", ""),
                    "wflow_submodel_id": row.get("wflow_submodel_id", ""),
                    "geometry": point,
                }
            )
    columns = [
        "river_index",
        "_boundary_river_uid",
        "river_id",
        "river_geometry_source",
        "river_geometry_source_path",
        "wflow_submodel_id",
        "geometry",
    ]
    return gpd.GeoDataFrame(rows, columns=columns, geometry="geometry", crs=rivers.crs)


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


def _select_upstream_stream_boundary_intersection(point, rivers, candidates, handoff_row) -> dict:
    submodel_id = str(handoff_row.get("wflow_submodel_id", "") or "")
    if _is_crossing_derived_handoff(handoff_row):
        return _select_nearest_stream_boundary_intersection(point, rivers, candidates, submodel_id)

    river_pool = rivers
    if submodel_id and "wflow_submodel_id" in rivers.columns:
        submodel_rivers = rivers[rivers["wflow_submodel_id"].astype(str) == submodel_id]
        if not submodel_rivers.empty:
            river_pool = submodel_rivers

    distances = river_pool.geometry.distance(point)
    river_index = int(distances.idxmin())
    river = river_pool.loc[river_index].geometry
    river_uid = int(river_pool.loc[river_index].get("_boundary_river_uid", river_index))
    same_river = candidates[candidates["_boundary_river_uid"] == river_uid].copy()
    if same_river.empty:
        return _select_nearest_stream_boundary_intersection(
            point,
            rivers,
            candidates,
            submodel_id,
            review_status="review_required_stream_boundary_nearest_reach_fallback",
        )

    gage_measure = float(river.project(point))
    same_river["stream_measure_m"] = [float(river.project(candidate)) for candidate in same_river.geometry]
    upstream = same_river[same_river["stream_measure_m"] <= gage_measure + 1.0e-6].copy()
    if upstream.empty:
        same_river["stream_seed_distance_m"] = same_river.geometry.distance(point)
        selected = same_river.sort_values("stream_seed_distance_m").iloc[0]
        return _stream_boundary_selection(
            selected,
            distances,
            river_index,
            gage_measure,
            same_river_count=len(same_river),
            selected_count=0,
            review_status="review_required_stream_boundary_direction_fallback",
        )

    upstream = upstream.sort_values("stream_measure_m")
    selected = upstream.iloc[0]
    return _stream_boundary_selection(
        selected,
        distances,
        river_index,
        gage_measure,
        same_river_count=len(same_river),
        selected_count=len(upstream),
    )


def _select_nearest_stream_boundary_intersection(
    point,
    rivers,
    candidates,
    submodel_id: str,
    *,
    review_status="review_required_stream_boundary_intersection",
) -> dict:
    candidate_pool = candidates
    if submodel_id and "wflow_submodel_id" in candidates.columns:
        submodel_candidates = candidates[candidates["wflow_submodel_id"].astype(str) == submodel_id]
        if not submodel_candidates.empty:
            candidate_pool = submodel_candidates
    if candidate_pool.empty:
        raise ValueError("No stream/SFINCS-boundary intersections found for crossing-derived handoff")

    candidate_pool = candidate_pool.copy()
    candidate_pool["stream_seed_distance_m"] = candidate_pool.geometry.distance(point)
    selected = candidate_pool.sort_values("stream_seed_distance_m").iloc[0].copy()
    river_index = int(selected["river_index"])
    river = rivers.loc[river_index].geometry
    gage_measure = float(river.project(point))
    selected["stream_measure_m"] = float(river.project(selected.geometry))
    distances = pd.Series({river_index: float(river.distance(point))})
    return _stream_boundary_selection(
        selected,
        distances,
        river_index,
        gage_measure,
        same_river_count=len(candidate_pool),
        selected_count=len(candidate_pool),
        review_status=review_status,
    )


def _is_crossing_derived_handoff(handoff_row) -> bool:
    placement = str(handoff_row.get("handoff_placement", "") or "").lower()
    return placement in STREAM_BOUNDARY_HANDOFF_MODES


def _stream_boundary_selection(
    selected,
    distances,
    river_index,
    gage_measure,
    *,
    same_river_count,
    selected_count,
    review_status="review_required_stream_boundary_intersection",
) -> dict:
    return {
        "geometry": selected.geometry,
        "handoff_placement": "stream_boundary_intersection",
        "handoff_location_review_status": review_status,
        "stream_boundary_river_index": int(selected["river_index"]),
        "stream_boundary_river_id": str(selected["river_id"]),
        "stream_boundary_river_source": str(selected.get("river_geometry_source", "")),
        "stream_boundary_river_source_path": str(selected.get("river_geometry_source_path", "")),
        "stream_boundary_measure_m": float(selected["stream_measure_m"]),
        "stream_gage_measure_m": gage_measure,
        "stream_snap_distance_m": float(distances.loc[river_index]),
        "stream_boundary_candidate_count": int(same_river_count),
        "stream_boundary_upstream_candidate_count": int(selected_count),
    }


def _river_identifier(row) -> str:
    for column in ("river_id", "id", "NHDPlusID", "nhdplusid", "featureid", "FeatureID", "COMID", "comid", "idx"):
        if column in row and not pd.isna(row[column]):
            return str(row[column])
    return str(row.name)


def _sfincs_domain_wflow_submodel_ids(config, location_root: Path, assigned: gpd.GeoDataFrame) -> list[str]:
    if assigned.empty:
        return []
    domain_set = config.get("wflow", {}).get("domain_set", {})
    if domain_set.get("allow_multiple_submodels") is False:
        return [
            str(
                domain_set.get("single_submodel_id")
                or f"{config.get('project', {}).get('name', location_root.name)}_main"
            )
        ]
    return sorted(assigned["wflow_submodel_id"].dropna().astype(str).unique().tolist())


def _single_domain_wflow_submodel_id(config, location_root: Path, sfincs_domain_id: str) -> str | None:
    manifest = _location_path(
        location_root,
        config.get("sfincs_domain_set", {}).get("domain_manifest", "data/sfincs/domains/domain_set.yaml"),
    )
    if not manifest.exists():
        return None
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    for domain in payload.get("domains", []):
        if str(domain.get("sfincs_domain_id")) != str(sfincs_domain_id):
            continue
        submodel_ids = [str(value) for value in domain.get("wflow_submodel_ids", ()) if str(value).strip()]
        return submodel_ids[0] if submodel_ids else None
    return None


def _domain_base_plan(config, paths, domain: dict) -> InlandSfincsBasePlan:
    base_plan = plan_inland_sfincs_base(config, paths)
    region = _location_path(_location_root(paths), domain["region"])
    base_model_root = _location_path(_location_root(paths), domain["base_model_root"])
    missing = tuple(
        path
        for path in (
            region,
            base_plan.dem,
            base_plan.landcover,
            base_plan.hsg,
            base_plan.ksat,
            base_plan.handoff_manifest,
        )
        if not path.exists()
    )
    return InlandSfincsBasePlan(
        base_model_root=base_model_root,
        region=region,
        dem=base_plan.dem,
        landcover=base_plan.landcover,
        hsg=base_plan.hsg,
        ksat=base_plan.ksat,
        handoff_manifest=base_plan.handoff_manifest,
        model_crs=base_plan.model_crs,
        grid_resolution_m=base_plan.grid_resolution_m,
        ready_to_build=not missing,
        missing_inputs=missing,
        built=is_built_sfincs_base(base_model_root),
    )


def _raster_source(path, *, crs):
    return {
        "uri": str(path),
        "data_type": "RasterDataset",
        "driver": {"name": "rasterio"},
        "metadata": {"crs": crs},
    }


def _epsg_code(crs):
    text = str(crs)
    if text.upper().startswith("EPSG:"):
        try:
            return int(text.split(":", 1)[1])
        except ValueError:
            return None
    return None


def _hydromt_sfincs_datadir():
    try:
        from hydromt_sfincs import DATADIR
    except Exception:
        return None
    return DATADIR


def _esa_worldcover_mapping() -> Path:
    datadir = _hydromt_sfincs_datadir()
    if datadir is None:
        return Path("lulc/esa_worldcover_mapping.csv")
    return Path(datadir) / "lulc" / "esa_worldcover_mapping.csv"


def _crossing_handoff_sources(config, location_root: Path) -> gpd.GeoDataFrame:
    """SFINCS discharge source points from stream/coverage-box crossings (gage-free).

    Reads the Wflow domain plan (via the dispatcher, so it works for both the per-crossing
    subbasin and the single encompassing-HUC basin) and emits one source point per crossing.
    SFINCS sources and Wflow gauges therefore coincide by construction; no USGS gage network
    is read.
    """
    from wflow_runs.build_plan import plan_wflow_domain_set

    plan = plan_wflow_domain_set(config, {"location_root": location_root})
    if plan.status != "ready":
        raise ValueError(f"Crossing-derived Wflow domain plan is not ready: {plan.status}: {plan.issues}")
    return crossing_handoff_sources(plan)


def _accepted_handoff_gages(config, location_root: Path) -> gpd.GeoDataFrame:
    outlet_source = str(config.get("wflow", {}).get("domain_set", {}).get("outlet_source", "reviewed_streamgages"))
    stream_boundary_mode = handoff_location_mode(config) in {
        "stream_boundary_intersection",
        "sfincs_stream_boundary",
        "boundary_stream_intersection",
    }
    if stream_boundary_mode or outlet_source in {
        "stream_boundary_crossings",
        "encompassing_huc",
        "boundary_handoff_watershed",
        "stream_boundary_watershed",
        "sfincs_boundary_watershed",
    }:
        return _crossing_handoff_sources(config, location_root)
    network_value = (
        config.get("wflow", {})
        .get("streamgage_network", {})
        .get("reviewed_network", "data/sources/usgs_streamgages/streamgage_network.geojson")
    )
    network_path = _location_path(location_root, network_value)
    if not network_path.exists():
        raise FileNotFoundError(network_path)
    gages = gpd.read_file(network_path)
    required = {"site_no", "sfincs_handoff_id", "wflow_submodel_id", "sfincs_domain_id"}
    missing = sorted(required - set(gages.columns))
    if missing:
        raise ValueError(f"Reviewed streamgage network missing handoff columns: {missing}")

    review_status = gages.get("review_status", pd.Series("", index=gages.index)).astype(str)
    handoff = gages[
        gages["sfincs_handoff_id"].notna()
        & gages["sfincs_handoff_id"].astype(str).str.strip().ne("")
        & review_status.str.startswith("accepted")
    ].copy()
    if handoff.empty:
        raise ValueError(f"Reviewed streamgage network has no accepted SFINCS handoff gages: {network_path}")
    return handoff


def _exposure_components(exposure: gpd.GeoDataFrame):
    geometry = unary_union([geom for geom in exposure.geometry if geom is not None and not geom.is_empty])
    if geometry.is_empty:
        return []
    polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    return sorted(polygons, key=lambda geom: (geom.centroid.x, geom.centroid.y))


def _exposure_component_records(exposure: gpd.GeoDataFrame):
    if "subregion_id" not in exposure.columns:
        return [{"geometry": geometry, "subregion_id": None} for geometry in _exposure_components(exposure)]
    has_source_subregion = "source_subregion_id" in exposure.columns
    records = []
    for _, row in exposure.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue
        # The region-setup SFINCS coverage (bbox.geojson) carries the raw exposure name in
        # source_subregion_id and the already-final domain id in subregion_id. Prefer the raw
        # name so _auto_domain_id derives the same id whether the source is the raw study_area
        # footprint or the generated coverage boxes (otherwise the id gets a double prefix).
        subregion_id = row.get("source_subregion_id") if has_source_subregion else None
        if subregion_id is None or pd.isna(subregion_id):
            subregion_id = row.get("subregion_id")
        subregion_id = None if pd.isna(subregion_id) else str(subregion_id)
        polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        for polygon in polygons:
            if polygon is not None and not polygon.is_empty:
                records.append({"geometry": polygon, "subregion_id": subregion_id})
    return sorted(records, key=lambda record: (str(record["subregion_id"] or ""), record["geometry"].centroid.x, record["geometry"].centroid.y))


def _assign_handoffs_to_components(handoff: gpd.GeoDataFrame, components, *, domain_ids=None):
    assignments = {index: [] for index in range(len(components))}
    domain_lookup = {str(domain_id): index for index, domain_id in enumerate(domain_ids or [])}
    for _, row in handoff.iterrows():
        domain_id = row.get("sfincs_domain_id") if hasattr(row, "get") else None
        if domain_id is not None and not pd.isna(domain_id) and str(domain_id) in domain_lookup:
            assignments[domain_lookup[str(domain_id)]].append(row)
            continue
        point = row.geometry
        distances = [component.distance(point) for component in components]
        index = int(min(range(len(distances)), key=lambda item: distances[item]))
        assignments[index].append(row)
    return {
        index: gpd.GeoDataFrame(rows, geometry="geometry", crs=handoff.crs)
        if rows
        else handoff.iloc[[]]
        for index, rows in assignments.items()
    }


def _included_domain_ids(domain_set) -> set[str]:
    values = domain_set.get("include_domain_ids", ())
    return {str(value).strip() for value in values if str(value).strip()}


def _auto_domain_id(project_name: str, index: int, count: int, *, subregion_id: str | None = None) -> str:
    if subregion_id:
        suffix = "".join(char.lower() if char.isalnum() else "_" for char in str(subregion_id)).strip("_")
        if suffix:
            return f"{project_name}_{suffix}"
    if count == 1:
        return f"{project_name}_main"
    if count == 2:
        return f"{project_name}_{'west' if index == 0 else 'east'}"
    return f"{project_name}_{index + 1:02d}"


def _domain_geometry_with_handoffs(component, handoff: gpd.GeoDataFrame, buffer_m: float):
    pieces = [component]
    for point in handoff.geometry if not handoff.empty else []:
        if component.covers(point):
            continue
        component_point, _ = nearest_points(component, point)
        pieces.append(LineString([component_point, point]).buffer(buffer_m))
        pieces.append(point.buffer(buffer_m))
    return unary_union(pieces)


def _location_root(paths):
    if paths.get("location_root") is not None:
        return Path(paths["location_root"])
    repo_root = Path(paths.get("repo_root", Path.cwd()))
    location_name = paths.get("location_name")
    if location_name is None:
        raise ValueError("paths must include 'location_root' or 'location_name'")
    return repo_root / "locations" / str(location_name)


def _location_path(location_root, value):
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts[:2] == ("locations", Path(location_root).name):
        return Path(location_root).parents[1] / path
    return Path(location_root) / path
