from __future__ import annotations

from copy import deepcopy
import os
import json
from pathlib import Path
import shutil

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import Point
from shapely.ops import unary_union

from paths import location_root_from_paths, relative_to_or_absolute, resolve_location_path
from coupling.handoff_sources import (
    STREAM_BOUNDARY_HANDOFF_MODES,
    read_stream_boundary_handoff_location_artifacts,
    read_stream_boundary_handoff_locations,
)
from wflow_runs.domain import (
    configured_or_manifest_submodels,
    render_hydromt_build_steps,
)
import wflow_runs.repairs as wflow_repairs
import wflow_runs.reservoirs as wflow_reservoirs
import wflow_runs.staticmaps_qa as staticmaps_qa
import wflow_runs.types as wflow_types

_GENERATED_NOTICE = (
    "# GENERATED FILE — do not edit. Overwritten when {source} runs.\n"
    "# Source of truth is the location config and the code that produces this file.\n"
)


DEFAULT_US_WFLOW_SOURCE_STRATEGY = {
    "hydrography": {
        "preferred": "us_hydrography",
        "hydromt_basemap": "us_hydrography_basemap",
        "river_geometry": "nhdplus_hr_river_geometry",
        "api": "hydromt_gis_flw_d8_from_dem",
        "fallback": "merit_hydro",
    },
    "soils": {
        "preferred": "ssurgo",
        "wflow_parameters": "ssurgo_wflow_soil_parameters",
        "fallback": "soilgrids",
    },
}

REVIEWED_STREAMGAGE_SCHEMA = [
    "site_no",
    "site_name",
    "status",
    "drainage_area_sqmi",
    "period_start",
    "period_end",
    "record_years",
    "completeness_score",
    "roles",
    "frequency_basis",
    "wflow_submodel_id",
    "sfincs_domain_id",
    "sfincs_handoff_id",
    "review_status",
    "review_notes",
]
NULLABLE_REVIEWED_STREAMGAGE_FIELDS = {"sfincs_handoff_id", "review_notes"}


def build_wflow_build_plan(config, paths) -> wflow_types.WflowBuildPlan:
    """Return the notebook-facing HydroMT-Wflow build/update plan."""
    location_root = location_root_from_paths(paths)
    wflow = config.get("wflow", {})
    plugin = str(wflow.get("plugin", "wflow_sbm"))
    base_model_root = resolve_location_path(location_root, wflow.get("base_model_root", "data/wflow/base"))
    events_root = resolve_location_path(location_root, wflow.get("events_root", "data/wflow/events"))
    data_catalog = resolve_location_path(location_root, wflow.get("data_catalog", "data/wflow/data_catalog.yml"))
    build_config = resolve_location_path(location_root, wflow.get("build_config", "wflow_build.yml"))
    update_forcing_config = resolve_location_path(
        location_root,
        wflow.get("update_forcing_config", "wflow_update_forcing.yml"),
    )
    _ensure_model_recipe_file(config, "wflow_build", build_config)
    _ensure_model_recipe_file(config, "wflow_update_forcing", update_forcing_config)

    build_workflow = _read_workflow(build_config)
    update_workflow = _read_workflow(update_forcing_config)
    build_steps = _step_names(build_workflow)
    update_steps = _step_names(update_workflow)
    region_kind = _region_kind(build_workflow)
    domain_set = wflow.get("domain_set", {})
    domain_status = _domain_status(region_kind, domain_set)
    review_required = bool(domain_set.get("review_required", False)) or domain_status != "configured"

    return wflow_types.WflowBuildPlan(
        study_location=str(config.get("project", {}).get("name", paths.get("location_name", location_root.name))),
        plugin=plugin,
        base_model_root=base_model_root,
        events_root=events_root,
        data_catalog=data_catalog,
        build_config=build_config,
        update_forcing_config=update_forcing_config,
        build_steps=build_steps,
        update_steps=update_steps,
        region_kind=region_kind,
        review_required=review_required,
        domain_status=domain_status,
        build_command=(
            f"hydromt build {plugin} {base_model_root} "
            f"-i {build_config} -d {data_catalog} -vvv"
        ),
        update_command=(
            f"hydromt update {plugin} {base_model_root} "
            f"-i {update_forcing_config} -d {data_catalog} "
            f"-o {events_root / '<event_id>'} -vvv"
        ),
    )


def build_wflow_data_catalog(config, paths) -> Path:
    """Write the local HydroMT-Wflow catalog entries for one Location Workspace."""
    location_root = location_root_from_paths(paths)
    wflow = config.get("wflow", {})
    catalog_path = resolve_location_path(location_root, wflow.get("data_catalog", "data/wflow/data_catalog.yml"))
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    reference_crs = str(config.get("project", {}).get("reference_crs", "EPSG:4326"))
    study_location = str(config.get("project", {}).get("name", "") or location_root.name)
    streamgage_network = (
        config.get("collection", {})
        .get("usgs_streamgages", {})
        .get("reviewed_network", "data/sources/usgs_streamgages/streamgage_network.geojson")
    )
    event_precip = (
        config.get("collection", {})
        .get("aorc_sst", {})
        .get("event_precip", "data/wflow/events/<event_id>/precip.nc")
    )
    event_temp_pet = (
        (wflow.get("event_forcing", {}) or {})
        .get("temp_pet", {})
        .get(
            "event_temp_pet",
            config.get("collection", {})
            .get("aorc_sst", {})
            .get("event_temp_pet", "data/wflow/events/<event_id>/temp_pet.nc"),
        )
    )
    national_hydrography = config.get("collection", {}).get("national_hydrography", {})
    hydromt_basemap = national_hydrography.get(
        "hydromt_basemap",
        "data/wflow/hydrography/us_hydrography_basemap.nc",
    )
    river_geometry = national_hydrography.get(
        "river_geometry",
        "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg",
    )
    reservoirs_cfg = national_hydrography.get("reservoirs", {})
    reservoirs = reservoirs_cfg.get(
        "output",
        national_hydrography.get("reservoirs_output", "data/sources/national_hydrography/nhdplus_hr_wflow_reservoirs.gpkg"),
    )
    river_geometry_required = _wflow_build_uses_river_geometry(
        _ensure_model_recipe_file(
            config,
            "wflow_build",
            resolve_location_path(location_root, wflow.get("build_config", "wflow_build.yml")),
        )
    )
    sources = config.get("static_sources", {})
    wflow_extent = sources.get("wflow_collection_extent", {})
    ssurgo = sources.get("ssurgo", {})
    hsg = wflow_extent.get("hsg_output", ssurgo.get("hsg_output", "data/static/soils/hsg.tif"))
    ksat = wflow_extent.get("ksat_output", ssurgo.get("ksat_output", "data/static/soils/ksat_mmhr.tif"))
    landcover = wflow_extent.get(
        "landcover_output",
        sources.get("landcover", {}).get("output", "data/static/processed/landcover_region_setup.tif"),
    )
    soil_parameters = wflow.get("source_strategy", {}).get("soils", {}).get(
        "wflow_parameters",
        DEFAULT_US_WFLOW_SOURCE_STRATEGY["soils"]["wflow_parameters"],
    )
    ssurgo_wflow_soil_parameters = national_hydrography.get(
        "wflow_soil_parameters",
        "data/wflow/static/ssurgo_wflow_soil_parameters.nc",
    )
    strategy = plan_wflow_us_source_strategy(config)
    river_geometry_source = strategy.river_geometry_source or "nhdplus_hr_river_geometry"
    catalog = {
        "meta": {
            "roots": [".."],
            "source_policy": {
                "hydrography": strategy.hydrography_policy,
                "soils": strategy.soil_policy,
            },
            "global_fallback_dependencies": list(strategy.global_fallbacks),
        },
        strategy.hydromt_basemap_source: _raster_xarray_entry(
            resolve_location_path(location_root, hydromt_basemap).resolve().as_posix(),
            category="hydrography",
            status="review_required",
            required_for_build=True,
            required_variables=["flwdir", "elevtn", "uparea", "strord"],
        ),
        f"{study_location}_streamgage_network": {
            "data_type": "GeoDataFrame",
            "driver": {"name": "pyogrio"},
            "uri": resolve_location_path(location_root, streamgage_network).resolve().as_posix(),
            "metadata": {"crs": reference_crs, "category": "hydrography"},
        },
        river_geometry_source: {
            "data_type": "GeoDataFrame",
            "driver": {"name": "pyogrio"},
            "uri": resolve_location_path(location_root, river_geometry).resolve().as_posix(),
            "metadata": {
                "crs": reference_crs,
                "category": "hydrography",
                "required_for_build": river_geometry_required,
                "required_columns": ["rivwth", "rivdph", "qbankfull"],
            },
        },
        "nhdplus_hr_wflow_reservoirs": {
            "data_type": "GeoDataFrame",
            "driver": {"name": "pyogrio"},
            "uri": resolve_location_path(location_root, reservoirs).resolve().as_posix(),
            "metadata": {
                "crs": reference_crs,
                "category": "hydrography",
                "source": "USGS NHDPlus HR NHDWaterbody",
                "required_for_build": bool(reservoirs_cfg.get("enabled", False)),
                "required_columns": ["waterbody_id", "Area_avg", "Depth_avg", "Vol_avg", "Dis_avg"],
                "operation": reservoirs_cfg.get("operation", "no_control"),
                "status": "review_required_public_waterbody_estimates",
            },
        },
        # ESA WorldCover landcover collected in 01_region_setup. Named
        # `esa_worldcover` so HydroMT-Wflow setup_lulcmaps auto-resolves its
        # shipped `esa_worldcover_mapping_default` table. CRS is read from the
        # GeoTIFF; setup_lulcmaps requests this single band as `landuse`.
        "esa_worldcover": {
            "data_type": "RasterDataset",
            "driver": {"name": "rasterio"},
            "uri": resolve_location_path(location_root, landcover).resolve().as_posix(),
            "metadata": {
                "category": "landuse",
                "source": "ESA WorldCover (01_region_setup)",
                "required_for_build": True,
            },
        },
        "ssurgo_hydrologic_soil_group": _rasterio_entry(
            resolve_location_path(location_root, hsg).resolve().as_posix(),
            crs=reference_crs,
            category="soils",
            source="SSURGO",
        ),
        "ssurgo_saturated_conductivity": _rasterio_entry(
            resolve_location_path(location_root, ksat).resolve().as_posix(),
            crs=reference_crs,
            category="soils",
            source="SSURGO",
        ),
        soil_parameters: _raster_xarray_entry(
            resolve_location_path(location_root, ssurgo_wflow_soil_parameters).resolve().as_posix(),
            category="soils",
            status="review_required",
            required_for_build=True,
            derived_from=["ssurgo_hydrologic_soil_group", "ssurgo_saturated_conductivity"],
        ),
        "event_precip": _raster_xarray_entry(
            resolve_location_path(location_root, event_precip).resolve().as_posix(),
            category="event_forcing",
        ),
        "event_temp_pet": _raster_xarray_entry(
            resolve_location_path(location_root, event_temp_pet).resolve().as_posix(),
            category="event_forcing",
        ),
    }
    catalog = _normalize_catalog_metadata(catalog)
    catalog_path.write_text(
        _GENERATED_NOTICE.format(source="the Wflow data-catalog build")
        + yaml.safe_dump(catalog, sort_keys=False),
        encoding="utf-8",
    )
    return catalog_path


def _wflow_build_uses_river_geometry(build_config: Path) -> bool:
    if not Path(build_config).exists():
        return True
    workflow = yaml.safe_load(Path(build_config).read_text(encoding="utf-8")) or {}
    steps = workflow.get("steps") if isinstance(workflow.get("steps"), list) else [workflow]
    for step in steps:
        if not isinstance(step, dict) or "setup_rivers" not in step:
            continue
        river_geom = (step.get("setup_rivers") or {}).get("river_geom_fn")
        return river_geom not in (None, "")
    return False


def wflow_catalog_source_readiness(catalog_path: str | Path) -> list[dict]:
    """Return local-file readiness rows for a HydroMT-Wflow data catalog."""
    catalog_path = Path(catalog_path)
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    rows = []
    for source, entry in catalog.items():
        if source == "meta" or not isinstance(entry, dict):
            continue
        uri = entry.get("uri")
        metadata = entry.get("metadata") or {}
        if not uri:
            continue
        uri_text = str(uri)
        path = Path(uri_text)
        local_file = "://" not in uri_text and "<" not in uri_text and ">" not in uri_text
        exists = path.exists() if local_file else None
        rows.append(
            {
                "source": source,
                "data_type": entry.get("data_type"),
                "uri": uri_text,
                "local_file": local_file,
                "exists": exists,
                "required_for_build": bool(metadata.get("required_for_build", False)),
                "status": metadata.get("status"),
                "category": metadata.get("category"),
            }
        )
    return rows


def plan_wflow_us_source_strategy(config) -> wflow_types.WflowSourceStrategy:
    """Return the USA-first Wflow source strategy for notebook review."""
    project = config.get("project", {})
    wflow = config.get("wflow", {})
    strategy = _merged_source_strategy(wflow.get("source_strategy", {}))
    hydrography = strategy["hydrography"]
    soils = strategy["soils"]
    sources = config.get("static_sources", {})
    wflow_extent = sources.get("wflow_collection_extent", {})
    ssurgo = sources.get("ssurgo", {})
    ssurgo_inputs = tuple(
        value
        for value in (
            wflow_extent.get("hsg_output", ssurgo.get("hsg_output")),
            wflow_extent.get("ksat_output", ssurgo.get("ksat_output")),
        )
        if value
    )
    country = str(project.get("country", project.get("country_code", ""))).upper()
    issues = []
    if country and country not in {"US", "USA", "UNITED STATES", "UNITED STATES OF AMERICA"}:
        issues.append("USA-first source strategy selected for a non-USA Study Location")
    if hydrography.get("preferred") == "us_hydrography":
        issues.append(
            "HydroMT-Wflow setup_basemaps requires a local DEM-derived RasterDataset "
            "with flwdir, elevtn, uparea, and strord; prepare and review the LDD basemap "
            "before production."
        )
    if soils.get("preferred") == "ssurgo" and len(ssurgo_inputs) < 2:
        issues.append("SSURGO-first soil policy requires both HSG and Ksat rasters")
    if soils.get("preferred") == "ssurgo":
        issues.append(
            "SSURGO HSG/Ksat cover SFINCS infiltration evidence; Wflow SBM still needs "
            "reviewed soil parameter maps derived or augmented from those local soils."
        )
    global_fallbacks = tuple(
        value
        for value in (hydrography.get("fallback"), soils.get("fallback"))
        if value
    )

    return wflow_types.WflowSourceStrategy(
        status="review_required" if issues else "ready",
        hydrography_policy=(
            "us_hydrography_first"
            if hydrography.get("preferred") == "us_hydrography"
            else "configured"
        ),
        hydromt_basemap_source=str(hydrography.get("hydromt_basemap")),
        river_geometry_source=hydrography.get("river_geometry"),
        catchment_source=hydrography.get("catchments"),
        hydrography_api=str(hydrography.get("api")),
        soil_policy="ssurgo_first" if soils.get("preferred") == "ssurgo" else "configured",
        wflow_soil_parameter_source=str(soils.get("wflow_parameters")),
        ssurgo_inputs=ssurgo_inputs,
        global_fallbacks=global_fallbacks,
        issues=tuple(issues),
    )


def plan_wflow_domain_set_from_streamgages(config, paths) -> wflow_types.WflowDomainSetPlan:
    """Plan Wflow Submodels from the reviewed USGS Streamgage Network."""
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
    issues = []
    issues.extend(_schema_issues(gages))
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
        region_kind = "subbasin"
        hydromt_region = _hydromt_subbasin_region(
            outlet_xy,
            outlet,
            None,
        )
        submodels.append(
            {
                "wflow_submodel_id": submodel_id,
                "region_kind": region_kind,
                "region": hydromt_region,
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
    """Plan the Wflow Domain Set from the configured outlet source.

    ``wflow.domain_set.outlet_source`` selects how Wflow outlets are found:
    ``reviewed_streamgages`` (default) uses the reviewed USGS network; the gage-free
    ``stream_boundary_crossings`` derives them from where streams cross each SFINCS box.
    """
    outlet_source = str(config.get("wflow", {}).get("domain_set", {}).get("outlet_source", "reviewed_streamgages"))
    if outlet_source in {"boundary_handoff_watershed", "stream_boundary_watershed", "sfincs_boundary_watershed"}:
        return plan_wflow_domain_set_from_boundary_handoff_watersheds(config, paths)
    if outlet_source == "encompassing_huc":
        return plan_wflow_domain_set_from_encompassing_huc(config, paths)
    if outlet_source == "stream_boundary_crossings":
        return plan_wflow_domain_set_from_stream_boundary_crossings(config, paths)
    return plan_wflow_domain_set_from_streamgages(config, paths)


def plan_wflow_domain_set_from_stream_boundary_crossings(config, paths) -> wflow_types.WflowDomainSetPlan:
    """Plan Wflow Submodels from where streams cross each SFINCS coverage box.

    Gage-free: each inflow crossing is a Wflow subbasin outlet and the SFINCS discharge
    source it feeds. The contributing subbasin is delineated upstream of the crossing at
    build time from the DEM-derived LDD basemap, so the Wflow domain spans the true
    upstream watershed rather than only catchments overlapping the exposure footprint.
    """
    from coupling.domain_set import (
        stream_boundary_inflow_crossings,
        subbasin_submodels_from_crossings,
    )

    location_root = location_root_from_paths(paths)
    domain_set = config.get("wflow", {}).get("domain_set", {})
    crossings_cfg = domain_set.get("crossings", {})
    project_name = str(config.get("project", {}).get("name", location_root.name))
    min_uparea_km2 = float(crossings_cfg.get("min_uparea_km2", 5.0))

    coverage_path, boxes = _sfincs_coverage_boxes(config, location_root)
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
        return wflow_types.WflowDomainSetPlan(coverage_path, "missing_coverage_bbox", 0, 0, 0, (), (f"SFINCS coverage source has no polygons: {coverage_path}",))
    try:
        rivers = _load_crossing_rivers(config, location_root)
    except FileNotFoundError as exc:
        return wflow_types.WflowDomainSetPlan(coverage_path, "missing_river_geometry", 0, 0, 0, (), (str(exc),))

    submodels = []
    issues = []
    for index, row in boxes.iterrows():
        domain_id = _coverage_domain_id(row, project_name, int(index), len(boxes))
        crossings = stream_boundary_inflow_crossings(rivers, row.geometry, min_uparea_km2=min_uparea_km2)
        if crossings.empty:
            issues.append(f"{domain_id}: no stream/coverage-box inflow crossings above {min_uparea_km2} km2")
            continue
        submodels.extend(subbasin_submodels_from_crossings(crossings, project_name=project_name, sfincs_domain_id=domain_id))

    status = "ready" if submodels and not issues else "review_required"
    return wflow_types.WflowDomainSetPlan(
        reviewed_network=coverage_path,
        status=status,
        gage_count=0,
        submodel_count=len(submodels),
        handoff_count=len(submodels),
        submodels=tuple(submodels),
        issues=tuple(issues),
    )


def plan_wflow_domain_set_from_boundary_handoff_watersheds(config, paths) -> wflow_types.WflowDomainSetPlan:
    """Plan one Wflow watershed per SFINCS domain from its boundary inflow points.

    Unlike ``stream_boundary_crossings`` this does not create a Wflow model per crossing.
    Unlike ``encompassing_huc`` it does not keep the whole HUC that contains the SFINCS
    coverage box. The HydroMT build region is a documented ``subbasin`` region whose
    outlets are the stream/coverage-boundary inflow points that feed the SFINCS domain.
    """
    from coupling.domain_set import coverage_box_crossings

    location_root = location_root_from_paths(paths)
    domain_set = config.get("wflow", {}).get("domain_set", {})
    crossings_cfg = domain_set.get("crossings", {})
    project_name = str(config.get("project", {}).get("name", location_root.name))
    min_uparea_km2 = float(crossings_cfg.get("min_uparea_km2", 5.0))

    coverage_path, boxes = _sfincs_coverage_boxes(config, location_root)
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

    network_path = resolve_location_path(
        location_root,
        config.get("wflow", {})
        .get("streamgage_network", {})
        .get("reviewed_network", "data/sources/usgs_streamgages/streamgage_network.geojson"),
    )
    accepted_gages = _accepted_streamgages_frame(network_path)
    ignore_source_artifacts = bool(
        domain_set.get("ignore_sfincs_handoff_artifacts")
        or crossings_cfg.get("ignore_sfincs_handoff_artifacts")
    )
    artifact_submodels = (
        []
        if ignore_source_artifacts
        else _boundary_handoff_submodels_from_source_artifacts(
            config,
            location_root,
            boxes,
            accepted_gages=accepted_gages,
            project_name=project_name,
            min_uparea_km2=min_uparea_km2,
        )
    )
    if artifact_submodels:
        return wflow_types.WflowDomainSetPlan(
            reviewed_network=coverage_path,
            status="ready",
            gage_count=len(accepted_gages),
            submodel_count=len(artifact_submodels),
            handoff_count=sum(len(submodel["handoff_points"]) for submodel in artifact_submodels),
            submodels=tuple(artifact_submodels),
            issues=(),
        )

    try:
        rivers = _load_crossing_rivers(config, location_root)
    except FileNotFoundError as exc:
        return wflow_types.WflowDomainSetPlan(coverage_path, "missing_river_geometry", 0, 0, 0, (), (str(exc),))

    crossings = coverage_box_crossings(
        boxes,
        rivers,
        project_name=project_name,
        min_uparea_km2=min_uparea_km2,
    )
    if crossings.empty:
        return wflow_types.WflowDomainSetPlan(
            coverage_path,
            "review_required",
            0,
            0,
            0,
            (),
            (f"no stream/coverage-box inflow crossings above {min_uparea_km2} km2",),
        )

    submodels = []
    gages_by_domain = _accepted_streamgages_by_sfincs_domain(accepted_gages, boxes, project_name)
    domain_geometries = _coverage_domain_geometries(boxes, project_name)
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
        handoff_region = _hydromt_boundary_handoff_subbasin_region(handoff_points, min_uparea_km2=min_uparea_km2)
        reviewed_region, subbasin_geometry, watershed_source = _reviewed_wflow_watershed_region(
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

    return wflow_types.WflowDomainSetPlan(
        reviewed_network=coverage_path,
        status="ready",
        gage_count=len(accepted_gages),
        submodel_count=len(submodels),
        handoff_count=sum(len(submodel["handoff_points"]) for submodel in submodels),
        submodels=tuple(submodels),
        issues=(),
    )


def _boundary_handoff_submodels_from_source_artifacts(
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
        _coverage_domain_id(box_row, project_name, int(index), count)
        for index, box_row in boxes.reset_index(drop=True).iterrows()
    }
    points_by_domain = _source_artifact_handoff_points_by_domain(
        config,
        location_root,
        boxes,
        project_name=project_name,
    )
    if not points_by_domain or set(points_by_domain) != domain_ids:
        return []

    submodels = []
    gages_by_domain = _accepted_streamgages_by_sfincs_domain(accepted_gages, boxes, project_name)
    domain_geometries = _coverage_domain_geometries(boxes, project_name)
    for domain_id in sorted(points_by_domain):
        handoff_points = points_by_domain[domain_id]
        domain_gages = gages_by_domain.get(str(domain_id), [])
        handoff_region = _hydromt_boundary_handoff_subbasin_region(handoff_points, min_uparea_km2=min_uparea_km2)
        reviewed_region, subbasin_geometry, watershed_source = _reviewed_wflow_watershed_region(
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


def _accepted_streamgages_by_sfincs_domain(
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
        _coverage_domain_id(box_row, project_name, int(index), len(boxes))
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
                domain_id = _coverage_domain_id(box_row, project_name, int(index), len(boxes_wgs))
                by_domain.setdefault(domain_id, []).append(gage)
    return by_domain


def _source_artifact_handoff_points_by_domain(
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
        _coverage_domain_id(box_row, project_name, int(index), count)
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


def _hydromt_boundary_handoff_subbasin_region(
    handoff_points: list[dict],
    *,
    min_uparea_km2: float | None,
) -> dict:
    """Return the fallback HydroMT subbasin region for one SFINCS domain.

    The grouped Wflow/SFINCS coupling mode keeps all SFINCS boundary crossings as
    candidate handoff gauges. When a reviewed full-watershed geometry is unavailable, use
    the dominant upstream-area handoff as a single HydroMT outlet; never use every SFINCS
    source as a multi-outlet ``subbasin`` region.
    """
    if not handoff_points:
        raise ValueError("Boundary-handoff Wflow region requires at least one handoff point")
    point = _primary_handoff_subbasin_point(handoff_points)
    region = {"subbasin": [float(point["lon"]), float(point["lat"])]}
    if min_uparea_km2 is not None and float(min_uparea_km2) > 0:
        region["uparea"] = float(min_uparea_km2)
    return region


def _coverage_domain_geometries(boxes: gpd.GeoDataFrame, project_name: str) -> dict[str, object]:
    count = len(boxes)
    return {
        _coverage_domain_id(box_row, project_name, int(index), count): box_row.geometry
        for index, box_row in boxes.reset_index(drop=True).iterrows()
        if box_row.geometry is not None and not box_row.geometry.is_empty
    }


def _reviewed_wflow_watershed_region(
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


def _configured_reference_gage_site_nos(config: dict) -> set[str]:
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


def _coverage_domain_id(row, project_name: str, index: int, count: int) -> str:
    subregion_id = row.get("subregion_id")
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


def _sfincs_coverage_boxes(config, location_root: Path) -> tuple[Path, gpd.GeoDataFrame]:
    """Return the SFINCS hydraulic coverage boxes that Wflow must enclose.

    Wflow HUC domains are chosen from the same exposure source and domain-count
    setting as SFINCS, so a single-SFINCS Austin setup cannot accidentally keep
    planning six stale SMART-DS component boxes from ``data/static/aoi/bbox.geojson``.
    """
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
        domain_id = _coverage_domain_id({"subregion_id": record.get("source_subregion_id")}, project_name, index, count)
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
        rows = [
            row
            for row in rows
            if row["subregion_id"] in include_domain_ids
        ]
    return coverage_path, gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _included_sfincs_domain_ids(domain_set) -> set[str]:
    values = domain_set.get("include_domain_ids", ())
    return {str(value).strip() for value in values if str(value).strip()}


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
        return sorted(records, key=lambda record: (str(record["source_subregion_id"] or ""), record["geometry"].centroid.x, record["geometry"].centroid.y))

    geometry = source.geometry.union_all()
    if geometry.is_empty:
        return []
    polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    return [
        {"source_subregion_id": None, "geometry": polygon}
        for polygon in sorted(polygons, key=lambda geom: (geom.centroid.x, geom.centroid.y))
    ]


def plan_wflow_domain_set_from_encompassing_huc(config, paths) -> wflow_types.WflowDomainSetPlan:
    """Plan one Wflow basin per SFINCS coverage box from the smallest encapsulating WBD HUC.

    Each coverage box gets its own HUC domain (a single HUC, or -- for a box straddling a
    basin divide, like Greensboro's Roanoke/Cape Fear split -- the union of the HUCs it
    intersects). The box's stream crossings are the SFINCS discharge sources / Wflow gauges
    within that basin. HUC boundaries come from the national WBD, so a domain is not capped by
    whatever NHDPlus extent happened to be fetched. A combined union of all per-box HUCs is
    also written for the region-setup watershed/envelope plot.
    """
    from coupling.domain_set import coverage_box_crossings, coverage_domain_id, select_encompassing_huc

    location_root = location_root_from_paths(paths)
    domain_set = config.get("wflow", {}).get("domain_set", {})
    crossings_cfg = domain_set.get("crossings", {})
    huc_cfg = domain_set.get("huc", {})
    project_name = str(config.get("project", {}).get("name", location_root.name))
    min_uparea_km2 = float(crossings_cfg.get("min_uparea_km2", 5.0))
    levels = tuple(int(level) for level in huc_cfg.get("levels", (8, 6, 4)))
    allow_union = bool(huc_cfg.get("allow_union", True))

    coverage_path, boxes = _sfincs_coverage_boxes(config, location_root)
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
        return wflow_types.WflowDomainSetPlan(coverage_path, "missing_coverage_bbox", 0, 0, 0, (), (f"SFINCS coverage source has no polygons: {coverage_path}",))
    try:
        rivers = _load_crossing_rivers(config, location_root)
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
    artifact_handoff_points = _source_artifact_handoff_points_by_domain(
        config,
        location_root,
        boxes,
        project_name=project_name,
    )
    if crossings.empty and not artifact_handoff_points:
        return wflow_types.WflowDomainSetPlan(coverage_path, "review_required", 0, 0, 0, (), (f"no stream/coverage-box inflow crossings above {min_uparea_km2} km2",))

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
                box_row.geometry, _wbd_huc_loader(config, location_root, box_row.geometry), levels=levels, allow_union=allow_union
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
        return wflow_types.WflowDomainSetPlan(coverage_path, "review_required", 0, 0, 0, (), tuple(issues) or ("no Wflow HUC domains could be formed",))

    combined_path = resolve_location_path(location_root, huc_cfg.get("output", "data/wflow/wflow_domain_huc.geojson"))
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(combined_rows, geometry="geometry", crs="EPSG:4326").to_file(combined_path, driver="GeoJSON")

    status = "ready" if not issues else "review_required"
    handoff_count = sum(len(submodel["handoff_points"]) for submodel in submodels)
    return wflow_types.WflowDomainSetPlan(coverage_path, status, len(accepted_gages), len(submodels), handoff_count, tuple(submodels), tuple(issues))


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


def _load_crossing_rivers(config, location_root: Path) -> gpd.GeoDataFrame:
    """Load the stream network with a normalised ``uparea`` (km2) column.

    Uses the NHDPlus HR river geometry collected in 02_collect_sources; this carries
    a drainage-area attribute and exists before Wflow is built (the domain must be
    chosen before the model exists).
    """
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


def write_wflow_domain_set_manifest(plan: wflow_types.WflowDomainSetPlan, config, paths) -> Path:
    """Write the reviewed Wflow-SFINCS Domain Set manifest."""
    if plan.status != "ready":
        raise ValueError(f"Wflow Domain Set plan is not ready: {plan.status}")
    location_root = location_root_from_paths(paths)
    wflow = config.get("wflow", {})
    submodels = _manifest_submodels_from_active_handoff_sources(plan, config, location_root)
    manifest_path = resolve_location_path(
        location_root,
        wflow.get("domain_set_manifest", "data/wflow/domain_set.yaml"),
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sfincs_domains_by_handoff = _sfincs_domain_ids_by_handoff(config, location_root)
    manifest = {
        "event_catalog_scope": wflow.get("domain_set", {}).get(
            "event_catalog_scope",
            "shared_across_domain_set",
        ),
        "reviewed_network": relative_to_or_absolute(plan.reviewed_network, location_root),
        "subbasin_fabric": wflow.get("domain_set", {}).get(
            "subbasin_fabric",
            "data/wflow/domain_set_subbasins.gpkg",
        ),
        "handoff": {
            "source_variable": wflow.get("handoff", {}).get("source_variable", "river_q"),
            "source_standard_name": wflow.get("handoff", {}).get(
                "source_standard_name",
                "river_water__volume_flow_rate",
            ),
            "target": wflow.get("handoff", {}).get("target", "sfincs_discharge_forcing"),
            "sfincs_boundary_type": "discharge",
        },
        "submodels": [
            {
                "wflow_submodel_id": submodel["wflow_submodel_id"],
                "hydromt_region": submodel["region"],
                "handoff_outlet_region": submodel.get("outlet_region", submodel["region"]),
                "region_kind": submodel.get("region_kind"),
                "subbasin_geometry": relative_to_or_absolute(Path(submodel["subbasin_geometry"]), location_root)
                if submodel.get("subbasin_geometry")
                else None,
                "sfincs_domain_ids": _manifest_sfincs_domain_ids(submodel, sfincs_domains_by_handoff),
                "sfincs_handoff_ids": list(submodel["sfincs_handoff_ids"]),
                "sfincs_boundary_ids": list(submodel["sfincs_handoff_ids"]),
                "gauge_site_nos": list(submodel["gauge_site_nos"]),
                "frequency_basis": list(submodel["frequency_basis"]),
                "role_counts": dict(submodel["role_counts"]),
            }
            for submodel in submodels
        ],
    }
    manifest_path.write_text(
        _GENERATED_NOTICE.format(source="the Wflow domain-set build")
        + yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path


def _manifest_submodels_from_active_handoff_sources(
    plan: wflow_types.WflowDomainSetPlan,
    config: dict,
    location_root: Path,
) -> tuple[dict, ...]:
    """Use generated SFINCS source artifacts as the final stream-boundary handoff IDs."""
    if config.get("wflow", {}).get("domain_set", {}).get("ignore_sfincs_handoff_artifacts"):
        return plan.submodels
    locations = read_stream_boundary_handoff_location_artifacts(
        config,
        location_root,
        location_path=resolve_location_path,
    )
    if locations is None or locations.empty or "wflow_submodel_id" not in locations:
        return plan.submodels

    by_submodel: dict[str, gpd.GeoDataFrame] = {}
    for submodel_id, group in locations.groupby(locations["wflow_submodel_id"].astype(str), sort=True):
        by_submodel[str(submodel_id)] = group.copy()
    if not by_submodel:
        return plan.submodels

    synced = []
    for submodel in plan.submodels:
        submodel_id = str(submodel.get("wflow_submodel_id", ""))
        group = by_submodel.get(submodel_id)
        if group is None or group.empty:
            synced.append(submodel)
            continue
        group = group.sort_values("sfincs_handoff_id").reset_index(drop=True)
        ids = [str(value) for value in group["sfincs_handoff_id"]]
        updated = dict(submodel)
        updated["sfincs_handoff_ids"] = ids
        updated["sfincs_boundary_ids"] = ids
        if "sfincs_domain_id" in group:
            updated["sfincs_domain_ids"] = _sorted_values(group["sfincs_domain_id"].astype(str))
        uparea_col = "uparea_km2" if "uparea_km2" in group else "uparea" if "uparea" in group else None
        group_wgs = group.to_crs("EPSG:4326") if group.crs is not None else group
        updated["handoff_points"] = [
            {
                "sfincs_handoff_id": str(row["sfincs_handoff_id"]),
                "sfincs_domain_id": str(row.get("sfincs_domain_id", "")),
                "lon": float(row.geometry.x),
                "lat": float(row.geometry.y),
                "uparea_km2": float(row[uparea_col]) if uparea_col and pd.notna(row[uparea_col]) else float("nan"),
            }
            for _, row in group_wgs.iterrows()
        ]
        synced.append(updated)
    return tuple(synced)


def _manifest_sfincs_domain_ids(submodel: dict, sfincs_domains_by_handoff: dict[str, tuple[str, ...]]) -> list[str]:
    domain_ids = set()
    for handoff_id in submodel.get("sfincs_handoff_ids", ()):
        domain_ids.update(sfincs_domains_by_handoff.get(str(handoff_id), ()))
    if domain_ids:
        return sorted(domain_ids)
    return list(submodel["sfincs_domain_ids"])


def _sfincs_domain_ids_by_handoff(config, location_root: Path) -> dict[str, tuple[str, ...]]:
    manifest_value = config.get("sfincs_domain_set", {}).get(
        "domain_manifest",
        "data/sfincs/domains/domain_set.yaml",
    )
    manifest_path = resolve_location_path(location_root, manifest_value)
    if not manifest_path.exists():
        return {}
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    handoff_domains: dict[str, set[str]] = {}
    for domain in manifest.get("domains", []):
        domain_id = domain.get("sfincs_domain_id")
        if not domain_id:
            continue
        for handoff_id in domain.get("handoff_source_ids", []):
            handoff_domains.setdefault(str(handoff_id), set()).add(str(domain_id))
    return {
        handoff_id: tuple(sorted(domain_ids))
        for handoff_id, domain_ids in handoff_domains.items()
    }


def write_wflow_subbasin_fabric_from_nhdplus(config, paths) -> dict:
    """Write reviewed Wflow subbasin polygons from NHDPlus HR catchments.

    Outlets come from the configured outlet source (reviewed gages or stream/coverage-box
    crossings); each is routed upstream through the NHDPlus network, so the resulting
    fabric -- and the Wflow collection envelope derived from it -- spans the full
    contributing watershed rather than only catchments overlapping the footprint.
    """
    location_root = location_root_from_paths(paths)
    plan = plan_wflow_domain_set(config, paths)
    if plan.status != "ready":
        raise ValueError(f"Wflow Domain Set plan is not ready: {plan.status}")

    collection = config.get("collection", {}).get("national_hydrography", {})
    wflow = config.get("wflow", {})
    catchments_path = resolve_location_path(
        location_root,
        collection.get("catchments", "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg"),
    )
    rivers_path = resolve_location_path(
        location_root,
        collection.get("river_geometry", "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"),
    )
    output_path = resolve_location_path(
        location_root,
        wflow.get("domain_set", {}).get("subbasin_fabric", "data/wflow/domain_set_subbasins.gpkg"),
    )
    diagnostics_path = resolve_location_path(
        location_root,
        wflow.get("domain_set", {}).get("subbasin_fabric_diagnostics", "data/wflow/readiness/nhdplus_subbasin_fabric.csv"),
    )
    if not catchments_path.exists():
        raise FileNotFoundError(catchments_path)
    if not rivers_path.exists():
        raise FileNotFoundError(rivers_path)

    catchments = gpd.read_file(catchments_path).to_crs("EPSG:4326")
    rivers = gpd.read_file(rivers_path).to_crs("EPSG:4326")
    if catchments.empty:
        raise ValueError(f"NHDPlus HR catchments artifact has no features: {catchments_path}")
    if rivers.empty:
        raise ValueError(f"NHDPlus HR river geometry artifact has no features: {rivers_path}")

    handoff = wflow.get("handoff", {})
    rows = []
    diagnostics = []
    for submodel in plan.submodels:
        selected_catchments, outlet_matches, method = _select_submodel_upstream_nhdplus_catchments(
            submodel,
            rivers,
            catchments,
        )
        geometry = selected_catchments.geometry.union_all()
        nhdplus_area_km2 = _equal_area_km2(geometry)
        reviewed_area_km2 = _reviewed_submodel_handoff_area_km2(submodel)
        area_difference_pct = _area_difference_pct(nhdplus_area_km2, reviewed_area_km2)
        area_match_status = _area_match_status(area_difference_pct)
        review_status = (
            "review_required_nhdplus_upstream"
            if "routed_upstream_catchments" in method
            else "review_required_nearest_catchment_only"
        )
        representative = outlet_matches[0]
        rows.append(
            {
                "wflow_submodel_id": submodel["wflow_submodel_id"],
                "region_role": "handoff_drainage",
                "sfincs_handoff_ids": ",".join(submodel["sfincs_handoff_ids"]),
                "sfincs_boundary_id": ",".join(submodel["sfincs_handoff_ids"]),
                "sfincs_boundary_type": "discharge",
                "sfincs_forcing_target": handoff.get("target", "sfincs_discharge_forcing"),
                "wflow_source_variable": handoff.get("source_variable", "river_q"),
                "wflow_source_standard_name": handoff.get(
                    "source_standard_name",
                    "river_water__volume_flow_rate",
                ),
                "gauge_site_nos": ",".join(submodel["gauge_site_nos"]),
                "outlet_lon": float(representative["lon"]),
                "outlet_lat": float(representative["lat"]),
                "catchment_count": int(len(selected_catchments)),
                "nhdplus_area_km2": round(nhdplus_area_km2, 3),
                "reviewed_drainage_area_km2": round(float(reviewed_area_km2), 3) if reviewed_area_km2 is not None else float("nan"),
                "area_difference_pct": round(area_difference_pct, 3) if area_difference_pct is not None else float("nan"),
                "area_match_status": area_match_status,
                "aggregation_method": method,
                "review_status": review_status,
                "source": "USGS NHDPlus HR NHDPlusCatchment",
                "geometry": geometry,
            }
        )
        for match in outlet_matches:
            diagnostics.append(
                {
                    "wflow_submodel_id": submodel["wflow_submodel_id"],
                    "sfincs_handoff_id": match["sfincs_handoff_id"],
                    "outlet_lon": float(match["lon"]),
                    "outlet_lat": float(match["lat"]),
                    "matched_river_index": int(match["river_index"]),
                    "matched_catchment_index": int(match["catchment_index"]),
                    "river_snap_distance_m": float(match["river_distance_m"]),
                    "catchment_match_distance_m": float(match["catchment_distance_m"]),
                    "catchment_count": int(len(selected_catchments)),
                    "nhdplus_area_km2": round(nhdplus_area_km2, 3),
                    "reviewed_drainage_area_km2": round(float(reviewed_area_km2), 3) if reviewed_area_km2 is not None else float("nan"),
                    "area_difference_pct": round(area_difference_pct, 3) if area_difference_pct is not None else float("nan"),
                    "area_match_status": area_match_status,
                    "aggregation_method": method,
                    "review_status": review_status,
                }
            )

    handoff_rows = list(rows)
    submodel_count = len(handoff_rows)
    handoff_catchment_count = int(sum(int(row["catchment_count"]) for row in handoff_rows))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    fabric = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    fabric.to_file(output_path, driver="GPKG")
    fabric_dir = _subbasin_fabric_directory(output_path)
    fabric_dir.mkdir(parents=True, exist_ok=True)
    member_ids = {str(value) for value in fabric["wflow_submodel_id"]}
    for stale_path in fabric_dir.glob("*.geojson"):
        if stale_path.stem not in member_ids:
            stale_path.unlink()
    subbasin_geometry_files = []
    coverage_geometry_file = None
    for _, row in fabric.iterrows():
        member_id = str(row["wflow_submodel_id"])
        geojson_path = fabric_dir / f"{member_id}.geojson"
        gpd.GeoDataFrame([row], geometry="geometry", crs=fabric.crs).to_file(geojson_path, driver="GeoJSON")
        if row.get("region_role") == "evaluation_coverage":
            coverage_geometry_file = geojson_path
        else:
            subbasin_geometry_files.append(geojson_path)
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)
    result = {
        "subbasin_fabric": output_path,
        "subbasin_geometry_files": tuple(subbasin_geometry_files),
        "diagnostics_csv": diagnostics_path,
        "submodel_count": submodel_count,
        "catchment_count": handoff_catchment_count,
        "area_mismatch_count": int(sum(1 for row in handoff_rows if row.get("area_match_status") == "review_required_area_mismatch")),
        "area_mismatch_submodels": tuple(
            str(row["wflow_submodel_id"])
            for row in handoff_rows
            if row.get("area_match_status") == "review_required_area_mismatch"
        ),
        "statuses": tuple(sorted(fabric["review_status"].unique())),
        "coverage_region": str(coverage_geometry_file) if coverage_geometry_file else None,
        "coverage_catchment_count": 0,
        "coverage_status": "handoff_watershed_only",
    }
    return result


def _select_submodel_upstream_nhdplus_catchments(submodel: dict, rivers, catchments):
    selections = []
    matches = []
    methods = []
    for handoff in _submodel_handoff_points(submodel):
        outlet = gpd.GeoDataFrame(
            {"_outlet": [handoff["sfincs_handoff_id"]]},
            geometry=[Point(float(handoff["lon"]), float(handoff["lat"]))],
            crs="EPSG:4326",
        )
        match = _match_nhdplus_outlet(outlet, rivers, catchments)
        selected, method = _select_upstream_nhdplus_catchments(
            rivers,
            catchments,
            match["river_index"],
            match["catchment_index"],
        )
        selections.append(selected)
        methods.append(method)
        matches.append({**handoff, **match})
    if not selections:
        raise ValueError(f"Wflow Submodel {submodel.get('wflow_submodel_id')} has no handoff outlet points")
    selected_catchments = gpd.GeoDataFrame(
        pd.concat(selections, ignore_index=True),
        geometry="geometry",
        crs=catchments.crs,
    )
    id_col = _find_column(selected_catchments, ("featureid", "nhdplusid", "comid", "gridcode"))
    if id_col:
        selected_catchments = selected_catchments.drop_duplicates(subset=[id_col]).copy()
    else:
        selected_catchments = selected_catchments.drop_duplicates(subset=["geometry"]).copy()
    method = methods[0] if len(set(methods)) == 1 else "union_" + "_and_".join(sorted(set(methods)))
    return selected_catchments, matches, method


def _submodel_handoff_points(submodel: dict) -> list[dict]:
    points = submodel.get("handoff_points")
    if points:
        return [
            {
                "sfincs_handoff_id": str(point["sfincs_handoff_id"]),
                "lon": float(point["lon"]),
                "lat": float(point["lat"]),
                "uparea_km2": float(point["uparea_km2"]) if point.get("uparea_km2") is not None else np.nan,
            }
            for point in points
        ]

    outlet_xy = (submodel.get("outlet_region") or submodel.get("region") or {}).get("subbasin")
    if not outlet_xy:
        return []
    handoff_id = next(
        (str(value) for value in submodel.get("sfincs_handoff_ids", ()) if value),
        str(submodel.get("wflow_submodel_id", "")),
    )
    uparea = (submodel.get("region") or {}).get("uparea")
    return [
        {
            "sfincs_handoff_id": handoff_id,
            "lon": float(outlet_xy[0]),
            "lat": float(outlet_xy[1]),
            "uparea_km2": float(uparea) if uparea is not None else np.nan,
        }
    ]


def _reviewed_submodel_handoff_area_km2(submodel: dict) -> float | None:
    points = _submodel_handoff_points(submodel)
    areas = [
        float(point["uparea_km2"])
        for point in points
        if point.get("uparea_km2") is not None and not pd.isna(point.get("uparea_km2"))
    ]
    if not areas:
        return None
    return float(sum(areas)) if len(areas) > 1 else float(areas[0])


def _area_difference_pct(nhdplus_area_km2, reviewed_area_km2) -> float | None:
    if reviewed_area_km2 in (None, "") or pd.isna(reviewed_area_km2):
        return None
    reviewed_area_km2 = float(reviewed_area_km2)
    if reviewed_area_km2 <= 0:
        return None
    return abs(float(nhdplus_area_km2) - reviewed_area_km2) / reviewed_area_km2 * 100.0


def _area_match_status(area_difference_pct: float | None) -> str:
    if area_difference_pct is None:
        return "missing_reviewed_area"
    if area_difference_pct > 50.0:
        return "review_required_area_mismatch"
    return "within_review_tolerance"


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

    boundary_locations = _sfincs_boundary_handoff_locations(
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


def _sfincs_boundary_handoff_locations(
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


def write_wflow_crossing_gauge_locations(config, paths, submodel: dict, *, output=None) -> dict:
    """Write the HydroMT-Wflow gauge for a crossing-derived (gage-free) submodel.

    The gauge is the inflow crossing itself, which is also the subbasin outlet, so no
    reviewed gage network is read. Wflow reports ``river_q`` here and hands it to SFINCS
    as the discharge boundary source. ``uparea`` (from the crossing) lets HydroMT snap the
    gauge to the LDD cell with the matching drainage area.
    """
    location_root = location_root_from_paths(paths)
    points = submodel.get("handoff_points")
    if points:
        # Encompassing-HUC basin: one gauge per crossing it feeds.
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
        # Per-crossing subbasin: the gauge is the single outlet (= the crossing).
        region = submodel.get("region", {}) or {}
        outlet_xy = (submodel.get("outlet_region") or region).get("subbasin") or region.get("subbasin")
        if not outlet_xy:
            raise ValueError(f"Wflow Submodel {submodel.get('wflow_submodel_id')} has no crossing outlet for a gauge")
        handoff_id = next((str(value) for value in submodel.get("sfincs_handoff_ids", ()) if value), str(submodel.get("wflow_submodel_id")))
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


def write_wflow_observation_gauge_locations(config, paths, submodel: dict, *, output=None) -> dict:
    """Write all reviewed streamgages in the submodel as Wflow observation gauges.

    Unlike :func:`write_wflow_sfincs_gauge_locations` (which keeps only the SFINCS
    discharge handoff source points), this keeps every accepted active reviewed gage that
    belongs to the submodel so Wflow reports modelled discharge at the full streamgage
    network for calibration and validation. SFINCS coupling is unaffected.
    """
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


def _write_wflow_sfincs_handoff_gauge_locations(config, paths, submodel: dict) -> dict:
    """Write Wflow gauges at the active SFINCS handoff geometry for this submodel."""
    domain_set = config.get("wflow", {}).get("domain_set", {})
    outlet_source = str(domain_set.get("outlet_source", "reviewed_streamgages"))
    if domain_set.get("ignore_sfincs_handoff_artifacts"):
        return write_wflow_crossing_gauge_locations(config, paths, submodel)
    if outlet_source in {"stream_boundary_crossings", "boundary_handoff_watershed", "stream_boundary_watershed", "sfincs_boundary_watershed"}:
        location_root = location_root_from_paths(paths)
        handoff_ids = {str(value) for value in submodel.get("sfincs_handoff_ids", ()) if value}
        if _sfincs_boundary_handoff_locations(
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
        if handoff_ids and _sfincs_boundary_handoff_locations(
            config,
            location_root,
            handoff_ids,
            submodel_id=str(submodel.get("wflow_submodel_id", "")),
        ) is not None:
            return write_wflow_sfincs_gauge_locations(config, paths, submodel)
        return write_wflow_crossing_gauge_locations(config, paths, submodel)
    return write_wflow_sfincs_gauge_locations(config, paths, submodel)


def build_wflow_steps_for_submodel(
    build_config: Path,
    submodel: dict,
    *,
    gauges_fn=None,
    sfincs_snap_to_river: bool = True,
    sfincs_snap_uparea: bool = True,
    obs_gauges_fn=None,
    obs_snap_to_river: bool = True,
    obs_snap_uparea: bool = True,
    handoff_config: dict | None = None,
) -> list[dict]:
    """Return HydroMT-Wflow build steps with the reviewed submodel region.

    ``gauges_fn`` adds the SFINCS discharge handoff source points (basename ``sfincs``),
    whose Wflow output is handed to SFINCS as boundary inflow. ``obs_gauges_fn`` adds the
    full reviewed streamgage network as a separate observation/calibration gauge layer
    (basename ``usgs``) so Wflow reports discharge at every reviewed gage without making
    them SFINCS sources.
    """
    submodel = dict(submodel)
    if gauges_fn is not None:
        submodel["gauges_fn"] = gauges_fn
    if obs_gauges_fn is not None:
        submodel["observation_gauges_fn"] = obs_gauges_fn
    config = {"wflow": {"handoff": dict(handoff_config or {})}}
    return render_hydromt_build_steps(
        config,
        Path(),
        build_config,
        submodel,
        sfincs_snap_to_river=sfincs_snap_to_river,
        sfincs_snap_uparea=sfincs_snap_uparea,
        obs_snap_to_river=obs_snap_to_river,
        obs_snap_uparea=obs_snap_uparea,
    )


def build_wflow_submodel(
    config,
    paths,
    *,
    submodel_id: str | None = None,
    model_cls=None,
    force: bool = False,
    write_catalog: bool = True,
) -> dict:
    """Build one reviewed HydroMT-Wflow submodel for a Location Workspace."""
    location_root = location_root_from_paths(paths)
    build_plan = build_wflow_build_plan(config, paths)
    domain_plan = plan_wflow_domain_set(config, paths)
    if domain_plan.status != "ready":
        raise RuntimeError(f"Wflow Domain Set plan is not ready: {domain_plan.status}: {domain_plan.issues}")
    submodel = _select_wflow_submodel(domain_plan.submodels, submodel_id)
    selected_id = str(submodel["wflow_submodel_id"])
    model_root = build_plan.base_model_root / selected_id
    apply_repairs = _legacy_wflow_repairs_enabled(config)
    if (
        _is_built_wflow_model(model_root)
        and not force
        and _sfincs_gauge_layer_matches(model_root, submodel, config=config, paths=paths)
        and _observation_gauge_layer_matches(model_root, submodel, config)
    ):
        wflow_reservoirs.assert_wflow_reservoir_staticmaps_current(config, model_root, selected_id)
        wflow_repairs.normalize_wflow_staticmaps_nodata(model_root)
        if apply_repairs:
            wflow_repairs.repair_wflow_river_width(model_root)
            wflow_repairs.repair_wflow_canopy_parameters(model_root)
            wflow_repairs.repair_wflow_gauge_map(model_root)
        qa = _wflow_staticmap_qa(model_root, config)
        catalog_path = build_wflow_data_catalog(config, paths) if write_catalog else build_plan.data_catalog
        model = staticmaps_qa.open_wflow_model(model_root, catalog_path, model_cls=model_cls, mode="r")
        return {
            "status": "reused",
            "wflow_submodel_id": selected_id,
            "base_model_root": model_root,
            "data_catalog": catalog_path,
            "built": True,
            "staticmap_qa_status": staticmaps_qa.qa_status(qa),
            "model": model,
        }
    if _is_built_wflow_model(model_root) and not force:
        force = True
    if force and model_root.exists():
        shutil.rmtree(model_root)

    wflow_repairs.ensure_wflow_hydrography_basemap_nodata(config, paths)
    catalog_path = build_wflow_data_catalog(config, paths) if write_catalog else build_plan.data_catalog
    missing_required = [
        row
        for row in wflow_catalog_source_readiness(catalog_path)
        if row["required_for_build"] and row["local_file"] and row["exists"] is False
    ]
    if missing_required:
        raise FileNotFoundError(
            "Missing required HydroMT-Wflow source files before build: "
            + json.dumps(
                [{"source": row["source"], "uri": row["uri"]} for row in missing_required],
                indent=2,
            )
        )

    # Crossing gauges seed the first HUC build; once SFINCS writes boundary
    # source points, those exact points become the Wflow gauges handed to SFINCS.
    outlet_source = str(config.get("wflow", {}).get("domain_set", {}).get("outlet_source", "reviewed_streamgages"))
    gauge_summary = _write_wflow_sfincs_handoff_gauge_locations(config, paths, submodel)
    if outlet_source in {"stream_boundary_crossings", "boundary_handoff_watershed", "stream_boundary_watershed", "sfincs_boundary_watershed"}:
        obs_gauge_summary = (
            write_wflow_observation_gauge_locations(config, paths, submodel)
            if submodel.get("gauge_site_nos")
            else None
        )
    elif outlet_source == "encompassing_huc":
        obs_gauge_summary = (
            write_wflow_observation_gauge_locations(config, paths, submodel)
            if submodel.get("gauge_site_nos")
            else None
        )
    else:
        gauge_summary = write_wflow_sfincs_gauge_locations(config, paths, submodel)
        obs_gauge_summary = write_wflow_observation_gauge_locations(config, paths, submodel)
    steps = build_wflow_steps_for_submodel(
        build_plan.build_config,
        submodel,
        gauges_fn=gauge_summary["gauges_fn"],
        sfincs_snap_to_river=bool(gauge_summary.get("snap_to_river", True)),
        sfincs_snap_uparea=bool(gauge_summary.get("snap_uparea", True)),
        obs_gauges_fn=obs_gauge_summary["gauges_fn"] if obs_gauge_summary else None,
        obs_snap_to_river=obs_gauge_summary.get("snap_to_river", True) if obs_gauge_summary else True,
        obs_snap_uparea=obs_gauge_summary["snap_uparea"] if obs_gauge_summary else True,
        handoff_config=config.get("wflow", {}).get("handoff", {}),
    )
    model = staticmaps_qa.open_wflow_model(model_root, catalog_path, model_cls=model_cls, mode="w+")
    model.build(steps=steps)
    wflow_repairs.normalize_wflow_staticmaps_nodata(model_root)
    if apply_repairs:
        wflow_repairs.repair_wflow_river_width(model_root)
        wflow_repairs.repair_wflow_canopy_parameters(model_root)
    qa = _wflow_staticmap_qa(model_root, config)
    return {
        "status": "built",
        "wflow_submodel_id": selected_id,
        "base_model_root": model_root,
        "data_catalog": catalog_path,
        "gauges_fn": gauge_summary["gauges_fn"],
        "gauge_count": gauge_summary["gauge_count"],
        "observation_gauges_fn": obs_gauge_summary["gauges_fn"] if obs_gauge_summary else None,
        "observation_gauge_count": obs_gauge_summary["gauge_count"] if obs_gauge_summary else 0,
        "built": _is_built_wflow_model(model_root),
        "staticmap_qa_status": staticmaps_qa.qa_status(qa),
        "model": model,
    }


def _legacy_wflow_repairs_enabled(config: dict) -> bool:
    return bool((config.get("wflow", {}) or {}).get("apply_legacy_repairs", False))


def _wflow_staticmap_qa(model_root: Path, config: dict) -> pd.DataFrame:
    try:
        report = staticmaps_qa.validate_staticmaps(
            model_root,
            river_upa_km2=config.get("inland_coupling", {}).get("discharge_forcing", {}).get("river_upa_km2"),
            raise_on_error=False,
        )
        if wflow_reservoirs.wflow_reservoirs_enabled(config):
            reservoir_report = wflow_reservoirs.validate_wflow_reservoir_staticmaps(
                model_root,
                required=True,
                raise_on_error=False,
            )
            report = pd.concat([report, reservoir_report], ignore_index=True)
        return report
    except FileNotFoundError as exc:
        return pd.DataFrame(
            [{"check": "staticmaps", "status": "not_available", "message": str(exc)}]
        )


def _select_wflow_submodel(submodels: tuple[dict, ...], submodel_id: str | None) -> dict:
    if submodel_id is None:
        return submodels[0]
    for submodel in submodels:
        if str(submodel["wflow_submodel_id"]) == str(submodel_id):
            return submodel
    available = ", ".join(str(submodel["wflow_submodel_id"]) for submodel in submodels)
    raise ValueError(f"Wflow Submodel not found: {submodel_id}. Available submodels: {available}")


def _configured_wflow_submodels(config: dict, location_root: Path) -> list[dict]:
    wflow = config.get("wflow", {}) or {}
    submodels = list(((wflow.get("domain_set", {}) or {}).get("submodels", []) or []))
    if submodels:
        return submodels
    manifest = Path(wflow.get("domain_set_manifest", "data/wflow/domain_set.yaml"))
    if not manifest.is_absolute():
        manifest = Path(location_root) / manifest
    if not manifest.exists():
        return []
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    return list(payload.get("submodels", []) or [])


def _is_built_wflow_model(model_root: Path) -> bool:
    if not model_root.exists():
        return False
    files = {path.name for path in model_root.rglob("*") if path.is_file() and path.name != ".gitkeep"}
    return bool(files & {"wflow_sbm.toml", "staticmaps.nc", "staticgeoms.nc"})


def _sfincs_gauge_layer_matches(
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
        for source_path in _sfincs_handoff_source_paths(config, paths, submodel):
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


def _sfincs_handoff_source_paths(config: dict, paths: dict, submodel: dict) -> list[Path]:
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


def _observation_gauge_layer_matches(model_root: Path, submodel: dict, config: dict) -> bool:
    submodel_sites = {str(value).zfill(8) for value in submodel.get("gauge_site_nos", ()) if value}
    reference_sites = _configured_reference_gage_site_nos(config) & submodel_sites
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


def _equal_area_km2(geometry) -> float:
    if geometry is None or geometry.is_empty:
        return 0.0
    return float(gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs("EPSG:5070").area.iloc[0]) / 1.0e6


def _configured_wflow_region(config, location_root: Path) -> dict | None:
    build_config = resolve_location_path(
        location_root,
        config.get("wflow", {}).get("build_config", "wflow_build.yml"),
    )
    build_config = _ensure_model_recipe_file(config, "wflow_build", build_config)
    if not build_config.exists():
        return None
    workflow = _read_workflow(build_config)
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


def _hydromt_subbasin_region(outlet_xy: list[float], outlet: dict, subbasin_geometry: Path | None) -> dict:
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


def _subbasin_fabric_directory(output_path: Path) -> Path:
    return output_path.with_suffix("")


def _match_nhdplus_outlet(outlet, rivers, catchments):
    outlet_m = outlet.to_crs("EPSG:5070")
    rivers_m = rivers.to_crs("EPSG:5070")
    catchments_m = catchments.to_crs("EPSG:5070")
    point = outlet_m.geometry.iloc[0]

    river_distances = rivers_m.geometry.distance(point)
    river_index = int(river_distances.idxmin())
    containing = catchments_m[catchments_m.geometry.contains(point) | catchments_m.geometry.touches(point)]
    if containing.empty:
        catchment_distances = catchments_m.geometry.distance(point)
        catchment_index = int(catchment_distances.idxmin())
        catchment_distance = float(catchment_distances.loc[catchment_index])
    else:
        catchment_index = int(containing.index[0])
        catchment_distance = 0.0
    return {
        "river_index": river_index,
        "catchment_index": catchment_index,
        "river_distance_m": float(river_distances.loc[river_index]),
        "catchment_distance_m": catchment_distance,
    }


def _select_upstream_nhdplus_catchments(rivers, catchments, river_index, catchment_index):
    hydroseq_col = _find_column(rivers, ("hydroseq",))
    downstream_col = _find_column(rivers, ("dnhydroseq", "dn_hydroseq", "tohydroseq", "to_hydroseq"))
    river_id_col = _find_column(rivers, ("nhdplusid", "featureid", "comid"))
    catchment_id_col = _find_column(catchments, ("featureid", "nhdplusid", "comid"))
    if hydroseq_col and downstream_col and river_id_col and catchment_id_col:
        selected_hydroseq = _upstream_hydroseq_values(rivers, river_index, hydroseq_col, downstream_col)
        selected_flowline_ids = set(
            pd.to_numeric(
                rivers.loc[rivers[hydroseq_col].isin(selected_hydroseq), river_id_col],
                errors="coerce",
            ).dropna().astype("int64")
        )
        catchment_ids = pd.to_numeric(catchments[catchment_id_col], errors="coerce")
        selected = catchments[catchment_ids.isin(selected_flowline_ids)].copy()
        if not selected.empty:
            return selected, "routed_upstream_catchments"
    return catchments.loc[[catchment_index]].copy(), "nearest_or_containing_catchment"


def _upstream_hydroseq_values(rivers, outlet_index, hydroseq_col, downstream_col):
    hydroseq = pd.to_numeric(rivers[hydroseq_col], errors="coerce")
    downstream = pd.to_numeric(rivers[downstream_col], errors="coerce")
    outlet_value = hydroseq.loc[outlet_index]
    if pd.isna(outlet_value):
        return set()
    selected = {int(outlet_value)}
    changed = True
    while changed:
        changed = False
        upstream = set(hydroseq[downstream.isin(selected)].dropna().astype("int64"))
        new_values = upstream - selected
        if new_values:
            selected.update(new_values)
            changed = True
    return selected


def _find_column(frame, candidates):
    lookup = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _read_workflow(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _normalize_workflow(workflow)


def _ensure_model_recipe_file(config: dict, key: str, path: Path) -> Path:
    recipe = (config.get("_model_recipes") or {}).get(key)
    if recipe is None:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _GENERATED_NOTICE.format(source=f"the {key} model YAML extraction")
        + yaml.safe_dump(recipe, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _normalize_workflow(workflow: dict) -> dict:
    """Return HydroMT workflow steps from either supported YAML shape.

    HydroMT model objects consume ``steps=[{"setup_*": {...}}, ...]``. The
    coupling reference files use a more readable top-level recipe shape, so
    normalize that form at the code boundary instead of making stakeholder
    YAML carry the internal list representation.
    """
    if "steps" in workflow:
        return workflow

    steps = []
    passthrough = {}
    for name, options in workflow.items():
        if _is_hydromt_step_name(name):
            steps.append({name: {} if options is None else options})
        else:
            passthrough[name] = options
    return {**passthrough, "steps": steps}


def _is_hydromt_step_name(name) -> bool:
    text = str(name)
    return text.startswith("setup_") or text.startswith("write_") or "." in text


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


def _step_names(workflow: dict) -> tuple[str, ...]:
    names = []
    for step in workflow.get("steps", []):
        if isinstance(step, dict) and step:
            names.append(str(next(iter(step))))
    return tuple(names)


def _region_kind(workflow: dict) -> str:
    for step in workflow.get("steps", []):
        if not isinstance(step, dict):
            continue
        basemaps = step.get("setup_basemaps")
        if not isinstance(basemaps, dict):
            continue
        region = basemaps.get("region", {})
        if isinstance(region, dict) and region:
            return str(next(iter(region)))
    return "missing"


def _domain_status(region_kind: str, domain_set: dict) -> str:
    submodels = domain_set.get("submodels") or []
    if region_kind == "bbox" and domain_set.get("allow_multiple_submodels") is False:
        return "configured"
    if region_kind == "bbox" and not submodels:
        return "review_required_bbox_placeholder"
    if region_kind in {"basin", "subbasin", "interbasin"} or submodels:
        return "configured"
    return "review_required_missing_region"


def _merged_source_strategy(overrides: dict) -> dict:
    strategy = deepcopy(DEFAULT_US_WFLOW_SOURCE_STRATEGY)
    for section, values in (overrides or {}).items():
        if isinstance(values, dict) and isinstance(strategy.get(section), dict):
            strategy[section].update(values)
    return strategy


def _raster_xarray_entry(uri: str, *, category: str, **metadata) -> dict:
    entry_metadata = {"category": category}
    entry_metadata.update({key: value for key, value in metadata.items() if value is not None})
    return {
        "data_type": "RasterDataset",
        "driver": {"name": "raster_xarray"},
        "uri": uri,
        "metadata": entry_metadata,
    }


def _rasterio_entry(uri: str, *, crs: str, category: str, source: str | None = None) -> dict:
    metadata = {"crs": crs, "category": category}
    if source:
        metadata["source"] = source
    return {
        "data_type": "RasterDataset",
        "driver": {"name": "rasterio"},
        "uri": uri,
        "metadata": metadata,
    }


def _normalize_catalog_metadata(catalog: dict) -> dict:
    normalized = deepcopy(catalog)
    for entry in normalized.values():
        if not isinstance(entry, dict) or not isinstance(entry.get("metadata"), dict):
            continue
        entry["metadata"] = {
            key: _netcdf_safe_metadata_value(value)
            for key, value in entry["metadata"].items()
        }
    return normalized


def _netcdf_safe_metadata_value(value):
    if isinstance(value, bool):
        return int(value)
    return value
