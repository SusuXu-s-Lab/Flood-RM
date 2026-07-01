from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from paths import location_root_from_paths, resolve_location_path
import wflow_runs.hydromt_recipe as hydromt_recipe
import wflow_runs.types as wflow_types


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
    build_config = hydromt_recipe.ensure_model_recipe_file(
        config,
        "wflow_build",
        resolve_location_path(location_root, wflow.get("build_config", "wflow_build.yml")),
    )
    river_geometry_required = hydromt_recipe.wflow_build_uses_river_geometry(build_config)
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
        strategy.hydromt_basemap_source: raster_xarray_entry(
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
        "ssurgo_hydrologic_soil_group": rasterio_entry(
            resolve_location_path(location_root, hsg).resolve().as_posix(),
            crs=reference_crs,
            category="soils",
            source="SSURGO",
        ),
        "ssurgo_saturated_conductivity": rasterio_entry(
            resolve_location_path(location_root, ksat).resolve().as_posix(),
            crs=reference_crs,
            category="soils",
            source="SSURGO",
        ),
        soil_parameters: raster_xarray_entry(
            resolve_location_path(location_root, ssurgo_wflow_soil_parameters).resolve().as_posix(),
            category="soils",
            status="review_required",
            required_for_build=True,
            derived_from=["ssurgo_hydrologic_soil_group", "ssurgo_saturated_conductivity"],
        ),
        "event_precip": raster_xarray_entry(
            resolve_location_path(location_root, event_precip).resolve().as_posix(),
            category="event_forcing",
        ),
        "event_temp_pet": raster_xarray_entry(
            resolve_location_path(location_root, event_temp_pet).resolve().as_posix(),
            category="event_forcing",
        ),
    }
    catalog = normalize_catalog_metadata(catalog)
    catalog_path.write_text(
        hydromt_recipe.GENERATED_NOTICE.format(source="the Wflow data-catalog build")
        + yaml.safe_dump(catalog, sort_keys=False),
        encoding="utf-8",
    )
    return catalog_path


def plan_wflow_us_source_strategy(config) -> wflow_types.WflowSourceStrategy:
    """Return the USA-first Wflow source strategy for notebook review."""
    project = config.get("project", {})
    wflow = config.get("wflow", {})
    strategy = merged_source_strategy(wflow.get("source_strategy", {}))
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


def merged_source_strategy(overrides: dict) -> dict:
    strategy = deepcopy(DEFAULT_US_WFLOW_SOURCE_STRATEGY)
    for section, values in (overrides or {}).items():
        if isinstance(values, dict) and isinstance(strategy.get(section), dict):
            strategy[section].update(values)
    return strategy


def raster_xarray_entry(uri: str, *, category: str, **metadata) -> dict:
    entry_metadata = {"category": category}
    entry_metadata.update({key: value for key, value in metadata.items() if value is not None})
    return {
        "data_type": "RasterDataset",
        "driver": {"name": "raster_xarray"},
        "uri": uri,
        "metadata": entry_metadata,
    }


def rasterio_entry(uri: str, *, crs: str, category: str, source: str | None = None) -> dict:
    metadata = {"crs": crs, "category": category}
    if source:
        metadata["source"] = source
    return {
        "data_type": "RasterDataset",
        "driver": {"name": "rasterio"},
        "uri": uri,
        "metadata": metadata,
    }


def normalize_catalog_metadata(catalog: dict) -> dict:
    normalized = deepcopy(catalog)
    for entry in normalized.values():
        if not isinstance(entry, dict) or not isinstance(entry.get("metadata"), dict):
            continue
        entry["metadata"] = {
            key: netcdf_safe_metadata_value(value)
            for key, value in entry["metadata"].items()
        }
    return normalized


def netcdf_safe_metadata_value(value):
    if isinstance(value, bool):
        return int(value)
    return value
