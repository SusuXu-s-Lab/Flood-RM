from __future__ import annotations

from dataclasses import dataclass
import os
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import shapely
import xarray as xr
from shapely.geometry import LineString, box
from shapely.ops import split

from design_events.collect_sources.ssurgo import ssurgo_attribute_columns
from design_events.collect_sources.national_hydrography import WBD_MAPSERVER, fetch_nhdplus_hr_catchments, fetch_wbd_huc
from sfincs_runs.build_base.crossings import select_encompassing_huc
from sfincs_runs.build_base.inland_base import (
    plan_inland_sfincs_domain_set,
    write_inland_sfincs_domain_set_manifest,
)
from sfincs_runs.build_base.static_catalog import build_static_data_catalog
from sfincs_runs.build_base.static_intake import (
    build_region_setup,
    clip_dem_and_landcover_to_bbox,
    collect_ssurgo_infiltration_inputs,
    collect_static_region_inputs,
    collect_wflow_static_region_inputs,
    download_file,
    fetch_usgs_3dep_dem,
    fetch_worldcover_landcover,
    static_sources_with_defaults,
    worldcover_tile_urls,
)
from sfincs_runs.config import build_paths as build_sfincs_paths
from study_location import build_study_area, define_location
from wflow_runs.build_plan import write_wflow_crossing_gauge_locations, write_wflow_domain_set_manifest
from wflow_runs.notebook import exists_table, prepare_wflow_subbasin_fabric, domain_summary, subbasins


@dataclass(frozen=True)
class RegionSetupNotebookRuntime:
    location_root: Path
    location_name: str
    repo_root: Path
    runtime_config: dict
    config: dict
    grid_config: dict
    data_sources: dict
    sfincs_config: dict
    wflow_config: dict
    region_setup: object
    collect_static_inputs: bool
    fetch_dem: bool
    fetch_landcover: bool
    fetch_ssurgo: bool
    fetch_nhdplus: bool

    @property
    def static_sources(self) -> dict:
        return self.data_sources["static_sources"]

    def resolve_location_path(self, value) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.location_root / path


@dataclass(frozen=True)
class CoastalRegionSetupRuntime:
    location_root: Path
    location_name: str
    repo_root: Path
    config: dict
    paths: dict
    region_setup: object
    collect_static_inputs: bool
    fetch_dem: bool
    fetch_landcover: bool
    fetch_ssurgo: bool

    def resolve_location_path(self, value) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.location_root / path


@dataclass(frozen=True)
class EvaluationFootprint:
    aoi_result: object
    study_area_wgs: gpd.GeoDataFrame
    study_union: object
    evaluation_output_path: Path
    evaluation_footprint: gpd.GeoDataFrame
    summary: pd.Series


@dataclass(frozen=True)
class AoiBuildResult:
    output_path: Path
    source_format: str
    n_points: int
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class InlandRegionDomains:
    footprint: EvaluationFootprint
    sfincs_coverage: gpd.GeoDataFrame
    wflow_watersheds: gpd.GeoDataFrame
    wflow_collection_boundary: gpd.GeoDataFrame
    summary: pd.Series


@dataclass
class CoveragePreflight:
    bbox_gdf: gpd.GeoDataFrame
    bbox_geom: object
    bbox_source: str
    bbox_wgs84: tuple[float, float, float, float]
    coverage_target_gdf: gpd.GeoDataFrame
    coverage_target_geom: object
    huc_mode: bool
    huc_domain_path: Path
    huc_domain_sources: list[str]
    nhd_watersheds: gpd.GeoDataFrame
    nhd_watershed_gdf: gpd.GeoDataFrame
    nhd_watershed_geom: object
    nhd_watershed_source: str
    wflow_extent_gdf: gpd.GeoDataFrame
    wflow_extent_geom: object
    wflow_extent_path: Path
    wflow_extent_wgs84: tuple[float, float, float, float]
    wflow_domain_plan: object
    wflow_domain_summary: pd.Series
    wflow_subbasin_review: pd.DataFrame
    wflow_domain_manifest: Path | None
    wflow_crossing_gauge_summaries: list[dict]
    fabric_inputs: pd.DataFrame
    fabric_summary: pd.Series
    summary: pd.Series


@dataclass(frozen=True)
class SfincsDomainSetReview:
    domain_plan: object
    domain_manifest: Path | None
    figure: object
    summary: pd.Series


@dataclass
class StaticInputCollection:
    terrain_status: pd.Series
    terrain_landcover_summary: pd.DataFrame | pd.Series
    wflow_static_summary: pd.Series | pd.DataFrame


@dataclass(frozen=True)
class RequiredStaticDataCollection:
    wflow_summary: pd.Series | pd.DataFrame
    sfincs_summary: pd.Series | pd.DataFrame


@dataclass(frozen=True)
class CoastalRegionDomain:
    aoi_result: object
    study_area_wgs: gpd.GeoDataFrame
    bbox_gdf: gpd.GeoDataFrame
    bbox_wgs84: tuple[float, float, float, float]
    summary: pd.Series


@dataclass(frozen=True)
class CoastalStaticDataCollection:
    terrain_landcover: object
    dem_refresh: pd.Series | None
    coastal_region: pd.Series
    wave_geometry: pd.Series | None
    ssurgo: pd.Series
    summary: pd.Series


@dataclass
class CoastalTerrainLandcover:
    dem: object
    dem_clip: object
    landcover_clip: object
    landcover_raw_path: Path
    landcover_source: str
    summary: pd.Series


@dataclass
class CoastalStaticQa:
    hsg_summary: pd.DataFrame
    infiltration_summary: pd.Series


@dataclass(frozen=True)
class SsurgoInputCollection:
    summary: pd.Series | pd.DataFrame
    pedology_readiness: pd.Series
    pedology_columns_ready: bool


def static_input_settings_from_env(environ=None) -> dict[str, bool]:
    """Read notebook static-source gates from the process environment."""
    environ = os.environ if environ is None else environ
    return {
        "collect_static_inputs": environ.get("FLOOD_RM_COLLECT_STATIC_INPUTS", "1") != "0",
        "fetch_dem": environ.get("FLOOD_RM_FETCH_DEM", "0") == "1",
        "fetch_landcover": environ.get("FLOOD_RM_FETCH_LANDCOVER", "1") != "0",
        "fetch_ssurgo": environ.get("FLOOD_RM_FETCH_SSURGO", "1") != "0",
        "fetch_nhdplus": environ.get("FLOOD_RM_FETCH_NHDPLUS", "1") != "0",
    }


def load_runtime(
    location_root,
    *,
    static_input_settings: dict | None = None,
    wflow_domain_review_required: bool | None = False,
) -> RegionSetupNotebookRuntime | CoastalRegionSetupRuntime:
    """Load the region-setup notebook runtime for a Location Workspace."""
    location_root = Path(location_root).resolve()
    config = define_location(location_root / "config.yaml").config
    if config.get("flood_setting") == "coastal":
        return _load_coastal_runtime(
            location_root,
            static_input_settings=static_input_settings,
        )
    return _load_inland_runtime(
        location_root,
        static_input_settings=static_input_settings,
        wflow_domain_review_required=wflow_domain_review_required,
    )


def _load_inland_runtime(
    location_root,
    *,
    static_input_settings: dict | None = None,
    wflow_domain_review_required: bool | None = False,
) -> RegionSetupNotebookRuntime:
    """Load the inland region-setup notebook runtime."""
    location_root = Path(location_root).resolve()
    repo_root = location_root.parents[1]
    runtime_config = define_location(location_root / "config.yaml").config
    runtime_config["static_sources"] = static_sources_with_defaults(runtime_config)
    runtime_config.setdefault("paths", {}).setdefault("data_catalog", "data/static/data_catalogue.yaml")

    if wflow_domain_review_required is not None and runtime_config.get("wflow"):
        runtime_config["wflow"]["domain_set"]["review_required"] = bool(wflow_domain_review_required)

    settings = static_input_settings_from_env() | (static_input_settings or {})
    context = {
        "repo_root": repo_root,
        "location_name": location_root.name,
        "location_root": location_root,
    }

    return RegionSetupNotebookRuntime(
        location_root=location_root,
        location_name=location_root.name,
        repo_root=repo_root,
        runtime_config=runtime_config,
        config=runtime_config,
        grid_config=runtime_config,
        data_sources=runtime_config,
        sfincs_config=runtime_config,
        wflow_config={"wflow": runtime_config.get("wflow", {})},
        region_setup=build_region_setup(runtime_config, context),
        collect_static_inputs=bool(settings["collect_static_inputs"]),
        fetch_dem=bool(settings["fetch_dem"]),
        fetch_landcover=bool(settings["fetch_landcover"]),
        fetch_ssurgo=bool(settings["fetch_ssurgo"]),
        fetch_nhdplus=bool(settings["fetch_nhdplus"]),
    )


def runtime_summary(runtime: RegionSetupNotebookRuntime) -> pd.Series:
    return pd.Series(
        {
            "location": runtime.config["project"]["place_name"],
            "flood_setting": runtime.config["flood_setting"],
            "model_crs": runtime.config["project"]["model_crs"],
            "location_root": str(runtime.location_root),
        },
        name="runtime",
    )


def static_input_settings_summary(runtime: RegionSetupNotebookRuntime) -> pd.Series:
    return pd.Series(
        {
            "collect_static_inputs": runtime.collect_static_inputs,
            "fetch_dem": runtime.fetch_dem,
            "fetch_landcover": runtime.fetch_landcover,
            "fetch_ssurgo": runtime.fetch_ssurgo,
            "fetch_nhdplus": runtime.fetch_nhdplus,
        },
        name="static_input_settings",
    )


def region_source_record_locations(runtime: RegionSetupNotebookRuntime) -> pd.DataFrame:
    sources = runtime.static_sources
    records = {
        "SMART-DS evaluation footprint": runtime.grid_config["smart_ds_evaluation_footprint"]["output"],
        "SFINCS coverage boxes": sources["bbox"]["output"],
        "Wflow watersheds": sources["wflow_collection_extent"]["watersheds"],
        "Wflow collection envelope": sources["wflow_collection_extent"]["boundary"],
        "raw terrain": sources["terrain"]["raw"],
        "processed terrain": sources["terrain"]["output"],
        "raw landcover": sources["landcover"]["raw"],
        "processed landcover": sources["landcover"]["output"],
        "SSURGO polygons": sources["ssurgo"]["output"],
        "SSURGO attributes": sources["ssurgo"]["attributes_output"],
        "HSG raster": sources["ssurgo"]["hsg_output"],
        "Ksat raster": sources["ssurgo"]["ksat_output"],
    }
    return _location_record_table(runtime.location_root, records)


def _load_coastal_runtime(
    location_root,
    *,
    static_input_settings: dict | None = None,
) -> CoastalRegionSetupRuntime:
    """Load a coastal region-setup notebook runtime."""
    location_root = Path(location_root).resolve()
    repo_root = location_root.parents[1]
    config = define_location(location_root / "config.yaml").config
    config["static_sources"] = static_sources_with_defaults(config)
    config.setdefault("paths", {}).setdefault("data_catalog", "data/static/data_catalogue.yaml")
    paths = build_sfincs_paths(config)
    settings = static_input_settings_from_env() | (static_input_settings or {})
    region_setup = build_region_setup(
        config,
        {
            "repo_root": repo_root,
            "location_name": location_root.name,
            "location_root": location_root,
        },
        buffer_degrees=0.0,
    )
    return CoastalRegionSetupRuntime(
        location_root=location_root,
        location_name=location_root.name,
        repo_root=repo_root,
        config=config,
        paths=paths,
        region_setup=region_setup,
        collect_static_inputs=bool(settings["collect_static_inputs"]),
        fetch_dem=bool(settings["fetch_dem"]),
        fetch_landcover=bool(settings["fetch_landcover"]),
        fetch_ssurgo=bool(settings["fetch_ssurgo"]),
    )


def build_domains(runtime: RegionSetupNotebookRuntime | CoastalRegionSetupRuntime):
    """Build the configured region domains for inland or coastal notebooks."""
    if isinstance(runtime, CoastalRegionSetupRuntime):
        return _build_coastal_domains(runtime)
    return _build_inland_domains(runtime)


def _build_coastal_domains(runtime: CoastalRegionSetupRuntime) -> CoastalRegionDomain:
    """Build/read the AOI and read the configured SFINCS bbox from Region config."""
    aoi_result = _build_aoi(runtime.config, runtime.repo_root)
    study_area_wgs = gpd.read_file(aoi_result.output_path).to_crs("EPSG:4326")
    bbox_path = runtime.region_setup.bbox_output
    if not bbox_path.exists():
        raise FileNotFoundError(
            "Configured SFINCS bbox is missing. Set static_sources.bbox.output in the Location override "
            f"and create the GeoJSON before running region setup: {bbox_path}"
        )
    bbox_gdf = gpd.read_file(bbox_path).to_crs("EPSG:4326")
    bbox_geom = bbox_gdf.geometry.union_all()
    study_geom = study_area_wgs.geometry.union_all()
    if not bbox_geom.contains(study_geom) and not bbox_geom.covers(study_geom):
        raise RuntimeError(f"Configured SFINCS bbox does not contain the study area: {bbox_path}")
    bbox_wgs84 = tuple(float(value) for value in bbox_gdf.total_bounds)
    summary = pd.Series(
        {
            "study_area": str(aoi_result.output_path.relative_to(runtime.repo_root)),
            "bbox": str(bbox_path.relative_to(runtime.repo_root)),
            "bbox_wgs84": tuple(round(value, 6) for value in bbox_wgs84),
            "source_points": aoi_result.n_points,
            "bbox_source": "configured static_sources.bbox.output",
        },
        name="configured_coastal_region",
    )
    return CoastalRegionDomain(aoi_result, study_area_wgs, bbox_gdf, bbox_wgs84, summary)


def collect_static(
    runtime: RegionSetupNotebookRuntime | CoastalRegionSetupRuntime,
    domains=None,
):
    """Collect required static data for inland or coastal notebooks."""
    if isinstance(runtime, CoastalRegionSetupRuntime):
        if domains is None:
            raise TypeError("collect_static(runtime, domains) requires coastal domains")
        return _collect_coastal_static(runtime, domains)
    return _collect_inland_static(runtime)


def _collect_coastal_static(
    runtime: CoastalRegionSetupRuntime,
    domain: CoastalRegionDomain,
) -> CoastalStaticDataCollection:
    """Collect required coastal static data. Marshfield has no Wflow collection."""
    dem_refresh = None
    if runtime.collect_static_inputs and runtime.fetch_dem:
        dem_refresh = ensure_usgs_3dep_dem_covers_bbox(
            dem_raw=runtime.region_setup.dem_raw,
            bbox_wgs84=domain.bbox_wgs84,
            repo_root=runtime.repo_root,
            gsd=10,
        )
    terrain_landcover = collect_coastal_terrain_landcover_inputs(
        region_setup=runtime.region_setup,
        bbox_gdf=domain.bbox_gdf,
        bbox_wgs84=domain.bbox_wgs84,
        config=runtime.config,
        repo_root=runtime.repo_root,
    )
    coastal_region, coast = fetch_configured_coastal_region(runtime, domain)
    wave_geometry = derive_configured_wave_geometry(runtime) if runtime.config.get("coastal_waves", False) else None
    ssurgo = collect_configured_coastal_ssurgo(runtime, domain)
    summary = pd.Series(
        {
            "terrain_landcover": "ready",
            "dem_refresh": dem_refresh.get("status") if dem_refresh is not None else "not requested",
            "coastal_region": coastal_region["coastal_region"],
            "wave_geometry": "ready" if wave_geometry is not None else "not enabled",
            "ssurgo": ssurgo["hsg_raster"],
        },
        name="coastal_static_collection",
    )
    terrain_landcover.coast = coast
    return CoastalStaticDataCollection(terrain_landcover, dem_refresh, coastal_region, wave_geometry, ssurgo, summary)


def fetch_configured_coastal_region(
    runtime: CoastalRegionSetupRuntime,
    domain: CoastalRegionDomain,
) -> tuple[pd.Series, gpd.GeoDataFrame]:
    """Fetch OSM coastline geometry and write the configured coastal land region."""
    import osmnx as ox
    from shapely.geometry import Point
    from shapely.ops import polygonize, unary_union

    coast_config = runtime.config["static_sources"]["coastline"]
    land_seed = Point(coast_config["land_seed"])
    ocean_seed = Point(coast_config["ocean_seed"])
    aoi_box = box(*domain.bbox_wgs84)
    raw_coast = ox.features.features_from_bbox(bbox=domain.bbox_wgs84, tags={"natural": "coastline"})
    coast = raw_coast[raw_coast.geometry.type.isin(["LineString", "MultiLineString"])]
    coast = coast[coast["natural"] == "coastline"].to_crs("EPSG:4326").clip(aoi_box)

    coastline_output = runtime.location_root / "data/static/processed/osm_coastline.geojson"
    coastline_output.parent.mkdir(parents=True, exist_ok=True)
    coast.to_file(coastline_output, driver="GeoJSON")

    merged = unary_union(list(coast.geometry) + [aoi_box.boundary])
    polygons = gpd.GeoDataFrame(geometry=list(polygonize(merged)), crs="EPSG:4326")
    land = polygons[polygons.contains(land_seed)]
    ocean = polygons[polygons.contains(ocean_seed)]
    other = polygons[~polygons.index.isin(land.index.union(ocean.index))]

    runtime.region_setup.coastal_region_output.parent.mkdir(parents=True, exist_ok=True)
    land.to_crs(runtime.config["project"]["model_crs"]).to_file(runtime.region_setup.coastal_region_output, driver="GeoJSON")
    summary = pd.Series(
        {
            "raw_coastline_segments": len(coast),
            "land_polygons": len(land),
            "ocean_polygons": len(ocean),
            "other_polygons": len(other),
            "raw_coastline": str(coastline_output.relative_to(runtime.repo_root)),
            "coastal_region": str(runtime.region_setup.coastal_region_output.relative_to(runtime.repo_root)),
        },
        name="coastline",
    )
    return summary, coast


def derive_configured_wave_geometry(runtime: CoastalRegionSetupRuntime) -> pd.Series:
    """Derive offshore/water-level geometry for coastal wave coupling."""
    from shapely.geometry import Point
    from sfincs_runs.snapwave_setup import derive_seaward_boundary

    wave_config = runtime.config.get("coastal_wave_coupling") or {}
    boundary_buffer_m = float((wave_config.get("quadtree") or {}).get("waterlevel_boundary_buffer_m", 180.0))
    model_crs = str(runtime.config["project"].get("model_crs") or "EPSG:26919")
    domain_geom = gpd.read_file(runtime.region_setup.bbox_output).to_crs(model_crs).geometry.union_all()
    land_geom = gpd.read_file(runtime.region_setup.coastal_region_output).to_crs(model_crs).geometry.union_all()
    ocean_seed_lonlat = runtime.config.get("static_sources", {}).get("coastline", {}).get("ocean_seed")
    ocean_seed = (
        gpd.GeoSeries([Point(*ocean_seed_lonlat)], crs="EPSG:4326").to_crs(model_crs).iloc[0]
        if ocean_seed_lonlat
        else None
    )

    offshore_polygon, seaward_edge = derive_seaward_boundary(domain_geom, land_geom, ocean_seed=ocean_seed)
    static_root = Path(runtime.paths["static_root"])
    outputs = {
        "offshore_region": (static_root / "offshore_region.geojson", offshore_polygon),
        "seaward_edge": (static_root / "seaward_edge.geojson", seaward_edge),
        "waterlevel_boundary": (static_root / "waterlevel_boundary.geojson", seaward_edge.buffer(boundary_buffer_m)),
    }
    for path, geometry in outputs.values():
        gpd.GeoDataFrame(geometry=[geometry], crs=model_crs).explode(index_parts=False).reset_index(drop=True).to_file(path, driver="GeoJSON")
    return pd.Series(
        {
            "offshore_region": str(outputs["offshore_region"][0].relative_to(runtime.repo_root)),
            "seaward_edge": str(outputs["seaward_edge"][0].relative_to(runtime.repo_root)),
            "waterlevel_boundary": str(outputs["waterlevel_boundary"][0].relative_to(runtime.repo_root)),
            "boundary_buffer_m": boundary_buffer_m,
            "offshore_area_km2": round(offshore_polygon.area / 1e6, 2),
            "seaward_edge_km": round(seaward_edge.length / 1e3, 2),
        },
        name="wave_geometry",
    )


def collect_configured_coastal_ssurgo(
    runtime: CoastalRegionSetupRuntime,
    domain: CoastalRegionDomain,
) -> pd.Series:
    """Collect SSURGO polygons/attributes and write coastal infiltration rasters."""
    from design_events.collect_sources.ssurgo import (
        fetch_ssurgo_mapunit_attributes,
        fetch_ssurgo_mapunit_polygons,
        normalize_ssurgo_axis_order,
    )
    from sfincs_runs.hydrology import write_ssurgo_infiltration_rasters

    setup = runtime.region_setup
    soils = (
        gpd.read_file(setup.ssurgo_output)
        if setup.ssurgo_output.exists()
        else fetch_ssurgo_mapunit_polygons(domain.bbox_wgs84, setup.ssurgo_output, keep_gml=True)
    )
    soils = soils.set_crs("EPSG:4326") if soils.crs is None else soils.to_crs("EPSG:4326")
    original_bounds = tuple(round(value, 6) for value in soils.total_bounds) if not soils.empty else None
    soils = normalize_ssurgo_axis_order(soils, domain.bbox_wgs84)
    normalized_bounds = tuple(round(value, 6) for value in soils.total_bounds) if not soils.empty else None
    if original_bounds != normalized_bounds:
        setup.ssurgo_output.unlink(missing_ok=True)
        soils.to_file(setup.ssurgo_output, driver="GPKG")

    soil_mukeys = sorted(soils["mukey"].dropna().astype(str).unique()) if "mukey" in soils else []
    attributes = (
        pd.read_csv(setup.ssurgo_attributes_output)
        if setup.ssurgo_attributes_output.exists()
        else fetch_ssurgo_mapunit_attributes(soil_mukeys, setup.ssurgo_attributes_output)
    )
    legacy_landcover_template = runtime.paths["static_root"] / "worldcover_mfield_mesh.tif"
    landcover_template = setup.landcover_output if setup.landcover_output.exists() else legacy_landcover_template
    infiltration = write_ssurgo_infiltration_rasters(
        soils,
        attributes,
        landcover_template,
        hsg_out=setup.ssurgo_hsg_output,
        land_domain=setup.coastal_region_output,
        ksat_out=setup.ssurgo_ksat_output,
        drainage_condition="undrained",
        ksat_units="um/s",
    )
    return pd.Series(
        {
            "soil_polygons": len(soils),
            "soil_mukeys": len(soil_mukeys),
            "ssurgo_attribute_rows": len(attributes),
            "bounds_before_axis_check": original_bounds,
            "bounds_after_axis_check": normalized_bounds,
            "ssurgo": str(setup.ssurgo_output.relative_to(runtime.repo_root)),
            "ssurgo_attributes": str(setup.ssurgo_attributes_output.relative_to(runtime.repo_root)),
            "hsg_raster": str(setup.ssurgo_hsg_output.relative_to(runtime.repo_root)),
            "ksat_raster": str(setup.ssurgo_ksat_output.relative_to(runtime.repo_root)),
            "rasterized_soil_polygons": infiltration["rasterized_polygons"],
            "soil_pixels": infiltration["hsg_pixels"],
        },
        name="ssurgo",
    )


def _plot_coastal_domains(runtime: CoastalRegionSetupRuntime, domain: CoastalRegionDomain):
    import matplotlib.pyplot as plt

    coastal_region = (
        gpd.read_file(runtime.region_setup.coastal_region_output).to_crs("EPSG:4326")
        if runtime.region_setup.coastal_region_output.exists()
        else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    )
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    domain.bbox_gdf.boundary.plot(ax=ax, color="#dc2626", linewidth=1.5, label="configured SFINCS bbox")
    coastal_region.boundary.plot(ax=ax, color="#2563eb", linewidth=1.2, label="coastal land region")
    domain.study_area_wgs.boundary.plot(ax=ax, color="black", linewidth=1.3, label="study area")
    ax.set_title(f"{runtime.location_name.title()} configured coastal region")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="best")
    return fig


def plot_static(
    runtime: RegionSetupNotebookRuntime | CoastalRegionSetupRuntime,
    static_data=None,
    domains=None,
):
    """Plot collected static data QA for inland or coastal notebooks."""
    if isinstance(runtime, CoastalRegionSetupRuntime):
        if static_data is None or domains is None:
            raise TypeError("plot_static(runtime, static_data, domains) requires coastal static data and domains")
        return _plot_coastal_static(runtime, static_data, domains)
    return _plot_inland_static(runtime)


def _plot_coastal_static(
    runtime: CoastalRegionSetupRuntime,
    static_data: CoastalStaticDataCollection,
    domain: CoastalRegionDomain,
) -> CoastalStaticQa:
    return plot_coastal_static_input_qa(
        region_setup=runtime.region_setup,
        bbox_gdf=domain.bbox_gdf,
        bbox_wgs84=domain.bbox_wgs84,
        dem=static_data.terrain_landcover.dem,
        dem_clip=static_data.terrain_landcover.dem_clip,
        landcover_clip=static_data.terrain_landcover.landcover_clip,
        coast=getattr(static_data.terrain_landcover, "coast", None),
    )


def build_smart_ds_evaluation_footprint(runtime: RegionSetupNotebookRuntime) -> EvaluationFootprint:
    """Build the study AOI and copy it to the configured evaluation footprint."""
    aoi_config = runtime.runtime_config.get("aoi", {})
    aoi_output_path = runtime.resolve_location_path(
        aoi_config.get("output", "data/static/aoi/study_area.geojson")
    )
    if aoi_output_path.exists():
        study_area_wgs = gpd.read_file(aoi_output_path).to_crs("EPSG:4326")
        aoi_result = AoiBuildResult(
            output_path=aoi_output_path,
            source_format=str(aoi_config.get("source_format", "asset_registry")),
            n_points=0,
            bounds=tuple(float(value) for value in study_area_wgs.total_bounds),
        )
    else:
        aoi_result = _build_aoi(runtime.runtime_config, runtime.repo_root)
        study_area_wgs = gpd.read_file(aoi_result.output_path).to_crs("EPSG:4326")

    if aoi_config.get("preserve_disconnected_subregions") and "subregion_id" not in study_area_wgs:
        study_area_wgs = _split_disconnected_study_area(study_area_wgs, aoi_config)

    evaluation_output_path = runtime.resolve_location_path(
        runtime.grid_config["smart_ds_evaluation_footprint"]["output"]
    )
    evaluation_output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_footprint = study_area_wgs.copy()
    evaluation_footprint.to_file(evaluation_output_path, driver="GeoJSON")
    study_union = study_area_wgs.geometry.union_all()

    return EvaluationFootprint(
        aoi_result=aoi_result,
        study_area_wgs=study_area_wgs,
        study_union=study_union,
        evaluation_output_path=evaluation_output_path,
        evaluation_footprint=evaluation_footprint,
        summary=pd.Series(
            {
                "study_area": str(aoi_result.output_path.relative_to(runtime.repo_root)),
                "evaluation_footprint": str(evaluation_output_path.relative_to(runtime.repo_root)),
                "source_format": aoi_result.source_format,
                "subregion_count": _subregion_count(study_area_wgs, study_union),
                "bounds": tuple(round(value, 6) for value in aoi_result.bounds),
                "minimum_flood_coverage": runtime.grid_config["smart_ds_evaluation_footprint"][
                    "minimum_flood_coverage"
                ],
            },
            name="smart_ds_aoi",
        ),
    )


def _build_inland_domains(runtime: RegionSetupNotebookRuntime) -> InlandRegionDomains:
    """Write the selected SFINCS coverage and its encompassing Wflow HUC watershed."""
    footprint = build_smart_ds_evaluation_footprint(runtime)
    sfincs_coverage = write_selected_sfincs_coverage(runtime, footprint.study_area_wgs)
    wflow_watersheds = write_encompassing_wflow_huc_watersheds(runtime, sfincs_coverage)
    wflow_collection_boundary = write_wflow_collection_boundary(runtime, sfincs_coverage, wflow_watersheds)
    summary = pd.Series(
        {
            "selected_sfincs_domain_ids": ", ".join(sfincs_coverage["subregion_id"].astype(str)),
            "sfincs_coverage": str(runtime.region_setup.bbox_output.relative_to(runtime.repo_root)),
            "wflow_huc_watersheds": str(
                runtime.resolve_location_path(
                    runtime.static_sources["wflow_collection_extent"]["watersheds"]
                ).relative_to(runtime.repo_root)
            ),
            "wflow_collection_boundary": str(
                runtime.resolve_location_path(
                    runtime.static_sources["wflow_collection_extent"]["boundary"]
                ).relative_to(runtime.repo_root)
            ),
            "wbd_service": WBD_MAPSERVER,
            "huc_level": ", ".join(sorted(wflow_watersheds["huc_level"].astype(str).unique())),
            "huc_kind": ", ".join(sorted(wflow_watersheds["huc_kind"].astype(str).unique())),
        },
        name="selected_inland_region_domains",
    )
    return InlandRegionDomains(
        footprint=footprint,
        sfincs_coverage=sfincs_coverage,
        wflow_watersheds=wflow_watersheds,
        wflow_collection_boundary=wflow_collection_boundary,
        summary=summary,
    )


def write_selected_sfincs_coverage(
    runtime: RegionSetupNotebookRuntime,
    study_area_wgs: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Write SFINCS coverage boxes only for configured include_domain_ids."""
    selected_domain_ids = {
        str(value).strip()
        for value in runtime.sfincs_config["sfincs_domain_set"].get("include_domain_ids", [])
        if str(value).strip()
    }
    if not selected_domain_ids:
        raise RuntimeError("sfincs_domain_set.include_domain_ids must name the SMART-DS subregion(s) to model")

    study_geom = study_area_wgs.geometry.union_all()
    study_components = _study_components(study_area_wgs, study_geom)
    coverage, _ = _write_sfincs_coverage_boxes(
        study_area_wgs=study_area_wgs,
        study_components=study_components,
        requested_domain_ids=selected_domain_ids,
        location_name=runtime.location_name,
        output_path=runtime.region_setup.bbox_output,
    )
    return coverage


def write_encompassing_wflow_huc_watersheds(
    runtime: RegionSetupNotebookRuntime,
    sfincs_coverage: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Fetch/select WBD HUC polygons that contain the selected SFINCS coverage."""
    huc_config = runtime.wflow_config["wflow"]["domain_set"].get("huc", {})
    huc_root = runtime.resolve_location_path(huc_config.get("root", "data/wflow/domain_huc"))
    huc_root.mkdir(parents=True, exist_ok=True)
    levels = tuple(int(level) for level in huc_config.get("levels", (8, 6, 4)))
    allow_union = bool(huc_config.get("allow_union", True))
    service_url = str(huc_config.get("service_url", WBD_MAPSERVER))

    records = []
    for _, row in sfincs_coverage.iterrows():
        domain_id = str(row["subregion_id"])
        selected = select_encompassing_huc(
            row.geometry,
            _wbd_loader_for_geometry(runtime, row.geometry, service_url=service_url),
            levels=levels,
            allow_union=allow_union,
        )
        huc_path = huc_root / f"{domain_id}.geojson"
        huc_row = {
            "wflow_submodel_id": domain_id,
            "sfincs_domain_id": domain_id,
            "huc_id": "_".join(selected["huc_ids"]),
            "huc_level": int(selected["level"]),
            "huc_kind": selected["kind"],
            "geometry": selected["geometry"],
        }
        gpd.GeoDataFrame([huc_row], geometry="geometry", crs="EPSG:4326").to_file(
            huc_path,
            driver="GeoJSON",
        )
        records.append(huc_row)

    watersheds = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    combined_path = runtime.resolve_location_path(
        runtime.wflow_config["wflow"]["domain_set"].get("huc", {}).get(
            "output",
            "data/wflow/wflow_domain_huc.geojson",
        )
    )
    watersheds_path = runtime.resolve_location_path(
        runtime.static_sources["wflow_collection_extent"]["watersheds"]
    )
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    watersheds_path.parent.mkdir(parents=True, exist_ok=True)
    watersheds.to_file(combined_path, driver="GeoJSON")
    watersheds.to_file(watersheds_path, driver="GeoJSON")
    return watersheds


def write_wflow_collection_boundary(
    runtime: RegionSetupNotebookRuntime,
    sfincs_coverage: gpd.GeoDataFrame,
    wflow_watersheds: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Write the static-data collection envelope around the selected HUC watershed."""
    extent = runtime.static_sources["wflow_collection_extent"]
    padding = float(extent.get("padding_degrees", 0.02))
    geom = wflow_watersheds.geometry.union_all()
    west, south, east, north = gpd.GeoSeries(
        [geom, sfincs_coverage.geometry.union_all()],
        crs="EPSG:4326",
    ).total_bounds
    boundary = gpd.GeoDataFrame(
        [
            {
                "name": "wflow_collection_region",
                "method": "selected_sfincs_domain_encompassing_huc",
                "source": "USGS WBD HUC",
                "padding_degrees": padding,
                "geometry": box(
                    west - padding,
                    south - padding,
                    east + padding,
                    north + padding,
                ),
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    output_path = runtime.resolve_location_path(extent["boundary"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    boundary.to_file(output_path, driver="GeoJSON")
    return boundary


def _collect_inland_static(
    runtime: RegionSetupNotebookRuntime,
) -> RequiredStaticDataCollection:
    """Collect required static data for Wflow first, then SFINCS."""
    paths = {
        "repo_root": runtime.repo_root,
        "location_name": runtime.location_name,
        "location_root": runtime.location_root,
    }
    wflow_summary = collect_wflow_static_region_inputs(
        runtime.runtime_config,
        paths,
        fetch_dem=runtime.fetch_dem,
        fetch_landcover=runtime.fetch_landcover,
        fetch_ssurgo=runtime.fetch_ssurgo,
    )
    sfincs_summary = collect_static_region_inputs(
        runtime.runtime_config,
        paths,
        fetch_dem=runtime.fetch_dem,
        fetch_landcover=runtime.fetch_landcover,
        fetch_ssurgo=runtime.fetch_ssurgo,
    )
    return RequiredStaticDataCollection(
        wflow_summary=_stringify_paths(wflow_summary, name="wflow_static_data"),
        sfincs_summary=_stringify_paths(sfincs_summary, name="sfincs_static_data"),
    )


def _plot_inland_domains(runtime: RegionSetupNotebookRuntime, domains: InlandRegionDomains):
    """Plot the selected SMART-DS/SFINCS footprint inside the Wflow HUC watershed."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    domains.wflow_watersheds.boundary.plot(
        ax=ax,
        color="#2563eb",
        linewidth=1.6,
        label="Wflow HUC watershed",
    )
    domains.wflow_collection_boundary.boundary.plot(
        ax=ax,
        color="#7c3aed",
        linestyle="--",
        linewidth=1.2,
        label="static collection boundary",
    )
    domains.footprint.study_area_wgs.boundary.plot(
        ax=ax,
        color="black",
        linewidth=1.3,
        label="SMART-DS evaluation footprint",
    )
    domains.sfincs_coverage.boundary.plot(
        ax=ax,
        color="#dc2626",
        linewidth=1.6,
        label="selected SFINCS coverage",
    )
    ax.set_title(f"{runtime.location_name.title()} selected inland flood domains")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="best")
    return fig


def plot_domains(runtime: RegionSetupNotebookRuntime | CoastalRegionSetupRuntime, domains):
    """Plot the configured flood domains for inland or coastal setup notebooks."""
    if isinstance(runtime, CoastalRegionSetupRuntime):
        return _plot_coastal_domains(runtime, domains)
    return _plot_inland_domains(runtime, domains)


def _plot_inland_static(runtime: RegionSetupNotebookRuntime):
    """Plot Wflow and SFINCS DEM, landcover, HSG, and Ksat rasters when present."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm

    raster_paths = _required_static_raster_paths(runtime)
    missing = {name: path for name, path in raster_paths.items() if not path.exists()}
    if missing:
        return exists_table(Path("/"), missing)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    hsg_cmap = ListedColormap(["#4daf4a", "#377eb8", "#ffbf00", "#e41a1c"], name="hsg_abcd")
    hsg_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], hsg_cmap.N)

    for row_index, model in enumerate(["Wflow", "SFINCS"]):
        prefix = model.lower()
        dem = open_plot_raster(raster_paths[f"{prefix}_dem"], resampling=Resampling.average)
        landcover = open_plot_raster(raster_paths[f"{prefix}_landcover"])
        hsg = open_plot_raster(raster_paths[f"{prefix}_hsg"])
        ksat = open_plot_raster(raster_paths[f"{prefix}_ksat"], resampling=Resampling.average)

        dem.plot(ax=axes[row_index, 0], cmap="terrain", add_colorbar=True)
        landcover.plot(ax=axes[row_index, 1], add_colorbar=True)
        hsg.where(hsg > 0).plot(
            ax=axes[row_index, 2],
            cmap=hsg_cmap,
            norm=hsg_norm,
            add_colorbar=False,
        )
        _plot_ksat(ksat, axes[row_index, 3])

        for column_index, label in enumerate(["DEM", "landcover", "HSG", "Ksat"]):
            axes[row_index, column_index].set_title(f"{model} {label}")
            axes[row_index, column_index].set_xlabel("")
            axes[row_index, column_index].set_ylabel("")
    return fig


def coverage_preflight_choices(
    runtime: RegionSetupNotebookRuntime,
    *,
    write_crossing_gauges: bool = True,
) -> pd.Series:
    return pd.Series(
        {
            "coverage_geometry": "one SFINCS bbox per selected SMART-DS component",
            "allow_multiple_sfincs_domains": runtime.sfincs_config["sfincs_domain_set"].get(
                "allow_multiple_domains",
                True,
            ),
            "include_domain_ids": runtime.sfincs_config["sfincs_domain_set"].get("include_domain_ids", []),
            "wflow_outlet_source": runtime.wflow_config["wflow"]["domain_set"].get("outlet_source"),
            "fetch_nhdplus_review_evidence": runtime.fetch_nhdplus,
            "write_crossing_gauges": write_crossing_gauges,
        },
        name="coverage_and_hydrologic_preflight_choices",
    )


def build_region_coverage_preflight(
    runtime: RegionSetupNotebookRuntime,
    study_area_wgs: gpd.GeoDataFrame,
    *,
    write_crossing_gauges: bool = True,
) -> CoveragePreflight:
    return build_sfincs_coverage_and_wflow_preflight(
        runtime_config=runtime.runtime_config,
        location_root=runtime.location_root,
        repo_root=runtime.repo_root,
        region_setup=runtime.region_setup,
        study_area_wgs=study_area_wgs,
        fetch_nhdplus=runtime.fetch_nhdplus,
        write_crossing_gauges=write_crossing_gauges,
    )


def write_inland_domain_set_and_plot(
    runtime: RegionSetupNotebookRuntime,
    coverage_plan: CoveragePreflight,
    study_area_wgs: gpd.GeoDataFrame,
) -> SfincsDomainSetReview:
    """Write the SFINCS domain set and plot it in its Wflow context."""
    if not _contains_or_covers(coverage_plan.wflow_extent_geom, coverage_plan.bbox_geom):
        raise RuntimeError("Wflow collection review envelope does not contain the selected SFINCS coverage boxes")

    domain_plan = plan_inland_sfincs_domain_set(
        runtime.runtime_config,
        {"location_root": runtime.location_root},
    )
    domain_manifest = (
        write_inland_sfincs_domain_set_manifest(
            domain_plan,
            runtime.runtime_config,
            {"location_root": runtime.location_root},
        )
        if domain_plan.domains
        else None
    )
    figure = _plot_coverage_context(coverage_plan, study_area_wgs)
    summary = pd.Series(
        {
            "bbox_source": coverage_plan.bbox_source,
            "bbox_count": int(len(coverage_plan.bbox_gdf)),
            "bbox_wgs84": tuple(round(value, 6) for value in coverage_plan.bbox_wgs84),
            "bbox_geojson": str(runtime.region_setup.bbox_output.relative_to(runtime.repo_root)),
            "sfincs_domain_status": domain_plan.status,
            "sfincs_domain_count": domain_plan.domain_count,
            "sfincs_domain_manifest": str(domain_manifest.relative_to(runtime.repo_root))
            if domain_manifest
            else "review_required",
            "wflow_collection_bounds_wgs84": tuple(
                round(value, 6) for value in coverage_plan.wflow_extent_wgs84
            ),
            "sfincs_domain_equals_wflow_collection": False,
            "contains_sfincs_coverage": _contains_or_covers(
                coverage_plan.wflow_extent_geom,
                coverage_plan.bbox_geom,
            ),
        },
        name="sfincs_coverage_bboxes",
    )
    return SfincsDomainSetReview(domain_plan, domain_manifest, figure, summary)


def static_source_plan_table(runtime: RegionSetupNotebookRuntime) -> pd.DataFrame:
    sources = runtime.static_sources
    return pd.DataFrame(
        [
            {
                "component": "terrain",
                "raw": sources["terrain"]["raw"],
                "processed": sources["terrain"]["output"],
                "method": "USGS 3DEP raw DEM, clipped to bbox",
            },
            {
                "component": "landcover",
                "raw": sources["landcover"]["raw"],
                "processed": sources["landcover"]["output"],
                "method": "ESA WorldCover raw tile mosaic, clipped/reprojected to DEM",
            },
            {
                "component": "ssurgo",
                "raw": sources["ssurgo"]["output"],
                "processed": sources["ssurgo"]["hsg_output"],
                "method": "USDA SSURGO mapunit + attributes rasterized to landcover",
            },
            {
                "component": "ksat",
                "raw": sources["ssurgo"]["attributes_output"],
                "processed": sources["ssurgo"]["ksat_output"],
                "method": "SSURGO Ksat harmonic aggregation",
            },
        ]
    )


def static_collection_policy_summary() -> pd.Series:
    return pd.Series(
        {
            "sfincs_dem": "USGS 3DEP raw DEM clipped to selected SFINCS coverage bbox",
            "sfincs_landcover": "ESA WorldCover raw tile mosaic aligned to the DEM grid",
            "wflow_static_envelope": "reviewed Wflow watershed envelope with configured padding",
            "huc_wide_wflow_dem": "required unless FLOOD_RM_FETCH_DEM=1",
        },
        name="static_collection_policy",
    )


def collect_region_static_inputs(
    runtime: RegionSetupNotebookRuntime,
    coverage_plan: CoveragePreflight,
    *,
    allow_wflow_landcover_only_without_dem: bool = False,
) -> StaticInputCollection:
    return collect_inland_static_inputs(
        runtime_config=runtime.runtime_config,
        location_root=runtime.location_root,
        repo_root=runtime.repo_root,
        region_setup=runtime.region_setup,
        bbox_gdf=coverage_plan.bbox_gdf,
        bbox_wgs84=coverage_plan.bbox_wgs84,
        wflow_extent_path=coverage_plan.wflow_extent_path,
        wflow_extent_wgs84=coverage_plan.wflow_extent_wgs84,
        collect_static_inputs=runtime.collect_static_inputs,
        fetch_dem=runtime.fetch_dem,
        fetch_landcover=runtime.fetch_landcover,
        fetch_ssurgo=runtime.fetch_ssurgo,
        allow_wflow_landcover_only_without_dem=allow_wflow_landcover_only_without_dem,
    )


def collect_or_report_ssurgo_inputs(
    runtime: RegionSetupNotebookRuntime,
    coverage_plan: CoveragePreflight,
) -> SsurgoInputCollection:
    sources = runtime.static_sources
    if runtime.collect_static_inputs and runtime.fetch_ssurgo:
        summary = collect_ssurgo_infiltration_inputs(runtime.region_setup, coverage_plan.bbox_wgs84)
    else:
        summary = exists_table(
            runtime.location_root,
            {
                "SSURGO polygons": sources["ssurgo"]["output"],
                "SSURGO attributes": sources["ssurgo"]["attributes_output"],
                "hydrologic soil group raster": sources["ssurgo"]["hsg_output"],
                "saturated conductivity raster": sources["ssurgo"]["ksat_output"],
            },
        )

    attributes_path = runtime.resolve_location_path(sources["ssurgo"]["attributes_output"])
    columns_ready = (
        attributes_path.exists()
        and set(ssurgo_attribute_columns()).issubset(set(pd.read_csv(attributes_path, nrows=0).columns))
    )
    readiness = pd.Series(
        {
            "ssurgo_attributes": str(attributes_path),
            "wflow_pedology_columns_ready": columns_ready,
            "required_wflow_pedology_columns": ", ".join(ssurgo_attribute_columns()),
        },
        name="ssurgo_wflow_pedology_readiness",
    )
    return SsurgoInputCollection(summary, readiness, columns_ready)


def static_data_catalog_summary(runtime: RegionSetupNotebookRuntime) -> pd.Series:
    data_catalog_path = build_static_data_catalog(
        runtime.runtime_config,
        {
            "location_root": runtime.location_root,
            "data_catalog": runtime.resolve_location_path(runtime.sfincs_config["paths"]["data_catalog"]),
        },
    )
    return pd.Series(
        {
            "data_catalog": str(data_catalog_path.relative_to(runtime.repo_root)),
            "exists": data_catalog_path.exists(),
            "bytes": data_catalog_path.stat().st_size if data_catalog_path.exists() else 0,
        },
        name="hydromt_catalog",
    )


def plot_region_static_input_qa(runtime: RegionSetupNotebookRuntime):
    return plot_static_input_qa(runtime.region_setup, location_name=runtime.location_name)


def domain_planning_inputs_table(runtime: RegionSetupNotebookRuntime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "Wflow",
                "manifest": runtime.wflow_config["wflow"]["domain_set_manifest"],
                "allow_multiple": runtime.wflow_config["wflow"]["domain_set"]["allow_multiple_submodels"],
                "event_catalog_scope": runtime.wflow_config["wflow"]["domain_set"]["event_catalog_scope"],
            },
            {
                "model": "SFINCS",
                "manifest": runtime.sfincs_config["sfincs_domain_set"]["domain_manifest"],
                "allow_multiple": runtime.sfincs_config["sfincs_domain_set"]["allow_multiple_domains"],
                "event_catalog_scope": runtime.sfincs_config["sfincs_domain_set"]["event_catalog_scope"],
            },
        ]
    )


def region_setup_readiness_table(
    runtime: RegionSetupNotebookRuntime,
    *,
    ssurgo_wflow_columns_ready: bool,
) -> pd.DataFrame:
    sources = runtime.static_sources
    wflow_build_recipe = runtime.config.get("includes", {}).get(
        "wflow_build",
        runtime.wflow_config.get("wflow", {}).get("build_config", "wflow_build.yml"),
    )
    qa = exists_table(
        runtime.location_root,
        {
            "evaluation footprint": runtime.grid_config["smart_ds_evaluation_footprint"]["output"],
            "raw terrain": sources["terrain"]["raw"],
            "processed terrain": sources["terrain"]["output"],
            "raw landcover": sources["landcover"]["raw"],
            "processed landcover": sources["landcover"]["output"],
            "SSURGO polygons": sources["ssurgo"]["output"],
            "SSURGO attributes": sources["ssurgo"]["attributes_output"],
            "SSURGO Wflow pedology source": sources["ssurgo"]["attributes_output"],
            "soil group raster": sources["ssurgo"]["hsg_output"],
            "ksat raster": sources["ssurgo"]["ksat_output"],
            "wflow build recipe": wflow_build_recipe,
            "sfincs data catalog": runtime.sfincs_config["paths"]["data_catalog"],
        },
    )
    qa["status"] = qa["exists"].map({True: "ready", False: "missing or pending"})
    mask = qa["artifact"].eq("SSURGO Wflow pedology source")
    qa.loc[mask, "status"] = "ready" if ssurgo_wflow_columns_ready else "missing Wflow pedology columns"
    return qa


def build_sfincs_coverage_and_wflow_preflight(
    *,
    runtime_config: dict,
    location_root: Path,
    repo_root: Path,
    region_setup,
    study_area_wgs: gpd.GeoDataFrame,
    fetch_nhdplus: bool,
    write_crossing_gauges: bool,
) -> CoveragePreflight:
    """Write SFINCS coverage boxes, derive Wflow review domains, and return audit tables."""
    location_root = Path(location_root)
    repo_root = Path(repo_root)
    location_name = location_root.name
    data_sources = runtime_config
    sfincs_config = runtime_config
    wflow_domain_set = runtime_config["wflow"]["domain_set"]
    static_sources = data_sources["static_sources"]
    wflow_extent_config = static_sources.get("wflow_collection_extent", {})

    study_geom = study_area_wgs.geometry.union_all()
    study_components = _study_components(study_area_wgs, study_geom)
    allow_multiple_sfincs_domains = bool(sfincs_config["sfincs_domain_set"].get("allow_multiple_domains", True))
    if not allow_multiple_sfincs_domains:
        study_components = [{"source_subregion_id": None, "geometry": study_geom}]

    requested_sfincs_domain_ids = {
        str(value).strip()
        for value in sfincs_config["sfincs_domain_set"].get("include_domain_ids", [])
        if str(value).strip()
    }
    bbox_gdf, coverage_target_gdf = _write_sfincs_coverage_boxes(
        study_area_wgs=study_area_wgs,
        study_components=study_components,
        requested_domain_ids=requested_sfincs_domain_ids,
        location_name=location_name,
        output_path=region_setup.bbox_output,
    )
    bbox_source = (
        "smart_ds_subregion_component_envelopes"
        if allow_multiple_sfincs_domains and "subregion_id" in study_area_wgs.columns
        else "smart_ds_evaluation_component_envelopes"
        if allow_multiple_sfincs_domains
        else "smart_ds_outer_evaluation_envelope"
    )
    bbox_geom = bbox_gdf.geometry.union_all()
    bbox_wgs84 = tuple(float(value) for value in bbox_gdf.total_bounds)
    coverage_target_geom = coverage_target_gdf.geometry.union_all()

    huc_mode = str(wflow_domain_set.get("outlet_source")) == "encompassing_huc"
    huc_domain_path = _location_path(
        location_root,
        wflow_domain_set.get("huc", {}).get("output", "data/wflow/wflow_domain_huc.geojson"),
    )
    nhd_fabric_path = _location_path(
        location_root,
        wflow_extent_config.get(
            "source_fabric",
            wflow_domain_set.get("subbasin_fabric", "data/wflow/domain_set_subbasins.gpkg"),
        ),
    )
    nhd_catchments_path = _location_path(
        location_root,
        wflow_extent_config.get(
            "source_catchments",
            data_sources["collection"]["national_hydrography"].get(
                "catchments", "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg"
            ),
        ),
    )
    wflow_watersheds_path = _location_path(
        location_root, wflow_extent_config.get("watersheds", "data/static/aoi/wflow_nhdplus_watersheds.geojson")
    )
    wflow_extent_path = _location_path(
        location_root, wflow_extent_config.get("boundary", "data/static/aoi/wflow_collection_region.geojson")
    )

    _, wflow_domain_plan, wflow_domain_summary = domain_summary(runtime_config, location_root)
    wflow_subbasin_review = subbasins(wflow_domain_plan)
    wflow_domain_manifest, wflow_crossing_gauge_summaries, fabric_inputs, fabric_summary, wflow_domain_plan = (
        _write_wflow_domain_artifacts(
            runtime_config=runtime_config,
            location_root=location_root,
            huc_mode=huc_mode,
            huc_domain_path=huc_domain_path,
            nhd_fabric_path=nhd_fabric_path,
            write_crossing_gauges=write_crossing_gauges,
            wflow_domain_plan=wflow_domain_plan,
        )
    )

    nhd_watersheds, nhd_watershed_source = _load_or_fetch_wflow_watersheds(
        runtime_config=runtime_config,
        location_root=location_root,
        bbox_gdf=bbox_gdf,
        bbox_geom=bbox_geom,
        huc_mode=huc_mode,
        huc_domain_path=huc_domain_path,
        nhd_fabric_path=nhd_fabric_path,
        nhd_catchments_path=nhd_catchments_path,
        fetch_nhdplus=fetch_nhdplus,
        wflow_domain_plan=wflow_domain_plan,
    )
    nhd_watershed_geom = nhd_watersheds.geometry.union_all()
    nhd_watershed_gdf = _wflow_watershed_review_frame(
        nhd_watersheds=nhd_watersheds,
        nhd_watershed_geom=nhd_watershed_geom,
        nhd_watershed_source=nhd_watershed_source,
        huc_mode=huc_mode,
        location_name=location_name,
        method=wflow_extent_config.get("method", "reviewed_nhdplus_watersheds"),
    )

    wflow_extent_gdf = _write_wflow_collection_envelope(
        wflow_extent_config=wflow_extent_config,
        huc_mode=huc_mode,
        nhd_watershed_source=nhd_watershed_source,
        nhd_watershed_geom=nhd_watershed_geom,
        bbox_geom=bbox_geom,
        wflow_watersheds_path=wflow_watersheds_path,
        wflow_extent_path=wflow_extent_path,
        nhd_watershed_gdf=nhd_watershed_gdf,
    )
    wflow_extent_geom = wflow_extent_gdf.geometry.union_all()
    wflow_extent_wgs84 = tuple(float(value) for value in wflow_extent_gdf.total_bounds)
    huc_domain_sources = _huc_domain_sources(huc_mode, wflow_domain_plan, location_root, repo_root, huc_domain_path)

    summary = pd.Series(
        {
            "method": "encompassing_huc" if huc_mode else wflow_extent_config.get("method", "reviewed_nhdplus_watersheds"),
            "source": nhd_watershed_source,
            "wflow_domain_status": wflow_domain_plan.status,
            "wflow_domain_manifest": str(wflow_domain_manifest.relative_to(repo_root)) if wflow_domain_manifest else "review_required",
            "wflow_domain_huc": ", ".join(huc_domain_sources)
            if huc_domain_sources
            else str(huc_domain_path.relative_to(repo_root))
            if huc_mode and huc_domain_path.exists()
            else "n/a",
            "wflow_sfincs_gauge_files": ", ".join(
                str(summary["gauges_fn"].relative_to(repo_root)) for summary in wflow_crossing_gauge_summaries
            ),
            "watershed_count": int(len(nhd_watersheds)),
            "watersheds_geojson": str(wflow_watersheds_path.relative_to(repo_root)),
            "collection_boundary_geojson": str(wflow_extent_path.relative_to(repo_root)),
            "collection_bounds_wgs84": tuple(round(value, 6) for value in wflow_extent_wgs84),
            "contains_sfincs_coverage": bool(nhd_watershed_geom.contains(bbox_geom) or nhd_watershed_geom.covers(bbox_geom)),
            "review_required": wflow_extent_config.get("review_required", True) or wflow_domain_plan.status != "ready",
        },
        name="wflow_hydrologic_preflight",
    )

    return CoveragePreflight(
        bbox_gdf=bbox_gdf,
        bbox_geom=bbox_geom,
        bbox_source=bbox_source,
        bbox_wgs84=bbox_wgs84,
        coverage_target_gdf=coverage_target_gdf,
        coverage_target_geom=coverage_target_geom,
        huc_mode=huc_mode,
        huc_domain_path=huc_domain_path,
        huc_domain_sources=huc_domain_sources,
        nhd_watersheds=nhd_watersheds,
        nhd_watershed_gdf=nhd_watershed_gdf,
        nhd_watershed_geom=nhd_watershed_geom,
        nhd_watershed_source=nhd_watershed_source,
        wflow_extent_gdf=wflow_extent_gdf,
        wflow_extent_geom=wflow_extent_geom,
        wflow_extent_path=wflow_extent_path,
        wflow_extent_wgs84=wflow_extent_wgs84,
        wflow_domain_plan=wflow_domain_plan,
        wflow_domain_summary=wflow_domain_summary,
        wflow_subbasin_review=wflow_subbasin_review,
        wflow_domain_manifest=wflow_domain_manifest,
        wflow_crossing_gauge_summaries=wflow_crossing_gauge_summaries,
        fabric_inputs=fabric_inputs,
        fabric_summary=fabric_summary,
        summary=summary,
    )


def collect_inland_static_inputs(
    *,
    runtime_config: dict,
    location_root: Path,
    repo_root: Path,
    region_setup,
    bbox_gdf: gpd.GeoDataFrame,
    bbox_wgs84,
    wflow_extent_path: Path,
    wflow_extent_wgs84,
    collect_static_inputs: bool,
    fetch_dem: bool,
    fetch_landcover: bool,
    fetch_ssurgo: bool,
    allow_wflow_landcover_only_without_dem: bool = False,
) -> StaticInputCollection:
    """Collect SFINCS rasters and coarse Wflow static rasters for inland locations."""
    location_root = Path(location_root)
    repo_root = Path(repo_root)
    location_name = location_root.name
    static_sources = runtime_config["static_sources"]

    dem_fetch_summary = _maybe_fetch_dem(
        collect_static_inputs=collect_static_inputs,
        fetch_dem=fetch_dem,
        bbox_wgs84=bbox_wgs84,
        dem_raw=region_setup.dem_raw,
        repo_root=repo_root,
    )
    landcover_raw_path, landcover_source = _maybe_fetch_landcover(
        collect_static_inputs=collect_static_inputs,
        fetch_landcover=fetch_landcover,
        bbox_wgs84=bbox_wgs84,
        landcover_raw=region_setup.landcover_raw,
        repo_root=repo_root,
    )
    terrain_landcover_summary = _clip_or_report_static_rasters(
        collect_static_inputs=collect_static_inputs,
        region_setup=region_setup,
        landcover_raw_path=landcover_raw_path,
        bbox_gdf=bbox_gdf,
        config=runtime_config,
    )
    terrain_status = pd.Series(
        {
            "dem_raw": str(region_setup.dem_raw.relative_to(repo_root)),
            "dem_source": dem_fetch_summary.get("source"),
            "dem_next_action": dem_fetch_summary.get("next_action", "none"),
            "dem_processed": str(region_setup.dem_output.relative_to(repo_root)),
            "landcover_raw": str(region_setup.landcover_raw.relative_to(repo_root)),
            "landcover_source": landcover_source,
            "landcover_processed": str(region_setup.landcover_output.relative_to(repo_root)),
            "clip_ready": region_setup.dem_raw.exists() and region_setup.landcover_raw.exists(),
            "dem_exists": region_setup.dem_output.exists(),
            "landcover_exists": region_setup.landcover_output.exists(),
        },
        name="terrain_landcover",
    )
    wflow_static_summary = _collect_or_report_wflow_static_inputs(
        runtime_config=runtime_config,
        location_root=location_root,
        repo_root=repo_root,
        location_name=location_name,
        static_sources=static_sources,
        wflow_extent_path=Path(wflow_extent_path),
        wflow_extent_wgs84=wflow_extent_wgs84,
        collect_static_inputs=collect_static_inputs,
        fetch_dem=fetch_dem,
        fetch_landcover=fetch_landcover,
        fetch_ssurgo=fetch_ssurgo,
        allow_landcover_only_without_dem=allow_wflow_landcover_only_without_dem,
    )
    return StaticInputCollection(
        terrain_status=terrain_status,
        terrain_landcover_summary=_series_or_frame(terrain_landcover_summary),
        wflow_static_summary=wflow_static_summary,
    )


def plot_static_input_qa(region_setup, *, location_name: str, max_plot_cells: int = 500_000):
    """Plot DEM, landcover, HSG, and Ksat if all processed rasters are ready."""
    required = {
        "DEM": region_setup.dem_output,
        "landcover": region_setup.landcover_output,
        "HSG": region_setup.ssurgo_hsg_output,
        "Ksat": region_setup.ssurgo_ksat_output,
    }
    missing = [name for name, path in required.items() if not Path(path).exists()]
    if missing:
        return exists_table(Path("/"), {name: path for name, path in required.items()})

    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
    from matplotlib.patches import Patch

    dem_clip = open_plot_raster(region_setup.dem_output, max_plot_cells=max_plot_cells, resampling=Resampling.average)
    landcover_clip = open_plot_raster(region_setup.landcover_output, max_plot_cells=max_plot_cells)
    hsg = open_plot_raster(region_setup.ssurgo_hsg_output, max_plot_cells=max_plot_cells)
    ksat = open_plot_raster(region_setup.ssurgo_ksat_output, max_plot_cells=max_plot_cells, resampling=Resampling.average)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    dem_clip.plot(ax=axes[0, 0], cmap="terrain", add_colorbar=True)
    axes[0, 0].set_title(f"{location_name.title()} DEM")
    landcover_clip.plot(ax=axes[0, 1], add_colorbar=True)
    axes[0, 1].set_title(f"{location_name.title()} land-surface cover")

    hsg_colors = ["#4daf4a", "#377eb8", "#ffbf00", "#e41a1c"]
    hsg_cmap = ListedColormap(hsg_colors, name="hsg_abcd")
    hsg_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], hsg_cmap.N)
    hsg.where(hsg > 0).plot(ax=axes[1, 0], cmap=hsg_cmap, norm=hsg_norm, add_colorbar=False)
    axes[1, 0].legend(
        handles=[Patch(facecolor=color, edgecolor="none", label=label) for color, label in zip(hsg_colors, ["A", "B", "C", "D"])],
        title="HSG",
        loc="lower left",
        fontsize=8,
    )
    axes[1, 0].set_title("SSURGO hydrologic soil group")

    ksat_positive = ksat.where(ksat > 0)
    ksat_values = ksat_positive.to_numpy()
    ksat_values = ksat_values[np.isfinite(ksat_values)]
    ksat_vmin = max(float(np.nanpercentile(ksat_values, 2)), 0.01) if ksat_values.size else 0.01
    ksat_vmax = float(np.nanpercentile(ksat_values, 98)) if ksat_values.size else 1.0
    ksat_positive.plot(
        ax=axes[1, 1],
        cmap="viridis",
        norm=LogNorm(vmin=ksat_vmin, vmax=ksat_vmax),
        cbar_kwargs={"label": "Ksat (mm/hr)"},
    )
    axes[1, 1].set_title("SSURGO saturated conductivity")
    for ax in axes.ravel():
        ax.set_xlabel("")
        ax.set_ylabel("")
    return fig


def collect_coastal_terrain_landcover_inputs(*, region_setup, bbox_gdf, bbox_wgs84, config, repo_root) -> CoastalTerrainLandcover:
    """Clip Marshfield terrain and landcover over the reviewed coastal SFINCS bbox."""
    repo_root = Path(repo_root)
    if not region_setup.dem_raw.exists():
        raise FileNotFoundError(region_setup.dem_raw)

    if region_setup.landcover_raw.exists():
        landcover_raw_path = region_setup.landcover_raw
        landcover_source = "local"
    else:
        landcover_raw_path, landcover_source = _fetch_worldcover_for_bbox_geometry(
            bbox_wgs84,
            region_setup.landcover_raw,
            bbox_gdf=bbox_gdf,
        )

    region_setup.dem_output.parent.mkdir(parents=True, exist_ok=True)
    region_setup.landcover_output.parent.mkdir(parents=True, exist_ok=True)

    dem = rxr.open_rasterio(region_setup.dem_raw, masked=True).squeeze(drop=True)
    if dem.rio.crs is None:
        dem = dem.rio.write_crs(config["project"]["model_crs"])

    landcover = rxr.open_rasterio(landcover_raw_path, masked=True).squeeze(drop=True)
    if landcover.rio.crs is None:
        landcover = landcover.rio.write_crs(config["project"].get("reference_crs", "EPSG:4326"))

    bbox_model = bbox_gdf.to_crs(dem.rio.crs)
    dem_clip = dem.rio.clip_box(*bbox_model.total_bounds)
    landcover_clip = landcover.rio.clip_box(*bbox_wgs84)
    landcover_clip = landcover_clip.rio.clip(bbox_gdf.geometry, bbox_gdf.crs)
    landcover_clip = landcover_clip.rio.reproject_match(dem_clip)
    dem_clip.rio.to_raster(region_setup.dem_output)
    landcover_clip.rio.to_raster(region_setup.landcover_output)

    return CoastalTerrainLandcover(
        dem=dem,
        dem_clip=dem_clip,
        landcover_clip=landcover_clip,
        landcover_raw_path=Path(landcover_raw_path),
        landcover_source=landcover_source,
        summary=pd.Series(
            {
                "dem": str(region_setup.dem_output.relative_to(repo_root)),
                "landcover_raw": str(Path(landcover_raw_path).relative_to(repo_root)),
                "landcover_source": landcover_source,
                "landcover": str(region_setup.landcover_output.relative_to(repo_root)),
            },
            name="clipped_rasters",
        ),
    )


def ensure_usgs_3dep_dem_covers_bbox(*, dem_raw, bbox_wgs84, repo_root, gsd=10) -> pd.Series:
    """Refresh the raw DEM only when the existing raster does not cover the reviewed bbox."""
    dem_raw = Path(dem_raw)
    repo_root = Path(repo_root)
    dem_raw.parent.mkdir(parents=True, exist_ok=True)

    coverage_ok = _raster_bounds_cover_wgs84(dem_raw, bbox_wgs84) if dem_raw.exists() else False
    if coverage_ok:
        status = "kept (existing tif already covers bbox)"
        tiles_merged = 0
    else:
        summary = fetch_usgs_3dep_dem(bbox_wgs84, dem_raw, gsd=gsd, force=dem_raw.exists())
        status = "replaced with USGS 3DEP (stale/undersized CUDEM overwritten)"
        tiles_merged = int(summary.get("tiles_merged", 0))

    return pd.Series(
        {
            "dem_path": str(dem_raw.relative_to(repo_root)),
            "status": status,
            "tiles_merged": tiles_merged,
            "bytes": dem_raw.stat().st_size,
        },
        name="inland_dem",
    )


def plot_coastal_static_input_qa(
    *,
    region_setup,
    bbox_gdf,
    bbox_wgs84,
    dem,
    dem_clip,
    landcover_clip,
    coast=None,
) -> CoastalStaticQa:
    """Plot Marshfield terrain, landcover/coast, and SSURGO infiltration rasters."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    coastal_region = (
        gpd.read_file(region_setup.coastal_region_output).to_crs("EPSG:4326")
        if region_setup.coastal_region_output.exists()
        else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    )
    coast_wgs = coast.to_crs("EPSG:4326") if coast is not None and not coast.empty else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    hsg = rxr.open_rasterio(region_setup.ssurgo_hsg_output, masked=True).squeeze(drop=True)
    ksat = rxr.open_rasterio(region_setup.ssurgo_ksat_output, masked=True).squeeze(drop=True)
    hsg_wgs = hsg.rio.reproject("EPSG:4326")
    ksat_wgs = ksat.rio.reproject("EPSG:4326")

    _plot_marshfield_dem(dem_clip, bbox_gdf, dem)
    _plot_marshfield_landcover(landcover_clip, bbox_gdf, coastal_region, coast_wgs)
    _plot_marshfield_soils(hsg_wgs, ksat_wgs, bbox_gdf, bbox_wgs84, coastal_region)

    hsg_values, hsg_counts = np.unique(hsg.values[np.isfinite(hsg.values) & (hsg.values > 0)].astype(int), return_counts=True)
    hsg_summary = pd.DataFrame(
        {
            "hsg_code": hsg_values,
            "hsg": pd.Series(hsg_values).map({1: "A", 2: "B", 3: "C", 4: "D"}).to_numpy(),
            "pixel_count": hsg_counts,
        }
    )
    infiltration_summary = pd.Series(
        {
            "hsg_pixels": int(np.isfinite(hsg.values).sum()),
            "ksat_pixels": int(np.isfinite(ksat.values).sum()),
            "ksat_min_mmhr": float(np.nanmin(ksat.values)),
            "ksat_p50_mmhr": float(np.nanpercentile(ksat.values, 50)),
            "ksat_p98_mmhr": float(np.nanpercentile(ksat.values, 98)),
        },
        name="ssurgo_infiltration_rasters",
    )
    return CoastalStaticQa(hsg_summary=hsg_summary, infiltration_summary=infiltration_summary)


def open_plot_raster(path, *, max_plot_cells=500_000, resampling=Resampling.nearest):
    with rasterio.open(path) as src:
        factor = max(1, int(math.ceil(math.sqrt((src.width * src.height) / max_plot_cells))))
        out_height = max(1, int(math.ceil(src.height / factor)))
        out_width = max(1, int(math.ceil(src.width / factor)))
        data = src.read(1, out_shape=(out_height, out_width), masked=True, resampling=resampling)
        transform = src.transform * src.transform.scale(src.width / out_width, src.height / out_height)
        x = np.array([transform * (col + 0.5, 0.5) for col in range(out_width)])[:, 0]
        y = np.array([transform * (0.5, row + 0.5) for row in range(out_height)])[:, 1]
        preview = np.ma.asarray(data).astype("float32")
        raster = xr.DataArray(np.ma.filled(preview, np.nan), dims=("y", "x"), coords={"y": y, "x": x})
        if raster.size > max_plot_cells:
            raster = raster.coarsen(y=2, x=2, boundary="trim").mean()
        raster = raster.rio.write_crs(src.crs)
        raster.rio.write_transform(transform, inplace=True)
        return raster


def _wbd_loader_for_geometry(runtime: RegionSetupNotebookRuntime, geometry, *, service_url: str):
    huc_config = runtime.wflow_config["wflow"]["domain_set"].get("huc", {})
    pad = float(huc_config.get("query_pad_degrees", 0.1))
    west, south, east, north = geometry.bounds
    bbox = (west - pad, south - pad, east + pad, north + pad)

    def load(level):
        return fetch_wbd_huc(bbox, huc_level=level, service_url=service_url)

    return load


def _stringify_paths(values: dict, *, name: str) -> pd.Series:
    return pd.Series(
        {key: str(value) if isinstance(value, Path) else value for key, value in values.items()},
        name=name,
    )


def _required_static_raster_paths(runtime: RegionSetupNotebookRuntime) -> dict[str, Path]:
    extent = runtime.static_sources["wflow_collection_extent"]
    return {
        "wflow_dem": runtime.resolve_location_path(extent["terrain_output"]),
        "wflow_landcover": runtime.resolve_location_path(extent["landcover_output"]),
        "wflow_hsg": runtime.resolve_location_path(extent["hsg_output"]),
        "wflow_ksat": runtime.resolve_location_path(extent["ksat_output"]),
        "sfincs_dem": runtime.region_setup.dem_output,
        "sfincs_landcover": runtime.region_setup.landcover_output,
        "sfincs_hsg": runtime.region_setup.ssurgo_hsg_output,
        "sfincs_ksat": runtime.region_setup.ssurgo_ksat_output,
    }


def _plot_ksat(ksat, ax):
    from matplotlib.colors import LogNorm

    ksat_positive = ksat.where(ksat > 0)
    values = ksat_positive.to_numpy()
    values = values[np.isfinite(values)]
    vmin = max(float(np.nanpercentile(values, 2)), 0.01) if values.size else 0.01
    vmax = float(np.nanpercentile(values, 98)) if values.size else 1.0
    ksat_positive.plot(
        ax=ax,
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cbar_kwargs={"label": "Ksat (mm/hr)"},
    )


def _location_record_table(location_root: Path, records: dict[str, object]) -> pd.DataFrame:
    rows = []
    for label, value in records.items():
        path = Path(value)
        resolved = path if path.is_absolute() else Path(location_root) / path
        rows.append(
            {
                "record": label,
                "configured": str(value),
                "location_root_syntax": f'location_root / "{value}"' if not path.is_absolute() else str(value),
                "exists": resolved.exists(),
            }
        )
    return pd.DataFrame(rows)


def _split_disconnected_study_area(study_area_wgs, aoi_config) -> gpd.GeoDataFrame:
    study_geom_split = _preserve_disconnected_subregions(
        study_area_wgs.geometry.union_all(),
        max_bridge_edge_degrees=float(aoi_config.get("subregion_bridge_max_edge_degrees", 0.03)),
    )
    return gpd.GeoDataFrame(
        {"name": ["study_area"]},
        geometry=[study_geom_split],
        crs="EPSG:4326",
    )


def _build_aoi(config, repo_root) -> AoiBuildResult:
    raw = build_study_area(config, repo_root)
    if hasattr(raw, "output_path"):
        return raw
    name = config["project"]["name"]
    aoi_config = config.get("aoi", {})
    output = Path(repo_root) / "locations" / name / aoi_config.get("output", "data/static/aoi/study_area.geojson")
    bounds = raw.get("bounds", gpd.read_file(output).total_bounds)
    return AoiBuildResult(
        output_path=output,
        source_format=str(raw.get("source_format", aoi_config.get("source_format", "asset_registry"))),
        n_points=int(raw.get("n_points", 0)),
        bounds=tuple(float(value) for value in bounds),
    )


def _preserve_disconnected_subregions(geometry, *, max_bridge_edge_degrees: float):
    if geometry.geom_type != "Polygon":
        return geometry
    coords = list(geometry.exterior.coords)
    long_edges = [
        (coords[index], coords[index + 1])
        for index in range(len(coords) - 1)
        if LineString([coords[index], coords[index + 1]]).length > max_bridge_edge_degrees
    ]
    if len(long_edges) != 2:
        return geometry
    p0 = LineString(long_edges[0]).interpolate(0.5, normalized=True)
    p1 = LineString(long_edges[1]).interpolate(0.5, normalized=True)
    parts = split(geometry, LineString([p0, p1]))
    geoms = [part for part in parts.geoms if part.geom_type == "Polygon"]
    return geometry if len(geoms) < 2 else shapely.MultiPolygon(geoms)


def _subregion_count(study_area_wgs, study_union) -> int:
    if "subregion_id" in study_area_wgs.columns:
        return int(len(study_area_wgs))
    return len(getattr(study_union, "geoms", [study_union]))


def _contains_or_covers(container, candidate) -> bool:
    return bool(container.contains(candidate) or container.covers(candidate))


def _plot_coverage_context(coverage_plan: CoveragePreflight, study_area_wgs: gpd.GeoDataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    coverage_plan.wflow_extent_gdf.boundary.plot(
        ax=ax,
        color="purple",
        linewidth=1.4,
        linestyle="--",
        label="Wflow collection envelope",
    )
    coverage_plan.nhd_watershed_gdf.boundary.plot(
        ax=ax,
        color="tab:blue",
        linewidth=1.4,
        label="larger Wflow watersheds",
    )
    study_area_wgs.boundary.plot(
        ax=ax,
        color="black",
        linewidth=1.5,
        label="SMART-DS evaluation footprint",
    )
    bbox_label = "SFINCS coverage box" if len(coverage_plan.bbox_gdf) == 1 else "SFINCS coverage boxes"
    coverage_plan.bbox_gdf.boundary.plot(
        ax=ax,
        color="red",
        linewidth=1.5,
        label=bbox_label,
    )
    ax.set_title(f"{bbox_label} inside the larger Wflow watersheds")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(loc="best")
    return fig


def _study_components(study_area_wgs, study_geom):
    if "subregion_id" in study_area_wgs.columns:
        return [
            {"source_subregion_id": str(row["subregion_id"]), "geometry": row.geometry}
            for _, row in study_area_wgs.iterrows()
            if row.geometry is not None and not row.geometry.is_empty
        ]
    component_geometries = list(study_geom.geoms) if study_geom.geom_type == "MultiPolygon" else [study_geom]
    component_geometries = sorted(component_geometries, key=lambda geom: (geom.centroid.x, geom.centroid.y))
    return [{"source_subregion_id": None, "geometry": geometry} for geometry in component_geometries]


def _write_sfincs_coverage_boxes(
    *, study_area_wgs, study_components, requested_domain_ids, location_name, output_path
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    bbox_records = [
        {
            "name": "sfincs_coverage_bbox",
            "subregion_id": _coverage_subregion_id(index, len(study_components), record["source_subregion_id"], location_name),
            "source_subregion_id": record["source_subregion_id"],
            "component_index": index,
            "target_geometry": record["geometry"],
            "geometry": record["geometry"].envelope,
        }
        for index, record in enumerate(study_components)
    ]
    if requested_domain_ids:
        bbox_records = [record for record in bbox_records if record["subregion_id"] in requested_domain_ids]
        missing = sorted(requested_domain_ids - {record["subregion_id"] for record in bbox_records})
        if missing:
            raise RuntimeError("Configured SFINCS domain ids were not found in the SMART-DS components: " + ", ".join(missing))
    if not bbox_records:
        raise RuntimeError("No SFINCS coverage boxes remain after include_domain_ids filtering")
    target_geom = gpd.GeoSeries([record.pop("target_geometry") for record in bbox_records], crs=study_area_wgs.crs).union_all()
    target_gdf = gpd.GeoDataFrame({"name": ["selected_smart_ds_footprint"]}, geometry=[target_geom], crs=study_area_wgs.crs)
    bbox_gdf = gpd.GeoDataFrame(bbox_records, geometry="geometry", crs=study_area_wgs.crs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bbox_gdf.to_file(output_path, driver="GeoJSON")
    return bbox_gdf.to_crs("EPSG:4326"), target_gdf.to_crs("EPSG:4326")


def _coverage_subregion_id(index, count, source_subregion_id, location_name):
    if source_subregion_id:
        suffix = "".join(char.lower() if char.isalnum() else "_" for char in str(source_subregion_id)).strip("_")
        return f"{location_name}_{suffix}"
    if count == 1:
        return f"{location_name}_main"
    if count == 2:
        return f"{location_name}_{'west' if index == 0 else 'east'}"
    return f"{location_name}_{index + 1:02d}"


def _write_wflow_domain_artifacts(
    *, runtime_config, location_root, huc_mode, huc_domain_path, nhd_fabric_path, write_crossing_gauges, wflow_domain_plan
):
    if wflow_domain_plan.status != "ready":
        fabric_inputs = exists_table(
            location_root,
            {
                "NHDPlus HR river geometry": runtime_config["collection"]["national_hydrography"]["river_geometry"],
                "NHDPlus HR catchments": runtime_config["collection"]["national_hydrography"]["catchments"],
            },
        )
        fabric_summary = pd.Series(
            {
                "status": wflow_domain_plan.status,
                "issue": "; ".join(wflow_domain_plan.issues),
                "subbasin_fabric": str(nhd_fabric_path),
            },
            name="wflow_subbasin_fabric_result",
        )
        return None, [], fabric_inputs, fabric_summary, wflow_domain_plan

    manifest = write_wflow_domain_set_manifest(wflow_domain_plan, runtime_config, {"location_root": location_root})
    crossing_summaries = (
        [
            write_wflow_crossing_gauge_locations(runtime_config, {"location_root": location_root}, submodel)
            for submodel in wflow_domain_plan.submodels
        ]
        if write_crossing_gauges
        and str(runtime_config["wflow"]["domain_set"].get("outlet_source"))
        in {
            "stream_boundary_crossings",
            "encompassing_huc",
            "boundary_handoff_watershed",
            "stream_boundary_watershed",
            "sfincs_boundary_watershed",
        }
        else []
    )
    if huc_mode:
        fabric_inputs = exists_table(location_root, {"WBD per-box HUC domains": str(huc_domain_path)})
        fabric_summary = pd.Series({"status": "encompassing_huc", "huc_domain": str(huc_domain_path)}, name="wflow_subbasin_fabric_result")
        return manifest, crossing_summaries, fabric_inputs, fabric_summary, wflow_domain_plan

    _, fabric_inputs, replanned_domain, fabric_summary = prepare_wflow_subbasin_fabric(runtime_config, location_root, wflow_domain_plan)
    return manifest, crossing_summaries, fabric_inputs, fabric_summary, replanned_domain


def _load_or_fetch_wflow_watersheds(
    *, runtime_config, location_root, bbox_gdf, bbox_geom, huc_mode, huc_domain_path, nhd_fabric_path, nhd_catchments_path, fetch_nhdplus, wflow_domain_plan
):
    if huc_mode and wflow_domain_plan.submodels:
        frames = []
        for submodel in wflow_domain_plan.submodels:
            huc_frame = gpd.read_file(_location_path(location_root, submodel["subbasin_geometry"])).to_crs("EPSG:4326")
            huc_frame["wflow_submodel_id"] = submodel["wflow_submodel_id"]
            huc_frame["sfincs_domain_id"] = next(iter(submodel["sfincs_domain_ids"]), submodel["wflow_submodel_id"])
            frames.append(huc_frame)
        watersheds = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326")
        source = "WBD per-box HUC domains (Wflow domain set)"
    elif huc_mode and huc_domain_path.exists():
        watersheds = gpd.read_file(huc_domain_path).to_crs("EPSG:4326")
        source = "WBD per-box HUC domains (Wflow domain set)"
    elif nhd_fabric_path.exists():
        watersheds = gpd.read_file(nhd_fabric_path).to_crs("EPSG:4326")
        source = "NHDPlus handoff-subbasin review fabric"
    elif nhd_catchments_path.exists():
        catchments = gpd.read_file(nhd_catchments_path).to_crs("EPSG:4326")
        watersheds = catchments[catchments.intersects(bbox_geom)].copy()
        source = "NHDPlus catchments intersecting selected SFINCS coverage"
    elif fetch_nhdplus:
        catchments = fetch_nhdplus_hr_catchments(tuple(float(value) for value in bbox_gdf.total_bounds)).to_crs("EPSG:4326")
        nhd_catchments_path.parent.mkdir(parents=True, exist_ok=True)
        catchments.to_file(nhd_catchments_path, driver="GPKG")
        watersheds = catchments[catchments.intersects(bbox_geom)].copy()
        source = "NHDPlus HR catchments fetched for review evidence"
    else:
        raise FileNotFoundError(
            "Missing local NHDPlus review evidence (set FLOOD_RM_FETCH_NHDPLUS=1 to fetch): "
            f"{nhd_fabric_path} or {nhd_catchments_path}"
        )
    watersheds = watersheds[watersheds.geometry.notna() & ~watersheds.geometry.is_empty].copy()
    if watersheds.empty:
        raise RuntimeError("Local NHDPlus review evidence produced no catchment polygons")
    return watersheds, source


def _wflow_watershed_review_frame(*, nhd_watersheds, nhd_watershed_geom, nhd_watershed_source, huc_mode, location_name, method):
    if huc_mode:
        frame = nhd_watersheds.copy()
        if "wflow_submodel_id" not in frame.columns:
            frame["wflow_submodel_id"] = [f"{location_name}_{index + 1:02d}" for index in range(len(frame))]
        if "sfincs_domain_id" not in frame.columns:
            frame["sfincs_domain_id"] = frame["wflow_submodel_id"]
        frame["name"] = frame["wflow_submodel_id"]
        frame["method"] = "encompassing_huc"
        frame["source"] = nhd_watershed_source
        frame["watershed_count"] = int(len(nhd_watersheds))
        return frame
    return gpd.GeoDataFrame(
        [{"name": "wflow_nhdplus_watersheds", "method": method, "source": nhd_watershed_source, "watershed_count": int(len(nhd_watersheds)), "geometry": nhd_watershed_geom}],
        geometry="geometry",
        crs="EPSG:4326",
    )


def _write_wflow_collection_envelope(
    *, wflow_extent_config, huc_mode, nhd_watershed_source, nhd_watershed_geom, bbox_geom, wflow_watersheds_path, wflow_extent_path, nhd_watershed_gdf
):
    collection_padding_degrees = float(wflow_extent_config.get("padding_degrees", 0.02))
    west, south, east, north = gpd.GeoSeries([nhd_watershed_geom, bbox_geom], crs="EPSG:4326").total_bounds
    extent_gdf = gpd.GeoDataFrame(
        [
            {
                "name": "wflow_collection_region",
                "method": "encompassing_huc_envelope" if huc_mode else "nhdplus_review_envelope",
                "source": nhd_watershed_source,
                "padding_degrees": collection_padding_degrees,
                "geometry": box(
                    west - collection_padding_degrees,
                    south - collection_padding_degrees,
                    east + collection_padding_degrees,
                    north + collection_padding_degrees,
                ),
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    wflow_watersheds_path.parent.mkdir(parents=True, exist_ok=True)
    wflow_extent_path.parent.mkdir(parents=True, exist_ok=True)
    nhd_watershed_gdf.to_file(wflow_watersheds_path, driver="GeoJSON")
    extent_gdf.to_file(wflow_extent_path, driver="GeoJSON")
    return extent_gdf


def _huc_domain_sources(huc_mode, wflow_domain_plan, location_root, repo_root, huc_domain_path):
    if not huc_mode:
        return []
    sources = [
        str(_location_path(location_root, submodel["subbasin_geometry"]).relative_to(repo_root))
        for submodel in wflow_domain_plan.submodels
        if _location_path(location_root, submodel["subbasin_geometry"]).exists()
    ]
    if not sources and huc_domain_path.exists():
        sources = [str(huc_domain_path.relative_to(repo_root))]
    return sources


def _maybe_fetch_dem(*, collect_static_inputs, fetch_dem, bbox_wgs84, dem_raw, repo_root):
    if collect_static_inputs and fetch_dem and not dem_raw.exists():
        print(f"Fetching USGS 3DEP DEM into {dem_raw.relative_to(repo_root)}")
        return fetch_usgs_3dep_dem(bbox_wgs84, dem_raw)
    if dem_raw.exists():
        return {"dem_raw": dem_raw, "source": "existing"}
    return {
        "dem_raw": dem_raw,
        "source": "not fetched",
        "next_action": "Set fetch_dem = True or launch with FLOOD_RM_FETCH_DEM=1, then rerun this cell.",
    }


def _maybe_fetch_landcover(*, collect_static_inputs, fetch_landcover, bbox_wgs84, landcover_raw, repo_root):
    if collect_static_inputs and fetch_landcover and not landcover_raw.exists():
        print(f"Fetching ESA WorldCover landcover into {landcover_raw.relative_to(repo_root)}")
        return fetch_worldcover_landcover(bbox_wgs84, landcover_raw)
    return landcover_raw, "existing" if landcover_raw.exists() else "not fetched"


def _clip_or_report_static_rasters(*, collect_static_inputs, region_setup, landcover_raw_path, bbox_gdf, config):
    if collect_static_inputs and region_setup.dem_raw.exists() and region_setup.landcover_raw.exists():
        return clip_dem_and_landcover_to_bbox(
            region_setup.dem_raw,
            landcover_raw_path,
            region_setup.dem_output,
            region_setup.landcover_output,
            bbox_gdf,
            model_crs=config["project"]["model_crs"],
            reference_crs=config["project"].get("reference_crs", "EPSG:4326"),
        )
    return pd.DataFrame(
        [
            {"artifact": "raw terrain", "path": region_setup.dem_raw, "exists": region_setup.dem_raw.exists()},
            {"artifact": "processed terrain", "path": region_setup.dem_output, "exists": region_setup.dem_output.exists()},
            {"artifact": "raw landcover", "path": region_setup.landcover_raw, "exists": region_setup.landcover_raw.exists()},
            {"artifact": "processed landcover", "path": region_setup.landcover_output, "exists": region_setup.landcover_output.exists()},
        ]
    )


def _collect_or_report_wflow_static_inputs(
    *,
    runtime_config,
    location_root,
    repo_root,
    location_name,
    static_sources,
    wflow_extent_path,
    wflow_extent_wgs84,
    collect_static_inputs,
    fetch_dem,
    fetch_landcover,
    fetch_ssurgo,
    allow_landcover_only_without_dem,
):
    extent_config = static_sources.get("wflow_collection_extent", {})
    wflow_dem_raw = _location_path(location_root, extent_config.get("terrain_raw", "data/wflow/static/raw/topo/dem_wflow.tif"))
    wflow_landcover_raw = _location_path(
        location_root, extent_config.get("landcover_raw", "data/wflow/static/raw/landcover/landcover_wflow.tif")
    )
    wflow_landcover_output = _location_path(
        location_root, extent_config.get("landcover_output", "data/wflow/static/processed/landcover_wflow_coarse.tif")
    )
    _require_wflow_static_inputs(
        collect_static_inputs=collect_static_inputs,
        allow_landcover_only_without_dem=allow_landcover_only_without_dem,
        wflow_extent_path=wflow_extent_path,
        wflow_dem_raw=wflow_dem_raw,
        wflow_landcover_raw=wflow_landcover_raw,
        wflow_landcover_output=wflow_landcover_output,
        fetch_dem=fetch_dem,
        fetch_landcover=fetch_landcover,
        repo_root=repo_root,
    )
    if collect_static_inputs and (fetch_dem or wflow_dem_raw.exists()):
        summary = collect_wflow_static_region_inputs(
            runtime_config,
            {"repo_root": repo_root, "location_name": location_name, "location_root": location_root},
            fetch_dem=fetch_dem,
            fetch_landcover=fetch_landcover,
            fetch_ssurgo=fetch_ssurgo,
        )
        return pd.Series({key: str(value) if isinstance(value, Path) else value for key, value in summary.items()}, name="wflow_static_collection")
    if collect_static_inputs and allow_landcover_only_without_dem:
        return _collect_wflow_landcover_without_dem(
            extent_config=extent_config,
            wflow_extent_path=wflow_extent_path,
            wflow_extent_wgs84=wflow_extent_wgs84,
            wflow_dem_raw=wflow_dem_raw,
            wflow_landcover_output=wflow_landcover_output,
            fetch_landcover=fetch_landcover,
        )
    return exists_table(
        location_root,
        {
            "Wflow collection boundary": str(wflow_extent_path),
            "Wflow raw DEM": str(wflow_dem_raw),
            "Wflow raw landcover": str(wflow_landcover_raw),
            "Wflow coarse DEM": extent_config.get("terrain_output", "data/wflow/static/processed/dem_wflow_coarse.tif"),
            "Wflow coarse landcover": extent_config.get("landcover_output", "data/wflow/static/processed/landcover_wflow_coarse.tif"),
        },
    )


def _require_wflow_static_inputs(
    *,
    collect_static_inputs,
    allow_landcover_only_without_dem,
    wflow_extent_path,
    wflow_dem_raw,
    wflow_landcover_raw,
    wflow_landcover_output,
    fetch_dem,
    fetch_landcover,
    repo_root,
):
    if not collect_static_inputs:
        return
    missing = []
    if not wflow_extent_path.exists():
        missing.append(f"Wflow collection boundary is missing: {wflow_extent_path.relative_to(repo_root)}")
    if not allow_landcover_only_without_dem and not wflow_dem_raw.exists() and not fetch_dem:
        missing.append(f"Wflow raw DEM is missing: {wflow_dem_raw.relative_to(repo_root)}; set FLOOD_RM_FETCH_DEM=1")
    if allow_landcover_only_without_dem:
        if not wflow_landcover_raw.exists() and not wflow_landcover_output.exists() and not fetch_landcover:
            missing.append(f"Wflow coarse landcover is missing: {wflow_landcover_output.relative_to(repo_root)}; set FLOOD_RM_FETCH_LANDCOVER=1")
    elif not wflow_landcover_raw.exists() and not fetch_landcover:
        missing.append(f"Wflow raw landcover is missing: {wflow_landcover_raw.relative_to(repo_root)}; set FLOOD_RM_FETCH_LANDCOVER=1")
    if missing:
        raise FileNotFoundError("Wflow static collection is enabled but cannot run:\n" + "\n".join(missing))


def _collect_wflow_landcover_without_dem(
    *, extent_config, wflow_extent_path, wflow_extent_wgs84, wflow_dem_raw, wflow_landcover_output, fetch_landcover
):
    if fetch_landcover and not wflow_landcover_output.exists():
        landcover_path, landcover_source = fetch_worldcover_landcover(
            wflow_extent_wgs84,
            wflow_landcover_output,
            target_resolution_degrees=extent_config.get("landcover_resolution_degrees", extent_config.get("terrain_resolution_degrees")),
        )
    else:
        landcover_path = wflow_landcover_output
        landcover_source = "existing" if wflow_landcover_output.exists() else "not fetched"
    return pd.Series(
        {
            "mode": "landcover_only_until_wflow_dem_fetch",
            "reason": "HydroMT-Wflow elevation comes from us_hydrography_basemap; HUC-wide DEM review raster remains opt-in.",
            "bbox": str(wflow_extent_path),
            "dem_raw": str(wflow_dem_raw),
            "dem_next_action": "Set FLOOD_RM_FETCH_DEM=1 only if a HUC-wide Wflow DEM review raster is needed.",
            "landcover": str(landcover_path),
            "landcover_source": landcover_source,
            "landcover_exists": wflow_landcover_output.exists(),
        },
        name="wflow_static_collection",
    )


def _series_or_frame(value):
    if isinstance(value, pd.DataFrame):
        return value.assign(path=lambda frame: frame["path"].astype(str)) if "path" in value.columns else value
    return pd.Series(value, name="clip_summary")


def _location_path(location_root, value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(location_root) / path


def _fetch_worldcover_for_bbox_geometry(bbox_wgs84, output_path, *, bbox_gdf, year=2021, version="v200"):
    output_path = Path(output_path)
    tile_cache = output_path.parent / "worldcover_tiles"
    tile_paths = []
    for url in worldcover_tile_urls(bbox_wgs84, year=year, version=version):
        tile_path = tile_cache / Path(url).name
        if not tile_path.exists():
            download_file(url, tile_path)
        tile_paths.append(tile_path)
    arrays = [rxr.open_rasterio(path, masked=True).squeeze(drop=True) for path in tile_paths]
    try:
        landcover_full = arrays[0] if len(arrays) == 1 else merge_arrays(arrays)
        landcover_bbox = landcover_full.rio.clip_box(*bbox_wgs84)
        landcover_bbox = landcover_bbox.rio.clip(bbox_gdf.geometry, bbox_gdf.crs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        landcover_bbox.rio.to_raster(output_path)
    finally:
        for array in arrays:
            close = getattr(array, "close", None)
            if close:
                close()
    return output_path, "downloaded"


def _raster_bounds_cover_wgs84(path, bbox_wgs84, *, eps=1e-9):
    try:
        with rxr.open_rasterio(path) as src:
            bounds = src.rio.bounds()
            crs = src.rio.crs
        existing_bounds = bounds if crs is None or crs.to_epsg() == 4326 else transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    except Exception:
        return False
    ew, es, ee, en = existing_bounds
    bw, bs, be, bn = bbox_wgs84
    return ew <= bw + eps and es <= bs + eps and ee >= be - eps and en >= bn - eps


def _plot_marshfield_dem(dem_clip, bbox_gdf, dem):
    import matplotlib.pyplot as plt

    fig, dem_ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    dem_clip.plot(ax=dem_ax, cmap="terrain", add_colorbar=True)
    bbox_gdf.to_crs(dem.rio.crs).boundary.plot(ax=dem_ax, color="red", linewidth=1.0)
    dem_ax.set_title("DEM")
    dem_ax.set_xlabel("")
    dem_ax.set_ylabel("")
    plt.show()


def _plot_marshfield_landcover(landcover_clip, bbox_gdf, coastal_region, coast_wgs):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, landcover_ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    landcover_clip.plot(ax=landcover_ax, add_colorbar=True)
    bbox_gdf.to_crs(landcover_clip.rio.crs).boundary.plot(ax=landcover_ax, color="red", linewidth=1.0)
    if not coastal_region.empty:
        coastal_region.to_crs(landcover_clip.rio.crs).boundary.plot(ax=landcover_ax, color="#00a884", linewidth=0.9)
    if not coast_wgs.empty:
        coast_wgs.to_crs(landcover_clip.rio.crs).plot(ax=landcover_ax, color="navy", linewidth=0.8)
    landcover_ax.set_title("land-surface cover")
    landcover_ax.legend(
        handles=[
            Line2D([0], [0], color="red", linewidth=1.0, label="bbox"),
            Line2D([0], [0], color="navy", linewidth=0.8, label="raw coastline"),
            Patch(facecolor="none", edgecolor="#00a884", label="coastal land mask"),
        ],
        loc="lower left",
        fontsize=8,
    )
    landcover_ax.set_xlabel("")
    landcover_ax.set_ylabel("")
    plt.show()


def _plot_marshfield_soils(hsg_wgs, ksat_wgs, bbox_gdf, bbox_wgs84, coastal_region):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
    hsg_colors = ["#4daf4a", "#377eb8", "#ffbf00", "#e41a1c"]
    hsg_cmap = ListedColormap(hsg_colors, name="hsg_abcd")
    hsg_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], hsg_cmap.N)
    hsg_wgs.where(hsg_wgs > 0).plot(ax=axes[0], cmap=hsg_cmap, norm=hsg_norm, add_colorbar=False)
    axes[0].legend(
        handles=[Patch(facecolor=color, edgecolor="none", label=label) for color, label in zip(hsg_colors, ["A", "B", "C", "D"])],
        title="HSG",
        loc="lower left",
        fontsize=8,
    )
    axes[0].set_title("SSURGO hydrologic soil group raster")

    ksat_positive = ksat_wgs.where(ksat_wgs > 0)
    ksat_values = ksat_positive.values[np.isfinite(ksat_positive.values)]
    ksat_vmin = max(float(np.nanpercentile(ksat_values, 2)), 0.01) if ksat_values.size else 0.01
    ksat_vmax = float(np.nanpercentile(ksat_values, 98)) if ksat_values.size else 1.0
    ksat_positive.plot(
        ax=axes[1],
        cmap="viridis",
        norm=LogNorm(vmin=ksat_vmin, vmax=ksat_vmax),
        cbar_kwargs={"label": "Ksat (mm/hr)"},
    )
    axes[1].set_title("SSURGO Ksat raster, land domain")

    for ax in axes:
        bbox_gdf.boundary.plot(ax=ax, color="red", linewidth=1.0)
        if not coastal_region.empty:
            coastal_region.boundary.plot(ax=ax, color="black", linewidth=0.7)
        ax.set(xlim=(bbox_wgs84[0], bbox_wgs84[2]), ylim=(bbox_wgs84[1], bbox_wgs84[3]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("")
        ax.set_ylabel("")
    plt.show()
