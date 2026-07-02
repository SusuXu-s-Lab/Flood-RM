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

from collect_sources.national_hydrography import WBD_MAPSERVER, fetch_wbd_huc
from coupling.domain_geometry import select_encompassing_huc
from sfincs_runs.static_intake import (
    build_region_setup,
    collect_static_region_inputs,
    collect_wflow_static_region_inputs,
    download_file,
    fetch_usgs_3dep_dem,
    static_sources_with_defaults,
    worldcover_tile_urls,
)
from sfincs_runs.config import build_paths as build_sfincs_paths
from study_location import build_study_area, define_location
from coupling.wflow_sfincs_build import exists_table


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
    from collect_sources.ssurgo import (
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
