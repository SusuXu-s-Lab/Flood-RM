from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rioxarray as rxr
from shapely.geometry import box
import xarray as xr

from paths import resolve_location_path
from collect_sources.reservoir_conditions import (
    enrich_wflow_reservoirs_with_public_conditions,
)
from collect_sources.ssurgo import (
    fetch_ssurgo_mapunit_attributes,
    ssurgo_attribute_columns,
)

NHDPLUS_HR_MAPSERVER = "https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer"
NHDPLUS_HR_NETWORK_FLOWLINE_LAYER = 3
NHDPLUS_HR_WATERBODY_LAYER = 9
NHDPLUS_HR_CATCHMENT_LAYER = 10

# USGS Watershed Boundary Dataset (WBD) ArcGIS service: HUC polygons by level.
WBD_MAPSERVER = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer"
WBD_HUC_LAYER_BY_LEVEL = {2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6}
STREAM_GEO_FIGSHARE_ARTICLE_ID = 24463240
NLDI_BASE_URL = "https://api.water.usgs.gov/nldi/linked-data"


def collect_national_hydrography(settings, *, skip_existing=True, smoke=False):
    """Prepare USA-first HydroMT-Wflow hydrography and soil source artifacts."""
    config = settings["config"]
    paths = settings["paths"]
    location_root = Path(paths["location_root"])
    collection = config.get("collection", {}).get("national_hydrography", {})
    static_sources = config.get("static_sources", {})
    wflow_extent = static_sources.get("wflow_collection_extent", {})

    hydrography_nc = _location_path(location_root, collection.get("hydromt_basemap", "data/wflow/hydrography/us_hydrography_basemap.nc"))
    river_gpkg = _location_path(location_root, collection.get("river_geometry", "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"))
    catchments_gpkg = _location_path(location_root, collection.get("catchments", "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg"))
    reservoirs_gpkg = _location_path(
        location_root,
        collection.get("reservoirs", {}).get("output", collection.get("reservoirs_output", "data/sources/national_hydrography/nhdplus_hr_wflow_reservoirs.gpkg")),
    )
    soil_nc = _location_path(location_root, collection.get("wflow_soil_parameters", "data/wflow/static/ssurgo_wflow_soil_parameters.nc"))
    manifest = Path(paths.get("source_artifacts_root", location_root / "data/sources/source_artifacts")) / "national_hydrography_wflow_sources.json"
    target_resolution_degrees = float(collection.get("basemap_source_resolution_degrees", 1 / 1080))
    collect_review_vectors = bool(collection.get("collect_review_vectors", False))
    reservoir_cfg = collection.get("reservoirs", {})
    collect_reservoirs = bool(reservoir_cfg.get("enabled", False))
    stream_geo_spec = config.get("collection", {}).get("stream_geo_nldi", {})
    stream_geo_table = _optional_location_path(
        location_root,
        collection.get("stream_geo_table", stream_geo_spec.get("stream_geo_table")),
    )
    nldi_lookup_cache = _optional_location_path(
        location_root,
        collection.get(
            "nldi_lookup_cache",
            stream_geo_spec.get("nldi_lookup_cache", "data/sources/national_hydrography/nldi_stream_geo_comid_cache.csv"),
        ),
    )
    nhdplus_v2_flowlines = _optional_location_path(
        location_root,
        collection.get(
            "nhdplus_v2_flowlines",
            stream_geo_spec.get("nhdplus_v2_flowlines", "data/sources/national_hydrography/nhdplus_v2_flowlines.gpkg"),
        ),
    )

    required_outputs = [hydrography_nc, soil_nc]
    if collect_review_vectors:
        required_outputs.extend([river_gpkg, catchments_gpkg])
    if collect_reservoirs:
        required_outputs.append(reservoirs_gpkg)
    base_outputs = [path for path in required_outputs if path != reservoirs_gpkg]
    if (
        skip_existing
        and collect_reservoirs
        and not reservoirs_gpkg.exists()
        and all(path.exists() for path in base_outputs)
        and _hydromt_basemap_ready(
            hydrography_nc,
            target_resolution_degrees,
            expected_outlets=str(collection.get("dem_outlets", "edge")),
            expected_mask_geometry=_expected_mask_geometry(location_root, collection, wflow_extent),
        )
    ):
        reservoir_summary = write_wflow_reservoir_waterbodies(
            static_sources.get("wflow_collection_extent", {}).get("watersheds", "data/static/aoi/wflow_nhdplus_watersheds.geojson"),
            reservoirs_gpkg,
            river_geometry=river_gpkg if river_gpkg.exists() else None,
            service_url=str(collection.get("nhdplus_hr_service_url", NHDPLUS_HR_MAPSERVER)),
            timeout_seconds=float(collection.get("request_timeout_seconds", 120)),
            min_area_km2=float(reservoir_cfg.get("min_area_km2", 1.0)),
            default_depth_m=float(reservoir_cfg.get("default_depth_m", 5.0)),
            default_discharge_m3s=float(reservoir_cfg.get("default_discharge_m3s", 0.01)),
            source_layer_id=int(reservoir_cfg.get("source_layer_id", NHDPLUS_HR_WATERBODY_LAYER)),
            condition_cfg=reservoir_cfg.get("conditions", {}),
            location_root=location_root,
        )
        return {
            **_result(
                "collected",
                hydrography_nc,
                river_gpkg,
                catchments_gpkg,
                soil_nc,
                manifest,
                collect_review_vectors=collect_review_vectors,
                reservoirs_gpkg=reservoirs_gpkg,
            ),
            **reservoir_summary,
            "reservoir_only": True,
        }
    if (
        skip_existing
        and all(path.exists() for path in required_outputs)
        and _hydromt_basemap_ready(
            hydrography_nc,
            target_resolution_degrees,
            expected_outlets=str(collection.get("dem_outlets", "edge")),
            expected_mask_geometry=_expected_mask_geometry(location_root, collection, wflow_extent),
        )
    ):
        result = _result(
            "reused",
            hydrography_nc,
            river_gpkg,
            catchments_gpkg,
            soil_nc,
            manifest,
            collect_review_vectors=collect_review_vectors,
            reservoirs_gpkg=reservoirs_gpkg if collect_reservoirs else None,
        )
        if collect_reservoirs:
            result.update(
                _refresh_existing_reservoir_conditions(
                    reservoirs_gpkg,
                    reservoir_cfg,
                    location_root=location_root,
                    skip_existing=skip_existing,
                )
            )
        return result

    dem_path = _location_path(
        location_root,
        collection.get(
            "dem_source",
            wflow_extent.get("terrain_output", static_sources.get("terrain", {}).get("output", "data/static/processed/dem_region_setup.tif")),
        ),
    )
    hsg_path = _location_path(
        location_root,
        collection.get(
            "soil_template",
            wflow_extent.get("hsg_output", static_sources.get("ssurgo", {}).get("hsg_output", "data/static/soils/hsg.tif")),
        ),
    )
    if not dem_path.exists():
        raise FileNotFoundError(f"USGS 3DEP DEM is required before Wflow hydrography preparation: {dem_path}")
    if not hsg_path.exists():
        raise FileNotFoundError(f"SSURGO raster template is required before Wflow soil preparation: {hsg_path}")

    hydrography_summary = write_dem_derived_hydromt_basemap(
        dem_path,
        hydrography_nc,
        target_resolution_degrees=target_resolution_degrees,
        min_stream_uparea_km2=float(collection.get("min_stream_uparea_km2", 5.0)),
        river_geometry=river_gpkg if river_gpkg.exists() else None,
        river_burn_method=str(collection.get("dem_river_burn_method", "uparea")),
        river_burn_depth=float(collection.get("dem_river_burn_depth_m", 5.0)),
        outlets=str(collection.get("dem_outlets", "edge")),
        mask_geometry=_optional_location_path(
            location_root,
            collection.get("dem_mask_geometry", wflow_extent.get("watersheds")),
        ),
    )
    if collect_review_vectors:
        river_summary = write_review_hydrography_vectors(
            hydrography_nc,
            river_gpkg,
            catchments_gpkg,
            service_url=str(collection.get("nhdplus_hr_service_url", NHDPLUS_HR_MAPSERVER)),
            timeout_seconds=float(collection.get("request_timeout_seconds", 120)),
            stream_geo_table=stream_geo_table,
            nldi_lookup_cache=nldi_lookup_cache,
            use_nldi_lookup=bool(collection.get("use_nldi_stream_geo_join", True)),
            nldi_timeout_seconds=float(collection.get("nldi_request_timeout_seconds", 30)),
            nldi_max_workers=int(collection.get("nldi_max_workers", 4)),
            nldi_progress_interval=int(collection.get("nldi_progress_interval", 500)),
            stream_geo_join_method=str(collection.get("stream_geo_join_method", "attribute_transfer")),
            nhdplus_v2_flowlines=nhdplus_v2_flowlines,
            stream_geo_join_max_distance_m=float(collection.get("stream_geo_join_max_distance_m", 2500)),
        )
        hydrography_summary = write_dem_derived_hydromt_basemap(
            dem_path,
            hydrography_nc,
            target_resolution_degrees=target_resolution_degrees,
            min_stream_uparea_km2=float(collection.get("min_stream_uparea_km2", 5.0)),
            river_geometry=river_gpkg,
            river_burn_method=str(collection.get("dem_river_burn_method", "uparea")),
            river_burn_depth=float(collection.get("dem_river_burn_depth_m", 5.0)),
            outlets=str(collection.get("dem_outlets", "edge")),
            mask_geometry=_optional_location_path(
                location_root,
                collection.get("dem_mask_geometry", wflow_extent.get("watersheds")),
            ),
        )
    else:
        river_summary = {
            "review_vector_status": "skipped",
            "review_vector_note": "NHDPlus/3DHP vectors are optional QA artifacts and are not Wflow build inputs.",
        }
    if collect_reservoirs:
        reservoir_summary = write_wflow_reservoir_waterbodies(
            static_sources.get("wflow_collection_extent", {}).get("watersheds", "data/static/aoi/wflow_nhdplus_watersheds.geojson"),
            reservoirs_gpkg,
            river_geometry=river_gpkg if river_gpkg.exists() else None,
            service_url=str(collection.get("nhdplus_hr_service_url", NHDPLUS_HR_MAPSERVER)),
            timeout_seconds=float(collection.get("request_timeout_seconds", 120)),
            min_area_km2=float(reservoir_cfg.get("min_area_km2", 1.0)),
            default_depth_m=float(reservoir_cfg.get("default_depth_m", 5.0)),
            default_discharge_m3s=float(reservoir_cfg.get("default_discharge_m3s", 0.01)),
            source_layer_id=int(reservoir_cfg.get("source_layer_id", NHDPLUS_HR_WATERBODY_LAYER)),
            condition_cfg=reservoir_cfg.get("conditions", {}),
            location_root=location_root,
        )
    else:
        reservoir_summary = {
            "reservoir_status": "skipped",
            "reservoir_note": "Reservoir/waterbody source collection is not enabled.",
        }
    ssurgo_polygons = _location_path(
        location_root,
        collection.get(
            "soil_polygons",
            wflow_extent.get("ssurgo_output", static_sources.get("ssurgo", {}).get("output", "data/static/soils/ssurgo_mapunitpoly.gpkg")),
        ),
    )
    ssurgo_attributes = _location_path(
        location_root,
        collection.get(
            "soil_attributes",
            wflow_extent.get(
                "ssurgo_attributes_output",
                static_sources.get("ssurgo", {}).get("attributes_output", "data/static/soils/ssurgo_mapunit_attributes.csv"),
            ),
        ),
    )
    _ensure_ssurgo_wflow_attributes(ssurgo_polygons, ssurgo_attributes)
    soil_summary = write_ssurgo_wflow_soil_dataset(
        ssurgo_polygons,
        ssurgo_attributes,
        hsg_path,
        soil_nc,
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "source": "national_hydrography",
                "kind": "wflow_build_sources",
                "status": "review_required",
                "metadata": {
                    "service": collection.get("service", "usgs_3dhp_or_nhdplus_hr"),
                    "hydrography_method": "usgs_3dep_dem_derived_hydromt_basemap",
                    "review_vectors_collected": collect_review_vectors,
                    "stream_geo_table": str(stream_geo_table) if stream_geo_table else "",
                    "stream_geo_article_id": STREAM_GEO_FIGSHARE_ARTICLE_ID,
                    "soil_method": "ssurgo_horizon_pedology",
                    "smoke": bool(smoke),
                    **hydrography_summary,
                    **river_summary,
                    **reservoir_summary,
                    **soil_summary,
                },
                "artifacts": {
                    "hydromt_basemap": str(hydrography_nc),
                    "river_geometry": str(river_gpkg),
                    "catchments": str(catchments_gpkg),
                    "reservoirs": str(reservoirs_gpkg),
                    "wflow_soil_parameters": str(soil_nc),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return _result(
        "collected",
        hydrography_nc,
        river_gpkg,
        catchments_gpkg,
        soil_nc,
        manifest,
        collect_review_vectors=collect_review_vectors,
        reservoirs_gpkg=reservoirs_gpkg if collect_reservoirs else None,
    )


def _refresh_existing_reservoir_conditions(
    reservoirs_gpkg: Path,
    reservoir_cfg: dict,
    *,
    location_root: Path,
    skip_existing: bool,
):
    condition_cfg = reservoir_cfg.get("conditions", {}) or {}
    if not condition_cfg.get("enabled", False) or not reservoirs_gpkg.exists():
        return {}
    enriched = enrich_wflow_reservoirs_with_public_conditions(
        reservoirs_gpkg,
        condition_cfg,
        location_root=location_root,
        output_path=reservoirs_gpkg,
        skip_existing=skip_existing,
    )
    return {
        "reservoir_condition_status": enriched.provenance.get("status", "collected"),
        "reservoir_condition_provider": enriched.provenance.get("provider", ""),
        "reservoir_condition_matched": enriched.provenance.get("matched_reservoirs", 0),
        "reservoir_condition_unmatched": enriched.provenance.get("unmatched_reservoirs", 0),
        "reservoir_condition_summary": enriched.provenance.get("summary_csv", ""),
    }


def refresh_wflow_hydrography_basemap(settings, *, skip_existing=False):
    """Refresh only the HydroMT-Wflow hydrography basemap from the local DEM."""
    config = settings["config"]
    paths = settings["paths"]
    location_root = Path(paths["location_root"])
    collection = config.get("collection", {}).get("national_hydrography", {})
    static_sources = config.get("static_sources", {})
    wflow_extent = static_sources.get("wflow_collection_extent", {})

    hydrography_nc = _location_path(location_root, collection.get("hydromt_basemap", "data/wflow/hydrography/us_hydrography_basemap.nc"))
    river_gpkg = _location_path(location_root, collection.get("river_geometry", "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"))
    manifest = Path(paths.get("source_artifacts_root", location_root / "data/sources/source_artifacts")) / "national_hydrography_wflow_sources.json"
    target_resolution_degrees = float(collection.get("basemap_source_resolution_degrees", 1 / 1080))

    if (
        skip_existing
        and hydrography_nc.exists()
        and _hydromt_basemap_ready(
            hydrography_nc,
            target_resolution_degrees,
            expected_outlets=str(collection.get("dem_outlets", "edge")),
            expected_mask_geometry=_expected_mask_geometry(location_root, collection, wflow_extent),
        )
    ):
        return {
            "status": "reused",
            "reused": True,
            "hydrography_only": True,
            "hydromt_basemap": hydrography_nc,
            "source_artifact_json": manifest,
            "artifact_count": 1,
        }

    dem_path = _location_path(
        location_root,
        collection.get(
            "dem_source",
            wflow_extent.get("terrain_output", static_sources.get("terrain", {}).get("output", "data/static/processed/dem_region_setup.tif")),
        ),
    )
    if not dem_path.exists():
        raise FileNotFoundError(f"USGS 3DEP DEM is required before Wflow hydrography preparation: {dem_path}")

    hydrography_summary = write_dem_derived_hydromt_basemap(
        dem_path,
        hydrography_nc,
        target_resolution_degrees=target_resolution_degrees,
        min_stream_uparea_km2=float(collection.get("min_stream_uparea_km2", 5.0)),
        river_geometry=river_gpkg if river_gpkg.exists() else None,
        river_burn_method=str(collection.get("dem_river_burn_method", "uparea")),
        river_burn_depth=float(collection.get("dem_river_burn_depth_m", 5.0)),
        outlets=str(collection.get("dem_outlets", "edge")),
        mask_geometry=_optional_location_path(
            location_root,
            collection.get("dem_mask_geometry", wflow_extent.get("watersheds")),
        ),
    )
    _update_hydrography_manifest(
        manifest,
        collection,
        hydrography_nc,
        hydrography_summary,
    )
    return {
        "status": "collected",
        "reused": False,
        "hydrography_only": True,
        "hydromt_basemap": hydrography_nc,
        "source_artifact_json": manifest,
        "artifact_count": 1,
        **hydrography_summary,
    }


def refresh_wflow_river_geometry_sources(settings, *, skip_existing=False):
    """Refresh NHDPlus/STREAM-geo river geometry without rebuilding DEM or soil sources."""
    config = settings["config"]
    paths = settings["paths"]
    location_root = Path(paths["location_root"])
    collection = config.get("collection", {}).get("national_hydrography", {})
    stream_geo_spec = config.get("collection", {}).get("stream_geo_nldi", {})

    hydrography_nc = _location_path(location_root, collection.get("hydromt_basemap", "data/wflow/hydrography/us_hydrography_basemap.nc"))
    river_gpkg = _location_path(location_root, collection.get("river_geometry", "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"))
    catchments_gpkg = _location_path(location_root, collection.get("catchments", "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg"))
    manifest = Path(paths.get("source_artifacts_root", location_root / "data/sources/source_artifacts")) / "national_hydrography_wflow_sources.json"
    stream_geo_table = _optional_location_path(
        location_root,
        collection.get("stream_geo_table", stream_geo_spec.get("stream_geo_table")),
    )
    nldi_lookup_cache = _optional_location_path(
        location_root,
        collection.get(
            "nldi_lookup_cache",
            stream_geo_spec.get("nldi_lookup_cache", "data/sources/national_hydrography/nldi_stream_geo_comid_cache.csv"),
        ),
    )
    nhdplus_v2_flowlines = _optional_location_path(
        location_root,
        collection.get(
            "nhdplus_v2_flowlines",
            stream_geo_spec.get("nhdplus_v2_flowlines", "data/sources/national_hydrography/nhdplus_v2_flowlines.gpkg"),
        ),
    )

    if skip_existing and river_gpkg.exists() and catchments_gpkg.exists() and nldi_lookup_cache and nldi_lookup_cache.exists():
        return {
            "status": "reused",
            "reused": True,
            "hydrography_only": False,
            "river_geometry_only": True,
            "river_geometry": river_gpkg,
            "catchments": catchments_gpkg,
            "nldi_lookup_cache": nldi_lookup_cache,
            "source_artifact_json": manifest,
            "artifact_count": 2,
        }
    method = str(collection.get("stream_geo_join_method", "attribute_transfer")).lower()
    if (
        method in {"attribute_transfer", "stream_geo_attribute_transfer", "hydrologic_attribute_transfer"}
        and river_gpkg.exists()
        and stream_geo_table
        and stream_geo_table.exists()
    ):
        river_summary = enrich_existing_wflow_river_geometry_with_stream_geo(
            river_gpkg,
            stream_geo_table,
            catchments_gpkg=catchments_gpkg,
        )
        _update_river_geometry_manifest(
            manifest,
            hydrography_nc,
            river_gpkg,
            catchments_gpkg,
            nldi_lookup_cache,
            river_summary,
        )
        return {
            "status": "collected",
            "reused": False,
            "hydrography_only": False,
            "river_geometry_only": True,
            "used_existing_river_geometry": True,
            "hydromt_basemap": hydrography_nc,
            "river_geometry": river_gpkg,
            "catchments": catchments_gpkg,
            "nldi_lookup_cache": nldi_lookup_cache,
            "source_artifact_json": manifest,
            "artifact_count": 1,
            **river_summary,
        }
    if not hydrography_nc.exists():
        raise FileNotFoundError(f"HydroMT-Wflow hydrography basemap is required before river-geometry refresh: {hydrography_nc}")

    river_summary = write_review_hydrography_vectors(
        hydrography_nc,
        river_gpkg,
        catchments_gpkg,
        service_url=str(collection.get("nhdplus_hr_service_url", NHDPLUS_HR_MAPSERVER)),
        timeout_seconds=float(collection.get("request_timeout_seconds", 120)),
        stream_geo_table=stream_geo_table,
        nldi_lookup_cache=nldi_lookup_cache,
        use_nldi_lookup=bool(collection.get("use_nldi_stream_geo_join", True)),
        nldi_timeout_seconds=float(collection.get("nldi_request_timeout_seconds", 30)),
        nldi_max_workers=int(collection.get("nldi_max_workers", 4)),
        nldi_progress_interval=int(collection.get("nldi_progress_interval", 500)),
        stream_geo_join_method=str(collection.get("stream_geo_join_method", "attribute_transfer")),
        nhdplus_v2_flowlines=nhdplus_v2_flowlines,
        stream_geo_join_max_distance_m=float(collection.get("stream_geo_join_max_distance_m", 2500)),
    )
    _update_river_geometry_manifest(
        manifest,
        hydrography_nc,
        river_gpkg,
        catchments_gpkg,
        nldi_lookup_cache,
        river_summary,
    )
    return {
        "status": "collected",
        "reused": False,
        "hydrography_only": False,
        "river_geometry_only": True,
        "hydromt_basemap": hydrography_nc,
        "river_geometry": river_gpkg,
        "catchments": catchments_gpkg,
        "nldi_lookup_cache": nldi_lookup_cache,
        "source_artifact_json": manifest,
        "artifact_count": 2,
        **river_summary,
    }


def _update_hydrography_manifest(manifest, collection, hydrography_nc, hydrography_summary):
    manifest = Path(manifest)
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        payload = {
            "source": "national_hydrography",
            "kind": "wflow_build_sources",
            "status": "review_required",
            "metadata": {},
            "artifacts": {},
        }
    metadata = payload.setdefault("metadata", {})
    metadata.update(
        {
            "hydrography_method": "usgs_3dep_dem_derived_hydromt_basemap",
            "hydrography_only_refresh": True,
            **hydrography_summary,
        }
    )
    artifacts = payload.setdefault("artifacts", {})
    artifacts.update(
        {
            "hydromt_basemap": str(hydrography_nc),
            "river_geometry": artifacts.get(
                "river_geometry",
                str(collection.get("river_geometry", "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg")),
            ),
            "catchments": artifacts.get(
                "catchments",
                str(collection.get("catchments", "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg")),
            ),
            "wflow_soil_parameters": artifacts.get(
                "wflow_soil_parameters",
                str(collection.get("wflow_soil_parameters", "data/wflow/static/ssurgo_wflow_soil_parameters.nc")),
            ),
        }
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def enrich_existing_wflow_river_geometry_with_stream_geo(river_gpkg, stream_geo_table, *, catchments_gpkg=None):
    """Enrich an existing NHDPlus HR river-geometry artifact without refetching NHDPlus."""
    river_gpkg = Path(river_gpkg)
    stream_geo_table = Path(stream_geo_table)
    rivers = gpd.read_file(river_gpkg)
    stream_geo = load_stream_geo_table(stream_geo_table)
    enriched = _prepare_nhdplus_river_geometry(
        rivers,
        stream_geo=stream_geo,
        stream_geo_join_method="attribute_transfer",
        use_nldi_lookup=False,
    )
    tmp_path = river_gpkg.with_name(f"{river_gpkg.stem}_tmp{river_gpkg.suffix}")
    with suppress(FileNotFoundError):
        tmp_path.unlink()
    enriched.to_file(tmp_path, driver="GPKG", layer=river_gpkg.stem)
    tmp_path.replace(river_gpkg)
    catchment_features = 0
    if catchments_gpkg and Path(catchments_gpkg).exists():
        with suppress(Exception):
            catchment_features = int(len(gpd.read_file(catchments_gpkg, rows=1)))
    return {
        "river_features": int(len(enriched)),
        "catchment_features": catchment_features,
        "river_source": "USGS NHDPlus HR NetworkNHDFlowline",
        "catchment_source": "USGS NHDPlus HR NHDPlusCatchment",
        "stream_geo_join_method": "attribute_transfer_existing_river_geometry",
        "rivwth_unique": int(pd.to_numeric(enriched["rivwth"], errors="coerce").nunique(dropna=True)) if "rivwth" in enriched else 0,
        "rivdph_unique": int(pd.to_numeric(enriched["rivdph"], errors="coerce").nunique(dropna=True)) if "rivdph" in enriched else 0,
    }


def _update_river_geometry_manifest(manifest, hydrography_nc, river_gpkg, catchments_gpkg, nldi_lookup_cache, river_summary):
    manifest = Path(manifest)
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        payload = {
            "source": "national_hydrography",
            "kind": "wflow_build_sources",
            "status": "review_required",
            "metadata": {},
            "artifacts": {},
        }
    metadata = payload.setdefault("metadata", {})
    metadata.update(
        {
            "river_geometry_only_refresh": True,
            "nldi_lookup_cache": str(nldi_lookup_cache) if nldi_lookup_cache else "",
            **river_summary,
        }
    )
    artifacts = payload.setdefault("artifacts", {})
    artifacts.update(
        {
            "hydromt_basemap": str(hydrography_nc),
            "river_geometry": str(river_gpkg),
            "catchments": str(catchments_gpkg),
        }
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_dem_derived_hydromt_basemap(
    dem_path,
    output_path,
    *,
    target_resolution_degrees=1 / 1080,
    min_stream_uparea_km2=5.0,
    river_geometry=None,
    river_burn_method="uparea",
    river_burn_depth=5.0,
    outlets="edge",
    mask_geometry=None,
):
    """Write HydroMT-Wflow RasterDataset variables from a local USGS 3DEP DEM.

    When reviewed river geometry is available, burn it into the DEM before deriving D8
    flow directions. This keeps flat reservoirs/floodplains from being resolved as
    artificial diagonal flow paths by the depression-filling algorithm.
    """
    import hydromt  # noqa: F401
    from hydromt.gis import flw

    dem_path = Path(dem_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dem = None
    try:
        dem = rxr.open_rasterio(dem_path, masked=True).squeeze(drop=True)
        # HydroMT-Wflow only treats EPSG:4326 as a geographic grid when deriving
        # slopes from the basemap. 3DEP is commonly NAD83 / EPSG:4269; reproject
        # explicitly so setup_basemaps computes land_slope in m/m rather than
        # treating degree spacing as metres.
        if dem.rio.crs is None:
            dem = dem.rio.write_crs("EPSG:4326")
        if dem.rio.crs.to_epsg() != 4326:
            dem = dem.rio.reproject("EPSG:4326", resolution=target_resolution_degrees)
        factor = max(1, int(round(float(target_resolution_degrees) / abs(float(dem.rio.resolution()[0])))))
        if factor > 1:
            dem = dem.coarsen(y=factor, x=factor, boundary="trim").mean()
            dem = dem.rio.write_crs(dem.rio.crs)
        mask = _load_mask_geometry(mask_geometry, target_crs=dem.rio.crs)
        if mask is not None:
            dem = dem.rio.clip(mask.geometry.values, mask.crs, drop=False, all_touched=True)
        dem = dem.astype("float32")
        nodata = dem.rio.nodata
        values = dem.values
        if nodata is not None and np.isfinite(nodata):
            values = np.where(values == nodata, np.nan, values)
        finite = np.isfinite(values)
        if not np.any(finite):
            raise ValueError(f"DEM has no finite elevation values: {dem_path}")
        nodata_value = np.float32(-9999)
        elevtn = np.where(finite, values, nodata_value).astype("float32")
        dem = dem.copy(data=elevtn)
        dem = dem.rio.write_nodata(nodata_value, encoded=False)
        burn_rivers = _load_river_burn_geometry(river_geometry, target_crs=dem.rio.crs)
        da_flwdir = flw.d8_from_dem(
            dem,
            max_depth=-1.0,
            outlets=str(outlets or "edge"),
            gdf_riv=burn_rivers,
            riv_burn_method=str(river_burn_method or "uparea"),
            riv_depth=float(river_burn_depth),
        )
        flwdir = flw.flwdir_from_da(da_flwdir, ftype="infer", check_ftype=True)
        flwdir_arr = da_flwdir.values.astype("uint8")
        uparea = flwdir.upstream_area(unit="km2").astype("float32")
        strord = flwdir.stream_order().astype("int16")
        basins = flwdir.basins(idxs=flwdir.idxs_pit).astype("int32")
        valid = elevtn != nodata_value
        uparea = np.where(valid, uparea, np.float32(-9999)).astype("float32")
        strord = np.where(valid, strord, np.int16(0)).astype("int16")
        basins = np.where(valid, basins, np.int32(0)).astype("int32")
        stream_mask = (valid & (uparea >= float(min_stream_uparea_km2))).astype("uint8")

        ds = xr.Dataset(
            data_vars={
                "flwdir": (dem.dims, flwdir_arr, {"long_name": "D8 flow direction"}),
                "elevtn": (dem.dims, elevtn.astype("float32"), {"_FillValue": np.float32(-9999), "unit": "m+REF"}),
                "uparea": (dem.dims, uparea, {"_FillValue": np.float32(-9999), "unit": "km2"}),
                "strord": (dem.dims, strord, {"_FillValue": np.int16(0)}),
                "basins": (dem.dims, basins, {"_FillValue": np.int32(0)}),
                "rivmsk_review": (dem.dims, stream_mask, {"_FillValue": np.uint8(0)}),
            },
            coords={dem.rio.y_dim: dem[dem.rio.y_dim], dem.rio.x_dim: dem[dem.rio.x_dim]},
            attrs={
                "source": "USGS 3DEP DEM",
                "status": "review_required",
                "review_note": "HydroMT d8_from_dem LDD basemap from local DEM; review stream order, upstream area, and outlets before production.",
                "river_burn_features": int(len(burn_rivers)) if burn_rivers is not None else 0,
                "river_burn_method": str(river_burn_method or ""),
                "outlets": str(outlets or "edge"),
                "mask_geometry": str(mask_geometry or ""),
                "masked_cells": int(np.count_nonzero(~valid)),
            },
        )
        ds.raster.set_crs(_hydromt_crs(dem.rio.crs))
        _write_netcdf_atomically(
            ds,
            output_path,
            encoding={
                "flwdir": {"dtype": "uint8"},
                "strord": {"dtype": "int16", "_FillValue": np.int16(0)},
                "basins": {"dtype": "int32", "_FillValue": np.int32(0)},
                "rivmsk_review": {"dtype": "uint8", "_FillValue": np.uint8(0)},
            },
        )
        return {
            "hydrography_cells": int(np.size(uparea)),
            "max_uparea_km2": float(np.nanmax(uparea)),
            "stream_cells": int(np.count_nonzero(stream_mask)),
            "resolution_degrees": _coordinate_resolution_degrees(dem, dem.rio.x_dim),
            "river_burn_features": int(len(burn_rivers)) if burn_rivers is not None else 0,
            "river_burn_method": str(river_burn_method or ""),
            "outlets": str(outlets or "edge"),
            "mask_geometry": str(mask_geometry or ""),
            "masked_cells": int(np.count_nonzero(~valid)),
        }
    finally:
        if dem is not None:
            with suppress(Exception):
                dem.close()


def _load_river_burn_geometry(river_geometry, *, target_crs) -> gpd.GeoDataFrame | None:
    if river_geometry is None:
        return None
    river_path = Path(river_geometry)
    if not river_path.exists():
        return None
    rivers = gpd.read_file(river_path)
    if rivers.empty:
        return None
    if rivers.crs is None:
        rivers = rivers.set_crs("EPSG:4326")
    rivers = rivers.to_crs(target_crs)
    uparea_col = _first_column(rivers, ("uparea", "uparea_km2", "TotDASqKm", "totdasqkm", "drainage_area_km2"))
    if uparea_col is not None and uparea_col != "uparea":
        rivers = rivers.copy()
        rivers["uparea"] = pd.to_numeric(rivers[uparea_col], errors="coerce")
    elif uparea_col is None:
        rivers = rivers.copy()
        rivers["uparea"] = 1.0
    rivers = rivers[rivers.geometry.notna() & ~rivers.geometry.is_empty].copy()
    rivers = rivers[pd.to_numeric(rivers["uparea"], errors="coerce").fillna(0.0) > 0].copy()
    if rivers.empty:
        return None
    return rivers


def _load_mask_geometry(mask_geometry, *, target_crs) -> gpd.GeoDataFrame | None:
    if mask_geometry in (None, ""):
        return None
    mask_path = Path(mask_geometry)
    if not mask_path.exists():
        return None
    mask = gpd.read_file(mask_path)
    if mask.empty:
        return None
    if mask.crs is None:
        mask = mask.set_crs("EPSG:4326")
    mask = mask.to_crs(target_crs)
    mask = mask[mask.geometry.notna() & ~mask.geometry.is_empty].copy()
    if mask.empty:
        return None
    return mask


def hydromt_basemap_readiness(
    hydrography_nc: Path,
    target_resolution_degrees: float,
    *,
    expected_outlets: str | None = None,
    expected_mask_geometry: str | None = None,
) -> dict:
    """Return fast QA for the collected HydroMT-Wflow hydrography basemap.

    Wflow land-slope derivation is sensitive to stale geographic CRS metadata. We
    require the collected DEM-derived basemap to be explicitly EPSG:4326 so HydroMT
    treats the degree grid as geographic and converts distances to metres.
    """
    import hydromt  # noqa: F401

    hydrography_nc = Path(hydrography_nc)
    if not hydrography_nc.exists():
        return {
            "status": "missing",
            "path": str(hydrography_nc),
            "crs_epsg": None,
            "resolution_degrees": None,
            "message": "missing basemap",
        }
    ds = None
    try:
        ds = xr.open_dataset(hydrography_nc)
        crs = ds.raster.crs
        epsg = crs.to_epsg() if crs is not None else None
        resolution = None
        for coord in ("x", "longitude", "lon"):
            if coord in ds.coords and ds[coord].size > 1:
                values = ds[coord].values
                resolution = float(abs(values[1] - values[0]))
                break
        resolution_ok = (
            resolution is not None
            and abs(resolution - float(target_resolution_degrees)) <= max(
                1.0e-9,
                float(target_resolution_degrees) * 0.01,
            )
        )
        crs_ok = epsg == 4326
        outlets = str(ds.attrs.get("outlets", "") or "")
        outlets_ok = expected_outlets is None or outlets == str(expected_outlets)
        mask_geometry = str(ds.attrs.get("mask_geometry", "") or "")
        mask_ok = expected_mask_geometry is None or mask_geometry == str(expected_mask_geometry or "")
        status = "ready" if resolution_ok and crs_ok and outlets_ok and mask_ok else "stale"
        messages = []
        if not crs_ok:
            messages.append(f"crs_epsg={epsg}; expected EPSG:4326")
        if not resolution_ok:
            messages.append(f"resolution={resolution}; expected {target_resolution_degrees}")
        if not outlets_ok:
            messages.append(f"outlets={outlets or 'unset'}; expected {expected_outlets}")
        if not mask_ok:
            messages.append(f"mask_geometry={mask_geometry or 'unset'}; expected {expected_mask_geometry}")
        return {
            "status": status,
            "path": str(hydrography_nc),
            "crs_epsg": epsg,
            "resolution_degrees": resolution,
            "river_burn_features": int(ds.attrs.get("river_burn_features", 0) or 0),
            "river_burn_method": str(ds.attrs.get("river_burn_method", "") or ""),
            "outlets": outlets,
            "mask_geometry": mask_geometry,
            "masked_cells": int(ds.attrs.get("masked_cells", 0) or 0),
            "message": "; ".join(messages) if messages else "ready",
        }
    except Exception as exc:
        return {
            "status": "stale",
            "path": str(hydrography_nc),
            "crs_epsg": None,
            "resolution_degrees": None,
            "message": f"unreadable basemap: {exc}",
        }
    finally:
        if ds is not None:
            ds.close()


def _hydromt_basemap_ready(
    hydrography_nc: Path,
    target_resolution_degrees: float,
    *,
    expected_outlets: str | None = None,
    expected_mask_geometry: str | None = None,
) -> bool:
    return (
        hydromt_basemap_readiness(
            hydrography_nc,
            target_resolution_degrees,
            expected_outlets=expected_outlets,
            expected_mask_geometry=expected_mask_geometry,
        )["status"]
        == "ready"
    )


def _coordinate_resolution_degrees(data_array, coord: str) -> float:
    values = data_array[coord].values
    if len(values) < 2:
        return 0.0
    return float(abs(values[1] - values[0]))


def write_review_hydrography_vectors(
    hydrography_nc,
    river_output,
    catchment_output,
    *,
    service_url=NHDPLUS_HR_MAPSERVER,
    timeout_seconds=120,
    stream_geo_table=None,
    nldi_lookup_cache=None,
    use_nldi_lookup=False,
    nldi_timeout_seconds=30,
    nldi_max_workers=4,
    nldi_progress_interval=500,
    stream_geo_join_method="attribute_transfer",
    nhdplus_v2_flowlines=None,
    stream_geo_join_max_distance_m=2500,
    session_get=None,
):
    import hydromt  # noqa: F401

    hydrography_nc = Path(hydrography_nc)
    river_output = Path(river_output)
    catchment_output = Path(catchment_output)
    river_output.parent.mkdir(parents=True, exist_ok=True)
    catchment_output.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(hydrography_nc)
    try:
        ds.raster.set_crs(_hydromt_crs(ds.raster.crs) or "EPSG:4269")
        bbox_wgs84 = tuple(float(value) for value in ds.raster.bounds)
        rivers = fetch_nhdplus_hr_layer(
            bbox_wgs84,
            layer_id=NHDPLUS_HR_NETWORK_FLOWLINE_LAYER,
            service_url=service_url,
            timeout_seconds=timeout_seconds,
            session_get=session_get,
        )
        catchments = fetch_nhdplus_hr_layer(
            bbox_wgs84,
            layer_id=NHDPLUS_HR_CATCHMENT_LAYER,
            service_url=service_url,
            timeout_seconds=timeout_seconds,
            session_get=session_get,
        )
        if rivers.empty:
            raise ValueError(f"NHDPlus HR NetworkNHDFlowline returned no features for bbox {bbox_wgs84}")
        if catchments.empty:
            raise ValueError(f"NHDPlus HR NHDPlusCatchment returned no features for bbox {bbox_wgs84}")
        stream_geo = load_stream_geo_table(stream_geo_table) if stream_geo_table and Path(stream_geo_table).exists() else None
        stream_geo_comid_keys = None
        method = str(stream_geo_join_method or "").lower()
        if stream_geo is not None and not stream_geo.empty and method in {"pynhd_waterdata", "waterdata", "nhdplus_v2_nearest"}:
            nhdplus_v2 = load_or_fetch_nhdplus_v2_flowlines(
                bbox_wgs84,
                nhdplus_v2_flowlines,
                skip_existing=True,
            )
            stream_geo_comid_keys = _stream_geo_comid_keys_from_flowlines(
                rivers,
                nhdplus_v2,
                river_id_column=_first_column(rivers, ("comid", "COMID", "ComID", "nhdplusid", "NHDPlusID", "featureid", "FeatureID")),
                max_distance_m=stream_geo_join_max_distance_m,
            )
        rivers = _prepare_nhdplus_river_geometry(
            rivers,
            stream_geo=stream_geo,
            stream_geo_comid_keys=stream_geo_comid_keys,
            stream_geo_join_method=method,
            nldi_lookup_cache=nldi_lookup_cache,
            use_nldi_lookup=use_nldi_lookup and stream_geo_comid_keys is None,
            nldi_timeout_seconds=nldi_timeout_seconds,
            nldi_max_workers=nldi_max_workers,
            nldi_progress_interval=nldi_progress_interval,
            session_get=session_get,
        )
        rivers.to_file(river_output, driver="GPKG")
        catchments = _prepare_nhdplus_catchments(catchments)
        catchments.to_file(catchment_output, driver="GPKG")
        return {
            "river_features": int(len(rivers)),
            "catchment_features": int(len(catchments)),
            "river_source": "USGS NHDPlus HR NetworkNHDFlowline",
            "catchment_source": "USGS NHDPlus HR NHDPlusCatchment",
            "stream_geo_join_method": method or "direct_comid",
        }
    finally:
        ds.close()


def fetch_nhdplus_hr_layer(
    bbox_wgs84,
    *,
    layer_id,
    service_url=NHDPLUS_HR_MAPSERVER,
    timeout_seconds=120,
    session_get=None,
):
    """Fetch one NHDPlus HR ArcGIS REST layer as GeoDataFrame."""
    features = []
    offset = 0
    limit = 2000
    get = session_get or requests.get
    while True:
        response = get(
            f"{service_url.rstrip('/')}/{int(layer_id)}/query",
            params={
                "where": "1=1",
                "geometry": ",".join(str(float(value)) for value in bbox_wgs84),
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
                "f": "geojson",
                "resultOffset": offset,
                "resultRecordCount": limit,
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        batch = payload.get("features", [])
        features.extend(batch)
        if not payload.get("exceededTransferLimit") and len(batch) < limit:
            break
        offset += len(batch)
        if not batch:
            break
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


def fetch_wbd_huc(
    bbox_wgs84,
    *,
    huc_level,
    service_url=WBD_MAPSERVER,
    layer_id=None,
    timeout_seconds=120,
    session_get=None,
):
    """Fetch WBD HUC polygons of one level intersecting a bbox, with a normalised id.

    Returns a GeoDataFrame with a ``huc_id`` (the ``huc<level>`` code) and ``huc_level``
    column. Used to pick the smallest single watershed that encapsulates the SFINCS
    coverage boxes as the Wflow domain.
    """
    if layer_id is None:
        layer_id = WBD_HUC_LAYER_BY_LEVEL[int(huc_level)]
    hucs = fetch_nhdplus_hr_layer(
        bbox_wgs84,
        layer_id=layer_id,
        service_url=service_url,
        timeout_seconds=timeout_seconds,
        session_get=session_get,
    )
    if hucs.empty:
        return hucs
    code_column = next(
        (column for column in (f"huc{int(huc_level)}", f"HUC{int(huc_level)}", "huc", "HUC") if column in hucs.columns),
        None,
    )
    hucs = hucs.copy()
    hucs["huc_id"] = hucs[code_column].astype(str) if code_column else ""
    hucs["huc_level"] = int(huc_level)
    return hucs


def fetch_nhdplus_hr_catchments(
    bbox_wgs84,
    *,
    service_url=NHDPLUS_HR_MAPSERVER,
    timeout_seconds=180,
    session_get=None,
):
    """Fetch + prepare NHDPlus HR catchments for a bbox (on-demand region-setup evidence)."""
    catchments = fetch_nhdplus_hr_layer(
        bbox_wgs84,
        layer_id=NHDPLUS_HR_CATCHMENT_LAYER,
        service_url=service_url,
        timeout_seconds=timeout_seconds,
        session_get=session_get,
    )
    if catchments.empty:
        raise ValueError(f"NHDPlus HR NHDPlusCatchment returned no features for bbox {bbox_wgs84}")
    return _prepare_nhdplus_catchments(catchments)


def write_wflow_reservoir_waterbodies(
    search_geometry,
    output_path,
    *,
    river_geometry=None,
    service_url=NHDPLUS_HR_MAPSERVER,
    timeout_seconds=120,
    min_area_km2=1.0,
    default_depth_m=5.0,
    default_discharge_m3s=0.01,
    source_layer_id=NHDPLUS_HR_WATERBODY_LAYER,
    condition_cfg=None,
    location_root=None,
    session_get=None,
):
    """Fetch public NHDPlus HR waterbodies and write Wflow reservoir input fields."""
    location_root = Path(location_root or ".")
    search_path = _location_path(location_root, search_geometry)
    if not search_path.exists():
        raise FileNotFoundError(search_path)
    region = gpd.read_file(search_path)
    if region.empty:
        raise ValueError(f"Wflow reservoir search geometry is empty: {search_path}")
    if region.crs is None:
        region = region.set_crs("EPSG:4326")
    region = region.to_crs("EPSG:4326")
    geom = region.geometry.union_all()
    minx, miny, maxx, maxy = geom.bounds
    raw = fetch_nhdplus_hr_layer(
        (minx, miny, maxx, maxy),
        layer_id=source_layer_id,
        service_url=service_url,
        timeout_seconds=timeout_seconds,
        session_get=session_get,
    )
    prepared = prepare_nhdplus_hr_waterbodies_for_wflow(
        raw,
        geom,
        river_geometry=river_geometry,
        min_area_km2=min_area_km2,
        default_depth_m=default_depth_m,
        default_discharge_m3s=default_discharge_m3s,
    )
    condition_summary = {}
    if condition_cfg and condition_cfg.get("enabled", False):
        enriched = enrich_wflow_reservoirs_with_public_conditions(
            prepared,
            condition_cfg,
            location_root=location_root,
            skip_existing=True,
            session_get=session_get,
        )
        prepared = enriched.frame
        condition_summary = {
            "reservoir_condition_status": enriched.provenance.get("status", "collected"),
            "reservoir_condition_provider": enriched.provenance.get("provider", ""),
            "reservoir_condition_matched": enriched.provenance.get("matched_reservoirs", 0),
            "reservoir_condition_unmatched": enriched.provenance.get("unmatched_reservoirs", 0),
            "reservoir_condition_summary": enriched.provenance.get("summary_csv", ""),
        }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_file(output_path, driver="GPKG")
    return {
        "reservoir_status": "collected",
        "reservoir_source": "USGS NHDPlus HR NHDWaterbody",
        "reservoir_features": int(len(prepared)),
        "reservoir_min_area_km2": float(min_area_km2),
        "reservoir_parameter_status": "source_backed_no_control" if condition_summary else "estimated_no_control",
        "reservoir_output": str(output_path),
        **condition_summary,
    }


def prepare_nhdplus_hr_waterbodies_for_wflow(
    waterbodies,
    search_geometry,
    *,
    river_geometry=None,
    min_area_km2=1.0,
    default_depth_m=5.0,
    default_discharge_m3s=0.01,
):
    """Return NHDPlus HR waterbodies with HydroMT-Wflow reservoir columns."""
    if waterbodies is None or waterbodies.empty:
        return _empty_wflow_reservoir_frame()
    gdf = waterbodies.copy()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty & gdf.geometry.intersects(search_geometry)].copy()
    if "ftype" in gdf:
        gdf = gdf[pd.to_numeric(gdf["ftype"], errors="coerce").eq(390)].copy()
    area_km2 = pd.to_numeric(gdf.get("areasqkm", pd.Series(dtype=float)), errors="coerce")
    gdf = gdf.loc[area_km2 >= float(min_area_km2)].copy()
    if gdf.empty:
        return _empty_wflow_reservoir_frame()
    gdf["_area_km2"] = pd.to_numeric(gdf["areasqkm"], errors="coerce")
    gdf = gdf.sort_values(["_area_km2", "nhdplusid"], ascending=[False, True]).reset_index(drop=True)
    gdf["waterbody_id"] = np.arange(1, len(gdf) + 1, dtype="int64")
    gdf["source_nhdplusid"] = gdf.get("nhdplusid", pd.Series(index=gdf.index, dtype=object)).astype(str)
    gdf["waterbody_name"] = gdf.get("gnis_name", pd.Series(index=gdf.index, dtype=object)).fillna("").astype(str)
    gdf["Area_avg"] = gdf["_area_km2"].astype(float) * 1_000_000.0
    gdf["Depth_avg"] = float(default_depth_m)
    gdf["Vol_avg"] = gdf["Area_avg"] * gdf["Depth_avg"]
    gdf["Dis_avg"] = _waterbody_discharge_estimates(
        gdf,
        river_geometry,
        default_discharge_m3s=float(default_discharge_m3s),
    )
    gdf["reservoir_parameter_source"] = "NHDPlus HR geometry + estimated no-control defaults"
    gdf["review_status"] = "review_required_public_waterbody_estimates"
    gdf["reservoir_operation"] = "no_control"
    keep = [
        "waterbody_id",
        "source_nhdplusid",
        "waterbody_name",
        "Area_avg",
        "Depth_avg",
        "Vol_avg",
        "Dis_avg",
        "areasqkm",
        "elevation",
        "ftype",
        "fcode",
        "purpcode",
        "onoffnet",
        "reservoir_operation",
        "reservoir_parameter_source",
        "review_status",
        "geometry",
    ]
    return gpd.GeoDataFrame(gdf[[column for column in keep if column in gdf.columns]], geometry="geometry", crs="EPSG:4326")


def _waterbody_discharge_estimates(waterbodies, river_geometry, *, default_discharge_m3s):
    estimates = pd.Series(default_discharge_m3s, index=waterbodies.index, dtype=float)
    if river_geometry in (None, ""):
        return estimates
    river_path = Path(river_geometry)
    if not river_path.exists():
        return estimates
    rivers = gpd.read_file(river_path)
    if rivers.empty or "qbankfull" not in rivers:
        return estimates
    if rivers.crs is None:
        rivers = rivers.set_crs("EPSG:4326")
    projected_crs = _local_projected_crs(waterbodies)
    wb = waterbodies.to_crs(projected_crs).copy()
    rv = rivers.to_crs(projected_crs).copy()
    rv = rv[pd.to_numeric(rv["qbankfull"], errors="coerce").notna()].copy()
    if rv.empty:
        return estimates
    joined = gpd.sjoin_nearest(
        wb[["waterbody_id", "geometry"]],
        rv[["qbankfull", "geometry"]],
        how="left",
        max_distance=5000,
        distance_col="river_distance_m",
    )
    q = pd.to_numeric(joined["qbankfull"], errors="coerce").groupby(joined.index).max()
    estimates.loc[q.index] = q.fillna(default_discharge_m3s).clip(lower=default_discharge_m3s)
    return estimates


def _empty_wflow_reservoir_frame():
    return gpd.GeoDataFrame(
        columns=[
            "waterbody_id",
            "source_nhdplusid",
            "waterbody_name",
            "Area_avg",
            "Depth_avg",
            "Vol_avg",
            "Dis_avg",
            "reservoir_operation",
            "reservoir_parameter_source",
            "review_status",
            "geometry",
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )


def load_or_fetch_nhdplus_v2_flowlines(
    bbox_wgs84,
    output_path,
    *,
    skip_existing=True,
) -> gpd.GeoDataFrame:
    """Load/cache NHDPlusV2/MR flowlines for bulk STREAM-geo COMID joins."""
    if output_path in (None, ""):
        raise RuntimeError(
            "STREAM-geo bulk join requires a configured NHDPlusV2 flowline cache path "
            "(collection.national_hydrography.nhdplus_v2_flowlines)."
        )
    output_path = Path(output_path)
    if skip_existing and output_path.exists():
        return gpd.read_file(output_path).to_crs("EPSG:4326")
    try:
        from pynhd import WaterData
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "STREAM-geo bulk join should use PyNHD WaterData instead of thousands of NLDI point calls. "
            "Install/sync the optional source dependency `pynhd`, then rerun 02_collect_sources. "
            "The researched path is WaterData('nhdflowline_network').bybox(...) followed by a local nearest join."
        ) from exc

    flowlines = WaterData("nhdflowline_network").bybox(tuple(float(value) for value in bbox_wgs84))
    if flowlines.empty:
        raise ValueError(f"PyNHD WaterData nhdflowline_network returned no features for bbox {bbox_wgs84}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flowlines.to_crs("EPSG:4326").to_file(output_path, driver="GPKG")
    return flowlines.to_crs("EPSG:4326")


def _stream_geo_comid_keys_from_flowlines(
    rivers: gpd.GeoDataFrame,
    flowlines: gpd.GeoDataFrame,
    *,
    river_id_column: str | None,
    max_distance_m=2500,
) -> pd.DataFrame:
    if river_id_column is None:
        return pd.DataFrame(columns=["_river_id_key", "_stream_geo_comid_key", "stream_geo_join_distance_m"])
    flowline_comid = _first_column(flowlines, ("comid", "COMID", "ComID", "nhdplus_comid", "NHDPlus_COMID", "featureid", "FeatureID"))
    if flowline_comid is None:
        return pd.DataFrame(columns=["_river_id_key", "_stream_geo_comid_key", "stream_geo_join_distance_m"])

    river_points = rivers[[river_id_column, "geometry"]].copy()
    river_points["_river_id_key"] = river_points[river_id_column].astype(str)
    river_points["geometry"] = river_points.geometry.map(_river_lookup_point)
    river_points = gpd.GeoDataFrame(river_points.dropna(subset=["geometry"]), geometry="geometry", crs=rivers.crs or "EPSG:4326").to_crs("EPSG:4326")
    flowlines = flowlines[[flowline_comid, "geometry"]].copy()
    flowlines["_stream_geo_comid_key"] = flowlines[flowline_comid].astype(str)
    flowlines = gpd.GeoDataFrame(flowlines.dropna(subset=["geometry"]), geometry="geometry", crs=flowlines.crs or "EPSG:4326").to_crs("EPSG:4326")
    if river_points.empty or flowlines.empty:
        return pd.DataFrame(columns=["_river_id_key", "_stream_geo_comid_key", "stream_geo_join_distance_m"])

    projected_crs = _local_projected_crs(river_points)
    left = river_points[["_river_id_key", "geometry"]].to_crs(projected_crs)
    right = flowlines[["_stream_geo_comid_key", "geometry"]].to_crs(projected_crs)
    joined = gpd.sjoin_nearest(
        left,
        right,
        how="left",
        max_distance=float(max_distance_m) if max_distance_m else None,
        distance_col="stream_geo_join_distance_m",
    )
    return pd.DataFrame(joined[["_river_id_key", "_stream_geo_comid_key", "stream_geo_join_distance_m"]])


def _local_projected_crs(frame: gpd.GeoDataFrame):
    with suppress(Exception):
        crs = frame.estimate_utm_crs()
        if crs is not None:
            return crs
    return "EPSG:5070"


def _prepare_nhdplus_river_geometry(
    rivers,
    *,
    stream_geo=None,
    stream_geo_comid_keys=None,
    stream_geo_join_method="direct_comid",
    nldi_lookup_cache=None,
    use_nldi_lookup=False,
    nldi_timeout_seconds=30,
    nldi_max_workers=4,
    nldi_progress_interval=500,
    session_get=None,
):
    rivers = rivers.to_crs("EPSG:4326").copy()
    if stream_geo is not None and not stream_geo.empty:
        if str(stream_geo_join_method or "").lower() in {"attribute_transfer", "stream_geo_attribute_transfer", "hydrologic_attribute_transfer"}:
            rivers = enrich_river_geometry_with_stream_geo_attribute_transfer(rivers, stream_geo)
        else:
            rivers = enrich_river_geometry_with_stream_geo(
                rivers,
                stream_geo,
                stream_geo_comid_keys=stream_geo_comid_keys,
                nldi_lookup_cache=nldi_lookup_cache,
                use_nldi_lookup=use_nldi_lookup,
                nldi_timeout_seconds=nldi_timeout_seconds,
                nldi_max_workers=nldi_max_workers,
                nldi_progress_interval=nldi_progress_interval,
                session_get=session_get,
            )
    if "rivwth" not in rivers or rivers["rivwth"].isna().all():
        rivers["rivwth"] = _estimate_river_width(rivers)
        rivers["rivwth_source"] = "drainage_area_formula_fallback"
    else:
        rivers["rivwth"] = pd.to_numeric(rivers["rivwth"], errors="coerce").fillna(_estimate_river_width(rivers)).astype("float32")
        rivers["rivwth_source"] = rivers.get("rivwth_source", "source_geometry")
    if "rivdph" not in rivers or rivers["rivdph"].isna().all():
        rivers["rivdph"] = np.nan
        rivers["rivdph_source"] = "missing_native_powlaw_fallback"
    if "qbankfull" not in rivers or rivers["qbankfull"].isna().all():
        rivers["qbankfull"] = _estimate_bankfull_discharge(rivers)
        rivers["qbankfull_source"] = "drainage_area_formula_fallback"
    rivwth_source = rivers["rivwth_source"] if "rivwth_source" in rivers else pd.Series("", index=rivers.index)
    rivers["review_status"] = np.where(
        rivwth_source.astype(str).str.contains("STREAM-geo", na=False),
        "review_required_stream_geo",
        "review_required_formula_fallback",
    )
    rivers["source"] = "USGS NHDPlus HR NetworkNHDFlowline"
    return rivers


def load_stream_geo_table(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".gpkg", ".geojson", ".shp"}:
        return gpd.read_file(path)
    return pd.read_csv(path)


def enrich_river_geometry_with_stream_geo(
    rivers,
    stream_geo,
    *,
    stream_geo_comid_keys=None,
    nldi_lookup_cache=None,
    use_nldi_lookup=False,
    nldi_timeout_seconds=30,
    nldi_max_workers=4,
    nldi_progress_interval=500,
    session_get=None,
) -> gpd.GeoDataFrame:
    """Join STREAM-geo width/depth estimates onto NHDPlus river geometry."""
    rivers = rivers.copy()
    stream_geo = pd.DataFrame(stream_geo).copy()
    river_comid = _first_column(rivers, ("comid", "COMID", "ComID", "nhdplusid", "NHDPlusID", "featureid", "FeatureID"))
    stream_comid = _first_column(stream_geo, ("comid", "COMID", "ComID", "nhdplusid", "NHDPlusID", "featureid", "FeatureID"))
    if river_comid is None or stream_comid is None:
        rivers["rivwth_source"] = "missing_comid_formula_fallback"
        return rivers
    width_col = _first_column(
        stream_geo,
        ("rivwth", "XGB_Width_m", "RF_Width_m", "MLP_Width_m", "width_m", "width", "pred_width", "river_width"),
    )
    depth_col = _first_column(
        stream_geo,
        ("rivdph", "XGB_Depth_m", "RF_Depth_m", "MLP_Depth_m", "depth_m", "depth", "pred_depth", "river_depth"),
    )
    q_col = _first_column(stream_geo, ("qbankfull", "bankfull_discharge", "qbf", "Qbf"))
    keep = [stream_comid, *[col for col in (width_col, depth_col, q_col) if col is not None]]
    lookup = stream_geo[keep].copy()
    lookup["_comid_key"] = lookup[stream_comid].astype(str)
    lookup = lookup.drop_duplicates("_comid_key")
    rename = {}
    if width_col:
        rename[width_col] = "_stream_geo_rivwth"
    if depth_col:
        rename[depth_col] = "_stream_geo_rivdph"
    if q_col:
        rename[q_col] = "_stream_geo_qbankfull"
    lookup = lookup.rename(columns=rename)
    rivers["_comid_key"] = rivers[river_comid].astype(str)
    rivers["_river_id_key"] = rivers[river_comid].astype(str)
    source_label = "STREAM-geo"
    if stream_geo_comid_keys is not None and not stream_geo_comid_keys.empty:
        rivers = rivers.merge(stream_geo_comid_keys, on="_river_id_key", how="left")
        rivers["_comid_key"] = rivers["_stream_geo_comid_key"].where(rivers["_stream_geo_comid_key"].notna(), rivers["_comid_key"])
        if "stream_geo_join_distance_m" in rivers:
            rivers["stream_geo_join_distance_m"] = pd.to_numeric(rivers["stream_geo_join_distance_m"], errors="coerce")
        source_label = "STREAM-geo/NHDPlusV2-nearest"
    elif use_nldi_lookup:
        nldi_keys = _nldi_stream_geo_comid_keys(
            rivers,
            river_id_column=river_comid,
            cache_path=nldi_lookup_cache,
            timeout_seconds=nldi_timeout_seconds,
            max_workers=nldi_max_workers,
            progress_interval=nldi_progress_interval,
            session_get=session_get,
        )
        if not nldi_keys.empty:
            rivers = rivers.merge(nldi_keys, on="_river_id_key", how="left")
            rivers["_comid_key"] = rivers["_nldi_comid_key"].where(rivers["_nldi_comid_key"].notna(), rivers["_comid_key"])
            source_label = "STREAM-geo/NLDI"
    rivers = rivers.merge(lookup.drop(columns=[stream_comid]), on="_comid_key", how="left")
    if "_stream_geo_rivwth" in rivers:
        rivers["rivwth"] = pd.to_numeric(rivers["_stream_geo_rivwth"], errors="coerce")
        rivers["rivwth_source"] = np.where(rivers["rivwth"].notna(), source_label, "drainage_area_formula_fallback")
    if "_stream_geo_rivdph" in rivers:
        rivers["rivdph"] = pd.to_numeric(rivers["_stream_geo_rivdph"], errors="coerce")
        rivers["rivdph_source"] = np.where(rivers["rivdph"].notna(), source_label, "missing_native_powlaw_fallback")
    if "_stream_geo_qbankfull" in rivers:
        rivers["qbankfull"] = pd.to_numeric(rivers["_stream_geo_qbankfull"], errors="coerce")
        rivers["qbankfull_source"] = np.where(rivers["qbankfull"].notna(), source_label, "drainage_area_formula_fallback")
    drop_columns = [
        col
        for col in rivers.columns
        if col.startswith("_stream_geo_") or col in {"_comid_key", "_river_id_key", "_nldi_comid_key"}
    ]
    return rivers.drop(columns=drop_columns)


def enrich_river_geometry_with_stream_geo_attribute_transfer(
    rivers,
    stream_geo,
    *,
    neighbors=5,
) -> gpd.GeoDataFrame:
    """Transfer STREAM-geo width/depth to NHDPlus HR reaches by hydrologic attributes."""
    from sklearn.neighbors import NearestNeighbors

    rivers = rivers.copy()
    stream_geo = pd.DataFrame(stream_geo).copy()
    river_order = _first_column(rivers, ("streamorde", "StreamOrde", "stream_order", "strord"))
    river_area = _first_column(rivers, ("totdasqkm", "TotDASqKM", "areasqkm", "uparea", "uparea_km2"))
    river_length = _first_column(rivers, ("lengthkm", "LENGTHKM", "slopelenkm", "Shape_Length"))
    stream_order = _first_column(stream_geo, ("StreamOrde", "streamorde", "stream_order", "strord"))
    stream_area = _first_column(stream_geo, ("TotDASqKM", "totdasqkm", "areasqkm", "uparea", "uparea_km2"))
    stream_length = _first_column(stream_geo, ("LENGTHKM", "lengthkm", "slopelenkm"))
    width_col = _first_column(stream_geo, ("XGB_Width_m", "RF_Width_m", "MLP_Width_m", "rivwth", "width_m", "width"))
    depth_col = _first_column(stream_geo, ("XGB_Depth_m", "RF_Depth_m", "MLP_Depth_m", "rivdph", "depth_m", "depth"))
    if None in (river_order, river_area, river_length, stream_order, stream_area, stream_length, width_col, depth_col):
        rivers["rivwth_source"] = "missing_stream_geo_attribute_transfer_inputs"
        rivers["rivdph_source"] = "missing_stream_geo_attribute_transfer_inputs"
        return rivers

    stream_features = pd.DataFrame(
        {
            "order": pd.to_numeric(stream_geo[stream_order], errors="coerce"),
            "area": pd.to_numeric(stream_geo[stream_area], errors="coerce"),
            "length": pd.to_numeric(stream_geo[stream_length], errors="coerce"),
            "width": pd.to_numeric(stream_geo[width_col], errors="coerce"),
            "depth": pd.to_numeric(stream_geo[depth_col], errors="coerce"),
        }
    )
    valid_stream = (
        np.isfinite(stream_features[["order", "area", "length", "width", "depth"]]).all(axis=1)
        & (stream_features["area"] > 0)
        & (stream_features["length"] > 0)
        & (stream_features["width"] > 0)
        & (stream_features["depth"] > 0)
    )
    stream_features = stream_features.loc[valid_stream].reset_index(drop=True)
    if stream_features.empty:
        rivers["rivwth_source"] = "empty_stream_geo_attribute_transfer_inputs"
        rivers["rivdph_source"] = "empty_stream_geo_attribute_transfer_inputs"
        return rivers

    river_features = pd.DataFrame(
        {
            "order": pd.to_numeric(rivers[river_order], errors="coerce"),
            "area": pd.to_numeric(rivers[river_area], errors="coerce"),
            "length": pd.to_numeric(rivers[river_length], errors="coerce"),
        },
        index=rivers.index,
    )
    query_features = _stream_geo_attribute_matrix(river_features)
    source_features = _stream_geo_attribute_matrix(stream_features)
    model = NearestNeighbors(n_neighbors=max(1, min(int(neighbors), len(stream_features))), algorithm="auto", metric="euclidean", n_jobs=-1)
    model.fit(source_features)
    distances, indices = model.kneighbors(query_features)
    widths = np.nanmedian(stream_features["width"].to_numpy()[indices], axis=1)
    depths = np.nanmedian(stream_features["depth"].to_numpy()[indices], axis=1)
    rivers["rivwth"] = widths.astype("float32")
    rivers["rivdph"] = depths.astype("float32")
    rivers["stream_geo_attribute_distance"] = np.nanmedian(distances, axis=1).astype("float32")
    rivers["rivwth_source"] = "STREAM-geo_attribute_transfer"
    rivers["rivdph_source"] = "STREAM-geo_attribute_transfer"
    return rivers


def _stream_geo_attribute_matrix(frame: pd.DataFrame) -> np.ndarray:
    order = pd.to_numeric(frame["order"], errors="coerce").fillna(0).clip(lower=0)
    area = pd.to_numeric(frame["area"], errors="coerce").fillna(0).clip(lower=1e-6)
    length = pd.to_numeric(frame["length"], errors="coerce").fillna(0).clip(lower=1e-6)
    return np.column_stack(
        [
            order.to_numpy(dtype="float64") * 2.0,
            np.log10(area.to_numpy(dtype="float64")),
            np.log10(length.to_numpy(dtype="float64")),
        ]
    )


def _nldi_stream_geo_comid_keys(
    rivers: gpd.GeoDataFrame,
    *,
    river_id_column: str,
    cache_path,
    timeout_seconds=30,
    max_workers=4,
    progress_interval=500,
    session_get=None,
) -> pd.DataFrame:
    """Return cached NHDPlus HR id -> NHDPlusV2 COMID keys from NLDI point lookup."""
    if cache_path in (None, ""):
        return pd.DataFrame(columns=["_river_id_key", "_nldi_comid_key"])
    cache_path = Path(cache_path)
    if cache_path.exists():
        cache = pd.read_csv(cache_path, dtype=str)
    else:
        cache = pd.DataFrame(columns=["river_id", "lon", "lat", "nldi_comid", "status"])
    if "river_id" not in cache:
        cache["river_id"] = pd.Series(dtype=str)
    known = set(cache["river_id"].dropna().astype(str))

    pending = []
    for _, row in rivers.iterrows():
        river_id = str(row[river_id_column])
        if river_id in known:
            continue
        point = _river_lookup_point(row.geometry)
        if point is None:
            pending.append({"river_id": river_id, "lon": "", "lat": "", "status": "missing_geometry"})
            continue
        pending.append({"river_id": river_id, "lon": float(point.x), "lat": float(point.y), "status": "pending"})

    rows = []
    completed = 0

    def lookup(item):
        if item["status"] == "missing_geometry":
            return {**item, "nldi_comid": ""}
        try:
            comid = fetch_nldi_comid(
                float(item["lon"]),
                float(item["lat"]),
                timeout_seconds=timeout_seconds,
                session_get=session_get,
            )
            return {**item, "nldi_comid": comid or "", "status": "matched" if comid else "no_match"}
        except Exception as exc:
            return {**item, "nldi_comid": "", "status": f"error:{type(exc).__name__}"}

    def flush_rows():
        nonlocal cache, rows
        if not rows:
            return
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache = cache.drop_duplicates("river_id", keep="last")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(cache_path, index=False)
        rows = []

    if pending:
        max_workers = max(1, int(max_workers or 1))
        progress_interval = max(0, int(progress_interval or 0))
        if max_workers == 1 or len(pending) == 1:
            get = session_get
            close_session = None
            if get is None:
                session = requests.Session()
                get = session.get
                close_session = session.close
            try:
                for item in pending:
                    rows.append(lookup(item) if session_get is not None else _nldi_lookup_item(item, timeout_seconds, get))
                    completed += 1
                    if progress_interval and completed % progress_interval == 0:
                        flush_rows()
                        print(f"NLDI COMID lookup cached {completed}/{len(pending)} new river reaches")
            finally:
                if close_session is not None:
                    close_session()
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(lookup, item) for item in pending]
                for future in as_completed(futures):
                    rows.append(future.result())
                    completed += 1
                    if progress_interval and completed % progress_interval == 0:
                        flush_rows()
                        print(f"NLDI COMID lookup cached {completed}/{len(pending)} new river reaches")
        flush_rows()

    out = cache[["river_id", "nldi_comid"]].copy()
    out["_river_id_key"] = out["river_id"].astype(str)
    out["_nldi_comid_key"] = out["nldi_comid"].replace("", np.nan).astype("string")
    return out[["_river_id_key", "_nldi_comid_key"]]


def _nldi_lookup_item(item, timeout_seconds, session_get):
    if item["status"] == "missing_geometry":
        return {**item, "nldi_comid": ""}
    try:
        comid = fetch_nldi_comid(
            float(item["lon"]),
            float(item["lat"]),
            timeout_seconds=timeout_seconds,
            session_get=session_get,
        )
        return {**item, "nldi_comid": comid or "", "status": "matched" if comid else "no_match"}
    except Exception as exc:
        return {**item, "nldi_comid": "", "status": f"error:{type(exc).__name__}"}


def _river_lookup_point(geometry):
    if geometry is None or geometry.is_empty:
        return None
    try:
        return geometry.interpolate(0.5, normalized=True)
    except Exception:
        return geometry.representative_point()


def fetch_nldi_comid(lon: float, lat: float, *, base_url=NLDI_BASE_URL, timeout_seconds=30, session_get=None) -> str | None:
    get = session_get or requests.get
    response = get(
        f"{base_url.rstrip('/')}/comid/position",
        params={"coords": f"POINT({float(lon)} {float(lat)})"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    features = payload.get("features") or []
    if not features:
        return None
    props = features[0].get("properties") or {}
    value = props.get("identifier") or props.get("comid") or props.get("COMID")
    return None if value in (None, "") else str(value)


def _first_column(frame, candidates) -> str | None:
    lower = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        if str(candidate).lower() in lower:
            return lower[str(candidate).lower()]
    return None


def _prepare_nhdplus_catchments(catchments):
    catchments = catchments.to_crs("EPSG:4326").copy()
    if "basid" not in catchments:
        id_candidates = [name for name in ("featureid", "FeatureID", "nhdplusid", "NHDPlusID", "hydroseq", "HydroSeq") if name in catchments]
        if id_candidates:
            catchments["basid"] = pd.to_numeric(catchments[id_candidates[0]], errors="coerce").fillna(0).astype("int64")
        else:
            catchments["basid"] = np.arange(1, len(catchments) + 1, dtype="int64")
    catchments["review_status"] = "review_required_usgs_nhdplus_hr"
    catchments["source"] = "USGS NHDPlus HR NHDPlusCatchment"
    return catchments


def _estimate_river_width(rivers):
    if "TotDASqKm" in rivers:
        drainage_area = pd.to_numeric(rivers["TotDASqKm"], errors="coerce")
    elif "areasqkm" in rivers:
        drainage_area = pd.to_numeric(rivers["areasqkm"], errors="coerce")
    else:
        drainage_area = pd.Series(np.nan, index=rivers.index)
    width = 5.0 + 2.5 * np.sqrt(drainage_area.fillna(100.0).clip(lower=1.0))
    return width.clip(lower=10.0).astype("float32")


def _estimate_bankfull_discharge(rivers):
    if "TotDASqKm" in rivers:
        drainage_area = pd.to_numeric(rivers["TotDASqKm"], errors="coerce")
    elif "areasqkm" in rivers:
        drainage_area = pd.to_numeric(rivers["areasqkm"], errors="coerce")
    else:
        drainage_area = pd.Series(np.nan, index=rivers.index)
    qbankfull = 2.0 * drainage_area.fillna(100.0).clip(lower=1.0) ** 0.8
    return qbankfull.clip(lower=1.0).astype("float32")


def write_ssurgo_wflow_soil_dataset(soil_polygons, soil_attributes, template_raster, output_path):
    """Write SoilGrids-shaped SSURGO horizon properties consumed by setup_soilmaps."""
    import hydromt  # noqa: F401
    import rasterio
    from rasterio.features import rasterize

    soil_polygons = Path(soil_polygons)
    soil_attributes = Path(soil_attributes)
    template_raster = Path(template_raster)
    output_path = Path(output_path)
    required = {
        "mukey",
        "hzdept_r",
        "hzdepb_r",
        "sandtotal_r",
        "silttotal_r",
        "claytotal_r",
        "dbthirdbar_r",
        "om_r",
        "ph1to1h2o_r",
    }
    if not soil_polygons.exists():
        raise FileNotFoundError(f"SSURGO polygons are required before Wflow soil preparation: {soil_polygons}")
    if not soil_attributes.exists():
        raise FileNotFoundError(f"SSURGO horizon attributes are required before Wflow soil preparation: {soil_attributes}")
    attrs = pd.read_csv(soil_attributes)
    missing = required - set(attrs.columns)
    if missing:
        raise ValueError(
            "SSURGO attributes are missing Wflow pedology columns. Rerun 01_region_setup "
            f"so Soil Data Access refreshes {soil_attributes}. Missing: {sorted(missing)}"
        )
    soils = gpd.read_file(soil_polygons)
    if "mukey" not in soils:
        raise ValueError("SSURGO polygons must contain a mukey column")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(template_raster) as src:
        profile = src.profile
        shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs

    if soils.crs is None:
        soils = soils.set_crs(crs)
    elif crs is not None and soils.crs != crs:
        soils = soils.to_crs(crs)

    attrs["mukey"] = attrs["mukey"].astype(str)
    soils = soils.copy()
    soils["mukey"] = soils["mukey"].astype(str)
    layer_values = _ssurgo_soilgrids_layer_values(attrs)
    merged = soils.merge(layer_values, on="mukey", how="left")
    ds = xr.Dataset(
        coords={
            "y": np.arange(shape[0]) * profile["transform"].e + profile["transform"].f + profile["transform"].e / 2,
            "x": np.arange(shape[1]) * profile["transform"].a + profile["transform"].c + profile["transform"].a / 2,
        }
    )
    for column in [col for col in layer_values.columns if col != "mukey"]:
        values = rasterize(
            (
                (geom, float(value))
                for geom, value in zip(merged.geometry, merged[column])
                if geom is not None and not geom.is_empty and pd.notna(value)
            ),
            out_shape=shape,
            transform=transform,
            fill=-9999.0,
            dtype="float32",
        )
        ds[column] = (("y", "x"), values.astype("float32"))
        ds[column].raster.set_nodata(-9999.0)
    ds.attrs.update(
        {
            "source": "SSURGO Soil Data Access mapunit/component/chorizon",
            "status": "review_required",
            "review_note": "SoilGrids-shaped SSURGO horizon properties for HydroMT-Wflow setup_soilmaps.",
        }
    )
    ds.raster.set_crs(_hydromt_crs(crs))
    _write_netcdf_atomically(ds, output_path)
    valid = ds["soilthickness"].values != -9999 if "soilthickness" in ds else []
    return {"wflow_soil_pixels": int(np.count_nonzero(valid)), "ssurgo_mapunits": int(len(layer_values))}


def _ssurgo_soilgrids_layer_values(attrs):
    depths_cm = [0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0]
    numeric = [
        "hzdept_r",
        "hzdepb_r",
        "sandtotal_r",
        "silttotal_r",
        "claytotal_r",
        "dbthirdbar_r",
        "om_r",
        "ph1to1h2o_r",
    ]
    frame = attrs.copy()
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    rows = []
    for mukey, group in frame.groupby("mukey", sort=True):
        horizons = group.dropna(subset=["hzdept_r", "hzdepb_r"]).copy()
        horizons = horizons[horizons["hzdepb_r"] > horizons["hzdept_r"]]
        if horizons.empty:
            continue
        row = {"mukey": str(mukey), "soilthickness": float(min(200.0, horizons["hzdepb_r"].max()))}
        for idx, depth in enumerate(depths_cm, start=1):
            selected = horizons[(horizons["hzdept_r"] <= depth) & (horizons["hzdepb_r"] > depth)]
            if selected.empty and depth >= horizons["hzdepb_r"].max():
                selected = horizons[horizons["hzdepb_r"] >= depth]
            if selected.empty:
                selected = horizons.iloc[[int((horizons["hzdept_r"] - depth).abs().argmin())]]
            record = selected.iloc[0]
            row[f"bd_sl{idx}"] = float(record["dbthirdbar_r"]) if pd.notna(record["dbthirdbar_r"]) else np.nan
            row[f"oc_sl{idx}"] = float(record["om_r"]) / 1.724 if pd.notna(record["om_r"]) else np.nan
            row[f"ph_sl{idx}"] = float(record["ph1to1h2o_r"]) if pd.notna(record["ph1to1h2o_r"]) else np.nan
            row[f"clyppt_sl{idx}"] = float(record["claytotal_r"]) if pd.notna(record["claytotal_r"]) else np.nan
            row[f"sltppt_sl{idx}"] = float(record["silttotal_r"]) if pd.notna(record["silttotal_r"]) else np.nan
            row[f"sndppt_sl{idx}"] = float(record["sandtotal_r"]) if pd.notna(record["sandtotal_r"]) else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    required = [f"{prefix}_sl{idx}" for prefix in ("bd", "oc", "ph", "clyppt", "sltppt", "sndppt") for idx in range(1, 8)]
    return out.dropna(subset=["soilthickness", *required], how="any").reset_index(drop=True)


def _write_netcdf_atomically(ds, output_path, *, encoding=None):
    output_path = Path(output_path)
    temp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    with suppress(FileNotFoundError):
        temp_path.unlink()
    if encoding:
        ds = ds.copy()
        for variable in encoding:
            if variable in ds:
                ds[variable].attrs.pop("_FillValue", None)
    ds.to_netcdf(temp_path, encoding=encoding)
    temp_path.replace(output_path)


def _hydromt_crs(crs):
    if crs is None:
        return None
    to_epsg = getattr(crs, "to_epsg", None)
    epsg = to_epsg() if to_epsg is not None else None
    if epsg is not None:
        return f"EPSG:{epsg}"
    to_wkt = getattr(crs, "to_wkt", None)
    if to_wkt is not None:
        return to_wkt()
    return str(crs)


def _ensure_ssurgo_wflow_attributes(soil_polygons, soil_attributes):
    soil_polygons = Path(soil_polygons)
    soil_attributes = Path(soil_attributes)
    required = set(ssurgo_attribute_columns())
    if soil_attributes.exists():
        with suppress(Exception):
            if required.issubset(set(pd.read_csv(soil_attributes, nrows=0).columns)):
                return soil_attributes
    if not soil_polygons.exists():
        raise FileNotFoundError(f"SSURGO polygons are required before refreshing Wflow soil attributes: {soil_polygons}")
    soils = gpd.read_file(soil_polygons)
    if "mukey" not in soils:
        raise ValueError("SSURGO polygons must contain a mukey column")
    mukeys = sorted(soils["mukey"].dropna().astype(str).unique())
    fetch_ssurgo_mapunit_attributes(mukeys, soil_attributes)
    return soil_attributes


def _location_path(location_root, value):
    return resolve_location_path(location_root, value)


def _optional_location_path(location_root, value):
    if value in (None, ""):
        return None
    return _location_path(location_root, value)


def _expected_mask_geometry(location_root, collection, wflow_extent):
    return str(
        _optional_location_path(
            location_root,
            collection.get("dem_mask_geometry", wflow_extent.get("watersheds")),
        )
        or ""
    )


def _result(
    status,
    hydrography_nc,
    river_gpkg,
    catchments_gpkg,
    soil_nc,
    manifest,
    *,
    collect_review_vectors=False,
    reservoirs_gpkg=None,
):
    result = {
        "reused": status == "reused",
        "status": status,
        "hydromt_basemap": Path(hydrography_nc),
        "river_geometry": Path(river_gpkg),
        "catchments": Path(catchments_gpkg),
        "wflow_soil_parameters": Path(soil_nc),
        "source_artifact_json": Path(manifest),
        "artifact_count": 4 if collect_review_vectors else 2,
    }
    if reservoirs_gpkg is not None:
        result["reservoirs"] = Path(reservoirs_gpkg)
        result["artifact_count"] += 1
    return result
