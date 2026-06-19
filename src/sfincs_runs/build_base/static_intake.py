from __future__ import annotations

from contextlib import suppress
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import requests
import rioxarray as rxr
from rioxarray.exceptions import NoDataInBounds
from rioxarray.merge import merge_arrays
from shapely.geometry import box

from design_events.collect_sources.ssurgo import (
    fetch_ssurgo_mapunit_attributes,
    fetch_ssurgo_mapunit_polygons,
    normalize_ssurgo_axis_order,
    ssurgo_attribute_columns,
)
from sfincs_runs.build_base.region_setup import RegionSetup, build_region_setup
from sfincs_runs.hydrology import write_ssurgo_infiltration_rasters


def worldcover_tile_urls(bbox_wgs84, *, year=2021, version="v200"):
    west, south, east, north = bbox_wgs84
    west_tile = _worldcover_tile_origin(west)
    east_tile = _worldcover_tile_origin(east)
    south_tile = _worldcover_tile_origin(south)
    north_tile = _worldcover_tile_origin(north)
    urls = []
    for lon in range(west_tile, east_tile + 1, 3):
        for lat in range(south_tile, north_tile + 1, 3):
            tile = _worldcover_tile_name(lon, lat)
            urls.append(
                f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/{version}/{year}/map/"
                f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
            )
    return urls


def fetch_worldcover_landcover(
    bbox_wgs84,
    output_path,
    *,
    tile_cache=None,
    year=2021,
    version="v200",
    target_resolution_degrees=None,
    force=False,
):
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return output_path, "local"
    tile_cache = Path(tile_cache or output_path.parent / "worldcover_tiles")
    tile_paths = []
    for url in worldcover_tile_urls(bbox_wgs84, year=year, version=version):
        tile_path = tile_cache / Path(url).name
        if tile_path.exists():
            tile_paths.append(tile_path)
        elif target_resolution_degrees is not None:
            tile_paths.append(url)
        else:
            download_file(url, tile_path)
            tile_paths.append(tile_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if target_resolution_degrees is not None:
        from rasterio.enums import Resampling

        resampling = Resampling.nearest
    arrays = []
    clipped_arrays = []
    merged = None
    try:
        for tile_path in tile_paths:
            array = rxr.open_rasterio(tile_path, masked=True).squeeze(drop=True)
            arrays.append(array)
            try:
                clipped = array.rio.clip_box(*bbox_wgs84, crs="EPSG:4326")
            except NoDataInBounds:
                continue
            if target_resolution_degrees is not None:
                try:
                    clipped_arrays.append(
                        clipped.rio.reproject(
                            "EPSG:4326",
                            resolution=float(target_resolution_degrees),
                            resampling=resampling,
                        )
                    )
                finally:
                    _close_arrays([clipped])
            else:
                clipped_arrays.append(clipped)
        if not clipped_arrays:
            raise RuntimeError(f"No WorldCover pixels intersect bbox {bbox_wgs84}")
        merged = clipped_arrays[0] if len(clipped_arrays) == 1 else merge_arrays(clipped_arrays)
        _write_raster_atomically(merged, output_path)
    finally:
        _close_arrays([merged, *clipped_arrays, *arrays])
    return output_path, "downloaded"


def fetch_usgs_3dep_dem(
    bbox_wgs84,
    output_path,
    *,
    gsd=10,
    request_timeout=60,
    target_resolution_degrees=None,
    force=False,
):
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return {"dem_raw": output_path, "tiles_merged": 0, "source": "existing"}

    import planetary_computer
    import pystac_client
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
        timeout=request_timeout,
    )
    search = catalog.search(collections=["3dep-seamless"], bbox=list(bbox_wgs84))
    items = [item for item in search.items() if item.properties.get("gsd") == gsd]
    if not items:
        raise RuntimeError(f"No 3DEP items returned for bbox {bbox_wgs84} and gsd={gsd}")

    if target_resolution_degrees is not None:
        from rasterio.enums import Resampling

        resampling = Resampling.bilinear
    arrays = []
    clipped_arrays = []
    merged = None
    try:
        for item in items:
            array = rxr.open_rasterio(item.assets["data"].href, masked=True).squeeze(drop=True)
            arrays.append(array)
            try:
                clipped = array.rio.clip_box(*bbox_wgs84, crs="EPSG:4326")
            except NoDataInBounds:
                continue
            if target_resolution_degrees is not None:
                try:
                    clipped_arrays.append(
                        clipped.rio.reproject(
                            "EPSG:4269",
                            resolution=float(target_resolution_degrees),
                            resampling=resampling,
                        )
                    )
                finally:
                    _close_arrays([clipped])
            else:
                clipped_arrays.append(clipped)
        if not clipped_arrays:
            raise RuntimeError(f"No 3DEP pixels intersect bbox {bbox_wgs84} and gsd={gsd}")
        merged = clipped_arrays[0] if len(clipped_arrays) == 1 else merge_arrays(clipped_arrays)
        _write_raster_atomically(merged, output_path)
    finally:
        _close_arrays([merged, *clipped_arrays, *arrays])
    return {"dem_raw": output_path, "tiles_merged": len(items), "source": "usgs_3dep"}


def clip_dem_and_landcover_to_bbox(
    dem_raw,
    landcover_raw,
    dem_output,
    landcover_output,
    bbox_gdf,
    *,
    model_crs,
    reference_crs="EPSG:4326",
    target_resolution_degrees=None,
):
    dem_raw = Path(dem_raw)
    landcover_raw = Path(landcover_raw)
    if not dem_raw.exists():
        raise FileNotFoundError(dem_raw)
    if not landcover_raw.exists():
        raise FileNotFoundError(landcover_raw)
    dem_output = Path(dem_output)
    landcover_output = Path(landcover_output)
    dem_output.parent.mkdir(parents=True, exist_ok=True)
    landcover_output.parent.mkdir(parents=True, exist_ok=True)
    dem = None
    landcover = None
    dem_clip = None
    landcover_clip = None
    try:
        dem = rxr.open_rasterio(dem_raw, masked=True).squeeze(drop=True)
        if dem.rio.crs is None:
            dem = dem.rio.write_crs(model_crs)
        landcover = rxr.open_rasterio(landcover_raw, masked=True).squeeze(drop=True)
        if landcover.rio.crs is None:
            landcover = landcover.rio.write_crs(reference_crs)
        bbox_model = bbox_gdf.to_crs(dem.rio.crs)
        bbox_landcover = bbox_gdf.to_crs(landcover.rio.crs)
        dem_clip = dem.rio.clip_box(*bbox_model.total_bounds)
        if target_resolution_degrees is not None:
            target_crs = dem_clip.rio.crs if dem_clip.rio.crs.is_geographic else "EPSG:4269"
            dem_clip = dem_clip.rio.reproject(target_crs, resolution=float(target_resolution_degrees))
        # Clip landcover to the AOI's bounding rectangle (matching the DEM), then align it
        # to the DEM grid. Do NOT clip to the AOI polygon geometry: when the AOI is several
        # disjoint sub-region boxes, a geometry clip punches nodata holes in the corners
        # they leave uncovered, and any SFINCS domain spanning a gap loses its landcover and
        # falls back to a flat manning value.
        landcover_clip = landcover.rio.clip_box(*bbox_landcover.total_bounds)
        landcover_clip = landcover_clip.rio.reproject_match(dem_clip)
        _write_raster_atomically(dem_clip, dem_output)
        _write_raster_atomically(landcover_clip, landcover_output)
        return {
            "dem": dem_output,
            "landcover": landcover_output,
            "dem_pixels": int(dem_clip.size),
            "landcover_pixels": int(landcover_clip.size),
        }
    finally:
        _close_arrays([landcover_clip, dem_clip, landcover, dem])


def collect_static_region_inputs(
    config,
    paths,
    *,
    fetch_dem=False,
    fetch_landcover=True,
    fetch_ssurgo=True,
    soil_polygons=None,
    soil_attributes=None,
):
    region_setup = build_region_setup(config, paths, buffer_degrees=0.01)
    bbox_gdf = _ensure_bbox(config, paths, region_setup)
    bbox_wgs84 = tuple(float(value) for value in bbox_gdf.to_crs("EPSG:4326").total_bounds)
    if fetch_dem and not region_setup.dem_raw.exists():
        fetch_usgs_3dep_dem(bbox_wgs84, region_setup.dem_raw)
    if fetch_landcover and not region_setup.landcover_raw.exists():
        fetch_worldcover_landcover(bbox_wgs84, region_setup.landcover_raw)
    clip_summary = clip_dem_and_landcover_to_bbox(
        region_setup.dem_raw,
        region_setup.landcover_raw,
        region_setup.dem_output,
        region_setup.landcover_output,
        bbox_gdf,
        model_crs=config["project"]["model_crs"],
        reference_crs=config["project"].get("reference_crs", "EPSG:4326"),
    )
    ssurgo_summary = {}
    if fetch_ssurgo:
        ssurgo_summary = collect_ssurgo_infiltration_inputs(
            region_setup,
            bbox_wgs84,
            soil_polygons=soil_polygons,
            soil_attributes=soil_attributes,
        )
    return {
        **clip_summary,
        **ssurgo_summary,
        "bbox": region_setup.bbox_output,
        "dem_exists": region_setup.dem_output.exists(),
        "landcover_exists": region_setup.landcover_output.exists(),
    }


def collect_wflow_static_region_inputs(
    config,
    paths,
    *,
    fetch_dem=False,
    fetch_landcover=True,
    fetch_ssurgo=True,
    soil_polygons=None,
    soil_attributes=None,
):
    """Collect coarse Wflow static inputs over the reviewed Wflow collection envelope."""
    location_root = Path(paths["location_root"])
    extent = config.get("static_sources", {}).get("wflow_collection_extent", {})
    boundary = _location_path(location_root, extent.get("boundary", "data/static/aoi/wflow_collection_region.geojson"))
    if not boundary.exists():
        raise FileNotFoundError(f"Wflow collection boundary is required before Wflow static collection: {boundary}")
    bbox_gdf = gpd.read_file(boundary).to_crs("EPSG:4326")
    bbox_wgs84 = tuple(float(value) for value in bbox_gdf.total_bounds)

    setup = _wflow_region_setup(config, paths, boundary, bbox_wgs84)
    target_resolution = extent.get("terrain_resolution_degrees")
    dem_input = setup.dem_raw
    dem_covers_extent = _raster_covers_bounds(dem_input, bbox_wgs84)
    if not dem_covers_extent and _raster_covers_bounds(setup.dem_output, bbox_wgs84):
        dem_input = setup.dem_output
        dem_covers_extent = True
    if fetch_dem and not dem_covers_extent:
        fetch_usgs_3dep_dem(
            bbox_wgs84,
            setup.dem_raw,
            gsd=int(extent.get("terrain_fetch_gsd", 30)),
            target_resolution_degrees=target_resolution,
            force=setup.dem_raw.exists(),
        )
        dem_input = setup.dem_raw
    elif not dem_covers_extent:
        raise RuntimeError(
            f"Wflow raw DEM does not cover Wflow collection boundary {bbox_wgs84}: {setup.dem_raw}. "
            "Set FLOOD_RM_FETCH_DEM=1 to collect it over the larger Wflow region."
        )
    landcover_input = setup.landcover_raw
    landcover_covers_extent = _raster_covers_bounds(landcover_input, bbox_wgs84)
    if not landcover_covers_extent and _raster_covers_bounds(setup.landcover_output, bbox_wgs84):
        landcover_input = setup.landcover_output
        landcover_covers_extent = True
    if fetch_landcover and not landcover_covers_extent:
        fetch_worldcover_landcover(
            bbox_wgs84,
            setup.landcover_raw,
            target_resolution_degrees=extent.get("landcover_resolution_degrees", target_resolution),
            force=setup.landcover_raw.exists(),
        )
        landcover_input = setup.landcover_raw
    elif not landcover_covers_extent:
        raise RuntimeError(
            f"Wflow raw landcover does not cover Wflow collection boundary {bbox_wgs84}: {setup.landcover_raw}. "
            "Set FLOOD_RM_FETCH_LANDCOVER=1 to collect it over the larger Wflow region."
        )
    clip_summary = clip_dem_and_landcover_to_bbox(
        dem_input,
        landcover_input,
        setup.dem_output,
        setup.landcover_output,
        bbox_gdf,
        model_crs=config["project"]["model_crs"],
        reference_crs=config["project"].get("reference_crs", "EPSG:4326"),
        target_resolution_degrees=target_resolution,
    )
    if not _raster_covers_bounds(setup.dem_output, bbox_wgs84):
        raise RuntimeError(f"Wflow processed DEM does not cover Wflow collection boundary {bbox_wgs84}: {setup.dem_output}")
    if not _raster_covers_bounds(setup.landcover_output, bbox_wgs84):
        raise RuntimeError(
            f"Wflow processed landcover does not cover Wflow collection boundary {bbox_wgs84}: {setup.landcover_output}"
        )
    ssurgo_summary = {}
    if fetch_ssurgo:
        ssurgo_summary = collect_ssurgo_infiltration_inputs(
            setup,
            bbox_wgs84,
            soil_polygons=soil_polygons,
            soil_attributes=soil_attributes,
        )
    return {
        **clip_summary,
        **ssurgo_summary,
        "bbox": boundary,
        "dem_exists": setup.dem_output.exists(),
        "landcover_exists": setup.landcover_output.exists(),
    }


def collect_ssurgo_infiltration_inputs(region_setup, bbox_wgs84, *, soil_polygons=None, soil_attributes=None):
    if soil_polygons is not None:
        soils = soil_polygons.copy()
        region_setup.ssurgo_output.parent.mkdir(parents=True, exist_ok=True)
        soils.to_file(region_setup.ssurgo_output, driver="GPKG")
    elif region_setup.ssurgo_output.exists():
        soils = gpd.read_file(region_setup.ssurgo_output)
        if not _geodataframe_bounds_cover(soils, bbox_wgs84):
            soils = fetch_ssurgo_mapunit_polygons(bbox_wgs84, region_setup.ssurgo_output)
    else:
        soils = fetch_ssurgo_mapunit_polygons(bbox_wgs84, region_setup.ssurgo_output)
    if soils.crs is None:
        soils = soils.set_crs("EPSG:4326")
    soils_wgs = soils.to_crs("EPSG:4326")
    soils_wgs = normalize_ssurgo_axis_order(soils_wgs, bbox_wgs84)
    if not _geodataframe_bounds_cover(soils_wgs, bbox_wgs84):
        raise RuntimeError(f"SSURGO polygons do not cover collection boundary {bbox_wgs84}: {region_setup.ssurgo_output}")
    soil_mukeys = sorted(soils_wgs["mukey"].dropna().astype(str).unique()) if "mukey" in soils_wgs else []
    if soil_attributes is not None:
        attrs = soil_attributes.copy()
        region_setup.ssurgo_attributes_output.parent.mkdir(parents=True, exist_ok=True)
        attrs.to_csv(region_setup.ssurgo_attributes_output, index=False)
    elif region_setup.ssurgo_attributes_output.exists() and _ssurgo_attributes_cover_wflow_soils(
        region_setup.ssurgo_attributes_output
    ):
        attrs = pd.read_csv(region_setup.ssurgo_attributes_output)
    else:
        attrs = fetch_ssurgo_mapunit_attributes(soil_mukeys, region_setup.ssurgo_attributes_output)
    infiltration = write_ssurgo_infiltration_rasters(
        soils,
        attrs,
        region_setup.landcover_output,
        hsg_out=region_setup.ssurgo_hsg_output,
        ksat_out=region_setup.ssurgo_ksat_output,
        drainage_condition="undrained",
        ksat_units="um/s",
    )
    return {
        "ssurgo_polygons": int(len(soils)),
        "ssurgo_attribute_rows": int(len(attrs)),
        "soil_mukeys": int(len(soil_mukeys)),
        "hsg": Path(infiltration["hsg"]),
        "ksat": Path(infiltration["ksat"]),
    }


def _ensure_bbox(config, paths, region_setup):
    if region_setup.bbox_output.exists():
        return gpd.read_file(region_setup.bbox_output).to_crs("EPSG:4326")
    from shapely.geometry import box
    from study_location import study_area_bbox

    bbox_geom = box(*study_area_bbox(config, paths["repo_root"], buffer_degrees=0.01))
    bbox_gdf = gpd.GeoDataFrame({"name": ["sfincs_bbox"]}, geometry=[bbox_geom], crs="EPSG:4326")
    region_setup.bbox_output.parent.mkdir(parents=True, exist_ok=True)
    bbox_gdf.to_file(region_setup.bbox_output, driver="GeoJSON")
    return bbox_gdf


def _wflow_static_config(config):
    cfg = {**config, "static_sources": {**config.get("static_sources", {})}}
    sources = cfg["static_sources"]
    extent = sources.get("wflow_collection_extent", {})
    sources["bbox"] = {"output": extent.get("boundary", "data/static/aoi/wflow_collection_region.geojson")}
    sources["terrain"] = {
        "raw": extent.get("terrain_raw", "data/wflow/static/raw/topo/dem_wflow.tif"),
        "output": extent.get("terrain_output", "data/wflow/static/processed/dem_wflow_coarse.tif"),
    }
    sources["landcover"] = {
        "raw": extent.get("landcover_raw", "data/wflow/static/raw/landcover/landcover_wflow.tif"),
        "output": extent.get("landcover_output", "data/wflow/static/processed/landcover_wflow_coarse.tif"),
    }
    sources["ssurgo"] = {
        "output": extent.get("ssurgo_output", "data/wflow/static/soils/ssurgo_mapunitpoly_wflow.gpkg"),
        "attributes_output": extent.get("ssurgo_attributes_output", "data/wflow/static/soils/ssurgo_mapunit_attributes_wflow.csv"),
        "hsg_output": extent.get("hsg_output", "data/wflow/static/soils/hsg_wflow.tif"),
        "ksat_output": extent.get("ksat_output", "data/wflow/static/soils/ksat_mmhr_wflow.tif"),
    }
    return cfg


def _wflow_region_setup(config, paths, boundary: Path, bbox_wgs84) -> RegionSetup:
    cfg = _wflow_static_config(config)
    sources = cfg["static_sources"]
    location_root = Path(paths["location_root"])
    return RegionSetup(
        study_location=str(paths.get("location_name") or config.get("project", {}).get("name", "")),
        bbox_wgs84=tuple(float(value) for value in bbox_wgs84),
        study_area_path=boundary,
        dem_raw=_location_path(location_root, sources["terrain"]["raw"]),
        landcover_raw=_location_path(location_root, sources["landcover"]["raw"]),
        dem_output=_location_path(location_root, sources["terrain"]["output"]),
        landcover_output=_location_path(location_root, sources["landcover"]["output"]),
        bbox_output=boundary,
        coastal_region_output=_location_path(location_root, "data/static/processed/coastal_region.geojson"),
        ssurgo_output=_location_path(location_root, sources["ssurgo"]["output"]),
        ssurgo_attributes_output=_location_path(location_root, sources["ssurgo"]["attributes_output"]),
        ssurgo_hsg_output=_location_path(location_root, sources["ssurgo"]["hsg_output"]),
        ssurgo_ksat_output=_location_path(location_root, sources["ssurgo"]["ksat_output"]),
    )


def _raster_covers_bounds(path, bbox_wgs84, *, tolerance=1e-8) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                return False
            target_bounds = gpd.GeoSeries([box(*bbox_wgs84)], crs="EPSG:4326").to_crs(src.crs).total_bounds
            raster_bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
    except Exception:
        return False
    return _bounds_cover(raster_bounds, target_bounds, tolerance=tolerance)


def _geodataframe_bounds_cover(frame: gpd.GeoDataFrame, bbox_wgs84, *, tolerance=1e-8) -> bool:
    if frame.empty or frame.crs is None:
        return False
    frame_bounds = frame.to_crs("EPSG:4326").total_bounds
    return _bounds_cover(frame_bounds, bbox_wgs84, tolerance=tolerance)


def _bounds_cover(outer_bounds, inner_bounds, *, tolerance=1e-8) -> bool:
    outer_west, outer_south, outer_east, outer_north = (float(value) for value in outer_bounds)
    inner_west, inner_south, inner_east, inner_north = (float(value) for value in inner_bounds)
    return (
        outer_west <= inner_west + tolerance
        and outer_south <= inner_south + tolerance
        and outer_east >= inner_east - tolerance
        and outer_north >= inner_north - tolerance
    )


def _location_path(location_root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(location_root) / path


def _ssurgo_attributes_cover_wflow_soils(path):
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    return set(ssurgo_attribute_columns()).issubset(columns)


def _worldcover_tile_origin(value, tile_size=3):
    return int(math.floor(float(value) / tile_size) * tile_size)


def _worldcover_tile_name(lon, lat):
    west = _worldcover_tile_origin(lon)
    south = _worldcover_tile_origin(lat)
    ns = "N" if south >= 0 else "S"
    ew = "E" if west >= 0 else "W"
    return f"{ns}{abs(south):02d}{ew}{abs(west):03d}"


def download_file(url, output_path, *, timeout_seconds=120):
    return _download_file(url, output_path, timeout_seconds=timeout_seconds)


def _download_file(url, output_path, *, timeout_seconds=120):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=timeout_seconds)
    response.raise_for_status()
    with output_path.open("wb") as stream:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                stream.write(chunk)
    return output_path


def _write_raster_atomically(data_array, output_path):
    output_path = Path(output_path)
    temp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    with suppress(FileNotFoundError):
        temp_path.unlink()
    copy = getattr(data_array, "copy", None)
    raster = copy(deep=False) if copy is not None else data_array
    encoding = getattr(raster, "encoding", {})
    attrs = getattr(raster, "attrs", {})
    for key in ("_FillValue", "missing_value", "scale_factor", "add_offset"):
        if key in encoding:
            attrs.pop(key, None)
    raster.rio.to_raster(temp_path)
    temp_path.replace(output_path)


def _close_arrays(arrays):
    seen = set()
    for array in arrays:
        if array is None:
            continue
        marker = id(array)
        if marker in seen:
            continue
        seen.add(marker)
        close = getattr(array, "close", None)
        if close is not None:
            with suppress(Exception):
                close()
