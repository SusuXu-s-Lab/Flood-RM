from __future__ import annotations

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

from design_events.collect_sources.ssurgo import (
    fetch_ssurgo_mapunit_attributes,
    ssurgo_attribute_columns,
)

NHDPLUS_HR_MAPSERVER = "https://hydro.nationalmap.gov/arcgis/rest/services/NHDPlus_HR/MapServer"
NHDPLUS_HR_NETWORK_FLOWLINE_LAYER = 3
NHDPLUS_HR_CATCHMENT_LAYER = 10

# USGS Watershed Boundary Dataset (WBD) ArcGIS service: HUC polygons by level.
WBD_MAPSERVER = "https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer"
WBD_HUC_LAYER_BY_LEVEL = {2: 1, 4: 2, 6: 3, 8: 4, 10: 5, 12: 6}


def collect_national_hydrography(settings, *, skip_existing=True, smoke=False):
    """Prepare USA-first HydroMT-Wflow hydrography and soil source artifacts."""
    config = settings["config"]
    paths = settings["paths"]
    location_root = Path(paths["location_root"])
    collection = config.get("collection", {}).get("national_hydrography", {})
    static_sources = config.get("static_sources", {})

    hydrography_nc = _location_path(location_root, collection.get("hydromt_basemap", "data/wflow/hydrography/us_hydrography_basemap.nc"))
    river_gpkg = _location_path(location_root, collection.get("river_geometry", "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"))
    catchments_gpkg = _location_path(location_root, collection.get("catchments", "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg"))
    soil_nc = _location_path(location_root, collection.get("wflow_soil_parameters", "data/wflow/static/ssurgo_wflow_soil_parameters.nc"))
    manifest = Path(paths.get("source_artifacts_root", location_root / "data/sources/source_artifacts")) / "national_hydrography_wflow_sources.json"
    target_resolution_degrees = float(collection.get("basemap_source_resolution_degrees", 1 / 1080))
    collect_review_vectors = bool(collection.get("collect_review_vectors", False))

    required_outputs = [hydrography_nc, soil_nc]
    if collect_review_vectors:
        required_outputs.extend([river_gpkg, catchments_gpkg])
    if (
        skip_existing
        and all(path.exists() for path in required_outputs)
        and _hydromt_basemap_resolution_matches(hydrography_nc, target_resolution_degrees)
    ):
        return _result(
            "reused",
            hydrography_nc,
            river_gpkg,
            catchments_gpkg,
            soil_nc,
            manifest,
            collect_review_vectors=collect_review_vectors,
        )

    wflow_extent = static_sources.get("wflow_collection_extent", {})
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
    )
    if collect_review_vectors:
        river_summary = write_review_hydrography_vectors(
            hydrography_nc,
            river_gpkg,
            catchments_gpkg,
            service_url=str(collection.get("nhdplus_hr_service_url", NHDPLUS_HR_MAPSERVER)),
            timeout_seconds=float(collection.get("request_timeout_seconds", 120)),
        )
    else:
        river_summary = {
            "review_vector_status": "skipped",
            "review_vector_note": "NHDPlus/3DHP vectors are optional QA artifacts and are not Wflow build inputs.",
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
                    "soil_method": "ssurgo_horizon_pedology",
                    "smoke": bool(smoke),
                    **hydrography_summary,
                    **river_summary,
                    **soil_summary,
                },
                "artifacts": {
                    "hydromt_basemap": str(hydrography_nc),
                    "river_geometry": str(river_gpkg),
                    "catchments": str(catchments_gpkg),
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
    )


def write_dem_derived_hydromt_basemap(
    dem_path,
    output_path,
    *,
    target_resolution_degrees=1 / 1080,
    min_stream_uparea_km2=5.0,
):
    """Write HydroMT-Wflow RasterDataset variables from a local USGS 3DEP DEM."""
    import hydromt  # noqa: F401
    from hydromt.gis import flw

    dem_path = Path(dem_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dem = None
    try:
        dem = rxr.open_rasterio(dem_path, masked=True).squeeze(drop=True)
        if dem.rio.crs is None:
            dem = dem.rio.write_crs("EPSG:4269")
        if not dem.rio.crs.is_geographic:
            dem = dem.rio.reproject("EPSG:4269", resolution=target_resolution_degrees)
        factor = max(1, int(round(float(target_resolution_degrees) / abs(float(dem.rio.resolution()[0])))))
        if factor > 1:
            dem = dem.coarsen(y=factor, x=factor, boundary="trim").mean()
            dem = dem.rio.write_crs(dem.rio.crs)
        dem = dem.astype("float32")
        nodata = dem.rio.nodata
        values = dem.values
        if nodata is not None and np.isfinite(nodata):
            values = np.where(values == nodata, np.nan, values)
        finite = np.isfinite(values)
        if not np.any(finite):
            raise ValueError(f"DEM has no finite elevation values: {dem_path}")
        fill_value = float(np.nanmedian(values))
        elevtn = np.where(finite, values, fill_value).astype("float32")
        dem = dem.copy(data=elevtn)
        dem = dem.rio.write_nodata(np.float32(-9999), encoded=False)
        da_flwdir = flw.d8_from_dem(dem, max_depth=-1.0, outlets="edge")
        flwdir = flw.flwdir_from_da(da_flwdir, ftype="infer", check_ftype=True)
        flwdir_arr = da_flwdir.values.astype("uint8")
        uparea = flwdir.upstream_area(unit="km2").astype("float32")
        strord = flwdir.stream_order().astype("int16")
        basins = flwdir.basins(idxs=flwdir.idxs_pit).astype("int32")
        stream_mask = (uparea >= float(min_stream_uparea_km2)).astype("uint8")

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
        }
    finally:
        if dem is not None:
            with suppress(Exception):
                dem.close()


def _hydromt_basemap_resolution_matches(hydrography_nc: Path, target_resolution_degrees: float) -> bool:
    ds = None
    try:
        ds = xr.open_dataset(hydrography_nc)
        for coord in ("x", "longitude", "lon"):
            if coord in ds.coords and ds[coord].size > 1:
                values = ds[coord].values
                resolution = float(abs(values[1] - values[0]))
                return abs(resolution - float(target_resolution_degrees)) <= max(
                    1.0e-9,
                    float(target_resolution_degrees) * 0.01,
                )
        return False
    except Exception:
        return False
    finally:
        if ds is not None:
            ds.close()


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
        rivers = _prepare_nhdplus_river_geometry(rivers)
        rivers.to_file(river_output, driver="GPKG")
        catchments = _prepare_nhdplus_catchments(catchments)
        catchments.to_file(catchment_output, driver="GPKG")
        return {
            "river_features": int(len(rivers)),
            "catchment_features": int(len(catchments)),
            "river_source": "USGS NHDPlus HR NetworkNHDFlowline",
            "catchment_source": "USGS NHDPlus HR NHDPlusCatchment",
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


def _prepare_nhdplus_river_geometry(rivers):
    rivers = rivers.to_crs("EPSG:4326").copy()
    if "rivwth" not in rivers:
        rivers["rivwth"] = _estimate_river_width(rivers)
    if "qbankfull" not in rivers:
        rivers["qbankfull"] = _estimate_bankfull_discharge(rivers)
    rivers["review_status"] = "review_required_usgs_nhdplus_hr"
    rivers["source"] = "USGS NHDPlus HR NetworkNHDFlowline"
    return rivers


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
    path = Path(value)
    return path if path.is_absolute() else Path(location_root) / path


def _result(status, hydrography_nc, river_gpkg, catchments_gpkg, soil_nc, manifest, *, collect_review_vectors=False):
    return {
        "reused": status == "reused",
        "status": status,
        "hydromt_basemap": Path(hydrography_nc),
        "river_geometry": Path(river_gpkg),
        "catchments": Path(catchments_gpkg),
        "wflow_soil_parameters": Path(soil_nc),
        "source_artifact_json": Path(manifest),
        "artifact_count": 4 if collect_review_vectors else 2,
    }
