import json
from pathlib import Path

import geopandas as gpd
import hydromt  # noqa: F401
import numpy as np
import pandas as pd
import pytest
import rasterio
import pyflwdir
from rasterio.transform import from_origin
from shapely.geometry import box
import xarray as xr

from design_events.collect_sources.national_hydrography import (
    NHDPLUS_HR_CATCHMENT_LAYER,
    NHDPLUS_HR_WATERBODY_LAYER,
    WBD_HUC_LAYER_BY_LEVEL,
    collect_national_hydrography,
    enrich_river_geometry_with_stream_geo_attribute_transfer,
    enrich_river_geometry_with_stream_geo,
    fetch_nldi_comid,
    fetch_nhdplus_hr_catchments,
    fetch_nhdplus_hr_layer,
    fetch_wbd_huc,
    prepare_nhdplus_hr_waterbodies_for_wflow,
    refresh_wflow_hydrography_basemap,
    refresh_wflow_river_geometry_sources,
    _stream_geo_comid_keys_from_flowlines,
    write_dem_derived_hydromt_basemap,
    write_review_hydrography_vectors,
    write_wflow_reservoir_waterbodies,
    write_ssurgo_wflow_soil_dataset,
)
from design_events.collect_sources.stream_geo_nldi import (
    collect_stream_geo_nldi,
    select_stream_geo_file,
)


def test_fetch_wbd_huc_queries_level_layer_and_normalizes_huc_id():
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"huc6": "030300", "name": "Cape Fear"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-80.0, 36.0], [-79.0, 36.0], [-79.0, 37.0], [-80.0, 37.0], [-80.0, 36.0]]],
                        },
                    }
                ],
            }

    def fake_get(url, params, timeout):
        calls.append(url)
        return Response()

    hucs = fetch_wbd_huc(
        (-80.0, 36.0, -79.0, 37.0),
        huc_level=6,
        service_url="https://example.test/MapServer",
        session_get=fake_get,
    )

    assert calls[0] == f"https://example.test/MapServer/{WBD_HUC_LAYER_BY_LEVEL[6]}/query"
    assert hucs["huc_id"].iloc[0] == "030300"
    assert int(hucs["huc_level"].iloc[0]) == 6


def test_fetch_nldi_comid_uses_position_query():
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"features": [{"properties": {"identifier": "12345"}}]}

    def fake_get(url, params, timeout):
        calls.append((url, params))
        return Response()

    comid = fetch_nldi_comid(-89.509, 43.087, base_url="https://example.test/nldi", session_get=fake_get)

    assert comid == "12345"
    assert calls[0][0] == "https://example.test/nldi/comid/position"
    assert calls[0][1]["coords"] == "POINT(-89.509 43.087)"


def test_enrich_river_geometry_with_stream_geo_adds_native_width_depth_fields():
    rivers = gpd.GeoDataFrame(
        {"COMID": ["1", "2"]},
        geometry=[box(0, 0, 1, 1).boundary, box(1, 1, 2, 2).boundary],
        crs="EPSG:4326",
    )
    stream_geo = pd.DataFrame(
        {
            "COMID": ["1"],
            "width_m": [42.0],
            "depth_m": [2.5],
            "qbankfull": [100.0],
        }
    )

    enriched = enrich_river_geometry_with_stream_geo(rivers, stream_geo)

    assert enriched.loc[0, "rivwth"] == pytest.approx(42.0)
    assert enriched.loc[0, "rivdph"] == pytest.approx(2.5)
    assert enriched.loc[0, "qbankfull"] == pytest.approx(100.0)
    assert enriched.loc[0, "rivwth_source"] == "STREAM-geo"


def test_enrich_river_geometry_with_stream_geo_caches_nldi_bridge_incrementally(tmp_path):
    rivers = gpd.GeoDataFrame(
        {"nhdplusid": ["hr-1", "hr-2"]},
        geometry=[box(0, 0, 1, 1).boundary, box(2, 0, 3, 1).boundary],
        crs="EPSG:4326",
    )
    stream_geo = pd.DataFrame(
        {
            "COMID": ["101", "202"],
            "XGB_Width_m": [12.0, 24.0],
            "XGB_Depth_m": [1.2, 2.4],
        }
    )
    cache = tmp_path / "nldi_stream_geo_comid_cache.csv"

    class Response:
        def __init__(self, comid):
            self.comid = comid

        def raise_for_status(self):
            return None

        def json(self):
            return {"features": [{"properties": {"identifier": self.comid}}]}

    calls = []

    def fake_get(url, params, timeout):
        calls.append(params["coords"])
        x = float(params["coords"].split("(")[1].split()[0])
        return Response("101" if x < 2 else "202")

    enriched = enrich_river_geometry_with_stream_geo(
        rivers,
        stream_geo,
        nldi_lookup_cache=cache,
        use_nldi_lookup=True,
        nldi_max_workers=1,
        nldi_progress_interval=1,
        session_get=fake_get,
    )

    assert enriched["rivwth"].tolist() == [12.0, 24.0]
    assert enriched["rivdph"].tolist() == [1.2, 2.4]
    assert enriched["rivwth_source"].tolist() == ["STREAM-geo/NLDI", "STREAM-geo/NLDI"]
    cached = pd.read_csv(cache, dtype=str)
    assert cached["river_id"].tolist() == ["hr-1", "hr-2"]
    assert cached["nldi_comid"].tolist() == ["101", "202"]
    assert len(calls) == 2

    enrich_river_geometry_with_stream_geo(
        rivers,
        stream_geo,
        nldi_lookup_cache=cache,
        use_nldi_lookup=True,
        nldi_max_workers=1,
        nldi_progress_interval=1,
        session_get=fake_get,
    )

    assert len(calls) == 2


def test_stream_geo_comid_keys_from_flowlines_uses_local_nearest_join():
    rivers = gpd.GeoDataFrame(
        {"nhdplusid": ["hr-1", "hr-2"]},
        geometry=[
            box(-80.001, 35.999, -80.000, 36.000).boundary,
            box(-79.901, 36.099, -79.900, 36.100).boundary,
        ],
        crs="EPSG:4326",
    )
    flowlines = gpd.GeoDataFrame(
        {"comid": ["101", "202"]},
        geometry=[
            box(-80.002, 35.998, -79.999, 36.001).boundary,
            box(-79.902, 36.098, -79.899, 36.101).boundary,
        ],
        crs="EPSG:4326",
    )

    keys = _stream_geo_comid_keys_from_flowlines(
        rivers,
        flowlines,
        river_id_column="nhdplusid",
        max_distance_m=1000,
    )

    assert keys["_river_id_key"].tolist() == ["hr-1", "hr-2"]
    assert keys["_stream_geo_comid_key"].tolist() == ["101", "202"]
    assert keys["stream_geo_join_distance_m"].max() < 1000


def test_stream_geo_attribute_transfer_adds_nonconstant_width_depth():
    rivers = gpd.GeoDataFrame(
        {
            "streamorde": [1, 4],
            "totdasqkm": [1.0, 1000.0],
            "lengthkm": [0.5, 10.0],
        },
        geometry=[box(0, 0, 1, 1).boundary, box(1, 1, 2, 2).boundary],
        crs="EPSG:4326",
    )
    stream_geo = pd.DataFrame(
        {
            "StreamOrde": [1, 1, 4, 4],
            "TotDASqKM": [1.0, 2.0, 900.0, 1100.0],
            "LENGTHKM": [0.4, 0.8, 9.0, 11.0],
            "XGB_Width_m": [3.0, 4.0, 90.0, 110.0],
            "XGB_Depth_m": [0.3, 0.4, 1.8, 2.0],
        }
    )

    enriched = enrich_river_geometry_with_stream_geo_attribute_transfer(rivers, stream_geo, neighbors=1)

    assert enriched["rivwth"].tolist() == pytest.approx([3.0, 110.0])
    assert enriched["rivdph"].tolist() == pytest.approx([0.3, 2.0])
    assert enriched["rivwth"].nunique() == 2
    assert enriched["rivdph_source"].tolist() == ["STREAM-geo_attribute_transfer", "STREAM-geo_attribute_transfer"]


def test_prepare_nhdplus_hr_waterbodies_writes_native_wflow_reservoir_fields():
    waterbodies = gpd.GeoDataFrame(
        {
            "nhdplusid": [3001, 3002, 3003],
            "gnis_name": ["Lake Travis", "Small Pond", "Flooded Area"],
            "areasqkm": [72.854, 0.2, 2.0],
            "elevation": [207.6, 100.0, 120.0],
            "ftype": [390, 390, 436],
            "fcode": [39009, 39004, 43624],
            "purpcode": ["WB", "SC", "UF"],
        },
        geometry=[
            box(-98.0, 30.4, -97.9, 30.5),
            box(-98.1, 30.1, -98.0, 30.2),
            box(-97.9, 30.2, -97.8, 30.3),
        ],
        crs="EPSG:4326",
    )

    prepared = prepare_nhdplus_hr_waterbodies_for_wflow(
        waterbodies,
        box(-98.2, 30.0, -97.7, 30.6),
        min_area_km2=1.0,
        default_depth_m=5.0,
        default_discharge_m3s=0.25,
    )

    assert len(prepared) == 1
    row = prepared.iloc[0]
    assert row["waterbody_id"] == 1
    assert row["source_nhdplusid"] == "3001"
    assert row["Area_avg"] == pytest.approx(72.854 * 1_000_000)
    assert row["Depth_avg"] == pytest.approx(5.0)
    assert row["Vol_avg"] == pytest.approx(72.854 * 1_000_000 * 5.0)
    assert row["Dis_avg"] == pytest.approx(0.25)
    assert row["reservoir_operation"] == "no_control"


def test_write_wflow_reservoir_waterbodies_uses_nhdplus_waterbody_layer(tmp_path):
    location_root = tmp_path / "locations/austin"
    search = location_root / "data/static/aoi/wflow_nhdplus_watersheds.geojson"
    search.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        [{"geometry": box(-98.2, 30.0, -97.7, 30.6)}],
        crs="EPSG:4326",
    ).to_file(search, driver="GeoJSON")
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "nhdplusid": 3001,
                            "gnis_name": "Lake Travis",
                            "areasqkm": 72.854,
                            "elevation": 207.6,
                            "ftype": 390,
                            "fcode": 39009,
                            "purpcode": "WB",
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-98.0, 30.4], [-97.9, 30.4], [-97.9, 30.5], [-98.0, 30.5], [-98.0, 30.4]]],
                        },
                    }
                ],
            }

    def fake_get(url, params, timeout):
        calls.append((url, params))
        return Response()

    output = location_root / "data/sources/national_hydrography/nhdplus_hr_wflow_reservoirs.gpkg"
    summary = write_wflow_reservoir_waterbodies(
        "data/static/aoi/wflow_nhdplus_watersheds.geojson",
        output,
        location_root=location_root,
        session_get=fake_get,
    )

    assert f"/{NHDPLUS_HR_WATERBODY_LAYER}/query" in calls[0][0]
    assert summary["reservoir_features"] == 1
    written = gpd.read_file(output)
    assert set(["waterbody_id", "Area_avg", "Depth_avg", "Vol_avg", "Dis_avg"]).issubset(written.columns)


def test_select_stream_geo_file_prefers_stream_parquet():
    selected = select_stream_geo_file(
        {
            "files": [
                {"name": "readme.txt", "download_url": "https://example.test/readme.txt"},
                {"name": "STREAM-geo.parquet", "download_url": "https://example.test/stream.parquet"},
            ]
        }
    )

    assert selected["name"] == "STREAM-geo.parquet"


def test_collect_stream_geo_nldi_reuses_cached_table(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    table = location_root / "data/sources/national_hydrography/stream_geo.parquet"
    table.parent.mkdir(parents=True)
    pd.DataFrame({"COMID": ["1"], "width_m": [12.0], "depth_m": [1.2]}).to_parquet(table, index=False)

    result = collect_stream_geo_nldi(
        {
            "config": {
                "collection": {
                    "stream_geo_nldi": {
                        "stream_geo_table": "data/sources/national_hydrography/stream_geo.parquet",
                    },
                    "national_hydrography": {
                        "stream_geo_table": "data/sources/national_hydrography/stream_geo.parquet",
                    },
                }
            },
            "paths": {
                "location_root": location_root,
                "location_name": "greensboro",
                "repo_root": tmp_path,
                "source_artifacts_root": location_root / "data/sources/source_artifacts",
            },
        },
        skip_existing=True,
    )

    assert result["reused"] is True
    assert result["rows"] == 1
    assert result["stream_geo_table"] == table
    assert result["source_artifact_json"].exists()


def test_write_dem_derived_hydromt_basemap_accepts_rasterio_crs(tmp_path):
    dem = tmp_path / "dem.tif"
    output = tmp_path / "us_hydrography_basemap.nc"

    values = np.array(
        [
            [9, 8, 7, 6, 5],
            [8, 7, 6, 5, 4],
            [7, 6, 5, 4, 3],
            [6, 5, 4, 3, 2],
            [5, 4, 3, 2, 1],
        ],
        dtype="float32",
    )
    with rasterio.open(
        dem,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-80, 37, 0.01, 0.01),
        nodata=-9999.0,
    ) as dst:
        dst.write(values, 1)

    summary = write_dem_derived_hydromt_basemap(
        dem,
        output,
        target_resolution_degrees=0.01,
        min_stream_uparea_km2=0.01,
    )

    assert output.exists()
    assert summary["hydrography_cells"] == 25
    ds = xr.open_dataset(output)
    try:
        assert {"flwdir", "elevtn", "uparea", "strord", "basins"}.issubset(ds.data_vars)
        assert ds.raster.crs.to_epsg() == 4326
        assert ds["flwdir"].dtype == "uint8"
        assert ds["flwdir"].raster.nodata is None
        assert ds["strord"].raster.nodata is not None
        assert ds["basins"].raster.nodata is not None
        assert summary["resolution_degrees"] == pytest.approx(0.01)
        pyflwdir.from_array(ds["flwdir"].values, ftype="infer", transform=ds.raster.transform, latlon=True)
    finally:
        ds.close()


def test_write_dem_derived_hydromt_basemap_normalizes_geographic_crs_to_wgs84(tmp_path):
    dem = tmp_path / "dem_4269.tif"
    output = tmp_path / "us_hydrography_basemap.nc"

    values = np.array(
        [
            [9, 8, 7, 6, 5],
            [8, 7, 6, 5, 4],
            [7, 6, 5, 4, 3],
            [6, 5, 4, 3, 2],
            [5, 4, 3, 2, 1],
        ],
        dtype="float32",
    )
    with rasterio.open(
        dem,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=1,
        dtype="float32",
        crs="EPSG:4269",
        transform=from_origin(-80, 37, 0.01, 0.01),
        nodata=-9999.0,
    ) as dst:
        dst.write(values, 1)

    write_dem_derived_hydromt_basemap(
        dem,
        output,
        target_resolution_degrees=0.01,
        min_stream_uparea_km2=0.01,
    )

    ds = xr.open_dataset(output)
    try:
        assert ds.raster.crs.to_epsg() == 4326
    finally:
        ds.close()


def test_collect_national_hydrography_rebuilds_stale_hydromt_basemap_resolution(tmp_path, monkeypatch):
    location_root = tmp_path / "locations/greensboro"
    hydrography = location_root / "data/wflow/hydrography/us_hydrography_basemap.nc"
    river = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    catchments = location_root / "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg"
    soil = location_root / "data/wflow/static/ssurgo_wflow_soil_parameters.nc"
    dem = location_root / "data/static/processed/dem_region_setup.tif"
    hsg = location_root / "data/static/soils/hsg_greensboro.tif"
    for path in (river, catchments, soil, dem, hsg):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")
    hydrography.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(coords={"y": [0.0, 0.01], "x": [0.0, 0.01]}).to_netcdf(hydrography)

    calls = []

    def fake_hydrography(dem_path, output_path, *, target_resolution_degrees, min_stream_uparea_km2):
        calls.append(target_resolution_degrees)
        xr.Dataset(coords={"y": [0.0, target_resolution_degrees], "x": [0.0, target_resolution_degrees]}).to_netcdf(output_path)
        return {"hydrography_cells": 4, "max_uparea_km2": 1.0, "stream_cells": 1}

    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography.write_dem_derived_hydromt_basemap",
        fake_hydrography,
    )
    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography.write_review_hydrography_vectors",
        lambda *args, **kwargs: pytest.fail("NHDPlus review vectors should be optional by default"),
    )
    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography._ensure_ssurgo_wflow_attributes",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography.write_ssurgo_wflow_soil_dataset",
        lambda *args, **kwargs: {"soil_cells": 1},
    )

    result = collect_national_hydrography(
        {
            "config": {
                "collection": {
                    "national_hydrography": {
                        "hydromt_basemap": "data/wflow/hydrography/us_hydrography_basemap.nc",
                        "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg",
                        "catchments": "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg",
                        "wflow_soil_parameters": "data/wflow/static/ssurgo_wflow_soil_parameters.nc",
                        "basemap_source_resolution_degrees": 1 / 1800,
                    }
                },
                "static_sources": {
                    "terrain": {"output": "data/static/processed/dem_region_setup.tif"},
                    "ssurgo": {"hsg_output": "data/static/soils/hsg_greensboro.tif"},
                },
            },
            "paths": {"location_root": location_root},
        },
        skip_existing=True,
    )

    assert result["status"] == "collected"
    assert result["artifact_count"] == 2
    assert calls == [1 / 1800]


def test_collect_national_hydrography_rebuilds_stale_hydromt_basemap_crs(tmp_path, monkeypatch):
    location_root = tmp_path / "locations/greensboro"
    hydrography = location_root / "data/wflow/hydrography/us_hydrography_basemap.nc"
    river = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    catchments = location_root / "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg"
    soil = location_root / "data/wflow/static/ssurgo_wflow_soil_parameters.nc"
    dem = location_root / "data/static/processed/dem_region_setup.tif"
    hsg = location_root / "data/static/soils/hsg_greensboro.tif"
    for path in (river, catchments, soil, dem, hsg):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")
    hydrography.parent.mkdir(parents=True, exist_ok=True)
    stale = xr.Dataset(coords={"y": [1.0, 1.01], "x": [0.0, 0.01]})
    stale.raster.set_crs("EPSG:4269")
    stale.to_netcdf(hydrography)

    calls = []

    def fake_hydrography(dem_path, output_path, *, target_resolution_degrees, min_stream_uparea_km2):
        calls.append(target_resolution_degrees)
        ready = xr.Dataset(coords={"y": [1.0, 1.01], "x": [0.0, 0.01]})
        ready.raster.set_crs("EPSG:4326")
        ready.to_netcdf(output_path)
        return {"hydrography_cells": 4, "max_uparea_km2": 1.0, "stream_cells": 1}

    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography.write_dem_derived_hydromt_basemap",
        fake_hydrography,
    )
    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography.write_review_hydrography_vectors",
        lambda *args, **kwargs: pytest.fail("NHDPlus review vectors should be optional by default"),
    )
    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography._ensure_ssurgo_wflow_attributes",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography.write_ssurgo_wflow_soil_dataset",
        lambda *args, **kwargs: {"soil_cells": 1},
    )

    result = collect_national_hydrography(
        {
            "config": {
                "collection": {
                    "national_hydrography": {
                        "hydromt_basemap": "data/wflow/hydrography/us_hydrography_basemap.nc",
                        "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg",
                        "catchments": "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg",
                        "wflow_soil_parameters": "data/wflow/static/ssurgo_wflow_soil_parameters.nc",
                        "basemap_source_resolution_degrees": 0.01,
                    }
                },
                "static_sources": {
                    "terrain": {"output": "data/static/processed/dem_region_setup.tif"},
                    "ssurgo": {"hsg_output": "data/static/soils/hsg_greensboro.tif"},
                },
            },
            "paths": {"location_root": location_root},
        },
        skip_existing=True,
    )

    assert result["status"] == "collected"
    assert calls == [0.01]


def test_collect_national_hydrography_collects_review_vectors_when_configured(tmp_path, monkeypatch):
    location_root = tmp_path / "locations/austin"
    hydrography = location_root / "data/wflow/hydrography/us_hydrography_basemap.nc"
    river = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    catchments = location_root / "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg"
    soil = location_root / "data/wflow/static/ssurgo_wflow_soil_parameters.nc"
    dem = location_root / "data/wflow/static/processed/dem_wflow_coarse.tif"
    hsg = location_root / "data/wflow/static/soils/hsg_wflow.tif"
    for path in (dem, hsg):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")
    hydrography_inputs = []
    soil_inputs = []

    def fake_hydrography(dem_path, output_path, *, target_resolution_degrees, min_stream_uparea_km2):
        hydrography_inputs.append(Path(dem_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset(coords={"y": [0.0, target_resolution_degrees], "x": [0.0, target_resolution_degrees]}).to_netcdf(output_path)
        return {"hydrography_cells": 4, "max_uparea_km2": 1.0, "stream_cells": 1}

    def fake_review_vectors(hydrography_nc, river_output, catchment_output, **kwargs):
        Path(river_output).parent.mkdir(parents=True, exist_ok=True)
        Path(river_output).write_text("rivers", encoding="utf-8")
        Path(catchment_output).write_text("catchments", encoding="utf-8")
        return {"river_features": 2, "catchment_features": 3}

    def fake_soil_dataset(soil_polygons, soil_attributes, template_raster, output_path):
        soil_inputs.append((Path(soil_polygons), Path(soil_attributes), Path(template_raster)))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("soil", encoding="utf-8")
        return {"soil_cells": 1}

    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography.write_dem_derived_hydromt_basemap",
        fake_hydrography,
    )
    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography.write_review_hydrography_vectors",
        fake_review_vectors,
    )
    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography._ensure_ssurgo_wflow_attributes",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography.write_ssurgo_wflow_soil_dataset",
        fake_soil_dataset,
    )

    result = collect_national_hydrography(
        {
            "config": {
                "collection": {
                    "national_hydrography": {
                        "hydromt_basemap": "data/wflow/hydrography/us_hydrography_basemap.nc",
                        "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg",
                        "catchments": "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg",
                        "wflow_soil_parameters": "data/wflow/static/ssurgo_wflow_soil_parameters.nc",
                        "basemap_source_resolution_degrees": 1 / 1800,
                        "collect_review_vectors": True,
                    }
                },
                "static_sources": {
                    "terrain": {"output": "data/static/processed/dem_region_setup.tif"},
                    "ssurgo": {
                        "output": "data/static/soils/ssurgo_mapunitpoly.gpkg",
                        "attributes_output": "data/static/soils/ssurgo_mapunit_attributes.csv",
                        "hsg_output": "data/static/soils/hsg_austin.tif",
                    },
                    "wflow_collection_extent": {
                        "terrain_output": "data/wflow/static/processed/dem_wflow_coarse.tif",
                        "ssurgo_output": "data/wflow/static/soils/ssurgo_mapunitpoly_wflow.gpkg",
                        "ssurgo_attributes_output": "data/wflow/static/soils/ssurgo_mapunit_attributes_wflow.csv",
                        "hsg_output": "data/wflow/static/soils/hsg_wflow.tif",
                    },
                },
            },
            "paths": {"location_root": location_root},
        },
        skip_existing=True,
    )

    assert result["status"] == "collected"
    assert result["artifact_count"] == 4
    assert hydrography.exists()
    assert river.exists()
    assert catchments.exists()
    assert soil.exists()
    manifest = json.loads(result["source_artifact_json"].read_text(encoding="utf-8"))
    assert manifest["metadata"]["review_vectors_collected"] is True
    assert manifest["metadata"]["river_features"] == 2
    assert manifest["metadata"]["catchment_features"] == 3
    assert hydrography_inputs == [location_root / "data/wflow/static/processed/dem_wflow_coarse.tif"]
    assert soil_inputs == [
        (
            location_root / "data/wflow/static/soils/ssurgo_mapunitpoly_wflow.gpkg",
            location_root / "data/wflow/static/soils/ssurgo_mapunit_attributes_wflow.csv",
            location_root / "data/wflow/static/soils/hsg_wflow.tif",
        )
    ]


def test_refresh_wflow_hydrography_basemap_only_rewrites_hydromt_basemap(tmp_path, monkeypatch):
    location_root = tmp_path
    hydrography = location_root / "data/wflow/hydrography/us_hydrography_basemap.nc"
    soil = location_root / "data/wflow/static/ssurgo_wflow_soil_parameters.nc"
    dem = location_root / "data/wflow/static/processed/dem_wflow_coarse.tif"
    manifest = location_root / "data/sources/source_artifacts/national_hydrography_wflow_sources.json"
    dem.parent.mkdir(parents=True)
    dem.write_text("dem placeholder", encoding="utf-8")
    soil.parent.mkdir(parents=True, exist_ok=True)
    soil.write_text("keep me", encoding="utf-8")
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "source": "national_hydrography",
                "kind": "wflow_build_sources",
                "status": "review_required",
                "metadata": {"soil_method": "ssurgo_horizon_pedology"},
                "artifacts": {
                    "hydromt_basemap": str(hydrography),
                    "river_geometry": "existing_rivers.gpkg",
                    "catchments": "existing_catchments.gpkg",
                    "wflow_soil_parameters": str(soil),
                },
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_hydrography(dem_path, output_path, *, target_resolution_degrees, min_stream_uparea_km2):
        calls.append(
            {
                "dem_path": Path(dem_path),
                "output_path": Path(output_path),
                "target_resolution_degrees": target_resolution_degrees,
                "min_stream_uparea_km2": min_stream_uparea_km2,
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("fresh hydrography", encoding="utf-8")
        return {"hydrography_cells": 4, "max_uparea_km2": 2.0, "stream_cells": 1}

    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography.write_dem_derived_hydromt_basemap",
        fake_hydrography,
    )

    result = refresh_wflow_hydrography_basemap(
        {
            "config": {
                "collection": {
                    "national_hydrography": {
                        "hydromt_basemap": "data/wflow/hydrography/us_hydrography_basemap.nc",
                        "wflow_soil_parameters": "data/wflow/static/ssurgo_wflow_soil_parameters.nc",
                        "dem_source": "data/wflow/static/processed/dem_wflow_coarse.tif",
                        "basemap_source_resolution_degrees": 0.01,
                        "min_stream_uparea_km2": 3.0,
                    }
                }
            },
            "paths": {"location_root": location_root, "source_artifacts_root": manifest.parent},
        },
        skip_existing=False,
    )

    assert result["status"] == "collected"
    assert result["hydrography_only"] is True
    assert result["artifact_count"] == 1
    assert hydrography.read_text(encoding="utf-8") == "fresh hydrography"
    assert soil.read_text(encoding="utf-8") == "keep me"
    assert calls == [
        {
            "dem_path": dem,
            "output_path": hydrography,
            "target_resolution_degrees": 0.01,
            "min_stream_uparea_km2": 3.0,
        }
    ]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["metadata"]["hydrography_only_refresh"] is True
    assert payload["metadata"]["soil_method"] == "ssurgo_horizon_pedology"
    assert payload["artifacts"]["wflow_soil_parameters"] == str(soil)
    assert payload["artifacts"]["river_geometry"] == "existing_rivers.gpkg"


def test_refresh_wflow_river_geometry_sources_skips_dem_and_soil_rebuild(tmp_path, monkeypatch):
    location_root = tmp_path
    hydrography = location_root / "data/wflow/hydrography/us_hydrography_basemap.nc"
    river = location_root / "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    catchments = location_root / "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg"
    stream_geo = location_root / "data/sources/national_hydrography/stream_geo.parquet"
    nldi_cache = location_root / "data/sources/national_hydrography/nldi_stream_geo_comid_cache.csv"
    manifest = location_root / "data/sources/source_artifacts/national_hydrography_wflow_sources.json"
    hydrography.parent.mkdir(parents=True)
    xr.Dataset(coords={"y": [0.0, 0.01], "x": [0.0, 0.01]}).to_netcdf(hydrography)
    stream_geo.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"COMID": ["1"], "XGB_Width_m": [10.0], "XGB_Depth_m": [1.0]}).to_parquet(stream_geo, index=False)

    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography.write_dem_derived_hydromt_basemap",
        lambda *args, **kwargs: pytest.fail("river-geometry refresh should not rebuild the DEM-derived basemap"),
    )
    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography.write_ssurgo_wflow_soil_dataset",
        lambda *args, **kwargs: pytest.fail("river-geometry refresh should not rebuild soil parameters"),
    )

    def fake_review_vectors(hydrography_nc, river_output, catchment_output, **kwargs):
        assert Path(hydrography_nc) == hydrography
        assert kwargs["stream_geo_table"] == stream_geo
        assert kwargs["nldi_lookup_cache"] == nldi_cache
        Path(river_output).parent.mkdir(parents=True, exist_ok=True)
        Path(river_output).write_text("rivers", encoding="utf-8")
        Path(catchment_output).write_text("catchments", encoding="utf-8")
        return {"river_features": 2, "catchment_features": 3}

    monkeypatch.setattr(
        "design_events.collect_sources.national_hydrography.write_review_hydrography_vectors",
        fake_review_vectors,
    )

    result = refresh_wflow_river_geometry_sources(
        {
            "config": {
                "collection": {
                    "stream_geo_nldi": {
                        "stream_geo_table": "data/sources/national_hydrography/stream_geo.parquet",
                        "nldi_lookup_cache": "data/sources/national_hydrography/nldi_stream_geo_comid_cache.csv",
                    },
                    "national_hydrography": {
                        "hydromt_basemap": "data/wflow/hydrography/us_hydrography_basemap.nc",
                        "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg",
                        "catchments": "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg",
                        "stream_geo_table": "data/sources/national_hydrography/stream_geo.parquet",
                        "nldi_lookup_cache": "data/sources/national_hydrography/nldi_stream_geo_comid_cache.csv",
                    },
                }
            },
            "paths": {"location_root": location_root, "source_artifacts_root": manifest.parent},
        },
        skip_existing=False,
    )

    assert result["status"] == "collected"
    assert result["river_geometry_only"] is True
    assert river.read_text(encoding="utf-8") == "rivers"
    assert catchments.read_text(encoding="utf-8") == "catchments"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["metadata"]["river_geometry_only_refresh"] is True
    assert payload["metadata"]["river_features"] == 2
    assert payload["artifacts"]["river_geometry"] == str(river)


def test_fetch_nhdplus_hr_layer_uses_arcgis_geojson_query(tmp_path):
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"nhdplusid": 1},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-79.8, 36.1], [-79.7, 36.2]],
                        },
                    }
                ],
            }

    def fake_get(url, params, timeout):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return Response()

    frame = fetch_nhdplus_hr_layer(
        (-80.0, 36.0, -79.5, 36.4),
        layer_id=3,
        service_url="https://example.test/MapServer",
        session_get=fake_get,
    )

    assert len(frame) == 1
    assert calls[0]["url"] == "https://example.test/MapServer/3/query"
    assert calls[0]["params"]["f"] == "geojson"
    assert calls[0]["params"]["geometryType"] == "esriGeometryEnvelope"
    assert calls[0]["params"]["outSR"] == "4326"


def test_fetch_nhdplus_hr_catchments_queries_catchment_layer_and_prepares(tmp_path):
    # Regression: 01_region_setup fetches NHDPlus HR catchments on demand for fresh
    # locations. The wrapper must hit the catchment layer and return prepared columns.
    calls = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"nhdplusid": 42},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-98.1, 30.1], [-98.0, 30.1], [-98.0, 30.2], [-98.1, 30.2], [-98.1, 30.1]]],
                        },
                    }
                ],
            }

    def fake_get(url, params, timeout):
        calls.append(url)
        return Response()

    catchments = fetch_nhdplus_hr_catchments(
        (-98.2, 30.0, -97.4, 30.6),
        service_url="https://example.test/MapServer",
        session_get=fake_get,
    )

    assert calls[0] == f"https://example.test/MapServer/{NHDPLUS_HR_CATCHMENT_LAYER}/query"
    assert len(catchments) == 1
    assert catchments["review_status"].iloc[0] == "review_required_usgs_nhdplus_hr"
    assert catchments["basid"].iloc[0] == 42
    assert str(catchments.crs).upper().endswith("4326")


def test_write_review_hydrography_vectors_uses_nhdplus_hr_layers(tmp_path):
    dem = tmp_path / "dem.tif"
    hydrography = tmp_path / "us_hydrography_basemap.nc"
    river_output = tmp_path / "nhdplus_hr_river_geometry.gpkg"
    catchment_output = tmp_path / "nhdplus_hr_catchments.gpkg"
    calls = []

    with rasterio.open(
        dem,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-80, 37, 0.01, 0.01),
        nodata=-9999.0,
    ) as dst:
        dst.write(np.arange(25, dtype="float32").reshape(5, 5), 1)
    write_dem_derived_hydromt_basemap(dem, hydrography, target_resolution_degrees=0.01)

    class Response:
        def __init__(self, features):
            self.features = features

        def raise_for_status(self):
            return None

        def json(self):
            return {"type": "FeatureCollection", "features": self.features}

    def fake_get(url, params, timeout):
        calls.append(url)
        if url.endswith("/3/query"):
            return Response(
                [
                    {
                        "type": "Feature",
                        "properties": {"nhdplusid": 101, "TotDASqKm": 25.0},
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-79.99, 36.96], [-79.97, 36.94]],
                        },
                    }
                ]
            )
        if url.endswith("/10/query"):
            return Response(
                [
                    {
                        "type": "Feature",
                        "properties": {"featureid": 101},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-80.0, 36.95], [-79.95, 36.95], [-79.95, 37.0], [-80.0, 37.0], [-80.0, 36.95]]],
                        },
                    }
                ]
            )
        raise AssertionError(url)

    summary = write_review_hydrography_vectors(
        hydrography,
        river_output,
        catchment_output,
        service_url="https://example.test/MapServer",
        session_get=fake_get,
    )

    assert summary["river_source"] == "USGS NHDPlus HR NetworkNHDFlowline"
    assert calls == ["https://example.test/MapServer/3/query", "https://example.test/MapServer/10/query"]
    rivers = gpd.read_file(river_output)
    catchments = gpd.read_file(catchment_output)
    assert rivers["source"].unique().tolist() == ["USGS NHDPlus HR NetworkNHDFlowline"]
    assert {"rivwth", "qbankfull", "review_status"}.issubset(rivers.columns)
    assert catchments["basid"].tolist() == [101]


def test_write_ssurgo_wflow_soil_dataset_uses_horizon_pedology(tmp_path):
    polygons = tmp_path / "ssurgo_mapunitpoly.gpkg"
    attributes = tmp_path / "ssurgo_mapunit_attributes.csv"
    template = tmp_path / "hsg.tif"
    output = tmp_path / "ssurgo_wflow_soil_parameters.nc"

    gpd.GeoDataFrame(
        {"mukey": ["1001"]},
        geometry=[box(0, 0, 2, 2)],
        crs="EPSG:4326",
    ).to_file(polygons, driver="GPKG")
    pd.DataFrame(
        [
            {
                "mukey": "1001",
                "hydgrp": "B",
                "ksat_r": 12.5,
                "hzdept_r": 0,
                "hzdepb_r": 30,
                "sandtotal_r": 55,
                "silttotal_r": 30,
                "claytotal_r": 15,
                "dbthirdbar_r": 1.4,
                "om_r": 2.0,
                "ph1to1h2o_r": 6.4,
            },
            {
                "mukey": "1001",
                "hydgrp": "B",
                "ksat_r": 6.0,
                "hzdept_r": 30,
                "hzdepb_r": 200,
                "sandtotal_r": 45,
                "silttotal_r": 35,
                "claytotal_r": 20,
                "dbthirdbar_r": 1.5,
                "om_r": 1.0,
                "ph1to1h2o_r": 6.1,
            },
        ]
    ).to_csv(attributes, index=False)
    with rasterio.open(
        template,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
        nodata=0,
    ) as dst:
        dst.write(np.ones((2, 2), dtype="uint8"), 1)

    summary = write_ssurgo_wflow_soil_dataset(polygons, attributes, template, output)

    assert summary["wflow_soil_pixels"] == 4
    ds = xr.open_dataset(output)
    try:
        for prefix in ["bd", "oc", "ph", "clyppt", "sltppt", "sndppt"]:
            for layer in range(1, 8):
                assert f"{prefix}_sl{layer}" in ds
        assert "soilthickness" in ds
        assert float(ds["sndppt_sl1"].max()) == 55.0
        assert float(ds["sndppt_sl4"].max()) == 45.0
        assert ds.attrs["status"] == "review_required"
    finally:
        ds.close()
