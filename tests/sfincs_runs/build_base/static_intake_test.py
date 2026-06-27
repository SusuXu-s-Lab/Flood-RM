from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
import xarray as xr

from sfincs_runs.build_base.static_intake import (
    clip_dem_and_landcover_to_bbox,
    collect_ssurgo_infiltration_inputs,
    collect_static_region_inputs,
    collect_wflow_static_region_inputs,
    fetch_usgs_3dep_dem,
    fetch_worldcover_landcover,
    worldcover_tile_urls,
)
import sfincs_runs.build_base.static_intake as static_intake


def _write_raster(path, values, *, crs="EPSG:4326", transform=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values)
    if transform is None:
        transform = from_origin(-80.0, 36.5, 0.1, 0.1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(values, 1)


def test_worldcover_tile_urls_cover_bbox_tiles():
    urls = worldcover_tile_urls((-80.1, 35.9, -79.7, 36.2))

    assert len(urls) == 2
    assert urls[0].endswith("ESA_WorldCover_10m_2021_v200_N33W081_Map.tif")
    assert urls[1].endswith("ESA_WorldCover_10m_2021_v200_N36W081_Map.tif")


def test_fetch_worldcover_landcover_clips_tiles_before_merge_and_writes_atomically(tmp_path, monkeypatch):
    landcover_raw = tmp_path / "raw/landcover/landcover.tif"
    tile_cache = tmp_path / "raw/landcover/worldcover_tiles"
    bbox = (-80.1, 35.9, -79.7, 36.2)
    clip_calls = []
    merged_inputs = []
    opened_paths = []
    written_paths = []

    for url in worldcover_tile_urls(bbox):
        tile_path = tile_cache / Path(url).name
        tile_path.parent.mkdir(parents=True, exist_ok=True)
        tile_path.write_bytes(b"tile")

    class FakeRio:
        def __init__(self, parent):
            self.parent = parent

        def clip_box(self, *bounds, crs=None):
            clip_calls.append((self.parent.name, bounds, crs))
            return FakeArray(f"clipped:{self.parent.name}")

        def to_raster(self, path):
            written_paths.append(Path(path))
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"landcover")

    class FakeArray:
        def __init__(self, name):
            self.name = name
            self.rio = FakeRio(self)

        def squeeze(self, drop=False):
            return self

        def close(self):
            pass

    def fake_open_rasterio(path, **kwargs):
        opened_paths.append((Path(path), kwargs))
        return FakeArray(Path(path).name)

    def fake_merge_arrays(arrays):
        merged_inputs.extend(array.name for array in arrays)
        return FakeArray("merged")

    monkeypatch.setattr(static_intake.rxr, "open_rasterio", fake_open_rasterio)
    monkeypatch.setattr(static_intake, "merge_arrays", fake_merge_arrays)

    result_path, source = fetch_worldcover_landcover(bbox, landcover_raw, tile_cache=tile_cache)

    assert result_path == landcover_raw
    assert source == "downloaded"
    assert [path.name for path, _ in opened_paths] == [
        "ESA_WorldCover_10m_2021_v200_N33W081_Map.tif",
        "ESA_WorldCover_10m_2021_v200_N36W081_Map.tif",
    ]
    assert all(kwargs == {"masked": True} for _, kwargs in opened_paths)
    assert clip_calls == [
        ("ESA_WorldCover_10m_2021_v200_N33W081_Map.tif", bbox, "EPSG:4326"),
        ("ESA_WorldCover_10m_2021_v200_N36W081_Map.tif", bbox, "EPSG:4326"),
    ]
    assert merged_inputs == [
        "clipped:ESA_WorldCover_10m_2021_v200_N33W081_Map.tif",
        "clipped:ESA_WorldCover_10m_2021_v200_N36W081_Map.tif",
    ]
    assert written_paths == [landcover_raw.with_name(".landcover.tmp.tif")]
    assert landcover_raw.exists()
    assert not landcover_raw.with_name(".landcover.tmp.tif").exists()


def test_fetch_worldcover_landcover_coarsens_each_tile_before_merge(tmp_path, monkeypatch):
    landcover_raw = tmp_path / "raw/landcover/landcover.tif"
    tile_cache = tmp_path / "raw/landcover/worldcover_tiles"
    bbox = (-80.1, 35.9, -79.7, 36.2)
    reprojections = []
    merged_inputs = []

    for url in worldcover_tile_urls(bbox):
        tile_path = tile_cache / Path(url).name
        tile_path.parent.mkdir(parents=True, exist_ok=True)
        tile_path.write_bytes(b"tile")

    class FakeRio:
        def __init__(self, parent):
            self.parent = parent

        def clip_box(self, *bounds, crs=None):
            return FakeArray(f"clipped:{self.parent.name}")

        def reproject(self, crs, *, resolution, resampling):
            reprojections.append((self.parent.name, crs, resolution, resampling.name))
            return FakeArray(f"coarse:{self.parent.name}")

        def to_raster(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"landcover")

    class FakeArray:
        def __init__(self, name):
            self.name = name
            self.rio = FakeRio(self)

        def squeeze(self, drop=False):
            return self

        def close(self):
            pass

    def fake_open_rasterio(path, **kwargs):
        return FakeArray(Path(path).name)

    def fake_merge_arrays(arrays):
        merged_inputs.extend(array.name for array in arrays)
        return FakeArray("merged")

    monkeypatch.setattr(static_intake.rxr, "open_rasterio", fake_open_rasterio)
    monkeypatch.setattr(static_intake, "merge_arrays", fake_merge_arrays)

    fetch_worldcover_landcover(bbox, landcover_raw, tile_cache=tile_cache, target_resolution_degrees=0.01)

    assert reprojections == [
        ("clipped:ESA_WorldCover_10m_2021_v200_N33W081_Map.tif", "EPSG:4326", 0.01, "nearest"),
        ("clipped:ESA_WorldCover_10m_2021_v200_N36W081_Map.tif", "EPSG:4326", 0.01, "nearest"),
    ]
    assert merged_inputs == [
        "coarse:clipped:ESA_WorldCover_10m_2021_v200_N33W081_Map.tif",
        "coarse:clipped:ESA_WorldCover_10m_2021_v200_N36W081_Map.tif",
    ]


def test_fetch_worldcover_landcover_streams_uncached_tiles_for_coarse_target(tmp_path, monkeypatch):
    landcover_raw = tmp_path / "raw/landcover/landcover.tif"
    tile_cache = tmp_path / "raw/landcover/worldcover_tiles"
    bbox = (-80.1, 35.9, -79.7, 36.2)
    opened_sources = []

    class FakeRio:
        def __init__(self, parent):
            self.parent = parent

        def clip_box(self, *bounds, crs=None):
            return FakeArray(f"clipped:{self.parent.name}")

        def reproject(self, crs, *, resolution, resampling):
            return FakeArray(f"coarse:{self.parent.name}")

        def to_raster(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"landcover")

    class FakeArray:
        def __init__(self, name):
            self.name = name
            self.rio = FakeRio(self)

        def squeeze(self, drop=False):
            return self

        def close(self):
            pass

    def fake_open_rasterio(source, **kwargs):
        opened_sources.append(str(source))
        return FakeArray(Path(source).name)

    def fake_download_file(url, output_path):
        raise AssertionError("coarse Wflow landcover should not cache full-resolution WorldCover tiles")

    monkeypatch.setattr(static_intake.rxr, "open_rasterio", fake_open_rasterio)
    monkeypatch.setattr(static_intake, "merge_arrays", lambda arrays: FakeArray("merged"))
    monkeypatch.setattr(static_intake, "_download_file", fake_download_file)

    fetch_worldcover_landcover(bbox, landcover_raw, tile_cache=tile_cache, target_resolution_degrees=0.01)

    assert opened_sources == worldcover_tile_urls(bbox)
    assert not tile_cache.exists()


def test_write_raster_atomically_removes_conflicting_fill_value_metadata(tmp_path):
    output = tmp_path / "landcover.tif"
    raster = xr.DataArray(
        np.array([[1, 2], [3, 4]], dtype="uint8"),
        dims=("y", "x"),
        coords={"y": [1.5, 0.5], "x": [0.5, 1.5]},
    )
    raster = raster.rio.write_crs("EPSG:4326")
    raster.rio.write_transform(from_origin(0, 2, 1, 1), inplace=True)
    raster.attrs["_FillValue"] = 0
    raster.encoding["_FillValue"] = 0

    static_intake._write_raster_atomically(raster, output)

    with rasterio.open(output) as src:
        assert src.read(1).tolist() == [[1, 2], [3, 4]]
    assert not output.with_name(".landcover.tmp.tif").exists()


def test_clip_dem_and_landcover_to_bbox_writes_processed_rasters(tmp_path):
    dem_raw = tmp_path / "raw/topo/dem.tif"
    landcover_raw = tmp_path / "raw/landcover/landcover.tif"
    dem_output = tmp_path / "processed/dem_region_setup.tif"
    landcover_output = tmp_path / "processed/landcover_region_setup.tif"
    _write_raster(dem_raw, np.arange(100, dtype="float32").reshape(10, 10))
    _write_raster(landcover_raw, np.ones((10, 10), dtype="uint8"))
    bbox = gpd.GeoDataFrame({"name": ["bbox"]}, geometry=[box(-79.9, 35.9, -79.4, 36.4)], crs="EPSG:4326")

    summary = clip_dem_and_landcover_to_bbox(
        dem_raw,
        landcover_raw,
        dem_output,
        landcover_output,
        bbox,
        model_crs="EPSG:4326",
    )

    assert summary["dem"] == dem_output
    assert summary["landcover"] == landcover_output
    assert not dem_output.with_name(".dem_region_setup.tmp.tif").exists()
    assert not landcover_output.with_name(".landcover_region_setup.tmp.tif").exists()
    with rasterio.open(dem_output) as src:
        assert src.width < 10
        assert src.height < 10
    with rasterio.open(landcover_output) as src:
        assert src.crs.to_epsg() == 4326


def test_clip_landcover_covers_full_dem_extent_for_multi_polygon_bbox(tmp_path):
    """Landcover must cover the same footprint as the DEM even when the AOI is several
    disjoint sub-region boxes that do not tile their bounding rectangle. Otherwise SFINCS
    domains spanning the gaps lose landcover and fall back to a flat manning value."""
    import rioxarray as rxr

    dem_raw = tmp_path / "raw/topo/dem.tif"
    landcover_raw = tmp_path / "raw/landcover/landcover.tif"
    dem_output = tmp_path / "processed/dem_region_setup.tif"
    landcover_output = tmp_path / "processed/landcover_region_setup.tif"
    # 10x10 raster covering lon [-80, -79], lat [35.5, 36.5].
    _write_raster(dem_raw, np.arange(100, dtype="float32").reshape(10, 10) + 1.0)
    _write_raster(landcover_raw, np.ones((10, 10), dtype="float32"))
    # Two diagonal sub-region boxes (SW + NE) whose union covers only ~half the hull.
    bbox = gpd.GeoDataFrame(
        {"name": ["sw", "ne"]},
        geometry=[box(-80.0, 35.5, -79.5, 36.0), box(-79.5, 36.0, -79.0, 36.5)],
        crs="EPSG:4326",
    )

    clip_dem_and_landcover_to_bbox(
        dem_raw, landcover_raw, dem_output, landcover_output, bbox, model_crs="EPSG:4326"
    )

    dem = rxr.open_rasterio(dem_output, masked=True).squeeze(drop=True)
    landcover = rxr.open_rasterio(landcover_output, masked=True).squeeze(drop=True)
    dem_valid = float(np.isfinite(dem.values).mean())
    landcover_valid = float(np.isfinite(landcover.values).mean())
    assert dem_valid > 0.99
    # Landcover must match the DEM footprint, not be punched full of holes in the
    # corners where the AOI boxes leave gaps.
    assert landcover_valid == pytest.approx(dem_valid, abs=0.02)


def test_fetch_usgs_3dep_dem_reuses_existing_raw_dem_without_catalog_request(tmp_path):
    dem_raw = tmp_path / "raw/topo/dem.tif"
    _write_raster(dem_raw, np.ones((2, 2), dtype="float32"))

    summary = fetch_usgs_3dep_dem((-80.1, 35.9, -79.7, 36.2), dem_raw)

    assert summary == {"dem_raw": dem_raw, "tiles_merged": 0, "source": "existing"}


def test_fetch_usgs_3dep_dem_clips_remote_items_before_merge_and_writes_atomically(tmp_path, monkeypatch):
    dem_raw = tmp_path / "raw/topo/dem.tif"
    bbox = (-80.1, 35.9, -79.7, 36.2)
    clip_calls = []
    merged_inputs = []
    opened_urls = []
    written_paths = []

    class FakeRio:
        def __init__(self, parent):
            self.parent = parent

        def clip_box(self, *bounds, crs=None):
            clip_calls.append((self.parent.name, bounds, crs))
            return FakeArray(f"clipped:{self.parent.name}")

        def to_raster(self, path):
            written_paths.append(Path(path))
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"dem")

    class FakeArray:
        def __init__(self, name):
            self.name = name
            self.rio = FakeRio(self)
            self.closed = False

        def squeeze(self, drop=False):
            return self

        def close(self):
            self.closed = True

    class FakeSearch:
        def items(self):
            return [
                SimpleNamespace(properties={"gsd": 10}, assets={"data": SimpleNamespace(href="https://example.test/a.tif")}),
                SimpleNamespace(properties={"gsd": 10}, assets={"data": SimpleNamespace(href="https://example.test/b.tif")}),
                SimpleNamespace(properties={"gsd": 30}, assets={"data": SimpleNamespace(href="https://example.test/skip.tif")}),
            ]

    class FakeCatalog:
        def search(self, *, collections, bbox):
            assert collections == ["3dep-seamless"]
            assert bbox == list((-80.1, 35.9, -79.7, 36.2))
            return FakeSearch()

    def fake_open_rasterio(url, **kwargs):
        opened_urls.append((url, kwargs))
        return FakeArray(Path(url).name)

    def fake_merge_arrays(arrays):
        merged_inputs.extend(array.name for array in arrays)
        return FakeArray("merged")

    monkeypatch.setattr("pystac_client.Client.open", lambda *args, **kwargs: FakeCatalog())
    monkeypatch.setattr(static_intake.rxr, "open_rasterio", fake_open_rasterio)
    monkeypatch.setattr(static_intake, "merge_arrays", fake_merge_arrays)

    summary = fetch_usgs_3dep_dem(bbox, dem_raw)

    assert summary == {"dem_raw": dem_raw, "tiles_merged": 2, "source": "usgs_3dep"}
    assert [url for url, _ in opened_urls] == ["https://example.test/a.tif", "https://example.test/b.tif"]
    assert all(kwargs == {"masked": True} for _, kwargs in opened_urls)
    assert clip_calls == [
        ("a.tif", bbox, "EPSG:4326"),
        ("b.tif", bbox, "EPSG:4326"),
    ]
    assert merged_inputs == ["clipped:a.tif", "clipped:b.tif"]
    assert written_paths == [dem_raw.with_name(".dem.tmp.tif")]
    assert dem_raw.exists()
    assert not dem_raw.with_name(".dem.tmp.tif").exists()


def test_fetch_usgs_3dep_dem_coarsens_each_item_before_merge(tmp_path, monkeypatch):
    dem_raw = tmp_path / "raw/topo/dem.tif"
    bbox = (-80.1, 35.9, -79.7, 36.2)
    reprojections = []
    merged_inputs = []

    class FakeRio:
        def __init__(self, parent):
            self.parent = parent

        def clip_box(self, *bounds, crs=None):
            return FakeArray(f"clipped:{self.parent.name}")

        def reproject(self, crs, *, resolution, resampling):
            reprojections.append((self.parent.name, crs, resolution, resampling.name))
            return FakeArray(f"coarse:{self.parent.name}")

        def to_raster(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"dem")

    class FakeArray:
        def __init__(self, name):
            self.name = name
            self.rio = FakeRio(self)

        def squeeze(self, drop=False):
            return self

        def close(self):
            pass

    class FakeSearch:
        def items(self):
            return [
                SimpleNamespace(properties={"gsd": 30}, assets={"data": SimpleNamespace(href="https://example.test/a.tif")}),
                SimpleNamespace(properties={"gsd": 30}, assets={"data": SimpleNamespace(href="https://example.test/b.tif")}),
            ]

    class FakeCatalog:
        def search(self, *, collections, bbox):
            return FakeSearch()

    def fake_open_rasterio(url, **kwargs):
        return FakeArray(Path(url).name)

    def fake_merge_arrays(arrays):
        merged_inputs.extend(array.name for array in arrays)
        return FakeArray("merged")

    monkeypatch.setattr("pystac_client.Client.open", lambda *args, **kwargs: FakeCatalog())
    monkeypatch.setattr(static_intake.rxr, "open_rasterio", fake_open_rasterio)
    monkeypatch.setattr(static_intake, "merge_arrays", fake_merge_arrays)

    fetch_usgs_3dep_dem(bbox, dem_raw, gsd=30, target_resolution_degrees=0.01)

    assert reprojections == [
        ("clipped:a.tif", "EPSG:4269", 0.01, "bilinear"),
        ("clipped:b.tif", "EPSG:4269", 0.01, "bilinear"),
    ]
    assert merged_inputs == ["coarse:clipped:a.tif", "coarse:clipped:b.tif"]


def test_collect_static_region_inputs_reuses_local_inputs_and_writes_soil_rasters(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    dem_raw = location_root / "data/static/raw/topo/dem.tif"
    landcover_raw = location_root / "data/static/raw/landcover/landcover.tif"
    _write_raster(dem_raw, np.ones((4, 4), dtype="float32"), crs="EPSG:32617", transform=from_origin(600000, 4000000, 10, 10))
    _write_raster(landcover_raw, np.ones((4, 4), dtype="uint8"), crs="EPSG:32617", transform=from_origin(600000, 4000000, 10, 10))
    bbox = gpd.GeoDataFrame(
        {"name": ["bbox"]},
        geometry=[box(600000, 3999960, 600040, 4000000)],
        crs="EPSG:32617",
    ).to_crs("EPSG:4326")
    bbox_path = location_root / "data/static/aoi/bbox.geojson"
    bbox_path.parent.mkdir(parents=True, exist_ok=True)
    bbox.to_file(bbox_path, driver="GeoJSON")
    soils = gpd.GeoDataFrame(
        {"mukey": ["1"]},
        geometry=[box(600000, 3999960, 600040, 4000000)],
        crs="EPSG:32617",
    )
    attrs = __import__("pandas").DataFrame(
        {
            "mukey": ["1"],
            "hydgrp": ["B"],
            "ksat_r": [20.0],
            "hzdept_r": [0.0],
            "hzdepb_r": [40.0],
        }
    )
    config = {
        "project": {"name": "greensboro", "model_crs": "EPSG:32617", "reference_crs": "EPSG:4326"},
        "grid_footprint": {"source": "data/static/aoi/bbox.geojson"},
        "static_sources": {
            "bbox": {"output": "data/static/aoi/bbox.geojson"},
            "terrain": {"raw": "data/static/raw/topo/dem.tif", "output": "data/static/processed/dem_region_setup.tif"},
            "landcover": {"raw": "data/static/raw/landcover/landcover.tif", "output": "data/static/processed/landcover_region_setup.tif"},
            "ssurgo": {
                "output": "data/static/soils/ssurgo_mapunitpoly.gpkg",
                "attributes_output": "data/static/soils/ssurgo_mapunit_attributes.csv",
                "hsg_output": "data/static/soils/hsg_greensboro.tif",
                "ksat_output": "data/static/soils/ksat_mmhr_greensboro.tif",
            },
        },
    }
    paths = {
        "repo_root": tmp_path,
        "location_name": "greensboro",
        "location_root": location_root,
        "static_root": location_root / "data/static/processed",
    }

    summary = collect_static_region_inputs(
        config,
        paths,
        fetch_dem=False,
        fetch_landcover=False,
        fetch_ssurgo=True,
        soil_polygons=soils,
        soil_attributes=attrs,
    )

    assert summary["dem_exists"] is True
    assert summary["landcover_exists"] is True
    assert summary["ssurgo_polygons"] == 1
    assert (location_root / "data/static/soils/hsg_greensboro.tif").exists()
    assert (location_root / "data/static/soils/ksat_mmhr_greensboro.tif").exists()
    assert not (location_root / "data/static/soils/.hsg_greensboro.tmp.tif").exists()
    assert not (location_root / "data/static/soils/.ksat_mmhr_greensboro.tmp.tif").exists()


def test_collect_wflow_static_region_inputs_uses_large_envelope_and_coarse_outputs(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    wflow_boundary = location_root / "data/static/aoi/wflow_collection_region.geojson"
    dem_raw = location_root / "data/wflow/static/raw/topo/dem_wflow.tif"
    landcover_raw = location_root / "data/wflow/static/raw/landcover/landcover_wflow.tif"
    sfincs_dem = location_root / "data/static/processed/dem_region_setup.tif"
    _write_raster(dem_raw, np.arange(100, dtype="float32").reshape(10, 10) + 1.0)
    _write_raster(landcover_raw, np.ones((10, 10), dtype="uint8"))
    wflow_boundary.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"name": ["wflow_collection"]},
        geometry=[box(-80.0, 35.5, -79.0, 36.5)],
        crs="EPSG:4326",
    ).to_file(wflow_boundary, driver="GeoJSON")
    soils = gpd.GeoDataFrame(
        {"mukey": ["1"]},
        geometry=[box(-80.0, 35.5, -79.0, 36.5)],
        crs="EPSG:4326",
    )
    attrs = __import__("pandas").DataFrame(
        {
            "mukey": ["1"],
            "hydgrp": ["B"],
            "ksat_r": [20.0],
            "hzdept_r": [0.0],
            "hzdepb_r": [40.0],
        }
    )
    config = {
        "project": {"name": "greensboro", "model_crs": "EPSG:32617", "reference_crs": "EPSG:4326"},
        "grid_footprint": {"source": "data/static/aoi/study_area.geojson"},
        "static_sources": {
            "bbox": {"output": "data/static/aoi/bbox.geojson"},
            "terrain": {"raw": "data/static/raw/topo/dem.tif", "output": "data/static/processed/dem_region_setup.tif"},
            "landcover": {"raw": "data/static/raw/landcover/landcover.tif", "output": "data/static/processed/landcover_region_setup.tif"},
            "ssurgo": {
                "output": "data/static/soils/ssurgo_mapunitpoly.gpkg",
                "attributes_output": "data/static/soils/ssurgo_mapunit_attributes.csv",
                "hsg_output": "data/static/soils/hsg_greensboro.tif",
                "ksat_output": "data/static/soils/ksat_mmhr_greensboro.tif",
            },
            "wflow_collection_extent": {
                "boundary": "data/static/aoi/wflow_collection_region.geojson",
                "terrain_raw": "data/wflow/static/raw/topo/dem_wflow.tif",
                "terrain_output": "data/wflow/static/processed/dem_wflow_coarse.tif",
                "terrain_resolution_degrees": 0.2,
                "landcover_raw": "data/wflow/static/raw/landcover/landcover_wflow.tif",
                "landcover_output": "data/wflow/static/processed/landcover_wflow_coarse.tif",
                "ssurgo_output": "data/wflow/static/soils/ssurgo_mapunitpoly_wflow.gpkg",
                "ssurgo_attributes_output": "data/wflow/static/soils/ssurgo_mapunit_attributes_wflow.csv",
                "hsg_output": "data/wflow/static/soils/hsg_wflow.tif",
                "ksat_output": "data/wflow/static/soils/ksat_mmhr_wflow.tif",
            },
        },
    }
    paths = {"repo_root": tmp_path, "location_name": "greensboro", "location_root": location_root}

    summary = collect_wflow_static_region_inputs(
        config,
        paths,
        fetch_dem=False,
        fetch_landcover=False,
        fetch_ssurgo=True,
        soil_polygons=soils,
        soil_attributes=attrs,
    )

    assert summary["bbox"] == wflow_boundary
    assert summary["dem"] == location_root / "data/wflow/static/processed/dem_wflow_coarse.tif"
    assert summary["landcover"] == location_root / "data/wflow/static/processed/landcover_wflow_coarse.tif"
    assert summary["hsg"] == location_root / "data/wflow/static/soils/hsg_wflow.tif"
    assert summary["ksat"] == location_root / "data/wflow/static/soils/ksat_mmhr_wflow.tif"
    assert not sfincs_dem.exists()
    with rasterio.open(summary["dem"]) as src:
        assert src.width < 10
        assert src.height < 10


def test_collect_wflow_static_region_inputs_refreshes_stale_raw_inputs_for_large_envelope(tmp_path, monkeypatch):
    location_root = tmp_path / "locations/greensboro"
    wflow_boundary = location_root / "data/static/aoi/wflow_collection_region.geojson"
    dem_raw = location_root / "data/wflow/static/raw/topo/dem_wflow.tif"
    landcover_raw = location_root / "data/wflow/static/raw/landcover/landcover_wflow.tif"
    _write_raster(dem_raw, np.ones((10, 10), dtype="float32"))
    _write_raster(landcover_raw, np.ones((10, 10), dtype="uint8"))
    wflow_boundary.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"name": ["wflow_collection"]},
        geometry=[box(-80.5, 35.0, -78.5, 36.7)],
        crs="EPSG:4326",
    ).to_file(wflow_boundary, driver="GeoJSON")
    soils = gpd.GeoDataFrame(
        {"mukey": ["1"]},
        geometry=[box(-80.5, 35.0, -78.5, 36.7)],
        crs="EPSG:4326",
    )
    attrs = __import__("pandas").DataFrame(
        {
            "mukey": ["1"],
            "hydgrp": ["B"],
            "ksat_r": [20.0],
            "hzdept_r": [0.0],
            "hzdepb_r": [40.0],
        }
    )
    config = {
        "project": {"name": "greensboro", "model_crs": "EPSG:32617", "reference_crs": "EPSG:4326"},
        "static_sources": {
            "wflow_collection_extent": {
                "boundary": "data/static/aoi/wflow_collection_region.geojson",
                "terrain_raw": "data/wflow/static/raw/topo/dem_wflow.tif",
                "terrain_output": "data/wflow/static/processed/dem_wflow_coarse.tif",
                "terrain_resolution_degrees": 0.2,
                "landcover_raw": "data/wflow/static/raw/landcover/landcover_wflow.tif",
                "landcover_output": "data/wflow/static/processed/landcover_wflow_coarse.tif",
                "ssurgo_output": "data/wflow/static/soils/ssurgo_mapunitpoly_wflow.gpkg",
                "ssurgo_attributes_output": "data/wflow/static/soils/ssurgo_mapunit_attributes_wflow.csv",
                "hsg_output": "data/wflow/static/soils/hsg_wflow.tif",
                "ksat_output": "data/wflow/static/soils/ksat_mmhr_wflow.tif",
            },
        },
    }
    paths = {"repo_root": tmp_path, "location_name": "greensboro", "location_root": location_root}
    calls = []

    def fake_fetch_dem(bbox_wgs84, output_path, **kwargs):
        calls.append(("dem", bbox_wgs84, kwargs.get("force")))
        _write_raster(output_path, np.ones((20, 20), dtype="float32"), transform=from_origin(-80.5, 36.7, 0.1, 0.1))
        return {"dem_raw": output_path, "source": "fake"}

    def fake_fetch_landcover(bbox_wgs84, output_path, **kwargs):
        calls.append(("landcover", bbox_wgs84, kwargs.get("force")))
        _write_raster(output_path, np.ones((20, 20), dtype="uint8"), transform=from_origin(-80.5, 36.7, 0.1, 0.1))
        return output_path, "fake"

    monkeypatch.setattr(static_intake, "fetch_usgs_3dep_dem", fake_fetch_dem)
    monkeypatch.setattr(static_intake, "fetch_worldcover_landcover", fake_fetch_landcover)

    summary = collect_wflow_static_region_inputs(
        config,
        paths,
        fetch_dem=True,
        fetch_landcover=True,
        fetch_ssurgo=True,
        soil_polygons=soils,
        soil_attributes=attrs,
    )

    assert calls == [
        ("dem", (-80.5, 35.0, -78.5, 36.7), True),
        ("landcover", (-80.5, 35.0, -78.5, 36.7), True),
    ]
    assert summary["dem_exists"] is True
    assert summary["landcover_exists"] is True


def test_collect_wflow_static_region_inputs_reuses_existing_coarse_landcover_when_raw_missing(tmp_path):
    location_root = tmp_path / "locations/austin"
    wflow_boundary = location_root / "data/static/aoi/wflow_collection_region.geojson"
    dem_raw = location_root / "data/wflow/static/raw/topo/dem_wflow.tif"
    landcover_output = location_root / "data/wflow/static/processed/landcover_wflow_coarse.tif"
    _write_raster(dem_raw, np.ones((20, 20), dtype="float32"), transform=from_origin(-80.5, 36.7, 0.1, 0.1))
    _write_raster(landcover_output, np.ones((20, 20), dtype="uint8"), transform=from_origin(-80.5, 36.7, 0.1, 0.1))
    wflow_boundary.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"name": ["wflow_collection"]},
        geometry=[box(-80.5, 35.0, -78.5, 36.7)],
        crs="EPSG:4326",
    ).to_file(wflow_boundary, driver="GeoJSON")
    config = {
        "project": {"name": "austin", "model_crs": "EPSG:32614", "reference_crs": "EPSG:4326"},
        "static_sources": {
            "wflow_collection_extent": {
                "boundary": "data/static/aoi/wflow_collection_region.geojson",
                "terrain_raw": "data/wflow/static/raw/topo/dem_wflow.tif",
                "terrain_output": "data/wflow/static/processed/dem_wflow_coarse.tif",
                "terrain_resolution_degrees": 0.2,
                "landcover_raw": "data/wflow/static/raw/landcover/landcover_wflow.tif",
                "landcover_output": "data/wflow/static/processed/landcover_wflow_coarse.tif",
            },
        },
    }

    summary = collect_wflow_static_region_inputs(
        config,
        {"repo_root": tmp_path, "location_name": "austin", "location_root": location_root},
        fetch_dem=False,
        fetch_landcover=False,
        fetch_ssurgo=False,
    )

    assert summary["dem"] == location_root / "data/wflow/static/processed/dem_wflow_coarse.tif"
    assert summary["landcover"] == landcover_output
    assert summary["dem_exists"] is True
    assert summary["landcover_exists"] is True


def test_collect_wflow_static_region_inputs_reuses_existing_coarse_dem_when_raw_missing(tmp_path):
    location_root = tmp_path / "locations/austin"
    wflow_boundary = location_root / "data/static/aoi/wflow_collection_region.geojson"
    dem_output = location_root / "data/wflow/static/processed/dem_wflow_coarse.tif"
    landcover_output = location_root / "data/wflow/static/processed/landcover_wflow_coarse.tif"
    _write_raster(dem_output, np.ones((20, 20), dtype="float32"), transform=from_origin(-80.5, 36.7, 0.1, 0.1))
    _write_raster(landcover_output, np.ones((20, 20), dtype="uint8"), transform=from_origin(-80.5, 36.7, 0.1, 0.1))
    wflow_boundary.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"name": ["wflow_collection"]},
        geometry=[box(-80.5, 35.0, -78.5, 36.7)],
        crs="EPSG:4326",
    ).to_file(wflow_boundary, driver="GeoJSON")
    config = {
        "project": {"name": "austin", "model_crs": "EPSG:32614", "reference_crs": "EPSG:4326"},
        "static_sources": {
            "wflow_collection_extent": {
                "boundary": "data/static/aoi/wflow_collection_region.geojson",
                "terrain_raw": "data/wflow/static/raw/topo/dem_wflow.tif",
                "terrain_output": "data/wflow/static/processed/dem_wflow_coarse.tif",
                "terrain_resolution_degrees": 0.2,
                "landcover_raw": "data/wflow/static/raw/landcover/landcover_wflow.tif",
                "landcover_output": "data/wflow/static/processed/landcover_wflow_coarse.tif",
            },
        },
    }

    summary = collect_wflow_static_region_inputs(
        config,
        {"repo_root": tmp_path, "location_name": "austin", "location_root": location_root},
        fetch_dem=False,
        fetch_landcover=False,
        fetch_ssurgo=False,
    )

    assert summary["dem"] == dem_output
    assert summary["landcover"] == landcover_output
    assert summary["dem_exists"] is True
    assert summary["landcover_exists"] is True


def test_collect_wflow_static_region_inputs_rejects_stale_raw_inputs_without_fetch(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    wflow_boundary = location_root / "data/static/aoi/wflow_collection_region.geojson"
    dem_raw = location_root / "data/wflow/static/raw/topo/dem_wflow.tif"
    landcover_raw = location_root / "data/wflow/static/raw/landcover/landcover_wflow.tif"
    _write_raster(dem_raw, np.ones((10, 10), dtype="float32"))
    _write_raster(landcover_raw, np.ones((10, 10), dtype="uint8"))
    wflow_boundary.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"name": ["wflow_collection"]},
        geometry=[box(-80.5, 35.0, -78.5, 36.7)],
        crs="EPSG:4326",
    ).to_file(wflow_boundary, driver="GeoJSON")
    config = {
        "project": {"name": "greensboro", "model_crs": "EPSG:32617", "reference_crs": "EPSG:4326"},
        "static_sources": {
            "wflow_collection_extent": {
                "boundary": "data/static/aoi/wflow_collection_region.geojson",
                "terrain_raw": "data/wflow/static/raw/topo/dem_wflow.tif",
                "landcover_raw": "data/wflow/static/raw/landcover/landcover_wflow.tif",
            },
        },
    }

    with pytest.raises(RuntimeError, match="Wflow raw DEM.*does not cover"):
        collect_wflow_static_region_inputs(
            config,
            {"repo_root": tmp_path, "location_name": "greensboro", "location_root": location_root},
            fetch_dem=False,
            fetch_landcover=False,
            fetch_ssurgo=False,
        )


def test_collect_ssurgo_infiltration_inputs_refetches_stale_polygons_for_large_bbox(tmp_path, monkeypatch):
    ssurgo_output = tmp_path / "ssurgo_mapunitpoly_wflow.gpkg"
    stale_soils = gpd.GeoDataFrame(
        {"mukey": ["1"]},
        geometry=[box(-80.0, 35.5, -79.0, 36.5)],
        crs="EPSG:4326",
    )
    stale_soils.to_file(ssurgo_output, driver="GPKG")
    setup = SimpleNamespace(
        ssurgo_output=ssurgo_output,
        ssurgo_attributes_output=tmp_path / "ssurgo_mapunit_attributes_wflow.csv",
        landcover_output=tmp_path / "landcover_wflow_coarse.tif",
        ssurgo_hsg_output=tmp_path / "hsg_wflow.tif",
        ssurgo_ksat_output=tmp_path / "ksat_mmhr_wflow.tif",
    )
    attrs = __import__("pandas").DataFrame(
        {
            "mukey": ["2"],
            "hydgrp": ["C"],
            "ksat_r": [10.0],
            "hzdept_r": [0.0],
            "hzdepb_r": [50.0],
        }
    )
    calls = []

    def fake_fetch_soils(bbox_wgs84, output_path):
        calls.append((bbox_wgs84, output_path))
        fresh = gpd.GeoDataFrame(
            {"mukey": ["2"]},
            geometry=[box(*bbox_wgs84)],
            crs="EPSG:4326",
        )
        fresh.to_file(output_path, driver="GPKG")
        return fresh

    def fake_write_infiltration(soils, attributes, landcover, *, hsg_out, ksat_out, **kwargs):
        Path(hsg_out).parent.mkdir(parents=True, exist_ok=True)
        Path(hsg_out).write_bytes(b"hsg")
        Path(ksat_out).write_bytes(b"ksat")
        return {"hsg": hsg_out, "ksat": ksat_out}

    monkeypatch.setattr(static_intake, "fetch_ssurgo_mapunit_polygons", fake_fetch_soils)
    monkeypatch.setattr(static_intake, "write_ssurgo_infiltration_rasters", fake_write_infiltration)

    summary = collect_ssurgo_infiltration_inputs(
        setup,
        (-80.5, 35.0, -78.5, 36.7),
        soil_attributes=attrs,
    )

    assert calls == [((-80.5, 35.0, -78.5, 36.7), ssurgo_output)]
    assert summary["ssurgo_polygons"] == 1
    assert gpd.read_file(ssurgo_output)["mukey"].tolist() == ["2"]
