from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
from shapely.geometry import box

from sfincs_runs.build_base import region_notebook as region


def test_load_runtime_exposes_static_source_defaults_for_thin_inland_config():
    runtime = region.load_runtime(Path("locations/greensboro"))

    assert runtime.static_sources["terrain"]["raw"] == "data/static/raw/topo/dem.tif"
    assert runtime.static_sources["landcover"]["output"] == "data/static/processed/landcover_region_setup.tif"
    assert runtime.static_sources["ssurgo"]["output"] == "data/static/soils/ssurgo_mapunitpoly.gpkg"
    assert runtime.static_sources["ssurgo"]["hsg_output"] == "data/static/soils/hsg_greensboro.tif"
    assert runtime.static_sources["wflow_collection_extent"]["watersheds"] == (
        "data/static/aoi/wflow_nhdplus_watersheds.geojson"
    )
    assert runtime.config["paths"]["data_catalog"] == "data/static/data_catalogue.yaml"


def test_plot_domains_dispatches_without_recursing(monkeypatch, tmp_path):
    monkeypatch.setattr(region, "_plot_inland_domains", lambda runtime, domains: "inland")
    monkeypatch.setattr(region, "_plot_coastal_domains", lambda runtime, domains: "coastal")

    coastal_runtime = region.CoastalRegionSetupRuntime(
        location_root=tmp_path,
        location_name="test",
        repo_root=tmp_path,
        config={},
        paths={},
        region_setup=SimpleNamespace(coastal_region_output=tmp_path / "missing.geojson"),
        collect_static_inputs=False,
        fetch_dem=False,
        fetch_landcover=False,
        fetch_ssurgo=False,
    )

    assert region.plot_domains(object(), object()) == "inland"
    assert region.plot_domains(coastal_runtime, object()) == "coastal"


def test_build_smart_ds_evaluation_footprint_reuses_existing_study_area(monkeypatch, tmp_path):
    location_root = tmp_path / "locations" / "test"
    study_area_path = location_root / "data/static/aoi/study_area.geojson"
    study_area_path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"name": ["study_area"]},
        geometry=[box(-1.0, -1.0, 1.0, 1.0)],
        crs="EPSG:4326",
    ).to_file(study_area_path, driver="GeoJSON")

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("existing study area should be reused")

    monkeypatch.setattr(region, "_build_aoi", fail_rebuild)
    config = {
        "project": {"name": "test"},
        "aoi": {
            "source": "data/smart_ds/2016",
            "source_format": "smart_ds_buscoords",
            "output": "data/static/aoi/study_area.geojson",
        },
        "smart_ds_evaluation_footprint": {
            "output": "data/static/aoi/evaluation_footprint.geojson",
            "minimum_flood_coverage": True,
        },
    }
    runtime = region.RegionSetupNotebookRuntime(
        location_root=location_root,
        location_name="test",
        repo_root=tmp_path,
        runtime_config=config,
        config=config,
        grid_config=config,
        data_sources={"static_sources": {}},
        sfincs_config={},
        wflow_config={"wflow": {}},
        region_setup=SimpleNamespace(),
        collect_static_inputs=False,
        fetch_dem=False,
        fetch_landcover=False,
        fetch_ssurgo=False,
        fetch_nhdplus=False,
    )

    footprint = region.build_smart_ds_evaluation_footprint(runtime)

    assert footprint.aoi_result.output_path == study_area_path
    assert footprint.summary["source_format"] == "smart_ds_buscoords"
    assert footprint.evaluation_output_path.exists()
