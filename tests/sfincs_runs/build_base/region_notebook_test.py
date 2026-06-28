from pathlib import Path
from types import SimpleNamespace

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
