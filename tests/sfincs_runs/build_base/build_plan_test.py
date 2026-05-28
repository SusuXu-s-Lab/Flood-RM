from pathlib import Path

from sfincs_runs.build_base import build_baseline_build_plan, build_static_intake_plan
from sfincs_runs.config import load_runtime


def suffix(path):
    return Path(path).relative_to(Path(__file__).resolve().parents[3]).as_posix()


def test_baseline_build_plan_selects_wave_coupled_notebook_for_marshfield(tmp_path):
    config = {
        "project": {"name": "marshfield"},
        "coastal_waves": True,
        "grid_footprint": {"source": "data/static/aoi/study_area.geojson"},
        "notebooks": {
            "build_sfincs": "02_flood/04/a_build_standard.ipynb",
            "build_sfincs_wave_coupled": "02_flood/04/a_build_waves.ipynb",
        },
    }
    paths = {
        "root": tmp_path / "locations/marshfield/data/sfincs",
        "base_model_root": tmp_path / "locations/marshfield/data/sfincs/base",
        "data_catalog": tmp_path / "locations/marshfield/data/static/data_catalogue.yaml",
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "location_root": tmp_path / "locations/marshfield",
    }

    plan = build_baseline_build_plan(config, paths)

    assert plan.study_location == "marshfield"
    assert plan.build_kind == "wave_coupled"
    assert plan.truth_set_kind == "wave_coupled_truth_set"
    assert plan.notebook_path == tmp_path / "locations/marshfield/02_flood/04/a_build_waves.ipynb"
    assert plan.grid_footprint_source == tmp_path / "locations/marshfield/data/static/aoi/study_area.geojson"
    assert plan.required_sources == ("era5_waves",)


def test_baseline_build_plan_selects_regular_notebook_without_coastal_waves(tmp_path):
    config = {
        "project": {"name": "austin"},
        "coastal_waves": False,
        "grid_footprint": {"source": "grid_footprint.geojson"},
    }
    paths = {
        "root": tmp_path / "locations/austin/data/sfincs",
        "base_model_root": tmp_path / "locations/austin/data/sfincs/base",
        "data_catalog": tmp_path / "locations/austin/data/static/data_catalogue.yaml",
        "repo_root": tmp_path,
        "location_name": "austin",
        "location_root": tmp_path / "locations/austin",
    }

    plan = build_baseline_build_plan(config, paths)

    assert plan.build_kind == "regular_grid"
    assert plan.truth_set_kind == "hydrodynamic_truth_set"
    assert plan.notebook_path == tmp_path / "locations/austin/02_flood/04/a_build_standard.ipynb"
    assert plan.grid_footprint_source == tmp_path / "locations/austin/grid_footprint.geojson"
    assert plan.required_sources == ()


def test_baseline_build_plan_uses_marshfield_runtime_config():
    config, paths = load_runtime("locations/marshfield/config.yaml")

    plan = build_baseline_build_plan(config, paths)

    assert plan.study_location == "marshfield"
    assert plan.build_kind == "wave_coupled"
    assert suffix(plan.notebook_path) == "locations/marshfield/02_flood/04/a_build_waves.ipynb"
    assert suffix(plan.base_model_root) == "locations/marshfield/data/sfincs/base"
    assert suffix(plan.grid_footprint_source) == "locations/marshfield/data/static/aoi/study_area.geojson"


def test_static_intake_plan_uses_location_data_workspace(tmp_path):
    config = {
        "project": {"name": "marshfield", "model_crs": "EPSG:26919"},
        "grid_footprint": {"source": "data/static/aoi/study_area.geojson"},
    }
    paths = {
        "location_name": "marshfield",
        "location_root": tmp_path / "locations/marshfield",
        "static_root": tmp_path / "locations/marshfield/data/static/processed",
        "raw_root": tmp_path / "locations/marshfield/data/static/raw",
        "data_catalog": tmp_path / "locations/marshfield/data/static/data_catalogue.yaml",
    }

    plan = build_static_intake_plan(config, paths)

    assert plan.study_location == "marshfield"
    assert plan.model_crs == "EPSG:26919"
    assert plan.static_root == tmp_path / "locations/marshfield/data/static/processed"
    assert plan.raw_root == tmp_path / "locations/marshfield/data/static/raw"
    assert plan.data_catalog == tmp_path / "locations/marshfield/data/static/data_catalogue.yaml"
    assert plan.required_static_inputs == ("terrain", "landcover", "coastline", "ssurgo")
