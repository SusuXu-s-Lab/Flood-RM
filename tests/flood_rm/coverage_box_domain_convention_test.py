from pathlib import Path
import json

import geopandas as gpd
from shapely.geometry import box

import design_events.collect_sources.national_hydrography as national_hydrography
from sfincs_runs.build_base import plan_inland_sfincs_domain_set
from study_location import define_location
from wflow_runs.build_plan import plan_wflow_domain_set


repo_root = Path(__file__).resolve().parents[2]


def _location_config(location_name):
    location_root = repo_root / "locations" / location_name
    return location_root, define_location(location_root / "config.yaml").config


def _fake_wbd(monkeypatch):
    def fake_fetch(bbox_arg, *, huc_level, **kwargs):
        west, south, east, north = bbox_arg
        return gpd.GeoDataFrame(
            {"huc_id": [f"h{huc_level}"], "huc_level": [huc_level]},
            geometry=[box(west - 0.5, south - 0.5, east + 0.5, north + 0.5)],
            crs="EPSG:4326",
        )

    monkeypatch.setattr(national_hydrography, "fetch_wbd_huc", fake_fetch)


def test_greensboro_plans_one_enclosing_wflow_watershed_for_selected_sfincs_box(monkeypatch, tmp_path):
    _fake_wbd(monkeypatch)
    location_root, runtime_config = _location_config("greensboro")
    runtime_config["wflow"]["domain_set"]["huc"]["root"] = str(tmp_path / "greensboro_domain_huc")
    runtime_config["wflow"]["domain_set"]["huc"]["output"] = str(tmp_path / "greensboro_wflow_domain_huc.geojson")

    wflow_plan = plan_wflow_domain_set(runtime_config, {"location_root": location_root})
    sfincs_plan = plan_inland_sfincs_domain_set(runtime_config, {"location_root": location_root})

    assert runtime_config["wflow"]["domain_set"]["allow_multiple_submodels"] is True
    assert wflow_plan.status == "ready"
    assert [submodel["wflow_submodel_id"] for submodel in wflow_plan.submodels] == ["greensboro_rural"]
    assert runtime_config["sfincs_domain_set"]["allow_multiple_domains"] is True
    assert runtime_config["sfincs_domain_set"]["include_domain_ids"] == ["greensboro_rural"]
    assert sfincs_plan.status == "ready"
    assert sfincs_plan.domain_count == 1
    assert [domain["sfincs_domain_id"] for domain in sfincs_plan.domains] == ["greensboro_rural"]
    for domain in sfincs_plan.domains:
        assert domain["wflow_submodel_ids"] == [domain["sfincs_domain_id"]]


def test_greensboro_region_setup_watershed_artifact_keeps_selected_rural_domain():
    location_root, _ = _location_config("greensboro")

    coverage_boxes = gpd.read_file(location_root / "data/static/aoi/bbox.geojson")
    watersheds = gpd.read_file(location_root / "data/static/aoi/wflow_nhdplus_watersheds.geojson")

    assert set(coverage_boxes["subregion_id"]) == {"greensboro_rural"}
    assert len(watersheds) == 1
    assert set(watersheds["wflow_submodel_id"]) == {"greensboro_rural"}


def test_greensboro_wflow_watersheds_use_real_hucs_and_padded_collection_envelope():
    location_root, _ = _location_config("greensboro")

    watersheds = gpd.read_file(location_root / "data/static/aoi/wflow_nhdplus_watersheds.geojson")
    collection_envelope = gpd.read_file(location_root / "data/static/aoi/wflow_collection_region.geojson")

    for _, row in watersheds.iterrows():
        huc_level = int(row["huc_level"])
        huc_parts = str(row["huc_id"]).split("_")
        assert all(part.isdigit() and len(part) == huc_level for part in huc_parts)

    watershed_west, watershed_south, watershed_east, watershed_north = watersheds.total_bounds
    envelope_west, envelope_south, envelope_east, envelope_north = collection_envelope.total_bounds
    assert envelope_west < watershed_west
    assert envelope_south < watershed_south
    assert envelope_east > watershed_east
    assert envelope_north > watershed_north


def test_greensboro_region_setup_writes_crossing_gauge_points_for_selected_east_domain():
    location_root, _ = _location_config("greensboro")

    east = gpd.read_file(location_root / "data/wflow/domain_set_gauges/greensboro_east_sfincs_gauges.geojson")

    assert len(east) == 8
    assert set(east["wflow_submodel_id"]) == {"greensboro_east"}
    assert east["name"].str.startswith("greensboro_east_inflow_").all()
    assert set(east["gauge_location_source"]) == {"sfincs_stream_boundary_intersection"}


def test_greensboro_04_uses_the_coupled_build_notebook():
    location_root, _ = _location_config("greensboro")
    notebook_root = location_root / "02_flood/04"

    assert {path.name for path in notebook_root.iterdir() if path.is_file()} == {
        "a_build_coupled_model.ipynb",
        "b_prepare_wflow_dynamic_handoff.ipynb",
        "c_run_example.ipynb",
    }
    assert list(notebook_root.glob("build_greensboro_*.ipynb")) == []


def test_greensboro_region_setup_does_not_silently_skip_wflow_static_collection():
    location_root, _ = _location_config("greensboro")
    notebook = json.loads((location_root / "02_flood/01_region_setup.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    helper_text = (repo_root / "src/sfincs_runs/build_base/region_notebook.py").read_text(encoding="utf-8")

    assert "collect_required_inland_static_data(runtime)" in text
    assert "collect_wflow_static_region_inputs" in helper_text
    assert "wflow_static_ready" not in text


def test_austin_region_setup_does_not_silently_skip_wflow_static_collection():
    location_root, _ = _location_config("austin")
    notebook = json.loads((location_root / "02_flood/01_region_setup.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    helper_text = (repo_root / "src/sfincs_runs/build_base/region_notebook.py").read_text(encoding="utf-8")
    intake_text = (repo_root / "src/sfincs_runs/build_base/static_intake.py").read_text(encoding="utf-8")

    assert "collect_required_inland_static_data(runtime)" in text
    assert "allow_wflow_landcover_only_without_dem=True" not in text
    assert "collect_wflow_static_region_inputs" in helper_text
    assert "dem_input = setup.dem_raw" in intake_text
    assert "landcover_input = setup.landcover_raw" in intake_text


def test_austin_uses_one_selected_sfincs_box_and_one_enclosing_wflow_watershed(monkeypatch, tmp_path):
    _fake_wbd(monkeypatch)
    location_root, runtime_config = _location_config("austin")
    runtime_config["wflow"]["domain_set"]["huc"]["root"] = str(tmp_path / "austin_domain_huc")
    runtime_config["wflow"]["domain_set"]["huc"]["output"] = str(tmp_path / "austin_wflow_domain_huc.geojson")

    wflow_plan = plan_wflow_domain_set(runtime_config, {"location_root": location_root})
    sfincs_plan = plan_inland_sfincs_domain_set(runtime_config, {"location_root": location_root})

    assert runtime_config["wflow"]["domain_set"]["allow_multiple_submodels"] is True
    assert runtime_config["sfincs_domain_set"]["allow_multiple_domains"] is True
    assert runtime_config["sfincs_domain_set"]["include_domain_ids"] == ["austin_p4u"]
    assert runtime_config["aoi"]["source"] == "data/smart_ds/2016"
    assert wflow_plan.status == "ready"
    assert [submodel["wflow_submodel_id"] for submodel in wflow_plan.submodels] == ["austin_p4u"]
    assert sfincs_plan.status == "ready"
    assert sfincs_plan.domain_count == 1
    assert sfincs_plan.domains[0]["sfincs_domain_id"] == "austin_p4u"
    assert sfincs_plan.domains[0]["wflow_submodel_ids"] == ["austin_p4u"]


def test_austin_region_setup_artifacts_keep_selected_p4u_domain():
    location_root, _ = _location_config("austin")

    coverage_boxes = gpd.read_file(location_root / "data/static/aoi/bbox.geojson")
    watersheds = gpd.read_file(location_root / "data/static/aoi/wflow_nhdplus_watersheds.geojson")
    collection_envelope = gpd.read_file(location_root / "data/static/aoi/wflow_collection_region.geojson")

    assert set(coverage_boxes["subregion_id"]) == {"austin_p4u"}
    assert len(watersheds) == 1
    assert set(watersheds["wflow_submodel_id"]) == {"austin_p4u"}
    assert watersheds.to_crs("EPSG:5070").geometry.union_all().covers(
        coverage_boxes.to_crs("EPSG:5070").geometry.union_all()
    )

    watershed_west, watershed_south, watershed_east, watershed_north = watersheds.total_bounds
    envelope_west, envelope_south, envelope_east, envelope_north = collection_envelope.total_bounds
    assert envelope_west < watershed_west
    assert envelope_south < watershed_south
    assert envelope_east > watershed_east
    assert envelope_north > watershed_north


def test_austin_streamgage_discovery_uses_wflow_watershed_search_geometry():
    _, runtime_config = _location_config("austin")

    discovery = runtime_config["collection"]["usgs_streamgages"]["discovery"]

    assert discovery["search_geometry"] == "data/static/aoi/wflow_nhdplus_watersheds.geojson"


def test_austin_coupled_notebook_plots_manifest_selected_footprint():
    location_root, _ = _location_config("austin")
    notebook = json.loads((location_root / "02_flood/04/a_build_coupled_model.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "selected_source_subregion_ids" in text
    assert "exposure_subregion_id" in text
    assert "study_components[study_components.intersects(sfincs_coverage_geom)]" not in text


def test_austin_coupled_notebook_names_missing_wflow_static_inputs():
    location_root, _ = _location_config("austin")
    notebook = json.loads((location_root / "02_flood/04/a_build_coupled_model.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "Missing required HydroMT-Wflow source files before build:\\n" in text
    assert "missing_lines" in text
    assert "01_region_setup.ipynb" in text


def test_austin_coupled_notebook_opens_missing_sfincs_root_in_build_mode():
    location_root, _ = _location_config("austin")
    notebook = json.loads((location_root / "02_flood/04/a_build_coupled_model.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "sfincs_grid_resolution_matches(root, expected_sfincs_res_m)" in text
    assert "sfincs_model_ready = (root / \"sfincs.inp\").exists() and sfincs_model_resolution_ready" in text
    assert "sfincs_model_mode = \"w+\" if force_sfincs_domain_build or not sfincs_model_ready else \"r+\"" in text
    assert "mode=sfincs_model_mode" in text


def test_austin_coupled_notebook_forces_boundary_handoff_rebuild_without_replotting_static_ldd():
    location_root, _ = _location_config("austin")
    notebook = json.loads((location_root / "02_flood/04/a_build_coupled_model.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "force_wflow_boundary_gauge_build" not in text
    assert "force_wflow_river_build = True" in text
    assert "force=rebuild_wflow_with_boundary_handoffs" in text
    assert "### Step 16 · Final Wflow gauge QA" in text
    assert "\"wflow_ldd_components.png\"" not in text
    assert "filled Wflow basin and SFINCS coverage areas" in text
    assert "plot_sfincs_handoff_basemap(" in text


def test_austin_collect_sources_notebook_exposes_refresh_switch_and_reuses_skip_policy():
    location_root, _ = _location_config("austin")
    notebook = json.loads((location_root / "02_flood/02_collect_sources.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "force_source_collection_refresh = False" in text
    assert "refresh_wflow_hydrography_only = False" in text
    assert "collect.refresh_wflow_hydrography_basemap(runtime, force=True)" in text
    assert "skip_existing=source_skip_existing" in text


def test_greensboro_coupled_notebook_opens_missing_sfincs_root_in_build_mode():
    location_root, _ = _location_config("greensboro")
    notebook = json.loads((location_root / "02_flood/04/a_build_coupled_model.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "sfincs_grid_resolution_matches(root, expected_sfincs_res_m)" in text
    assert "sfincs_model_ready = (root / \"sfincs.inp\").exists() and sfincs_model_resolution_ready" in text
    assert "sfincs_model_mode = \"w+\" if force_sfincs_domain_build or not sfincs_model_ready else \"r+\"" in text
    assert "mode=sfincs_model_mode" in text


def test_austin_coupled_notebook_labels_native_wflow_handoff_reduction():
    location_root, _ = _location_config("austin")
    notebook = json.loads((location_root / "02_flood/04/a_build_coupled_model.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "planned_boundary_crossings" in text
    assert "native_snapped_wflow_gauges" in text
