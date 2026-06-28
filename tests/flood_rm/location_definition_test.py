import json
from pathlib import Path

import pytest

from study_location import define_location


def suffix(path):
    return Path(path).relative_to(Path(__file__).resolve().parents[2]).as_posix()


repo_root = Path(__file__).resolve().parents[2]
greensboro_flood = repo_root / "locations/greensboro/02_flood"


def notebook_text(relative_path):
    notebook = json.loads((greensboro_flood / relative_path).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_define_location_reads_split_yaml():
    definition = define_location("locations/marshfield/config.yaml")

    assert definition.name == "marshfield"
    assert suffix(definition.root) == "locations/marshfield"
    assert definition.config["project"]["name"] == "marshfield"
    assert definition.config["includes"] == {
        "grid": "grid.yaml",
        "sfincs": "sfincs.yaml",
        "snapwave": "snapwave.yaml",
    }
    assert (definition.root / "grid.yaml").exists()
    assert (definition.root / "sfincs.yaml").exists()
    assert (definition.root / "snapwave.yaml").exists()
    assert not (definition.root / "data_sources.yaml").exists()
    assert not (definition.root / "sfincs_build.yml").exists()
    assert definition.config["collection"]["start"] == "1979-02-01"
    assert definition.data_sources == {}
    assert definition.grid["asset_registry"] == "data/power_grid/asset_registry"
    assert definition.config["static_sources"]["terrain"]["raw"] == "data/static/raw/topo/cudem_10_marshfield.tif"
    assert definition.config["coastal_wave_coupling"]["quadtree"]["base_model_root"] == "data/sfincs/base_quadtree_snapwave"
    assert "notebooks" not in definition.config
    assert "setup_grid_from_region" in definition.model_recipes["sfincs_build"]
    assert "setup_wave_forcing" in definition.model_recipes["snapwave_update_forcing"]


def test_define_location_requires_grid_and_sfincs_includes(tmp_path):
    location = tmp_path / "locations" / "demo"
    location.mkdir(parents=True)
    config = location / "config.yaml"
    config.write_text("project:\n  name: demo\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must include grid.yaml and sfincs.yaml"):
        define_location(config)


def test_define_location_reads_model_yaml_hydromt_recipes(tmp_path):
    location = tmp_path / "locations" / "austin"
    location.mkdir(parents=True)
    (location / "config.yaml").write_text(
        "\n".join(
            [
                "project:",
                "  name: austin",
                "includes:",
                "  smartds: smartds.yaml",
                "  sfincs: sfincs.yaml",
                "  wflow: wflow.yaml",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (location / "smartds.yaml").write_text("grid:\n  asset_registry: data/power_grid/asset_registry\n", encoding="utf-8")
    (location / "sfincs.yaml").write_text(
        "\n".join(
            [
                "sfincs:",
                "  boundary_file: data/sfincs/base/sfincs.bnd",
                "  build_config: data/sfincs/config/sfincs_build.yml",
                "  update_forcing_config: data/sfincs/config/sfincs_update_forcing.yml",
                "hydromt:",
                "  build:",
                "    setup_grid_from_region:",
                "      res: 100",
                "  update_forcing:",
                "    setup_config:",
                "      tref: 20000101 000000",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (location / "wflow.yaml").write_text(
        "\n".join(
            [
                "wflow:",
                "  enabled: true",
                "  build_config: data/wflow/config/wflow_build.yml",
                "  update_forcing_config: data/wflow/config/wflow_update_forcing.yml",
                "hydromt:",
                "  build:",
                "    steps:",
                "      - setup_basemaps:",
                "          region:",
                "            bbox: [0, 0, 1, 1]",
                "  update_forcing:",
                "    steps:",
                "      - setup_config:",
                "          data: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    definition = define_location(location / "config.yaml")

    assert definition.config["wflow"]["build_config"] == "data/wflow/config/wflow_build.yml"
    assert definition.config["wflow"]["update_forcing_config"] == "data/wflow/config/wflow_update_forcing.yml"
    assert definition.model_recipes["wflow_build"]["steps"][0]["setup_basemaps"]["region"]["bbox"] == [0, 0, 1, 1]


def test_austin_collects_nhdplus_review_vectors_for_stream_boundary_handoff():
    definition = define_location("locations/austin/config.yaml")
    national_hydrography = definition.config["collection"]["national_hydrography"]

    assert national_hydrography["collect_review_vectors"] is True
    assert national_hydrography["basemap_source_resolution_degrees"] == pytest.approx(1 / 1080)
    assert national_hydrography["river_geometry"] == "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg"
    assert national_hydrography["catchments"] == "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg"


def test_greensboro_flood_notebook_setup_matches_inland_wflow_sequence():
    assert [path.name for path in sorted(greensboro_flood.glob("[0-9][0-9]_*.ipynb"))] == [
        "01_region_setup.ipynb",
        "02_collect_sources.ipynb",
        "03_build_event_catalog.ipynb",
        "05_create_scenarios.ipynb",
        "06_evaluate.ipynb",
    ]
    assert [path.name for path in sorted((greensboro_flood / "04").glob("*.ipynb"))] == [
        "a_build_coupled_model.ipynb",
        "b_prepare_wflow_dynamic_handoff.ipynb",
        "c_run_example.ipynb",
    ]


def test_greensboro_flood_notebooks_have_stage_contracts_and_compile():
    notebooks = sorted(greensboro_flood.glob("[0-9][0-9]_*.ipynb"))
    notebooks += sorted((greensboro_flood / "04").glob("*.ipynb"))

    assert notebooks
    for notebook_path in notebooks:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "Stage Contract" in text, notebook_path
        assert "Requires:" in text, notebook_path
        assert "Produces:" in text, notebook_path
        assert "Next:" in text, notebook_path
        if notebook_path.parent.name == "04":
            assert "load_runtime(Path(\"../..\").resolve()" in text, notebook_path
            assert "load_coupled_runtime" not in text, notebook_path
        else:
            assert "location_root" in text, notebook_path

        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"{notebook_path}:cell{index}", "exec")


def test_greensboro_notebooks_include_inland_wflow_components():
    expected_sections = {
        "01_region_setup.ipynb": [
            "## SMART-DS AOI and Evaluation Footprint",
            "## SFINCS and Wflow Terrain/Landcover Inputs",
            "## Greensboro Region Setup",
        ],
        "02_collect_sources.ipynb": [
            "## Source Collection Plan",
            "## USGS Active Streamgage Discovery",
            "## Reviewed Streamgage Network Writer",
            "## Direct AORC SST Rainfall Members",
            "## NWM Soil-Moisture Context",
            "## Run Collection",
            "## Collected Data Overview",
        ],
        "03_build_event_catalog.ipynb": [
            "## Design Event Catalogue Parameters",
            "## Stage 1 - Source inventory and pairing policy",
            "## Stage 2 - Streamflow validation anchor (Primary Reference Gage)",
            "## Stage 3 - Rainfall-driven design catalog",
            "## Inland Storm Timing Descriptors",
            "## Stage 7 - Wflow readiness replay set",
            "## Stage 8 - Hand off to Wflow and SFINCS",
        ],
        "04/a_build_coupled_model.ipynb": [
            "## Part 1 — Coupled Domain Plans",
            "### Step 1 · Wflow watershed and SFINCS coverage plans",
            "### Step 2 · Hydrologic boundary and handoff map",
            "## Part 2 — HydroMT-Wflow Native Build",
            "### Step 3 · HydroMT-Wflow data catalog readiness",
            "### Step 4 · Wflow build steps",
            "### Step 5 · Build or reuse Wflow submodels",
            "### Step 8 · Wflow LDD component plots",
            "## Part 3 — HydroMT-SFINCS Coverage Build",
            "### Step 9 · Initialize SFINCS coverage models",
            "### Step 10 · SFINCS build configuration",
            "### Step 11 · Build or reuse SFINCS grids, masks, subgrid, and Wflow source points",
            "### Step 13 · SFINCS handoff source table",
            "## Part 4 — Coupling Back Into Wflow",
            "### Step 14 · Rebuild Wflow gauges at SFINCS boundary sources",
            "### Step 15 · Final coupled Wflow basemaps",
            "### Step 16 · Final Wflow gauge QA",
        ],
        "04/c_run_example.ipynb": [
            "## 1 - Select a catalog event and verify dynamic Wflow handoff",
            "## 2 - Configure SFINCS run",
            "## 3 - Stage SFINCS and apply rainfall + dynamic Wflow discharge",
            "## 4 - Pre-run forcing QA",
            "## 5 - Run SFINCS",
            "## 6 - Flood + discharge animation (mp4)",
            "## 7 - Post-run diagnostics",
        ],
        "04/b_prepare_wflow_dynamic_handoff.ipynb": [
            "## 1 - Select Event and Plan Handoff",
            "## 2 - Verify Wflow Source Geometry and Static Maps",
            "## 3 - Collect Shared Warmup AORC Forcing",
            "## 4 - Stage Wflow Event Meteo",
            "## 5 - Run Dynamic Wflow Handoff QA",
            "## 6 - Acceptance Artifact",
        ],
        "05_create_scenarios.ipynb": [
            "## Package Worklist",
            "## Sync Handoff",
            "## Cluster Launch Plan",
            "## Sample Driver QA",
        ],
        "06_evaluate.ipynb": [
            "## Stats Inputs",
            "## Wflow Readiness",
            "## Completed Events",
            "## Multi-Domain Depth Merge",
            "## SMART-DS Asset Evaluation",
            "## Alignment QA",
            "## Write Notebook Outputs",
        ],
    }

    for relative_path, headings in expected_sections.items():
        text = notebook_text(relative_path)
        for heading in headings:
            assert heading in text, f"{heading} missing from {relative_path}"


def test_greensboro_notebooks_reference_usgs_wflow_and_sfincs_handoff():
    collect_text = notebook_text("02_collect_sources.ipynb")
    combined_text = notebook_text("04/a_build_coupled_model.ipynb")
    build_text = combined_text

    assert "usgs_streamgages" in collect_text
    assert "discover_active_streamgage_candidates" in collect_text
    assert "write_reviewed_streamgage_network" in collect_text
    assert "reviewed streamgage network" in collect_text
    assert "build_wflow_submodel" in build_text
    assert "wflow_source_readiness" in build_text
    assert "handoff_contract" in build_text
    assert "build_domains" not in build_text
    assert "run_sfincs_domain_build = True" in combined_text
    assert "force_sfincs_domain_build = rerun" in combined_text
    assert "plan_inland_sfincs_base(config, paths)" in build_text
    assert "plan_inland_sfincs_domain_set(config, paths)" in build_text
    assert "write_inland_sfincs_domain_set_manifest(sfincs_domain_plan, config, paths)" in build_text
    assert "create_handoffs(" in build_text
    assert "from hydromt_sfincs import DATADIR, SfincsModel" in combined_text
    assert "sf = SfincsModel(" in combined_text
    assert "setup_grid_from_region" in combined_text
    assert "setup_dep" in combined_text
    assert "datasets_rgh" in combined_text
    assert "setup_subgrid" in combined_text
    assert "plot_sfincs_handoff_basemap(" in combined_text
    assert "plot_inland_sfincs_domain_set_basemaps(" not in combined_text
    assert "sf.write()" in build_text


def test_greensboro_notebooks_run_requested_inland_build_actions():
    build_text = "\n".join(
        [
            notebook_text("04/a_build_coupled_model.ipynb"),
        ]
    )
    assert "build_wflow = True" in build_text
    assert "build_sfincs_base = False" not in build_text
    assert "base_build_summary = build_inland_sfincs_base(" not in build_text
    assert "run_sfincs_domain_build = True" in build_text
    assert "build_domains(" not in build_text
    assert "sf = SfincsModel(" in build_text

    collect_text = notebook_text("02_collect_sources.ipynb")
    assert "soil_moisture_ready = soil_moisture_csv_has_variables(soil_moisture_path, soil_variables)" in collect_text
    assert "collect_missing_sources = force_source_collection_refresh or not collectable_readiness[\"ready\"].all()" in collect_text
    assert "run_collection=collect_missing_sources" in collect_text
    assert "write_reviewed_streamgage_network_file = True" in collect_text
    assert "reviewed_network_result = write_reviewed_streamgage_network(" in collect_text
    assert 'os.environ.pop("DEBUG", None)' in build_text


def test_greensboro_notebooks_do_not_retain_coastal_event_workflow():
    banned_terms_by_notebook = {
        "02_collect_sources.ipynb": [
            "CORA Coastal Water Level",
            "ERA5 Ocean Waves",
            "SnapWave",
            "collect_cora",
            "collect_era5_waves",
            "total offshore water level",
        ],
        "03_build_event_catalog.ipynb": [
            "CORA",
            "ERA5",
            "SnapWave",
            "coastal POT",
            "offshore",
            "template_assignment",
            "tail_morph",
            "load_hourly_waterlevel",
            "SLR",
        ],
        "04/a_build_coupled_model.ipynb": [
            "Coastal boundary",
            "setup_waterlevel_forcing",
        ],
        "05_create_scenarios.ipynb": [
            "SnapWave",
            "surge",
            "include-waves",
            "coastal_wave_coupling",
            "snapwave",
        ],
        "06_evaluate.ipynb": [
            "SLR",
            "tail_morph",
        ],
    }

    for relative_path, banned_terms in banned_terms_by_notebook.items():
        text = notebook_text(relative_path)
        for banned_term in banned_terms:
            assert banned_term not in text, f"{banned_term!r} retained in {relative_path}"


def test_greensboro_notebooks_are_clean_copies_without_outputs_or_execution_counts():
    notebooks = sorted(greensboro_flood.glob("[0-9][0-9]_*.ipynb"))
    notebooks += sorted((greensboro_flood / "04").glob("*.ipynb"))

    for notebook_path in notebooks:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            assert cell.get("outputs", []) == [], f"{notebook_path}:cell{index} has stale outputs"
            assert cell.get("execution_count") is None, f"{notebook_path}:cell{index} has stale execution count"
