from pathlib import Path
import json


repo_root = Path(__file__).resolve().parents[2]

# The 04 notebooks live in a subdirectory. Build notebooks are lightweight
# base-model runbooks; example notebooks own the Single-Use-Case Test Plan.
flood_workflow_notebooks = [
    "01_region_setup.ipynb",
    "02_collect_sources.ipynb",
    "03_build_event_catalog.ipynb",
    "05_create_scenarios.ipynb",
    "06_evaluate.ipynb",
]

flood_workflow_04_notebooks = [
    "a_build_standard.ipynb",
    "a_build_waves.ipynb",
    "b_example_standard.ipynb",
    "b_example_waves.ipynb",
]

grid_workflow_notebooks = [
    "01_base_network.ipynb",
    "03_audit_network.ipynb",
]


def test_template_location_workspace_has_been_removed():
    assert not (repo_root / "locations/template").exists()


def test_marshfield_location_workspace_has_stage_oriented_data_folders():
    workspace = repo_root / "locations/marshfield"

    assert (workspace / "config.yaml").exists()
    assert [path.name for path in sorted((workspace / "01_grid").glob("[0-9][0-9]_*.ipynb"))] == grid_workflow_notebooks
    assert [path.name for path in sorted((workspace / "02_flood").glob("[0-9][0-9]_*.ipynb"))] == flood_workflow_notebooks
    assert [path.name for path in sorted((workspace / "02_flood/04").glob("*.ipynb"))] == flood_workflow_04_notebooks
    assert not (workspace / "notebooks_grid").exists()
    assert not (workspace / "notebooks_flood").exists()
    assert sorted(path.name for path in (workspace / "data").iterdir() if path.is_dir()) == [
        "cache",
        "evaluation",
        "event_catalog",
        "power_grid",
        "sfincs",
        "sources",
        "static",
    ]
    assert not (workspace / "01_grid/cache").exists()
    assert not (workspace / "02_flood/cache").exists()
    assert (workspace / "data/cache/01_grid/notebook_cache").exists()
    assert (workspace / "data/cache/02_flood/notebook_cache").exists()


def test_marshfield_notebooks_have_stage_contract_cards():
    notebooks = sorted((repo_root / "locations/marshfield/01_grid").rglob("*.ipynb"))
    notebooks += sorted((repo_root / "locations/marshfield/02_flood").rglob("*.ipynb"))

    assert notebooks
    for notebook_path in notebooks:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "Stage Contract" in text, notebook_path
        assert "Requires:" in text, notebook_path
        assert "Produces:" in text, notebook_path
        assert "Next:" in text, notebook_path


def test_old_top_level_workflow_folders_are_drained():
    assert not (repo_root / "notebooks").exists()
    assert not (repo_root / "scripts").exists()
    assert not (repo_root / "design_events").exists()
    assert not (repo_root / "sfincs_runs").exists()
    assert not (repo_root / "reference_data").exists()
    assert not (repo_root / "tools").exists()


def test_reference_and_tooling_files_have_domain_homes():
    assert (repo_root / "locations/marshfield/data/sources/aorc_sst/transposition_regions/transposition_region_100km.geojson").exists()
    assert (repo_root / "locations/marshfield/data/evaluation/reference_media/riley_90m_flood_ocean_animation.mp4").exists()
    assert (repo_root / "locations/marshfield/data/sfincs/reports/scenario_build_report.csv").exists()
    assert (repo_root / "locations/marshfield/data/static/aoi/study_area.geojson").exists()
    assert (repo_root / "locations/marshfield/data/power_grid/asset_registry/buses.csv").exists()
    assert (repo_root / "locations/marshfield/data/power_grid/augmented/assets.parquet").exists()
    assert (repo_root / "src/design_events/collect_sources/fetch_era5_waves.py").exists()
    assert (repo_root / "src/design_events/collect_sources/fetch_ssurgo.py").exists()
    assert (repo_root / "cluster/run_sfincs_dsai_wave_coupled.slurm").exists()


def test_workflow_packages_do_not_nest_an_extra_pipeline_folder():
    assert not (repo_root / "src/design_events/pipeline").exists()
    assert not (repo_root / "src/sfincs_runs/pipeline").exists()


def test_marshfield_notebooks_preserve_location_specific_workflow_content():
    notebooks = repo_root / "locations/marshfield/02_flood"
    expected = {
        "01_region_setup.ipynb": ("# 01 Region Setup", 10),
        "02_collect_sources.ipynb": ("# 02 Collect Sources", 8),
        "03_build_event_catalog.ipynb": ("# Build Event Catalog", 30),
        "04/a_build_standard.ipynb": ("# Build SFINCS Model", 15),
        "04/a_build_waves.ipynb": ("# Build SFINCS Model", 15),
        "04/b_example_standard.ipynb": ("# Example SFINCS Run", 10),
        "04/b_example_waves.ipynb": ("# Example SFINCS Run", 10),
        "05_create_scenarios.ipynb": ("# 05 Create SFINCS Scenarios", 6),
        "06_evaluate.ipynb": ("# SFINCS Scenario Stats", 18),
    }

    for filename, (heading, min_cells) in expected.items():
        notebook = json.loads((notebooks / filename).read_text(encoding="utf-8"))
        text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert heading in text
        assert len(notebook["cells"]) >= min_cells


def test_marshfield_event_catalog_notebook_tells_staged_catalog_story():
    notebook = json.loads(
        (repo_root / "locations/marshfield/02_flood/03_build_event_catalog.ipynb").read_text(
            encoding="utf-8"
        )
    )
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    stages = [
        "## Stage 1 — Source inventory",
        "## Stage 2 — Driver libraries and marginal evidence",
        "## Stage 3 — Coastal-driver sampling and benchmark design slices",
        "## Stage 4 — Build event hydrographs",
        "## Stage 5 — Compound forcing pairing",
        "## Stage 6 — Probability catalog and resilience stress/training set",
        "## Stage 7 — SLR scenario comparison",
        "## Stage 8 — Hand off to SFINCS",
    ]
    positions = [text.index(stage) for stage in stages]

    assert positions == sorted(positions)
    assert "source_inventory_frame" in text
    assert "plot_rainfall_member_distribution" in text
    assert "wave_analog_diagnostics" in text
    assert "forcing_selection_frame" in text
    assert "plot_forcing_marginal_comparison" in text
    assert "Coastal Driver Return Period" in text
    assert "probability_weight" in text
    assert "Probability Catalog" in text
    assert "Resilience Stress/Training Set" in text
    assert "select_resilience_stress_training_set" in text
    assert "plot_return_period_benchmark_coverage" in text
    assert "plot_catalog_set_severity_comparison" in text
    assert "coastal_driver_return_period_range" in text
    assert "0.2% annual-chance" in text
    assert "plot_distinct_oscillatory_proxies" in text
    assert "plot_msl_shift_scenario_comparison" in text
    assert "notebook-only" not in text


def test_marshfield_region_setup_notebook_uses_modular_bbox_not_hand_mesh():
    notebook = json.loads(
        (repo_root / "locations/marshfield/02_flood/01_region_setup.ipynb").read_text(
            encoding="utf-8"
        )
    )
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"01_region_setup.ipynb:cell{index}", "exec")

    assert "bbox_geom_wgs = box(*study_area_bbox(config, repo_root, buffer_degrees=0.01))" in text
    assert "bbox_wgs84 = tuple(float(v) for v in bbox_gdf.total_bounds)" in text
    assert "def fetch_worldcover_landcover" in text
    assert "esa-worldcover" in text
    assert "if region_setup.landcover_raw.exists()" in text
    assert "fetch_ssurgo_mapunit_polygons" in text
    assert "normalize_ssurgo_axis_order" in text
    assert "features_from_bbox" in text
    assert "rio.clip_box" in text
    assert "raw coastline" in text
    assert "coastal land mask" in text
    assert "SSURGO map units" in text
    assert "write_ssurgo_infiltration_rasters" in text
    assert "hsg_summary" in text
    assert "hsg_abcd" in text
    assert "LogNorm" in text
    assert "ksat_positive" in text
    assert "ssurgo_infiltration_rasters" in text
    assert "ListedColormap" in text
    assert "top_soil_types" not in text
    assert '"other"' not in text
    assert "study_area.to_crs(dem.rio.crs).boundary.plot" not in text
    assert "study_area.to_crs(landcover_clip.rio.crs).boundary.plot" not in text
    assert "study_area.boundary.plot(ax=soils_ax" not in text
    assert 'label="study area"' not in text
    assert "coastal_region.boundary.plot(ax=ax" in text
    assert "Patch(facecolor=\"none\", edgecolor=\"#00a884\", label=\"coastal land mask\")" in text
    assert "## Raster Detail" not in text
    assert "fig, dem_ax = plt.subplots" in text
    assert "fig, landcover_ax = plt.subplots" in text
    assert "fig, axes = plt.subplots(1, 2" in text
    assert "plt.subplots(1, 4" not in text
    assert "load_unstructured_mesh_from_hdf" not in text
    assert "mfield_mesh_v2.hdf" not in text


def test_marshfield_collect_sources_notebook_is_collection_ready():
    notebook = json.loads(
        (repo_root / "locations/marshfield/02_flood/02_collect_sources.ipynb").read_text(
            encoding="utf-8"
        )
    )
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"02_collect_sources.ipynb:cell{index}", "exec")

    assert "run_collection = True" in text
    assert "from tqdm.auto import tqdm" not in text
    assert "import importlib" in text
    assert "source_plan_module = importlib.reload(source_plan_module)" in text
    assert "run_collect_module = importlib.reload(run_collect_module)" in text
    assert "tqdm(plan.steps" not in text
    assert "def _record(collection_rows, source, status, started, **details):" not in text
    assert "\"rows\": rows" not in text
    assert "run_collect(" in text
    assert "collect_aorc_sst(settings" not in text
    assert "SMOKE = True" not in text
    assert "source_skip_existing = True" in text
    assert "skip_existing=source_skip_existing" in text
    assert "reused_smoke" not in text
    assert "will_reuse_existing" in text
    assert "source_artifact_covers(paths, *source_artifacts[step.name], step.start, step.end)" in text
    assert "wave_dataset_covers(output_path, step.start, step.end)" in text
    assert "\"nwm\": (\"nwm\", \"retrospective_hydrologic_state\")" in text
    assert "paths[\"aorc_sst_rainfall_members_csv\"]" in text
    assert "audit.get(\"gates\", [])" in text
    assert "## Collected Data Overview" in text
    assert "Collected source geography" in text
    assert "rainfall transposition targets" in text
    assert "72h rainfall magnitude" in text
    assert "ERA5 wave bbox" not in text
    assert "CORA boundary points" not in text
    assert "NWM soil points" not in text
    assert "CORA boundary water level" in text
    assert "NWM soil moisture" in text
    assert "AORC SST rainfall member maxima" in text
    assert "ERA5 {name}" in text
    assert "audit.get(\"checks\", [])" not in text
    assert "collect_all_sources" not in text
    assert "cli_equivalent" not in text
    assert "python -m design_events" not in text


def test_marshfield_collect_sources_rainfall_plot_accepts_current_aorc_columns():
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    notebook_path = repo_root / "locations/marshfield/02_flood/02_collect_sources.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    helper_source = next(source for source in cells if "def _first_column" in source)
    rainfall_plot_source = next(source for source in cells if "# Plot compact AORC SST rainfall member summaries." in source)

    namespace = {
        "Path": Path,
        "paths": {
            "aorc_sst_rainfall_members_csv": repo_root / "locations/marshfield/data/sources/aorc_sst/rainfall_members.csv"
        },
        "pd": pd,
    }

    exec(helper_source, namespace)
    namespace["plt"].show = lambda: None
    exec(rainfall_plot_source, namespace)

    assert namespace["max_column"] == "max_precip_mm"
    assert namespace["mean_column"] == "mean_precip_mm"
    assert namespace["unit"] == "mm"
    plt.close("all")


def test_marshfield_collect_sources_notebook_contains_sst_region_plot():
    notebooks = repo_root / "locations/marshfield/02_flood"
    notebook = json.loads((notebooks / "02_collect_sources.ipynb").read_text(encoding="utf-8"))
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert not (notebooks / "sst_region.ipynb").exists()
    assert "## Stochastic Storm Transposition Region" in text
    assert "The SST region is the configured AORC SST transposition polygon" in text
    assert "def plot_sst_region" in text
    assert "plot_sst_region(config, paths)" in text
    assert "config[\"collection\"][\"aorc_sst\"][\"transposition_region\"][\"geometry_file\"]" in text
    assert "ctx.add_basemap" in text


def test_marshfield_event_catalog_notebook_writes_catalog_audit():
    notebook = json.loads(
        (repo_root / "locations/marshfield/02_flood/03_build_event_catalog.ipynb").read_text(
            encoding="utf-8"
        )
    )
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"03_build_event_catalog.ipynb:cell{index}", "exec")

    assert "from design_events.build_events.event_catalog import build_event_catalog" in text
    assert "event_catalog = build_event_catalog(config, paths)" in text
    assert "paths[\"event_catalog_audit_json\"]" in text


def test_marshfield_config_is_location_definition_not_notebook_index():
    config = (repo_root / "locations/marshfield/config.yaml").read_text(encoding="utf-8")

    assert "notebooks:" not in config
    assert "data_sources: data_sources.yaml" in config
    assert "grid: grid.yaml" in config
    assert "sfincs: sfincs.yaml" in config


def test_marshfield_augment_network_notebooks_use_current_artifact_apis():
    notebooks = repo_root / "locations/marshfield/01_grid/02_augment_network"
    expected = [
        "01_der_inventory.ipynb",
        "02_load_profiles.ipynb",
        "03_switch_synthesis.ipynb",
        "04_load_blocks.ipynb",
        "05_onm_export.ipynb",
    ]

    assert sorted(path.name for path in notebooks.glob("*.ipynb")) == expected

    for filename in expected:
        notebook = json.loads((notebooks / filename).read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"{filename}:cell{index}", "exec")

    der_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads((notebooks / "01_der_inventory.ipynb").read_text(encoding="utf-8"))["cells"]
        if cell.get("cell_type") == "code"
    )
    load_profiles_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads((notebooks / "02_load_profiles.ipynb").read_text(encoding="utf-8"))["cells"]
        if cell.get("cell_type") == "code"
    )
    switch_notebook = json.loads((notebooks / "03_switch_synthesis.ipynb").read_text(encoding="utf-8"))
    switch_text = "\n".join("".join(cell.get("source", [])) for cell in switch_notebook["cells"])
    switch_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in switch_notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    load_blocks_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads((notebooks / "04_load_blocks.ipynb").read_text(encoding="utf-8"))["cells"]
        if cell.get("cell_type") == "code"
    )
    onm_export_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads((notebooks / "05_onm_export.ipynb").read_text(encoding="utf-8"))["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "build_layer_1_der_inventory" in der_source
    assert "build_layer1_der_inventory" not in der_source
    assert "critical_load_assignments.parquet" in der_source
    assert "der_inventory_pyarrow_schema" in der_source
    assert "battery_kwh" not in der_source
    assert "build_location_load_profile_inputs" in load_profiles_source
    assert "build_marshfield_load_profile_inputs" not in load_profiles_source
    assert "assign_load_profile_archetypes" not in load_profiles_source
    assert "run_marshfield_ssap" not in switch_source
    assert "build_ssap_components" in switch_source
    assert "physical_lines_only" in switch_source
    assert "physical_switch_candidate_edges" in switch_source
    assert "select_policy_candidate_edges" in switch_source
    assert "solve_ssap_per_feeder" in switch_source
    assert "ensure_frontier_point" in switch_source
    assert "switch_budget = 150" in switch_source
    assert "component_switch_cap = 4" in switch_source
    assert "max_candidate_edges=20" in switch_source
    assert "A polynomial-time exact algorithm for the sectionalizing switch allocation.pdf" in switch_text
    assert "switch_placement_dnmg_graph_based.pdf" not in switch_text
    assert "assemble_switch_artifact" in switch_source
    assert "associated_line_name" in switch_source
    assert "transformers = pd.read_csv" in load_blocks_source
    assert "transformers=transformers" in load_blocks_source
    assert "build_location_block_overview" in load_blocks_source
    assert "build_marshfield_block_overview" not in load_blocks_source
    assert "build_ocean_bluff_block_detail" in load_blocks_source
    assert "from power.event_window import build_event_window_bundle" in onm_export_source
    assert "build_switch_line_overlay" in onm_export_source
    assert "build_marshfield_switch_line_overlay" not in onm_export_source
    assert "display(Image(filename=result[\"output_path\"]))" in onm_export_source


def test_marshfield_grid_notebooks_are_digestible_dataset_runbooks():
    notebooks = repo_root / "locations/marshfield/01_grid"
    expected = {
        "01_base_network.ipynb": "# 01 Baseline Network",
        "03_audit_network.ipynb": "# 03 Network Audit",
    }

    for filename, heading in expected.items():
        notebook = json.loads((notebooks / filename).read_text(encoding="utf-8"))
        text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert heading in text
        assert "optimization" not in text.lower()
        assert len(notebook["cells"]) >= 4
        if filename == "03_audit_network.ipynb":
            assert "smart_ds_compat_dir=grid[\"augmented_artifacts\"]" in text
            assert "plot_validation_region_report_card" in text
            assert "krishnan_validation_region_report_card.png" in text


def test_marshfield_base_network_notebook_builds_shift_systems_with_explicit_equipment_mappers():
    notebook = json.loads(
        (repo_root / "locations/marshfield/01_grid/01_base_network.ipynb").read_text(
            encoding="utf-8"
        )
    )
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "build_shift_example_equipment_catalog" in text
    assert "ShiftExampleEdgeEquipmentMapper" in text
    assert "in-memory example equipment catalog" in text
    assert "DistributionSystemBuilder(\n                region_name,\n                graph,\n                phase_mappers[idx],\n                voltage_mappers[idx],\n                equipment_mappers[idx],\n            ).get_system()" in text
    assert "Failed to export" in text
    assert "No region systems were exported." in text
    assert "DistributionSystemBuilder(region_name, graph).build()" not in text
    assert "resources\" / \"equipment_catalog.json\"" not in text


def test_marshfield_base_network_notebook_uses_current_shift_graph_api_and_cache_contract():
    notebook = json.loads(
        (repo_root / "locations/marshfield/01_grid/01_base_network.ipynb").read_text(
            encoding="utf-8"
        )
    )
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "study_polygons" in text
    assert "bounds = study_area.bounds" in text
    assert "graph_cache_params" in text
    assert "source_anchor_policy" in text
    assert "source_anchor_ids" in text
    assert "resolve_grid_source_area" in text
    assert "resolve_grid_source_anchors" in text
    assert "mhmp_substations.json" not in text
    assert "docs\" / \"sandboxes\"" not in text
    assert "source_assignments" in text
    assert "clusters = get_kmeans_clusters(max(len(pts) // customers_per_transformer, 1), pts)" in text
    assert "PRSG(groups=clusters, source_location=GeoLocation(source[\"lon\"], source[\"lat\"])).get_distribution_graph()" in text
    assert "get_kmeans_clusters(parcels, tile_size_deg=tile_size_deg)" not in text
    assert "customers_per_transformer=customers_per_transformer" not in text
    assert ".run()" not in text


def test_marshfield_base_network_notebook_uses_current_gdm_quantity_imports():
    notebook = json.loads(
        (repo_root / "locations/marshfield/01_grid/01_base_network.ipynb").read_text(
            encoding="utf-8"
        )
    )
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    outputs = "\n".join(json.dumps(cell.get("outputs", [])) for cell in notebook["cells"])

    assert "from gdm.quantities import ApparentPower" in text
    assert "from infrasys.quantities import ApparentPower" not in text
    assert "cannot import name 'ApparentPower'" not in outputs
