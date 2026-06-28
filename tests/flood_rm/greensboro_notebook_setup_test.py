import ast
import json
from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]
locations_root = repo_root / "locations"
inland_locations = ["austin", "greensboro"]
catalog_locations = ["austin", "greensboro", "marshfield"]


def _notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_text(path: Path) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in _notebook(path)["cells"])


def _code_sources(path: Path):
    for index, cell in enumerate(_notebook(path)["cells"]):
        if cell.get("cell_type") == "code":
            yield index, "".join(cell.get("source", []))


def _flood_root(location: str) -> Path:
    return locations_root / location / "02_flood"


def test_inland_flood_notebook_inventory_matches_current_wflow_sequence():
    expected = [
        "01_region_setup.ipynb",
        "02_collect_sources.ipynb",
        "03_build_event_catalog.ipynb",
        "05_create_scenarios.ipynb",
        "05b_calibrate_wshed.ipynb",
        "05c_ship_calibrated.ipynb",
        "06_evaluate.ipynb",
    ]
    expected_04 = [
        "a_build_coupled_model.ipynb",
        "b_prepare_wflow_dynamic_handoff.ipynb",
        "c_run_example.ipynb",
    ]

    for location in inland_locations:
        flood_root = _flood_root(location)
        assert [path.name for path in sorted(flood_root.glob("[0-9][0-9]*.ipynb"))] == expected
        assert [path.name for path in sorted((flood_root / "04").glob("*.ipynb"))] == expected_04


def test_flood_notebook_code_cells_compile():
    notebooks = []
    for location in catalog_locations:
        flood_root = _flood_root(location)
        notebooks.extend(sorted(flood_root.glob("[0-9][0-9]*.ipynb")))
        notebooks.extend(sorted((flood_root / "04").glob("*.ipynb")))

    assert notebooks
    for notebook_path in notebooks:
        for index, source in _code_sources(notebook_path):
            ast.parse(source, filename=f"{notebook_path}:cell{index}")


def test_collect_sources_notebooks_show_sst_geography_and_rainfall_members():
    for location in catalog_locations:
        text = _source_text(_flood_root(location) / "02_collect_sources.ipynb")

        assert "collect.load_runtime(" in text
        assert "collect.plot_collected_sst_geography(config, paths)" in text
        assert "collect.plot_aorc_sst_rainfall(paths)" in text


def test_event_catalog_notebooks_keep_one_historical_and_one_design_severity_plot():
    for location in inland_locations:
        text = _source_text(_flood_root(location) / "03_build_event_catalog.ipynb")

        assert text.count("P.plot_severity_bands(") == 2
        assert "Historical rainfall-driver severity distribution" in text
        assert "P.plot_severity_bands(df_catalog);" in text
        assert "materialize_inland_catalog_outputs(" in text
        assert "catalog_outputs[\"preview\"]" in text
        assert "catalog_outputs[\"summary\"]" in text
        assert "location_relative_member_file" not in text

    marshfield_text = _source_text(_flood_root("marshfield") / "03_build_event_catalog.ipynb")
    assert marshfield_text.count("P.plot_severity_bands(") == 2
    assert "Historical paired-driver severity distribution" in marshfield_text
    assert "P.plot_severity_bands(stress_training_catalog);" in marshfield_text


def test_event_catalog_notebooks_do_not_render_removed_budget_plots():
    for location in catalog_locations:
        text = _source_text(_flood_root(location) / "03_build_event_catalog.ipynb")

        assert "P.plot_tail_budget(" not in text
        assert "P.plot_joint_tail_budget(" not in text
        assert "P.plot_catalog_set_severity_comparison(" not in text
        assert "Candidate pool to selected stress set" not in text
        assert "Selection rate by severity band" not in text
        assert "Fitted probability mass by band" not in text


def test_inland_catalog_notebooks_use_rainfall_driven_wflow_response_contract():
    builder_text = (
        repo_root / "src/design_events/build_events/probability/inland_dependence.py"
    ).read_text(encoding="utf-8")
    workflow_text = (repo_root / "src/design_events/build_events/workflow.py").read_text(encoding="utf-8")
    assert "rainfall_marginal_with_antecedent_moisture_wflow_response" in builder_text
    assert 'catalog["streamflow_design_role"] = "wflow_response"' in builder_text
    assert '"streamflow_design_role"' in workflow_text

    for location in inland_locations:
        text = _source_text(_flood_root(location) / "03_build_event_catalog.ipynb")

        assert "build_inland_catalog(" in text
        assert 'df_catalog["forcing_pairing_policy"].iloc[0]' in text
        assert "P.plot_scaling(df_catalog, \"rainfall\")" in text
        assert "design_driver" in text
        assert "rainfall + antecedent moisture" in text


def test_inland_build_notebooks_use_current_wflow_sfincs_handoff_flow():
    for location in inland_locations:
        text = _source_text(_flood_root(location) / "04/a_build_coupled_model.ipynb")

        assert "load_runtime(Path(\"../..\").resolve(), wflow_domain_review_required=True)" in text
        assert "build_wflow_submodel(" in text
        assert "create_handoffs(" in text
        assert "sfincs_rivers_inflow_geoms(sf)" in text
        assert "set_observations(" in text
        assert "plot_wflow_ldd_components" in text
        assert "load_coupled_runtime" not in text
        assert "force_wflow_river_build = True" in text


def test_inland_scenario_notebooks_submit_joint_wflow_sfincs_pipeline():
    for location in inland_locations:
        text = _source_text(_flood_root(location) / "05_create_scenarios.ipynb")

        assert "cluster/run_wflow_sfincs_dsai_inland_coupled.slurm" in text
        assert "cluster/run_sfincs_dsai_inland_coupled.slurm" not in text
        assert "FORCE_WFLOW" in text
        assert "OVERWRITE_METEO" in text
        assert "joint_worklist_csv = runtime.joint_worklist_path" in text
        assert "joint_wflow_sfincs_cluster_plan.json" in text
        assert "Wflow dynamic handoff -> native SFINCS staging -> SFINCS run" in text


def test_inland_run_example_and_evaluate_notebooks_include_wflow_output_plots():
    for location in inland_locations:
        run_text = _source_text(_flood_root(location) / "04/c_run_example.ipynb")
        evaluate_text = _source_text(_flood_root(location) / "06_evaluate.ipynb")

        assert "plot_wflow_event_handoff" in run_text
        assert "plot_event_precipitation_peak_discharge" in evaluate_text
        assert "load_runtime(Path(\"../..\").resolve(), wflow_domain_review_required=False)" in run_text
        assert "load_runtime(location_root, wflow_domain_review_required=True)" in evaluate_text


def test_marshfield_wave_example_uses_event_catalog_selection():
    text = _source_text(_flood_root("marshfield") / "04/b_example_waves.ipynb")

    assert 'SINGLE_USE_EVENT_ID = "evt_' not in text
    assert "SINGLE_USE_EVENT_ID = None" in text
    assert "event_id=SINGLE_USE_EVENT_ID" in text
