from pathlib import Path

from fiat_runs import load_notebook_runtime
from study_location import define_location
from sfincs_runs.config import build_grid_paths
from sfincs_runs.config import load_runtime as load_config_runtime
from sfincs_runs.config import load_sfincs_runtime
from wflow_runs.notebook import load_calibration_runtime, load_runtime as load_wflow_runtime


repo_root = Path(__file__).resolve().parents[2]


def test_collect_sources_runtime_derives_inland_source_paths_from_location_workspace():
    from design_events.collect_sources.workflow import load_runtime, source_records

    location_root = repo_root / "locations" / "greensboro"

    runtime = load_runtime(location_root)
    records = source_records(runtime)

    assert runtime.candidate_path == location_root / "data/sources/usgs_streamgages/streamgage_candidates.geojson"
    assert runtime.reviewed_network_path == location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"
    assert runtime.streamflow_records_path == location_root / "data/sources/usgs_streamgages/streamflow_records.csv"
    assert set(["active USGS streamgage candidates", "AORC rainfall members", "NWM soil-moisture members"]).issubset(
        set(records["record"])
    )


def test_event_catalog_runtime_derives_inland_source_inventory_paths():
    from design_events.build_events.workflow import event_catalog_source_inventory, load_runtime

    location_root = repo_root / "locations" / "greensboro"

    runtime = load_runtime(location_root)
    inventory = event_catalog_source_inventory(runtime)

    assert set(["reviewed streamgage network", "streamflow members", "rainfall members", "soil moisture"]).issubset(
        set(inventory["artifact"])
    )
    assert (
        inventory.loc[inventory["artifact"].eq("rainfall members"), "path"].iloc[0]
        == str(location_root / "data/sources/aorc_sst/rainfall_members.csv")
    )


def test_wflow_coupled_runtime_derives_location_artifacts_without_mutating_definition():
    location_root = repo_root / "locations" / "greensboro"

    fresh_config = define_location(location_root / "config.yaml").config
    default_runtime = load_wflow_runtime(location_root)
    runtime = load_wflow_runtime(location_root, wflow_domain_review_required=False)
    reviewed_runtime = load_wflow_runtime(location_root, wflow_domain_review_required=True)

    assert runtime.location_root == location_root
    assert runtime.location_name == "greensboro"
    assert default_runtime.config["wflow"]["domain_set"]["review_required"] == fresh_config["wflow"]["domain_set"]["review_required"]
    assert runtime.config["wflow"]["domain_set"]["review_required"] is False
    assert reviewed_runtime.config["wflow"]["domain_set"]["review_required"] is True
    assert "scenario_run" in runtime.config
    assert runtime.joint_worklist_path == location_root / "data/sfincs/scenarios/greensboro_joint_wflow_sfincs_worklist.csv"
    assert runtime.readiness_path == location_root / "data/sfincs/scenarios/greensboro_dynamic_handoff_readiness.csv"
    assert runtime.events_root == location_root / "data/wflow/events"
    assert runtime.design_paths["usgs_streamgage_network_geojson"] == location_root / "data/sources/usgs_streamgages/streamgage_network.geojson"

    assert define_location(location_root / "config.yaml").config == fresh_config


def test_wflow_calibration_runtime_exposes_validation_paths_without_creating_dirs():
    location_root = repo_root / "locations" / "greensboro"

    runtime = load_calibration_runtime(location_root, create_audit_dirs=False)

    assert runtime.scenario_catalog_path == location_root / "data/event_catalog/catalog/scenario_catalog.csv"
    assert runtime.streamflow_records_path == location_root / "data/sources/usgs_streamgages/streamflow_records.csv"
    assert runtime.event_streamflow_iv_root == location_root / "data/sources/usgs_streamgages/event_streamflow_iv"
    assert runtime.audit_plots_dir == location_root / "data/wflow/audit/plots"


def test_grid_notebooks_resolve_grid_paths_without_power_notebook_facade():
    location_root = repo_root / "locations" / "marshfield"

    config, paths = load_config_runtime(location_root / "config.yaml")
    grid = build_grid_paths(config)

    assert paths["location_root"] == location_root
    assert paths["location_name"] == "marshfield"
    assert grid["asset_registry"] == location_root / "data/power_grid/asset_registry"
    assert grid["augmented_artifacts"] == location_root / "data/static/power_grid/smart_ds_compat"
    assert grid["figures"] == location_root / "data/power_grid/figures"


def test_sfincs_notebook_runtime_derives_standard_and_wave_paths():
    location_root = repo_root / "locations" / "marshfield"

    standard = load_sfincs_runtime(location_root, create_base_model_dir=False)
    wave = load_sfincs_runtime(location_root, wave=True, create_base_model_dir=False)

    assert standard.base_model == location_root / "data/sfincs/base"
    assert standard.catalog_dir == location_root / "data/event_catalog/catalog"
    assert standard.scenarios_root == location_root / "data/sfincs/scenarios"
    assert wave.base_model == location_root / "data/sfincs/base_quadtree_snapwave"
    assert wave.quadtree_cfg["base_model_root"] == "data/sfincs/base_quadtree_snapwave"


def test_fiat_notebook_runtime_layers_risk_paths_without_creating_tide_gauge_dirs():
    location_root = repo_root / "locations" / "marshfield"

    runtime = load_notebook_runtime(location_root, create_tide_gauge_dirs=False)

    assert runtime.catalog_csv == location_root / "data/event_catalog/catalog/event_catalog.csv"
    assert runtime.metadata_json == location_root / "data/event_catalog/catalog/catalog_risk_metadata.json"
    assert runtime.model_root == location_root / "data/fiat/model"
    assert runtime.per_event_damage_csv == location_root / "data/fiat/risk/per_event_damage.csv"
    assert runtime.tide_gauge_fig_root == location_root / "data/sfincs/tide_gauges/figures"
