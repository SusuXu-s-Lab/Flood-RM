from pathlib import Path

import pandas as pd

from sfincs_runs.config import load_runtime
from sfincs_runs.single_use_case import build_single_use_case_plan


def suffix(path):
    return Path(path).relative_to(Path(__file__).resolve().parents[3]).as_posix()


def test_single_use_case_plan_selects_first_event_from_catalog(tmp_path):
    event_catalog = tmp_path / "design/catalog/event_catalog.csv"
    event_catalog.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": ["evt_0042", "evt_0099", "evt_0100"],
            "event_family": ["surge_synthetic", "historical_extreme", "historical_extreme"],
            "coastal_template_peak_time": ["2020-01-01", "2004-01-01", "2020-02-01"],
            "sample_rp_years": [500.0, 800.0, 100.0],
        }
    ).to_csv(event_catalog, index=False)
    paths = {
        "location_name": "marshfield",
        "outputs_root": tmp_path / "sfincs_outputs",
        "design_outputs_root": tmp_path / "design",
        "base_model_root": tmp_path / "sfincs_outputs/base",
        "event_catalog_csv": event_catalog,
    }

    plan = build_single_use_case_plan({}, paths, reference_date="2026-05-25")

    assert plan.event_id == "evt_0100"
    assert plan.selection_reason == "recent_historical_extreme"
    assert {"item": "selection_reason", "value": "recent_historical_extreme"} in plan.summary_rows()
    assert plan.scenarios_dir == tmp_path / "sfincs_outputs/single_use_case/scenarios"
    assert plan.storage_dir == tmp_path / "sfincs_outputs/single_use_case/run_outputs"
    assert plan.run_root == tmp_path / "sfincs_outputs/single_use_case/run_stage"
    assert plan.stats_dir == tmp_path / "sfincs_outputs/single_use_case/stats"
    assert plan.build_command[-5:] == ["--event-id", "evt_0100", "--force", "--limit", "1"]
    assert plan.required_inputs == ("event_catalog", "event_catalog_audit", "base_model")


def test_single_use_case_plan_accepts_event_override(tmp_path):
    event_catalog = tmp_path / "design/catalog/event_catalog.csv"
    event_catalog.parent.mkdir(parents=True)
    pd.DataFrame({"event_id": ["evt_0042", "evt_0099"]}).to_csv(event_catalog, index=False)
    paths = {
        "location_name": "marshfield",
        "outputs_root": tmp_path / "sfincs_outputs",
        "design_outputs_root": tmp_path / "design",
        "base_model_root": tmp_path / "sfincs_outputs/base",
        "event_catalog_csv": event_catalog,
    }

    plan = build_single_use_case_plan({}, paths, event_id="evt_0099")

    assert plan.event_id == "evt_0099"
    assert plan.selection_reason == "explicit_event_id"
    assert "--dry-run" in plan.dry_run_command
    assert plan.run_command[-3:] == ["--event-id", "evt_0099", "--force-rerun"]
    assert plan.stats_command[-2:] == ["--event-id", "evt_0099"]


def test_single_use_case_plan_falls_back_to_recent_template_proxy(tmp_path):
    event_catalog = tmp_path / "design/catalog/event_catalog.csv"
    event_catalog.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": ["evt_0001", "evt_0002"],
            "event_family": ["surge_synthetic", "surge_synthetic"],
            "coastal_template_peak_time": ["2018-01-04", "2021-02-01"],
            "sample_rp_years": [50.0, 10.0],
        }
    ).to_csv(event_catalog, index=False)
    paths = {
        "location_name": "marshfield",
        "outputs_root": tmp_path / "sfincs_outputs",
        "design_outputs_root": tmp_path / "design",
        "base_model_root": tmp_path / "sfincs_outputs/base",
        "event_catalog_csv": event_catalog,
    }

    plan = build_single_use_case_plan({}, paths, reference_date="2026-05-25")

    assert plan.event_id == "evt_0001"
    assert plan.selection_reason == "recent_template_extreme_proxy"


def test_single_use_case_plan_uses_marshfield_runtime_config():
    config, paths = load_runtime("locations/marshfield/config.yaml")

    plan = build_single_use_case_plan(config, paths, event_id="evt_0001")

    assert plan.study_location == "marshfield"
    assert suffix(plan.base_model_root) == "locations/marshfield/data/sfincs/base"
    assert suffix(plan.design_outputs_root) == "locations/marshfield/data/event_catalog"
    assert suffix(plan.scenarios_dir) == "locations/marshfield/data/sfincs/single_use_case/scenarios"
    assert plan.event_id == "evt_0001"


def test_single_use_case_plan_accepts_base_model_override(tmp_path):
    event_catalog = tmp_path / "design/catalog/event_catalog.csv"
    event_catalog.parent.mkdir(parents=True)
    pd.DataFrame({"event_id": ["evt_0001"]}).to_csv(event_catalog, index=False)
    paths = {
        "location_name": "marshfield",
        "outputs_root": tmp_path / "sfincs_outputs",
        "design_outputs_root": tmp_path / "design",
        "base_model_root": tmp_path / "sfincs_outputs/base",
        "event_catalog_csv": event_catalog,
    }

    plan = build_single_use_case_plan(
        {},
        paths,
        event_id="evt_0001",
        base_model_root=tmp_path / "sfincs_outputs/base_quadtree_snapwave",
    )

    assert plan.base_model_root == tmp_path / "sfincs_outputs/base_quadtree_snapwave"
    assert str(plan.base_model_root) in plan.build_command
