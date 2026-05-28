from pathlib import Path

from design_events.build_events import build_event_catalog_plan
from design_events.config import load_runtime


def suffix(path):
    return Path(path).relative_to(Path(__file__).resolve().parents[3]).as_posix()


def test_event_catalog_plan_uses_marshfield_runtime_config():
    config, paths = load_runtime("locations/marshfield/config.yaml")

    plan = build_event_catalog_plan(config, paths)

    assert plan.study_location == "marshfield"
    assert plan.scenario_name == "base"
    assert suffix(plan.event_summary_csv) == "locations/marshfield/data/event_catalog/events/surge_event_members_summary.csv"
    assert suffix(plan.event_members_nc) == "locations/marshfield/data/event_catalog/events/surge_event_members.nc"
    assert suffix(plan.event_catalog_csv) == "locations/marshfield/data/event_catalog/catalog/event_catalog.csv"
    assert suffix(plan.audit_json) == "locations/marshfield/data/event_catalog/catalog/event_catalog_audit.json"
    assert plan.forcing_names == ("rainfall", "soil_moisture")
    assert suffix(plan.forcing("rainfall").member_path) == "locations/marshfield/data/sources/aorc_sst/rainfall_members.csv"
    assert suffix(plan.forcing("soil_moisture").member_path) == "locations/marshfield/data/sources/nwm/soil_moisture.csv"
    assert plan.forcing("rainfall").pairing_policy["strategy"] == "seasonal_window_permutation"
    assert plan.required_forcings == ("coastal", "rainfall", "soil_moisture")


def test_event_catalog_plan_records_wave_analog_requirement_for_coastal_waves(tmp_path):
    config = {
        "project": {"name": "marshfield"},
        "coastal_waves": True,
        "event_catalog": {
            "forcing_members": {"rainfall": "rainfall_members.csv"},
            "pairing": {"rainfall": {"strategy": "independent_permutation", "seed": 3}},
        },
    }
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "event_summary_csv": tmp_path / "events/summary.csv",
        "event_members_nc": tmp_path / "events/members.nc",
        "event_catalog_csv": tmp_path / "catalog/event_catalog.csv",
        "event_catalog_audit_json": tmp_path / "catalog/audit.json",
        "scenario": {"name": "base"},
    }

    plan = build_event_catalog_plan(config, paths)

    assert plan.wave_analog_policy == "same_historical_analog"
    assert plan.required_source_artifacts == ("event_summary", "event_members", "rainfall_members", "era5_waves")


def test_event_catalog_plan_summarizes_notebook_rows(tmp_path):
    config = {"project": {"name": "austin"}, "event_catalog": {"forcing_members": {}}}
    paths = {
        "repo_root": tmp_path,
        "location_name": "austin",
        "event_summary_csv": tmp_path / "events/summary.csv",
        "event_members_nc": tmp_path / "events/members.nc",
        "event_catalog_csv": tmp_path / "catalog/event_catalog.csv",
        "event_catalog_audit_json": tmp_path / "catalog/audit.json",
        "scenario": {"name": "base"},
    }

    plan = build_event_catalog_plan(config, paths)

    assert plan.summary_rows() == [
        {"item": "study_location", "value": "austin"},
        {"item": "scenario_name", "value": "base"},
        {"item": "event_summary_csv", "value": str(tmp_path / "events/summary.csv")},
        {"item": "event_catalog_csv", "value": str(tmp_path / "catalog/event_catalog.csv")},
        {"item": "forcings", "value": ""},
        {"item": "wave_analog_policy", "value": "not_required"},
    ]
