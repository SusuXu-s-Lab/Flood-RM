import pandas as pd

from design_events.collect_sources import build_source_collection_plan


def test_source_collection_plan_describes_configured_sources(tmp_path):
    config = {
        "collection": {
            "start": "2020-01-01",
            "end": "2020-01-31",
            "cora": {"variable": "zeta"},
            "national_hydrography": {"service": "usgs_3dhp_or_nhdplus_hr"},
            "nwm": {"start": "2020-01-03", "end": "2020-01-10"},
            "aorc_sst": {"start_date": "2020-01-05", "end_date": "2020-01-07"},
            "era5_waves": {"provider": "earthdatahub"},
        }
    }

    plan = build_source_collection_plan(config, {"outputs_root": tmp_path})

    assert plan.source_names == ("cora", "national_hydrography", "nwm", "aorc_sst", "era5_waves")
    assert plan.start == pd.Timestamp("2020-01-01")
    assert plan.end == pd.Timestamp("2020-01-31")
    assert plan.step("nwm").start == pd.Timestamp("2020-01-03")
    assert plan.step("nwm").end == pd.Timestamp("2020-01-10")
    assert plan.step("aorc_sst").start_date == "2020-01-05"
    assert plan.step("aorc_sst").end_date == "2020-01-07"
    assert plan.settings_for("era5_waves")["era5_waves"] == {"provider": "earthdatahub"}
    assert plan.settings_for("national_hydrography")["national_hydrography"] == {
        "service": "usgs_3dhp_or_nhdplus_hr"
    }


def test_source_collection_plan_ignores_unrecognized_legacy_sources(tmp_path):
    config = {
        "collection": {
            "start": "2020-01-01",
            "end": "2020-01-31",
            "aorc_sst": {"start_date": "2020-01-03", "end_date": "2020-01-07"},
            "legacy_rainfall": {"start_date": "2020-01-05", "end_date": "2020-01-09"},
            "era5_waves": {"provider": "earthdatahub"},
        }
    }

    plan = build_source_collection_plan(config, {"outputs_root": tmp_path})

    assert plan.source_names == ("aorc_sst", "era5_waves")
    assert plan.step("aorc_sst").start_date == "2020-01-03"
    assert plan.step("aorc_sst").end_date == "2020-01-07"


def test_source_collection_plan_summarizes_notebook_rows(tmp_path):
    config = {
        "collection": {
            "start": "2020-01-01",
            "end": "2020-01-31",
            "cora": {},
            "nwm": {"start": "2020-01-03", "end": "2020-01-10"},
        }
    }

    plan = build_source_collection_plan(config, {"outputs_root": tmp_path})

    assert plan.summary_rows() == [
        {"source": "cora", "start": "2020-01-01", "end": "2020-01-31"},
        {"source": "nwm", "start": "2020-01-03", "end": "2020-01-10"},
    ]
