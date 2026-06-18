import pytest
import pandas as pd

from design_events.collect_sources.all_sources import collect_all_sources
from design_events.collect_sources.plan import build_source_collection_plan
from design_events.collect_sources.run_collect import run_collect
from design_events.cli import build_parser


def test_collect_all_sources_runs_configured_direct_collectors(tmp_path):
    calls = []
    config = {
        "collection": {
            "start": "2020-01-01",
            "end": "2020-01-03",
            "cora": {},
            "usgs_streamgages": {},
            "national_hydrography": {},
            "nwm": {},
            "aorc_sst": {},
            "era5_waves": {},
        }
    }
    paths = {
        "waterlevel_csv": tmp_path / "cora.csv",
        "usgs_streamgage_candidates_geojson": tmp_path / "usgs/streamgage_candidates.geojson",
        "nwm_root": tmp_path / "nwm",
        "aorc_sst_rainfall_members_csv": tmp_path / "aorc_sst/rainfall_members.csv",
    }

    def collect_cora(settings, skip_existing=False, smoke=False):
        calls.append(("cora", settings["start"].date().isoformat(), settings["end"].date().isoformat(), skip_existing, smoke))
        return pd.DataFrame({"time": pd.date_range("2020-01-01", periods=1), "value": [1.0]})

    def collect_usgs_streamgages(settings, skip_existing=False, smoke=False):
        calls.append(("usgs_streamgages", settings["start"].date().isoformat(), settings["end"].date().isoformat(), skip_existing, smoke))
        return {"candidate_count": 2, "candidate_geojson": paths["usgs_streamgage_candidates_geojson"]}

    def collect_nwm(settings, skip_existing=False, smoke=False):
        calls.append(("nwm", settings["start"].date().isoformat(), settings["end"].date().isoformat(), skip_existing, smoke))
        return {"streamflow_rows": 2, "soil_moisture_rows": 3}

    def collect_national_hydrography(settings, skip_existing=False, smoke=False):
        calls.append(("national_hydrography", settings["start"].date().isoformat(), settings["end"].date().isoformat(), skip_existing, smoke))
        return {"artifact_count": 4}

    def collect_aorc_sst(settings, skip_existing=False):
        calls.append(("aorc_sst", settings["start"].date().isoformat(), settings["end"].date().isoformat(), skip_existing))
        return {"ranked_rows": 4, "event_window_count": 2}

    def collect_era5_waves(settings, skip_existing=False, smoke=False):
        calls.append(("era5_waves", settings["start"].date().isoformat(), settings["end"].date().isoformat(), skip_existing, smoke))
        return {"time_count": 5, "wave_netcdf": tmp_path / "waves.nc"}

    result = collect_all_sources(
        config,
        paths,
        skip_existing=True,
        smoke=True,
        funcs={
            "collect_cora": collect_cora,
            "collect_usgs_streamgages": collect_usgs_streamgages,
            "collect_nwm": collect_nwm,
            "collect_national_hydrography": collect_national_hydrography,
            "collect_aorc_sst": collect_aorc_sst,
            "collect_era5_waves": collect_era5_waves,
        },
    )

    assert calls == [
        ("cora", "2020-01-01", "2020-01-03", True, True),
        ("usgs_streamgages", "2020-01-01", "2020-01-03", True, True),
        ("national_hydrography", "2020-01-01", "2020-01-03", True, True),
        ("nwm", "2020-01-01", "2020-01-03", True, True),
        ("aorc_sst", "2020-01-01", "2020-01-03", True),
        ("era5_waves", "2020-01-01", "2020-01-03", True, True),
    ]
    assert result["cora_rows"] == 1
    assert result["usgs_streamgages"] == {
        "candidate_count": 2,
        "candidate_geojson": paths["usgs_streamgage_candidates_geojson"],
    }
    assert result["nwm"] == {"streamflow_rows": 2, "soil_moisture_rows": 3}
    assert result["national_hydrography"] == {"artifact_count": 4}
    assert result["aorc_sst"] == {"ranked_rows": 4, "event_window_count": 2}
    assert result["era5_waves"] == {"time_count": 5, "wave_netcdf": tmp_path / "waves.nc"}
    assert "legacy_rainfall_config" not in result


def test_collect_all_sources_supports_inland_location_without_cora(tmp_path):
    calls = []
    config = {
        "flood_setting": "inland",
        "event_drivers": ["rainfall", "streamflow", "soil_moisture"],
        "collection": {
            "start": "2020-01-01",
            "end": "2020-01-03",
            "usgs_streamgages": {},
            "national_hydrography": {},
            "nwm": {},
            "aorc_sst": {"start_date": "2020-01-01", "end_date": "2020-01-03"},
        },
    }
    paths = {
        "source_artifacts_root": tmp_path / "source_artifacts",
        "usgs_streamgage_candidates_geojson": tmp_path / "usgs/streamgage_candidates.geojson",
        "nwm_root": tmp_path / "nwm",
        "aorc_sst_rainfall_members_csv": tmp_path / "aorc_sst/rainfall_members.csv",
    }

    def collect_usgs_streamgages(settings, skip_existing=False, smoke=False):
        calls.append(("usgs_streamgages", settings["start"].date().isoformat(), settings["end"].date().isoformat()))
        return {"candidate_count": 2, "candidate_geojson": paths["usgs_streamgage_candidates_geojson"]}

    def collect_nwm(settings, skip_existing=False, smoke=False):
        calls.append(("nwm", settings["start"].date().isoformat(), settings["end"].date().isoformat()))
        return {"streamflow_rows": 2, "soil_moisture_rows": 3}

    def collect_national_hydrography(settings, skip_existing=False, smoke=False):
        calls.append(("national_hydrography", settings["start"].date().isoformat(), settings["end"].date().isoformat()))
        return {"artifact_count": 4}

    def collect_aorc_sst(settings, skip_existing=False):
        calls.append(("aorc_sst", settings["start"].date().isoformat(), settings["end"].date().isoformat()))
        return {"ranked_rows": 4}

    result = collect_all_sources(
        config,
        paths,
        funcs={
            "collect_usgs_streamgages": collect_usgs_streamgages,
            "collect_nwm": collect_nwm,
            "collect_national_hydrography": collect_national_hydrography,
            "collect_aorc_sst": collect_aorc_sst,
        },
    )

    assert calls == [
        ("usgs_streamgages", "2020-01-01", "2020-01-03"),
        ("national_hydrography", "2020-01-01", "2020-01-03"),
        ("nwm", "2020-01-01", "2020-01-03"),
        ("aorc_sst", "2020-01-01", "2020-01-03"),
    ]
    assert result["waterlevel_csv"] is None
    assert result["cora_rows"] == 0
    assert result["usgs_streamgages"] == {
        "candidate_count": 2,
        "candidate_geojson": paths["usgs_streamgage_candidates_geojson"],
    }
    assert result["nwm"] == {"streamflow_rows": 2, "soil_moisture_rows": 3}


def test_collect_all_sources_rejects_nwm_dates_outside_base_window(tmp_path):
    config = {
        "collection": {
            "start": "2020-01-01",
            "end": "2020-01-31",
            "cora": {},
            "nwm": {"start": "2019-12-31"},
        }
    }

    with pytest.raises(ValueError, match="nwm collection dates must stay within the base collection window"):
        collect_all_sources(
            config,
            {"waterlevel_csv": tmp_path / "cora.csv"},
            funcs={
                "collect_cora": lambda *args, **kwargs: pd.DataFrame({"value": [1.0]}),
                "collect_nwm": lambda *args, **kwargs: {},
            },
        )


def test_collect_all_sources_rejects_aorc_sst_dates_outside_base_window(tmp_path):
    config = {
        "collection": {
            "start": "2020-01-01",
            "end": "2020-01-31",
            "aorc_sst": {"start_date": "2020-01-01", "end_date": "2020-02-01"},
        }
    }

    with pytest.raises(ValueError, match="aorc_sst collection dates must stay within the base collection window"):
        collect_all_sources(config, {"aorc_sst_rainfall_members_csv": tmp_path / "rainfall_members.csv"})


def test_run_collect_returns_notebook_result_table(tmp_path):
    config = {
        "collection": {
            "start": "2020-01-01",
            "end": "2020-01-03",
            "cora": {},
            "usgs_streamgages": {},
            "national_hydrography": {},
            "nwm": {},
            "aorc_sst": {},
            "era5_waves": {},
        }
    }
    paths = {
        "waterlevel_csv": tmp_path / "cora.csv",
        "usgs_streamgage_candidates_geojson": tmp_path / "usgs/streamgage_candidates.geojson",
        "nwm_soil_moisture_csv": tmp_path / "soil.csv",
        "aorc_sst_rainfall_members_csv": tmp_path / "aorc_sst/rainfall_members.csv",
    }
    paths["aorc_sst_rainfall_members_csv"].parent.mkdir()
    pd.DataFrame({"member_id": ["r1"]}).to_csv(paths["aorc_sst_rainfall_members_csv"], index=False)
    plan = build_source_collection_plan(config, paths)

    table = run_collect(
        config,
        paths,
        plan,
        progress=False,
        funcs={
            "collect_cora": lambda settings, skip_existing=False, smoke=False: pd.DataFrame({"value": [1.0, 2.0]}),
            "collect_usgs_streamgages": lambda settings, skip_existing=False, smoke=False: {
                "candidate_count": 6,
                "candidate_geojson": paths["usgs_streamgage_candidates_geojson"],
            },
            "collect_nwm": lambda settings, skip_existing=False, smoke=False: {
                "reused": True,
                "soil_moisture_rows": 3,
                "soil_moisture_csv": paths["nwm_soil_moisture_csv"],
            },
            "collect_national_hydrography": lambda settings, skip_existing=False, smoke=False: {
                "artifact_count": 4,
                "hydromt_basemap": tmp_path / "hydrography.nc",
            },
            "collect_aorc_sst": lambda settings, skip_existing=False: {
                "ranked_rows": 4,
                "ranked_storms_csv": tmp_path / "ranked.csv",
            },
            "collect_era5_waves": lambda settings, skip_existing=False, smoke=False: {
                "time_count": 5,
                "wave_netcdf": tmp_path / "waves.nc",
            },
        },
    )

    assert table["source"].tolist() == ["cora", "usgs_streamgages", "national_hydrography", "nwm", "aorc_sst", "era5_waves", "rainfall_members"]
    assert table["status"].tolist() == ["collected", "collected", "collected", "reused", "collected", "collected", "collected"]
    assert table["rows"].tolist() == [2, 6, 4, 3, 4, 5, 1]


def test_run_collect_marks_rainfall_members_not_configured_without_aorc_sst(tmp_path):
    config = {"collection": {"start": "2020-01-01", "end": "2020-01-03", "nwm": {}}}
    paths = {
        "nwm_soil_moisture_csv": tmp_path / "soil.csv",
        "aorc_sst_rainfall_members_csv": tmp_path / "aorc_sst/rainfall_members.csv",
    }
    plan = build_source_collection_plan(config, paths)

    table = run_collect(
        config,
        paths,
        plan,
        progress=False,
        funcs={
            "collect_nwm": lambda settings, skip_existing=False, smoke=False: {
                "soil_moisture_rows": 3,
                "soil_moisture_csv": paths["nwm_soil_moisture_csv"],
            },
        },
    )

    assert table["source"].tolist() == ["nwm", "rainfall_members"]
    assert table["status"].tolist() == ["collected", "not_configured"]
    assert table["rows"].tolist() == [3, 0]


def test_run_collect_prints_source_failure_before_continuing(tmp_path, capsys):
    config = {"collection": {"start": "2020-01-01", "end": "2020-01-03", "nwm": {}}}
    paths = {
        "nwm_soil_moisture_csv": tmp_path / "soil.csv",
        "aorc_sst_rainfall_members_csv": tmp_path / "aorc_sst/rainfall_members.csv",
    }
    plan = build_source_collection_plan(config, paths)

    def fail_nwm(settings, skip_existing=False, smoke=False):
        raise RuntimeError("zarr open stalled")

    table = run_collect(
        config,
        paths,
        plan,
        progress=False,
        stop_on_error=False,
        funcs={"collect_nwm": fail_nwm},
    )

    captured = capsys.readouterr()
    assert "nwm: failed with RuntimeError: zarr open stalled" in captured.out
    assert table.loc[table["source"].eq("nwm"), "status"].item() == "failed"


def test_pipeline_accepts_collect_sources_stage_without_legacy_rainfall_options():
    args = build_parser().parse_args(
        [
            "collect_sources",
            "--config",
            "locations/marshfield/config.yaml",
            "--skip-existing",
        ]
    )

    assert args.stage == "collect_sources"
    assert args.config == "locations/marshfield/config.yaml"
    assert args.skip_existing is True


def test_pipeline_rejects_removed_legacy_rainfall_stages_and_options():
    parser = build_parser()
    legacy = "storm" + "hub"

    with pytest.raises(SystemExit):
        parser.parse_args([f"collect_{legacy}", "--config", "locations/marshfield/config.yaml"])
    with pytest.raises(SystemExit):
        parser.parse_args([f"preflight_{legacy}", "--config", "locations/marshfield/config.yaml"])
    with pytest.raises(SystemExit):
        parser.parse_args(["build_rainfall_members", "--config", "locations/marshfield/config.yaml"])
    with pytest.raises(SystemExit):
        parser.parse_args(["collect_sources", f"--run-{legacy}"])


def test_pipeline_accepts_collect_aorc_sst_stage():
    args = build_parser().parse_args(
        [
            "collect_aorc_sst",
            "--config",
            "locations/marshfield/config.yaml",
            "--skip-existing",
        ]
    )

    assert args.stage == "collect_aorc_sst"
    assert args.config == "locations/marshfield/config.yaml"
    assert args.skip_existing is True


def test_pipeline_accepts_check_readiness_stage():
    args = build_parser().parse_args(
        [
            "check_readiness",
            "--config",
            "locations/marshfield/config.yaml",
        ]
    )

    assert args.stage == "check_readiness"
    assert args.config == "locations/marshfield/config.yaml"
