import json
import pandas as pd
import yaml

from design_events.build_events.event_catalog import (
    attach_forcing_members,
    build_event_catalog,
    rebuild_forcing_pairing,
    validate_event_catalog,
    write_event_catalog_audit,
)
from design_events.build_events.inland_event_catalog import (
    build_inland_event_artifacts,
    write_wflow_sfincs_handoff_manifest,
)
from design_events.build_events.inland_streamflow import build_usgs_streamflow_event_members


def test_build_event_catalog_writes_surge_recipe_rows(tmp_path):
    paths = {
        "location_name": "marshfield",
        "event_summary_csv": tmp_path / "events/surge_event_members_summary.csv",
        "event_members_nc": tmp_path / "events/surge_event_members.nc",
        "event_catalog_csv": tmp_path / "catalog/event_catalog.csv",
        "event_catalog_audit_json": tmp_path / "catalog/event_catalog_audit.json",
        "scenario": {"name": "base", "slr_offset_m": 0.0},
    }
    paths["event_summary_csv"].parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": ["evt_0001", "evt_0002"],
            "sample_rp_years": [10.0, 100.0],
            "sampling_region": ["body", "tail"],
            "sampling_weight": [1.1, 0.6],
            "probability_weight": [0.8, 0.2],
            "peak": [1.2, 1.8],
            "absolute_peak_m": [1.2, 1.8],
            "valid_start_hour": [-24, -36],
            "valid_end_hour": [48, 60],
        }
    ).to_csv(paths["event_summary_csv"], index=False)

    catalog = build_event_catalog({}, paths)

    assert paths["event_catalog_csv"].exists()
    assert json.loads(paths["event_catalog_audit_json"].read_text())["passed"] is True
    assert catalog["event_id"].tolist() == ["evt_0001", "evt_0002"]
    assert catalog["study_location"].tolist() == ["marshfield", "marshfield"]
    assert catalog["event_family"].tolist() == ["surge_synthetic", "surge_synthetic"]
    assert catalog["scenario_name"].tolist() == ["base", "base"]
    assert catalog["sampling_region"].tolist() == ["body", "tail"]
    assert catalog["sampling_weight"].tolist() == [1.1, 0.6]
    assert catalog["probability_weight"].tolist() == [0.8, 0.2]
    assert catalog["coastal_source"].tolist() == ["cora", "cora"]
    assert catalog["coastal_member_file"].tolist() == [str(paths["event_members_nc"]), str(paths["event_members_nc"])]
    assert catalog["coastal_member_id"].tolist() == ["evt_0001", "evt_0002"]
    assert catalog["rainfall_source"].isna().all()
    assert catalog["streamflow_source"].isna().all()
    assert catalog["soil_moisture_source"].isna().all()
    assert catalog["infiltration_treatment"].tolist() == ["none", "none"]


def test_build_inland_event_artifacts_writes_greensboro_catalog_replay_and_audit(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    streamflow_members = location_root / "data/sources/usgs_streamgages/streamflow_members.csv"
    rainfall_members = location_root / "data/sources/aorc_sst/rainfall_members.csv"
    soil_moisture_members = location_root / "data/sources/nwm/soil_moisture.csv"
    for path in [streamflow_members, rainfall_members, soil_moisture_members]:
        path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "event_id": ["usgs_02095000_20200206", "usgs_02095500_20210917"],
            "site_no": ["02095000", "02095500"],
            "event_time": ["2020-02-06T12:00:00", "2021-09-17T09:00:00"],
            "peak_flow_cfs": [8300.0, 12400.0],
            "sample_rp_years": [25.0, 100.0],
            "sampling_region": ["body", "tail"],
            "sampling_weight": [0.7, 0.3],
            "probability_weight": [0.7, 0.3],
            "source": ["usgs", "usgs"],
        }
    ).to_csv(streamflow_members, index=False)
    pd.DataFrame(
        {
            "member_id": ["rain_feb", "rain_sep"],
            "member_file": ["rain_feb.nc", "rain_sep.nc"],
            "source": ["aorc_sst", "aorc_sst"],
            "storm_date": ["2020-02-06T06:00:00", "2021-09-17T06:00:00"],
        }
    ).to_csv(rainfall_members, index=False)
    pd.DataFrame(
        {
            "member_id": ["soil_feb", "soil_sep"],
            "member_file": ["soil.csv", "soil.csv"],
            "source": ["nwm", "nwm"],
            "time": ["2020-02-05T06:00:00", "2021-09-16T06:00:00"],
        }
    ).to_csv(soil_moisture_members, index=False)

    config = {
        "project": {"name": "greensboro"},
        "event_catalog": {
            "forcing_members": {
                "streamflow": "data/sources/usgs_streamgages/streamflow_members.csv",
                "rainfall": "data/sources/aorc_sst/rainfall_members.csv",
                "soil_moisture": "data/sources/nwm/soil_moisture.csv",
            },
            "pairing": {
                "rainfall": {
                    "strategy": "inland_rainfall_pairing_priority",
                    "same_storm_when_available": True,
                    "fallback_strategy": "seasonal_window_permutation",
                    "seed": 3,
                    "window_days": 30,
                },
                "soil_moisture": {
                    "strategy": "inland_antecedent_moisture_pairing",
                    "rainfall_relative_when_coherent": True,
                    "fallback_reference": "dominant_streamgage_network_peak",
                    "lead_time_hours": 24,
                },
            },
        },
        "sampling": {
            "severity_bands": [
                {"severity_band": "common", "rp_min_years": 0.0, "rp_max_years": 50.0},
                {"severity_band": "rare", "rp_min_years": 50.0, "rp_max_years": 500.0},
            ]
        },
        "inland_coupling": {"infiltration": {"method": "cn_with_recovery"}},
        "wflow": {"events_root": "data/wflow/events"},
        "paths": {"outputs_root": "data/event_catalog"},
    }

    artifacts = build_inland_event_artifacts(
        config,
        {
            "repo_root": tmp_path,
            "location_root": location_root,
            "location_name": "greensboro",
            "scenario": {"name": "base"},
        },
    )

    catalog = artifacts.catalog
    assert artifacts.probability_catalog_parquet.exists()
    assert artifacts.probability_catalog_csv.exists()
    assert artifacts.wflow_replay_set_parquet.exists()
    assert artifacts.event_manifest_yaml.exists()
    assert json.loads(artifacts.audit_json.read_text(encoding="utf-8"))["passed"] is True
    assert catalog["event_family"].tolist() == ["streamgage_network", "streamgage_network"]
    assert catalog["streamflow_source"].tolist() == ["usgs", "usgs"]
    assert catalog["streamflow_member_id"].tolist() == ["02095000_20200206T120000", "02095500_20210917T090000"]
    assert catalog["rainfall_member_id"].tolist() == ["rain_feb", "rain_sep"]
    assert catalog["rainfall_pairing_policy"].tolist() == ["seasonal_window_permutation", "seasonal_window_permutation"]
    assert catalog["soil_moisture_member_id"].tolist() == ["soil_feb", "soil_sep"]
    assert catalog["soil_moisture_pairing_policy"].tolist() == ["antecedent_to_forcing", "antecedent_to_forcing"]
    assert catalog["infiltration_treatment"].tolist() == ["cn_with_recovery", "cn_with_recovery"]

    manifest = yaml.safe_load(artifacts.event_manifest_yaml.read_text(encoding="utf-8"))
    assert manifest["study_location"] == "greensboro"
    assert manifest["event_count"] == 2
    assert manifest["artifacts"]["probability_catalog_parquet"].endswith("probability_catalog.parquet")


def test_build_inland_event_artifacts_expands_streamflow_templates_to_design_catalog(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    streamflow_members = location_root / "data/sources/usgs_streamgages/streamflow_members.csv"
    rainfall_members = location_root / "data/sources/aorc_sst/rainfall_members.csv"
    soil_moisture_members = location_root / "data/sources/nwm/soil_moisture.csv"
    for path in [streamflow_members, rainfall_members, soil_moisture_members]:
        path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "event_id": [f"usgs_02095000_20200{index:02d}" for index in range(1, 7)],
            "member_id": [f"02095000_20200{index:02d}T000000" for index in range(1, 7)],
            "site_no": ["02095000"] * 6,
            "event_time": pd.date_range("2020-01-01", periods=6, freq="30D").strftime("%Y-%m-%dT%H:%M:%S"),
            "peak_flow_cfs": [1200, 1800, 2500, 3600, 5200, 7600],
            "sample_rp_years": [2, 5, 10, 25, 100, 400],
            "sampling_region": ["body", "body", "body", "body", "tail", "tail"],
            "sampling_weight": [1.0] * 6,
            "probability_weight": [1 / 6] * 6,
            "source": ["usgs"] * 6,
        }
    ).to_csv(streamflow_members, index=False)
    pd.DataFrame(
        {
            "member_id": [f"rain_{index:02d}" for index in range(1, 7)],
            "source": ["aorc_sst"] * 6,
            "member_file": ["rain.csv"] * 6,
            "storm_date": pd.date_range("2019-01-01", periods=6, freq="60D").strftime("%Y-%m-%dT%H:%M:%S"),
        }
    ).to_csv(rainfall_members, index=False)
    pd.DataFrame(
        {
            "member_id": [f"soil_{index:02d}" for index in range(1, 7)],
            "source": ["nwm"] * 6,
            "member_file": ["soil.csv"] * 6,
            "time": pd.date_range("2018-12-31", periods=6, freq="60D").strftime("%Y-%m-%dT%H:%M:%S"),
        }
    ).to_csv(soil_moisture_members, index=False)

    artifacts = build_inland_event_artifacts(
        {
            "project": {"name": "greensboro"},
            "events": {"target_event_count": 20},
            "sampling": {
                "spacing": "log",
                "return_period_min_years": 2,
                "return_period_max_years": 500,
                "hybrid_splice_quantile": 0.8,
                "tail_sample_fraction": 0.3,
                "severity_bands": [
                    {"severity_band": "common", "rp_min_years": 0.0, "rp_max_years": 10.0},
                    {"severity_band": "significant", "rp_min_years": 10.0, "rp_max_years": 50.0},
                    {"severity_band": "rare", "rp_min_years": 50.0, "rp_max_years": 100.0},
                    {"severity_band": "extreme", "rp_min_years": 100.0, "rp_max_years": 500.0},
                    {"severity_band": "beyond_design", "rp_min_years": 500.0, "rp_max_years": None},
                ],
            },
            "template_assignment": {"random_seed": 7},
            "event_catalog": {
                "forcing_members": {
                    "streamflow": "data/sources/usgs_streamgages/streamflow_members.csv",
                    "rainfall": "data/sources/aorc_sst/rainfall_members.csv",
                    "soil_moisture": "data/sources/nwm/soil_moisture.csv",
                },
                "pairing": {
                    "rainfall": {"strategy": "inland_rainfall_pairing_priority", "window_days": 90, "seed": 5},
                    "soil_moisture": {"strategy": "inland_antecedent_moisture_pairing", "lead_time_hours": 24},
                },
            },
        },
        {"location_root": location_root, "location_name": "greensboro", "scenario": {"name": "base"}},
    )

    catalog = artifacts.catalog
    assert len(catalog) == 20
    assert catalog["event_id"].is_unique
    assert catalog["event_id"].str.startswith("usgs_design_").all()
    assert catalog["streamflow_template_member_id"].notna().all()
    assert catalog["streamflow_scale_factor"].notna().all()
    assert catalog["streamflow_scale_factor"].gt(0).all()
    assert set(catalog["sampling_region"]) == {"body", "tail"}
    assert round(float(catalog["probability_weight"].sum()), 6) == 1.0
    assert catalog["sample_rp_years"].max() > 100
    assert set(catalog["streamflow_member_id"]).issubset(
        {f"02095000_20200{index:02d}T000000" for index in range(1, 7)}
    )


def test_build_usgs_streamflow_event_members_declusters_network_peaks_and_preserves_site_ids(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    output = location_root / "data/sources/usgs_streamgages/streamflow_members.csv"
    times = pd.date_range("2020-01-01", periods=220, freq="h")
    rows = []
    for site_no, first_peak, second_peak in [
        ("02095000", 36, 150),
        ("02095500", 38, 152),
    ]:
        values = [100.0] * len(times)
        values[first_peak] = 1200.0 if site_no == "02095000" else 1600.0
        values[second_peak] = 2200.0 if site_no == "02095000" else 1500.0
        for time, value in zip(times, values):
            rows.append({"site_no": site_no, "time": time.isoformat(), "discharge_cfs": value})
    records = pd.DataFrame(rows)
    config = {
        "extremes": {
            "pot": {
                "threshold_quantile": 0.9,
                "min_peak_distance_hours": 48,
            }
        },
        "event_catalog": {
            "forcing_members": {
                "streamflow": "data/sources/usgs_streamgages/streamflow_members.csv",
            }
        },
    }

    members = build_usgs_streamflow_event_members(
        config,
        {"location_root": location_root},
        streamflow_records=records,
    )

    assert output.exists()
    assert members["site_no"].tolist() == ["02095000", "02095500"]
    assert members["member_id"].tolist() == ["02095000_20200107T060000", "02095500_20200102T140000"]
    assert members["event_time"].tolist() == ["2020-01-07T06:00:00", "2020-01-02T14:00:00"]
    assert "site_threshold_cfs" in members.columns
    assert members["site_threshold_cfs"].notna().all()
    assert members["network_site_count"].tolist() == [2, 2]
    assert members["sampling_region"].tolist() == ["tail", "body"]
    assert round(float(members["probability_weight"].sum()), 6) == 1.0


def test_build_usgs_streamflow_event_members_reads_configured_streamflow_record_output(tmp_path):
    location_root = tmp_path / "locations" / "greensboro"
    records_path = location_root / "data/sources/usgs_streamgages/streamflow_records.csv"
    records_path.parent.mkdir(parents=True)
    times = pd.date_range("2020-01-01", periods=121, freq="h")
    records = pd.DataFrame(
        {
            "site_no": ["02095000"] * len(times),
            "time": times.astype(str),
            "discharge_cfs": [100.0] * 30 + [1500.0] + [100.0] * 88 + [2200.0] + [100.0],
        }
    )
    records.to_csv(records_path, index=False)
    config = {
        "collection": {
            "usgs_streamgages": {
                "streamflow_records": {
                    "output": "data/sources/usgs_streamgages/streamflow_records.csv",
                }
            }
        },
        "extremes": {
            "pot": {
                "threshold_quantile": 0.9,
                "min_peak_distance_hours": 48,
            }
        },
        "event_catalog": {
            "forcing_members": {
                "streamflow": "data/sources/usgs_streamgages/streamflow_members.csv",
            }
        },
    }

    members = build_usgs_streamflow_event_members(config, {"location_root": location_root})

    assert len(members) == 2
    assert (location_root / "data/sources/usgs_streamgages/streamflow_members.csv").exists()


def test_write_wflow_sfincs_handoff_manifest_records_domain_set_events(tmp_path):
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_001", "evt_002"],
            "wflow_event_dir": ["data/wflow/events/evt_001", "data/wflow/events/evt_002"],
        }
    )
    config = {
        "wflow": {
            "handoff": {
                "manifest": "data/wflow/domain_set_handoff.yaml",
                "source_variable": "river_q",
                "source_standard_name": "river_water__volume_flow_rate",
                "target": "sfincs_discharge_forcing",
            },
            "domain_set": {
                "event_catalog_scope": "shared_across_domain_set",
                "submodels": [{"id": "upper_haw", "outlet_site_no": "02095000"}],
            },
        },
        "sfincs_domain_set": {
            "event_catalog_scope": "shared_across_domain_set",
            "evaluation_merge": "max_depth_per_asset_with_source_domain",
            "domains": [{"id": "greensboro_core"}],
        },
        "inland_coupling": {
            "forcing_mode": "dual_fluvial_pluvial",
            "direct_rainfall": {"enabled": True},
        },
    }

    manifest_path = write_wflow_sfincs_handoff_manifest(
        catalog,
        config,
        {"location_root": tmp_path / "locations/greensboro"},
    )

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["forcing_mode"] == "dual_fluvial_pluvial"
    assert manifest["source_variable"] == "river_q"
    assert manifest["event_catalog_scope"] == "shared_across_domain_set"
    assert manifest["submodels"] == [{"id": "upper_haw", "outlet_site_no": "02095000"}]
    assert manifest["sfincs_domains"] == [{"id": "greensboro_core"}]
    assert [event["event_id"] for event in manifest["events"]] == ["evt_001", "evt_002"]
    assert manifest["events"][0]["discharge_forcing"].endswith("evt_001/sfincs_discharge.nc")


def test_build_event_catalog_records_same_historical_wave_analog_for_coastal_waves(tmp_path):
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "event_summary_csv": tmp_path / "events/surge_event_members_summary.csv",
        "event_members_nc": tmp_path / "events/surge_event_members.nc",
        "event_catalog_csv": tmp_path / "catalog/event_catalog.csv",
        "event_catalog_audit_json": tmp_path / "catalog/event_catalog_audit.json",
        "era5_waves_nc": tmp_path / "waves/era5.nc",
        "scenario": {"name": "base", "slr_offset_m": 0.0},
    }
    paths["event_summary_csv"].parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": ["evt_0001"],
            "sample_rp_years": [50.0],
            "sampling_region": ["tail"],
            "sampling_weight": [0.25],
            "template_id": ["tpl_0017"],
            "template_peak_time": ["2018-01-04T17:00:00"],
            "peak": [2.9],
            "absolute_peak_m": [2.9],
            "valid_start_hour": [-72],
            "valid_end_hour": [72],
        }
    ).to_csv(paths["event_summary_csv"], index=False)

    catalog = build_event_catalog({"coastal_waves": True}, paths)
    audit = json.loads(paths["event_catalog_audit_json"].read_text(encoding="utf-8"))

    assert audit["passed"] is True
    assert catalog["coastal_analog_id"].tolist() == ["tpl_0017"]
    assert catalog["coastal_analog_peak_time"].tolist() == ["2018-01-04T17:00:00"]
    assert catalog["snapwave_source"].tolist() == ["era5"]
    assert catalog["snapwave_member_file"].tolist() == [str(paths["era5_waves_nc"])]
    assert catalog["snapwave_member_id"].tolist() == ["tpl_0017"]
    assert catalog["snapwave_valid_start_time"].tolist() == ["2018-01-01T17:00:00"]
    assert catalog["snapwave_valid_end_time"].tolist() == ["2018-01-07T17:00:00"]
    assert catalog["snapwave_pairing_policy"].tolist() == ["same_historical_analog"]


def test_attach_forcing_members_records_pairing_policy():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_0001", "evt_0002", "evt_0003"],
            "rainfall_source": [pd.NA, pd.NA, pd.NA],
            "rainfall_member_file": [pd.NA, pd.NA, pd.NA],
            "rainfall_member_id": [pd.NA, pd.NA, pd.NA],
        }
    )
    rainfall = pd.DataFrame(
        {
            "member_id": ["rain_a", "rain_b"],
            "member_file": ["a.nc", "b.nc"],
            "source": ["aorc_sst", "aorc_sst"],
        }
    )

    paired = attach_forcing_members(
        catalog,
        rainfall,
        forcing="rainfall",
        policy={"strategy": "independent_permutation", "seed": 7},
    )

    assert paired["rainfall_source"].tolist() == ["aorc_sst", "aorc_sst", "aorc_sst"]
    assert sorted(paired["rainfall_member_id"].head(2).tolist()) == ["rain_a", "rain_b"]
    assert paired["rainfall_pairing_policy"].tolist() == ["independent_permutation"] * 3
    assert paired["rainfall_pairing_seed"].tolist() == [7, 7, 7]


def test_attach_forcing_members_keeps_empty_forcing_schema():
    catalog = pd.DataFrame({"event_id": ["evt_0001"]})
    soil_moisture = pd.DataFrame(columns=["member_id", "member_file", "time", "soil_moisture_mean"])

    paired = attach_forcing_members(
        catalog,
        soil_moisture,
        forcing="soil_moisture",
        policy={
            "strategy": "antecedent_to_forcing",
            "reference_forcing": "rainfall",
            "lead_time_hours": 24,
        },
    )

    assert "soil_moisture_member_id" in paired
    assert "soil_moisture_pairing_policy" in paired
    assert paired["soil_moisture_member_id"].isna().all()
    assert paired["soil_moisture_pairing_policy"].isna().all()


def test_rebuild_forcing_pairing_replaces_one_forcing_from_member_path(tmp_path):
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_0001"],
            "rainfall_source": ["old"],
            "rainfall_member_file": ["old.nc"],
            "rainfall_member_id": ["old_rain"],
            "rainfall_pairing_policy": ["seasonal_window_permutation"],
            "soil_moisture_member_id": ["soil_a"],
        }
    )
    rainfall_csv = tmp_path / "rainfall_members.csv"
    pd.DataFrame(
        {
            "member_id": ["rain_a"],
            "member_file": ["rain_a.nc"],
            "source": ["aorc_sst"],
        }
    ).to_csv(rainfall_csv, index=False)

    rebuilt = rebuild_forcing_pairing(
        catalog,
        rainfall_csv,
        "rainfall",
        {"strategy": "independent_permutation", "seed": 3},
    )

    assert rebuilt["rainfall_source"].tolist() == ["aorc_sst"]
    assert rebuilt["rainfall_member_id"].tolist() == ["rain_a"]
    assert rebuilt["rainfall_pairing_policy"].tolist() == ["independent_permutation"]
    assert rebuilt["soil_moisture_member_id"].tolist() == ["soil_a"]


def test_attach_forcing_members_can_pair_within_seasonal_window():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_winter", "evt_summer"],
            "coastal_template_peak_time": ["2018-01-15T00:00:00", "2018-08-15T00:00:00"],
        }
    )
    rainfall = pd.DataFrame(
        {
            "member_id": ["rain_jan", "rain_aug"],
            "member_file": ["jan.nc", "aug.nc"],
            "source": ["aorc_sst", "aorc_sst"],
            "storm_date": ["2020-01-20T00", "2020-08-10T00"],
        }
    )

    paired = attach_forcing_members(
        catalog,
        rainfall,
        forcing="rainfall",
        policy={
            "strategy": "seasonal_window_permutation",
            "seed": 3,
            "window_days": 45,
        },
    )

    assert paired["rainfall_member_id"].tolist() == ["rain_jan", "rain_aug"]
    assert paired["rainfall_member_time"].tolist() == ["2020-01-20T00", "2020-08-10T00"]
    assert paired["rainfall_pairing_policy"].tolist() == ["seasonal_window_permutation"] * 2
    assert paired["rainfall_pairing_window_days"].tolist() == [45, 45]


def test_attach_forcing_members_can_fallback_to_nearest_seasonal_member():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_summer"],
            "coastal_template_peak_time": ["2018-08-15T00:00:00"],
        }
    )
    rainfall = pd.DataFrame(
        {
            "member_id": ["rain_jan"],
            "member_file": ["jan.nc"],
            "source": ["aorc_sst"],
            "storm_date": ["2020-01-20T00"],
        }
    )

    paired = attach_forcing_members(
        catalog,
        rainfall,
        forcing="rainfall",
        policy={
            "strategy": "seasonal_window_permutation",
            "seed": 3,
            "window_days": 45,
            "fallback_strategy": "nearest",
        },
    )

    assert paired["rainfall_member_id"].tolist() == ["rain_jan"]
    assert paired["rainfall_member_time"].tolist() == ["2020-01-20T00"]
    assert paired["rainfall_pairing_policy"].tolist() == ["seasonal_window_permutation"]
    assert paired["rainfall_pairing_window_days"].tolist() == [45]


def test_attach_forcing_members_can_pair_antecedent_soil_moisture_to_rainfall():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_0001", "evt_0002"],
            "rainfall_member_time": ["2020-03-10T12:00:00", "2020-09-02T06:00:00"],
        }
    )
    soil_moisture = pd.DataFrame(
        {
            "member_id": ["soil_mar09", "soil_mar10", "soil_sep01", "soil_sep02"],
            "member_file": ["soil.csv"] * 4,
            "source": ["nwm"] * 4,
            "time": [
                "2020-03-09T12:00:00",
                "2020-03-10T12:00:00",
                "2020-09-01T06:00:00",
                "2020-09-02T06:00:00",
            ],
        }
    )

    paired = attach_forcing_members(
        catalog,
        soil_moisture,
        forcing="soil_moisture",
        policy={
            "strategy": "antecedent_to_forcing",
            "reference_forcing": "rainfall",
            "lead_time_hours": 24,
        },
    )

    assert paired["soil_moisture_member_id"].tolist() == ["soil_mar09", "soil_sep01"]
    assert paired["soil_moisture_member_time"].tolist() == [
        "2020-03-09T12:00:00",
        "2020-09-01T06:00:00",
    ]
    assert paired["soil_moisture_pairing_policy"].tolist() == ["antecedent_to_forcing"] * 2
    assert paired["soil_moisture_pairing_reference_time"].tolist() == [
        "2020-03-10T12:00:00",
        "2020-09-02T06:00:00",
    ]
    assert paired["soil_moisture_pairing_lag_hours"].tolist() == [24, 24]


def test_build_event_catalog_attaches_configured_rainfall_members(tmp_path):
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "event_summary_csv": tmp_path / "events/surge_event_members_summary.csv",
        "event_members_nc": tmp_path / "events/surge_event_members.nc",
        "event_catalog_csv": tmp_path / "catalog/event_catalog.csv",
        "scenario": {"name": "base", "slr_offset_m": 0.0},
    }
    paths["event_summary_csv"].parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": ["evt_0001", "evt_0002"],
            "sample_rp_years": [10.0, 100.0],
            "peak": [1.2, 1.8],
            "absolute_peak_m": [1.2, 1.8],
            "valid_start_hour": [-24, -36],
            "valid_end_hour": [48, 60],
        }
    ).to_csv(paths["event_summary_csv"], index=False)
    rainfall_csv = tmp_path / "aorc_sst/rainfall_members.csv"
    rainfall_csv.parent.mkdir()
    pd.DataFrame(
        {
            "member_id": ["rain_a", "rain_b"],
            "member_file": ["rain_a.nc", "rain_b.nc"],
            "source": ["aorc_sst", "aorc_sst"],
        }
    ).to_csv(rainfall_csv, index=False)
    config = {
        "event_catalog": {
            "forcing_members": {"rainfall": str(rainfall_csv)},
            "pairing": {"rainfall": {"strategy": "independent_permutation", "seed": 11}},
        },
    }

    catalog = build_event_catalog(config, paths)

    assert sorted(catalog["rainfall_member_id"].tolist()) == ["rain_a", "rain_b"]
    assert catalog["rainfall_source"].tolist() == ["aorc_sst", "aorc_sst"]
    assert catalog["rainfall_pairing_policy"].tolist() == ["independent_permutation", "independent_permutation"]
    assert catalog["rainfall_pairing_seed"].tolist() == [11, 11]


def test_build_event_catalog_supports_configured_seasonal_rainfall_pairing(tmp_path):
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "event_summary_csv": tmp_path / "events/surge_event_members_summary.csv",
        "event_members_nc": tmp_path / "events/surge_event_members.nc",
        "event_catalog_csv": tmp_path / "catalog/event_catalog.csv",
        "scenario": {"name": "base", "slr_offset_m": 0.0},
    }
    paths["event_summary_csv"].parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": ["evt_0001", "evt_0002"],
            "sample_rp_years": [5.0, 100.0],
            "template_peak_time": ["2018-02-01T00:00:00", "2018-09-01T00:00:00"],
            "peak": [1.2, 1.8],
            "absolute_peak_m": [1.2, 1.8],
            "valid_start_hour": [-24, -36],
            "valid_end_hour": [48, 60],
        }
    ).to_csv(paths["event_summary_csv"], index=False)
    rainfall_csv = tmp_path / "aorc_sst/rainfall_members.csv"
    rainfall_csv.parent.mkdir()
    pd.DataFrame(
        {
            "member_id": ["rain_feb", "rain_sep"],
            "member_file": ["rain_feb.nc", "rain_sep.nc"],
            "source": ["aorc_sst", "aorc_sst"],
            "storm_date": ["2020-02-03T00", "2020-09-04T00"],
        }
    ).to_csv(rainfall_csv, index=False)
    config = {
        "event_catalog": {
            "forcing_members": {"rainfall": str(rainfall_csv)},
            "pairing": {
                "rainfall": {
                    "strategy": "seasonal_window_permutation",
                    "seed": 11,
                    "window_days": 30,
                }
            },
        },
    }

    catalog = build_event_catalog(config, paths)

    assert catalog["rainfall_member_id"].tolist() == ["rain_feb", "rain_sep"]
    assert catalog["rainfall_member_time"].tolist() == ["2020-02-03T00", "2020-09-04T00"]
    assert catalog["rainfall_pairing_window_days"].tolist() == [30, 30]


def test_build_event_catalog_normalizes_raw_soil_moisture_members(tmp_path):
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "event_summary_csv": tmp_path / "events/surge_event_members_summary.csv",
        "event_members_nc": tmp_path / "events/surge_event_members.nc",
        "event_catalog_csv": tmp_path / "catalog/event_catalog.csv",
        "scenario": {"name": "base", "slr_offset_m": 0.0},
    }
    paths["event_summary_csv"].parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": ["evt_0001"],
            "sample_rp_years": [5.0],
            "template_peak_time": ["2018-01-02T00:00:00"],
            "peak": [1.2],
            "absolute_peak_m": [1.2],
            "valid_start_hour": [-24],
            "valid_end_hour": [48],
        }
    ).to_csv(paths["event_summary_csv"], index=False)
    soil_csv = tmp_path / "nwm/soil_moisture.csv"
    soil_csv.parent.mkdir()
    pd.DataFrame(
        {
            "time": ["2018-01-01T00:00:00"],
            "point_id": ["center"],
            "SOIL_M": [0.31],
        }
    ).to_csv(soil_csv, index=False)
    config = {
        "event_catalog": {
            "forcing_members": {"soil_moisture": str(soil_csv)},
            "pairing": {
                "soil_moisture": {
                    "strategy": "antecedent_to_forcing",
                    "reference_time_column": "coastal_template_peak_time",
                    "lead_time_hours": 24,
                }
            },
        },
    }

    catalog = build_event_catalog(config, paths)

    assert catalog["soil_moisture_source"].tolist() == ["nwm"]
    assert catalog["soil_moisture_member_file"].tolist() == [str(soil_csv)]
    assert catalog["soil_moisture_member_id"].tolist() == ["soil_moisture_20180101T000000"]
    assert catalog["soil_moisture_member_time"].tolist() == ["2018-01-01T00:00:00"]
    assert catalog["soil_moisture_pairing_policy"].tolist() == ["antecedent_to_forcing"]
    assert catalog["soil_moisture_pairing_lag_hours"].tolist() == [24]


def test_validate_event_catalog_flags_missing_sampling_weight_and_incomplete_forcing():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_0001"],
            "study_location": ["marshfield"],
            "event_family": ["surge_synthetic"],
            "scenario_name": ["base"],
            "sample_rp_years": [100.0],
            "sampling_region": ["tail"],
            "sampling_weight": [pd.NA],
            "coastal_source": ["cora"],
            "coastal_member_file": ["events.nc"],
            "coastal_member_id": ["evt_0001"],
            "rainfall_source": ["aorc_sst"],
            "rainfall_member_file": [pd.NA],
            "rainfall_member_id": ["rain_001"],
            "rainfall_pairing_policy": [pd.NA],
            "rainfall_pairing_seed": [pd.NA],
        }
    )

    issues = validate_event_catalog(catalog)

    assert {
        "severity": "error",
        "code": "invalid_sampling_weight",
        "event_id": "evt_0001",
        "column": "sampling_weight",
    } in issues
    assert {
        "severity": "error",
        "code": "incomplete_forcing",
        "event_id": "evt_0001",
        "forcing": "rainfall",
        "column": "rainfall_member_file",
    } in issues
    assert {
        "severity": "error",
        "code": "incomplete_forcing",
        "event_id": "evt_0001",
        "forcing": "rainfall",
        "column": "rainfall_pairing_policy",
    } in issues


def test_validate_event_catalog_exempts_historical_reference_rows_from_sampling_weight():
    catalog = pd.DataFrame(
        {
            "event_id": ["synthetic_0001", "historical_20110825T180000"],
            "study_location": ["marshfield", "marshfield"],
            "event_family": ["surge_synthetic", "historical_compound_tail"],
            "scenario_name": ["base", "base"],
            "sample_rp_years": [100.0, 75.0],
            "sampling_region": ["tail", "tail"],
            "sampling_weight": [pd.NA, pd.NA],
            "event_origin": ["synthetic_tail", "historical_tail"],
            "catalog_role": [pd.NA, "historical_reference"],
            "coastal_source": ["cora", "cora"],
            "coastal_member_file": ["events.nc", "events.nc"],
            "coastal_member_id": ["synthetic_0001", "historical_20110825T180000"],
        }
    )

    issues = validate_event_catalog(catalog)
    flagged = {
        issue["event_id"] for issue in issues if issue["code"] == "invalid_sampling_weight"
    }

    # The synthetic row is still required to carry a sampling weight; the observed reference row
    # sits outside the probability budget and is exempt.
    assert flagged == {"synthetic_0001"}


def test_validate_event_catalog_flags_incomplete_seasonal_pairing_metadata():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_0001"],
            "study_location": ["marshfield"],
            "event_family": ["surge_synthetic"],
            "scenario_name": ["base"],
            "sample_rp_years": [100.0],
            "sampling_region": ["tail"],
            "sampling_weight": [0.5],
            "coastal_source": ["cora"],
            "coastal_member_file": ["events.nc"],
            "coastal_member_id": ["evt_0001"],
            "rainfall_source": ["aorc_sst"],
            "rainfall_member_file": ["rain.nc"],
            "rainfall_member_id": ["rain_001"],
            "rainfall_member_time": [pd.NA],
            "rainfall_pairing_policy": ["seasonal_window_permutation"],
            "rainfall_pairing_seed": [42],
            "rainfall_pairing_window_days": [pd.NA],
        }
    )

    issues = validate_event_catalog(catalog)

    assert {
        "severity": "error",
        "code": "incomplete_seasonal_pairing",
        "event_id": "evt_0001",
        "forcing": "rainfall",
        "column": "rainfall_member_time",
    } in issues
    assert {
        "severity": "error",
        "code": "incomplete_seasonal_pairing",
        "event_id": "evt_0001",
        "forcing": "rainfall",
        "column": "rainfall_pairing_window_days",
    } in issues


def test_validate_event_catalog_flags_missing_wave_analog_metadata():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_0001"],
            "study_location": ["marshfield"],
            "event_family": ["surge_synthetic"],
            "scenario_name": ["base"],
            "sample_rp_years": [100.0],
            "sampling_region": ["tail"],
            "sampling_weight": [0.5],
            "coastal_source": ["cora"],
            "coastal_member_file": ["events.nc"],
            "coastal_member_id": ["evt_0001"],
            "coastal_analog_id": [pd.NA],
            "snapwave_source": ["era5"],
            "snapwave_member_file": [pd.NA],
            "snapwave_member_id": [pd.NA],
            "snapwave_pairing_policy": ["same_historical_analog"],
        }
    )

    issues = validate_event_catalog(catalog, wave_analog_policy="same_historical_analog")

    assert {
        "severity": "error",
        "code": "incomplete_wave_analog",
        "event_id": "evt_0001",
        "column": "coastal_analog_id",
    } in issues
    assert {
        "severity": "error",
        "code": "incomplete_wave_analog",
        "event_id": "evt_0001",
        "column": "snapwave_member_file",
    } in issues


def test_validate_event_catalog_rejects_short_coastal_windows():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_0001", "evt_0002", "evt_0003"],
            "study_location": ["marshfield", "marshfield", "marshfield"],
            "event_family": ["surge_synthetic", "surge_synthetic", "surge_synthetic"],
            "scenario_name": ["base", "base", "base"],
            "sample_rp_years": [10.0, 50.0, 100.0],
            "sampling_region": ["body", "tail", "tail"],
            "sampling_weight": [1.0, 1.0, 1.0],
            "coastal_source": ["cora", "cora", "cora"],
            "coastal_member_file": ["events.nc", "events.nc", "events.nc"],
            "coastal_member_id": ["evt_0001", "evt_0002", "evt_0003"],
            "coastal_valid_start_hour": [-6.0, -6.0, -72.0],
            "coastal_valid_end_hour": [6.0, 6.0, 72.0],
        }
    )

    issues = validate_event_catalog(catalog)

    assert {
        "severity": "error",
        "code": "coastal_window_too_short",
        "column": "coastal_valid_start_hour/coastal_valid_end_hour",
    } in issues


def test_write_event_catalog_audit_records_pass_fail_and_issues(tmp_path):
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_0001"],
            "study_location": ["marshfield"],
            "event_family": ["surge_synthetic"],
            "scenario_name": ["base"],
            "sample_rp_years": [100.0],
            "sampling_region": ["tail"],
            "sampling_weight": [0.5],
            "coastal_source": ["cora"],
            "coastal_member_file": ["events.nc"],
            "coastal_member_id": ["evt_0001"],
        }
    )
    audit_path = tmp_path / "catalog/event_catalog_audit.json"

    audit = write_event_catalog_audit(catalog, audit_path)

    assert audit == {
        "passed": True,
        "event_count": 1,
        "issue_count": 0,
        "issues": [],
    }
    assert json.loads(audit_path.read_text()) == audit
