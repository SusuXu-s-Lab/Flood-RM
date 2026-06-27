import pandas as pd

from sfincs_runs.scenarios.joint_handoff import write_handoff


def test_write_handoff_overlays_stress_pairing_lag(tmp_path):
    peak_time = pd.Timestamp("2020-01-01T00:00:00")
    times = pd.date_range(peak_time - pd.Timedelta(hours=36), periods=73, freq="h")
    components = pd.DataFrame(
        {
            "msl": [0.0] * len(times),
            "tide": [0.0] * len(times),
            "ntr": [0.2] * len(times),
        },
        index=times,
    )
    components.loc[peak_time, "ntr"] = 0.8
    catalog = pd.DataFrame(
        {
            "event_id": ["design_0001"],
            "sample_rp_years": [25.0],
            "severity_band": ["significant"],
            "sampling_region": ["tail"],
            "sampling_weight": [1.0],
            "probability_weight": [1.0],
            "event_origin": ["synthetic_tail"],
            "storm_type": ["nor_easter"],
            "coastal_water_level_member_time": [peak_time.strftime("%Y-%m-%dT%H:%M:%S")],
            "coastal_water_level_member_id": ["coastal_20200101T000000"],
            "coastal_water_level_scale_factor": [1.0],
            "coastal_water_level": [0.9],
            "coastal_water_level_template_value": [0.8],
            "rainfall_source": ["aorc_sst"],
            "rainfall_member_file": ["rainfall_members.csv"],
            "rainfall_member_id": ["rain_001"],
            "rainfall_member_time": ["1991-06-01T00:00:00"],
            "rainfall_peak_time": ["1991-06-01T12:00:00"],
            "rainfall_peak_time_source": ["event_window_hourly_peak"],
            "rainfall_peak_offset_from_start_hours": [12.0],
            "rainfall_duration_hours": [72.0],
            "rainfall_pairing_policy": ["copula_joint_field_preserving_analog"],
            "rainfall_pairing_seed": [7],
            "rainfall_pairing_lag_hours": [pd.NA],
            "rainfall_realization_lag_hours": [0.0],
        }
    )
    stress_catalog = catalog[["event_id"]].copy()
    stress_catalog["rainfall_start_offset_hours"] = [-30.0]
    stress_catalog["rainfall_pairing_lag_hours"] = [6.0]
    stress_catalog["rainfall_peak_offset_hours"] = [6.0]
    stress_catalog["rainfall_end_offset_hours"] = [42.0]
    stress_catalog["compound_pairing_policy"] = ["operationally_severe_plausible_dependence"]
    stress_path = tmp_path / "resilience_stress_training_catalog.csv"
    stress_catalog.to_csv(stress_path, index=False)
    paths = {
        "location_name": "marshfield",
        "event_members_nc": tmp_path / "surge_event_members.nc",
        "event_summary_csv": tmp_path / "surge_event_members_summary.csv",
        "event_catalog_csv": tmp_path / "event_catalog.csv",
        "event_acceptance_json": tmp_path / "event_acceptance.json",
        "event_catalog_audit_json": tmp_path / "event_catalog_audit.json",
        "lagtimes_csv": tmp_path / "lagtimes.csv",
        "resilience_stress_training_catalog_csv": stress_path,
    }

    out = write_handoff(
        catalog,
        components,
        config={
            "event_drivers": ["coastal_water_level", "rainfall"],
            "coastal_waves": False,
            "design_events": {"tide_resolving_half_window_hours": 36},
        },
        paths=paths,
    )

    assert float(out["catalog"].loc[0, "rainfall_start_offset_hours"]) == -30.0
    assert float(out["catalog"].loc[0, "rainfall_pairing_lag_hours"]) == 6.0
    assert float(out["catalog"].loc[0, "rainfall_peak_offset_hours"]) == 6.0
    assert float(out["catalog"].loc[0, "rainfall_end_offset_hours"]) == 42.0
    assert out["catalog"].loc[0, "rainfall_peak_time"] == "1991-06-01T12:00:00"


def test_write_handoff_rejects_synthetic_copula_rows_without_bounded_lag(tmp_path):
    peak_time = pd.Timestamp("2020-01-01T00:00:00")
    times = pd.date_range(peak_time - pd.Timedelta(hours=36), periods=73, freq="h")
    components = pd.DataFrame(
        {"msl": [0.0] * len(times), "tide": [0.0] * len(times), "ntr": [0.2] * len(times)},
        index=times,
    )
    components.loc[peak_time, "ntr"] = 0.8
    catalog = pd.DataFrame(
        {
            "event_id": ["design_0001"],
            "sample_rp_years": [25.0],
            "severity_band": ["significant"],
            "sampling_region": ["tail"],
            "sampling_weight": [1.0],
            "probability_weight": [1.0],
            "event_origin": ["synthetic_tail"],
            "storm_type": ["nor_easter"],
            "coastal_water_level_member_time": [peak_time.strftime("%Y-%m-%dT%H:%M:%S")],
            "coastal_water_level_member_id": ["coastal_20200101T000000"],
            "coastal_water_level_scale_factor": [1.0],
            "coastal_water_level": [0.9],
            "rainfall_source": ["aorc_sst"],
            "rainfall_member_file": ["rainfall_members.csv"],
            "rainfall_member_id": ["rain_001"],
            "rainfall_member_time": ["1991-06-01T00:00:00"],
            "rainfall_pairing_policy": ["copula_joint_field_preserving_analog"],
            "rainfall_realization_lag_hours": [0.0],
        }
    )
    paths = {
        "location_name": "marshfield",
        "event_members_nc": tmp_path / "surge_event_members.nc",
        "event_summary_csv": tmp_path / "surge_event_members_summary.csv",
        "event_catalog_csv": tmp_path / "event_catalog.csv",
        "event_acceptance_json": tmp_path / "event_acceptance.json",
        "event_catalog_audit_json": tmp_path / "event_catalog_audit.json",
    }

    import pytest

    with pytest.raises(RuntimeError, match="finite rainfall timing offsets"):
        write_handoff(
            catalog,
            components,
            config={
                "event_drivers": ["coastal_water_level", "rainfall"],
                "coastal_waves": False,
                "design_events": {"tide_resolving_half_window_hours": 36},
            },
            paths=paths,
        )
