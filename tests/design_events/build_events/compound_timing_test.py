import pandas as pd

from design_events.build_events.compound_timing import (
    attach_empirical_rainfall_lags,
    enrich_rainfall_member_timing,
)


def test_attach_empirical_rainfall_lags_uses_weighted_observed_peak_lag():
    catalog = pd.DataFrame(
        {
            "event_id": ["design_0001"],
            "storm_type": ["nor_easter"],
            "rainfall": [98.0],
            "coastal_water_level": [1.22],
            "rainfall_member_id": ["rain_001"],
            "rainfall_member_time": ["1999-01-01T00:00:00"],
            "coastal_water_level_member_time": ["2020-01-15T00:00:00"],
        }
    )
    paired = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2010-01-10T00:00:00", "2010-08-01T00:00:00"]),
            "storm_type": ["nor_easter", "nor_easter"],
            "rainfall": [100.0, 220.0],
            "coastal_water_level": [1.20, 0.2],
            "rainfall_time": pd.to_datetime(["2010-01-10T05:00:00", "2010-08-01T18:00:00"]),
            "coastal_water_level_time": pd.to_datetime(["2010-01-10T00:00:00", "2010-08-01T00:00:00"]),
        }
    )
    members = pd.DataFrame(
        {
            "member_id": ["rain_001"],
            "storm_start": ["1999-01-01T00:00:00"],
            "rainfall_peak_time": ["1999-01-01T12:00:00"],
            "duration_hours": [72.0],
            "mean_precip_mm": [100.0],
        }
    )

    out = attach_empirical_rainfall_lags(catalog, paired, members, window_hours=72.0)

    assert float(out.loc[0, "rainfall_peak_offset_hours"]) == 5.0
    assert float(out.loc[0, "rainfall_start_offset_hours"]) == -7.0
    assert float(out.loc[0, "rainfall_end_offset_hours"]) == 65.0
    assert out.loc[0, "compound_pairing_policy"] == "conditional_empirical_weighted_knn_lag"
    assert out.loc[0, "compound_pairing_role"] == "empirical_analog_lag"
    assert out.loc[0, "scenario_timing_edge_case"] == "observed-analog-lag"
    assert out.loc[0, "empirical_lag_analog_candidate_count"] == 2


def test_weighted_lag_analog_draw_spreads_repeated_targets_across_local_pool():
    catalog = pd.DataFrame(
        {
            "event_id": [f"design_{i:04d}" for i in range(30)],
            "storm_type": ["nor_easter"] * 30,
            "rainfall": [100.0] * 30,
            "coastal_water_level": [1.2] * 30,
            "rainfall_member_id": ["rain_001"] * 30,
            "rainfall_member_time": ["1999-01-01T00:00:00"] * 30,
            "coastal_water_level_member_time": ["2020-01-15T00:00:00"] * 30,
        }
    )
    paired = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2010-01-10T00:00:00"] * 3),
            "storm_type": ["nor_easter"] * 3,
            "rainfall": [100.0, 101.0, 99.0],
            "coastal_water_level": [1.2, 1.21, 1.19],
            "rainfall_time": pd.to_datetime(
                ["2010-01-10T02:00:00", "2010-01-10T04:00:00", "2010-01-09T22:00:00"]
            ),
            "coastal_water_level_time": pd.to_datetime(["2010-01-10T00:00:00"] * 3),
        }
    )
    members = pd.DataFrame(
        {
            "member_id": ["rain_001"],
            "storm_start": ["1999-01-01T00:00:00"],
            "rainfall_peak_time": ["1999-01-01T12:00:00"],
            "duration_hours": [72.0],
            "mean_precip_mm": [100.0],
        }
    )

    out = attach_empirical_rainfall_lags(
        catalog,
        paired,
        members,
        lag_pool_size=3,
        reuse_penalty_lambda=0.5,
        seed=7,
    )

    reuse = out["empirical_lag_analog_index"].value_counts()
    assert len(reuse) > 1
    assert int(reuse.max()) < len(out)
    assert set(out["rainfall_peak_offset_hours"]).issubset({-2.0, 2.0, 4.0})


def test_enrich_rainfall_member_timing_marks_legacy_midpoint_when_peak_missing():
    members = pd.DataFrame(
        {
            "member_id": ["rain_001"],
            "member_file": ["missing/ranked-storms.csv"],
            "storm_start": ["1999-01-01T00:00:00"],
            "duration_hours": [72.0],
            "mean_precip_mm": [100.0],
        }
    )

    out = enrich_rainfall_member_timing(members)

    assert out.loc[0, "rainfall_peak_time"] == pd.Timestamp("1999-01-02T12:00:00")
    assert float(out.loc[0, "rainfall_peak_offset_from_start_hours"]) == 36.0
    assert out.loc[0, "rainfall_peak_time_source"] == "legacy_midpoint_inferred"
