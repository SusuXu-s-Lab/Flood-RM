import pandas as pd
import pytest
import numpy as np

from design_events.build_events.selection import (
    _compound_roles,
    _rainfall_offsets,
    _select_compound_rainfall_member,
    apply_compound_stress_pairing,
)


def test_historical_coastal_rainfall_pair_does_not_use_seasonal_fallback():
    reference = pd.Timestamp("2020-01-10T00:00:00")
    rainfall = pd.DataFrame(
        {
            "member_id": ["rain_001"],
            "member_time": [pd.Timestamp("1995-01-10T00:00:00")],
            "rainfall_metric": [100.0],
        }
    )

    selected = _select_compound_rainfall_member(
        rainfall,
        reference,
        "historical_coastal_rainfall_pair",
        season_window_days=7,
        real_window_hours=72.0,
        reuse_counts={},
        reuse_penalty=0.0,
    )

    assert selected is None


def test_historical_coastal_rainfall_pair_rejects_unbounded_offset():
    member = {"member_time": "1995-01-10T00:00:00"}

    with pytest.raises(RuntimeError, match="requires a rainfall member within"):
        _rainfall_offsets(
            "historical_coastal_rainfall_pair",
            72.0,
            member,
            pd.Timestamp("2020-01-10T00:00:00"),
            real_window_hours=72.0,
        )


def test_compound_roles_default_to_empirical_analog_lag_not_fixed_sensitivity_roles():
    frame = pd.DataFrame(
        {
            "event_id": [f"design_{i:04d}" for i in range(6)],
            "sample_rp_years": [1, 2, 3, 4, 5, 6],
        }
    )
    rainfall = pd.DataFrame(
        {
            "member_id": ["rain_001"],
            "member_time": [pd.Timestamp("1995-01-10T00:00:00")],
            "rainfall_peak_time": [pd.Timestamp("1995-01-10T12:00:00")],
            "rainfall_metric": [100.0],
        }
    )

    roles = _compound_roles(
        frame,
        rainfall,
        real_count=0,
        real_window_hours=72.0,
        rng=np.random.default_rng(0),
        settings={},
    )

    assert set(roles.values()) == {"empirical_analog_lag"}


def test_compound_roles_reject_fixed_timing_roles_without_sensitivity_opt_in():
    frame = pd.DataFrame({"event_id": ["design_0001"], "sample_rp_years": [10.0]})
    rainfall = pd.DataFrame(
        {
            "member_id": ["rain_001"],
            "member_time": [pd.Timestamp("1995-01-10T00:00:00")],
            "rainfall_metric": [100.0],
        }
    )

    with pytest.raises(RuntimeError, match="Fixed compound timing roles are sensitivity-only"):
        _compound_roles(
            frame,
            rainfall,
            real_count=0,
            real_window_hours=72.0,
            rng=np.random.default_rng(0),
            settings={"role_fractions": {"rainfall_before_coastal": 1.0}},
        )


def test_empirical_compound_pairing_preserves_target_based_rainfall_member():
    catalog = pd.DataFrame(
        {
            "event_id": ["design_0001"],
            "storm_type": ["nor_easter"],
            "sample_rp_years": [25.0],
            "coastal_water_level": [1.1],
            "coastal_water_level_member_time": ["2020-01-10T00:00:00"],
            "rainfall": [55.0],
            "rainfall_member_id": ["rain_target"],
            "rainfall_member_time": ["1999-01-01T00:00:00"],
            "rainfall_member_file": ["ranked-storms.csv"],
            "rainfall_scale_factor": [1.1],
        }
    )
    rainfall_members = pd.DataFrame(
        {
            "member_id": ["rain_target", "rain_extreme"],
            "member_file": ["ranked-storms.csv", "ranked-storms.csv"],
            "storm_start": ["1999-01-01T00:00:00", "2001-01-01T00:00:00"],
            "rainfall_peak_time": ["1999-01-01T12:00:00", "2001-01-01T12:00:00"],
            "duration_hours": [72.0, 72.0],
            "mean_precip_mm": [50.0, 300.0],
        }
    )
    paired = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2010-01-10T00:00:00"]),
            "storm_type": ["nor_easter"],
            "rainfall": [54.0],
            "coastal_water_level": [1.1],
            "rainfall_time": pd.to_datetime(["2010-01-10T06:00:00"]),
            "coastal_water_level_time": pd.to_datetime(["2010-01-10T00:00:00"]),
        }
    )

    out = apply_compound_stress_pairing(
        catalog,
        rainfall_members=rainfall_members,
        settings={"observed_cooccurrence_pool": paired.to_dict("records")},
    )

    assert out.loc[0, "rainfall_member_id"] == "rain_target"
    assert float(out.loc[0, "rainfall_metric_mm"]) == 55.0
    assert out.loc[0, "compound_pairing_role"] == "empirical_analog_lag"
    assert float(out.loc[0, "rainfall_peak_offset_hours"]) == 6.0
