import pytest
import pandas as pd

from sfincs_runs.scenarios.timing import (
    DriverWindow,
    MissingTimingDescriptorsError,
    plan_event_forcing_support_window,
    plan_forcing_support_window,
)


def test_plan_forcing_support_window_uses_driver_union_plus_padding():
    plan = plan_forcing_support_window(
        event_reference_time=pd.Timestamp("2000-01-02 00:00:00"),
        driver_windows=[
            DriverWindow("coastal", start_offset_hours=-6, peak_offset_hours=0, end_offset_hours=18),
            DriverWindow("rainfall", start_offset_hours=-24, peak_offset_hours=-12, end_offset_hours=48),
        ],
        spinup_hours=6,
        drain_down_hours=12,
    )

    assert plan.run_start == pd.Timestamp("2000-01-01 00:00:00") - pd.Timedelta(hours=6)
    assert plan.run_stop == pd.Timestamp("2000-01-04 00:00:00") + pd.Timedelta(hours=12)
    assert plan.duration_hours == 90
    assert plan.timing_policy == "descriptors"
    assert plan.driver_windows[0].driver == "coastal"


def test_plan_forcing_support_window_applies_minimum_and_maximum_duration_caps():
    short = plan_forcing_support_window(
        event_reference_time=pd.Timestamp("2000-01-02 00:00:00"),
        driver_windows=[
            DriverWindow("coastal", start_offset_hours=-3, peak_offset_hours=0, end_offset_hours=3),
        ],
        min_run_hours=24,
    )
    long = plan_forcing_support_window(
        event_reference_time=pd.Timestamp("2000-01-02 00:00:00"),
        driver_windows=[
            DriverWindow("rainfall", start_offset_hours=-96, peak_offset_hours=-12, end_offset_hours=120),
        ],
        max_run_hours=168,
    )

    assert short.duration_hours == 24
    assert short.run_start == pd.Timestamp("2000-01-01 12:00:00")
    assert short.run_stop == pd.Timestamp("2000-01-02 12:00:00")
    assert long.duration_hours == 168
    assert long.run_start == pd.Timestamp("1999-12-29 12:00:00")
    assert long.run_stop == pd.Timestamp("2000-01-05 12:00:00")


def test_plan_event_forcing_support_window_reads_timing_descriptors():
    plan = plan_event_forcing_support_window(
        {
            "event_reference_time": "2000-01-02 12:00:00",
            "coastal_start_offset_hours": -12,
            "coastal_peak_offset_hours": 0,
            "coastal_end_offset_hours": 18,
            "rainfall_start_offset_hours": -24,
            "rainfall_peak_offset_hours": -6,
            "rainfall_end_offset_hours": 48,
        },
        spinup_hours=6,
        drain_down_hours=12,
    )

    assert plan.run_start == pd.Timestamp("2000-01-01 12:00:00") - pd.Timedelta(hours=6)
    assert plan.run_stop == pd.Timestamp("2000-01-04 12:00:00") + pd.Timedelta(hours=12)
    assert plan.timing_policy == "descriptors"
    assert [window.driver for window in plan.driver_windows] == ["coastal", "rainfall"]


def test_plan_event_forcing_support_window_reads_event_catalog_coastal_columns():
    plan = plan_event_forcing_support_window(
        {
            "coastal_template_peak_time": "2018-01-04 17:00:00",
            "coastal_valid_start_hour": -6,
            "coastal_valid_end_hour": 6,
            "snapwave_valid_start_time": "2018-01-04T11:00:00",
            "snapwave_valid_end_time": "2018-01-04T23:00:00",
        },
        allow_legacy_inference=False,
    )

    assert plan.event_reference_time == pd.Timestamp("2018-01-04 17:00:00")
    assert plan.run_start == pd.Timestamp("2018-01-04 11:00:00")
    assert plan.run_stop == pd.Timestamp("2018-01-04 23:00:00")
    assert plan.timing_policy == "descriptors"
    assert [window.driver for window in plan.driver_windows] == ["coastal", "wave"]


def test_plan_event_forcing_support_window_labels_legacy_fallback_and_strict_rejects_it():
    legacy = plan_event_forcing_support_window(
        {"event_id": "evt_0001"},
        model_start_time=pd.Timestamp("2000-01-01 00:00:00"),
        coastal_sample_count=13,
        allow_legacy_inference=True,
    )

    assert legacy.event_reference_time == pd.Timestamp("2000-01-01 00:00:00")
    assert legacy.run_start == pd.Timestamp("2000-01-01 00:00:00")
    assert legacy.run_stop == pd.Timestamp("2000-01-01 12:00:00")
    assert legacy.timing_policy == "legacy_inferred"

    with pytest.raises(MissingTimingDescriptorsError):
        plan_event_forcing_support_window(
            {"event_id": "evt_0001"},
            model_start_time=pd.Timestamp("2000-01-01 00:00:00"),
            coastal_sample_count=13,
            allow_legacy_inference=False,
        )
