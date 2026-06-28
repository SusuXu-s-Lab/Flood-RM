import pandas as pd

from wflow_runs.replay import _catalog_rainfall_start


def test_catalog_rainfall_start_uses_event_reference_time_without_offset():
    row = pd.Series(
        {
            "event_reference_time": "1996-09-03T12:00:00",
            "rainfall_member_time": "2009-01-05T06:00:00",
        }
    )

    assert _catalog_rainfall_start(row) == pd.Timestamp("1996-09-03T12:00:00")


def test_catalog_rainfall_start_prefers_explicit_offset():
    row = pd.Series(
        {
            "event_reference_time": "2000-01-10T00:00:00",
            "rainfall_member_time": "2000-01-09T00:00:00",
            "rainfall_start_offset_hours": -12,
        }
    )

    assert _catalog_rainfall_start(row) == pd.Timestamp("2000-01-09T12:00:00")
