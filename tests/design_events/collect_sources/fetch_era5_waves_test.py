import pandas as pd

from design_events.collect_sources.era5_waves.cds import build_cds_request_payload


def test_build_cds_request_payload_uses_north_west_south_east_bbox():
    payload = build_cds_request_payload(
        bbox_wgs84=(-70.78, 42.04, -70.62, 42.16),  # (W, S, E, N)
        time_window=(pd.Timestamp("2020-03-10"), pd.Timestamp("2020-03-10 23:00")),
    )
    # CDS expects [N, W, S, E]
    assert payload["area"] == [42.16, -70.78, 42.04, -70.62]


def test_build_cds_request_payload_uses_canonical_wave_variables():
    payload = build_cds_request_payload(
        bbox_wgs84=(-71.0, 42.0, -70.0, 43.0),
        time_window=(pd.Timestamp("2020-03-10"), pd.Timestamp("2020-03-10 23:00")),
    )
    assert payload["data_format"] == "netcdf"
    assert payload["download_format"] == "unarchived"
    assert payload["variable"] == [
        "significant_height_of_combined_wind_waves_and_swell",
        "peak_wave_period",
        "mean_wave_direction",
        "wave_spectral_directional_width",
    ]


def test_build_cds_request_payload_expands_dates_for_single_day():
    payload = build_cds_request_payload(
        bbox_wgs84=(-71.0, 42.0, -70.0, 43.0),
        time_window=(pd.Timestamp("2020-03-10"), pd.Timestamp("2020-03-10 23:00")),
    )
    assert payload["year"] == ["2020"]
    assert payload["month"] == ["03"]
    assert payload["day"] == ["10"]
    assert payload["time"] == [f"{h:02d}:00" for h in range(24)]


def test_build_cds_request_payload_expands_dates_across_month_boundary():
    payload = build_cds_request_payload(
        bbox_wgs84=(-71.0, 42.0, -70.0, 43.0),
        time_window=(pd.Timestamp("2020-01-30"), pd.Timestamp("2020-02-02 12:00")),
    )
    assert payload["year"] == ["2020"]
    assert payload["month"] == ["01", "02"]
    assert set(payload["day"]) == {"30", "31", "01", "02"}


def test_build_cds_request_payload_honors_variable_override():
    payload = build_cds_request_payload(
        bbox_wgs84=(-71.0, 42.0, -70.0, 43.0),
        time_window=(pd.Timestamp("2020-03-10"), pd.Timestamp("2020-03-10 23:00")),
        variables=["peak_wave_period"],
    )
    assert payload["variable"] == ["peak_wave_period"]
