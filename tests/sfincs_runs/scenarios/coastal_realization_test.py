import numpy as np
import pandas as pd
import pytest

from sfincs_runs.scenarios.coastal_realization import (
    build_timeseries,
    build_coastal_hydrograph_from_analog,
)


def _synthetic_waterlevel():
    # 10 days hourly: tide + a storm bump peaking at a known time.
    idx = pd.date_range("2012-10-25", periods=240, freq="1h")
    t = np.arange(240)
    tide = 0.6 * np.sin(2 * np.pi * t / 12.42)
    peak_time = idx[120]
    storm = 1.5 * np.exp(-((t - 120) ** 2) / (2 * 18**2))
    return pd.Series(tide + storm, index=idx), peak_time


def test_scaling_amplifies_anomaly_preserves_baseline_and_centres_on_peak():
    wl, peak_time = _synthetic_waterlevel()
    base = build_coastal_hydrograph_from_analog(wl, peak_time, 1.0, window_hours=72)
    scaled = build_coastal_hydrograph_from_analog(wl, peak_time, 2.0, window_hours=72)

    assert base.index.name == "relative_hour"
    assert 0 in base.index  # window is centred on the analog peak
    baseline = float(min(base.to_numpy()))
    # baseline preserved; anomaly above baseline doubled
    np.testing.assert_allclose(scaled.to_numpy() - baseline, (base.to_numpy() - baseline) * 2.0, atol=1e-9)
    # the storm peak is amplified
    assert scaled.max() > base.max()


def test_msl_offset_is_a_rigid_translation():
    wl, peak_time = _synthetic_waterlevel()
    base = build_coastal_hydrograph_from_analog(wl, peak_time, 1.0, window_hours=48)
    shifted = build_coastal_hydrograph_from_analog(wl, peak_time, 1.0, window_hours=48, msl_offset_m=0.5)
    np.testing.assert_allclose(shifted.to_numpy(), base.to_numpy() + 0.5, atol=1e-9)


def test_build_timeseries_from_catalog_row():
    wl, peak_time = _synthetic_waterlevel()
    row = {"coastal_water_level_member_time": str(peak_time), "coastal_water_level_scale_factor": 1.3}
    forcing = build_timeseries(row, wl, window_hours=72)
    assert forcing["forcing_variable"] == "coastal_water_level"
    assert forcing["scale_factor"] == 1.3
    assert forcing["h"].index.name == "relative_hour"
    assert np.isfinite(forcing["h"].to_numpy()).all()


def test_missing_window_and_bad_scale_raise():
    wl, peak_time = _synthetic_waterlevel()
    with pytest.raises(ValueError):
        build_coastal_hydrograph_from_analog(wl, pd.Timestamp("1950-01-01"), 1.0, window_hours=24)
    with pytest.raises(ValueError):
        build_coastal_hydrograph_from_analog(wl, peak_time, 0.0)
