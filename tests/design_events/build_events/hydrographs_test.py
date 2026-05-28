import numpy as np
import pandas as pd

from design_events.build_events.hydrographs import build_surge_event_artifacts


def test_build_surge_event_artifacts_exports_long_tide_resolving_members(tmp_path):
    peak_time = pd.Timestamp("2018-01-04 12:00:00")
    times = pd.date_range(peak_time - pd.Timedelta(hours=120), periods=241, freq="h")
    rel_hours = ((times - peak_time) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
    tide = 0.25 * np.sin(2.0 * np.pi * rel_hours / 12.42)
    storm = 2.0 * np.exp(-0.5 * (rel_hours / 5.0) ** 2)
    waterlevel = pd.Series(1.0 + tide + storm, index=times)

    waterlevel_csv = tmp_path / "waterlevel.csv"
    historical_peaks_csv = tmp_path / "historical_peaks.csv"
    sampled_peaks_csv = tmp_path / "sampled_peaks.csv"
    pd.DataFrame({"time": times, "value": waterlevel.values}).to_csv(waterlevel_csv, index=False)
    pd.DataFrame({"time": [peak_time], "h": [float(waterlevel.loc[peak_time])]}).to_csv(
        historical_peaks_csv,
        index=False,
    )
    pd.DataFrame(
        {
            "event_id": ["evt_0001"],
            "peak_m": [3.0],
            "sample_rp_years": [100.0],
            "sampling_region": ["tail"],
            "sampling_weight": [1.0],
            "probability_weight": [1.0],
        }
    ).to_csv(sampled_peaks_csv, index=False)
    paths = {
        "waterlevel_csv": waterlevel_csv,
        "historical_peaks_csv": historical_peaks_csv,
        "sampled_peaks_csv": sampled_peaks_csv,
        "scenario": {"name": "base", "description": "Base", "slr_offset_m": 0.0},
    }
    config = {
        "design_events": {
            "max_event_hours": 168,
            "tide_resolving_half_window_hours": 72,
            "min_event_hours": 12,
            "event_threshold_fraction": 0.10,
            "event_threshold_min_m": 0.05,
            "pre_event_baseline_hours": 24,
            "dominant_peak_ratio_max": 1.10,
        },
        "template_assignment": {"nearest_pool_size": 1, "random_seed": 2},
    }

    artifacts = build_surge_event_artifacts(config, paths)

    ds = artifacts["member_dataset"]
    total = ds["water_level_total"].sel(event_id="evt_0001").to_series().dropna()
    summary = artifacts["member_summary"].set_index("event_id").loc["evt_0001"]
    sign = np.sign(np.diff(total.to_numpy(dtype=float)))
    sign_changes = int(np.sum((sign[1:] * sign[:-1]) < 0))
    assert int(total.index.min()) == -72
    assert int(total.index.max()) == 72
    assert len(total) == 145
    assert int(ds["valid_mask"].sel(event_id="evt_0001").sum()) == 145
    assert sign_changes >= 8
    assert summary["valid_start_hour"] == -72
    assert summary["valid_end_hour"] == 72
