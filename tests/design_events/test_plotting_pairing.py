import matplotlib
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
from matplotlib.table import Table

matplotlib.use("Agg")

from design_events import plotting


def test_seasonal_pairing_diagnostics_counts_year_end_wraparound_as_in_window():
    catalog = pd.DataFrame(
        {
            "coastal_template_peak_time": ["2018-01-04T00:00:00"],
            "rainfall_member_time": ["2022-12-22T00:00:00"],
            "rainfall_member_id": ["rain_dec"],
        }
    )

    diagnostics = plotting.seasonal_pairing_diagnostics(
        catalog,
        "rainfall",
        window_days=45,
    ).iloc[0]

    assert diagnostics["paired_rows"] == 1
    assert diagnostics["in_window_rows"] == 1
    assert diagnostics["max_gap_days"] == 14


def test_plot_seasonal_pairing_shades_circular_window_segments():
    catalog = pd.DataFrame(
        {
            "coastal_template_peak_time": ["2018-01-04T00:00:00"],
            "rainfall_member_time": ["2022-12-22T00:00:00"],
            "rainfall_member_id": ["rain_dec"],
        }
    )

    fig = plotting.plot_seasonal_pairing(catalog, "rainfall", window_days=45)
    ax = fig.axes[0]

    assert "in-window=1/1" in ax.get_title()
    assert len(ax.collections) >= 4


def test_antecedent_pairing_diagnostics_reports_reference_lag():
    catalog = pd.DataFrame(
        {
            "soil_moisture_member_time": ["2020-01-01T00:00:00", "2020-01-02T06:00:00"],
            "soil_moisture_pairing_reference_time": ["2020-01-02T00:00:00", "2020-01-03T06:00:00"],
            "soil_moisture_pairing_lag_hours": [24, 24],
            "soil_moisture_pairing_policy": ["antecedent_to_forcing", "antecedent_to_forcing"],
        }
    )

    diagnostics = plotting.antecedent_pairing_diagnostics(catalog, "soil_moisture").iloc[0]

    assert diagnostics["paired_rows"] == 2
    assert diagnostics["configured_lag_hours"] == 24
    assert diagnostics["median_lag_hours"] == 24
    assert diagnostics["on_lag_rows"] == 2


def test_forcing_pairing_diagnostics_uses_inland_event_reference_time():
    catalog = pd.DataFrame(
        {
            "event_reference_time": ["2021-06-01T00:00:00"],
            "rainfall_member_time": ["2020-06-15T00:00:00"],
            "rainfall_member_id": ["rain_jun"],
            "rainfall_pairing_policy": ["seasonal_window_permutation"],
            "rainfall_pairing_window_days": [45],
        }
    )

    diagnostics = plotting.forcing_pairing_diagnostics(catalog, forcings=["rainfall"]).iloc[0]

    assert diagnostics["forcing"] == "rainfall"
    assert diagnostics["policy"] == "seasonal_window_permutation"
    assert diagnostics["paired_rows"] == 1
    assert diagnostics["in_window_rows"] == 1


def test_forcing_pairing_diagnostics_normalizes_inland_alias_policies():
    catalog = pd.DataFrame(
        {
            "event_reference_time": ["2021-06-01T00:00:00"],
            "rainfall_member_time": ["2020-06-15T00:00:00"],
            "rainfall_member_id": ["rain_jun"],
            "rainfall_pairing_window_days": [45],
            "soil_moisture_member_time": ["2020-06-14T00:00:00"],
            "soil_moisture_pairing_reference_time": ["2020-06-15T00:00:00"],
            "soil_moisture_pairing_lag_hours": [24],
        }
    )

    diagnostics = plotting.forcing_pairing_diagnostics(
        catalog,
        forcings=["rainfall", "soil_moisture"],
        policies={
            "rainfall": {"strategy": "inland_rainfall_pairing_priority", "window_days": 45},
            "soil_moisture": {
                "strategy": "inland_antecedent_moisture_pairing",
                "lead_time_hours": 24,
            },
        },
    )

    assert diagnostics["policy"].tolist() == [
        "seasonal_window_permutation",
        "antecedent_to_forcing",
    ]
    assert diagnostics["paired_rows"].tolist() == [1, 1]


def test_plot_configured_pairing_dispatches_antecedent_policy():
    catalog = pd.DataFrame(
        {
            "soil_moisture_member_time": ["2020-01-01T00:00:00"],
            "soil_moisture_pairing_reference_time": ["2020-01-02T00:00:00"],
            "soil_moisture_pairing_lag_hours": [24],
            "soil_moisture_pairing_policy": ["antecedent_to_forcing"],
        }
    )

    fig = plotting.plot_configured_pairing(
        catalog,
        "soil_moisture",
        policy={"strategy": "antecedent_to_forcing", "lead_time_hours": 24},
    )
    ax = fig.axes[0]

    assert "Antecedent pairing: soil_moisture" in ax.get_title()


def test_plot_aic_model_selection_marks_lowest_displayed_aic_as_pick():
    peaks = pd.Series(
        np.array([
            1.00, 1.01, 1.02, 1.04, 1.08, 1.15, 1.25, 1.40,
            1.70, 2.20, 3.00, 4.50,
        ])
    )
    marginal = SimpleNamespace(dist_name="exp")

    fig = plotting.plot_aic_model_selection(peaks, marginal)
    labels = [text.get_text() for text in fig.axes[0].get_legend().texts]
    aic_by_dist = {}
    picked = None
    for label in labels:
        match = re.search(r"^(EXP|GPD) fit \(AIC=([-0-9.]+)(.*)\)$", label)
        if match is None:
            continue
        dist, aic, suffix = match.groups()
        aic_by_dist[dist.lower()] = float(aic)
        if "AIC pick" in suffix:
            picked = dist.lower()

    assert picked is not None
    assert aic_by_dist[picked] == min(aic_by_dist.values())


def test_severity_band_distribution_reports_counts_and_weighted_mass():
    catalog = pd.DataFrame(
        {
            "severity_band": ["mild", "mild", "rare", "rare"],
            "sampling_weight": [1.2, 1.2, 0.25, 0.25],
            "probability_weight": [0.45, 0.35, 0.15, 0.05],
        }
    )

    distribution = plotting.severity_band_distribution(catalog)

    mild = distribution.set_index("severity_band").loc["mild"]
    rare = distribution.set_index("severity_band").loc["rare"]
    assert mild["event_count"] == 2
    assert mild["weighted_mass"] == 2.4
    assert rare["event_count"] == 2
    assert rare["weighted_mass"] == 0.5


def test_severity_band_distribution_prefers_probability_weight_when_available():
    catalog = pd.DataFrame(
        {
            "severity_band": ["mild", "mild", "rare", "rare"],
            "sampling_weight": [1.0, 1.0, 1.0, 1.0],
            "probability_weight": [0.7, 0.1, 0.05, 0.15],
        }
    )

    distribution = plotting.severity_band_distribution(catalog)

    mild = distribution.set_index("severity_band").loc["mild"]
    rare = distribution.set_index("severity_band").loc["rare"]
    assert mild["weighted_mass"] == 2.0
    assert round(mild["probability_mass"], 6) == 0.8
    assert rare["weighted_mass"] == 2.0
    assert round(rare["probability_mass"], 6) == 0.2


def test_severity_band_distribution_derives_missing_band_from_return_period():
    catalog = pd.DataFrame(
        {
            "sample_rp_years": [1.2, 5.0, 75.0],
            "sampling_weight": [1.0, 1.0, 1.0],
        }
    )

    distribution = plotting.severity_band_distribution(catalog)

    rows = distribution.set_index("severity_band")
    assert rows.loc["mild", "event_count"] == 1
    assert rows.loc["common", "event_count"] == 1
    assert rows.loc["rare", "event_count"] == 1


def test_plot_severity_bands_shows_unweighted_and_weighted_views():
    catalog = pd.DataFrame(
        {
            "severity_band": ["mild", "mild", "rare", "rare"],
            "sampling_weight": [1.2, 1.2, 0.25, 0.25],
            "probability_weight": [0.45, 0.35, 0.15, 0.05],
        }
    )

    fig = plotting.plot_severity_bands(catalog)

    assert len(fig.axes) == 2
    assert "Unweighted event count" in fig.axes[0].get_title()
    assert "Probability-weighted mass" in fig.axes[1].get_title()


def test_plot_original_vs_design_severity_shows_row_and_probability_views():
    original = pd.DataFrame(
        {
            "sample_rp_years": [1.1, 1.4, 3.0, 80.0],
            "sampling_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    design = pd.DataFrame(
        {
            "severity_band": ["mild", "rare", "rare", "extreme"],
            "sampling_weight": [10.0, 0.1, 0.1, 0.05],
            "probability_weight": [0.8, 0.05, 0.05, 0.1],
        }
    )

    fig = plotting.plot_original_vs_design_severity(original, design)

    assert len(fig.axes) == 2
    assert "Original vs design row distribution" in fig.axes[0].get_title()
    assert "Probability-weighted distribution" in fig.axes[1].get_title()


def test_nearest_benchmark_events_reports_standard_annual_chance_slices():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_001", "evt_002", "evt_003", "evt_004", "evt_005"],
            "sample_rp_years": [8.0, 12.0, 47.0, 101.0, 490.0],
            "severity_band": ["common", "significant", "significant", "extreme", "extreme"],
            "probability_weight": [0.1, 0.2, 0.3, 0.2, 0.2],
        }
    )

    benchmarks = plotting.nearest_benchmark_events(catalog, benchmarks=[10, 50, 100, 500])

    rows = benchmarks.set_index("benchmark_return_period_years")
    assert rows.loc[10, "event_id"] == "evt_002"
    assert rows.loc[50, "event_id"] == "evt_003"
    assert rows.loc[100, "event_id"] == "evt_004"
    assert rows.loc[500, "event_id"] == "evt_005"
    assert rows.loc[500, "annual_chance_label"] == "0.2% annual chance"


def test_plot_return_period_benchmark_coverage_marks_standard_slices():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_001", "evt_002", "evt_003", "evt_004"],
            "sample_rp_years": [10.0, 50.0, 100.0, 500.0],
            "severity_band": ["significant", "rare", "extreme", "extreme"],
            "probability_weight": [0.25, 0.25, 0.25, 0.25],
            "catalog_role": ["design", "design", "design", "design"],
        }
    )

    fig = plotting.plot_return_period_benchmark_coverage(catalog)

    assert "10/50/100/500-year" in fig.axes[0].get_title()
    assert "event-driver" in fig.axes[0].get_xlabel()
    assert "Selected Design Catalog" in fig.axes[0].get_legend().get_texts()[0].get_text()
    labels = [text.get_text() for text in fig.axes[0].texts]
    assert any("0.2%" in label for label in labels)
    table_text = [
        cell.get_text().get_text()
        for child in fig.axes[1].get_children()
        if isinstance(child, Table)
        for cell in child.get_celld().values()
    ]
    assert "500-yr" in table_text
    assert "0.2% AEP" in table_text
    assert "5000" not in table_text


def test_plot_return_period_benchmark_coverage_labels_inland_streamflow_axis():
    catalog = pd.DataFrame(
        {
            "event_id": ["usgs_02095500_20220103T000000", "usgs_02094500_20200320T000000"],
            "basis_site_no": ["02095500", "02094500"],
            "sample_rp_years": [10.1, 53.0],
            "peak_flow_cfs": [4300.0, 8200.0],
            "severity_band": ["significant", "rare"],
            "probability_weight": [0.5, 0.5],
        }
    )

    fig = plotting.plot_return_period_benchmark_coverage(catalog, benchmarks=[10, 50])

    title_and_labels = " ".join(
        [fig.axes[0].get_title(), fig.axes[0].get_xlabel(), fig.axes[1].get_title()]
    ).lower()
    assert "coastal" not in title_and_labels
    assert "streamgage-network" in fig.axes[0].get_title()
    assert "streamgage-network return period" in fig.axes[0].get_xlabel()


def test_plot_return_period_benchmark_coverage_preserves_coastal_axis_for_coastal_catalog():
    catalog = pd.DataFrame(
        {
            "event_id": ["coastal_001", "coastal_002"],
            "coastal_peak_m": [1.2, 2.4],
            "sample_rp_years": [10.0, 50.0],
            "severity_band": ["significant", "rare"],
            "probability_weight": [0.5, 0.5],
        }
    )

    fig = plotting.plot_return_period_benchmark_coverage(catalog, benchmarks=[10, 50])

    assert "coastal-driver" in fig.axes[0].get_title()
    assert "coastal driver return period" in fig.axes[0].get_xlabel()


def test_plot_catalog_set_severity_comparison_contrasts_design_and_stress_sets():
    probability_catalog = pd.DataFrame(
        {
            "severity_band": ["mild", "mild", "mild", "extreme"],
            "probability_weight": [0.3, 0.3, 0.3, 0.1],
            "sampling_weight": [1.0, 1.0, 1.0, 1.0],
            "catalog_role": ["design", "design", "design", "design"],
        }
    )
    stress_catalog = pd.DataFrame(
        {
            "severity_band": ["mild", "extreme", "extreme", "extreme"],
            "probability_weight": [0.3, 0.1, 0.1, 0.1],
            "sampling_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )

    fig = plotting.plot_catalog_set_severity_comparison(probability_catalog, stress_catalog)

    assert "Selected Design Catalog vs Resilience Stress/Training Set" in fig.axes[0].get_title()
    assert len(fig.axes[0].patches) >= 4


def test_plot_joint_tail_budget_labels_selected_design_catalog():
    catalog = pd.DataFrame(
        {
            "severity_band": ["mild", "common", "extreme", "extreme"],
            "probability_weight": [0.8, 0.15, 0.025, 0.025],
            "catalog_role": ["design", "design", "design", "design"],
            "candidate_pool_count": [100_000, 100_000, 100_000, 100_000],
            "pool_band_support": [80_000, 15_000, 5_000, 5_000],
            "pool_band_probability": [0.80, 0.15, 0.05, 0.05],
            "driver_u": [0.1, 0.5, 0.98, 0.99],
        }
    )
    stress = {
        "target_event_count": 4,
        "severity_band_fractions": {"mild": 0.25, "common": 0.25, "extreme": 0.5},
    }

    fig = plotting.plot_joint_tail_budget(catalog, stress)

    assert "100,000 -> 4" in fig.axes[0].get_title()
    assert "candidate pool (n=100,000)" in fig.axes[0].get_legend().get_texts()[0].get_text()
    assert "Selection rate" in fig.axes[1].get_title()
    assert "Fitted probability mass" in fig.axes[2].get_title()


def test_wave_analog_diagnostics_reports_same_analog_completeness():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_0001", "evt_0002"],
            "coastal_analog_id": ["tpl_001", "tpl_002"],
            "snapwave_member_id": ["tpl_001", "tpl_002"],
            "snapwave_member_file": ["waves.nc", "waves.nc"],
            "snapwave_pairing_policy": ["same_historical_analog", "same_historical_analog"],
        }
    )

    diagnostics = plotting.wave_analog_diagnostics(catalog).iloc[0]

    assert diagnostics["policy"] == "same_historical_analog"
    assert diagnostics["paired_rows"] == 2
    assert diagnostics["missing_rows"] == 0
    assert diagnostics["same_analog_rows"] == 2


def test_forcing_selection_frame_compares_catalog_selection_to_source_members():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_0001", "evt_0002", "evt_0003"],
            "rainfall_member_id": ["rain_001", "rain_002", "rain_001"],
            "sampling_weight": [1.0, 1.0, 1.0],
            "probability_weight": [0.6, 0.3, 0.1],
        }
    )
    members = pd.DataFrame(
        {
            "member_id": ["rain_001", "rain_002", "rain_003"],
            "mean_precip_in": [4.0, 7.0, 2.0],
        }
    )

    frame = plotting.forcing_selection_frame(catalog, members, "rainfall")

    rows = frame.set_index("member_id")
    assert rows.loc["rain_001", "selected_count"] == 2
    assert rows.loc["rain_001", "selected_probability_mass"] == 0.7
    assert rows.loc["rain_002", "selected_probability_mass"] == 0.3
    assert rows.loc["rain_003", "selected_count"] == 0


def test_forcing_selection_frame_normalizes_soil_moisture_source_members():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_0001"],
            "soil_moisture_member_id": ["soil_moisture_20200102T000000"],
            "sampling_weight": [1.0],
            "probability_weight": [1.0],
        }
    )
    members = pd.DataFrame(
        {
            "time": ["2020-01-02T00:00:00", "2020-01-02T00:00:00", "2020-01-03T00:00:00"],
            "point_id": ["a", "b", "a"],
            "SOIL_M": [0.2, 0.4, 0.6],
            "SOILSAT_TOP": [0.7, 0.9, 0.6],
        }
    )

    frame = plotting.forcing_selection_frame(catalog, members, "soil_moisture")

    row = frame.set_index("member_id").loc["soil_moisture_20200102T000000"]
    assert row["selected_count"] == 1
    assert round(row["member_value"], 6) == 0.8


def test_forcing_marginal_and_joint_plots_compare_selected_member_values():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_0001", "evt_0002", "evt_0003"],
            "sample_rp_years": [2.0, 25.0, 100.0],
            "severity_band": ["common", "significant", "extreme"],
            "coastal_peak_m": [1.0, 1.8, 2.4],
            "rainfall_member_id": ["rain_001", "rain_002", "rain_001"],
            "sampling_weight": [1.0, 1.0, 1.0],
            "probability_weight": [0.6, 0.3, 0.1],
        }
    )
    members = pd.DataFrame(
        {
            "member_id": ["rain_001", "rain_002", "rain_003"],
            "mean_precip_in": [4.0, 7.0, 2.0],
        }
    )

    marginal = plotting.plot_forcing_marginal_comparison(catalog, members, "rainfall")
    joint = plotting.plot_coastal_forcing_joint(catalog, members, "rainfall")

    assert "rainfall marginal comparison" in marginal.axes[0].get_title()
    assert "Coastal driver return period vs rainfall" in joint.axes[0].get_title()
    assert joint.axes[0].get_xscale() == "log"


def test_driver_forcing_joint_labels_inland_streamflow_axis():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_001", "evt_002"],
            "basis_site_no": ["02095000", "02095500"],
            "peak_flow_cfs": [1200.0, 2400.0],
            "sample_rp_years": [2.0, 100.0],
            "rainfall_member_id": ["rain_001", "rain_002"],
            "probability_weight": [0.6, 0.4],
        }
    )
    members = pd.DataFrame(
        {
            "member_id": ["rain_001", "rain_002"],
            "mean_precip_mm": [100.0, 180.0],
        }
    )

    fig = plotting.plot_coastal_forcing_joint(catalog, members, "rainfall")

    text = fig.axes[0].get_title() + " " + fig.axes[0].get_xlabel()
    assert "coastal" not in text.lower()
    assert "Streamgage-network return period vs rainfall" in fig.axes[0].get_title()


def test_plot_streamflow_pot_members_shows_reviewed_gage_events():
    members = pd.DataFrame(
        {
            "site_no": ["02095000", "02095000", "02095500"],
            "event_time": ["2020-01-01T00:00:00", "2020-02-01T00:00:00", "2020-01-15T00:00:00"],
            "peak_flow_cfs": [1200.0, 2200.0, 1800.0],
            "sampling_region": ["body", "tail", "body"],
        }
    )

    fig = plotting.plot_streamflow_pot_members(members)

    assert "USGS streamflow POT members" in fig.axes[0].get_title()
    assert fig.axes[0].get_ylabel() == "peak discharge [cfs]"
    assert len(fig.axes[0].collections) == 2


def test_plot_streamflow_return_period_distribution_uses_log_axes():
    members = pd.DataFrame(
        {
            "event_id": ["evt_001", "evt_002"],
            "site_no": ["02095000", "02095500"],
            "event_time": ["2020-01-01T00:00:00", "2021-01-01T00:00:00"],
            "peak_flow_cfs": [1200.0, 2400.0],
            "sample_rp_years": [2.0, 100.0],
            "sampling_region": ["body", "tail"],
        }
    )

    fig = plotting.plot_streamflow_return_period_distribution(members)

    assert "Return-period ranked streamflow members" in fig.axes[0].get_title()
    assert fig.axes[0].get_xscale() == "log"
    assert fig.axes[0].get_yscale() == "log"


def test_plot_streamflow_pot_extraction_marks_threshold_like_coastal_pot_plot():
    times = pd.date_range("2020-01-01", periods=120, freq="h")
    records = pd.DataFrame(
        {
            "site_no": ["02095000"] * len(times),
            "time": times,
            "discharge_cfs": [100.0] * 20 + [900.0] + [100.0] * 49 + [1400.0] + [100.0] * 49,
        }
    )
    members = pd.DataFrame(
        {
            "site_no": ["02095000", "02095000"],
            "event_time": [times[20].isoformat(), times[70].isoformat()],
            "peak_flow_cfs": [900.0, 1400.0],
            "site_threshold_cfs": [800.0, 800.0],
        }
    )

    fig = plotting.plot_streamflow_pot_extraction(records, members, threshold_quantile=0.9)

    assert len(fig.axes) == 2
    assert "Streamflow POT extraction" in fig.axes[0].get_title()
    assert any("threshold = 800 cfs" in text.get_text() for text in fig.axes[0].get_legend().texts)
    assert fig.axes[1].get_xlabel() == "peak discharge [cfs]"


def test_plot_rainfall_member_distribution_shows_depth_and_seasonality():
    members = pd.DataFrame(
        {
            "storm_start": ["2020-01-01T00:00:00", "2020-07-01T00:00:00"],
            "mean_precip_mm": [3.0, 5.0],
            "rank": [2, 1],
        }
    )

    fig = plotting.plot_rainfall_member_distribution(members)

    assert len(fig.axes) == 2
    assert "AORC SST rainfall member depths" in fig.axes[0].get_title()
    assert "AORC SST member seasonality" in fig.axes[1].get_title()


def test_plot_distinct_oscillatory_proxies_returns_selected_rows():
    import xarray as xr

    member_dataset = xr.Dataset(coords={"relative_hour": [-1, 0, 1]})
    summary = pd.DataFrame(
        {
            "event_id": ["evt_0001", "evt_0002"],
            "template_id": ["tpl_001", "tpl_002"],
            "sample_rp_years": [1.0, 10.0],
            "peak": [2.0, 3.0],
            "volume": [4.0, 5.0],
            "duration_above_50pct_peak": [2.0, 3.0],
            "asymmetry_ratio": [1.0, 1.1],
        }
    )
    template_frame = pd.DataFrame(
        {
            "template_id": ["tpl_001", "tpl_002"],
            "peak_time": pd.to_datetime(["2020-01-02T00:00:00", "2020-01-03T00:00:00"]),
            "baseline_m": [1.0, 1.0],
            "peak_m": [2.0, 2.5],
        }
    )
    index = pd.date_range("2020-01-01T23:00:00", periods=50, freq="h")
    waterlevel = pd.Series(np.sin(np.arange(50)), index=index)

    fig, selected = plotting.plot_distinct_oscillatory_proxies(
        member_dataset,
        summary,
        template_frame,
        waterlevel,
        candidate_n=2,
        pick_n=2,
    )

    assert len(fig.axes) == 2
    assert len(selected) == 2
    assert "oscillation_score" in selected.columns


def test_plot_msl_shift_scenario_comparison_uses_offsets_and_return_curve():
    import xarray as xr

    scenario_datasets = {
        "base": xr.Dataset(
            {
                "peak": ("event_id", [1.0, 2.0]),
                "surge_absolute": (("event_id", "relative_hour"), [[0.0, 1.0], [0.0, 2.0]]),
            },
            coords={"event_id": ["evt_0001", "evt_0002"], "relative_hour": [0, 1]},
            attrs={"slr_offset_m": 0.0},
        ),
        "future": xr.Dataset(
            {
                "peak": ("event_id", [1.0, 2.0]),
                "surge_absolute": (("event_id", "relative_hour"), [[0.5, 1.5], [0.5, 2.5]]),
            },
            coords={"event_id": ["evt_0001", "evt_0002"], "relative_hour": [0, 1]},
            attrs={"slr_offset_m": 0.5},
        ),
    }
    marginal_ci = pd.DataFrame({"rps": [1.0, 10.0], "h_point": [1.0, 2.0], "h_lo": [0.9, 1.8], "h_hi": [1.1, 2.2]})
    marginal_params = pd.DataFrame({"detrend_reference_epoch_year": [2000.0]})

    fig = plotting.plot_msl_shift_scenario_comparison(
        scenario_datasets,
        marginal_ci,
        marginal_params,
        scenario_colors={"base": "#1f77b4", "future": "#d62728"},
        example_event_index=1,
    )

    assert len(fig.axes) == 3
    assert "Synthetic peak distribution per scenario" in fig.axes[0].get_title()
    assert "rigid translation under SLR" in fig.axes[1].get_title()
    assert "Coastal-driver return-period curve per scenario" in fig.axes[2].get_title()
