import pandas as pd

from design_events.build_events.selection import (
    apply_compound_stress_pairing,
    _soil_member_metrics,
    select_resilience_stress_training_set,
)


def test_resilience_stress_training_set_keeps_benchmarks_and_limits_mild_rows():
    catalog = pd.DataFrame(
        {
            "event_id": [f"evt_{index:04d}" for index in range(1, 13)],
            "sample_rp_years": [1.2, 1.4, 1.8, 2.5, 8.0, 9.7, 11.0, 49.0, 101.0, 420.0, 495.0, 4.0],
            "severity_band": [
                "mild",
                "mild",
                "mild",
                "common",
                "common",
                "common",
                "significant",
                "significant",
                "extreme",
                "extreme",
                "extreme",
                "common",
            ],
            "sampling_region": ["body", "body", "body", "body", "tail", "tail", "tail", "tail", "tail", "tail", "tail", "body"],
            "rainfall_member_id": [f"rain_{index:02d}" for index in range(12)],
            "soil_moisture_member_id": [f"soil_{index:02d}" for index in range(12)],
            "probability_weight": [1 / 12] * 12,
        }
    )
    rainfall = pd.DataFrame(
        {
            "member_id": [f"rain_{index:02d}" for index in range(12)],
            "mean_precip_mm": [1, 2, 3, 4, 5, 6, 7, 8, 30, 9, 10, 40],
        }
    )
    soil_moisture = pd.DataFrame(
        {
            "member_id": [f"soil_{index:02d}" for index in range(12)],
            "soil_moisture_mean": [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.45, 0.46, 0.19],
        }
    )

    selected = select_resilience_stress_training_set(
        catalog,
        rainfall_members=rainfall,
        soil_moisture_members=soil_moisture,
        config={
            "resilience_stress_training": {
                "target_event_count": 8,
                "max_mild_fraction": 0.25,
                "benchmark_return_period_years": [10, 50, 100, 500],
                "rainfall_heavy_fraction": 0.20,
                "wet_soil_fraction": 0.20,
            }
        },
    )

    assert len(selected) == 8
    assert set(selected["benchmark_return_period_years"].dropna().astype(str)) == {"10", "50", "100", "500"}
    assert (selected["severity_band"] == "mild").sum() <= 2
    assert "rainfall_heavy_sst_member" in ";".join(selected["selection_reason"])
    assert "wet_antecedent_soil_state" in ";".join(selected["selection_reason"])
    assert selected["event_set"].eq("resilience_stress_training").all()


def test_resilience_stress_training_reasons_are_driver_aware_for_streamflow_catalogs():
    catalog = pd.DataFrame(
        {
            "event_id": [f"evt_{index:04d}" for index in range(1, 7)],
            "basis_site_no": ["02095000"] * 6,
            "peak_flow_cfs": [1000, 1500, 2000, 3000, 5000, 8000],
            "sample_rp_years": [2, 9, 12, 52, 101, 498],
            "severity_band": ["common", "common", "significant", "rare", "extreme", "extreme"],
            "sampling_region": ["body", "body", "tail", "tail", "tail", "tail"],
            "rainfall_member_id": [f"rain_{index:02d}" for index in range(6)],
            "soil_moisture_member_id": [f"soil_{index:02d}" for index in range(6)],
        }
    )

    selected = select_resilience_stress_training_set(
        catalog,
        config={
            "resilience_stress_training": {
                "target_event_count": 5,
                "benchmark_return_period_years": [10, 50, 100, 500],
            }
        },
    )

    reasons = ";".join(selected["selection_reason"])
    assert "coastal_driver" not in reasons
    assert "streamflow_driver" in reasons
    assert "nearest_10pct_annual_chance_streamflow_driver" in reasons


def test_resilience_stress_training_set_uses_marshfield_like_severity_budget_by_default():
    rows = []
    bands = [
        ("mild", 1.2, "body"),
        ("common", 3.0, "body"),
        ("significant", 20.0, "tail"),
        ("rare", 75.0, "tail"),
        ("extreme", 250.0, "tail"),
    ]
    for band, base_rp, region in bands:
        for index in range(100):
            rows.append(
                {
                    "event_id": f"{band}_{index:03d}",
                    "sample_rp_years": base_rp + index * 0.01,
                    "severity_band": band,
                    "sampling_region": region,
                    "rainfall_member_id": f"rain_{band}_{index:03d}",
                    "soil_moisture_member_id": f"soil_{band}_{index:03d}",
                }
            )
    catalog = pd.DataFrame(rows)

    selected = select_resilience_stress_training_set(
        catalog,
        config={
            "resilience_stress_training": {
                "target_event_count": 100,
                "benchmark_return_period_years": [10, 50, 100, 500],
            }
        },
    )

    counts = selected["severity_band"].value_counts().to_dict()
    assert len(selected) == 100
    assert counts == {
        "significant": 28,
        "common": 28,
        "extreme": 27,
        "rare": 12,
        "mild": 5,
    }


def test_soil_member_metrics_prefers_soilsat_top_when_available():
    members = pd.DataFrame(
        {
            "time": ["2020-01-02T00:00:00", "2020-01-02T00:00:00"],
            "SOIL_M": [0.2, 0.4],
            "SOILSAT_TOP": [0.7, 0.9],
        }
    )

    metrics = _soil_member_metrics(members)

    assert metrics["member_id"].tolist() == ["soil_moisture_20200102T000000"]
    assert metrics["soil_moisture_mean"].tolist() == [0.8]


def test_compound_stress_pairing_overrides_rainfall_for_operational_cases():
    catalog = pd.DataFrame(
        {
            "event_id": [f"evt_{index:04d}" for index in range(1, 6)],
            "sample_rp_years": [500, 250, 100, 50, 10],
            "coastal_template_peak_time": [
                "2020-01-02T12:00:00",
                "2020-02-02T12:00:00",
                "2020-03-02T12:00:00",
                "2020-04-02T12:00:00",
                "2020-05-02T12:00:00",
            ],
            "rainfall_member_id": ["old"] * 5,
            "rainfall_member_time": ["2020-01-01T00:00:00"] * 5,
            "soil_moisture_member_id": ["old_soil"] * 5,
        }
    )
    rainfall = pd.DataFrame(
        {
            "member_id": ["real_jan", "heavy_feb", "heavy_mar", "heavy_apr", "heavy_may"],
            "source": ["aorc_sst"] * 5,
            "member_file": ["rain.csv"] * 5,
            "storm_start": [
                "2020-01-01T12:00:00",
                "1999-02-03T00:00:00",
                "1999-03-03T00:00:00",
                "1999-04-03T00:00:00",
                "1999-05-03T00:00:00",
            ],
            "duration_hours": [72] * 5,
            "mean_precip_mm": [100, 250, 240, 230, 220],
        }
    )
    soil = pd.DataFrame(
        {
            "time": ["1999-02-02T00:00:00", "1999-03-02T00:00:00", "1999-04-02T00:00:00"],
            "SOILSAT_TOP": [0.2, 0.9, 0.3],
        }
    )

    paired = apply_compound_stress_pairing(
        catalog,
        rainfall_members=rainfall,
        soil_moisture_members=soil,
        settings={
            "seed": 7,
            "seasonal_window_days": 45,
            "real_event_count": 1,
            "real_event_window_hours": 48,
            "role_fractions": {
                "high_rainfall_cooccurrence": 0.34,
                "rainfall_before_coastal": 0.33,
                "rainfall_after_coastal": 0.33,
                "wet_soil_high_rainfall": 0.01,
            },
        },
    )

    assert paired["rainfall_pairing_policy"].eq("compound_stress_operational").all()
    assert paired["compound_pairing_policy"].eq("operationally_severe_plausible_dependence").all()
    assert "historical_coastal_rainfall_pair" in set(paired["compound_pairing_role"])
    assert {"rainfall-coincident", "rainfall-before-coastal", "rainfall-after-coastal"}.issubset(
        set(paired["scenario_timing_edge_case"])
    )
    assert paired["rainfall_metric_mm"].notna().all()
    assert paired["rainfall_start_offset_hours"].notna().all()
