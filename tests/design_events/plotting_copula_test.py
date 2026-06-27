import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import design_events.plotting as P
from design_events.build_events.probability.design_catalog import build_joint_catalog
from design_events.fit_history.paired_observations import build_paired_observations


def _driver_frame(rho=0.75, years=6, seed=0):
    idx = pd.date_range("2000-01-01", periods=years * 365 * 24, freq="1h")
    rng = np.random.default_rng(seed)
    rainfall = pd.Series(rng.gamma(1.0, 2.0, len(idx)), index=idx)
    soil = pd.Series(rng.uniform(0.2, 0.35, len(idx)), index=idx)
    pos = rng.choice(len(idx), size=150, replace=False)
    z = rng.standard_normal((150, 2))
    z[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
    rainfall.iloc[pos] += 40 + 18 * np.maximum(z[:, 0], -1.5)
    soil.iloc[pos] += 0.15 * (1 / (1 + np.exp(-z[:, 1])))
    return pd.DataFrame({"rainfall": rainfall, "soil_moisture": soil})


def _member_libraries(seed=0):
    rng = np.random.default_rng(seed)
    rainfall = pd.DataFrame({
        "member_id": [f"rainfall_{i:04d}" for i in range(200)],
        "member_file": [f"data/sources/aorc_sst/event_windows/storm_{i:04d}.nc" for i in range(200)],
        "storm_start": pd.date_range("1990-01-01", periods=200, freq="20D").astype(str),
        "mean_precip_mm": np.sort(rng.gamma(2.0, 25.0, 200)),
    })
    soil = pd.DataFrame({
        "member_id": [f"soil_{i:04d}" for i in range(150)],
        "member_file": [f"data/sources/nwm/state_{i:04d}.nc" for i in range(150)],
        "time": pd.date_range("1990-01-05", periods=150, freq="25D").astype(str),
        "soil_moisture_mean": np.sort(rng.uniform(0.2, 0.5, 150)),
    })
    return {"rainfall": rainfall, "soil_moisture": soil}


CONFIG = {
    "project": {"name": "greensboro"},
    "events": {"target_event_count": 500},
    "event_catalog": {"dependence": {
        "driver_vector": ["rainfall", "soil_moisture"],
        "event_rate_per_year": 5.0,
        "copula_seed": 3,
        "pool_size": 100_000,
        "catalog_band_fractions": {"mild": 0.05, "common": 0.28, "significant": 0.28, "rare": 0.12, "extreme": 0.27},
    }},
    "resilience_stress_training": {"target_event_count": 500, "severity_band_fractions": {"mild": 0.05, "common": 0.28, "significant": 0.28, "rare": 0.12, "extreme": 0.27}},
}


@pytest.fixture(scope="module")
def artifacts():
    paired = build_paired_observations(_driver_frame(), threshold_quantiles=0.9)
    result = build_joint_catalog(
        CONFIG, {"location_name": "greensboro"}, paired_observations=paired, member_libraries=_member_libraries()
    )
    return paired, result


def test_cooccurrence_and_copula_figures(artifacts):
    paired, result = artifacts
    assert P.plot_driver_cooccurrence(paired, "rainfall", "soil_moisture") is not None
    assert P.plot_copula_fit_diagnostics(result.model, paired, n=2000) is not None
    plt.close("all")


def test_and_isolines_and_budget_and_realization(artifacts):
    paired, result = artifacts
    assert P.plot_and_joint_isolines(result.model, return_periods=(10, 50, 100), n_sample=2000, grid=40) is not None
    assert P.plot_joint_tail_budget(result.catalog, CONFIG["resilience_stress_training"]) is not None
    assert P.plot_realization_scaling(result.catalog, "rainfall") is not None
    plt.close("all")


def test_severity_plot_accepts_probability_only_historical_catalog():
    catalog = pd.DataFrame({
        "rainfall_mm": [42.0, 65.0, 110.0],
        "sample_rp_years": [0.5, 5.0, 50.0],
        "probability_weight": [1 / 3, 1 / 3, 1 / 3],
    })

    distribution = P.severity_band_distribution(catalog)

    assert distribution["event_count"].sum() == 3
    assert distribution["weighted_mass"].sum() == pytest.approx(3.0)
    assert distribution["probability_mass"].sum() == pytest.approx(1.0)
    assert distribution.attrs["has_probability_weight"] is True
    assert P.plot_severity_bands(catalog) is not None
    plt.close("all")


def test_compound_lag_diagnostic_labels_coastal_axis_as_ntr():
    frame = pd.DataFrame(
        {
            "coastal_water_level": [0.8],
            "coastal_absolute_peak_m": [2.1],
            "rainfall_metric_mm": [110.0],
            "rainfall_peak_offset_hours": [-24.0],
        }
    )

    panels = P._compound_lag_driver_panels(frame)

    assert panels[0] == ("coastal_water_level", "Coastal peak NTR / residual (m)")
    assert "MSL" not in panels[0][1]
