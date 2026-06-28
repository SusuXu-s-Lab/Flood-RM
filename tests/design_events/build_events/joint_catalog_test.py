import numpy as np
import pandas as pd
import pytest

from design_events.build_events.probability.dependence import DriverDependenceModel
from design_events.build_events.probability.design_catalog import (
    build_tail,
    build_joint_catalog,
    fit_index_marginal,
)


def _config(driver_vector):
    return {
        "project": {"name": "greensboro"},
        "events": {"target_event_count": 500},
        "event_catalog": {
            "dependence": {
                "method": "copula_joint",
                "driver_vector": driver_vector,
                "event_rate_per_year": 5.0,
                "copula_seed": 7,
                "pool_size": 120_000,
                "catalog_band_fractions": {"mild": 0.05, "common": 0.28, "significant": 0.28, "rare": 0.12, "extreme": 0.27},
            }
        },
        "resilience_stress_training": {
            "target_event_count": 500,
            "severity_band_fractions": {"mild": 0.05, "common": 0.28, "significant": 0.28, "rare": 0.12, "extreme": 0.27},
        },
        "wflow": {"events_root": "data/wflow/events"},
        "paths": {"sfincs_scenarios_root": "data/sfincs/scenarios"},
        "inland_coupling": {"infiltration": {"method": "curve_number"}},
    }


def _paired_observations(rho=0.7, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    z[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
    rainfall = 30.0 + 18.0 * np.log1p(np.exp(z[:, 0]))
    soil = 0.4 + 0.15 * (1 / (1 + np.exp(-z[:, 1])))
    return pd.DataFrame({"rainfall": rainfall, "soil_moisture": soil})


def _member_libraries(seed=0):
    rng = np.random.default_rng(seed)
    nr = 300
    rainfall = pd.DataFrame(
        {
            "member_id": [f"rainfall_72h_rank{i:04d}" for i in range(nr)],
            "member_file": [f"data/sources/aorc_sst/event_windows/storm_{i:04d}.nc" for i in range(nr)],
            "storm_start": pd.date_range("1990-01-01", periods=nr, freq="20D").astype(str),
            "mean_precip_mm": np.sort(rng.gamma(2.0, 25.0, nr)),
        }
    )
    ns = 250
    soil = pd.DataFrame(
        {
            "member_id": [f"soil_moisture_{i:04d}" for i in range(ns)],
            "member_file": [f"data/sources/nwm/soil_states/state_{i:04d}.nc" for i in range(ns)],
            "time": pd.date_range("1990-01-05", periods=ns, freq="25D").astype(str),
            "soil_moisture_mean": np.sort(rng.uniform(0.2, 0.55, ns)),
        }
    )
    return {"rainfall": rainfall, "soil_moisture": soil}


class LinearMarginal:
    def __init__(self, low, high):
        self.low = float(low)
        self.high = float(high)

    def cdf(self, values):
        values = np.asarray(values, dtype=float)
        return np.clip((values - self.low) / (self.high - self.low), 1e-9, 1.0 - 1e-9)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return self.low + q * (self.high - self.low)


class IndependentVine:
    def cdf(self, u):
        u = np.asarray(u, dtype=float)
        return np.prod(u, axis=1)


def test_build_joint_catalog_inland_vector_end_to_end():
    config = _config(["rainfall", "soil_moisture"])
    result = build_joint_catalog(
        config,
        {"location_name": "greensboro", "scenario": {"name": "base"}},
        paired_observations=_paired_observations(),
        member_libraries=_member_libraries(),
    )
    catalog = result.catalog

    assert len(catalog) == 500
    # standard catalog columns
    for column in ["event_id", "sample_rp_years", "severity_band", "sampling_region",
                   "sampling_weight", "probability_weight", "study_location", "scenario_name",
                   "forcing_pairing_policy", "wflow_event_dir", "sfincs_scenario_dir",
                   "infiltration_treatment", "event_origin", "catalog_role", "sampling_scheme",
                   "event_family"]:
        assert column in catalog.columns
    assert (catalog["forcing_pairing_policy"] == "copula_joint").all()
    assert set(catalog["event_family"]) == {"copula_joint_compound"}
    assert set(catalog["sampling_scheme"]) == {"band_stratified_importance"}
    assert set(catalog["catalog_role"]) == {"design"}

    # downstream-compatible realization columns for BOTH drivers
    for driver in ["rainfall", "soil_moisture"]:
        for suffix in [
            "_member_id", "_member_file", "_scale_factor", "_template_member_id",
            "_source", "_pairing_policy", "_pairing_seed",
        ]:
            assert f"{driver}{suffix}" in catalog.columns
        assert (catalog[f"{driver}_pairing_policy"] == "copula_joint_field_preserving_analog").all()
    assert catalog["rainfall_member_file"].str.endswith(".nc").all()

    # the stress-budget gate passed (rare+extreme filled)
    assert result.budget_report.loc[result.budget_report["stress_budget_count"] > 0, "meets_budget"].all()
    # probability mass mild-dominated despite tail enrichment
    assert catalog.groupby("severity_band")["probability_weight"].sum().get("mild", 0.0) > 0.5
    # fitted vine has the configured dimension
    assert result.model.dim == 2


def test_build_tail_appends_deduped_observed_tail():
    paired = pd.DataFrame(
        {
            "event_time": pd.to_datetime(["2000-01-01", "2000-01-03", "2000-03-01"]),
            "conditioned_on": ["rainfall", "coastal_water_level", "rainfall"],
            "rainfall": [95.0, 94.0, 40.0],
            "soil_moisture": [0.90, 0.89, 0.30],
            "storm_type": ["nor_easter", "nor_easter", "other"],
        }
    )
    model = DriverDependenceModel(
        vine=IndependentVine(),
        marginals=[LinearMarginal(0.0, 100.0), LinearMarginal(0.0, 1.0)],
        driver_names=["rainfall", "soil_moisture"],
        event_rate=1.0,
    )
    config = _config(["rainfall", "soil_moisture"])
    config["event_catalog"]["dependence"]["historical_tail_min_return_period_years"] = 10.0
    config["event_catalog"]["dependence"]["historical_tail_dedupe_hours"] = 120.0
    members = {
        "rainfall": pd.DataFrame(
            {
                "member_id": ["rain_high", "rain_near", "rain_body"],
                "member_file": ["rain_high.nc", "rain_near.nc", "rain_body.nc"],
                "storm_start": pd.to_datetime(["2000-01-01", "2000-01-03", "2000-03-01"]),
                "mean_precip_mm": [95.0, 94.0, 40.0],
                "source": "aorc_sst",
            }
        ),
        "soil_moisture": pd.DataFrame(
            {
                "member_id": ["soil_high", "soil_near", "soil_body"],
                "member_file": ["soil_high.nc", "soil_near.nc", "soil_body.nc"],
                "time": pd.to_datetime(["2000-01-01", "2000-01-03", "2000-03-01"]),
                "soil_moisture_mean": [0.90, 0.89, 0.30],
                "source": "nwm",
            }
        ),
    }

    catalog = build_tail(
        paired,
        model,
        config,
        {"location_name": "greensboro", "scenario": {"name": "base"}},
        member_libraries=members,
    )

    assert len(catalog) == 1
    row = catalog.iloc[0]
    assert row["event_id"] == "historical_20000101T000000"
    assert row["event_origin"] == "historical_tail"
    assert row["catalog_role"] == "historical_reference"
    assert row["sampling_scheme"] == "observed_historical_tail"
    assert row["storm_type"] == "nor_easter"
    assert row["rainfall_member_id"] == "rain_high"
    assert row["soil_moisture_member_id"] == "soil_high"
    assert row["rainfall_scale_factor"] == 1.0
    assert row["soil_moisture_scale_factor"] == 1.0


def test_build_tail_uses_observed_coastal_peak_time():
    paired = pd.DataFrame(
        {
            "event_time": [pd.Timestamp("2000-01-01 00:00:00")],
            "coastal_water_level_time": [pd.Timestamp("2000-01-02 06:00:00")],
            "rainfall_time": [pd.Timestamp("2000-01-01 00:00:00")],
            "conditioned_on": ["rainfall"],
            "coastal_water_level": [9.5],
            "rainfall": [95.0],
        }
    )
    model = DriverDependenceModel(
        vine=IndependentVine(),
        marginals=[LinearMarginal(0.0, 10.0), LinearMarginal(0.0, 100.0)],
        driver_names=["coastal_water_level", "rainfall"],
        event_rate=1.0,
    )
    config = _config(["coastal_water_level", "rainfall"])
    config["event_catalog"]["dependence"]["historical_tail_min_return_period_years"] = 10.0
    members = {
        "coastal_water_level": pd.DataFrame(
            {
                "member_id": ["coastal_far"],
                "member_file": ["cora.csv"],
                "time": [pd.Timestamp("1999-01-01 00:00:00")],
                "coastal_peak_m": [9.5],
            }
        ),
        "rainfall": pd.DataFrame(
            {
                "member_id": ["rain_exact"],
                "member_file": ["rain.csv"],
                "storm_start": [pd.Timestamp("2000-01-01 00:00:00")],
                "mean_precip_mm": [95.0],
            }
        ),
    }

    catalog = build_tail(
        paired,
        model,
        config,
        {"location_name": "greensboro", "scenario": {"name": "base"}},
        member_libraries=members,
    )

    row = catalog.iloc[0]
    assert row["coastal_water_level_member_id"] == "coastal_water_level_20000102T060000"
    assert row["coastal_water_level_member_time"] == "2000-01-02T06:00:00"
    assert row["coastal_water_level_template_value"] == 9.5
    assert row["rainfall_member_id"] == "rain_exact"


def test_driver_vector_is_required():
    config = _config([])
    with pytest.raises(ValueError, match="driver_vector"):
        build_joint_catalog(
            config, {"location_name": "greensboro"},
            paired_observations=_paired_observations(), member_libraries=_member_libraries(),
        )


def test_empirical_marginal_keeps_bounded_driver_within_observed_range():
    # Regression for the soil-isoline bug: a bounded antecedent driver given an empirical
    # marginal must never be sampled beyond its observed range, whereas an unbounded POT
    # tail extrapolates past it (the original defect).
    paired = _paired_observations()
    soil_max = float(paired["soil_moisture"].max())

    config = _config(["rainfall", "soil_moisture"])
    config["event_catalog"]["dependence"]["marginals"] = {"soil_moisture": {"kind": "empirical"}}
    empirical = build_joint_catalog(
        config, {"location_name": "greensboro"},
        paired_observations=paired, member_libraries=_member_libraries(),
    )
    assert empirical.catalog["soil_moisture"].max() <= soil_max + 1e-9  # bounded, no extrapolation

    config["event_catalog"]["dependence"]["marginals"] = {"soil_moisture": {"kind": "pot"}}
    pot = build_joint_catalog(
        config, {"location_name": "greensboro"},
        paired_observations=paired, member_libraries=_member_libraries(),
    )
    assert pot.catalog["soil_moisture"].max() > soil_max  # unbounded POT tail extrapolates (the bug)


def test_fit_index_marginal_kind_dispatch():
    from design_events.fit_history.return_curve import EmpiricalMarginal, HistoricalPeakMarginal
    rng = np.random.default_rng(0)
    bounded = rng.uniform(0.26, 0.41, 400)
    emp = fit_index_marginal(bounded, event_rate=5.0, kind="empirical")
    assert isinstance(emp, EmpiricalMarginal)
    assert emp.ppf(0.99999) <= bounded.max() + 1e-9  # saturates at observed max
    assert isinstance(fit_index_marginal(bounded, event_rate=5.0, kind="pot"), HistoricalPeakMarginal)
    with pytest.raises(ValueError):
        fit_index_marginal(bounded, event_rate=5.0, kind="beta")


def test_fit_index_marginal_roundtrips_through_quantiles():
    rng = np.random.default_rng(0)
    values = rng.gamma(2.0, 20.0, 500)
    marginal = fit_index_marginal(values, event_rate=5.0)
    q = np.array([0.5, 0.9, 0.99])
    np.testing.assert_allclose(marginal.cdf(marginal.ppf(q)), q, atol=1e-6)
    assert marginal.dist_name in {"exp", "gpd"}
