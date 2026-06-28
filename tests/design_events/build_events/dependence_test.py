import numpy as np
import pytest

from design_events.build_events.probability.dependence import (
    check_stress_budget,
    fit_driver_dependence,
    sample_tail_enriched_catalog,
    tail_enriched_catalog_band_fractions,
)
from design_events.fit_history.return_curve import HistoricalPeakMarginal


def _marginal(loc, scale):
    return HistoricalPeakMarginal(
        dist_name="exp", params=(0.0, loc, scale), extremes_rate=5.0, method="pot",
        threshold_quantile=0.98, peak_count=200,
    )


def _paired_observations(rho=0.7, n=4000, seed=0):
    # Concurrent (water level, rainfall) co-occurrence sample with positive dependence.
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    z[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
    wl = 1.5 + 0.3 * np.log1p(np.exp(z[:, 0]))
    rain = 20.0 + 8.0 * np.log1p(np.exp(z[:, 1]))
    return np.column_stack([wl, rain])


def _fit_model(rho=0.7, seed=0):
    x = _paired_observations(rho=rho, seed=seed)
    return fit_driver_dependence(
        x,
        marginals=[_marginal(1.5, 0.3), _marginal(20.0, 8.0)],
        driver_names=["water_level_m", "rainfall_mm"],
        event_rate=5.0,
        family_set=None,
        seed=1,
    )


STRESS = {
    "target_event_count": 500,
    "severity_band_fractions": {"mild": 0.05, "common": 0.28, "significant": 0.28, "rare": 0.12, "extreme": 0.27},
}


def test_fit_driver_dependence_validates_shapes():
    x = _paired_observations()
    with pytest.raises(ValueError):
        fit_driver_dependence(x, marginals=[_marginal(1.5, 0.3)], driver_names=["wl"], event_rate=5.0)
    model = _fit_model()
    assert model.dim == 2
    assert model.vine.dim == 2


def test_default_catalog_samples_the_fitted_distribution_by_count():
    model = _fit_model()
    catalog = sample_tail_enriched_catalog(model, n_catalog=2500, pool_size=120_000, seed=2)

    assert len(catalog) == 2500
    assert set(catalog["sampling_scheme"]) == {"probability_proportional"}
    assert set(catalog["catalog_role"]) == {"probability"}
    assert set(catalog["event_origin"]).issubset({"synthetic_body", "synthetic_tail"})
    assert catalog["probability_weight"].sum() == pytest.approx(1.0, abs=1e-9)
    counts = catalog["severity_band"].value_counts(normalize=True)
    assert counts.get("mild", 0.0) > counts.get("extreme", 0.0)


def test_configured_tail_enriched_catalog_fills_the_500_stress_budget():
    model = _fit_model()
    catalog = sample_tail_enriched_catalog(
        model,
        n_catalog=2500,
        pool_size=120_000,
        target_band_fractions=tail_enriched_catalog_band_fractions,
        seed=2,
    )

    assert len(catalog) == 2500
    assert set(catalog["sampling_scheme"]) == {"band_stratified_importance"}
    assert set(catalog["catalog_role"]) == {"design"}
    report = check_stress_budget(catalog, STRESS)  # raises if any band short
    assert report.loc[report["stress_budget_count"] > 0, "meets_budget"].all()
    rare_extreme = report.loc[report["severity_band"].isin(["rare", "extreme"]), "catalog_count"].sum()
    assert rare_extreme >= 195


def test_proportional_sampling_fails_the_budget_gate():
    # A small probability-proportional catalog should not masquerade as a full stress set.
    model = _fit_model()
    catalog = sample_tail_enriched_catalog(model, n_catalog=2500, pool_size=120_000, seed=3)
    with pytest.raises(ValueError, match="cannot fill"):
        check_stress_budget(catalog, STRESS)
    report = check_stress_budget(catalog, STRESS, raise_on_shortfall=False)
    assert not report.loc[report["severity_band"] == "extreme", "meets_budget"].iloc[0]


def test_probability_weight_reconstructs_true_mass_despite_enrichment():
    model = _fit_model()
    catalog = sample_tail_enriched_catalog(
        model,
        n_catalog=2500,
        pool_size=120_000,
        target_band_fractions=tail_enriched_catalog_band_fractions,
        seed=4,
    )

    # probability weights normalise to 1
    assert catalog["probability_weight"].sum() == pytest.approx(1.0, abs=1e-9)
    # tail-enriched by count, but mild dominates probability MASS (true distribution)
    mass = catalog.groupby("severity_band")["probability_weight"].sum()
    assert mass.get("mild", 0.0) > 0.5
    assert mass.get("mild", 0.0) > mass.get("extreme", 0.0)
    # importance weight is constant within a band, and mild > extreme
    sw_by_band = catalog.groupby("severity_band")["sampling_weight"].nunique()
    assert (sw_by_band <= 1).all()
    mean_sw = catalog.groupby("severity_band")["sampling_weight"].mean()
    assert mean_sw.get("mild", 0.0) > mean_sw.get("extreme", np.inf)


def test_catalog_columns_and_physical_monotonicity():
    model = _fit_model()
    catalog = sample_tail_enriched_catalog(model, n_catalog=2000, pool_size=100_000, seed=5)

    for column in ["event_id", "sample_rp_years", "severity_band", "sampling_region",
                   "sampling_weight", "probability_weight", "water_level_m", "rainfall_mm",
                   "water_level_m_u", "rainfall_mm_u", "event_origin", "catalog_role",
                   "sampling_scheme"]:
        assert column in catalog.columns
    assert np.isfinite(catalog[["water_level_m", "rainfall_mm", "sample_rp_years"]].to_numpy()).all()
    # rarer joint events carry larger driver magnitudes on average
    mild = catalog[catalog["severity_band"] == "mild"]
    extreme = catalog[catalog["severity_band"] == "extreme"]
    assert extreme["water_level_m"].mean() > mild["water_level_m"].mean()
    assert extreme["rainfall_mm"].mean() > mild["rainfall_mm"].mean()
