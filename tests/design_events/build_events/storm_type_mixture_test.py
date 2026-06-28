import numpy as np
import pandas as pd

from design_events.build_events.probability.dependence import (
    MixtureDependenceModel,
    StormTypePopulation,
    fit_storm_type_mixture,
    sample_mixture_catalog,
)
from design_events.build_events.probability.exceedance import combined_and_frequency
from design_events.build_events.probability.design_catalog import fit_index_marginal


class _UniformMarginal:
    def cdf(self, x):
        return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)

    def ppf(self, u):
        return np.clip(np.asarray(u, dtype=float), 0.0, 1.0)


class _IndepCopula:
    # independence copula: AND survival is exactly (1-u1)(1-u2)
    def cdf(self, u):
        u = np.atleast_2d(np.asarray(u, dtype=float))
        return u[:, 0] * u[:, 1]

    def simulate(self, n, **kwargs):
        return np.random.default_rng(0).random((int(n), 2))


def test_combined_frequency_is_rate_weighted_sum_of_population_survivals():
    pops = [
        StormTypePopulation("a", _IndepCopula(), [_UniformMarginal(), _UniformMarginal()], rate=2.0, n_events=50, low_confidence=False),
        StormTypePopulation("b", _IndepCopula(), [_UniformMarginal(), _UniformMarginal()], rate=3.0, n_events=80, low_confidence=False),
    ]
    # at x=(0.5,0.5) each population's AND survival is (1-.5)(1-.5)=0.25, so freq = (2+3)*0.25
    freq = combined_and_frequency([[0.5, 0.5]], pops)
    assert np.isclose(freq[0], 5.0 * 0.25)


def _synthetic_paired(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for storm_type, rho, n_st in [("nor_easter", 0.6, 140), ("tc", 0.2, 60)]:
        z = rng.standard_normal((n_st, 2))
        z[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
        coastal = 0.5 + 0.3 * np.maximum(z[:, 0], -1.5)
        rain = 40.0 + 20.0 * np.maximum(z[:, 1], -1.5)
        for c, r in zip(coastal, rain):
            rows.append({"storm_type": storm_type, "coastal_water_level": c, "rainfall": r})
    return pd.DataFrame(rows)


def test_mixture_rate_accounting_and_sampling():
    paired = _synthetic_paired()
    model, report = fit_storm_type_mixture(
        paired,
        ["coastal_water_level", "rainfall"],
        marginal_kinds={"coastal_water_level": "pot", "rainfall": "pot"},
        base_rate=5.0,
        fit_marginal=lambda values, rate, kind: fit_index_marginal(values, event_rate=rate, kind=kind),
        min_population_events=20,
        seed=1,
    )
    assert isinstance(model, MixtureDependenceModel)
    # population frequencies sum back to the base distinct-storm rate
    assert np.isclose(model.total_rate, 5.0)
    assert set(report["storm_type"]) == {"nor_easter", "tc"}

    catalog = sample_mixture_catalog(model, 100, target_band_fractions=None, pool_size=5000, seed=1)
    assert len(catalog) == 100
    assert "storm_type" in catalog.columns
    assert set(catalog["storm_type"]).issubset({"nor_easter", "tc"})
    assert np.isfinite(catalog["sample_rp_years"]).all()
