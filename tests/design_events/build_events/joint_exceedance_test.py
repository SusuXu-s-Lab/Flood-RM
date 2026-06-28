import numpy as np
import pyvinecopulib as pv
import pytest

from design_events.build_events.probability.exceedance import (
    and_joint_survival,
    and_label_frame,
    and_return_period,
    and_survival_empirical,
    and_survival_from_cdf,
    label_and_joint_exceedance,
    select_most_likely_design_events,
)
from design_events.fit_history.return_curve import HistoricalPeakMarginal


class _IndependenceCopula:
    """Independence copula: C(u) = prod(u_i); used to check the AND math in closed form."""

    def __init__(self, d):
        self.d = d

    def cdf(self, u):
        return np.prod(np.atleast_2d(np.asarray(u, dtype=float)), axis=1)

    def simulate(self, n, seeds=None):
        rng = np.random.default_rng(None if not seeds else int(seeds[0]))
        return rng.random((int(n), self.d))


def _fit_gaussian_like_vine(rho=0.8, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    z[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
    u = pv.to_pseudo_obs(z)
    controls = pv.FitControlsVinecop(
        family_set=[pv.gaussian, pv.student, pv.clayton, pv.gumbel],
        selection_criterion="aic",
        seeds=[1],
    )
    return pv.Vinecop.from_data(u, controls=controls)


def test_and_survival_independence_matches_closed_form():
    cop = _IndependenceCopula(2)
    u = np.array([[0.5, 0.5], [0.9, 0.2], [0.99, 0.99]])
    expected = np.prod(1.0 - u, axis=1)  # P(both exceed) under independence

    exact = and_survival_from_cdf(u, cop.cdf)
    np.testing.assert_allclose(exact, expected, atol=1e-12)

    reference = cop.simulate(200_000, seeds=[7])
    mc = and_survival_empirical(u, reference)
    np.testing.assert_allclose(mc, expected, atol=5e-3)


def test_and_survival_reduces_to_marginal_survival_in_1d():
    # d=1 inclusion-exclusion gives S = 1 - u, i.e. the marginal survival probability.
    cop = _IndependenceCopula(1)
    u = np.array([[0.2], [0.8], [0.98]])
    np.testing.assert_allclose(and_survival_from_cdf(u, cop.cdf), 1.0 - u.ravel(), atol=1e-12)


def test_and_return_period_formula_and_validation():
    aep, period = and_return_period(np.array([0.1, 0.01]), event_rate=5.0)
    np.testing.assert_allclose(aep, [0.5, 0.05])
    np.testing.assert_allclose(period, [2.0, 20.0])
    with pytest.raises(ValueError):
        and_return_period(np.array([0.1]), event_rate=0.0)


def test_positive_dependence_raises_joint_exceedance_in_the_tail():
    # Compound amplification: for positively dependent drivers, the probability that
    # BOTH exceed a high level is larger than under independence.
    high = np.array([[0.9, 0.9]])
    indep = and_survival_from_cdf(high, _IndependenceCopula(2).cdf)[0]
    dependent = and_joint_survival(high, copula=_fit_gaussian_like_vine(rho=0.85), method="cdf")[0]
    assert dependent > indep


def test_and_label_frame_columns_bands_and_ordering():
    cop = _IndependenceCopula(2)
    # Increasingly extreme joint levels -> longer return periods, more severe bands.
    u = np.array([[0.3, 0.3], [0.8, 0.8], [0.97, 0.97], [0.995, 0.995]])
    frame = and_label_frame(u, event_rate=5.0, copula=cop, method="cdf")

    assert list(frame.columns) == [
        "and_joint_exceedance_prob",
        "and_joint_aep",
        "and_joint_return_period_years",
        "and_severity_band",
    ]
    rp = frame["and_joint_return_period_years"].to_numpy()
    assert np.all(np.diff(rp) > 0)
    assert np.all(np.isfinite(frame["and_joint_exceedance_prob"]))
    # rarest row should not be a mild/common band
    assert frame["and_severity_band"].iloc[-1] in {"significant", "rare", "extreme", "beyond_design"}


def test_label_uses_paired_event_rate_consistently():
    cop = _IndependenceCopula(2)
    u = np.array([[0.9, 0.9]])
    survival = and_joint_survival(u, copula=cop, method="cdf")[0]
    labels = label_and_joint_exceedance(u, event_rate=5.0, copula=cop, method="cdf")
    np.testing.assert_allclose(labels.return_period_years[0], 1.0 / (5.0 * survival))


def test_empirical_auto_default_matches_simulate_proportion_pattern():
    # The recommended path for a fitted vine: AND survival = fraction of a large
    # simulated pool exceeding the level in all dims. Auto-defaults to empirical.
    vine = _fit_gaussian_like_vine(rho=0.8, n=4000, seed=0)
    pool = np.asarray(vine.simulate(200_000, seeds=[1, 2, 3]), dtype=float)
    u0 = np.array([[0.98, 0.98]])

    hand_rolled = np.all(pool > u0, axis=1).mean()  # the user's snippet
    via_reference = and_joint_survival(u0, reference=pool)[0]
    assert via_reference == pytest.approx(hand_rolled, abs=1e-12)

    # auto with only a copula also picks the empirical estimator (close to the pool value)
    via_auto = and_joint_survival(u0, copula=vine, n_reference=200_000, seed=11)[0]
    assert via_auto == pytest.approx(hand_rolled, rel=0.15)


def _marginal(loc, scale):
    # Exponential-tail POT marginal (the project's Exp/GPD family) with ~5 events/yr.
    return HistoricalPeakMarginal(
        dist_name="exp", params=(0.0, loc, scale), extremes_rate=5.0, method="pot",
        threshold_quantile=0.98, peak_count=200,
    )


def test_marginal_cdf_ppf_roundtrip_and_return_period_consistency():
    m = _marginal(loc=1.5, scale=0.3)
    q = np.array([0.1, 0.5, 0.9, 0.99])
    np.testing.assert_allclose(m.cdf(m.ppf(q)), q, atol=1e-9)
    # return_period(x) == 1 / (rate * (1 - cdf(x)))
    x = m.ppf(q)
    np.testing.assert_allclose(m.return_period(x), 1.0 / (m.extremes_rate * (1.0 - m.cdf(x))), rtol=1e-9)


def test_select_most_likely_design_events_are_monotone_and_physical():
    cop = _fit_gaussian_like_vine(rho=0.8, n=4000, seed=0)
    marginals = [_marginal(loc=1.5, scale=0.3), _marginal(loc=20.0, scale=8.0)]
    design = select_most_likely_design_events(
        cop,
        marginals,
        event_rate=5.0,
        target_return_periods=[10.0, 25.0, 50.0],
        driver_names=["water_level_m", "rainfall_mm"],
        n_sample=40_000,
        seed=3,
    )

    assert len(design) == 3
    # one design point per target, each found on a populated isoline band
    assert (design["isoline_points"] > 0).all()
    # achieved joint RP tracks the target (loose: KDE/QMC sampling tolerance)
    achieved = design["and_joint_return_period_years"].to_numpy()
    target = design["target_return_period_years"].to_numpy()
    assert np.all(achieved > 0)
    assert np.all((achieved > target / 3) & (achieved < target * 3))
    # rarer targets => larger design driver magnitudes
    assert np.all(np.diff(design["water_level_m"].to_numpy()) > 0)
    assert np.all(np.diff(design["rainfall_mm"].to_numpy()) > 0)
    # design point is a valid (uniform, physical) pair
    assert ((design["water_level_m_u"] > 0) & (design["water_level_m_u"] < 1)).all()
    assert np.isfinite(design[["water_level_m", "rainfall_mm"]].to_numpy()).all()
