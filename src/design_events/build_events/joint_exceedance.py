"""AND joint-exceedance labeling and design-event selection.

This is the labeling + event-selection stage of the compound-flood design-event
pipeline. It consumes a fitted copula over Driver Probability Indices (the scalar
per-driver summaries) and the paired-event rate, and produces:

- AND joint-exceedance probability, AEP, and return period for each event, and a
  severity band, following the "AND" hazard scenario used for compound flooding
  (Moftakhari et al. 2019; Salvadori et al. 2016; Jane et al. 2020).
- "Most-likely" design events on an AND isoline at target return periods, where
  the design point is the maximum joint-density point along the isoline
  (Salvadori & De Michele 2013; Bender et al. 2016; Maduwantha et al. 2026).

The copula is duck-typed: any object exposing ``cdf(u)`` (n*d uniforms -> n) and
``simulate(n, seeds=[...])`` (-> n*d uniforms) works, including a fitted
``pyvinecopulib.Vinecop``. The physical fields that realize each labeled event
(SST rainfall, hydrographs) are attached downstream; this stage only touches the
scalar indices, so it never collapses spatio-temporal structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from design_events.build_events.event_distribution import assign_severity_bands

clip_eps = 1e-12


def and_survival_from_cdf(u, cdf):
    """Exact AND survival ``S(u) = P(U1>u1, ..., Ud>ud)`` via inclusion-exclusion.

    ``S(u) = sum_{A subset of {1..d}} (-1)^|A| C(u^A)`` where ``u^A`` keeps ``u_i``
    for ``i in A`` and sets the other coordinates to 1 (marginalizing them).
    ``cdf`` is a callable mapping an (m, d) uniform array to an (m,) copula CDF.
    """
    u = np.atleast_2d(np.asarray(u, dtype=float))
    n, d = u.shape
    survival = np.zeros(n, dtype=float)
    for mask in range(1 << d):
        corner = np.ones((n, d), dtype=float)
        bits = 0
        for i in range(d):
            if (mask >> i) & 1:
                corner[:, i] = np.clip(u[:, i], clip_eps, 1.0 - clip_eps)
                bits += 1
        sign = -1.0 if bits % 2 else 1.0
        survival += sign * np.asarray(cdf(np.asfortranarray(corner)), dtype=float).reshape(n)
    return np.clip(survival, 0.0, 1.0)


def and_survival_empirical(u, reference, *, block_size=256):
    """Monte-Carlo AND survival: fraction of reference draws exceeding ``u`` in all dims.

    Robust for vine copulas whose analytic CDF is itself a QMC estimate, and the
    proportion-based estimator used by Jane et al. (2020) / Maduwantha et al. (2026).
    """
    u = np.atleast_2d(np.asarray(u, dtype=float))
    reference = np.atleast_2d(np.asarray(reference, dtype=float))
    n = u.shape[0]
    out = np.empty(n, dtype=float)
    for start in range(0, n, block_size):
        block = u[start : start + block_size]
        exceed = (reference[None, :, :] > block[:, None, :]).all(axis=2)
        out[start : start + block_size] = exceed.mean(axis=1)
    return out


def and_joint_survival(
    u,
    *,
    copula=None,
    reference=None,
    cdf=None,
    method="auto",
    n_reference=100_000,
    seed=0,
):
    """AND survival probability for each event, via the copula CDF or Monte Carlo.

    For a fitted vine the default ("auto") is the Monte-Carlo proportion estimator
    (sample a large pool, count draws exceeding the level in all dims) — it is robust,
    scales to 3+ drivers without inclusion-exclusion, and matches the proportion-based
    AND estimate of Jane et al. (2020) / Maduwantha et al. (2026). The closed-form
    ``cdf`` path is reserved for analytic copulas and single-event diagnostics. Both
    are Monte Carlo for vines, so tail estimates carry sampling uncertainty; use a
    large pool (and ``qrng=True`` on simulate) and treat low-hit bands as uncertain.
    """
    if method == "auto":
        if reference is not None:
            method = "empirical"
        elif cdf is not None:
            method = "cdf"
        elif copula is not None:
            method = "empirical"
        else:
            raise ValueError("provide a `copula`, a `cdf` callable, or a `reference` sample")
    if method == "cdf":
        cdf_fn = cdf if cdf is not None else (copula.cdf if copula is not None else None)
        if cdf_fn is None:
            raise ValueError("method='cdf' needs a `cdf` callable or a `copula` with .cdf")
        return and_survival_from_cdf(u, cdf_fn)
    if method == "empirical":
        if reference is None:
            if copula is None:
                raise ValueError("method='empirical' needs a `reference` sample or a `copula` to draw one")
            reference = np.asarray(copula.simulate(int(n_reference), seeds=[int(seed)]), dtype=float)
        return and_survival_empirical(u, reference)
    raise ValueError(f"unknown method {method!r}; use 'cdf', 'empirical', or 'auto'")


def and_return_period(survival, event_rate):
    """Convert AND survival to (annual exceedance probability, return period in years).

    ``rate`` is the paired-event rate (events per year, e.g. the two-sided POT rate).
    ``T = 1 / (rate * S)`` matches HistoricalPeakMarginal.return_period in 1-D.
    """
    survival = np.asarray(survival, dtype=float)
    rate = float(event_rate)
    if not (np.isfinite(rate) and rate > 0):
        raise ValueError(f"event_rate must be finite and > 0, got {event_rate!r}")
    aep = np.clip(rate * survival, 0.0, 1.0)
    with np.errstate(divide="ignore"):
        period = np.where(survival > 0, 1.0 / (rate * survival), np.inf)
    return aep, period


def combined_and_frequency(physical, populations):
    """Combined AND exceedance frequency (events/yr) across storm-type populations.

    For each population ``p`` with rate ``lambda_p``, marginals, and copula, the AND survival
    ``S_p(x) = P(all drivers exceed x | a p event)`` is evaluated in physical space (map x to
    that population's marginal CDFs, then its copula), and the population frequencies add:
    ``freq(x) = sum_p lambda_p * S_p(x)``. This is the paper's "combine the AEPs of the two
    populations" (Maduwantha et al. 2026), generalized to any number of populations. Returns
    the per-point frequency in 1/yr.
    """
    physical = np.atleast_2d(np.asarray(physical, dtype=float))
    freq = np.zeros(len(physical), dtype=float)
    for pop in populations:
        u = np.column_stack(
            [np.asarray(m.cdf(physical[:, j]), dtype=float) for j, m in enumerate(pop.marginals)]
        )
        survival = and_survival_from_cdf(np.clip(u, clip_eps, 1.0 - clip_eps), pop.vine.cdf)
        freq += float(pop.rate) * survival
    return freq


def combined_return_period(physical, populations):
    """Combined AND (frequency in 1/yr, return period in years) across populations."""
    freq = combined_and_frequency(physical, populations)
    with np.errstate(divide="ignore"):
        period = np.where(freq > 0, 1.0 / freq, np.inf)
    return freq, period


@dataclass(frozen=True)
class AndExceedanceLabels:
    survival: np.ndarray
    joint_aep: np.ndarray
    return_period_years: np.ndarray
    severity_band: pd.Series


def label_and_joint_exceedance(
    u,
    event_rate,
    *,
    copula=None,
    reference=None,
    cdf=None,
    bands=None,
    method="auto",
    n_reference=100_000,
    seed=0,
):
    """Label events with AND joint-exceedance probability, AEP, RP, and severity band."""
    survival = and_joint_survival(
        u, copula=copula, reference=reference, cdf=cdf, method=method, n_reference=n_reference, seed=seed
    )
    aep, period = and_return_period(survival, event_rate)
    band = assign_severity_bands(period, bands=bands)
    return AndExceedanceLabels(survival=survival, joint_aep=aep, return_period_years=period, severity_band=band)


def and_label_frame(u, event_rate, **kwargs):
    """AND labels as a DataFrame with the catalog's ``and_*`` columns."""
    labels = label_and_joint_exceedance(u, event_rate, **kwargs)
    return pd.DataFrame(
        {
            "and_joint_exceedance_prob": labels.survival,
            "and_joint_aep": labels.joint_aep,
            "and_joint_return_period_years": labels.return_period_years,
            "and_severity_band": np.asarray(labels.severity_band),
        }
    )


def _physical_from_uniform(u, marginals):
    cols = [np.asarray(m.ppf(np.clip(u[:, j], clip_eps, 1.0 - clip_eps)), dtype=float) for j, m in enumerate(marginals)]
    return np.column_stack(cols)


def select_most_likely_design_events(
    copula,
    marginals,
    event_rate,
    target_return_periods,
    *,
    driver_names=None,
    n_sample=50_000,
    rp_rel_tol=0.10,
    min_isoline_points=25,
    max_rel_tol=4.0,
    seed=0,
):
    """Select the most-likely AND design event at each target return period.

    Draws a large copula sample (the candidate isolines), labels each with its AND
    return period, maps it to physical magnitudes through the marginals, and picks
    the point of maximum joint density (Gaussian KDE) within a relative band around
    each target RP. Returns one row per target with the design point in both uniform
    and physical space. Mirrors the most-likely-event design of Salvadori & De Michele
    (2013) / Bender et al. (2016) used by Jane et al. (2020) and Maduwantha et al. (2026).
    """
    from scipy.stats import gaussian_kde

    targets = np.atleast_1d(np.asarray(target_return_periods, dtype=float))
    d = len(marginals)
    names = list(driver_names) if driver_names is not None else [f"driver_{j}" for j in range(d)]
    if len(names) != d:
        raise ValueError(f"driver_names length {len(names)} != number of marginals {d}")

    u_sample = np.asarray(copula.simulate(int(n_sample), seeds=[int(seed)]), dtype=float)
    survival = and_survival_from_cdf(u_sample, copula.cdf)
    _, period = and_return_period(survival, event_rate)

    x_sample = _physical_from_uniform(u_sample, marginals)
    density = gaussian_kde(x_sample.T)(x_sample.T)

    log_period = np.log(np.where(np.isfinite(period) & (period > 0), period, np.nan))
    rows = []
    for target in targets:
        log_target = np.log(float(target))
        tol = float(rp_rel_tol)
        band = np.abs(log_period - log_target) <= np.log1p(tol)
        while band.sum() < min_isoline_points and tol < max_rel_tol:
            tol *= 1.5
            band = np.abs(log_period - log_target) <= np.log1p(tol)
        row = {"target_return_period_years": float(target), "isoline_points": int(band.sum()), "relative_tolerance": float(tol)}
        if not band.any():
            row.update({"and_joint_return_period_years": np.nan, "and_joint_exceedance_prob": np.nan, "kde_density": np.nan})
            for name in names:
                row[f"{name}_u"] = np.nan
                row[name] = np.nan
            rows.append(row)
            continue
        idx_band = np.flatnonzero(band)
        best = idx_band[int(np.argmax(density[idx_band]))]
        row.update(
            {
                "and_joint_return_period_years": float(period[best]),
                "and_joint_exceedance_prob": float(survival[best]),
                "kde_density": float(density[best]),
            }
        )
        for j, name in enumerate(names):
            row[f"{name}_u"] = float(u_sample[best, j])
            row[name] = float(x_sample[best, j])
        rows.append(row)
    return pd.DataFrame(rows)
