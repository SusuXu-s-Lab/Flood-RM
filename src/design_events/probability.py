"""
The probability functions for the Design Event Catalog (Layer 1 of the two-layer framework).

    X_j               driver j physical value (a Driver Probability Index)
    F_j(x)            marginal cdf of driver j
    U_j = F_j(X_j)    probability-scale index
    C(u)              copula cdf,  P(U_1<=u_1, ..., U_d<=u_d)
    S_and(u)          AND survival, P(U_1>u_1, ..., U_d>u_d)
    nu(x)  = lambda * S_and(F(x))         annual exceedance frequency (1/yr)
    T_and(x) = 1 / nu(x)                  AND Joint Return Period (years)
    nu(x)  = sum_g lambda_g S_and,g(x)    storm-type mixture frequency
    w_b = p_b / q_b                       Sampling Weight (between-band importance)
    pi_i = p_b / n_b                      Probability Weight (row mass, sums to 1)

``p_b`` is a band's true probability mass in the fitted pool; ``q_b = n_b/n`` is its
design-catalog sampling fraction. 

The Sampling Weight is the importance weight ``w_b = p_b/q_b`` (``<1`` for an over-sampled tail band);
production's coastal path uses this convention, the inland path used its reciprocal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

clip_eps = 1e-12

# --------------------------------------------------------------------------- #
# Severity bands                                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Band:
    """A return-period severity band ``[lo, hi)`` in years (``hi=None`` is open)."""

    name: str
    lo: float
    hi: float | None


def default_bands() -> list[Band]:
    """The default severity bands (must match ``selection.default_severity_bands``)."""
    return [
        Band("mild", 0.0, 2.0),
        Band("common", 2.0, 10.0),
        Band("significant", 10.0, 50.0),
        Band("rare", 50.0, 100.0),
        Band("extreme", 100.0, 500.0),
        Band("beyond_design", 500.0, None),
    ]

def assign_band(return_periods, bands: Sequence[Band] | None = None) -> pd.Series:
    """Label each return period with its severity band (``"unclassified"`` for NaN/neg).

    Faithful to production ``assign_severity_bands``: later bands overwrite earlier
    ones, the open band has no upper edge, and non-finite/negative RPs are unclassified.
    """
    bands = list(bands) if bands is not None else default_bands()
    rp = pd.to_numeric(pd.Series(return_periods), errors="coerce")
    out = pd.Series(["unclassified"] * len(rp), index=rp.index, dtype="object")
    for band in bands:
        mask = rp >= float(band.lo)
        if band.hi is not None:
            mask &= rp < float(band.hi)
        out.loc[mask] = band.name
    out.loc[rp.isna() | (rp < 0)] = "unclassified"
    return out


def default_severity_bands():
    """Dict form of ``default_bands()`` — the config schema for ``sampling.severity_bands``."""
    return [{"severity_band": b.name, "rp_min_years": b.lo, "rp_max_years": b.hi} for b in default_bands()]


def assign_severity_bands(return_periods, bands=None):
    """``assign_band`` accepting the dict-band config schema (compat for the catalog cluster)."""
    if bands is None:
        resolved = default_bands()
    else:
        resolved = [
            Band(
                str(b["severity_band"]),
                float(b.get("rp_min_years", 0.0)),
                None if b.get("rp_max_years") is None else float(b["rp_max_years"]),
            )
            for b in bands
        ]
    return assign_band(return_periods, resolved)


# --------------------------------------------------------------------------- #
# AND joint-exceedance math                                                  #
# --------------------------------------------------------------------------- #
def and_survival_from_cdf(u, cdf) -> np.ndarray:
    """Exact AND survival ``S_and(u) = P(U_1>u_1, ..., U_d>u_d)`` via inclusion-exclusion.

    ``S_and(u) = sum_{A subset {1..d}} (-1)^|A| C(u^A)``, where ``u^A`` keeps ``u_i`` for
    ``i in A`` and sets the others to 1. ``cdf`` maps an ``(m, d)`` uniform array to the
    ``(m,)`` copula CDF (e.g. a fitted ``pyvinecopulib.Vinecop.cdf``).
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

def and_survival_empirical(u, reference, *, block_size=256) -> np.ndarray:
    """Monte-Carlo AND survival: fraction of ``reference`` draws exceeding ``u`` in all dims.

    Robust for vine copulas whose analytic CDF is itself a QMC estimate.
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


def seeded_cdf(copula, seed=0):
    """A reproducible copula CDF callable. A vine ``.cdf`` is QMC Monte-Carlo integration
    (nondeterministic unless seeded); pass ``seeds=[seed]``. Analytic copulas
    whose ``.cdf`` takes no ``seeds`` kwarg pass through unchanged.
    """
    def _cdf(corner):
        try:
            return copula.cdf(corner, seeds=[int(seed)])
        except TypeError:
            return copula.cdf(corner)
    return _cdf


def and_survival(u, copula, *, method="cdf", n_reference=500_000, seed=0) -> np.ndarray:
    """AND survival for each event. ``method`` resolves the auto/cdf ambiguity:
    ``"cdf"`` uses exact inclusion-exclusion over a
    seeded (reproducible) copula CDF; ``"mc"`` uses the Monte-Carlo proportion estimator.
    """
    if method == "cdf":
        return and_survival_from_cdf(u, seeded_cdf(copula, seed))
    if method == "mc":
        reference = np.asarray(copula.simulate(int(n_reference), seeds=[int(seed)]), dtype=float)
        return and_survival_empirical(u, reference)
    raise ValueError(f"unknown survival method {method!r}; use 'cdf' or 'mc'")


def and_return_period(survival, event_rate):
    """AND survival -> (annual exceedance probability, return period in years).

    ``nu = rate * S_and`` (clipped to a probability), ``T = 1 / nu``. ``rate`` is the
    distinct-storm paired-event rate ``lambda`` (events/yr).
    """
    survival = np.asarray(survival, dtype=float)
    rate = float(event_rate)
    if not (np.isfinite(rate) and rate > 0):
        raise ValueError(f"event_rate must be finite and > 0, got {event_rate!r}")
    aep = np.clip(rate * survival, 0.0, 1.0)
    with np.errstate(divide="ignore"):
        period = np.where(survival > 0, 1.0 / (rate * survival), np.inf)
    return aep, period


def combined_and_frequency(physical, populations, *, seed=0) -> np.ndarray:
    """Storm-type mixture frequency ``nu(x) = sum_g lambda_g S_and,g(F_g(x))`` (1/yr).

    Each population ``g`` maps ``x`` through its own marginals, evaluates its own copula
    AND survival (seeded for reproducibility), and the population frequencies add.
    """
    physical = np.atleast_2d(np.asarray(physical, dtype=float))
    freq = np.zeros(len(physical), dtype=float)
    for k, pop in enumerate(populations):
        u = np.column_stack(
            [np.asarray(m.cdf(physical[:, j]), dtype=float) for j, m in enumerate(pop.marginals)]
        )
        survival = and_survival_from_cdf(np.clip(u, clip_eps, 1.0 - clip_eps), seeded_cdf(pop.copula, int(seed) + k))
        freq += float(pop.rate) * survival
    return freq


def combined_return_period(physical, populations, *, seed=0):
    """Storm-type mixture (frequency 1/yr, return period years)."""
    freq = combined_and_frequency(physical, populations, seed=seed)
    with np.errstate(divide="ignore"):
        period = np.where(freq > 0, 1.0 / freq, np.inf)
    return freq, period


def _physical_from_uniform(u, marginals):
    return np.column_stack(
        [np.asarray(m.ppf(np.clip(u[:, j], clip_eps, 1.0 - clip_eps)), dtype=float) for j, m in enumerate(marginals)]
    )


def select_design_events_on_and_isolines(
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
    """Most-likely design events on AND Joint Return Period isolines.

    Draws a copula candidate pool, labels every candidate with
    ``T = 1 / (lambda * S_and(F(x)))``, maps candidates to physical magnitudes, and
    chooses the maximum kernel-density point near each target return period. This is
    the v2 reference equivalent of production's AND-isoline design-event selector.
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
        row = {
            "target_return_period_years": float(target),
            "isoline_points": int(band.sum()),
            "relative_tolerance": float(tol),
        }
        if not band.any():
            row.update(
                {
                    "and_joint_return_period_years": np.nan,
                    "and_joint_exceedance_prob": np.nan,
                    "kde_density": np.nan,
                }
            )
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


select_most_likely_design_events = select_design_events_on_and_isolines


# --------------------------------------------------------------------------- #
# Drivers and the fitted joint law                                           #
# --------------------------------------------------------------------------- #
@runtime_checkable
class Marginal(Protocol):
    """A fitted per-driver marginal ``F_j``: enough to move between physical and
    probability scale and to read a univariate return period."""

    def cdf(self, x): ...        # F(x)
    def ppf(self, u): ...        # F^-1(u)
    def pdf(self, x): ...        # f(x)
    def return_period(self, x): ...  # univariate Driver Return Period


@dataclass(frozen=True)
class Driver:
    """One catalog driver and its declared physical role."""

    name: str
    role: str                    # stochastic | conditioning | response | validation
    marginal: Marginal | None = None  # present iff role == "stochastic"


@dataclass
class JointLaw:
    """A fitted vine over the stochastic Driver Probability Indices, with marginals
    and the distinct-storm rate. The copula is duck-typed: it exposes ``cdf(u)`` and
    ``simulate(n, qrng=?, seeds=[...])`` (e.g. a ``pyvinecopulib.Vinecop``)."""

    drivers: tuple[Driver, ...]
    copula: object
    rate: float                  # lambda, events/year
    survival_method: str = "cdf"

    @property
    def stochastic(self) -> list[Driver]:
        return [d for d in self.drivers if d.role == "stochastic"]

    @property
    def driver_names(self) -> list[str]:
        return [d.name for d in self.stochastic]

    @property
    def dim(self) -> int:
        return len(self.stochastic)

    def u(self, x) -> np.ndarray:
        """Map physical magnitudes ``x`` (n, d) to probability indices ``U = F(x)``."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.column_stack([d.marginal.cdf(x[:, j]) for j, d in enumerate(self.stochastic)])

    def x(self, u) -> np.ndarray:
        """Map probability indices ``u`` (n, d) back to physical magnitudes ``F^-1(u)``."""
        u = np.atleast_2d(np.asarray(u, dtype=float))
        return np.column_stack(
            [d.marginal.ppf(np.clip(u[:, j], clip_eps, 1.0 - clip_eps)) for j, d in enumerate(self.stochastic)]
        )

    def S_and(self, u) -> np.ndarray:
        return and_survival(u, self.copula, method=self.survival_method, seed=0)

    def and_survival_probability(self, u) -> np.ndarray:
        """Glossary-facing alias for the AND survival probability."""
        return self.S_and(u)

    def T_and(self, x) -> np.ndarray:
        _, period = and_return_period(self.S_and(self.u(x)), self.rate)
        return period

    def and_joint_return_period(self, x) -> np.ndarray:
        """Glossary-facing alias for the AND Joint Return Period."""
        return self.T_and(x)

    def simulate(self, n, *, qrng=True, seed=0) -> np.ndarray:
        return np.asarray(self.copula.simulate(int(n), qrng=qrng, seeds=[int(seed)]), dtype=float)


@dataclass
class MixtureLaw:
    """A mixture of per-storm-type laws whose AND exceedance frequencies add."""

    populations: list
    driver_names: list

    @property
    def total_rate(self) -> float:
        return float(sum(p.rate for p in self.populations))

    @property
    def dim(self) -> int:
        return len(self.driver_names)

    def combined_return_period(self, physical, *, seed=0):
        return combined_return_period(physical, self.populations, seed=seed)


# --------------------------------------------------------------------------- #
# Severity-band importance sampling                                          #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WeightedSelection:
    """The chosen catalog rows plus their canonical weights and provenance."""

    idx: np.ndarray
    band: np.ndarray
    sampling_weight: np.ndarray      # w_b = p_b/q_b
    probability_weight: np.ndarray   # pi_i = p_b/n_b (normalized; sums to 1)
    sampling_scheme: str
    catalog_role: str


def _round_to_total(fractions: dict, total: int) -> dict:
    raw = {name: float(frac) * total for name, frac in fractions.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[: max(0, remainder)]:
        counts[name] += 1
    return counts


def select_catalog_indices(pool_band, band_names, n_catalog, target_band_fractions, rng) -> WeightedSelection:
    """Choose catalog rows from a band-labeled pool: proportional, or band-stratified
    importance (Tail-Enriched Design Ensemble).

    ``target_band_fractions=None`` -> proportional draw (counts follow the fitted law,
    ``w_b=1``, ``pi_i=1/n``). Otherwise fill each band ``b`` to its target ``n_b``; every
    event in the band carries the same Sampling Weight ``w_b = p_b/q_b`` (the canonical
    convention), and ``probability_weight`` normalizes ``w_b`` so it
    rebuilds the true mass ``p_b`` per band.
    """
    pool_band = np.asarray(pool_band)
    pool_total = len(pool_band)
    if target_band_fractions is None:
        idx = rng.choice(np.arange(pool_total), size=int(n_catalog), replace=pool_total < int(n_catalog))
        band_arr = pool_band[idx].astype(str)
        sampling_weight = np.ones(len(idx), dtype=float)
        probability_weight = np.full(len(idx), 1.0 / len(idx), dtype=float)
        return WeightedSelection(idx, band_arr, sampling_weight, probability_weight,
                                 "probability_proportional", "probability")

    fractions = dict(target_band_fractions)
    target_counts = _round_to_total({name: fractions.get(name, 0.0) for name in band_names}, int(n_catalog))
    chosen_idx, chosen_band, band_weight = [], [], {}
    for name in band_names:
        members = np.flatnonzero(pool_band == name)
        n_b = int(target_counts.get(name, 0))
        if n_b <= 0:
            continue
        p_b = members.size / pool_total
        q_b = n_b / int(n_catalog)
        band_weight[name] = (p_b / q_b) if (members.size > 0 and p_b > 0) else 0.0
        if members.size == 0:
            continue
        picks = rng.choice(members, size=n_b, replace=members.size < n_b)
        chosen_idx.append(picks)
        chosen_band.extend([name] * n_b)

    if not chosen_idx:
        raise ValueError("no catalog events sampled; check target_band_fractions and pool support")
    idx = np.concatenate(chosen_idx)
    band_arr = np.asarray(chosen_band)
    sampling_weight = np.asarray([band_weight[name] for name in band_arr], dtype=float)
    weight_total = float(sampling_weight.sum())
    probability_weight = sampling_weight / weight_total if weight_total > 0 else np.full(len(idx), np.nan)
    return WeightedSelection(idx, band_arr, sampling_weight, probability_weight,
                             "band_stratified_importance", "design")


def check_stress_budget(catalog, stress_settings, *, severity_bands=None, raise_on_shortfall=True):
    """Gate: can the catalog fill the Resilience Stress/Training Set severity budget?

    Per-band report (catalog count, distinct support, needed count, pass/fail). Raises when a
    band with a positive budget cannot be met, so a thin joint tail fails loudly instead of
    silently starving the stress set. Distinct support below budget is flagged: those bands are
    filled by resampling duplicates and carry deep-tail Monte-Carlo uncertainty.
    """
    bands = severity_bands or default_severity_bands()
    target = int(stress_settings.get("target_event_count", 500))
    fractions = stress_settings.get("severity_band_fractions", {})
    u_cols = [c for c in catalog.columns if c.endswith("_u")]
    counts = catalog["severity_band"].astype(str).value_counts()
    if u_cols:
        distinct = catalog.assign(
            _key=catalog[u_cols].round(9).astype(str).agg("|".join, axis=1)
        ).groupby("severity_band")["_key"].nunique()
    else:
        distinct = counts

    rows, ok = [], True
    for band in bands:
        name = str(band["severity_band"])
        need = int(round(float(fractions.get(name, 0.0)) * target))
        have = int(counts.get(name, 0))
        unique = int(distinct.get(name, 0))
        meets = have >= need
        if need > 0 and not meets:
            ok = False
        rows.append({
            "severity_band": name,
            "catalog_count": have,
            "distinct_support": unique,
            "stress_budget_count": need,
            "meets_budget": meets,
            "low_support_flag": bool(need > 0 and unique < need),
        })
    report = pd.DataFrame(rows)
    if raise_on_shortfall and not ok:
        short = report[(~report["meets_budget"]) & (report["stress_budget_count"] > 0)]
        raise ValueError(
            f"Probability Catalog cannot fill the {target}-event stress budget; enrich the "
            f"joint tail (larger pool / higher target fractions). Shortfalls:\n{short.to_string(index=False)}"
        )
    return report


def build_joint_catalog(*args, **kwargs):
    from design_events.build import build_joint_catalog as _build_joint_catalog

    return _build_joint_catalog(*args, **kwargs)


def build_inland_catalog(*args, **kwargs):
    from design_events.build import build_inland_catalog as _build_inland_catalog

    return _build_inland_catalog(*args, **kwargs)


def attach_field_preserving_realization(*args, **kwargs):
    from design_events.realization import attach_field_preserving_realization as _attach

    return _attach(*args, **kwargs)


__all__ = [
    "Band", "default_bands", "assign_band", "default_severity_bands", "assign_severity_bands",
    "build_joint_catalog", "build_inland_catalog", "attach_field_preserving_realization",
    "check_stress_budget",
    "and_survival_from_cdf", "and_survival_empirical", "and_survival", "and_return_period",
    "combined_and_frequency", "combined_return_period",
    "select_design_events_on_and_isolines", "select_most_likely_design_events",
    "Marginal", "Driver", "JointLaw", "MixtureLaw",
    "WeightedSelection", "select_catalog_indices",
]
