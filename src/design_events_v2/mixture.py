"""Storm-type mixture of joint laws (coastal, ADR-0011).

Different coastal storm mechanisms (tropical, nor'easter, other) have different
dependence. Fit a separate vine + marginals per **Storm Type** population, weight each by
its share of the distinct-storm rate, and add their AND exceedance frequencies:
``nu(x) = sum_g lambda_g S_and,g(F_g(x))``. Faithful to production
``dependence.fit_storm_type_mixture`` / ``sample_mixture_catalog`` (reconciled by test);
the math (``combined_return_period``) and the importance sampler
(``select_catalog_indices``) are shared with the single-law path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyvinecopulib as pv

from design_events_v2.probability import (
    MixtureLaw,
    assign_band,
    combined_return_period,
    default_bands,
    select_catalog_indices,
)
from design_events_v2.records import fit_marginal

clip_eps = 1e-12

# Parsimonious families for small populations (a few TCs overfit multi-parameter families).
_default_family_set = [pv.indep, pv.gaussian, pv.student, pv.clayton, pv.gumbel, pv.joe, pv.bb1, pv.bb7, pv.tawn]
_low_sample_family_set = [pv.indep, pv.gaussian, pv.clayton, pv.gumbel, pv.frank, pv.joe]


@dataclass
class Population:
    """One fitted Storm Type population: its copula, marginals, and annual rate."""

    storm_type: str
    copula: object          # has .cdf / .simulate (a pyvinecopulib.Vinecop)
    marginals: list
    rate: float             # lambda_g, its share of the base distinct-storm rate
    n_events: int
    low_confidence: bool


def fit_mixture_law(paired, driver_vector, *, base_rate, marginal_kinds=None, total_rows=None,
                    min_population_events=20, min_fit_events=8, seed=0):
    """Fit one joint law per Storm Type and weight populations by frequency.

    ``paired`` carries a ``storm_type`` column plus the driver columns. Each population's
    rate is ``base_rate * n_g/total`` so the population frequencies sum back to the base
    distinct-storm rate. Tiny populations use a parsimonious family set and are flagged
    ``low_confidence``; populations below ``min_fit_events`` are skipped.
    Returns ``(MixtureLaw, report_frame)``.
    """
    driver_vector = list(driver_vector)
    marginal_kinds = marginal_kinds or {}
    total_rows = int(total_rows if total_rows is not None else len(paired))
    populations, rows = [], []
    for k, (storm_type, group) in enumerate(paired.groupby("storm_type")):
        if storm_type == "unresolved":
            continue
        n = len(group)
        rate = float(base_rate) * (n / float(total_rows))
        if n < int(min_fit_events):
            rows.append({"storm_type": storm_type, "n_events": n, "rate": rate, "fitted": False, "low_confidence": True})
            continue
        observations = group[driver_vector].to_numpy(dtype=float)
        marginals = [
            fit_marginal(observations[:, j], extremes_rate=rate, kind=str(marginal_kinds.get(driver, "pot")))
            for j, driver in enumerate(driver_vector)
        ]
        low = n < int(min_population_events)
        families = _low_sample_family_set if low else _default_family_set
        u = pv.to_pseudo_obs(np.asfortranarray(observations), ties_method="random", seeds=[int(seed) + k])
        controls = pv.FitControlsVinecop(family_set=list(families), selection_criterion="aic", seeds=[int(seed) + k])
        vine = pv.Vinecop.from_data(np.asfortranarray(u), controls=controls)
        populations.append(Population(str(storm_type), vine, marginals, rate, n, low))
        rows.append({"storm_type": storm_type, "n_events": n, "rate": rate, "fitted": True, "low_confidence": low})
    if not populations:
        raise ValueError("no storm-type population had enough events to fit a copula")
    return MixtureLaw(populations, driver_vector), pd.DataFrame(rows)


def sample_mixture(law, n_catalog, *, band_fractions=None, pool_size=500_000, qrng=True, seed=0,
                   splice_rp_years=10.0, id_prefix="v2"):
    """Sample an AND-labeled catalog from the storm-type mixture.

    Draws each population's share of the pool from its own copula+marginals, labels every
    pooled event with the *combined* AND return period across populations, then applies the
    shared Tail-Enriched importance sampler. Each row keeps the ``storm_type`` it came from.
    """
    rng = np.random.default_rng(int(seed))
    total_rate = law.total_rate

    phys_parts, u_parts, type_parts = [], [], []
    for k, pop in enumerate(law.populations):
        n_pop = max(1, int(round(int(pool_size) * pop.rate / total_rate)))
        u = np.asarray(pop.copula.simulate(n_pop, qrng=qrng, seeds=[int(seed) + k]), dtype=float)
        phys = np.column_stack(
            [np.asarray(m.ppf(np.clip(u[:, j], clip_eps, 1.0 - clip_eps)), dtype=float) for j, m in enumerate(pop.marginals)]
        )
        phys_parts.append(phys)
        u_parts.append(u)
        type_parts.append(np.array([pop.storm_type] * n_pop))
    pool_phys = np.vstack(phys_parts)
    pool_u = np.vstack(u_parts)
    pool_type = np.concatenate(type_parts)

    freq, pool_rp = combined_return_period(pool_phys, law.populations, seed=seed)
    pool_band = assign_band(pool_rp).to_numpy()
    band_names = [b.name for b in default_bands()]
    sel = select_catalog_indices(pool_band, band_names, n_catalog, band_fractions, rng)

    sel_phys, sel_u, sel_rp, sel_freq, sel_type = (
        pool_phys[sel.idx], pool_u[sel.idx], pool_rp[sel.idx], freq[sel.idx], pool_type[sel.idx],
    )
    region = np.where(sel_rp < float(splice_rp_years), "synthetic_body", "synthetic_tail")
    catalog = pd.DataFrame(
        {
            "event_id": [f"{id_prefix}_{i:04d}" for i in range(1, len(sel.idx) + 1)],
            "sample_rp_years": sel_rp,
            "and_joint_exceedance_prob": np.clip(sel_freq / total_rate, 0.0, 1.0),
            "severity_band": sel.band,
            "sampling_region": np.where(sel_rp < float(splice_rp_years), "body", "tail"),
            "sampling_weight": sel.sampling_weight,
            "probability_weight": sel.probability_weight,
            "event_origin": region,
            "catalog_role": sel.catalog_role,
            "sampling_scheme": sel.sampling_scheme,
            "storm_type": sel_type,
        }
    )
    for j, name in enumerate(law.driver_names):
        catalog[name] = sel_phys[:, j]
        catalog[f"{name}_u"] = sel_u[:, j]
    return catalog.sort_values("sample_rp_years").reset_index(drop=True)


__all__ = ["Population", "fit_mixture_law", "sample_mixture"]
