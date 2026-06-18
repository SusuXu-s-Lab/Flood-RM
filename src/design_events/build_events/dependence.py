"""Vine-copula dependence fitting and joint catalog sampling.

This is the fitting + joint-sampling stage that sits between the marginals and the
AND labeling/selection stage (``joint_exceedance``). It:

1. fits a vine copula over the Driver Probability Indices (semiparametric: the
   copula is fit on rank-based pseudo-observations, the marginals stay parametric),
2. samples a Probability Catalog in joint-AND-return-period space from the fitted
   distribution by default, carrying equal ``probability_weight`` per event, and
3. optionally supports configured band-stratified tail enrichment for stress/design
   catalogs, carrying ``sampling_weight`` and ``probability_weight`` that reconstruct
   the true joint mass per Beck & Zuev (2015), and
4. gates the catalog against the stress-set severity budget so a thin joint tail
   fails loudly rather than silently starving the 500-event stress set.

When ``target_band_fractions`` is omitted, the returned catalog is an iid-style draw
from the fitted joint density: unweighted row counts are expected to reproduce the
historical/fitted severity distribution. When ``target_band_fractions`` is provided,
a large proportional pool is resampled by severity band. Drawing ``n_b`` catalog
events for band ``b`` (target fraction ``f_b``) from its ``m_b`` pool members gives
every event in the band the same between-band importance weight ``w_b = p_b / f_b``
where ``p_b = m_b / M`` is the band's true probability. Normalised,
``probability_weight`` reconstructs the true distribution even when the catalog is
tail-enriched by count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyvinecopulib as pv

from design_events.build_events.event_distribution import assign_severity_bands, default_severity_bands
from design_events.build_events.joint_exceedance import (
    and_return_period,
    and_survival_from_cdf,
    and_joint_survival,
    combined_return_period,
)

clip_eps = 1e-12

default_family_set = [
    pv.indep, pv.gaussian, pv.student, pv.clayton, pv.gumbel, pv.joe, pv.bb1, pv.bb7, pv.tawn,
]

# Parsimonious families for small storm-type populations (e.g. the handful of TC events):
# multi-parameter families (BB1/BB7/Tawn/Student) overfit a few points, so restrict to
# independence plus one-parameter copulas when a population is below the confidence threshold.
low_sample_family_set = [pv.indep, pv.gaussian, pv.clayton, pv.gumbel, pv.frank, pv.joe]

tail_enriched_catalog_band_fractions = {
    "mild": 0.05,
    "common": 0.28,
    "significant": 0.28,
    "rare": 0.12,
    "extreme": 0.27,
}


@dataclass
class DriverDependenceModel:
    """A fitted vine over Driver Probability Indices plus its marginals and rate."""

    vine: pv.Vinecop
    marginals: list
    driver_names: list
    event_rate: float

    @property
    def dim(self):
        return len(self.marginals)

    def vine_json(self):
        return self.vine.to_json()


def fit_driver_dependence(
    paired_observations,
    marginals,
    driver_names,
    event_rate,
    *,
    family_set=None,
    selection_criterion="aic",
    seed=0,
):
    """Fit a vine copula on rank pseudo-observations of paired driver indices.

    ``paired_observations`` is an (n, d) array of concurrent physical driver values
    (the two-sided POT co-occurrence sample). Marginals are kept parametric and the
    dependence is fit semiparametrically on pseudo-observations (Genest et al. 1995).
    """
    x = np.asarray(paired_observations, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"paired_observations must be 2-D (n, d), got shape {x.shape}")
    d = x.shape[1]
    if len(marginals) != d or len(driver_names) != d:
        raise ValueError(f"marginals ({len(marginals)}) and driver_names ({len(driver_names)}) must match d={d}")
    # Break rank ties randomly (coarsely-quantized drivers such as soil saturation have
    # many ties; average ranks would band the pseudo-observations and bias the fit).
    u = pv.to_pseudo_obs(np.asfortranarray(x), ties_method="random", seeds=[int(seed)])
    controls = pv.FitControlsVinecop(
        family_set=list(family_set) if family_set is not None else list(default_family_set),
        selection_criterion=selection_criterion,
        seeds=[int(seed)],
    )
    vine = pv.Vinecop.from_data(np.asfortranarray(u), controls=controls)
    return DriverDependenceModel(vine=vine, marginals=list(marginals), driver_names=list(driver_names), event_rate=float(event_rate))


def _round_to_total(fractions, total):
    raw = {name: float(frac) * total for name, frac in fractions.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    # hand out the rounding remainder to the largest fractional parts
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[: max(0, remainder)]:
        counts[name] += 1
    return counts


def _select_catalog_indices(pool_band, band_names, n_catalog, target_band_fractions, rng):
    """Choose catalog rows from a band-labeled pool: proportional, or band-stratified importance.

    With ``target_band_fractions=None`` the draw is proportional (unweighted counts follow the
    fitted distribution). Otherwise each band ``b`` is filled to its target count from its pool
    members and every event in the band carries the same between-band importance weight
    ``w_b = p_b / f_b`` (Beck & Zuev 2015), so ``probability_weight`` rebuilds the true mass.
    Returns ``(idx, band_arr, sampling_weight, probability_weight, sampling_scheme, catalog_role)``.
    """
    pool_total = len(pool_band)
    if target_band_fractions is None:
        idx = rng.choice(np.arange(pool_total), size=int(n_catalog), replace=pool_total < int(n_catalog))
        band_arr = pool_band[idx].astype(str)
        sampling_weight = np.ones(len(idx), dtype=float)
        probability_weight = np.full(len(idx), 1.0 / len(idx), dtype=float)
        return idx, band_arr, sampling_weight, probability_weight, "probability_proportional", "probability"

    fractions = dict(target_band_fractions)
    target_counts = _round_to_total({name: fractions.get(name, 0.0) for name in band_names}, int(n_catalog))
    chosen_idx, chosen_band, band_weight = [], [], {}
    for name in band_names:
        members = np.flatnonzero(pool_band == name)
        n_b = int(target_counts.get(name, 0))
        if n_b <= 0:
            continue
        p_b = members.size / pool_total
        # between-band importance weight; identical for every event in the band
        band_weight[name] = (p_b / (n_b / int(n_catalog))) if (members.size > 0 and p_b > 0) else 0.0
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
    return idx, band_arr, sampling_weight, probability_weight, "band_stratified_importance", "design"


def sample_tail_enriched_catalog(
    model,
    n_catalog,
    *,
    target_band_fractions=None,
    severity_bands=None,
    splice_rp_years=10.0,
    pool_size=500_000,
    qrng=True,
    seed=0,
    survival_method="cdf",
    id_prefix="design",
):
    """Sample a joint Probability Catalog labeled with AND joint return periods.

    Draws a large proportional pool from the fitted vine and labels each pool member
    with its AND joint return period. By default, the catalog is sampled
    proportionally from that pool, so unweighted row counts follow the fitted
    historical distribution. Passing ``target_band_fractions`` switches to deliberate
    band-stratified/importance sampling for enriched stress or design catalogs.
    """
    bands = severity_bands or default_severity_bands()
    band_names = [str(b["severity_band"]) for b in bands]
    rng = np.random.default_rng(int(seed))

    pool_u = np.asarray(model.vine.simulate(int(pool_size), qrng=qrng, seeds=[int(seed)]), dtype=float)
    if survival_method == "cdf":
        survival = and_survival_from_cdf(pool_u, model.vine.cdf)
    else:
        survival = and_joint_survival(pool_u, copula=model.vine, method="empirical", n_reference=int(pool_size), seed=int(seed) + 1)
    survival = np.clip(np.asarray(survival, dtype=float), clip_eps, 1.0)
    _, pool_rp = and_return_period(survival, model.event_rate)
    pool_band = assign_severity_bands(pool_rp, bands=bands).to_numpy()
    pool_total = len(pool_u)
    support = {name: int(np.sum(pool_band == name)) for name in band_names}
    support_probability = {name: support[name] / pool_total for name in band_names}

    idx, band_arr, sampling_weight, probability_weight, sampling_scheme, catalog_role = _select_catalog_indices(
        pool_band, band_names, n_catalog, target_band_fractions, rng
    )

    sel_u = pool_u[idx]
    sel_rp = pool_rp[idx]
    sel_survival = survival[idx]

    physical = np.column_stack(
        [np.asarray(m.ppf(np.clip(sel_u[:, j], clip_eps, 1.0 - clip_eps)), dtype=float) for j, m in enumerate(model.marginals)]
    )
    region = np.where(sel_rp < float(splice_rp_years), "body", "tail")

    catalog = pd.DataFrame(
        {
            "event_id": [f"{id_prefix}_{i:04d}" for i in range(1, len(idx) + 1)],
            "sample_rp_years": sel_rp,
            "and_joint_exceedance_prob": sel_survival,
            "severity_band": band_arr,
            "sampling_region": region,
            "sampling_weight": sampling_weight,
            "probability_weight": probability_weight,
            "pool_band_support": [support[name] for name in band_arr],
            "pool_band_probability": [support_probability[name] for name in band_arr],
            "candidate_pool_count": pool_total,
            "event_origin": np.where(region == "body", "synthetic_body", "synthetic_tail"),
            "catalog_role": catalog_role,
            "sampling_scheme": sampling_scheme,
        }
    )
    for j, name in enumerate(model.driver_names):
        catalog[name] = physical[:, j]
        catalog[f"{name}_u"] = sel_u[:, j]
    return catalog.sort_values("sample_rp_years").reset_index(drop=True)


def check_stress_budget(catalog, stress_settings, *, severity_bands=None, raise_on_shortfall=True):
    """Gate: can the catalog fill the Resilience Stress/Training Set severity budget?

    Returns a per-band report (catalog count, distinct-event support, the count the
    stress set needs, pass/fail). Raises when a band with a positive budget cannot be
    met, so a thin joint tail fails loudly instead of silently starving the 500 set.
    Distinct support below the budget is flagged: those bands are filled by resampling
    duplicates and carry the deep-tail Monte-Carlo uncertainty (Beck & Zuev 2015).
    """
    bands = severity_bands or default_severity_bands()
    target_500 = int(stress_settings.get("target_event_count", 500))
    fractions = stress_settings.get("severity_band_fractions", {})
    u_cols = [c for c in catalog.columns if c.endswith("_u")]
    counts = catalog["severity_band"].astype(str).value_counts()
    if u_cols:
        distinct = catalog.assign(_key=catalog[u_cols].round(9).astype(str).agg("|".join, axis=1)).groupby("severity_band")["_key"].nunique()
    else:
        distinct = counts

    rows, ok = [], True
    for band in bands:
        name = str(band["severity_band"])
        need = int(round(float(fractions.get(name, 0.0)) * target_500))
        have = int(counts.get(name, 0))
        unique = int(distinct.get(name, 0))
        meets = have >= need
        if need > 0 and not meets:
            ok = False
        rows.append(
            {
                "severity_band": name,
                "catalog_count": have,
                "distinct_support": unique,
                "stress_budget_count": need,
                "meets_budget": meets,
                "low_support_flag": bool(need > 0 and unique < need),
            }
        )
    report = pd.DataFrame(rows)
    if raise_on_shortfall and not ok:
        short = report[(~report["meets_budget"]) & (report["stress_budget_count"] > 0)]
        raise ValueError(
            "Probability Catalog cannot fill the "
            f"{target_500}-event stress budget; enrich the joint tail (larger pool / higher "
            f"target fractions). Shortfalls:\n{short.to_string(index=False)}"
        )
    return report


@dataclass
class StormTypePopulation:
    """One fitted storm-type population: its copula, marginals, and annual rate."""

    storm_type: str
    vine: pv.Vinecop
    marginals: list
    rate: float          # lambda_pop, events/yr — its share of the base distinct-storm rate
    n_events: int
    low_confidence: bool


@dataclass
class MixtureDependenceModel:
    """A mixture of per-storm-type copulas whose AND exceedance frequencies add (Fix 3)."""

    populations: list
    driver_names: list

    @property
    def dim(self):
        return len(self.driver_names)

    @property
    def total_rate(self):
        return float(sum(p.rate for p in self.populations))


def fit_storm_type_mixture(
    paired,
    driver_names,
    *,
    marginal_kinds,
    base_rate,
    fit_marginal,
    total_rows=None,
    min_population_events=20,
    min_fit_events=8,
    family_set=None,
    seed=0,
):
    """Fit a separate copula per storm-type population and weight them by frequency.

    ``paired`` carries a ``storm_type`` column plus the driver columns. Each population gets
    its own per-driver marginals (via the ``fit_marginal(values, rate, kind)`` callable, kept
    injected to avoid importing the catalog builder) and its own vine. Its rate is the base
    distinct-storm rate split by the population's share of events, so the population frequencies
    sum back to the Fix-1 realized rate. Tiny populations (``< min_population_events``, e.g. the
    handful of TCs) are fit with a parsimonious family set and flagged ``low_confidence``;
    populations below ``min_fit_events`` are skipped. Returns ``(model, report_frame)``.
    """
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
        observations = group[list(driver_names)].to_numpy(dtype=float)
        marginals = [
            fit_marginal(observations[:, j], rate, str((marginal_kinds or {}).get(driver, "pot")))
            for j, driver in enumerate(driver_names)
        ]
        low = n < int(min_population_events)
        families = low_sample_family_set if low else (family_set or default_family_set)
        fit = fit_driver_dependence(observations, marginals, list(driver_names), rate, family_set=families, seed=seed + k)
        populations.append(StormTypePopulation(str(storm_type), fit.vine, marginals, rate, n, low))
        rows.append({"storm_type": storm_type, "n_events": n, "rate": rate, "fitted": True, "low_confidence": low})
    if not populations:
        raise ValueError("no storm-type population had enough events to fit a copula")
    return MixtureDependenceModel(populations, list(driver_names)), pd.DataFrame(rows)


def sample_mixture_catalog(
    model,
    n_catalog,
    *,
    target_band_fractions=None,
    severity_bands=None,
    splice_rp_years=10.0,
    pool_size=500_000,
    qrng=True,
    seed=0,
    id_prefix="design",
):
    """Sample an AND-labeled catalog from the storm-type mixture (Fix 3).

    Draws each population's share of the candidate pool from its own copula+marginals, then
    labels every pooled event with the *combined* AND return period across all populations
    (``T = 1 / sum_p lambda_p S_p``). Band-enrichment/importance weighting is shared with the
    single-population sampler. Each row carries the ``storm_type`` it was drawn from.
    """
    bands = severity_bands or default_severity_bands()
    band_names = [str(b["severity_band"]) for b in bands]
    rng = np.random.default_rng(int(seed))
    total_rate = model.total_rate

    phys_parts, u_parts, type_parts = [], [], []
    for k, pop in enumerate(model.populations):
        n_pop = max(1, int(round(int(pool_size) * pop.rate / total_rate)))
        u = np.asarray(pop.vine.simulate(n_pop, qrng=qrng, seeds=[int(seed) + k]), dtype=float)
        phys = np.column_stack(
            [np.asarray(m.ppf(np.clip(u[:, j], clip_eps, 1.0 - clip_eps)), dtype=float) for j, m in enumerate(pop.marginals)]
        )
        phys_parts.append(phys)
        u_parts.append(u)
        type_parts.append(np.array([pop.storm_type] * n_pop))
    pool_phys = np.vstack(phys_parts)
    pool_u = np.vstack(u_parts)
    pool_type = np.concatenate(type_parts)

    freq, pool_rp = combined_return_period(pool_phys, model.populations)
    pool_band = assign_severity_bands(pool_rp, bands=bands).to_numpy()
    pool_total = len(pool_phys)
    support = {name: int(np.sum(pool_band == name)) for name in band_names}
    support_probability = {name: support[name] / pool_total for name in band_names}

    idx, band_arr, sampling_weight, probability_weight, sampling_scheme, catalog_role = _select_catalog_indices(
        pool_band, band_names, n_catalog, target_band_fractions, rng
    )
    sel_phys, sel_u, sel_rp, sel_freq, sel_type = pool_phys[idx], pool_u[idx], pool_rp[idx], freq[idx], pool_type[idx]
    region = np.where(sel_rp < float(splice_rp_years), "body", "tail")

    catalog = pd.DataFrame(
        {
            "event_id": [f"{id_prefix}_{i:04d}" for i in range(1, len(idx) + 1)],
            "sample_rp_years": sel_rp,
            "and_joint_exceedance_prob": np.clip(sel_freq / total_rate, 0.0, 1.0),
            "severity_band": band_arr,
            "sampling_region": region,
            "sampling_weight": sampling_weight,
            "probability_weight": probability_weight,
            "pool_band_support": [support[name] for name in band_arr],
            "pool_band_probability": [support_probability[name] for name in band_arr],
            "candidate_pool_count": pool_total,
            "event_origin": np.where(region == "body", "synthetic_body", "synthetic_tail"),
            "catalog_role": catalog_role,
            "sampling_scheme": sampling_scheme,
            "storm_type": sel_type,
        }
    )
    for j, name in enumerate(model.driver_names):
        catalog[name] = sel_phys[:, j]
        catalog[f"{name}_u"] = sel_u[:, j]
    return catalog.sort_values("sample_rp_years").reset_index(drop=True)
