"""Reference workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyvinecopulib as pv

from design_events.coastal import coastal_realization_metadata
from design_events.catalog import ReferenceBundle, write_reference_bundle
from design_events.audit_metrics import build_audit_metrics
from design_events.inland import build_inland_reference_bundle_inputs, inland_reference_metadata
from design_events.mixture import fit_mixture_law, sample_mixture
from design_events.probability import (
    Driver,
    JointLaw,
    MixtureLaw,
    and_return_period,
    assign_band,
    default_bands,
    select_catalog_indices,
)
from design_events.realization import realize_driver
from design_events.records import fit_marginal, load_records, paired_pot
from design_events.timing import attach_timing


@dataclass(frozen=True)
class _OneDimensionalCopula:
    """Uniform one-driver copula for rainfall-only inland reference bundles."""

    def cdf(self, u):
        return np.atleast_2d(np.asarray(u, dtype=float))[:, 0]

    def simulate(self, n, *, qrng=True, seeds=(0,)):
        rng = np.random.default_rng(int(seeds[0]) if seeds else 0)
        return rng.uniform(size=(int(n), 1))


def build_reference_bundle(config, *, output_dir, seed=0) -> ReferenceBundle:
    """Build and write the v2 reference Event Catalog bundle.

    Required config sections:

    ``records``
        Mapping of driver name to CSV read spec accepted by
        ``records.load_driver_series``.
    ``member_libraries``
        Mapping of driver name to CSV path/spec for Field-Preserving Realization.
    ``dependence.driver_vector``
        Ordered stochastic Driver Probability Indices.
    """

    cfg = build_inland_reference_bundle_inputs(dict(config or {}))
    dependence = _dependence(cfg)
    driver_vector = list(dependence.get("driver_vector") or [])
    if not driver_vector:
        raise ValueError("dependence.driver_vector is required")

    series = load_records(cfg, location_root=cfg.get("location_root"))
    missing = [driver for driver in driver_vector if driver not in series]
    if missing:
        raise ValueError(f"records missing driver series: {missing}")

    paired = paired_pot(series, cfg)
    event_rate = _event_rate(paired, dependence)
    law = fit_law(paired, driver_vector, event_rate, dependence, seed=seed)
    events, u_selected, x_selected = sample_catalog(law, cfg, dependence, seed=seed)
    member_tables = _load_member_libraries(cfg)
    drivers = _realize_drivers(events, u_selected, x_selected, driver_vector, cfg, member_tables, seed=seed)
    events, drivers, timing_audit = attach_timing(events, drivers, paired, member_tables, cfg, seed=seed)
    audit = _audit(cfg, dependence, paired, law, events, drivers, seed=seed)
    if cfg.get("runtime_audit"):
        audit["runtime"] = cfg["runtime_audit"]
    audit["timing"] = timing_audit
    if _is_coastal_reference(cfg, drivers):
        audit["coastal"] = coastal_realization_metadata(events, drivers, None, cfg)
    if _is_inland_reference(cfg):
        audit["inland"] = inland_reference_metadata(events, drivers, cfg)
    audit["diagnostics"] = build_audit_metrics(events, drivers, audit)
    return write_reference_bundle(events, drivers, audit, output_dir)


def _dependence(config):
    if "dependence" in config:
        return dict(config.get("dependence") or {})
    return dict(((config.get("event_catalog") or {}).get("dependence") or {}))


def _event_rate(paired, dependence):
    realized = float(paired.attrs.get("base_event_rate_per_year", float("nan")))
    if np.isfinite(realized) and realized > 0:
        return realized
    configured = float(dependence.get("event_rate_per_year", float("nan")))
    if np.isfinite(configured) and configured > 0:
        return configured
    raise ValueError("event_rate_per_year must be configured when paired observations do not carry a realized rate")


def fit_law(paired, driver_vector, event_rate, dependence, *, seed=0):
    """Fit the joint law over the stochastic Driver Probability Indices (the public seam).

    Dispatches to a storm-type **MixtureLaw** when ``storm_stratification`` is enabled and the
    paired sample carries a ``storm_type`` column, else a single **JointLaw**.
    """
    strat = dependence.get("storm_stratification", {}) or {}
    if strat.get("enabled") and "storm_type" in getattr(paired, "columns", []):
        marginal_cfg = dependence.get("marginals") or {}
        marginal_kinds = {d: str((marginal_cfg.get(d) or {}).get("kind", "pot")) for d in driver_vector}
        law, _report = fit_mixture_law(
            paired, driver_vector, base_rate=event_rate, marginal_kinds=marginal_kinds,
            min_population_events=int(strat.get("min_population_events", 20)), seed=seed,
        )
        return law
    return _fit_law(paired, driver_vector, event_rate, dependence, seed=seed)


def sample_catalog(law, config, dependence, *, seed=0, id_prefix="v2"):
    """Sample a long Event Catalog from a fitted law (single or mixture).

    Returns ``(events, u_selected, x_selected)`` — the same shape for both law kinds — so
    ``realize_events``/``attach_timing`` compose identically downstream.
    """
    if isinstance(law, MixtureLaw):
        return _sample_mixture_events(law, config, dependence, seed=seed, id_prefix=id_prefix)
    return _sample_events(law, config, dependence, seed=seed, id_prefix=id_prefix)


def _fit_law(paired, driver_vector, event_rate, dependence, *, seed):
    marginal_cfg = dict(dependence.get("marginals") or {})
    marginals = []
    for driver in driver_vector:
        spec = dict(marginal_cfg.get(driver) or {})
        marginals.append(
            fit_marginal(
                paired[driver].to_numpy(dtype=float),
                extremes_rate=event_rate,
                kind=str(spec.get("kind", "pot")),
                ev_type=str(spec.get("ev_type", "pot")),
                criterium=str(spec.get("criterium", "AIC")),
            )
        )

    observations = paired[driver_vector].to_numpy(dtype=float)
    if len(driver_vector) == 1:
        copula = _OneDimensionalCopula()
    else:
        u = pv.to_pseudo_obs(np.asfortranarray(observations), ties_method="random", seeds=[int(seed)])
        controls = pv.FitControlsVinecop(
            selection_criterion=str(dependence.get("selection_criterion", "aic")),
            seeds=[int(seed)],
        )
        copula = pv.Vinecop.from_data(np.asfortranarray(u), controls=controls)

    drivers = tuple(Driver(name=name, role="stochastic", marginal=marginal) for name, marginal in zip(driver_vector, marginals))
    return JointLaw(
        drivers=drivers,
        copula=copula,
        rate=event_rate,
        survival_method=str(dependence.get("survival_method", "cdf")),
    )


def _sample_events(law, config, dependence, *, seed, id_prefix="v2"):
    n_catalog = int((config.get("events") or {}).get("target_event_count", dependence.get("target_event_count", 2500)))
    pool_size = int(dependence.get("pool_size", max(10_000, n_catalog * 100)))
    rng = np.random.default_rng(int(seed))

    pool_u = law.simulate(pool_size, seed=seed)
    pool_x = law.x(pool_u)
    survival = np.clip(law.S_and(pool_u), 1e-12, 1.0)
    aep, return_period = and_return_period(survival, law.rate)
    pool_band = assign_band(return_period).to_numpy()
    band_names = [band.name for band in default_bands()]
    selection = select_catalog_indices(
        pool_band,
        band_names,
        n_catalog,
        dependence.get("catalog_band_fractions"),
        rng,
    )

    sel_u = pool_u[selection.idx]
    sel_x = pool_x[selection.idx]
    sel_survival = survival[selection.idx]
    sel_aep = aep[selection.idx]
    sel_rp = return_period[selection.idx]
    splice_rp = float(dependence.get("splice_rp_years", 10.0))
    event_family = str(config.get("event_family") or dependence.get("event_family") or "copula_joint_compound")
    scenario_name = str(config.get("scenario_name") or "base")

    events = pd.DataFrame(
        {
            "event_id": [f"{id_prefix}_{i:04d}" for i in range(1, len(selection.idx) + 1)],
            "event_role": "probability_design",
            "event_origin": np.where(sel_rp < splice_rp, "synthetic_body", "synthetic_tail"),
            "event_family": event_family,
            "scenario_name": scenario_name,
            "sample_rp_years": sel_rp,
            "and_joint_exceedance_prob": sel_survival,
            "and_joint_aep": sel_aep,
            "severity_band": selection.band,
            "sampling_region": np.where(sel_rp < splice_rp, "body", "tail"),
            "sampling_weight": selection.sampling_weight,
            "probability_weight": selection.probability_weight,
            "event_reference_time": pd.NA,
            "selection_reason": selection.sampling_scheme,
            "sampling_scheme": selection.sampling_scheme,
            "catalog_role": selection.catalog_role,
        }
    )
    for j, driver in enumerate(law.driver_names):
        events[driver] = sel_x[:, j]
        events[f"{driver}_u"] = sel_u[:, j]
    return events, sel_u, sel_x


def _sample_mixture_events(law, config, dependence, *, seed, id_prefix="v2"):
    n_catalog = int((config.get("events") or {}).get("target_event_count", dependence.get("target_event_count", 2500)))
    pool_size = int(dependence.get("pool_size", max(10_000, n_catalog * 100)))
    cat = sample_mixture(
        law, n_catalog, band_fractions=dependence.get("catalog_band_fractions"),
        pool_size=pool_size, seed=seed, id_prefix=id_prefix,
    )
    names = law.driver_names
    sel_x = cat[names].to_numpy(dtype=float)
    sel_u = cat[[f"{d}_u" for d in names]].to_numpy(dtype=float)
    # Add the reviewer-bundle columns the single-law sampler emits, so both paths align.
    events = cat.copy()
    events["event_role"] = "probability_design"
    events["event_family"] = str(config.get("event_family") or dependence.get("event_family") or "copula_joint_compound")
    events["scenario_name"] = str(config.get("scenario_name") or "base")
    events["and_joint_aep"] = np.clip(1.0 / events["sample_rp_years"].to_numpy(dtype=float), 0.0, 1.0)
    events["event_reference_time"] = pd.NA
    events["selection_reason"] = events["sampling_scheme"]
    return events, sel_u, sel_x


def _realize_drivers(events, u_selected, x_selected, driver_vector, config, member_tables, *, seed):
    member_specs = dict(config.get("member_libraries") or {})
    rows = []
    for j, driver in enumerate(driver_vector):
        if driver not in member_specs:
            raise ValueError(f"member_libraries missing driver {driver!r}")
        spec = dict(member_specs[driver] or {})
        members = member_tables[driver]
        index_column = str(spec.get("index_column", "value"))
        target_column = driver
        realized = realize_driver(
            events.assign(**{target_column: x_selected[:, j]}),
            members,
            driver=driver,
            target_column=target_column,
            index_column=index_column,
            member_id_column=str(spec.get("member_id_column", "member_id")),
            member_file_column=str(spec.get("member_file_column", "member_file")),
            time_column=spec.get("time_column", "time"),
            design_method=str(spec.get("realization_policy", f"v2_scaled_{driver}_analog")),
            observed_lags=spec.get("observed_lags"),
            default_lag_hours=float(spec.get("default_lag_hours", 0.0)),
            log_space=bool(spec.get("log_space", True)),
            pool_size=int(spec.get("pool_size", 75)),
            reuse_penalty_lambda=float(spec.get("reuse_penalty_lambda", 0.15)),
            seed=int(seed) + j,
        )
        realized["u"] = u_selected[:, j]
        realized["driver_role"] = str(spec.get("driver_role", "stochastic"))
        realized["time_policy"] = str(spec.get("time_policy", "member_time_plus_lag"))
        realized["source"] = str(spec.get("source", spec.get("path", driver)))
        rows.append(realized)
    return pd.concat(rows, ignore_index=True, sort=False)


def _load_member_libraries(config):
    return {
        name: _load_members(spec, config.get("location_root"))
        for name, spec in dict(config.get("member_libraries") or {}).items()
        if "path" in dict(spec or {})
    }


def _load_members(spec, location_root=None):
    path = Path(spec["path"])
    if not path.is_absolute() and location_root is not None:
        path = Path(location_root) / path
    if not path.exists():
        raise FileNotFoundError(f"member library not found: {path}")
    frame = pd.read_csv(path)
    frame.attrs["source_file"] = str(path)
    return frame


def _is_coastal_reference(config, drivers):
    family = str(config.get("event_family", ""))
    names = set(pd.DataFrame(drivers).get("driver", pd.Series(dtype=str)).astype(str))
    return "coastal" in family or bool({"coastal_ntr", "coastal_water_level"} & names)


def _is_inland_reference(config):
    return "inland" in str(config.get("event_family", "")) or bool(config.get("inland_wflow_coupled", False))


def _audit(config, dependence, paired, law, events, drivers, *, seed):
    band_mass = (
        events.groupby("severity_band")["probability_weight"].sum().astype(float).to_dict()
        if "probability_weight" in events
        else {}
    )
    band_fraction = events["severity_band"].value_counts(normalize=True).sort_index().to_dict()
    unique_members = drivers.groupby("driver")["member_id"].nunique().astype(int).to_dict()
    scale_quantiles = {}
    max_reuse = {}
    for driver, group in drivers.groupby("driver"):
        scale = pd.to_numeric(group["scale_factor"], errors="coerce")
        scale_quantiles[driver] = {
            "q05": float(scale.quantile(0.05)),
            "q50": float(scale.quantile(0.50)),
            "q95": float(scale.quantile(0.95)),
        }
        counts = group["member_id"].value_counts(normalize=True)
        max_reuse[driver] = float(counts.iloc[0]) if not counts.empty else 0.0

    return {
        "model": {
            "event_rate_per_year": float(law.rate),
            "drivers": law.driver_names,
            "joint_probability": "AND",
            "return_period_formula": "T = 1 / (lambda * S_and(F(x)))",
            "sampling_weight_formula": "w_b = p_b / q_b",
            "probability_weight_formula": "pi_i = p_b / n_b",
            "survival_method": law.survival_method,
        },
        "data": {
            "record_years": float(paired.attrs.get("record_years", float("nan"))),
            "paired_event_count": int(len(paired)),
            "distinct_storm_rate_per_year": float(paired.attrs.get("base_event_rate_per_year", float("nan"))),
        },
        "marginals": {
            driver.name: {
                "dist": getattr(driver.marginal, "dist_name", "unknown"),
                "params": list(getattr(driver.marginal, "params", [])),
                "extremes_rate": float(getattr(driver.marginal, "extremes_rate", float("nan"))),
                "peak_count": int(getattr(driver.marginal, "peak_count", 0)),
            }
            for driver in law.stochastic
        },
        "copula": {
            "class": type(law.copula).__name__,
            "dimension": len(law.driver_names),
        },
        "storm_type_populations": [],
        "sampling": {
            "catalog_count": int(len(events)),
            "pool_size": int(dependence.get("pool_size", max(10_000, len(events) * 100))),
            "band_true_mass": band_mass,
            "band_design_fraction": {str(k): float(v) for k, v in band_fraction.items()},
        },
        "realization": {
            "unique_members_by_driver": unique_members,
            "scale_factor_quantiles_by_driver": scale_quantiles,
            "max_member_reuse_fraction_by_driver": max_reuse,
        },
        "config": {
            "seed": int(seed),
            "event_family": config.get("event_family"),
            "scenario_name": config.get("scenario_name", "base"),
        },
        "checks": [
            {
                "name": "probability_weight_sum",
                "value": float(pd.to_numeric(events["probability_weight"], errors="coerce").sum()),
                "expected": 1.0,
            },
            {
                "name": "field_preserving_realization_rows",
                "value": int(len(drivers)),
                "expected": int(len(events) * len(law.driver_names)),
            },
        ],
    }


__all__ = ["build_reference_bundle", "fit_law", "sample_catalog"]
