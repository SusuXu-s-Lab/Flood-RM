"""Reference workflow.

The public Interface is intentionally small: ``build_reference_bundle`` takes an
Event Catalog-style config plus Source Artifact paths, then writes the reviewer-facing
``events.csv`` / ``drivers.csv`` / ``audit.json`` bundle. Production notebooks do not
call this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyvinecopulib as pv

from design_events.coastal import coastal_realization_metadata
from design_events.catalog import ReferenceBundle, write_reference_bundle
from design_events.runtime import build_paths
from design_events.audit import audit_from_catalog as _audit_from_catalog
from study_location import define_location
from design_events.diagnostics import audit_diagnostics
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
    audit["diagnostics"] = audit_diagnostics(events, drivers, audit)
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


# --------------------------------------------------------------------------------------
# Notebook-facing config policies + runtime + catalog/replay materialization
# (moved from the legacy nested workflow).
# --------------------------------------------------------------------------------------


def configure_coastal_dependence_policy(
    config,
    paths,
    *,
    coastal_latitude: float,
    storm_centroid=None,
    ntr_target_rate_per_year: float = 5.0,
    ntr_declustering_hours: float = 120.0,
    cooccurrence_pairing_window_hours: float = 72.0,
    storm_radius_km: float = 350.0,
    min_population_events: int = 20,
) -> dict:
    """Attach the reusable coastal NTR/rainfall dependence policy to config."""
    event_cfg = config.setdefault("event_catalog", {})
    location_root = Path(paths["location_root"])
    location_name = str(paths.get("location_name") or config["project"]["name"])
    duration_hours = int(config.get("collection", {}).get("aorc_sst", {}).get("storm_duration_hours", 72))
    rainfall_stats = Path(paths["aorc_sst_root"]) / location_name / f"{duration_hours}hr-events" / "storm-stats.csv"

    policy = _deep_merge_dict(
        {
            "method": "copula_joint",
            "driver_vector": ["coastal_water_level", "rainfall"],
            "primary_driver": "coastal_water_level",
            "event_rate_per_year": float(ntr_target_rate_per_year),
            "copula_seed": 0,
            "pool_size": 100000,
            "enforce_stress_budget": True,
            "catalog_band_fractions": {
                "mild": 0.05,
                "common": 0.20,
                "significant": 0.20,
                "rare": 0.25,
                "extreme": 0.30,
            },
            "cooccurrence": {
                "target_rate_per_year": float(ntr_target_rate_per_year),
                "condition_on": ["coastal_water_level", "rainfall"],
                "decluster_window_hours": float(ntr_declustering_hours),
                "pairing_window_hours": float(cooccurrence_pairing_window_hours),
            },
            "storm_stratification": {
                "enabled": True,
                "radius_km": float(storm_radius_km),
                "days_before": 2,
                "days_after": 1,
                "cool_season_months": [10, 11, 12, 1, 2, 3, 4],
                "min_population_events": int(min_population_events),
            },
            "marginals": {"coastal_water_level": {"kind": "pot"}, "rainfall": {"kind": "pot"}},
            "driver_records": {
                "coastal_water_level": {
                    "path": _location_relative_path(paths["waterlevel_csv"], location_root),
                    "time_column": "time",
                    "value_column": "value",
                    "transform": "ntr",
                    "latitude": float(coastal_latitude),
                },
                "rainfall": {
                    "path": _location_relative_path(rainfall_stats, location_root),
                    "time_column": "rainfall_peak_time",
                    "value_column": "mean",
                },
                "soil_moisture": {
                    "path": _location_relative_path(paths["nwm_soil_moisture_csv"], location_root),
                    "time_column": "time",
                    "value_column": "SOILSAT_TOP",
                    "aggregate": "mean",
                },
            },
            "member_libraries": {
                "coastal_water_level": {
                    "from": "records",
                    "index_column": "coastal_peak_m",
                    "decluster_window_hours": float(ntr_declustering_hours),
                    "target_rate_per_year": float(ntr_target_rate_per_year),
                },
                "rainfall": {"from": "member_table"},
            },
        },
        event_cfg.get("dependence", {}) or {},
    )
    policy["event_rate_per_year"] = float(ntr_target_rate_per_year)
    policy["cooccurrence"].update(
        {
            "target_rate_per_year": float(ntr_target_rate_per_year),
            "decluster_window_hours": float(ntr_declustering_hours),
            "pairing_window_hours": float(cooccurrence_pairing_window_hours),
        }
    )
    policy["storm_stratification"].update(
        {"radius_km": float(storm_radius_km), "min_population_events": int(min_population_events)}
    )
    policy["driver_records"]["coastal_water_level"].update(
        {
            "path": _location_relative_path(paths["waterlevel_csv"], location_root),
            "latitude": float(coastal_latitude),
        }
    )
    policy["driver_records"]["rainfall"]["path"] = _location_relative_path(rainfall_stats, location_root)
    policy["driver_records"]["soil_moisture"]["path"] = _location_relative_path(
        paths["nwm_soil_moisture_csv"], location_root
    )
    policy["member_libraries"]["coastal_water_level"].update(
        {
            "decluster_window_hours": float(ntr_declustering_hours),
            "target_rate_per_year": float(ntr_target_rate_per_year),
        }
    )
    if storm_centroid is not None:
        policy["storm_stratification"]["centroid"] = [float(value) for value in storm_centroid]

    event_cfg["dependence"] = policy
    return policy


def configure_coastal_design_event_policy(
    config,
    *,
    target_event_count: int = 500,
    severity_band_fractions: dict | None = None,
    benchmark_return_period_years=(10, 50, 100, 500),
) -> dict:
    """Attach compact coastal design-catalog defaults to config."""
    severity_band_fractions = dict(
        severity_band_fractions
        or {"mild": 0.05, "common": 0.20, "significant": 0.20, "rare": 0.25, "extreme": 0.30}
    )
    severity_bands = [
        {"severity_band": "mild", "rp_min_years": 0.0, "rp_max_years": 2.0},
        {"severity_band": "common", "rp_min_years": 2.0, "rp_max_years": 10.0},
        {"severity_band": "significant", "rp_min_years": 10.0, "rp_max_years": 50.0},
        {"severity_band": "rare", "rp_min_years": 50.0, "rp_max_years": 100.0},
        {"severity_band": "extreme", "rp_min_years": 100.0, "rp_max_years": 500.0},
        {"severity_band": "beyond_design", "rp_min_years": 500.0, "rp_max_years": None},
    ]
    event_cfg = config.setdefault("event_catalog", {})
    dependence = event_cfg.setdefault("dependence", {})
    dependence.update(
        {
            "method": "copula_joint",
            "pool_size": 100000,
            "catalog_band_fractions": severity_band_fractions,
        }
    )
    config["events"] = _deep_merge_dict(config.get("events", {}) or {}, {"target_event_count": int(target_event_count)})
    config["sampling"] = _deep_merge_dict(
        {
            "spacing": "log",
            "return_period_min_years": 1.5,
            "return_period_max_years": 500.0,
            "hybrid_splice_quantile": 0.95,
            "candidate_pool_count": 100000,
            "tail_sample_fraction": 0.05,
            "severity_bands": severity_bands,
        },
        config.get("sampling", {}) or {},
    )
    resilience = _deep_merge_dict(
        {
            "compound_pairing": {
                "enabled": True,
                "strategy": "operationally_severe_plausible_dependence",
                "seed": 0,
                "seasonal_window_days": 45,
                "real_event_count": 12,
                "real_event_window_hours": 72,
                "soil_moisture_lead_time_hours": 24,
                "role_fractions": {
                    "empirical_analog_lag": 1.0,
                },
            }
        },
        config.get("resilience_stress_training", {}) or {},
    )
    resilience.update(
        {
            "target_event_count": int(target_event_count),
            "severity_band_fractions": severity_band_fractions,
            "benchmark_return_period_years": list(benchmark_return_period_years),
        }
    )
    config["resilience_stress_training"] = resilience
    config["design_events"] = _deep_merge_dict(
        {
            "pre_event_baseline_hours": 24,
            "event_threshold_fraction": 0.1,
            "event_threshold_min_m": 0.05,
            "min_event_hours": 12,
            "max_event_hours": 168,
            "tide_resolving_half_window_hours": 72,
            "tail_morph_max_factor": 1.3,
            "tail_morph_trigger_quantile": 0.95,
        },
        config.get("design_events", {}) or {},
    )
    config["template_assignment"] = _deep_merge_dict(
        {
            "random_seed": 0,
            "nearest_pool_size": 75,
            "kernel_sigma_scale": 0.5,
            "kernel_sigma_min_m": 0.03,
            "kernel_sigma_max_m": 0.2,
            "reuse_penalty_lambda": 1.0,
            "dominant_peak_ratio_max": 0.9,
        },
        config.get("template_assignment", {}) or {},
    )
    return resilience


def _deep_merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


# Notebook runtime helpers

@dataclass(frozen=True)
class EventCatalogNotebookRuntime:
    location_root: Path
    location_name: str
    repo_root: Path
    runtime_config: dict
    config: dict
    grid_config: dict
    data_sources: dict
    sfincs_config: dict
    wflow_config: dict
    runtime_paths: dict

    def resolve_location_path(self, value) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.location_root / path

    def ensure_parent(self, value) -> Path:
        path = self.resolve_location_path(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def load_runtime(location_root) -> EventCatalogNotebookRuntime:
    location_root = Path(location_root).resolve()
    repo_root = location_root.parents[1]
    definition = define_location(location_root / "config.yaml")
    runtime_config = definition.config
    runtime_paths = build_paths(runtime_config)
    event_catalog = runtime_config.setdefault("event_catalog", {})
    event_catalog.setdefault("forcing_members", {})
    event_catalog["forcing_members"].setdefault("rainfall", runtime_paths["aorc_sst_rainfall_members_csv"])
    event_catalog["forcing_members"].setdefault("soil_moisture", runtime_paths["nwm_soil_moisture_csv"])
    if runtime_config.get("flood_setting") == "inland":
        event_catalog["forcing_members"].setdefault(
            "streamflow",
            location_root / "data/sources/usgs_streamgages/streamflow_members.csv",
        )
    usgs = runtime_config.get("collection", {}).get("usgs_streamgages")
    if usgs is not None and not isinstance(usgs.get("streamflow_records", {}), dict):
        usgs["streamflow_records"] = {"output": usgs["streamflow_records"]}
    return EventCatalogNotebookRuntime(
        location_root=location_root,
        location_name=location_root.name,
        repo_root=repo_root,
        runtime_config=runtime_config,
        config=runtime_config,
        grid_config=runtime_config,
        data_sources=runtime_config,
        sfincs_config=runtime_config,
        wflow_config={"wflow": runtime_config.get("wflow", {})},
        runtime_paths=runtime_paths,
    )


def event_catalog_source_inventory(runtime: EventCatalogNotebookRuntime) -> pd.DataFrame:
    forcing_members = runtime.data_sources["event_catalog"]["forcing_members"]
    paths = {"rainfall members": forcing_members["rainfall"], "soil moisture": forcing_members["soil_moisture"]}
    usgs = runtime.data_sources.get("collection", {}).get("usgs_streamgages")
    if usgs is not None:
        paths.update(
            {
                "reviewed streamgage network": usgs["reviewed_network"],
                "reviewed discharge records": usgs["streamflow_records"]["output"],
                "streamflow members": forcing_members["streamflow"],
            }
        )
    return pd.DataFrame(
        [
            {"artifact": name, "path": str(path), "exists": path.exists()}
            for name, value in paths.items()
            for path in [runtime.resolve_location_path(value)]
        ]
    )


def scenario_context(runtime: EventCatalogNotebookRuntime) -> dict:
    return {
        "repo_root": runtime.repo_root,
        "location_root": runtime.location_root,
        "location_name": runtime.config["project"]["name"],
        "scenario": {"name": runtime.sfincs_config["scenario_build"]["design_scenario"]},
    }


def materialize_inland_catalog_outputs(
    *,
    runtime: EventCatalogNotebookRuntime,
    event_catalog: pd.DataFrame,
    stress_training_catalog: pd.DataFrame,
    historical_tail_catalog: pd.DataFrame | None = None,
    selected_catalog_csv: Path | None = None,
    summary_fields: dict | None = None,
) -> dict:
    """Write catalog/replay artifacts after visible scientific selection."""
    catalog_root = runtime.ensure_parent("data/event_catalog/catalog/probability_catalog.csv").parent
    paths = {
        "inland_design_event_catalog_csv": selected_catalog_csv or catalog_root / "inland_design_event_catalog.csv",
        "selected_design_catalog_parquet": catalog_root / "probability_catalog.parquet",
        "selected_design_catalog_csv": catalog_root / "probability_catalog.csv",
        "historical_tail_catalog_csv": catalog_root / "historical_tail_catalog.csv",
        "scenario_catalog_csv": catalog_root / "scenario_catalog.csv",
        "wflow_replay_set_parquet": catalog_root / "wflow_replay_set.parquet",
        "wflow_replay_set_csv": catalog_root / "wflow_replay_set.csv",
        "wflow_scenario_replay_set_csv": catalog_root / "wflow_scenario_replay_set.csv",
    }
    if historical_tail_catalog is None:
        try:
            historical_tail_catalog = pd.read_csv(
                paths["historical_tail_catalog_csv"], dtype={"event_id": str}
            )
        except (FileNotFoundError, pd.errors.EmptyDataError):
            # A missing or empty (header-less) tail catalog just means no historical-tail
            # rows were written yet; treat it as empty rather than failing the handoff.
            historical_tail_catalog = pd.DataFrame()

    event_catalog = _location_relative_member_files(event_catalog, runtime.location_root)
    stress_training_catalog = _location_relative_member_files(stress_training_catalog, runtime.location_root)
    historical_tail_catalog = _location_relative_member_files(historical_tail_catalog, runtime.location_root)
    paths["inland_design_event_catalog_csv"].parent.mkdir(parents=True, exist_ok=True)
    event_catalog.to_csv(paths["inland_design_event_catalog_csv"], index=False)
    event_catalog.to_parquet(paths["selected_design_catalog_parquet"], index=False)
    event_catalog.to_csv(paths["selected_design_catalog_csv"], index=False)
    historical_tail_catalog.to_csv(paths["historical_tail_catalog_csv"], index=False)

    replay_columns = _replay_columns(event_catalog)
    wflow_replay_set = event_catalog[replay_columns].copy()
    wflow_replay_set.to_parquet(paths["wflow_replay_set_parquet"], index=False)
    wflow_replay_set.to_csv(paths["wflow_replay_set_csv"], index=False)

    stress_training_path = runtime.runtime_paths["resilience_stress_training_catalog_csv"]
    stress_training_path.parent.mkdir(parents=True, exist_ok=True)
    stress_training_catalog.to_csv(stress_training_path, index=False)

    scenario_catalog = pd.concat([stress_training_catalog, historical_tail_catalog], ignore_index=True, sort=False)
    scenario_catalog.to_csv(paths["scenario_catalog_csv"], index=False)
    scenario_catalog[_replay_columns(scenario_catalog)].copy().to_csv(paths["wflow_scenario_replay_set_csv"], index=False)

    # Reviewer-facing audit summary, emitted alongside the catalog without
    # requiring a notebook edit. Catalog-derived sections (band mass, realization
    # provenance, probability-weight check); model rate read from config when present.
    event_rate = (((runtime.config.get("event_catalog", {}) or {}).get("dependence", {}) or {}).get("event_rate_per_year"))
    audit = _audit_from_catalog(event_catalog, event_rate=(float(event_rate) if event_rate else None), drivers=["rainfall"])
    paths["audit_json"] = catalog_root / "audit.json"
    paths["audit_json"].write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    preview_columns = [
        column
        for column in [
            "event_id",
            "catalog_role",
            "sample_rp_years",
            "severity_band",
            "sampling_weight",
            "probability_weight",
            "rainfall_mm",
            "rainfall_member_id",
            "rainfall_scale_factor",
            "soil_moisture_member_id",
            "forcing_pairing_policy",
            "event_drivers",
            "streamflow_design_role",
        ]
        if column in event_catalog.columns
    ]
    summary = {
        "event_catalog_rows": len(event_catalog),
        "design_driver": "rainfall + antecedent moisture (discharge = Wflow response)",
        "inland_design_event_catalog_csv": str(paths["inland_design_event_catalog_csv"]),
        "probability_catalog_csv": str(paths["selected_design_catalog_csv"]),
        "scenario_catalog_csv": str(paths["scenario_catalog_csv"]),
        "scenario_catalog_rows": len(scenario_catalog),
        "wflow_scenario_replay_set_csv": str(paths["wflow_scenario_replay_set_csv"]),
    }
    if "rainfall_member_id" in event_catalog:
        summary["rainfall_analog_count"] = int(event_catalog["rainfall_member_id"].nunique())
    summary.update(summary_fields or {})

    return {
        **paths,
        "event_catalog": event_catalog,
        "stress_training_catalog_csv": stress_training_path,
        "stress_training_catalog": stress_training_catalog,
        "scenario_catalog": scenario_catalog,
        "wflow_replay_set": wflow_replay_set,
        "preview": event_catalog[preview_columns].head(8).round(
            {
                "sample_rp_years": 2,
                "sampling_weight": 3,
                "probability_weight": 8,
                "rainfall_mm": 1,
                "rainfall_scale_factor": 3,
            }
        ),
        "summary": pd.Series(summary, name="event_catalog_handoff"),
    }


def _replay_columns(catalog: pd.DataFrame) -> list[str]:
    return [
        column
        for column in [
            "event_id",
            "streamflow_member_id",
            "streamflow_member_file",
            "streamflow_member_time",
            "streamflow_scale_factor",
            "rainfall_member_id",
            "rainfall_member_file",
            "rainfall_member_time",
            "rainfall_scale_factor",
            "soil_moisture_member_id",
            "soil_moisture_member_file",
            "soil_moisture_member_time",
            "wflow_event_dir",
        ]
        if column in catalog.columns
    ]


def _location_relative_member_files(catalog: pd.DataFrame, location_root: Path) -> pd.DataFrame:
    frame = catalog.copy()
    for column in [c for c in frame.columns if c.endswith("_member_file")]:
        frame[column] = frame[column].map(lambda value: _location_relative_path(value, location_root))
    if "event_reference_time" in frame:
        frame["event_reference_time"] = pd.to_datetime(frame["event_reference_time"], errors="coerce")
    return frame


def _location_relative_path(value, location_root: Path):
    if value is None or pd.isna(value) or str(value).strip() == "":
        return pd.NA
    path = Path(str(value))
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.relative_to(location_root).as_posix()
    except ValueError:
        return path.as_posix()
