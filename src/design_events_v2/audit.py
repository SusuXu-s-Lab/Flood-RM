"""Stakeholder-readable audit derived from a (wide) Event Catalog.

The reviewer-facing summary of one catalog build: the AND/return-period and weight
formulas, band true-mass vs design-fraction (the Tail-Enriched importance picture), each
driver's Field-Preserving Realization provenance (unique members, scale-factor spread, max
reuse), and the probability-weight check. Computed from the catalog DataFrame alone so it
can be written as ``audit.json`` without the internal fit objects (ADR-0021).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _drivers_from_columns(catalog: pd.DataFrame) -> list[str]:
    return sorted(c[: -len("_member_id")] for c in catalog.columns if c.endswith("_member_id"))


def audit_from_catalog(catalog: pd.DataFrame, *, event_rate=None, drivers=None) -> dict:
    """Build the audit dict from a wide catalog.

    ``drivers`` defaults to those inferred from ``<driver>_member_id`` columns.
    """
    drivers = list(drivers) if drivers is not None else _drivers_from_columns(catalog)

    band_true_mass = {}
    if "severity_band" in catalog and "probability_weight" in catalog:
        band_true_mass = {
            str(k): float(v)
            for k, v in catalog.groupby("severity_band")["probability_weight"].sum().items()
        }
    band_design_fraction = (
        {str(k): float(v) for k, v in catalog["severity_band"].value_counts(normalize=True).items()}
        if "severity_band" in catalog
        else {}
    )

    unique_members, scale_quantiles, max_reuse = {}, {}, {}
    for driver in drivers:
        id_col, scale_col = f"{driver}_member_id", f"{driver}_scale_factor"
        if id_col in catalog:
            ids = catalog[id_col].astype(str)
            unique_members[driver] = int(ids.nunique())
            counts = ids.value_counts(normalize=True)
            max_reuse[driver] = float(counts.iloc[0]) if not counts.empty else 0.0
        if scale_col in catalog:
            scale = pd.to_numeric(catalog[scale_col], errors="coerce")
            scale_quantiles[driver] = {
                "q05": float(scale.quantile(0.05)),
                "q50": float(scale.quantile(0.50)),
                "q95": float(scale.quantile(0.95)),
            }

    checks = []
    if "probability_weight" in catalog:
        checks.append(
            {
                "name": "probability_weight_sum",
                "value": float(pd.to_numeric(catalog["probability_weight"], errors="coerce").sum()),
                "expected": 1.0,
            }
        )

    return {
        "model": {
            "event_rate_per_year": (float(event_rate) if event_rate is not None else None),
            "drivers": drivers,
            "joint_probability": "AND",
            "return_period_formula": "T = 1 / (lambda * S_and(F(x)))",
            "sampling_weight_formula": "w_b = p_b/q_b",
            "probability_weight_formula": "pi_i = p_b/n_b",
        },
        "sampling": {
            "catalog_count": int(len(catalog)),
            "band_true_mass": band_true_mass,
            "band_design_fraction": band_design_fraction,
        },
        "realization": {
            "unique_members_by_driver": unique_members,
            "scale_factor_quantiles_by_driver": scale_quantiles,
            "max_member_reuse_fraction_by_driver": max_reuse,
        },
        "checks": checks,
    }


__all__ = ["audit_from_catalog"]
