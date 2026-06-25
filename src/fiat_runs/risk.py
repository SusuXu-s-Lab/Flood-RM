"""Weighted-event Expected Annual Damage (EAD) integration (main env, pure pandas).

The Marshfield catalog is a band-stratified importance sample of compound coastal events.
Each synthetic event carries a ``probability_weight`` that de-biases the importance
sampling back to the true distribution and sums to 1 over the synthetic catalog (the
``historical_tail`` rows carry no weight and are validation-only). The catalog's copula
mixture occurrence rate ``total_rate_per_year`` (persisted by ``03`` to
``catalog_risk_metadata.json``) converts those conditional weights to annual rates:

    annual_rate_i = total_rate_per_year * probability_weight_i
    EAD           = sum_i annual_rate_i * loss_i = total_rate_per_year * sum_i w_i * loss_i

This is the standard event-based (catastrophe-model) risk integral. Because the events are
*compound* (no single 1-D return period), the probability integration is done here rather
than via Delft-FIAT's native return-period integrator (that path is the separate
cross-check in :mod:`fiat_runs.risk_native`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EAD_ORIGINS = ("synthetic_body", "synthetic_tail")


def load_catalog_weights(catalog_csv) -> pd.DataFrame:
    """Synthetic-event weights for EAD: event_id, probability_weight, severity, joint RP."""
    cat = pd.read_csv(catalog_csv)
    cols = ["event_id", "probability_weight", "event_origin", "severity_band", "sample_rp_years"]
    cat = cat[[c for c in cols if c in cat.columns]].copy()
    cat = cat[cat["event_origin"].isin(EAD_ORIGINS)]
    cat["probability_weight"] = pd.to_numeric(cat["probability_weight"], errors="coerce")
    return cat[cat["probability_weight"].notna()].reset_index(drop=True)


def total_rate_from_metadata(metadata_json) -> float:
    meta = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
    return float(meta["total_rate_per_year"])


def _join(damage_df: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    df = damage_df.merge(weights, on="event_id", how="inner")
    df["total_damage"] = pd.to_numeric(df["total_damage"], errors="coerce").fillna(0.0)
    return df


def weighted_ead(damage_df: pd.DataFrame, weights: pd.DataFrame, total_rate: float) -> dict:
    """EAD for a single set of per-event damages. Returns EAD + audit components."""
    df = _join(damage_df, weights)
    weight_coverage = float(df["probability_weight"].sum())
    expected_event_damage = float((df["probability_weight"] * df["total_damage"]).sum())
    return {
        "n_events": int(len(df)),
        "weight_coverage": weight_coverage,  # ~1.0 when every synthetic event is present
        "total_rate_per_year": float(total_rate),
        "expected_event_damage": expected_event_damage,  # E[loss | an event occurs]
        "ead": float(total_rate) * expected_event_damage,
    }


def ead_by_scenario(damage_df: pd.DataFrame, weights: pd.DataFrame, total_rate: float) -> pd.DataFrame:
    """Per-SLR-scenario EAD table. ``damage_df`` needs columns event_id, design_scenario, total_damage."""
    rows = []
    for scenario, sub in damage_df.groupby("design_scenario"):
        rows.append({"design_scenario": scenario, **weighted_ead(sub, weights, total_rate)})
    out = pd.DataFrame(rows).sort_values("ead").reset_index(drop=True)
    base = out.loc[out["design_scenario"] == "base", "ead"]
    if len(base):
        out["ead_delta_vs_base"] = out["ead"] - float(base.iloc[0])
    return out


def exceedance(damage_df: pd.DataFrame, weights: pd.DataFrame, total_rate: float) -> pd.DataFrame:
    """Loss vs annual exceedance for one scenario: the curve whose area under it is the EAD."""
    df = _join(damage_df, weights).sort_values("total_damage", ascending=False).reset_index(drop=True)
    df["annual_rate"] = total_rate * df["probability_weight"]
    df["exceedance_rate_per_year"] = df["annual_rate"].cumsum()
    df["return_period_years"] = 1.0 / df["exceedance_rate_per_year"].clip(lower=1e-12)
    return df[["event_id", "severity_band", "sample_rp_years", "total_damage",
               "annual_rate", "exceedance_rate_per_year", "return_period_years"]]


def ead_audit(damage_df: pd.DataFrame, weights: pd.DataFrame, total_rate: float, *, expected_event_count: int = 500) -> dict:
    """Auditable receipt: weight closure, rate, coverage, and per-scenario EAD."""
    present = set(damage_df["event_id"]) & set(weights["event_id"])
    table = ead_by_scenario(damage_df, weights, total_rate)
    return {
        "total_rate_per_year": float(total_rate),
        "synthetic_weight_sum": float(weights["probability_weight"].sum()),  # must be ~1.0
        "synthetic_event_count": int(len(weights)),
        "expected_event_count": int(expected_event_count),
        "events_with_damage_and_weight": int(len(present)),
        "ead_origins": list(EAD_ORIGINS),
        "ead_by_scenario": table.to_dict("records"),
    }
