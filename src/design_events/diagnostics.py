"""Diagnostics for ADR-0020 reference bundle tables.

Diagnostics consume ``events``, ``drivers``, and ``audit`` only. They do not rebuild
catalogs, read source data, or stage hydrodynamic forcing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def severity_distribution(events):
    """Probability and count summary by Event Catalog severity band."""

    frame = pd.DataFrame(events).copy()
    if frame.empty or "severity_band" not in frame:
        return {}
    frame["probability_weight"] = pd.to_numeric(frame.get("probability_weight", 0.0), errors="coerce").fillna(0.0)
    frame["sampling_weight"] = pd.to_numeric(frame.get("sampling_weight", 0.0), errors="coerce").fillna(0.0)
    grouped = frame.groupby("severity_band", dropna=False).agg(
        count=("event_id", "count"),
        probability_weight=("probability_weight", "sum"),
        mean_sampling_weight=("sampling_weight", "mean"),
    )
    return {
        str(index): {
            "count": int(row["count"]),
            "probability_weight": float(row["probability_weight"]),
            "mean_sampling_weight": float(row["mean_sampling_weight"]),
        }
        for index, row in grouped.iterrows()
    }


def probability_weight_check(events):
    """Check that ``probability_weight`` is finite, nonnegative, and sums to one."""

    frame = pd.DataFrame(events)
    weight = pd.to_numeric(frame.get("probability_weight", pd.Series(dtype=float)), errors="coerce")
    return {
        "sum": float(weight.sum()),
        "nonnegative": bool((weight.dropna() >= 0).all()),
        "finite_count": int(np.isfinite(weight.to_numpy(dtype=float, na_value=np.nan)).sum()),
        "row_count": int(len(frame)),
    }


def realization_reuse(drivers):
    """Member-reuse summary by driver."""

    frame = pd.DataFrame(drivers)
    if frame.empty or not {"driver", "member_id"}.issubset(frame.columns):
        return {}
    out = {}
    for driver, group in frame.groupby("driver"):
        counts = group["member_id"].value_counts(normalize=True)
        out[str(driver)] = {
            "rows": int(len(group)),
            "unique_members": int(group["member_id"].nunique()),
            "max_member_reuse_fraction": float(counts.iloc[0]) if not counts.empty else 0.0,
        }
    return out


def scale_factor_quantiles(drivers):
    """Scale-factor quantiles by driver."""

    frame = pd.DataFrame(drivers)
    if frame.empty or not {"driver", "scale_factor"}.issubset(frame.columns):
        return {}
    out = {}
    for driver, group in frame.groupby("driver"):
        scale = pd.to_numeric(group["scale_factor"], errors="coerce")
        out[str(driver)] = {
            "q05": float(scale.quantile(0.05)) if scale.notna().any() else None,
            "q50": float(scale.quantile(0.50)) if scale.notna().any() else None,
            "q95": float(scale.quantile(0.95)) if scale.notna().any() else None,
        }
    return out


def timing_summary(events, drivers):
    """Timing-field coverage summary."""

    event_frame = pd.DataFrame(events)
    driver_frame = pd.DataFrame(drivers)
    summary = {
        "events_with_reference_time": int(event_frame.get("event_reference_time", pd.Series(dtype=object)).notna().sum()),
        "event_rows": int(len(event_frame)),
    }
    if "lag_hours" in driver_frame:
        lag = pd.to_numeric(driver_frame["lag_hours"], errors="coerce")
        summary["driver_lag_hours_min"] = float(lag.min()) if lag.notna().any() else None
        summary["driver_lag_hours_max"] = float(lag.max()) if lag.notna().any() else None
    if "storm_loading_pattern" in event_frame:
        summary["storm_loading_pattern_counts"] = event_frame["storm_loading_pattern"].value_counts(dropna=False).to_dict()
    return summary


def audit_diagnostics(events, drivers, audit):
    """Build audit-ready diagnostics from v2 bundle tables only."""

    return {
        "severity_distribution": severity_distribution(events),
        "probability_weight_check": probability_weight_check(events),
        "realization_reuse": realization_reuse(drivers),
        "scale_factor_quantiles": scale_factor_quantiles(drivers),
        "timing_summary": timing_summary(events, drivers),
    }


__all__ = [
    "audit_diagnostics",
    "probability_weight_check",
    "realization_reuse",
    "scale_factor_quantiles",
    "severity_distribution",
    "timing_summary",
]
