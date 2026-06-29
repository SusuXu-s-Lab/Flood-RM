"""Inland Wflow-coupled reference helpers for ADR-0020.

For inland Wflow-coupled Study Locations, rainfall is the stochastic design driver,
antecedent soil moisture is conditioning, and streamflow is Wflow response or
validation provenance. This module normalizes that role split for v2 reference bundles.
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd


def build_inland_reference_bundle_inputs(config):
    """Return config normalized to the inland rainfall-only stochastic axis."""

    out = deepcopy(dict(config or {}))
    family = str(out.get("event_family", ""))
    if "inland" not in family and not out.get("inland_wflow_coupled", False):
        return out

    dependence = dict(out.get("dependence") or ((out.get("event_catalog") or {}).get("dependence") or {}))
    vector = [driver for driver in list(dependence.get("driver_vector") or []) if driver != "streamflow"]
    if "rainfall" not in vector:
        vector = ["rainfall", *[driver for driver in vector if driver != "rainfall"]]
    dependence["driver_vector"] = vector[:1] if vector and vector[0] == "rainfall" else ["rainfall"]
    out["dependence"] = dependence
    if "event_catalog" in out:
        out.setdefault("event_catalog", {})["dependence"] = dependence

    members = deepcopy(dict(out.get("member_libraries") or {}))
    members.setdefault("rainfall", {})["driver_role"] = "stochastic"
    if "soil_moisture" in members:
        members["soil_moisture"]["driver_role"] = "conditioning"
    if "streamflow" in members:
        members["streamflow"]["driver_role"] = "validation"
    out["member_libraries"] = members
    out["reference_time_policy"] = out.get("reference_time_policy", "rainfall_peak_time")
    return out


def inland_reference_metadata(events, drivers, config):
    """Summarize the inland role split from v2 bundle inputs/tables."""

    cfg = build_inland_reference_bundle_inputs(config)
    dependence = dict(cfg.get("dependence") or {})
    member_roles = {
        name: str(spec.get("driver_role", "stochastic"))
        for name, spec in dict(cfg.get("member_libraries") or {}).items()
    }
    driver_frame = pd.DataFrame(drivers)
    bundle_roles = {}
    if not driver_frame.empty and {"driver", "driver_role"}.issubset(driver_frame.columns):
        bundle_roles = (
            driver_frame.dropna(subset=["driver"])
            .drop_duplicates("driver")
            .set_index("driver")["driver_role"]
            .astype(str)
            .to_dict()
        )
    return {
        "stochastic_drivers": list(dependence.get("driver_vector") or []),
        "conditioning_drivers": sorted([name for name, role in member_roles.items() if role == "conditioning"]),
        "response_or_validation_drivers": sorted(
            [name for name, role in member_roles.items() if role in {"response", "validation"}]
        ),
        "streamflow_role": "response_or_validation" if "streamflow" in member_roles else "not_present",
        "bundle_driver_roles": bundle_roles,
        "event_reference_time_policy": cfg.get("reference_time_policy", "rainfall_peak_time"),
    }


__all__ = ["build_inland_reference_bundle_inputs", "inland_reference_metadata"]
