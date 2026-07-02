"""Inland Wflow-coupled reference helpers.
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import yaml

from paths import location_root_from_paths, resolve_location_path


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


__all__ = [
    "build_inland_reference_bundle_inputs",
    "inland_reference_metadata",
]


# --------------------------------------------------------------------------------------
# External-boundary fluvial streamgage path + Wflow handoff manifest
# (relocated out of legacy build_events.inland). Not a copula dup — the
# streamgage-network design path is retained as the external-boundary realization.
# --------------------------------------------------------------------------------------


_GENERATED_NOTICE = (
    "# GENERATED FILE — do not edit. Overwritten when {source} runs.\n"
    "# Source of truth is the location config and the code that produces this file.\n"
)


inland_catalog_columns = [
    "event_id",
    "study_location",
    "event_family",
    "scenario_name",
    "sample_rp_years",
    "severity_band",
    "sampling_region",
    "sampling_weight",
    "probability_weight",
    "event_reference_time",
    "basis_site_no",
    "peak_flow_cfs",
    "streamflow_template_event_id",
    "streamflow_template_member_id",
    "streamflow_template_time",
    "streamflow_template_peak_flow_cfs",
    "streamflow_scale_factor",
    "streamflow_design_method",
    "streamflow_source",
    "streamflow_member_file",
    "streamflow_member_id",
    "streamflow_member_time",
    "streamflow_pairing_policy",
    "streamflow_pairing_seed",
    "streamflow_pairing_window_days",
    "streamflow_pairing_reference_time",
    "streamflow_pairing_lag_hours",
    "rainfall_source",
    "rainfall_member_file",
    "rainfall_member_id",
    "rainfall_member_time",
    "rainfall_pairing_policy",
    "rainfall_pairing_seed",
    "rainfall_pairing_window_days",
    "rainfall_pairing_reference_time",
    "rainfall_pairing_lag_hours",
    "soil_moisture_source",
    "soil_moisture_member_file",
    "soil_moisture_member_id",
    "soil_moisture_member_time",
    "soil_moisture_pairing_policy",
    "soil_moisture_pairing_seed",
    "soil_moisture_pairing_window_days",
    "soil_moisture_pairing_reference_time",
    "soil_moisture_pairing_lag_hours",
    "infiltration_treatment",
    "wflow_event_dir",
    "sfincs_scenario_dir",
]


def write_handoff(catalog, config, paths):
    location_root = _location_root(paths)
    handoff = config.get("wflow", {}).get("handoff", {})
    manifest_path = _location_path(location_root, handoff.get("manifest", "data/wflow/domain_set_handoff.yaml"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    events = []
    for _, row in catalog.iterrows():
        event_id = str(row["event_id"])
        wflow_event_dir = str(row.get("wflow_event_dir") or f"data/wflow/events/{event_id}")
        events.append(
            {
                "event_id": event_id,
                "wflow_event_dir": wflow_event_dir,
                "discharge_forcing": f"{wflow_event_dir.rstrip('/')}/sfincs_discharge.nc",
            }
        )

    manifest = {
        "forcing_mode": config.get("inland_coupling", {}).get("forcing_mode", "dual_fluvial_pluvial"),
        "event_catalog_scope": config.get("wflow", {})
        .get("domain_set", {})
        .get("event_catalog_scope", "shared_across_domain_set"),
        "source_variable": handoff.get("source_variable", "river_q"),
        "source_standard_name": handoff.get(
            "source_standard_name",
            "river_water__volume_flow_rate",
        ),
        "target": handoff.get("target", "sfincs_discharge_forcing"),
        "direct_rainfall_enabled": bool(config.get("inland_coupling", {}).get("direct_rainfall", {}).get("enabled", True)),
        "submodels": list(config.get("wflow", {}).get("domain_set", {}).get("submodels", [])),
        "sfincs_domains": list(config.get("sfincs_domain_set", {}).get("domains", [])),
        "sfincs_evaluation_merge": config.get("sfincs_domain_set", {}).get(
            "evaluation_merge",
            "max_depth_per_asset_with_source_domain",
        ),
        "events": events,
    }
    manifest_path.write_text(
        _GENERATED_NOTICE.format(source="the inland event-catalog build")
        + yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path


def _normalize_member_table(members, forcing, member_path):
    frame = members.copy()
    if "source" not in frame:
        frame["source"] = forcing
    if "member_file" not in frame:
        frame["member_file"] = str(member_path)
    if "member_id" not in frame:
        if "time" in frame:
            times = pd.to_datetime(frame["time"], errors="coerce").dt.strftime("%Y%m%dT%H%M%S")
            frame["member_id"] = forcing + "_" + times.fillna("unknown")
        else:
            frame["member_id"] = [f"{forcing}_{index:04d}" for index in range(len(frame))]
    return frame


def _location_root(paths):
    return location_root_from_paths(paths)


def _location_path(location_root, value):
    return resolve_location_path(location_root, value)
