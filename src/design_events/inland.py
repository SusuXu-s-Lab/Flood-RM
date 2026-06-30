"""Inland Wflow-coupled reference helpers.
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd
import numpy as np
import yaml
import json
from dataclasses import dataclass
from pathlib import Path

from design_events.catalog import attach_forcing_members, validate_event_catalog
from design_events.probability import assign_severity_bands
from design_events.coastal import hybrid_peak_sample_frame
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


@dataclass(frozen=True)
class InlandEventArtifacts:
    catalog: pd.DataFrame
    probability_catalog_parquet: Path
    probability_catalog_csv: Path
    wflow_replay_set_parquet: Path
    wflow_replay_set_csv: Path
    event_manifest_yaml: Path
    audit_json: Path


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


def build_inland_event_artifacts(config, paths) -> InlandEventArtifacts:
    """Build inland event catalog artifacts from member tables.

    This is intentionally a file-producing integration API: notebooks can call
    it once reviewed streamflow, rainfall, and antecedent-state members exist,
    while tests can exercise the same behavior with small local fixtures.
    """
    location_root = _location_root(paths)
    outputs_root = _location_path(
        location_root,
        config.get("paths", {}).get("outputs_root", "data/event_catalog"),
    )
    catalog_root = outputs_root / "catalog"
    catalog_root.mkdir(parents=True, exist_ok=True)

    streamflow_path = _member_path(config, paths, "streamflow")
    streamflow_members = _normalize_streamflow_members(pd.read_csv(streamflow_path, dtype={"site_no": str}), streamflow_path)
    catalog = _base_inland_catalog(config, paths, streamflow_members)

    for forcing in ("rainfall", "soil_moisture"):
        member_value = config.get("event_catalog", {}).get("forcing_members", {}).get(forcing)
        if not member_value:
            continue
        member_path = _member_path(config, paths, forcing)
        members = _normalize_member_table(pd.read_csv(member_path), forcing, member_path)
        policy = config.get("event_catalog", {}).get("pairing", {}).get(forcing, {})
        policy = _inland_pairing_policy(forcing, policy)
        catalog = attach_forcing_members(catalog, members, forcing, policy)

    catalog = _with_missing_columns(catalog, inland_catalog_columns)
    catalog = catalog[inland_catalog_columns]

    probability_catalog_parquet = catalog_root / "probability_catalog.parquet"
    probability_catalog_csv = catalog_root / "probability_catalog.csv"
    wflow_replay_set_parquet = catalog_root / "wflow_replay_set.parquet"
    wflow_replay_set_csv = catalog_root / "wflow_replay_set.csv"
    event_manifest_yaml = outputs_root / "event_manifest.yaml"
    audit_json = catalog_root / "event_catalog_audit.json"

    replay = catalog[
        [
            "event_id",
            "event_reference_time",
            "basis_site_no",
            "streamflow_member_id",
            "streamflow_member_file",
            "streamflow_member_time",
            "rainfall_member_id",
            "rainfall_member_file",
            "rainfall_member_time",
            "soil_moisture_member_id",
            "soil_moisture_member_file",
            "soil_moisture_member_time",
            "wflow_event_dir",
        ]
    ].copy()

    catalog.to_parquet(probability_catalog_parquet, index=False)
    catalog.to_csv(probability_catalog_csv, index=False)
    replay.to_parquet(wflow_replay_set_parquet, index=False)
    replay.to_csv(wflow_replay_set_csv, index=False)

    audit = _write_inland_audit(catalog, audit_json)
    _write_event_manifest(
        catalog,
        {
            "probability_catalog_parquet": probability_catalog_parquet,
            "probability_catalog_csv": probability_catalog_csv,
            "wflow_replay_set_parquet": wflow_replay_set_parquet,
            "wflow_replay_set_csv": wflow_replay_set_csv,
            "audit_json": audit_json,
        },
        config,
        paths,
        event_manifest_yaml,
        audit,
    )
    return InlandEventArtifacts(
        catalog=catalog,
        probability_catalog_parquet=probability_catalog_parquet,
        probability_catalog_csv=probability_catalog_csv,
        wflow_replay_set_parquet=wflow_replay_set_parquet,
        wflow_replay_set_csv=wflow_replay_set_csv,
        event_manifest_yaml=event_manifest_yaml,
        audit_json=audit_json,
    )


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


def _base_inland_catalog(config, paths, streamflow_members):
    streamflow_members = _design_streamflow_members(config, streamflow_members)
    study_location = str(paths.get("location_name") or config.get("project", {}).get("name"))
    scenario_name = str(paths.get("scenario", {}).get("name", "base"))
    sample_rp_years = pd.to_numeric(streamflow_members["sample_rp_years"], errors="coerce")
    event_times = pd.to_datetime(streamflow_members["event_time"], errors="coerce")
    if event_times.isna().any():
        raise ValueError("streamflow members require parseable event_time values")
    catalog = pd.DataFrame(
        {
            "event_id": streamflow_members["event_id"].astype(str),
            "study_location": study_location,
            "event_family": "streamgage_network",
            "scenario_name": scenario_name,
            "sample_rp_years": sample_rp_years,
            "severity_band": _severity(config, sample_rp_years),
            "sampling_region": streamflow_members.get("sampling_region", "body"),
            "sampling_weight": pd.to_numeric(streamflow_members.get("sampling_weight", 1.0), errors="coerce"),
            "probability_weight": pd.to_numeric(streamflow_members.get("probability_weight", 1.0), errors="coerce"),
            "event_time": event_times.dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "event_reference_time": event_times.dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "basis_site_no": streamflow_members["site_no"].astype(str),
            "peak_flow_cfs": pd.to_numeric(streamflow_members.get("peak_flow_cfs", pd.NA), errors="coerce"),
            "streamflow_template_event_id": streamflow_members["streamflow_template_event_id"].astype(str),
            "streamflow_template_member_id": streamflow_members["streamflow_template_member_id"].astype(str),
            "streamflow_template_time": streamflow_members["streamflow_template_time"].astype(str),
            "streamflow_template_peak_flow_cfs": pd.to_numeric(
                streamflow_members["streamflow_template_peak_flow_cfs"],
                errors="coerce",
            ),
            "streamflow_scale_factor": pd.to_numeric(streamflow_members["streamflow_scale_factor"], errors="coerce"),
            "streamflow_design_method": streamflow_members["streamflow_design_method"].astype(str),
            "streamflow_source": streamflow_members.get("source", "usgs"),
            "streamflow_member_file": streamflow_members["member_file"].astype(str),
            "streamflow_member_id": streamflow_members["member_id"].astype(str),
            "streamflow_member_time": event_times.dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "streamflow_pairing_policy": "coherent_streamgage_network_event",
            "streamflow_pairing_seed": 0,
            "streamflow_pairing_window_days": pd.NA,
            "streamflow_pairing_reference_time": event_times.dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "streamflow_pairing_lag_hours": 0,
            "infiltration_treatment": config.get("inland_coupling", {})
            .get("infiltration", {})
            .get("method", config.get("infiltration", {}).get("treatment", "none")),
        }
    )
    events_root = config.get("wflow", {}).get("events_root", "data/wflow/events")
    scenarios_root = config.get("paths", {}).get("sfincs_scenarios_root", "data/sfincs/scenarios")
    catalog["wflow_event_dir"] = catalog["event_id"].map(lambda event_id: f"{events_root.rstrip('/')}/{event_id}")
    catalog["sfincs_scenario_dir"] = catalog["event_id"].map(lambda event_id: f"{scenarios_root.rstrip('/')}/{event_id}")
    return catalog


def _design_streamflow_members(config, streamflow_members):
    members = streamflow_members.copy().reset_index(drop=True)
    members["event_time"] = pd.to_datetime(members["event_time"], errors="coerce")
    members["peak_flow_cfs"] = pd.to_numeric(members.get("peak_flow_cfs"), errors="coerce")
    members["sample_rp_years"] = pd.to_numeric(members["sample_rp_years"], errors="coerce")
    members = members.dropna(subset=["event_time", "peak_flow_cfs", "sample_rp_years"]).copy()
    if members.empty:
        raise ValueError("streamflow members contain no valid design-template events")

    target_count = int(config.get("events", {}).get("target_event_count", len(members)))
    if target_count <= len(members):
        out = members.copy()
        out["streamflow_template_event_id"] = out["event_id"].astype(str)
        out["streamflow_template_member_id"] = out["member_id"].astype(str)
        out["streamflow_template_time"] = out["event_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        out["streamflow_template_peak_flow_cfs"] = out["peak_flow_cfs"]
        out["streamflow_scale_factor"] = 1.0
        out["streamflow_design_method"] = "historical_streamgage_network_event"
        out["event_time"] = out["event_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        return out

    marginal = _StreamflowPowerLawReturnCurve(members["sample_rp_years"], members["peak_flow_cfs"])
    seed = int(config.get("template_assignment", {}).get("random_seed", 42))
    sample = hybrid_peak_sample_frame(
        members["peak_flow_cfs"].to_numpy(dtype=float),
        target_count,
        config.get("sampling", {}),
        marginal,
        seed,
    )
    design_peak = pd.to_numeric(sample["peak_m"], errors="coerce").to_numpy(dtype=float)
    template_indices = _nearest_streamflow_templates(members, design_peak)
    templates = members.iloc[template_indices].reset_index(drop=True).copy()
    out = templates.copy()
    out["event_id"] = [f"usgs_design_{index:04d}" for index in range(1, target_count + 1)]
    out["peak_flow_cfs"] = design_peak
    out["sample_rp_years"] = marginal.return_period(design_peak)
    out["sampling_region"] = sample["sampling_region"].astype(str).to_numpy()
    out["sampling_weight"] = pd.to_numeric(sample["sampling_weight"], errors="coerce").to_numpy(dtype=float)
    probability_weight = pd.to_numeric(sample["probability_weight"], errors="coerce").fillna(0.0)
    total_probability = float(probability_weight.sum())
    if not np.isfinite(total_probability) or total_probability <= 0.0:
        probability_weight = pd.Series(np.full(target_count, 1.0 / target_count))
    else:
        probability_weight = probability_weight / total_probability
    out["probability_weight"] = probability_weight.to_numpy(dtype=float)
    template_peak = pd.to_numeric(templates["peak_flow_cfs"], errors="coerce").to_numpy(dtype=float)
    out["streamflow_template_event_id"] = templates["event_id"].astype(str).to_numpy()
    out["streamflow_template_member_id"] = templates["member_id"].astype(str).to_numpy()
    out["streamflow_template_time"] = pd.to_datetime(templates["event_time"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["streamflow_template_peak_flow_cfs"] = template_peak
    out["streamflow_scale_factor"] = np.divide(
        design_peak,
        template_peak,
        out=np.ones_like(design_peak, dtype=float),
        where=np.isfinite(template_peak) & (template_peak > 0),
    )
    out["streamflow_design_method"] = "scaled_streamgage_network_analog"
    out["event_time"] = pd.to_datetime(templates["event_time"], errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S")
    return out


def _nearest_streamflow_templates(members, design_peak):
    historical_peak = pd.to_numeric(members["peak_flow_cfs"], errors="coerce").to_numpy(dtype=float)
    historical_peak = np.where(np.isfinite(historical_peak) & (historical_peak > 0), historical_peak, np.nan)
    if np.isnan(historical_peak).all():
        return np.zeros(len(design_peak), dtype=int)
    log_historical = np.log(historical_peak)
    indices = []
    for peak in design_peak:
        if not np.isfinite(peak) or peak <= 0:
            indices.append(int(np.nanargmax(historical_peak)))
            continue
        distances = np.abs(log_historical - np.log(float(peak)))
        indices.append(int(np.nanargmin(distances)))
    return np.asarray(indices, dtype=int)


class _StreamflowPowerLawReturnCurve:
    def __init__(self, return_period_years, peak_flow_cfs):
        rp = pd.to_numeric(pd.Series(return_period_years), errors="coerce").to_numpy(dtype=float)
        peak = pd.to_numeric(pd.Series(peak_flow_cfs), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(rp) & np.isfinite(peak) & (rp > 0) & (peak > 0)
        if valid.sum() < 2:
            self.intercept = float(np.log(np.nanmax(peak[valid]) if valid.any() else 1.0))
            self.slope = 0.25
            return
        log_rp = np.log(rp[valid])
        log_peak = np.log(peak[valid])
        slope, intercept = np.polyfit(log_rp, log_peak, 1)
        self.slope = float(max(slope, 0.05))
        self.intercept = float(intercept)

    def magnitude(self, return_period_years):
        rp = np.asarray(return_period_years, dtype=float)
        value = np.exp(self.intercept + self.slope * np.log(np.maximum(rp, 1e-6)))
        return _maybe_scalar(value, return_period_years)

    def return_period(self, peak_flow_cfs):
        peak = np.asarray(peak_flow_cfs, dtype=float)
        value = np.exp((np.log(np.maximum(peak, 1e-6)) - self.intercept) / self.slope)
        return _maybe_scalar(value, peak_flow_cfs)


def _maybe_scalar(value, original):
    if np.ndim(original) == 0:
        return float(np.asarray(value))
    return value


def _normalize_streamflow_members(members, member_path):
    frame = members.copy()
    required = {"event_id", "site_no", "event_time", "sample_rp_years"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("streamflow members are missing required columns: " + ", ".join(sorted(missing)))
    if "source" not in frame:
        frame["source"] = "usgs"
    frame["member_file"] = str(member_path)
    if "member_id" not in frame:
        times = pd.to_datetime(frame["event_time"], errors="coerce").dt.strftime("%Y%m%dT%H%M%S")
        frame["member_id"] = frame["site_no"].astype(str) + "_" + times.fillna("unknown")
    return frame


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


def _inland_pairing_policy(forcing, policy):
    out = dict(policy or {})
    strategy = out.get("strategy")
    if forcing == "rainfall" and strategy == "inland_rainfall_pairing_priority":
        out["strategy"] = "seasonal_window_permutation"
        if out.get("fallback_strategy") == "seasonal_window_permutation":
            out.pop("fallback_strategy")
        out.setdefault("event_time_column", "event_reference_time")
    elif forcing == "soil_moisture" and strategy == "inland_antecedent_moisture_pairing":
        out["strategy"] = "antecedent_to_forcing"
        if out.get("rainfall_relative_when_coherent", True):
            out.setdefault("reference_forcing", "rainfall")
        else:
            out.setdefault("reference_time_column", "event_reference_time")
    return out


def _write_inland_audit(catalog, path):
    validation_catalog = catalog.rename(columns={"event_reference_time": "event_time"}).copy()
    issues = validate_event_catalog(
        validation_catalog,
        required_forcings=("streamflow", "rainfall", "soil_moisture"),
        wave_analog_policy="not_required",
    )
    audit = {
        "passed": len(issues) == 0,
        "event_count": int(len(catalog)),
        "issue_count": int(len(issues)),
        "issues": issues,
    }
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def _write_event_manifest(catalog, artifacts, config, paths, output_path, audit):
    manifest = {
        "study_location": str(paths.get("location_name") or config.get("project", {}).get("name")),
        "scenario_name": str(paths.get("scenario", {}).get("name", "base")),
        "event_family": "streamgage_network",
        "event_count": int(len(catalog)),
        "audit_passed": bool(audit["passed"]),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }
    output_path = Path(output_path)
    output_path.write_text(
        _GENERATED_NOTICE.format(source="the inland event-catalog build")
        + yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )


def _with_missing_columns(frame, columns):
    out = frame.copy()
    for column in columns:
        if column not in out:
            out[column] = pd.NA
    return out


def _severity(config, sample_rp_years):
    bands = config.get("sampling", {}).get("severity_bands")
    if not bands:
        return pd.Series([pd.NA] * len(sample_rp_years))
    return assign_severity_bands(sample_rp_years, bands)


def _member_path(config, paths, forcing):
    value = config.get("event_catalog", {}).get("forcing_members", {}).get(forcing)
    if not value:
        raise ValueError(f"{forcing} forcing member path is not configured")
    return _location_path(_location_root(paths), value)


def _location_root(paths):
    return location_root_from_paths(paths)


def _location_path(location_root, value):
    return resolve_location_path(location_root, value)
