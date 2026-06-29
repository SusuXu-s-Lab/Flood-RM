"""Canonical reviewer-facing Event Catalog bundle for ADR-0020.

The v2 reference emits two long tables plus one audit JSON:

* ``events.csv``: one row per Event Catalog row with probability labels and weights.
* ``drivers.csv``: one row per event-driver Field-Preserving Realization.
* ``audit.json``: formulas, source/config provenance, and summary checks.

This module owns the bundle Interface only. Probability fitting and realization stay in
``probability.py``, ``records.py``, and ``realization.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from design_events.runtime import plan
from design_events.probability import assign_severity_bands


EVENT_COLUMNS = [
    "event_id",
    "event_role",
    "event_origin",
    "event_family",
    "scenario_name",
    "sample_rp_years",
    "and_joint_exceedance_prob",
    "and_joint_aep",
    "severity_band",
    "sampling_region",
    "sampling_weight",
    "probability_weight",
    "event_reference_time",
    "selection_reason",
]


DRIVER_COLUMNS = [
    "event_id",
    "driver",
    "x",
    "u",
    "driver_role",
    "member_id",
    "member_file",
    "member_time",
    "template_value",
    "scale_factor",
    "lag_hours",
    "time_policy",
    "realization_policy",
    "source",
]


@dataclass(frozen=True)
class ReferenceBundle:
    """In-memory and on-disk handles for the ADR-0020 reference bundle."""

    events: pd.DataFrame
    drivers: pd.DataFrame
    audit: dict[str, Any]
    events_path: Path
    drivers_path: Path
    audit_path: Path


def normalize_events(events: pd.DataFrame) -> pd.DataFrame:
    """Return ``events`` with the canonical event columns first."""

    frame = events.copy()
    for column in EVENT_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[EVENT_COLUMNS + [c for c in frame.columns if c not in EVENT_COLUMNS]]


def normalize_drivers(drivers: pd.DataFrame) -> pd.DataFrame:
    """Return ``drivers`` with the canonical long driver columns first."""

    frame = drivers.copy()
    for column in DRIVER_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[DRIVER_COLUMNS + [c for c in frame.columns if c not in DRIVER_COLUMNS]]


def write_reference_bundle(
    events: pd.DataFrame,
    drivers: pd.DataFrame,
    audit: dict[str, Any],
    output_dir,
) -> ReferenceBundle:
    """Write ``events.csv``, ``drivers.csv``, and ``audit.json`` under ``output_dir``."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    events_out = normalize_events(events)
    drivers_out = normalize_drivers(drivers)
    events_path = root / "events.csv"
    drivers_path = root / "drivers.csv"
    audit_path = root / "audit.json"

    events_out.to_csv(events_path, index=False)
    drivers_out.to_csv(drivers_path, index=False)
    audit_path.write_text(json.dumps(_jsonable(audit), indent=2, sort_keys=True), encoding="utf-8")

    return ReferenceBundle(
        events=events_out,
        drivers=drivers_out,
        audit=_jsonable(audit),
        events_path=events_path,
        drivers_path=drivers_path,
        audit_path=audit_path,
    )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


__all__ = [
    "EVENT_COLUMNS",
    "DRIVER_COLUMNS",
    "ReferenceBundle",
    "normalize_events",
    "normalize_drivers",
    "write_reference_bundle",
]


# --------------------------------------------------------------------------------------
# Production wide Event Catalog assembly + forcing pairing + validation
# (moved from the legacy nested catalog builders; ADR-0021).
# --------------------------------------------------------------------------------------


catalog_columns = [
    "event_id",
    "study_location",
    "event_family",
    "scenario_name",
    "sample_rp_years",
    "severity_band",
    "sampling_region",
    "sampling_weight",
    "probability_weight",
    "coastal_source",
    "coastal_member_file",
    "coastal_member_id",
    "coastal_template_peak_time",
    "coastal_peak_m",
    "coastal_absolute_peak_m",
    "coastal_valid_start_hour",
    "coastal_valid_end_hour",
    "coastal_analog_id",
    "coastal_analog_peak_time",
    "snapwave_source",
    "snapwave_member_file",
    "snapwave_member_id",
    "snapwave_valid_start_time",
    "snapwave_valid_end_time",
    "snapwave_pairing_policy",
    "rainfall_source",
    "rainfall_member_file",
    "rainfall_member_id",
    "rainfall_member_time",
    "rainfall_pairing_policy",
    "rainfall_pairing_seed",
    "rainfall_pairing_window_days",
    "rainfall_pairing_reference_time",
    "rainfall_pairing_lag_hours",
    "streamflow_source",
    "streamflow_member_file",
    "streamflow_member_id",
    "streamflow_member_time",
    "streamflow_pairing_policy",
    "streamflow_pairing_seed",
    "streamflow_pairing_window_days",
    "streamflow_pairing_reference_time",
    "streamflow_pairing_lag_hours",
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
]


def _is_missing(value):
    return pd.isna(value) or value == ""


def _event_id(row):
    value = row.get("event_id", pd.NA)
    return "<missing>" if _is_missing(value) else str(value)


def _issue(code, event_id=None, column=None, forcing=None, severity="error"):
    issue = {"severity": severity, "code": code}
    if event_id is not None:
        issue["event_id"] = event_id
    if column is not None:
        issue["column"] = column
    if forcing is not None:
        issue["forcing"] = forcing
    return issue


def validate_event_catalog(catalog, required_forcings=None, wave_analog_policy=None, min_coastal_window_hours=72):
    required_forcings = set(required_forcings or ["coastal"])
    issues = []
    for column in ["event_id", "study_location", "event_family", "scenario_name"]:
        if column not in catalog:
            issues.append(_issue("missing_column", column=column))
    if issues:
        return issues

    if catalog["event_id"].duplicated().any():
        for event_id in catalog.loc[catalog["event_id"].duplicated(keep=False), "event_id"]:
            issues.append(_issue("duplicate_event_id", event_id=str(event_id), column="event_id"))

    if {"coastal_valid_start_hour", "coastal_valid_end_hour"}.issubset(catalog.columns):
        start = pd.to_numeric(catalog["coastal_valid_start_hour"], errors="coerce")
        end = pd.to_numeric(catalog["coastal_valid_end_hour"], errors="coerce")
        durations = (end - start).dropna()
        if len(durations) and float(durations.median()) < float(min_coastal_window_hours):
            issues.append(
                _issue(
                    "coastal_window_too_short",
                    column="coastal_valid_start_hour/coastal_valid_end_hour",
                )
            )

    for _, row in catalog.iterrows():
        event_id = _event_id(row)
        for column in ["study_location", "event_family", "scenario_name"]:
            if _is_missing(row.get(column, pd.NA)):
                issues.append(_issue("missing_value", event_id=event_id, column=column))
        # Observed historical-tail reference events sit outside the synthetic probability budget,
        # so they carry no sampling/probability weight by design; only synthetic rows are weighted.
        is_reference_row = (
            str(row.get("event_origin", "")) == "historical_tail"
            or str(row.get("catalog_role", "")) == "historical_reference"
        )
        weight = pd.to_numeric(pd.Series([row.get("sampling_weight", pd.NA)]), errors="coerce").iloc[0]
        if not is_reference_row and (pd.isna(weight) or weight <= 0):
            issues.append(_issue("invalid_sampling_weight", event_id=event_id, column="sampling_weight"))
        rp = pd.to_numeric(pd.Series([row.get("sample_rp_years", pd.NA)]), errors="coerce").iloc[0]
        if pd.isna(rp) or rp <= 0:
            issues.append(_issue("invalid_return_period", event_id=event_id, column="sample_rp_years"))
        if row.get("sampling_region", pd.NA) not in {"body", "tail"}:
            issues.append(_issue("invalid_sampling_region", event_id=event_id, column="sampling_region"))

        if wave_analog_policy == "same_historical_analog":
            wave_columns = [
                "coastal_analog_id",
                "coastal_analog_peak_time",
                "snapwave_source",
                "snapwave_member_file",
                "snapwave_member_id",
                "snapwave_valid_start_time",
                "snapwave_valid_end_time",
                "snapwave_pairing_policy",
            ]
            for column in wave_columns:
                if column not in catalog or _is_missing(row.get(column, pd.NA)):
                    issues.append(_issue("incomplete_wave_analog", event_id=event_id, column=column))
            policy = row.get("snapwave_pairing_policy", pd.NA)
            if not _is_missing(policy) and str(policy) != "same_historical_analog":
                issues.append(_issue("invalid_wave_pairing_policy", event_id=event_id, column="snapwave_pairing_policy"))

        for forcing in ["coastal", "rainfall", "streamflow", "soil_moisture"]:
            provenance = [
                f"{forcing}_source",
                f"{forcing}_member_file",
                f"{forcing}_member_id",
            ]
            pairing = [] if forcing == "coastal" else [
                f"{forcing}_pairing_policy",
                f"{forcing}_pairing_seed",
            ]
            columns = provenance + pairing
            values = [row.get(column, pd.NA) for column in columns]
            forcing_present = any(not _is_missing(value) for value in values)
            if forcing not in required_forcings and not forcing_present:
                continue
            for column, value in zip(columns, values):
                if _is_missing(value):
                    issues.append(
                        _issue(
                            "incomplete_forcing",
                            event_id=event_id,
                            forcing=forcing,
                            column=column,
                        )
                    )
            pairing_policy = row.get(f"{forcing}_pairing_policy", pd.NA)
            if not _is_missing(pairing_policy) and str(pairing_policy) == "seasonal_window_permutation":
                for column in [f"{forcing}_member_time", f"{forcing}_pairing_window_days"]:
                    if _is_missing(row.get(column, pd.NA)):
                        issues.append(
                            _issue(
                                "incomplete_seasonal_pairing",
                                event_id=event_id,
                                forcing=forcing,
                                column=column,
                            )
                        )
            if not _is_missing(pairing_policy) and str(pairing_policy) == "antecedent_to_forcing":
                for column in [
                    f"{forcing}_member_time",
                    f"{forcing}_pairing_reference_time",
                    f"{forcing}_pairing_lag_hours",
                ]:
                    if _is_missing(row.get(column, pd.NA)):
                        issues.append(
                            _issue(
                                "incomplete_antecedent_pairing",
                                event_id=event_id,
                                forcing=forcing,
                                column=column,
                            )
                        )
    return issues


def write_event_catalog_audit(catalog, path, required_forcings=None, wave_analog_policy=None):
    issues = validate_event_catalog(
        catalog,
        required_forcings=required_forcings,
        wave_analog_policy=wave_analog_policy,
    )
    audit = {
        "passed": len(issues) == 0,
        "event_count": int(len(catalog)),
        "issue_count": int(len(issues)),
        "issues": issues,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def _member_values(members, column, default=pd.NA):
    if column in members:
        return members[column]
    return pd.Series([default] * len(members))


def _first_existing_column(frame, candidates):
    for column in candidates:
        if column in frame:
            return column
    return None


def _summary_values(summary, column, default=pd.NA):
    if column in summary:
        return summary[column]
    return pd.Series([default] * len(summary))


def _event_window_time(summary, base_column, offset_column):
    if base_column not in summary or offset_column not in summary:
        return pd.Series([pd.NA] * len(summary), dtype="object")
    base = pd.to_datetime(summary[base_column], errors="coerce")
    offset = pd.to_numeric(summary[offset_column], errors="coerce")
    values = base + pd.to_timedelta(offset, unit="h")
    return values.dt.strftime("%Y-%m-%dT%H:%M:%S").where(values.notna(), pd.NA)


def _assignment_indices(n_events, n_members, seed):
    rng = np.random.default_rng(int(seed))
    assigned = []
    base = np.arange(n_members)
    while len(assigned) < n_events:
        assigned.extend(rng.permutation(base).tolist())
    return assigned[:n_events]


def _day_of_year_distance(a, b):
    diff = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return np.minimum(diff, 366.0 - diff)


def _seasonal_assignment_indices(catalog, members, policy):
    event_column = policy.get("event_time_column") or _first_existing_column(
        catalog,
        ["coastal_template_peak_time", "template_peak_time", "event_time", "event_date"],
    )
    member_column = policy.get("member_time_column") or _first_existing_column(
        members,
        ["storm_date", "storm_start", "member_time", "event_time", "event_date", "time"],
    )
    if event_column is None or member_column is None:
        raise ValueError("seasonal pairing requires event and member time columns")

    event_times = pd.to_datetime(catalog[event_column], errors="coerce")
    member_times = pd.to_datetime(members[member_column], errors="coerce")
    if event_times.isna().any() or member_times.isna().all():
        raise ValueError("seasonal pairing requires parseable event and member times")

    rng = np.random.default_rng(int(policy.get("seed", 0)))
    window_days = int(policy.get("window_days", 45))
    member_doy = member_times.dt.dayofyear.to_numpy(dtype=float)
    assigned = []
    for row_index, event_time in enumerate(event_times):
        distances = _day_of_year_distance(member_doy, event_time.dayofyear)
        candidates = np.flatnonzero(distances <= window_days)
        if candidates.size == 0:
            if policy.get("fallback_strategy") == "nearest":
                candidates = np.array([int(np.argmin(distances))])
            else:
                event_id = _event_id(catalog.iloc[row_index])
                raise ValueError(f"no {window_days}-day seasonal candidates for {event_id}")
        if candidates.size == 0:
            event_id = _event_id(catalog.iloc[row_index])
            raise ValueError(f"no {window_days}-day seasonal candidates for {event_id}")
        assigned.append(int(rng.choice(candidates)))
    return assigned, member_column


def _antecedent_assignment_indices(catalog, members, policy):
    reference_column = policy.get("reference_time_column")
    if reference_column is None:
        reference_forcing = policy.get("reference_forcing", "rainfall")
        reference_column = f"{reference_forcing}_member_time"
    member_column = policy.get("member_time_column") or _first_existing_column(
        members,
        ["time", "member_time", "storm_date", "storm_start", "event_time", "event_date"],
    )
    if reference_column not in catalog or member_column is None:
        raise ValueError("antecedent pairing requires reference and member time columns")

    reference_times = pd.to_datetime(catalog[reference_column], errors="coerce")
    member_times = pd.to_datetime(members[member_column], errors="coerce")
    if reference_times.isna().any() or member_times.isna().all():
        raise ValueError("antecedent pairing requires parseable reference and member times")

    lead_hours = float(policy.get("lead_time_hours", 24))
    fallback = policy.get("fallback_strategy", "nearest")
    assigned = []
    member_ns = member_times.astype("int64").to_numpy()
    for row_index, reference_time in enumerate(reference_times):
        target_time = reference_time - pd.Timedelta(hours=lead_hours)
        candidates = np.flatnonzero(member_times <= target_time)
        if candidates.size == 0:
            if fallback == "nearest":
                deltas = np.abs(member_ns - pd.Timestamp(target_time).value)
                candidates = np.array([int(np.nanargmin(deltas))])
            else:
                event_id = _event_id(catalog.iloc[row_index])
                raise ValueError(f"no antecedent member available for {event_id}")
        candidate_times = member_times.iloc[candidates].astype("int64").to_numpy()
        selected = int(candidates[np.argmin(np.abs(candidate_times - pd.Timestamp(target_time).value))])
        assigned.append(selected)
    return assigned, member_column, reference_column, lead_hours


def attach_forcing_members(catalog, members, forcing, policy=None):
    policy = policy or {}
    strategy = policy.get("strategy", "independent_permutation")
    seed = int(policy.get("seed", 0 if strategy == "antecedent_to_forcing" else 42))
    if strategy not in {"independent_permutation", "seasonal_window_permutation", "antecedent_to_forcing"}:
        raise ValueError(f"unsupported forcing pairing strategy {strategy!r}")
    out = catalog.copy()
    if members.empty:
        for suffix in [
            "source",
            "member_file",
            "member_id",
            "member_time",
            "pairing_policy",
            "pairing_seed",
            "pairing_window_days",
            "pairing_reference_time",
            "pairing_lag_hours",
        ]:
            column = f"{forcing}_{suffix}"
            if column not in out:
                out[column] = pd.NA
        return out

    if strategy == "seasonal_window_permutation":
        assigned, member_time_column = _seasonal_assignment_indices(out, members, policy)
        reference_column = None
        lag_hours = pd.NA
    elif strategy == "antecedent_to_forcing":
        assigned, member_time_column, reference_column, lag_hours = _antecedent_assignment_indices(out, members, policy)
    else:
        assigned = _assignment_indices(len(out), len(members), seed)
        member_time_column = _first_existing_column(
            members,
            ["storm_date", "storm_start", "member_time", "event_time", "event_date", "time"],
        )
        reference_column = None
        lag_hours = pd.NA
    selected = members.iloc[assigned].reset_index(drop=True)
    out[f"{forcing}_source"] = _member_values(selected, "source", forcing).to_numpy()
    out[f"{forcing}_member_file"] = _member_values(selected, "member_file").to_numpy()
    out[f"{forcing}_member_id"] = _member_values(selected, "member_id").to_numpy()
    out[f"{forcing}_member_time"] = _member_values(selected, member_time_column).to_numpy()
    out[f"{forcing}_pairing_policy"] = strategy
    out[f"{forcing}_pairing_seed"] = seed
    out[f"{forcing}_pairing_window_days"] = policy.get("window_days", pd.NA) if strategy == "seasonal_window_permutation" else pd.NA
    out[f"{forcing}_pairing_reference_time"] = out[reference_column].to_numpy() if reference_column else pd.NA
    out[f"{forcing}_pairing_lag_hours"] = lag_hours
    return out


def _attach_configured_forcing(catalog, plan):
    out = catalog
    for forcing in plan.forcings:
        members = pd.read_csv(forcing.member_path)
        members = _normalize_member_table(members, forcing.name, forcing.member_path)
        out = attach_forcing_members(out, members, forcing.name, forcing.pairing_policy)
    return out


def rebuild_forcing_pairing(catalog, member_path, forcing, policy=None):
    member_path = Path(member_path)
    members = pd.read_csv(member_path)
    members = _normalize_member_table(members, forcing, member_path)
    out = catalog.drop(
        columns=[column for column in catalog.columns if column.startswith(f"{forcing}_")],
        errors="ignore",
    )
    return attach_forcing_members(out, members, forcing, policy)


def _normalize_member_table(members, forcing, member_path):
    if forcing == "soil_moisture" and "time" in members and _soil_moisture_value_column(members):
        return _normalize_soil_moisture_member_table(members, member_path)
    out = members.copy()
    if "source" not in out:
        out["source"] = forcing
    if "member_file" not in out:
        out["member_file"] = str(member_path)
    if "member_id" not in out:
        if {"point_id", "time"}.issubset(out.columns):
            times = pd.to_datetime(out["time"], errors="coerce").dt.strftime("%Y%m%dT%H%M%S")
            out["member_id"] = out["point_id"].astype(str) + "_" + times.fillna("unknown")
        elif "time" in out:
            times = pd.to_datetime(out["time"], errors="coerce").dt.strftime("%Y%m%dT%H%M%S")
            out["member_id"] = forcing + "_" + times.fillna("unknown")
        else:
            out["member_id"] = [f"{forcing}_{index:04d}" for index in range(len(out))]
    return out


def _normalize_soil_moisture_member_table(members, member_path):
    frame = members.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"])
    value_column = _soil_moisture_value_column(frame)
    grouped = frame.groupby("time", as_index=False).agg(
        soil_moisture_mean=(value_column, "mean"),
        soil_moisture_min=(value_column, "min"),
        soil_moisture_max=(value_column, "max"),
        point_count=("point_id", "nunique") if "point_id" in frame else (value_column, "size"),
        layer_count=("soil_layers_stag", "nunique") if "soil_layers_stag" in frame else (value_column, "size"),
    )
    times = pd.to_datetime(grouped["time"], errors="coerce")
    grouped["member_id"] = "soil_moisture_" + times.dt.strftime("%Y%m%dT%H%M%S")
    grouped["source"] = "nwm"
    grouped["member_file"] = str(member_path)
    grouped["time"] = times.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return grouped


def _soil_moisture_value_column(frame):
    for column in ["SOILSAT_TOP", "SOIL_M"]:
        if column in frame:
            return column
    return None


def build_event_catalog(config, paths):
    catalog_plan = plan(config, paths)
    summary = pd.read_csv(catalog_plan.event_summary_csv)
    scenario = paths.get("scenario", {})
    coastal_analog_id = _summary_values(summary, "template_id", pd.NA)
    snapwave_required = catalog_plan.wave_analog_policy == "same_historical_analog"
    catalog = pd.DataFrame(
        {
            "event_id": summary["event_id"].astype(str),
            "study_location": catalog_plan.study_location,
            "event_family": "surge_synthetic",
            "scenario_name": catalog_plan.scenario_name or scenario.get("name", summary.get("scenario_name", "base")),
            "sample_rp_years": summary.get("sample_rp_years"),
            "severity_band": summary.get(
                "severity_band",
                assign_severity_bands(
                    summary.get("sample_rp_years"),
                    config.get("sampling", {}).get("severity_bands"),
                ),
            ),
            "sampling_region": summary.get("sampling_region"),
            "sampling_weight": summary.get("sampling_weight"),
            "probability_weight": summary.get("probability_weight"),
            "coastal_source": "cora",
            "coastal_member_file": str(catalog_plan.event_members_nc),
            "coastal_member_id": summary["event_id"].astype(str),
            "coastal_template_peak_time": summary.get("template_peak_time"),
            "coastal_peak_m": summary.get("peak"),
            "coastal_absolute_peak_m": summary.get("absolute_peak_m"),
            "coastal_valid_start_hour": summary.get("valid_start_hour"),
            "coastal_valid_end_hour": summary.get("valid_end_hour"),
            "coastal_analog_id": coastal_analog_id,
            "coastal_analog_peak_time": summary.get("template_peak_time"),
            "snapwave_source": "era5" if snapwave_required else pd.NA,
            "snapwave_member_file": str(paths.get("era5_waves_nc", "")) if snapwave_required else pd.NA,
            "snapwave_member_id": coastal_analog_id if snapwave_required else pd.NA,
            "snapwave_valid_start_time": _event_window_time(summary, "template_peak_time", "valid_start_hour")
            if snapwave_required
            else pd.NA,
            "snapwave_valid_end_time": _event_window_time(summary, "template_peak_time", "valid_end_hour")
            if snapwave_required
            else pd.NA,
            "snapwave_pairing_policy": catalog_plan.wave_analog_policy if snapwave_required else pd.NA,
            "rainfall_source": pd.NA,
            "rainfall_member_file": pd.NA,
            "rainfall_member_id": pd.NA,
            "rainfall_member_time": pd.NA,
            "rainfall_pairing_policy": pd.NA,
            "rainfall_pairing_seed": pd.NA,
            "rainfall_pairing_window_days": pd.NA,
            "rainfall_pairing_reference_time": pd.NA,
            "rainfall_pairing_lag_hours": pd.NA,
            "streamflow_source": pd.NA,
            "streamflow_member_file": pd.NA,
            "streamflow_member_id": pd.NA,
            "streamflow_member_time": pd.NA,
            "streamflow_pairing_policy": pd.NA,
            "streamflow_pairing_seed": pd.NA,
            "streamflow_pairing_window_days": pd.NA,
            "streamflow_pairing_reference_time": pd.NA,
            "streamflow_pairing_lag_hours": pd.NA,
            "soil_moisture_source": pd.NA,
            "soil_moisture_member_file": pd.NA,
            "soil_moisture_member_id": pd.NA,
            "soil_moisture_member_time": pd.NA,
            "soil_moisture_pairing_policy": pd.NA,
            "soil_moisture_pairing_seed": pd.NA,
            "soil_moisture_pairing_window_days": pd.NA,
            "soil_moisture_pairing_reference_time": pd.NA,
            "soil_moisture_pairing_lag_hours": pd.NA,
            "infiltration_treatment": config.get("infiltration", {}).get("treatment", "none"),
        }
    )
    catalog = _attach_configured_forcing(catalog, catalog_plan)
    catalog = catalog[catalog_columns]
    catalog_plan.event_catalog_csv.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(catalog_plan.event_catalog_csv, index=False)
    if catalog_plan.audit_json is not None:
        write_event_catalog_audit(
            catalog,
            catalog_plan.audit_json,
            required_forcings=catalog_plan.required_forcings,
            wave_analog_policy=catalog_plan.wave_analog_policy,
        )
    return catalog
