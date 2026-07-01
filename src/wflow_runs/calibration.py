from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from coupling.dynamic_handoff import plan_handoff
from coupling.wflow_sfincs_batch import run_handoffs
from wflow_runs.scores import usgs_calibration_table


@dataclass(frozen=True)
class ValidationEvents:
    summary: pd.Series
    scenarios: pd.DataFrame
    event_ids: list[str]


@dataclass(frozen=True)
class WflowValidationAudit:
    plan: pd.DataFrame
    report: pd.DataFrame


@dataclass(frozen=True)
class WflowCalibration:
    summary: pd.DataFrame
    artifacts: pd.Series
    patch: dict


def select_validation_events(
    config: dict,
    *,
    scenario_catalog_path,
    readiness_path,
    joint_worklist_path,
    n: int = 5,
) -> ValidationEvents:
    """Choose historical-like events for Wflow validation against observed USGS IV."""
    catalog = pd.read_csv(scenario_catalog_path, dtype={"event_id": str})
    readiness = _read_optional_csv(readiness_path, ["event_id", "status"])
    joint_worklist = _read_optional_csv(joint_worklist_path, ["event_id"])

    explicit_ids = (config.get("wflow", {}).get("validation", {}) or {}).get("historical_event_ids") or []
    if explicit_ids:
        scenarios = catalog[catalog["event_id"].astype(str).isin([str(e) for e in explicit_ids])].copy()
        basis = "config wflow.validation.historical_event_ids"
    elif "rainfall_scale_factor" in catalog:
        pool = catalog.copy()
        pool["historical_proximity"] = (pd.to_numeric(pool["rainfall_scale_factor"], errors="coerce") - 1.0).abs()
        scenarios = pool.nsmallest(n, "historical_proximity").sort_values("event_id").reset_index(drop=True)
        basis = "nearest-to-historical (rainfall_scale_factor approx 1.0)"
    else:
        scenarios = catalog.sort_values("sample_rp_years", ascending=False).head(n).reset_index(drop=True)
        basis = "largest sample_rp_years (fallback)"

    event_ids = scenarios["event_id"].astype(str).tolist()
    summary = pd.Series(
        {
            "catalog_rows": len(catalog),
            "readiness_rows": len(readiness),
            "joint_worklist_rows": len(joint_worklist),
            "validation_event_selection": basis,
            "validation_event_count": len(scenarios),
        },
        name="wflow_validation_inputs",
    )
    return ValidationEvents(summary=summary, scenarios=scenarios, event_ids=event_ids)


def run_wflow_validation_audit(
    config: dict,
    location_root,
    event_ids: list[str],
    *,
    scenario_catalog_path,
    rerun: bool = False,
    execute: bool = False,
) -> WflowValidationAudit:
    plan = pd.DataFrame(
        [plan_handoff(config, location_root, event_id, catalog_path=scenario_catalog_path).to_dict() for event_id in event_ids]
    )
    if not execute:
        report = pd.DataFrame(
            {
                "event_id": event_ids,
                "status": "not_run",
                "message": (
                    "Set EXECUTE_WFLOW_AUDIT=True to run the production dynamic Wflow handoff "
                    "for the historical validation events."
                ),
            }
        )
        return WflowValidationAudit(plan=plan, report=report)

    reports = [
        run_handoffs(
            config,
            location_root,
            catalog_path=scenario_catalog_path,
            event_ids=[event_id],
            status="all",
            execute=True,
            force=rerun,
            overwrite_meteo=rerun,
        )
        for event_id in event_ids
    ]
    report = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    return WflowValidationAudit(plan=plan, report=report)


def build_usgs_wflow_calibration(
    config: dict,
    location_root,
    event_ids: list[str],
    *,
    events_root,
    wflow_base_root,
    event_streamflow_iv_root,
    active_wflow_submodel_id: str | None = None,
) -> WflowCalibration:
    """Score Wflow against observed USGS IV and write the single-K calibration patch."""
    import yaml

    location_root = Path(location_root)
    tables = [
        usgs_calibration_table(
            event_id,
            events_root=events_root,
            wflow_base_root=wflow_base_root,
            event_streamflow_iv_root=event_streamflow_iv_root,
            submodel_id=active_wflow_submodel_id,
        )
        for event_id in event_ids
    ]
    summary = pd.concat([table for table in tables if not table.empty], ignore_index=True) if any(
        not table.empty for table in tables
    ) else pd.DataFrame()
    calibration_root = location_root / "data/wflow/calibration"
    calibration_root.mkdir(parents=True, exist_ok=True)
    summary_path = calibration_root / "usgs_wflow_calibration_summary.csv"
    summary.to_csv(summary_path, index=False)

    patch, status, suggested_k, event_count = _calibration_patch(config, location_root, summary, summary_path)
    json_path = calibration_root / "wflow_calibration_patch.json"
    yaml_path = calibration_root / "wflow_calibration_patch.yaml"
    json_path.write_text(json.dumps(patch, indent=2), encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(patch, sort_keys=False), encoding="utf-8")

    artifacts = pd.Series(
        {
            "calibration_summary_csv": str(summary_path),
            "calibration_patch_yaml": str(yaml_path),
            "calibration_status": status,
            "suggested_k_calibration": suggested_k,
            "validation_event_count": event_count,
        },
        name="wflow_calibration_artifacts",
    )
    return WflowCalibration(summary=summary, artifacts=artifacts, patch=patch)


def _calibration_patch(config: dict, location_root: Path, calibration_summary: pd.DataFrame, summary_path: Path) -> tuple[dict, str, float | None, int]:
    min_overlap_steps = 12
    min_events_for_ship = 2
    default_k_band = [0.25, 4.0]
    amp_cfg = ((config.get("inland_coupling", {}) or {}).get("amplification", {}) or {})
    primary_reference_gage = amp_cfg.get("primary_reference_gage") or (config.get("inland_coupling", {}) or {}).get("primary_reference_gage")

    valid = calibration_summary.copy()
    if primary_reference_gage and not valid.empty and "site_no" in valid:
        valid = valid[valid["site_no"].astype(str).eq(str(primary_reference_gage))].copy()
    if not valid.empty and {"n", "peak_bias_fraction"}.issubset(valid.columns):
        for column in ["n", "peak_bias_fraction", "volume_bias_fraction", "kge", "nse"]:
            if column in valid:
                valid[column] = pd.to_numeric(valid[column], errors="coerce")
        valid = valid[valid["n"].ge(min_overlap_steps) & valid["peak_bias_fraction"].gt(-0.95)].copy()
        valid["k_peak"] = 1.0 / (1.0 + valid["peak_bias_fraction"])
        if "volume_bias_fraction" in valid:
            valid["k_volume"] = 1.0 / (1.0 + valid["volume_bias_fraction"])
        valid = valid[np.isfinite(valid["k_peak"])]

    suggested_k = float(valid["k_peak"].median()) if not valid.empty and "k_peak" in valid else None
    event_count = int(valid["event_id"].nunique()) if not valid.empty and "event_id" in valid else 0
    status = "ready_to_ship" if suggested_k and event_count >= min_events_for_ship else "insufficient_validation_events"
    patch = {
        "location": location_root.name,
        "status": status,
        "method": "single-K peak-bias correction from Wflow-vs-USGS IV validation events",
        "primary_reference_gage": str(primary_reference_gage) if primary_reference_gage else None,
        "event_count": event_count,
        "row_count": int(len(valid)),
        "k_calibration": suggested_k,
        "k_band": amp_cfg.get("k_band", default_k_band),
        "source_summary_csv": str(summary_path.relative_to(location_root)),
        "runtime_config_patch": {
            "inland_coupling": {
                "amplification": {
                    "enabled": True,
                    "primary_reference_gage": str(primary_reference_gage) if primary_reference_gage else None,
                    "k_calibration": suggested_k,
                    "k_band": amp_cfg.get("k_band", default_k_band),
                }
            }
        },
    }
    return patch, status, suggested_k, event_count


def _read_optional_csv(path, columns: list[str]) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path, dtype={"event_id": str}) if path.exists() else pd.DataFrame(columns=columns)
