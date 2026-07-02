from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import pandas as pd
import yaml

from study_location import define_location
from location_runtime import WflowCalibrationRuntime, WflowRuntime, build_wflow_runtime
from paths import resolve_location_path
from coupling.wflow_domain_set import plan_wflow_domain_set
from wflow_runs.fabric import write_wflow_subbasin_fabric_from_nhdplus
from wflow_runs.hydromt_build import build_wflow_build_plan
from wflow_runs.hydromt_runtime import (
    _describe_hydromt_command,
    _hydromt_missing_message,
    _hydromt_subprocess_env,
    _resolve_hydromt_command,
)


@dataclass(frozen=True)
class WflowNotebookContext:
    location_root: Path
    repo_root: Path
    config: dict
    grid_config: dict
    data_sources: dict
    sfincs_config: dict
    wflow_config: dict
    runtime_config: dict


WflowCoupledNotebookRuntime = WflowRuntime
WflowCalibrationNotebookRuntime = WflowCalibrationRuntime


def _load_wflow_sfincs_runtime(
    location_root,
    *,
    wflow_domain_review_required: bool | None = None,
) -> WflowCoupledNotebookRuntime:
    """Load derived paths for Wflow-coupled Flood Notebook Workflow stages.

    Generated artifact paths stay derived from the Location Workspace convention
    instead of being repeated in notebook cells or location YAML.
    """
    location_root = Path(location_root).resolve()
    return build_wflow_runtime(
        define_location(location_root / "config.yaml"),
        wflow_domain_review_required=wflow_domain_review_required,
    )


def load_calibration_runtime(
    location_root,
    *,
    create_audit_dirs: bool = True,
) -> WflowCalibrationNotebookRuntime:
    """Load derived paths for the Wflow Readiness calibration notebook.

    When ``create_audit_dirs`` is true, the helper creates the calibration plot
    output directory used by the notebook.
    """
    location_root = Path(location_root).resolve()
    return build_wflow_runtime(
        define_location(location_root / "config.yaml"),
        workflow="calibration",
        create_audit_dirs=create_audit_dirs,
    )


def load_runtime(location_root, *, workflow: str = "coupled", **kwargs):
    """Load the Wflow notebook runtime for a named workflow."""
    if workflow == "coupled":
        return _load_wflow_sfincs_runtime(location_root, **kwargs)
    if workflow == "calibration":
        return load_calibration_runtime(location_root, **kwargs)
    raise ValueError("workflow must be 'coupled' or 'calibration'")


def find_location_root(location_name: str | None = None, *, start: Path | None = None) -> Path:
    # With no name, resolve the location from the working directory: the nearest
    # ancestor that holds a config.yaml and sits directly under locations/. A name
    # narrows the match to that specific location workspace.
    here = (start or Path.cwd()).resolve()
    for base in (here, *here.parents):
        if (base / "config.yaml").exists() and (
            base.parent.name == "locations" if location_name is None else base.name == location_name
        ):
            return base
        if location_name is not None:
            candidate = base / "locations" / location_name
            if (candidate / "config.yaml").exists():
                return candidate
    if location_name is not None:
        fallback = Path("locations") / location_name
        if (fallback / "config.yaml").exists():
            return fallback.resolve()
    raise FileNotFoundError(
        "Could not locate a locations/<name>/config.yaml above the working directory"
        if location_name is None
        else f"Could not locate locations/{location_name}/config.yaml"
    )


def exists_table(location_root: Path, named_paths: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "artifact": name,
                "path": str(resolve_location_path(location_root, relative_path)),
                "exists": resolve_location_path(location_root, relative_path).exists(),
            }
            for name, relative_path in named_paths.items()
        ]
    )


def domain_summary(config: dict, location_root: Path) -> tuple:
    build_plan = build_wflow_build_plan(config, {"location_root": location_root})
    domain_plan = plan_wflow_domain_set(config, {"location_root": location_root})
    wflow_cfg = config.get("wflow", {}) or {}
    domain_set = wflow_cfg.get("domain_set", {}) or {}
    # Use the same defaults the rest of wflow_runs applies via .get(), so locations that
    # don't spell out every optional domain_set key in YAML (greensboro, austin) summarize
    # instead of KeyError'ing. Defaults mirror coupling.domain_manifest.
    summary = pd.Series(
        {
            "allow_multiple_submodels": domain_set.get("allow_multiple_submodels", False),
            "review_required": build_plan.review_required,
            "domain_status": build_plan.domain_status,
            "reviewed_subbasin_plan_status": domain_plan.status,
            "hydromt_region_kind": build_plan.region_kind,
            "event_catalog_scope": domain_set.get("event_catalog_scope", "shared_across_domain_set"),
            "configured_submodel_count": len(domain_set.get("submodels", []) or []),
            "reviewed_submodel_count": domain_plan.submodel_count,
            "reviewed_handoff_count": domain_plan.handoff_count,
            "domain_set_manifest": wflow_cfg.get("domain_set_manifest", "data/wflow/domain_set.yaml"),
        }
    )
    return build_plan, domain_plan, summary


def subbasins(domain_plan) -> pd.DataFrame:
    if not domain_plan.submodels:
        return pd.DataFrame(
            [{"status": domain_plan.status, "issue": issue} for issue in domain_plan.issues]
        )
    rows = []
    for submodel in domain_plan.submodels:
        outlet_region = submodel.get("outlet_region", submodel["region"])
        outlet_xy = outlet_region.get("subbasin") if isinstance(outlet_region, dict) else None
        outlet_lon, outlet_lat = outlet_xy if outlet_xy else (None, None)
        rows.append(
            {
                "wflow_submodel_id": submodel["wflow_submodel_id"],
                "hydromt_region_kind": submodel["region_kind"],
                "hydromt_region": submodel["region"],
                "handoff_outlet_lon": outlet_lon,
                "handoff_outlet_lat": outlet_lat,
                "sfincs_domain_ids": ", ".join(submodel["sfincs_domain_ids"]),
                "sfincs_handoff_ids": ", ".join(submodel["sfincs_handoff_ids"]),
                "gauge_site_nos": ", ".join(submodel["gauge_site_nos"]),
                "frequency_basis": ", ".join(submodel["frequency_basis"]),
            }
        )
    return pd.DataFrame(rows)


def wflow_event_replay_plan(config: dict, location_root: Path, event_id: str | None) -> pd.Series:
    """Return the reviewed HydroMT-Wflow event replay command for one catalog event."""
    build_plan = build_wflow_build_plan(config, {"location_root": location_root})
    command = build_plan.update_command.replace("<event_id>", str(event_id or "<event_id>"))
    resolved_command, runner_status, runner_issue = _describe_hydromt_command(command, location_root)
    event_dir = build_plan.events_root / str(event_id) if event_id else build_plan.events_root / "<event_id>"
    return pd.Series(
        {
            "event_id": event_id,
            "wflow_event_dir": str(event_dir),
            "wflow_discharge_forcing": str(event_dir / "sfincs_discharge.nc"),
            "hydromt_wflow_update_command": command,
            "resolved_hydromt_wflow_update_command": resolved_command,
            "hydromt_runner_status": runner_status,
            "hydromt_runner_issue": runner_issue,
        },
        name="wflow_event_replay_plan",
    )


def run_wflow_event_replay(
    config: dict,
    location_root: Path,
    event_id: str,
    *,
    execute: bool = False,
) -> pd.Series:
    """Run or dry-run the HydroMT-Wflow event replay command for one event."""
    plan = wflow_event_replay_plan(config, location_root, event_id)
    command = str(plan["hydromt_wflow_update_command"])
    resolved_command = str(plan["resolved_hydromt_wflow_update_command"])
    runner_status = str(plan["hydromt_runner_status"])
    runner_issue = str(plan["hydromt_runner_issue"])
    if execute:
        command_parts = _resolve_hydromt_command(command, location_root)
        try:
            subprocess.run(command_parts, cwd=Path(location_root), check=True, env=_hydromt_subprocess_env(location_root))
        except FileNotFoundError as exc:
            raise RuntimeError(_hydromt_missing_message(command, location_root)) from exc
        status = "completed"
    else:
        status = "dry_run"
    return pd.Series(
        {
            "event_id": event_id,
            "status": status,
            "command": command,
            "resolved_command": resolved_command,
            "hydromt_runner_status": runner_status,
            "hydromt_runner_issue": runner_issue,
            "wflow_event_dir": plan["wflow_event_dir"],
            "wflow_discharge_forcing": plan["wflow_discharge_forcing"],
        },
        name="wflow_event_replay",
    )


def prepare_wflow_subbasin_fabric(config: dict, location_root: Path, domain_plan) -> tuple:
    wflow = config["wflow"]
    data_sources = config["collection"]["national_hydrography"]
    inputs = exists_table(
        location_root,
        {
            "NHDPlus HR river geometry": data_sources["river_geometry"],
            "NHDPlus HR catchments": data_sources["catchments"],
        },
    )
    if domain_plan.status == "ready" and inputs["exists"].all():
        result = write_wflow_subbasin_fabric_from_nhdplus(config, {"location_root": location_root})
    else:
        subbasin_fabric_path = resolve_location_path(
            location_root,
            wflow["domain_set"].get("subbasin_fabric", "data/wflow/domain_set_subbasins.gpkg"),
        )
        result = {
            "subbasin_fabric": subbasin_fabric_path,
            "subbasin_geometry_files": tuple(sorted(subbasin_fabric_path.with_suffix("").glob("*.geojson"))),
            "diagnostics_csv": resolve_location_path(
                location_root,
                wflow["domain_set"].get(
                    "subbasin_fabric_diagnostics",
                    "data/wflow/readiness/nhdplus_subbasin_fabric.csv",
                ),
            ),
            "submodel_count": 0,
            "catchment_count": 0,
            "statuses": ("missing_inputs_or_review_required",),
        }
    domain_plan = plan_wflow_domain_set(config, {"location_root": location_root})
    summary = pd.Series(
        {
            "subbasin_fabric": str(result["subbasin_fabric"]),
            "subbasin_geometry_files": len(result.get("subbasin_geometry_files", ())),
            "diagnostics_csv": str(result["diagnostics_csv"]),
            "submodel_count": result["submodel_count"],
            "catchment_count": result["catchment_count"],
            "statuses": ", ".join(result["statuses"]),
            "coverage_status": result.get("coverage_status"),
            "coverage_catchment_count": result.get("coverage_catchment_count", 0),
            "evaluation_footprint_within_domain": result.get("evaluation_footprint_within_domain"),
            "evaluation_footprint_uncovered_km2": result.get("evaluation_footprint_uncovered_km2"),
            "power_extent_within_domain": result.get("power_extent_within_domain"),
            "power_extent_uncovered_km2": result.get("power_extent_uncovered_km2"),
            "replanned_status": domain_plan.status,
            "replanned_hydromt_region_kinds": ", ".join(
                sorted({submodel["region_kind"] for submodel in domain_plan.submodels})
            ),
        },
        name="nhdplus_subbasin_fabric_result",
    )
    return result, inputs, domain_plan, summary


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))

