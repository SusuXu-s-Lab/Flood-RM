"""Runtime adapter from Flood-RM Location Configuration to the clean Wflow core."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from study_location import LocationDefinition


@dataclass(frozen=True)
class WflowRuntime:
    """Compatibility projection of the coupled Wflow notebook runtime."""

    definition: LocationDefinition
    location_root: Path
    location_name: str
    config: dict[str, Any]
    paths: dict[str, Path | str]
    design_paths: dict[str, Path | str]
    runtime_config: dict[str, Any]
    sfincs_config: dict[str, Any]
    wflow_config: dict[str, Any]
    sfincs_scenarios_root: Path
    scenario_catalog_path: Path
    probability_catalog_path: Path
    readiness_path: Path
    blocked_path: Path
    accepted_path: Path
    joint_worklist_path: Path
    incompatible_path: Path
    events_root: Path
    wflow_base_root: Path
    wflow_handoff_manifest: Path


@dataclass(frozen=True)
class WflowCalibrationRuntime(WflowRuntime):
    """Compatibility projection of the Wflow calibration notebook runtime."""

    streamflow_records_path: Path
    event_streamflow_iv_root: Path
    audit_plots_dir: Path


def build_wflow_runtime(
    definition: LocationDefinition,
    *,
    workflow: str = "coupled",
    wflow_domain_review_required: bool | None = None,
    create_audit_dirs: bool = True,
) -> WflowRuntime | WflowCalibrationRuntime:
    """Build the Stage 1 Wflow runtime without re-reading location YAML."""

    if workflow not in {"coupled", "calibration"}:
        raise ValueError("workflow must be 'coupled' or 'calibration'")
    base = _build_coupled_runtime(definition, wflow_domain_review_required=wflow_domain_review_required)
    if workflow == "coupled":
        return base
    audit_plots_dir = base.location_root / "data/wflow/audit/plots"
    if create_audit_dirs:
        audit_plots_dir.mkdir(parents=True, exist_ok=True)
    return WflowCalibrationRuntime(
        **base.__dict__,
        streamflow_records_path=base.location_root / "data/sources/usgs_streamgages/streamflow_records.csv",
        event_streamflow_iv_root=base.location_root / "data/sources/usgs_streamgages/event_streamflow_iv",
        audit_plots_dir=audit_plots_dir,
    )


def _build_coupled_runtime(
    definition: LocationDefinition,
    *,
    wflow_domain_review_required: bool | None = None,
) -> WflowRuntime:
    root = definition.root
    config = deepcopy(definition.config)
    if wflow_domain_review_required is not None:
        config.setdefault("wflow", {}).setdefault("domain_set", {})["review_required"] = bool(
            wflow_domain_review_required
        )
    config.setdefault("scenario_run", {})
    wflow = config.get("wflow", {}) or {}
    readiness_validation = wflow.setdefault("readiness_validation", {})
    readiness_validation.setdefault(
        "report",
        f"data/sfincs/scenarios/{definition.name}_dynamic_handoff_readiness.csv",
    )
    readiness_validation.setdefault("decision", "required")

    sfincs_scenarios_root = root / "data/sfincs/scenarios"
    return WflowRuntime(
        definition=definition,
        location_root=root,
        location_name=definition.name,
        config=config,
        paths=_sfincs_paths(root, definition.name, config),
        design_paths=_design_paths(root, definition.name, config),
        runtime_config=config,
        sfincs_config=config,
        wflow_config={"wflow": wflow},
        sfincs_scenarios_root=sfincs_scenarios_root,
        scenario_catalog_path=root / "data/event_catalog/catalog/scenario_catalog.csv",
        probability_catalog_path=root / "data/event_catalog/catalog/probability_catalog.csv",
        readiness_path=sfincs_scenarios_root / f"{definition.name}_dynamic_handoff_readiness.csv",
        blocked_path=sfincs_scenarios_root / f"{definition.name}_blocked_dynamic_handoffs.csv",
        accepted_path=sfincs_scenarios_root / f"{definition.name}_accepted_dynamic_handoffs.csv",
        joint_worklist_path=sfincs_scenarios_root / f"{definition.name}_joint_wflow_sfincs_worklist.csv",
        incompatible_path=sfincs_scenarios_root / f"{definition.name}_incompatible_dynamic_handoffs.csv",
        events_root=_location_path(root, wflow.get("events_root", "data/wflow/events")),
        wflow_base_root=_location_path(root, wflow.get("base_model_root", "data/wflow/base")),
        wflow_handoff_manifest=_location_path(
            root,
            (wflow.get("handoff") or {}).get("manifest", "data/wflow/domain_set_handoff.yaml"),
        ),
    )


def _sfincs_paths(root: Path, name: str, config: dict[str, Any]) -> dict[str, Path | str]:
    path_cfg = config.get("paths") or {}
    location_data_root = _location_path(root, path_cfg.get("data_root", "data"))
    return {
        "location_name": name,
        "location_root": root,
        "location_data_root": location_data_root,
        "location_config_path": root / "config.yaml",
        "project_config_path": root / "config.yaml",
        "outputs_root": _location_path(root, path_cfg.get("sfincs_outputs_root", "data/sfincs")),
        "base_model_root": _location_path(root, path_cfg.get("base_model_root", "data/sfincs/base")),
        "scenarios_root": _location_path(root, path_cfg.get("scenarios_root", "data/sfincs/scenarios")),
        "storage_root": _location_path(root, path_cfg.get("storage_root", "data/sfincs/run_outputs")),
        "run_root": _location_path(root, path_cfg.get("run_root", "data/sfincs/run_stage")),
        "stats_root": _location_path(root, path_cfg.get("stats_root", "data/sfincs/stats")),
        "design_outputs_root": _location_path(root, path_cfg.get("design_outputs_root", "data/event_catalog")),
    }


def _design_paths(root: Path, name: str, config: dict[str, Any]) -> dict[str, Path | str]:
    outputs_root = _location_path(root, (config.get("paths") or {}).get("outputs_root", "data/event_catalog"))
    return {
        "location_name": name,
        "location_root": root,
        "outputs_root": outputs_root,
        "catalog_root": outputs_root / "catalog",
        "events_root": outputs_root / "events",
    }


def _location_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
