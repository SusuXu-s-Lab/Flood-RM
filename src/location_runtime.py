"""Shared Location Workspace runtime projections for notebook adapters."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paths import default_location_config_path, find_repo_root, resolve_repo_path
from study_location import LocationDefinition, define_location, resolve_study_location


def load_runtime(path: str | Path | None = None, *, repo_root: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Path | str]]:
    """Load a Location Workspace config and legacy-compatible path projection.

    This is the non-model runtime helper for notebooks that need location and grid
    paths but should not depend on the SFINCS package name.
    """
    root = Path(repo_root) if repo_root is not None else find_repo_root()
    config_path = default_location_config_path(root) if path is None else resolve_repo_path(path, root)
    definition = define_location(config_path)
    config = apply_inland_runtime_defaults(deepcopy(definition.config))
    return config, build_sfincs_paths(definition.root, definition.name, config, repo_root=root)


def build_grid_paths(config: dict[str, Any], *, repo_root: str | Path | None = None) -> dict[str, Path]:
    """Resolve the grid section of a Location Workspace config to absolute paths."""
    root = Path(repo_root) if repo_root is not None else find_repo_root()
    location = resolve_study_location(config, root)
    grid_cfg = config.get("grid", {}) or {}

    def resolve(key: str, default: str) -> Path:
        raw = grid_cfg.get(key, default)
        path = Path(raw)
        return path if path.is_absolute() else (location.root / path).resolve()

    return {
        "power_extent": resolve("power_extent", "data/power_grid/power_extent.geojson"),
        "shift_cache": resolve("shift_cache", "data/power_grid/shift_cache"),
        "opendss_root": resolve("opendss_root", "data/power_grid/derived_opendss"),
        "asset_registry": resolve("asset_registry", "data/power_grid/asset_registry"),
        "augmented_artifacts": resolve("augmented_artifacts", "data/power_grid/augmented"),
        "onm_export": resolve("onm_export", "data/power_grid/onm_export"),
        "figures": resolve("figures", "data/power_grid/figures"),
    }


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
    base = _build_coupled_wflow_runtime(
        definition,
        wflow_domain_review_required=wflow_domain_review_required,
    )
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


def _build_coupled_wflow_runtime(
    definition: LocationDefinition,
    *,
    wflow_domain_review_required: bool | None = None,
) -> WflowRuntime:
    root = definition.root
    name = definition.name
    config = apply_inland_runtime_defaults(deepcopy(definition.config))
    if wflow_domain_review_required is not None:
        config.setdefault("wflow", {}).setdefault("domain_set", {})["review_required"] = bool(
            wflow_domain_review_required
        )
    config.setdefault("scenario_run", {})
    wflow = config.get("wflow", {}) or {}
    readiness_validation = wflow.setdefault("readiness_validation", {})
    readiness_validation.setdefault(
        "report",
        f"data/sfincs/scenarios/{name}_dynamic_handoff_readiness.csv",
    )
    readiness_validation.setdefault("decision", "required")

    sfincs_scenarios_root = root / "data/sfincs/scenarios"
    return WflowRuntime(
        definition=definition,
        location_root=root,
        location_name=name,
        config=config,
        paths=build_sfincs_paths(root, name, config),
        design_paths=build_design_paths(root, name, config),
        runtime_config=config,
        sfincs_config=config,
        wflow_config={"wflow": wflow},
        sfincs_scenarios_root=sfincs_scenarios_root,
        scenario_catalog_path=root / "data/event_catalog/catalog/scenario_catalog.csv",
        probability_catalog_path=root / "data/event_catalog/catalog/probability_catalog.csv",
        readiness_path=sfincs_scenarios_root / f"{name}_dynamic_handoff_readiness.csv",
        blocked_path=sfincs_scenarios_root / f"{name}_blocked_dynamic_handoffs.csv",
        accepted_path=sfincs_scenarios_root / f"{name}_accepted_dynamic_handoffs.csv",
        joint_worklist_path=sfincs_scenarios_root / f"{name}_joint_wflow_sfincs_worklist.csv",
        incompatible_path=sfincs_scenarios_root / f"{name}_incompatible_dynamic_handoffs.csv",
        events_root=location_path(root, wflow.get("events_root", "data/wflow/events"), location_name=name),
        wflow_base_root=location_path(root, wflow.get("base_model_root", "data/wflow/base"), location_name=name),
        wflow_handoff_manifest=location_path(
            root,
            (wflow.get("handoff") or {}).get("manifest", "data/wflow/domain_set_handoff.yaml"),
            location_name=name,
        ),
    )


def apply_inland_runtime_defaults(config: dict[str, Any]) -> dict[str, Any]:
    path_defaults = {
        "sfincs_outputs_root": "data/sfincs",
        "static_inputs_root": "data/static",
        "data_catalog": "data/static/data_catalogue.yaml",
        "static_root": "data/static/processed",
        "raw_root": "data/static/raw",
        "observations_root": "data/sources",
        "scenarios_root": "data/sfincs/scenarios",
        "storage_root": "data/sfincs/run_outputs",
        "run_root": "data/sfincs/run_stage",
        "stats_root": "data/sfincs/stats",
        "base_model_root": "data/sfincs/base",
        "design_outputs_root": "data/event_catalog",
    }
    path_config = config.setdefault("paths", {})
    for name, default in path_defaults.items():
        path_config.setdefault(name, default)

    evaluation = config.setdefault("evaluation", {})
    evaluation.setdefault("asset_source", "data/static/power_grid/smart_ds_compat/assets.parquet")
    evaluation.setdefault("output_root", "data/sfincs/evaluation")
    merge = evaluation.setdefault("multi_domain_merge", {})
    merge.setdefault("method", "max_depth_per_asset")
    merge.setdefault("retain_source_domain_id", True)
    merge.setdefault("write_overlap_diagnostics", True)

    sfincs_domain_set = config.get("sfincs_domain_set")
    if isinstance(sfincs_domain_set, dict):
        sfincs_domain_set.setdefault("domain_manifest", "data/sfincs/domains/domain_set.yaml")
        sfincs_domain_set.setdefault("event_catalog_scope", "shared_across_domain_set")
        sfincs_domain_set.setdefault("evaluation_merge", "max_depth_per_asset_with_source_domain")

    wflow = config.get("wflow")
    if isinstance(wflow, dict):
        wflow.setdefault("domain_set_manifest", "data/wflow/domain_set.yaml")
        domain_set = wflow.get("domain_set")
        if isinstance(domain_set, dict):
            domain_set.setdefault("event_catalog_scope", "shared_across_domain_set")
    return config


def repo_root_for_location(root: str | Path, location_name: str | None = None, *, fallback: str = "root") -> Path:
    root = Path(root)
    if root.parent.name == "locations" and (location_name is None or root.name == location_name):
        return root.parents[1]
    if fallback == "parent":
        return root.parent
    return root


def location_path(
    location_root: str | Path,
    value: str | Path,
    *,
    repo_root: str | Path | None = None,
    location_name: str | None = None,
) -> Path:
    root = Path(location_root)
    path = Path(value)
    if path.is_absolute():
        return path
    name = location_name or root.name
    if path.parts[:2] == ("locations", name):
        repo = Path(repo_root) if repo_root is not None else repo_root_for_location(root, name)
        return repo / path
    return root / path


def location_or_repo_path(
    location_root: str | Path,
    value: str | Path,
    *,
    repo_root: str | Path | None = None,
    location_name: str | None = None,
) -> Path:
    path = Path(value)
    if path.parts and path.parts[0] in {"data", "02_flood", "01_grid"}:
        return Path(location_root) / path
    repo = Path(repo_root) if repo_root is not None else repo_root_for_location(location_root, location_name)
    return repo / path


def build_sfincs_paths(
    location_root: str | Path,
    location_name: str,
    config: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Path | str]:
    root = Path(location_root)
    repo = Path(repo_root) if repo_root is not None else repo_root_for_location(root, location_name)
    path_cfg = config.get("paths") or {}
    location_data_root = location_path(root, path_cfg.get("data_root", "data"), repo_root=repo, location_name=location_name)
    return {
        "location_name": location_name,
        "location_root": root,
        "location_data_root": location_data_root,
        "repo_root": repo,
        "location_config_path": root / "config.yaml",
        "notebooks_root": root / "02_flood",
        "project_config_path": root / "config.yaml",
        "root": location_data_root / "sfincs",
        "outputs_root": location_path(root, path_cfg.get("sfincs_outputs_root", "data/sfincs"), repo_root=repo, location_name=location_name),
        "inputs_root": location_path(root, path_cfg.get("static_inputs_root", "data/static"), repo_root=repo, location_name=location_name),
        "data_catalog": location_path(root, path_cfg.get("data_catalog", "data/static/data_catalogue.yaml"), repo_root=repo, location_name=location_name),
        "static_root": location_path(root, path_cfg.get("static_root", "data/static/processed"), repo_root=repo, location_name=location_name),
        "raw_root": location_path(root, path_cfg.get("raw_root", "data/static/raw"), repo_root=repo, location_name=location_name),
        "observations_root": location_path(root, path_cfg.get("observations_root", "data/sources"), repo_root=repo, location_name=location_name),
        "base_model_root": location_path(root, path_cfg.get("base_model_root", "data/sfincs/base"), repo_root=repo, location_name=location_name),
        "scenarios_root": location_path(root, path_cfg.get("scenarios_root", "data/sfincs/scenarios"), repo_root=repo, location_name=location_name),
        "storage_root": location_path(root, path_cfg.get("storage_root", "data/sfincs/run_outputs"), repo_root=repo, location_name=location_name),
        "run_root": location_path(root, path_cfg.get("run_root", "data/sfincs/run_stage"), repo_root=repo, location_name=location_name),
        "stats_root": location_path(root, path_cfg.get("stats_root", "data/sfincs/stats"), repo_root=repo, location_name=location_name),
        "design_outputs_root": location_path(root, path_cfg.get("design_outputs_root", "data/event_catalog"), repo_root=repo, location_name=location_name),
    }


def build_design_paths(location_root: str | Path, location_name: str, config: dict[str, Any]) -> dict[str, Path | str]:
    root = Path(location_root)
    repo_root = repo_root_for_location(root, location_name)
    project_name = str((config.get("project") or {}).get("name", location_name)).strip()
    outputs_root = location_path(root, (config.get("paths") or {}).get("outputs_root", "data/event_catalog"), repo_root=repo_root, location_name=location_name)
    location_data_root = location_path(root, (config.get("paths") or {}).get("data_root", "data"), repo_root=repo_root, location_name=location_name)
    sources_root = location_data_root / "sources"
    data_root = sources_root / "cora_waterlevel"
    cache_root = data_root / "cache"
    catalog_root = outputs_root / "catalog"
    source_artifacts_root = sources_root / "source_artifacts"
    aorc_sst_root = sources_root / "aorc_sst"
    nwm_root = sources_root / "nwm"
    usgs_streamgages_root = sources_root / "usgs_streamgages"
    era5_waves_root = sources_root / "era5_waves"
    hurdat2_root = sources_root / "hurdat2"
    era5_waves_output = (config.get("collection") or {}).get("era5_waves", {}).get("output_path")
    era5_waves_nc = (
        location_path(root, era5_waves_output, repo_root=repo_root, location_name=location_name)
        if era5_waves_output
        else era5_waves_root / f"era5_{project_name}_offshore_hourly.nc"
    )
    cora_output = (config.get("collection") or {}).get("cora", {}).get("output_path")
    waterlevel_csv = (
        location_path(root, cora_output, repo_root=repo_root, location_name=location_name)
        if cora_output
        else data_root / f"{project_name}_cora_boundary_hourly_msl.csv"
    )
    soil_moisture_spec = (config.get("collection") or {}).get("nwm", {}).get("soil_moisture", {})
    soil_moisture_points = soil_moisture_spec.get("points_file") or "data/static/aoi/nwm_soil_moisture_points.geojson"
    sfincs_boundary_file = Path((config.get("sfincs") or {}).get("boundary_file", "data/sfincs/base/sfincs.bnd"))
    if not sfincs_boundary_file.is_absolute():
        sfincs_boundary_file = location_or_repo_path(root, sfincs_boundary_file, repo_root=repo_root, location_name=location_name)
    return {
        "repo_root": repo_root,
        "location_name": location_name,
        "location_root": root,
        "location_data_root": location_data_root,
        "location_config_path": root / "config.yaml",
        "notebooks_root": root / "02_flood",
        "root": outputs_root,
        "project_config_path": root / "config.yaml",
        "data_root": data_root,
        "outputs_root": outputs_root,
        "cache_root": cache_root,
        "waterlevel_csv": waterlevel_csv,
        "source_artifacts_root": source_artifacts_root,
        "usgs_streamgages_root": usgs_streamgages_root,
        "usgs_streamgage_candidates_geojson": usgs_streamgages_root / "streamgage_candidates.geojson",
        "usgs_streamgage_network_geojson": usgs_streamgages_root / "streamgage_network.geojson",
        "aorc_sst_root": aorc_sst_root,
        "aorc_sst_rainfall_members_csv": aorc_sst_root / "rainfall_members.csv",
        "cora_source_artifact_json": source_artifacts_root / "cora_boundary_water_level.json",
        "nwm_root": nwm_root,
        "nwm_streamflow_csv": nwm_root / "streamflow.csv",
        "nwm_soil_moisture_csv": nwm_root / "soil_moisture.csv",
        "nwm_soil_moisture_points_geojson": location_or_repo_path(root, soil_moisture_points, repo_root=repo_root, location_name=location_name),
        "era5_waves_root": era5_waves_root,
        "era5_waves_nc": era5_waves_nc,
        "hurdat2_root": hurdat2_root,
        "hurdat2_tracks_csv": hurdat2_root / "hurdat2_tracks.csv",
        "catalog_root": catalog_root,
        "historical_peaks_csv": catalog_root / "historical_peaks.csv",
        "marginal_params_csv": catalog_root / "marginal_params.csv",
        "marginal_rps_csv": catalog_root / "marginal_rps.csv",
        "marginal_plot_png": catalog_root / "marginal_fits.png",
        "sensitivity_csv": catalog_root / "threshold_model_sensitivity.csv",
        "marginal_rps_ci_csv": catalog_root / "marginal_rps_ci.csv",
        "marginal_bootstrap_json": catalog_root / "marginal_bootstrap.json",
        "stationarity_report_json": catalog_root / "stationarity_report.json",
        "sampled_peaks_csv": catalog_root / "sampled_peaks.csv",
        "event_catalog_csv": catalog_root / "event_catalog.csv",
        "event_catalog_audit_json": catalog_root / "event_catalog_audit.json",
        "resilience_stress_training_catalog_csv": catalog_root / "resilience_stress_training_catalog.csv",
        "event_distribution_summary_csv": catalog_root / "event_distribution_summary.csv",
        "event_distribution_summary_json": catalog_root / "event_distribution_summary.json",
        "event_distribution_plot_png": catalog_root / "event_distribution.png",
        "data_acquisition_readiness_json": outputs_root / "data_acquisition_readiness.json",
        "events_root": outputs_root / "events",
        "template_bank_nc": outputs_root / "events/surge_template_bank.nc",
        "event_members_nc": outputs_root / "events/surge_event_members.nc",
        "event_summary_csv": outputs_root / "events/surge_event_members_summary.csv",
        "event_acceptance_json": outputs_root / "events/surge_event_members_acceptance.json",
        "event_overview_png": outputs_root / "events/surge_event_members_overview.png",
        "lagtimes_csv": outputs_root / "events/lagtimes.csv",
        "sfincs_boundary_file": sfincs_boundary_file,
        "scenario": _resolve_design_scenario(config),
    }


def _resolve_design_scenario(config: dict[str, Any]) -> dict[str, Any]:
    scenario = ((config.get("scenarios") or {}).get("base") or {}).copy()
    return {
        "name": "base",
        "description": str(scenario.get("description", "")),
        "slr_offset_m": float(scenario.get("slr_offset_m", 0.0)),
    }
