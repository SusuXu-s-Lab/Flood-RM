"""Design-event runtime paths derived from one Study Location config."""

from __future__ import annotations
from pathlib import Path
from paths import find_repo_root, resolve_repo_path
from study_location import load_location_config, resolve_study_location

repo_root = find_repo_root(Path(__file__).resolve())

def load_runtime(path=None, scenario=None):
    config_path = None if path is None else resolve_repo_path(path, repo_root)
    config = load_location_config(config_path, repo_root)
    return config, build_paths(config, scenario=scenario)

def resolve_scenario(config, scenario=None):
    scenarios = config.get("scenarios") or {}
    name = str(scenario or "base").strip()
    if scenarios and name not in scenarios:
        raise ValueError(f"unknown scenario {name!r}; available: {sorted(scenarios.keys())}")
    spec = scenarios.get(name, {})
    out = {
        "name": name,
        "slr_offset_m": float(spec.get("slr_offset_m", 0.0)),
        "description": str(spec.get("description", "")),
    }
    for key in [
        "source",
        "source_url",
        "source_dataset",
        "source_location_basis",
        "source_baseline_year",
        "source_accessed",
        "scenario_family",
        "projection_year",
    ]:
        if key in spec:
            out[key] = spec[key]
    return out

def build_paths(config, scenario=None):
    project_name = str(config.get("project", {}).get("name", "")).strip()
    if not project_name:
        raise ValueError("project.name is required in config.yaml")
    location = resolve_study_location(config, repo_root)
    scenario_info = resolve_scenario(config, scenario)
    location_data_root = _location_data_root(config, location)
    sources_root = location_data_root / "sources"
    outputs_root = _location_or_absolute_path(
        location,
        config.get("paths", {}).get("outputs_root", "data/event_catalog"),
    )
    data_root = sources_root / "cora_waterlevel"
    cache_root = data_root / "cache"
    sfincs_boundary_value = config.get("sfincs", {}).get("boundary_file", "data/sfincs/base/sfincs.bnd")
    sfincs_boundary_file = Path(sfincs_boundary_value)
    if not sfincs_boundary_file.is_absolute():
        sfincs_boundary_file = _location_or_repo_path(location, sfincs_boundary_file)
    catalog_root = outputs_root / "catalog"
    source_artifacts_root = sources_root / "source_artifacts"
    aorc_sst_root = sources_root / "aorc_sst"
    nwm_root = sources_root / "nwm"
    usgs_streamgages_root = sources_root / "usgs_streamgages"
    era5_waves_root = sources_root / "era5_waves"
    hurdat2_root = sources_root / "hurdat2"
    era5_waves_output = config.get("collection", {}).get("era5_waves", {}).get("output_path")
    era5_waves_nc = (
        _location_or_absolute_path(location, era5_waves_output)
        if era5_waves_output
        else era5_waves_root / f"era5_{project_name}_offshore_hourly.nc"
    )
    cora_output = config.get("collection", {}).get("cora", {}).get("output_path")
    waterlevel_csv = (
        _location_or_absolute_path(location, cora_output)
        if cora_output
        else data_root / f"{project_name}_cora_boundary_hourly_msl.csv"
    )
    soil_moisture_spec = config.get("collection", {}).get("nwm", {}).get("soil_moisture", {})
    soil_moisture_points = soil_moisture_spec.get("points_file") or "data/static/aoi/nwm_soil_moisture_points.geojson"
    events_dirname = "events" if scenario_info["name"] == "base" else f"events_{scenario_info['name']}"
    events_root = outputs_root / events_dirname
    return {
        "repo_root": repo_root,
        "location_name": location.name,
        "location_root": location.root,
        "location_data_root": location_data_root,
        "location_config_path": location.config_path,
        "notebooks_root": location.notebooks_root,
        "root": outputs_root,
        "project_config_path": location.config_path,
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
        "nwm_soil_moisture_points_geojson": _location_or_repo_path(location, Path(soil_moisture_points)),
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
        "events_root": events_root,
        "template_bank_nc": events_root / "surge_template_bank.nc",
        "event_members_nc": events_root / "surge_event_members.nc",
        "event_summary_csv": events_root / "surge_event_members_summary.csv",
        "event_acceptance_json": events_root / "surge_event_members_acceptance.json",
        "event_overview_png": events_root / "surge_event_members_overview.png",
        "lagtimes_csv": events_root / "lagtimes.csv",
        "sfincs_boundary_file": sfincs_boundary_file,
        "scenario": scenario_info,
    }

def _location_data_root(config, location):
    value = config.get("paths", {}).get("data_root")
    if value is None:
        return location.data_root
    return _location_or_absolute_path(location, value)

def _location_or_absolute_path(location, value):
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts[:2] == ("locations", location.name):
        return repo_root / path
    return location.root / path

def _location_or_repo_path(location, path):
    if path.parts and path.parts[0] in {"data", "02_flood", "01_grid"}:
        return location.root / path
    return repo_root / path