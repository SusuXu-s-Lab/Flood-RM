from __future__ import annotations
import os
from pathlib import Path
import sys
import yaml

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if (_SOURCE_ROOT / "study_location.py").exists():
    sys.path = [entry for entry in sys.path if entry != str(_SOURCE_ROOT)]
    sys.path.insert(0, str(_SOURCE_ROOT))

from study_location import define_location, resolve_study_location

def find_repo_root(start=None):
    current = Path(start).expanduser() if start is not None else Path.cwd()
    current = current.resolve()
    candidates = [current] if current.is_dir() else [current.parent]
    candidates.extend(candidates[0].parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "locations").exists():
            return candidate
    raise FileNotFoundError("could not locate repo root")

repo_root = find_repo_root(Path(__file__).resolve())

def default_config_path():
    configured = os.environ.get("FLOOD_RM_LOCATION_CONFIG")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else (repo_root / path).resolve()
    location_name = os.environ.get("FLOOD_RM_LOCATION", "marshfield")
    return repo_root / "locations" / location_name / "config.yaml"

project_config_path = default_config_path()

def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        repo_root / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (repo_root / path).resolve()

def load_yaml(path):
    path = resolve_path(path)
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return data or {}

def load_config(path=None):
    config_path = default_config_path() if path is None else resolve_path(path)
    config = define_location(config_path).config
    config.setdefault("paths", {})
    return config

def resolve_scenario(config, scenario=None):
    # MSL-shift scenarios are metadata over the events tier only. Catalog
    # and sampled peaks are scenario-independent. Resolve the scenario
    # name and offset once, here, so every caller sees the same dict.
    scenarios = config.get("scenarios") or {}
    name = str(scenario or "base").strip()
    if scenarios and name not in scenarios:
        raise ValueError(
            f"unknown scenario {name!r}; available: {sorted(scenarios.keys())}"
        )
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

def build_paths(config=None, scenario=None):
    config = load_config() if config is None else config
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
    nwm_soil_moisture_spec = config.get("collection", {}).get("nwm", {}).get("soil_moisture", {})
    nwm_soil_moisture_points_value = (
        nwm_soil_moisture_spec.get("points_file") or "data/static/aoi/nwm_soil_moisture_points.geojson"
    )
    nwm_soil_moisture_points_geojson = _location_or_repo_path(
        location, Path(nwm_soil_moisture_points_value)
    )
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
        "project_config_path": default_config_path(),
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
        "nwm_soil_moisture_points_geojson": nwm_soil_moisture_points_geojson,
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

def load_runtime(path=None, scenario=None):
    config = load_config(path)
    return config, build_paths(config, scenario=scenario)


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
