# Runtime configuration and path helpers

from __future__ import annotations
from pathlib import Path
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if (_SOURCE_ROOT / "study_location.py").exists():
    sys.path = [entry for entry in sys.path if entry != str(_SOURCE_ROOT)]
    sys.path.insert(0, str(_SOURCE_ROOT))

from study_location import (
    default_location_config_path,
    find_repo_root,
    load_location_config,
    load_yaml_document,
    resolve_repo_path,
    resolve_study_location,
)


repo_root = find_repo_root(Path(__file__).resolve())


def default_config_path():
    return default_location_config_path(repo_root)


project_config_path = default_config_path()


def resolve_path(path):
    return resolve_repo_path(path, repo_root)


def load_yaml(path):
    return load_yaml_document(path, repo_root)


def load_config(path=None):
    return load_location_config(default_config_path() if path is None else resolve_path(path), repo_root)

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


# Progress helpers

from contextlib import contextmanager


def iter_progress(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


class _NullProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, amount=1):
        return None

    def set_description_str(self, value, refresh=True):
        return None

    def set_postfix_str(self, value, refresh=True):
        return None


@contextmanager
def progress_bar(**kwargs):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        yield _NullProgress()
        return

    with tqdm(**kwargs) as bar:
        yield bar


# Source artifact manifests

import json
from pathlib import Path

import pandas as pd


def _timestamp(value):
    if value is None:
        return None
    return pd.Timestamp(value).isoformat()


def _relative_path(path, root):
    path = Path(path)
    root = Path(root)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def source_artifact_path(paths, source, kind):
    return paths["source_artifacts_root"] / f"{source}_{kind}.json"


def read_source_artifact(paths, source, kind):
    path = source_artifact_path(paths, source, kind)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def source_artifact_covers(paths, source, kind, start, end):
    manifest = read_source_artifact(paths, source, kind)
    if manifest is None:
        return False
    if manifest.get("status") != "complete":
        return False
    if manifest.get("metadata", {}).get("smoke") is True:
        return False
    artifact_start = manifest.get("start")
    artifact_end = manifest.get("end")
    if not artifact_start or not artifact_end:
        return False
    return pd.Timestamp(artifact_start) <= pd.Timestamp(start) and pd.Timestamp(artifact_end) >= pd.Timestamp(end)


def write_source_artifact(
    paths,
    source,
    kind,
    start=None,
    end=None,
    artifacts=None,
    metadata=None,
    status="complete",
):
    manifest = {
        "study_location": paths["location_name"],
        "source": source,
        "kind": kind,
        "status": status,
        "start": _timestamp(start),
        "end": _timestamp(end),
        "artifacts": {
            key: _relative_path(value, paths["repo_root"])
            for key, value in (artifacts or {}).items()
        },
        "metadata": metadata or {},
    }
    path = source_artifact_path(paths, source, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


# Data acquisition readiness checks

import json
from pathlib import Path

import pandas as pd

def _read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(path):
    path = Path(path)
    if not path.exists():
        return None
    return int(len(pd.read_csv(path)))


def _rainfall_members_csv(config, paths):
    return paths["aorc_sst_rainfall_members_csv"]


def _collection_window(config):
    collection = config.get("collection", {})
    if not collection.get("start") or not collection.get("end"):
        return None, None
    return pd.Timestamp(collection["start"]), pd.Timestamp(collection["end"])


def _record_window_issues(record, label, config, start_key="start", end_key="end"):
    expected_start, expected_end = _collection_window(config)
    if expected_start is None or expected_end is None or record is None:
        return []
    issues = []
    actual_start = record.get(start_key)
    actual_end = record.get(end_key)
    if not actual_start or not actual_end:
        issues.append(f"{label} does not declare production window")
        return issues
    actual_start = pd.Timestamp(actual_start)
    actual_end = pd.Timestamp(actual_end)
    if actual_start > expected_start or actual_end < expected_end:
        issues.append(
            f"{label} does not cover production window: "
            f"{actual_start.isoformat()} to {actual_end.isoformat()}, "
            f"expected {expected_start.isoformat()} to {expected_end.isoformat()}"
        )
    return issues


def _source_manifest(paths, filename):
    return _read_json(paths["source_artifacts_root"] / filename)


def _source_artifact_issues(config, paths, filename, label):
    manifest = _source_manifest(paths, filename)
    if manifest is None:
        return [f"missing complete {label} source artifact"]
    issues = []
    if manifest.get("status") != "complete":
        issues.append(f"{label} source artifact status is {manifest.get('status')!r}")
    if manifest.get("metadata", {}).get("smoke") is True:
        issues.append(f"{label} source artifact is smoke-limited")
    issues.extend(_record_window_issues(manifest, f"{label} source artifact", config))
    return issues


def check_aorc_sst_collection(config, paths):
    storms = config.get("collection", {}).get("aorc_sst", {})
    duration = int(storms.get("storm_duration_hours", 72))
    collection_dir = paths["aorc_sst_root"] / paths["location_name"] / f"{duration}hr-events"
    stats_rows = _csv_rows(collection_dir / "storm-stats.csv")
    ranked_rows = _csv_rows(collection_dir / "ranked-storms.csv")
    artifact = _source_manifest(paths, "aorc_sst_rainfall_catalog.json")
    issues = []
    if artifact is None:
        issues.append("missing direct AORC SST source artifact")
    elif artifact.get("status") != "complete":
        issues.append(f"direct AORC SST source artifact status is {artifact.get('status')!r}")
    else:
        issues.extend(_record_window_issues(artifact, "AORC SST source artifact", config))
    if not stats_rows:
        issues.append("missing or empty AORC SST storm-stats.csv")
    if not ranked_rows:
        issues.append("missing or empty AORC SST ranked-storms.csv")
    return {
        "id": "aorc_sst_collection",
        "passed": not issues,
        "issues": issues,
        "details": {
            "status": None if artifact is None else artifact.get("status"),
            "storm_stats_rows": stats_rows or 0,
            "ranked_storm_rows": ranked_rows or 0,
        },
    }


def check_rainfall_catalog(config, paths):
    rainfall_rows = _csv_rows(_rainfall_members_csv(config, paths))
    catalog_rows = _csv_rows(paths["event_catalog_csv"])
    audit = _read_json(paths["event_catalog_audit_json"])
    paired_rows = 0
    issues = []
    if not rainfall_rows:
        issues.append("missing or empty rainfall_members.csv")
    if not catalog_rows:
        issues.append("missing or empty event_catalog.csv")
    else:
        catalog = pd.read_csv(paths["event_catalog_csv"])
        required = ["rainfall_source", "rainfall_member_file", "rainfall_member_id"]
        missing_columns = [column for column in required if column not in catalog]
        if missing_columns:
            issues.append(f"event catalog missing rainfall columns: {missing_columns}")
        else:
            paired = catalog[required].notna().all(axis=1) & (catalog[required] != "").all(axis=1)
            paired_rows = int(paired.sum())
            if paired_rows != int(len(catalog)):
                issues.append("event catalog has unpaired rainfall rows")
    if audit is None:
        issues.append("missing event catalog audit")
    elif audit.get("passed") is not True:
        issues.append("event catalog audit did not pass")
    return {
        "id": "rainfall_catalog",
        "passed": not issues,
        "issues": issues,
        "details": {
            "rainfall_member_rows": rainfall_rows or 0,
            "event_catalog_rows": catalog_rows or 0,
            "paired_event_rows": paired_rows,
            "event_catalog_audit_passed": None if audit is None else bool(audit.get("passed")),
        },
    }


def check_rainfall_catalog_smoke(config, paths):
    return check_rainfall_catalog(config, paths)


def _source_manifest_complete(paths, filename):
    manifest = _source_manifest(paths, filename)
    return manifest is not None and manifest.get("status") == "complete"


def _manifest_status(paths, filename):
    manifest = _source_manifest(paths, filename)
    if manifest is None:
        return "missing"
    return str(manifest.get("status", "unknown"))


def _pairing_strategy(config, forcing, default="not_applicable"):
    policy = config.get("event_catalog", {}).get("pairing", {}).get(forcing, {})
    return str(policy.get("strategy", default))


def source_inventory_frame(config, paths):
    collection = config.get("collection", {})
    nwm = collection.get("nwm", {})
    streamflow = nwm.get("streamflow", {})
    rows = [
        {
            "driver": "coastal_water_level",
            "source": "CORA",
            "role": "coastal event-index marginal and hydrograph templates",
            "pairing_policy": "event_index",
            "status": _manifest_status(paths, "cora_boundary_water_level.json"),
        }
    ]
    if config.get("coastal_waves", False):
        rows.append(
            {
                "driver": "coastal_waves",
                "source": "ERA5 SnapWave boundary",
                "role": "wave forcing from the same historical coastal analog",
                "pairing_policy": "same_historical_analog",
                "status": _manifest_status(paths, "era5_snapwave_boundary_forcing.json"),
            }
        )
    if "aorc_sst" in collection:
        rows.append(
            {
                "driver": "rainfall",
                "source": "Direct AORC SST",
                "role": "stochastic storm transposition rainfall members",
                "pairing_policy": _pairing_strategy(config, "rainfall"),
                "status": _manifest_status(paths, "aorc_sst_rainfall_catalog.json"),
            }
        )
    rows.append(
        {
            "driver": "soil_moisture",
            "source": "NWM retrospective",
            "role": "antecedent hydrologic state for the paired rainfall member",
            "pairing_policy": _pairing_strategy(config, "soil_moisture"),
            "status": _manifest_status(paths, "nwm_retrospective_hydrologic_state.json"),
        }
    )
    if streamflow.get("available") is False:
        rows.append(
            {
                "driver": "streamflow",
                "source": "not used",
                "role": streamflow.get("reason", "not a configured boundary driver"),
                "pairing_policy": "not_required",
                "status": "not_required",
            }
        )
    return pd.DataFrame(rows)


def _repo_path(paths, value):
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {"data", "02_flood", "01_grid"} and paths.get("location_root") is not None:
        return Path(paths["location_root"]) / path
    return paths["repo_root"] / path


def _wave_output_path(config, paths):
    spec = config.get("collection", {}).get("era5_waves", {})
    return _repo_path(paths, spec.get("output_path")) or paths.get("era5_waves_nc")


def _wave_dataset_variables(path):
    import xarray as xr

    with xr.open_dataset(path) as ds:
        return list(ds.data_vars)


def check_wave_forcing(config, paths):
    from design_events.collect_sources.era5_waves import era5_wave_short_variables, wave_dataset_covers

    required = bool(config.get("coastal_waves", False))
    spec = config.get("collection", {}).get("era5_waves")
    issues = []
    variables = []
    if not required:
        return {
            "id": "wave_forcing",
            "passed": True,
            "issues": [],
            "details": {"required": False},
        }
    if not spec:
        issues.append("missing collection.era5_waves settings")
    elif not spec.get("bbox_wgs84"):
        issues.append("missing era5_waves.bbox_wgs84")
    issues.extend(
        _source_artifact_issues(
            config,
            paths,
            "era5_snapwave_boundary_forcing.json",
            "ERA5 wave",
        )
    )
    output_path = _wave_output_path(config, paths)
    if output_path is None:
        issues.append("missing ERA5 wave output path")
    elif not Path(output_path).exists():
        issues.append(f"missing ERA5 wave NetCDF: {output_path}")
    else:
        try:
            variables = _wave_dataset_variables(output_path)
        except Exception as exc:
            issues.append(f"could not open ERA5 wave NetCDF: {type(exc).__name__}: {exc}")
        else:
            missing = [name for name in era5_wave_short_variables if name not in variables]
            if missing:
                issues.append(f"ERA5 wave NetCDF missing variables: {missing}")
            expected_start, expected_end = _collection_window(config)
            if expected_start is not None and expected_end is not None:
                if not wave_dataset_covers(output_path, expected_start, expected_end):
                    issues.append("ERA5 wave NetCDF does not cover collection window")
    return {
        "id": "wave_forcing",
        "passed": not issues,
        "issues": issues,
        "details": {
            "required": True,
            "variables": [name for name in era5_wave_short_variables if name in variables],
            "wave_netcdf": None if output_path is None else str(output_path),
        },
    }


def check_wave_forcing_smoke(config, paths):
    return check_wave_forcing(config, paths)


def check_source_acquisition(config, paths):
    collection = config.get("collection", {})
    rainfall_backend = "aorc_sst" if "aorc_sst" in collection else "not_configured"
    nwm = collection.get("nwm", {})
    streamflow = nwm.get("streamflow", {})
    soil_moisture = nwm.get("soil_moisture", {})
    required_dirs = [
        paths["outputs_root"],
        paths["source_artifacts_root"],
        paths["nwm_root"],
    ]
    if rainfall_backend == "aorc_sst":
        required_dirs.append(paths["aorc_sst_root"])
    issues = []
    for directory in required_dirs:
        if not Path(directory).exists():
            issues.append(f"missing output directory: {directory}")
    issues.extend(_source_artifact_issues(config, paths, "cora_boundary_water_level.json", "CORA"))
    issues.extend(_source_artifact_issues(config, paths, "nwm_retrospective_hydrologic_state.json", "NWM"))
    if rainfall_backend == "not_configured":
        issues.append("AORC SST rainfall collection is not configured")
    if streamflow.get("available") is not False:
        issues.append("configured coastal no-streamflow exception must be explicit")
    if streamflow.get("feature_ids") not in ([], None):
        issues.append("configured coastal no-streamflow feature_ids must be empty")
    if not streamflow.get("reason"):
        issues.append("configured coastal no-streamflow exception needs a reason")
    soil_points = soil_moisture.get("points") or []
    if not soil_points:
        issues.append("NWM soil moisture points are not configured")
    return {
        "id": "source_acquisition",
        "passed": not issues,
        "issues": issues,
        "details": {
            "rainfall_backend": rainfall_backend,
            "streamflow_available": streamflow.get("available"),
            "soil_moisture_point_count": len(soil_points),
            "cora_manifest_complete": _source_manifest_complete(paths, "cora_boundary_water_level.json"),
            "nwm_manifest_complete": _source_manifest_complete(paths, "nwm_retrospective_hydrologic_state.json"),
        },
    }


def check_acquisition_dry_run(config, paths):
    return check_source_acquisition(config, paths)


def write_data_acquisition_readiness(config, paths):
    gates = [
        check_aorc_sst_collection(config, paths),
        check_rainfall_catalog(config, paths),
        check_source_acquisition(config, paths),
    ]
    if config.get("coastal_waves", False):
        gates.append(check_wave_forcing(config, paths))
    audit = {
        "study_location": paths["location_name"],
        "passed": all(gate["passed"] for gate in gates),
        "gates": gates,
    }
    path = Path(paths["data_acquisition_readiness_json"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit
