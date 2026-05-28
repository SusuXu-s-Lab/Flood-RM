from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from design_events.collect_sources.era5_waves import era5_wave_short_variables, wave_dataset_covers


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
