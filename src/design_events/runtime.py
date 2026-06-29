"""Thin runtime_config adapter for reference bundles.

This module resolves Location Configuration and Source Artifact manifests into the
plain ``runtime_config`` consumed by ``workflow.build_reference_bundle``. It does
not collect data, plot, stage hydrodynamic forcing, or import production
``design_events`` modules.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import csv
import json
from pathlib import Path

from paths import find_repo_root, resolve_repo_path
from study_location import define_location, load_location_config, resolve_study_location

repo_root = find_repo_root(Path(__file__).resolve())


@dataclass(frozen=True)
class RuntimeConfigResult:
    runtime_config: dict
    source_artifacts: dict
    resolved_paths: dict
    checks: tuple[dict, ...]

    def for_bundle(self) -> dict:
        """Return ``runtime_config`` with runtime provenance attached for audit."""

        out = deepcopy(self.runtime_config)
        out["runtime_audit"] = {
            "source_artifacts": self.source_artifacts,
            "resolved_paths": self.resolved_paths,
            "checks": list(self.checks),
        }
        return out


def build_runtime_config(location_root, *, scenario="base", overrides=None) -> RuntimeConfigResult:
    """Build v2 ``runtime_config`` from one Location Workspace.

    Resolution order for paths is explicit config override, Source Artifact
    manifest artifact path, conventional location-relative fallback, then failure.
    """

    config_path = _config_path(location_root)
    definition = define_location(config_path)
    location = Path(definition.root)
    repo_root = _repo_root(location)
    config = deepcopy(definition.config)
    artifacts = _read_source_artifacts(location, config, repo_root)
    resolved_paths: dict[str, dict] = {}
    checks: list[dict] = []

    dependence = _dependence(config)
    records = _infer_records(config, artifacts, location, repo_root, resolved_paths)
    records = _merge_spec_overrides(
        records, dict(dependence.get("driver_records") or {}), resolved_paths, prefix="records"
    )
    _resolve_specs(records, location, repo_root, resolved_paths, prefix="records")

    member_libraries = _infer_member_libraries(config, artifacts, location, repo_root, resolved_paths)
    member_libraries = _merge_spec_overrides(
        member_libraries,
        dict((config.get("event_catalog") or {}).get("member_libraries") or {}),
        resolved_paths,
        prefix="member_libraries",
    )
    _resolve_specs(member_libraries, location, repo_root, resolved_paths, prefix="member_libraries")

    dependence = _normalize_dependence(config, dependence)
    runtime_config = {
        "location_root": str(location),
        "scenario_name": str(scenario or "base"),
        "event_family": _event_family(config),
        "inland_wflow_coupled": _is_inland(config),
        "records": records,
        "member_libraries": member_libraries,
        "dependence": dependence,
        "events": deepcopy(config.get("events") or {}),
        "sampling": deepcopy(config.get("sampling") or {}),
    }
    if overrides:
        runtime_config = _deep_merge(runtime_config, deepcopy(dict(overrides)))

    checks.extend(_validate_specs(runtime_config["records"], "records"))
    checks.extend(_validate_specs(runtime_config["member_libraries"], "member_libraries"))
    return RuntimeConfigResult(
        runtime_config=runtime_config,
        source_artifacts=artifacts,
        resolved_paths=resolved_paths,
        checks=tuple(checks),
    )


def _config_path(location_root):
    path = Path(location_root)
    return path if path.suffix in {".yaml", ".yml"} else path / "config.yaml"


def _repo_root(location: Path) -> Path:
    try:
        return find_repo_root(location)
    except FileNotFoundError:
        return location


def _source_artifacts_root(location: Path, config: dict, repo_root: Path) -> Path:
    data_root = Path((config.get("paths") or {}).get("data_root", "data"))
    if data_root.is_absolute():
        pass
    elif data_root.parts[:2] == ("locations", location.name):
        data_root = repo_root / data_root
    else:
        data_root = location / data_root
    return data_root / "sources" / "source_artifacts"


def _read_source_artifacts(location: Path, config: dict, repo_root: Path) -> dict:
    root = _source_artifacts_root(location, config, repo_root)
    artifacts = {}
    if not root.exists():
        return artifacts
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = f"{payload.get('source', path.stem)}_{payload.get('kind', '')}".strip("_")
        entry = {
            "manifest_path": str(path),
            "source": payload.get("source"),
            "kind": payload.get("kind"),
            "status": payload.get("status"),
            "start": payload.get("start"),
            "end": payload.get("end"),
            "artifacts": {
                name: str(_resolve_path(value, location, repo_root))
                for name, value in dict(payload.get("artifacts") or {}).items()
            },
            "metadata": payload.get("metadata") or {},
        }
        artifacts[key] = entry
    return artifacts


def _dependence(config: dict) -> dict:
    return deepcopy(((config.get("event_catalog") or {}).get("dependence") or {}))


def _normalize_dependence(config: dict, dependence: dict) -> dict:
    out = deepcopy(dependence)
    if _is_inland(config):
        out["driver_vector"] = ["rainfall"]
        out["condition_on"] = ["rainfall"]
        out.setdefault("cooccurrence", {})["condition_on"] = ["rainfall"]
        return out
    if not out.get("driver_vector"):
        drivers = [d for d in config.get("event_drivers", []) if d != "soil_moisture"]
        out["driver_vector"] = drivers or ["coastal_water_level", "rainfall"]
    return out


def _infer_records(config: dict, artifacts: dict, location: Path, repo_root: Path, resolved: dict) -> dict:
    out = {}
    rainfall_stats = _artifact(artifacts, "aorc_sst", "rainfall_catalog", "storm_stats_csv")
    if rainfall_stats:
        out["rainfall"] = _record(rainfall_stats, "rainfall_peak_time", "mean", "manifest", resolved, "records.rainfall.path")

    waterlevel = _artifact(artifacts, "cora", "boundary_water_level", "waterlevel_csv")
    if waterlevel:
        out["coastal_water_level"] = _record(
            waterlevel,
            "time",
            "value",
            "manifest",
            resolved,
            "records.coastal_water_level.path",
            transform="ntr",
            latitude=_coastal_latitude(config),
        )

    hydrologic = _artifact_entry(artifacts, "nwm", "retrospective_hydrologic_state")
    if hydrologic and hydrologic["artifacts"].get("soil_moisture_csv"):
        out["soil_moisture"] = _record(
            hydrologic["artifacts"]["soil_moisture_csv"],
            "time",
            "SOILSAT_TOP",
            "manifest",
            resolved,
            "records.soil_moisture.path",
            aggregate="mean",
        )

    streamflow = _conventional(location, "data/sources/usgs_streamgages/streamflow_records.csv", repo_root)
    if streamflow.exists():
        out["streamflow"] = _record(
            streamflow, "time", "discharge_cfs", "fallback", resolved, "records.streamflow.path", aggregate="max"
        )

    if not rainfall_stats:
        fallback = _conventional(
            location,
            f"data/sources/aorc_sst/{config.get('project', {}).get('name', location.name)}/72hr-events/storm-stats.csv",
            repo_root,
        )
        if fallback.exists():
            out["rainfall"] = _record(fallback, "rainfall_peak_time", "mean", "fallback", resolved, "records.rainfall.path")
    return out


def _infer_member_libraries(config: dict, artifacts: dict, location: Path, repo_root: Path, resolved: dict) -> dict:
    out = {}
    rainfall_members = _artifact(artifacts, "aorc_sst", "rainfall_catalog", "rainfall_members_csv")
    if rainfall_members:
        out["rainfall"] = _member(
            rainfall_members,
            "mean_precip_mm",
            "storm_start",
            "manifest",
            resolved,
            "member_libraries.rainfall.path",
            driver_role="stochastic",
        )
    elif (fallback := _conventional(location, "data/sources/aorc_sst/rainfall_members.csv", repo_root)).exists():
        out["rainfall"] = _member(
            fallback,
            "mean_precip_mm",
            "storm_start",
            "fallback",
            resolved,
            "member_libraries.rainfall.path",
            driver_role="stochastic",
        )
    return out


def _record(path, time_column, value_column, source, resolved, key, **extra) -> dict:
    resolved[key] = {"path": str(path), "resolution": source}
    spec = {"path": str(path), "time_column": time_column, "value_column": value_column}
    spec.update({k: v for k, v in extra.items() if v is not None})
    return spec


def _member(path, index_column, time_column, source, resolved, key, **extra) -> dict:
    resolved[key] = {"path": str(path), "resolution": source}
    spec = {"path": str(path), "index_column": index_column, "time_column": time_column}
    spec.update(extra)
    return spec


def _artifact(artifacts: dict, source: str, kind: str, name: str):
    entry = _artifact_entry(artifacts, source, kind)
    return None if entry is None else entry["artifacts"].get(name)


def _artifact_entry(artifacts: dict, source: str, kind: str):
    for entry in artifacts.values():
        if entry.get("source") == source and entry.get("kind") == kind:
            return entry
    return None


def _conventional(location: Path, value: str, repo_root: Path) -> Path:
    return _resolve_path(value, location, repo_root)


def _resolve_specs(specs: dict, location: Path, repo_root: Path, resolved: dict, *, prefix: str) -> None:
    for name, spec in specs.items():
        if "path" not in spec:
            continue
        key = f"{prefix}.{name}.path"
        path = _resolve_path(spec["path"], location, repo_root)
        spec["path"] = str(path)
        if key in resolved:
            resolved[key]["path"] = str(path)
        else:
            resolved[key] = {"path": str(path), "resolution": "config_override"}


def _resolve_path(value, location: Path, repo_root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    if path.parts[:2] == ("locations", location.name):
        return repo_root / path
    if path.parts and path.parts[0] == "data":
        return location / path
    candidate = repo_root / path
    return candidate if candidate.exists() else location / path


def _validate_specs(specs: dict, group: str) -> list[dict]:
    checks = []
    for name, spec in specs.items():
        path = Path(str(spec.get("path", "")))
        exists = path.exists()
        checks.append({"name": f"{group}.{name}.exists", "status": "pass" if exists else "fail", "path": str(path)})
        if exists and path.suffix.lower() == ".csv":
            required = [spec.get("time_column"), spec.get("value_column") or spec.get("index_column")]
            missing = [column for column in required if column and column not in _csv_header(path)]
            checks.append(
                {
                    "name": f"{group}.{name}.columns",
                    "status": "pass" if not missing else "fail",
                    "missing": missing,
                    "path": str(path),
                }
            )
    return checks


def _csv_header(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return set(next(csv.reader(stream), []))


def _event_family(config: dict) -> str:
    if _is_inland(config):
        return "inland_rainfall_wflow"
    if config.get("coastal_waves", False):
        return "coastal_compound_wave"
    return "coastal_compound"


def _is_inland(config: dict) -> bool:
    return str(config.get("flood_setting", "")).lower() == "inland"


def _coastal_latitude(config: dict):
    value = (((config.get("event_catalog") or {}).get("dependence") or {}).get("coastal_latitude"))
    if value is not None:
        return float(value)
    # Preserve the spec without inventing a site latitude if none is configured.
    return None


def _merge_spec_overrides(base: dict, override: dict, resolved: dict, *, prefix: str) -> dict:
    out = _deep_merge(base, override)
    for name, spec in override.items():
        if isinstance(spec, dict) and spec.get("path") is not None:
            resolved[f"{prefix}.{name}.path"] = {"path": str(spec["path"]), "resolution": "config_override"}
    return out


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out



# --------------------------------------------------------------------------------------
# Production path layout + Event Catalog Plan (relocated out of the legacy runtime and
# build_events.workflow modules).
# --------------------------------------------------------------------------------------


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


@dataclass(frozen=True)
class EventForcingPlan:
    name: str
    member_path: Path
    pairing_policy: dict


@dataclass(frozen=True)
class EventCatalogPlan:
    study_location: str
    scenario_name: str
    event_summary_csv: Path
    event_members_nc: Path
    event_catalog_csv: Path
    audit_json: Path | None
    forcings: tuple[EventForcingPlan, ...]
    required_forcings: tuple[str, ...]
    required_source_artifacts: tuple[str, ...]
    wave_analog_policy: str

    @property
    def forcing_names(self):
        return tuple(forcing.name for forcing in self.forcings)

    def forcing(self, name):
        for forcing in self.forcings:
            if forcing.name == name:
                return forcing
        raise KeyError(f"forcing is not configured: {name}")

    def summary_rows(self):
        return [
            {"item": "study_location", "value": self.study_location},
            {"item": "scenario_name", "value": self.scenario_name},
            {"item": "event_summary_csv", "value": self.event_summary_csv.as_posix()},
            {"item": "event_catalog_csv", "value": self.event_catalog_csv.as_posix()},
            {"item": "forcings", "value": ", ".join(self.forcing_names)},
            {"item": "wave_analog_policy", "value": self.wave_analog_policy},
        ]


forcing_order = ("rainfall", "streamflow", "soil_moisture")


def plan(config, paths):
    event_cfg = config.get("event_catalog", {})
    if "forcing_members" in event_cfg:
        member_paths = event_cfg.get("forcing_members", {})
    elif "event_catalog" not in config:
        member_paths = _member_paths_from_collection(config)
    else:
        member_paths = {}
    pairing = event_cfg.get("pairing", _pairing_from_collection(config))
    forcings = tuple(
        EventForcingPlan(
            name=name,
            member_path=_repo_path(paths, member_paths[name]),
            pairing_policy=dict(pairing.get(name, {})),
        )
        for name in forcing_order
        if member_paths.get(name) is not None
    )
    wave_analog_policy = "same_historical_analog" if config.get("coastal_waves", False) else "not_required"
    required_source_artifacts = ["event_summary", "event_members"]
    required_source_artifacts.extend(f"{forcing.name}_members" for forcing in forcings)
    if config.get("coastal_waves", False):
        required_source_artifacts.append("era5_waves")
    required_forcings = ("coastal", *tuple(forcing.name for forcing in forcings))
    return EventCatalogPlan(
        study_location=str(paths.get("location_name") or config.get("project", {}).get("name")),
        scenario_name=str(paths.get("scenario", {}).get("name", "base")),
        event_summary_csv=Path(paths["event_summary_csv"]),
        event_members_nc=Path(paths["event_members_nc"]),
        event_catalog_csv=Path(paths["event_catalog_csv"]),
        audit_json=None if paths.get("event_catalog_audit_json") is None else Path(paths["event_catalog_audit_json"]),
        forcings=forcings,
        required_forcings=required_forcings,
        required_source_artifacts=tuple(required_source_artifacts),
        wave_analog_policy=wave_analog_policy,
    )


def _repo_path(paths, value):
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {"data", "02_flood", "01_grid"} and paths.get("location_root") is not None:
        return Path(paths["location_root"]) / path
    return Path(paths["repo_root"]) / path


def _member_paths_from_collection(config):
    collection = config.get("collection", {})
    paths = {}
    if "aorc_sst" in collection:
        paths["rainfall"] = "data/sources/aorc_sst/rainfall_members.csv"
    if "nwm" in collection:
        paths["soil_moisture"] = "data/sources/nwm/soil_moisture.csv"
    if config.get("flood_setting") == "inland" and "usgs_streamgages" in collection:
        paths["streamflow"] = "data/sources/usgs_streamgages/streamflow_members.csv"
    return paths


def _pairing_from_collection(config):
    if config.get("flood_setting") == "inland":
        return {
            "rainfall": {
                "strategy": "inland_rainfall_pairing_priority",
                "same_storm_when_available": True,
                "fallback_strategy": "seasonal_window_permutation",
                "seed": 0,
                "window_days": 45,
            },
            "streamflow": {
                "strategy": "coherent_streamgage_network_event",
                "active_records_only": True,
                "allow_multiple_frequency_basis_gages": True,
                "design_event_method": "scaled_streamgage_network_analog",
            },
            "soil_moisture": {
                "strategy": "inland_antecedent_moisture_pairing",
                "rainfall_relative_when_coherent": True,
                "fallback_reference": "dominant_streamgage_network_peak",
                "lead_time_hours": 24,
            },
        }
    return {
        "rainfall": {"strategy": "seasonal_window_permutation", "seed": 0, "window_days": 45},
        "soil_moisture": {
            "strategy": "antecedent_to_forcing",
            "reference_forcing": "rainfall",
            "lead_time_hours": 24,
        },
    }

__all__ = ["RuntimeConfigResult", "build_runtime_config", "build_paths", "resolve_scenario", "load_runtime", "plan", "EventForcingPlan", "EventCatalogPlan"]
