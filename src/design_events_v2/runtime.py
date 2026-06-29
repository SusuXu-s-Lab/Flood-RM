"""Thin runtime_config adapter for ADR-0020 reference bundles.

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

from paths import find_repo_root
from study_location import define_location


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


__all__ = ["RuntimeConfigResult", "build_runtime_config"]
