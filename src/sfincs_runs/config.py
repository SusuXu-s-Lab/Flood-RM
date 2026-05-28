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
    candidates = [Path.cwd() / path, repo_root / path]
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


def build_paths(config=None):
    config = load_config() if config is None else config
    paths = config.get("paths", {})
    location = resolve_study_location(config, repo_root)
    location_name = location.name
    location_data_root = _location_data_root(config, location)
    sfincs_root = location_data_root / "sfincs"

    def root_path(name, default):
        value = Path(paths.get(name, default))
        if not value.is_absolute():
            value = _location_or_absolute_path(location, value)
        return value

    outputs_root = root_path("sfincs_outputs_root", "data/sfincs")
    inputs_root = root_path("static_inputs_root", "data/static")
    data_catalog = root_path("data_catalog", "data/static/data_catalogue.yaml")
    static_root = root_path("static_root", "data/static/processed")
    raw_root = root_path("raw_root", "data/static/raw")
    observations_root = root_path("observations_root", "data/sources")
    scenarios_root = root_path("scenarios_root", "data/sfincs/scenarios")
    storage_root = root_path("storage_root", "data/sfincs/run_outputs")
    run_root = root_path("run_root", "data/sfincs/run_stage")
    stats_root = root_path("stats_root", "data/sfincs/stats")
    base_model_root = root_path("base_model_root", "data/sfincs/base")
    design_outputs_root = root_path(
        "design_outputs_root",
        "data/event_catalog",
    )

    return {
        "root": sfincs_root,
        "repo_root": repo_root,
        "location_name": location.name,
        "location_root": location.root,
        "location_data_root": location_data_root,
        "location_config_path": location.config_path,
        "notebooks_root": location.notebooks_root,
        "project_config_path": default_config_path(),
        "outputs_root": outputs_root,
        "inputs_root": inputs_root,
        "data_catalog": data_catalog,
        "static_root": static_root,
        "raw_root": raw_root,
        "observations_root": observations_root,
        "base_model_root": base_model_root,
        "scenarios_root": scenarios_root,
        "storage_root": storage_root,
        "run_root": run_root,
        "stats_root": stats_root,
        "design_outputs_root": design_outputs_root,
    }


def load_runtime(path=None):
    config = load_config(path)
    return config, build_paths(config)


def build_grid_paths(config):
    """Resolve the grid: section of config.yaml to absolute Path objects.

    Returns a dict with the same keys as the grid: block (power_extent,
    shift_cache, opendss_root, asset_registry, augmented_artifacts,
    onm_export, figures).
    """
    location = resolve_study_location(config, repo_root)
    grid_cfg = config.get("grid", {})

    def _resolve(key, default):
        raw = grid_cfg.get(key, default)
        path = Path(raw)
        if path.is_absolute():
            return path
        # paths relative to the location root
        return (location.root / path).resolve()

    return {
        "power_extent": _resolve("power_extent", "data/power_grid/power_extent.geojson"),
        "shift_cache": _resolve("shift_cache", "data/power_grid/shift_cache"),
        "opendss_root": _resolve("opendss_root", "data/power_grid/derived_opendss"),
        "asset_registry": _resolve("asset_registry", "data/power_grid/asset_registry"),
        "augmented_artifacts": _resolve("augmented_artifacts", "data/power_grid/augmented"),
        "onm_export": _resolve("onm_export", "data/power_grid/onm_export"),
        "figures": _resolve("figures", "data/power_grid/figures"),
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
