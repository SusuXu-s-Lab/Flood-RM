from __future__ import annotations

from pathlib import Path
import sys

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
if (_SOURCE_ROOT / "study_location.py").exists():
    sys.path = [entry for entry in sys.path if entry != str(_SOURCE_ROOT)]
    sys.path.insert(0, str(_SOURCE_ROOT))

from paths import default_location_config_path, find_repo_root, resolve_repo_path
from study_location import (
    define_location,
    load_location_config,
    resolve_study_location,
)
from location_runtime import build_grid_paths as _shared_build_grid_paths
from sfincs_v2.runtime import SfincsRuntime, build_sfincs_runtime

repo_root = find_repo_root(Path(__file__).resolve())

def default_config_path():
    return default_location_config_path(repo_root)

project_config_path = None

def resolve_path(path):
    return resolve_repo_path(path, repo_root)

def load_config(path=None):
    return load_location_config(default_config_path() if path is None else resolve_path(path), repo_root)

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
        "project_config_path": location.config_path,
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

def _apply_inland_runtime_defaults(config):
    """Backfill optional config keys that inland notebooks read by bracket-subscript.

    These mirror the ``.get(..., default)`` defaults already used throughout
    ``wflow_runs``/``sfincs_runs`` so a location that doesn't spell out every optional key
    in YAML (greensboro, austin) still resolves in a notebook cell instead of raising
    ``KeyError``. Notebook-readable path and evaluation sections may be created in the
    runtime copy; location YAML is not mutated. The top-level ``wflow``/``sfincs_domain_set``
    sections are never created, so a coastal config stays coastal.
    """
    # static_sources: same defaults the region-setup/collect-sources runtimes apply.
    from sfincs_runs.build_base.static_intake import static_sources_with_defaults

    config["static_sources"] = static_sources_with_defaults(config)

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


def load_runtime(path=None):
    config = _apply_inland_runtime_defaults(load_config(path))
    return config, build_paths(config)

def load_sfincs_runtime(location_root, *, wave: bool = False, create_base_model_dir: bool = True) -> SfincsRuntime:
    """Load derived paths for SFINCS Location Workspace notebooks."""
    location_root = Path(location_root).resolve()
    return build_sfincs_runtime(
        define_location(location_root / "config.yaml"),
        wave=wave,
        create_base_model_dir=create_base_model_dir,
    )


def build_grid_paths(config):
    """Resolve the grid: section of config.yaml to absolute Path objects.

    Returns a dict with the same keys as the grid: block (power_extent,
    shift_cache, opendss_root, asset_registry, augmented_artifacts,
    onm_export, figures).
    """
    return _shared_build_grid_paths(config, repo_root=repo_root)


def parse_sfincs_inp(path):
    """Read a SFINCS ``sfincs.inp`` file as lowercase key/value strings."""
    path = Path(path)
    if not path.exists():
        return {}
    values = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip()
    return values


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
