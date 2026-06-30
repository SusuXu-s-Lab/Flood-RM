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
from location_runtime import (
    apply_inland_runtime_defaults,
    build_grid_paths as _shared_build_grid_paths,
    build_sfincs_paths,
    static_sources_with_defaults,
)
from sfincs_runs.io import parse_sfincs_inp
from sfincs_runs.runtime import SfincsRuntime, build_sfincs_runtime

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
    location = resolve_study_location(config, repo_root)
    return build_sfincs_paths(location.root, location.name, config, repo_root=repo_root)


def load_runtime(path=None):
    config = _apply_sfincs_runtime_defaults(load_config(path))
    return config, build_paths(config)


def _apply_sfincs_runtime_defaults(config):
    """Preserve legacy SFINCS notebook defaults while sharing generic runtime defaults."""
    config = apply_inland_runtime_defaults(config)
    config["static_sources"] = static_sources_with_defaults(config)
    return config

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
