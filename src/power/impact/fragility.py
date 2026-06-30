"""ERAD-derived flood-depth fragility curves for grid-asset notebooks."""
from functools import lru_cache
from pathlib import Path

from paths import default_location_config_path, find_repo_root
from power_v2.impact import (
    FloodDepthFragilityCurve,
    csv_failure_probability as _csv_failure_probability,
    erad_asset_type as _erad_asset_type,
    line_local_asset_type,
    load_asset_type_mapping as _load_asset_type_mapping,
    load_flood_depth_curves as _load_flood_depth_curves,
)
from study_location import define_location

project_root = Path(__file__).resolve().parents[3]
repo_root = find_repo_root(Path(__file__).resolve())


def power_grid_root():
    definition = define_location(default_location_config_path(repo_root))
    path = Path(definition.grid.get("power_grid_root", "data/power_grid"))
    return path if path.is_absolute() else definition.root / path


shared_fragility_dir = project_root / "artifacts" / "fragility"
default_curves_csv = shared_fragility_dir / "erad_flood_depth_curves.csv"
fragility_dir = power_grid_root() / "fragility"
default_mapping_csv = fragility_dir / "asset_type_mapping.csv"


@lru_cache(maxsize=None)
def load_flood_depth_curves(path=default_curves_csv):
    return _load_flood_depth_curves(path)


@lru_cache(maxsize=None)
def load_asset_type_mapping(path=default_mapping_csv):
    """Marshfield Asset Registry type -> ERAD asset type."""
    return _load_asset_type_mapping(path)


def erad_asset_type(local_asset_type, mapping=None):
    return _erad_asset_type(local_asset_type, mapping or load_asset_type_mapping())


def failure_probability(local_asset_type, depth_m, *, curves=None, mapping=None):
    """ERAD-derived flood failure probability for a Marshfield asset."""
    return _csv_failure_probability(
        local_asset_type,
        depth_m,
        curves=curves or load_flood_depth_curves(),
        mapping=mapping or load_asset_type_mapping(),
    )
