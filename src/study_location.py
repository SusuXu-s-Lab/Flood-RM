"""Cross-compatibility for Flood-RM workflows."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml
from aoi import build_study_area, study_area_bbox
from paths import (
    default_location_config_path,
    find_repo_root,
    resolve_optional_under_root,
    resolve_location_config_path as _resolve_location_config_path,
    resolve_repo_path,
)

def load_location_config(path=None, repo_root=None) -> dict:
    root = Path(repo_root) if repo_root is not None else find_repo_root()
    config_path = default_location_config_path(root) if path is None else resolve_repo_path(path, root)
    config = define_location(config_path).config
    config.setdefault("paths", {})
    return config

@dataclass(frozen=True)
class StudyLocation:
    name: str
    root: Path
    config_path: Path
    notebooks_root: Path
    data_root: Path
    flood_setting: str
    event_drivers: tuple[str, ...]
    grid_footprint_source: Path | None
    coastal_waves: bool

    @property
    def uses_coastal_water_level(self) -> bool:
        return "coastal_water_level" in self.event_drivers

    @property
    def is_configured(self) -> bool:
        return self.config_path.exists()

def resolve_study_location(config, repo_root):
    """Return the Study Location identity and key workspace paths for a config."""
    name = str(config.get("project", {}).get("name", "")).strip()
    if not name:
        raise ValueError("project.name is required in config.yaml")
    root = Path(repo_root) / "locations" / name
    flood_setting = str(config.get("flood_setting", "coastal")).strip() or "coastal"
    drivers = tuple(config.get("event_drivers") or ())
    grid_footprint_source = resolve_optional_under_root(
        repo_root,
        root,
        config.get("grid_footprint", {}).get("source"),
    )
    return StudyLocation(
        name=name,
        root=root,
        config_path=root / "config.yaml",
        notebooks_root=root / "02_flood",
        data_root=root / "data",
        flood_setting=flood_setting,
        event_drivers=drivers,
        grid_footprint_source=grid_footprint_source,
        coastal_waves=bool(config.get("coastal_waves", False)),
    )

@dataclass(frozen=True)
class LocationDefinition:
    name: str
    root: Path
    config_path: Path
    config: dict
    paths: dict
    grid: dict
    smartds: dict
    data_sources: dict
    sfincs: dict
    wflow: dict
    snapwave: dict
    model_recipes: dict

def define_location(config_path) -> LocationDefinition:
    config_path = _resolve_location_config_path(config_path)
    base = _load_yaml_file(config_path)
    name = str(base.get("project", {}).get("name", "")).strip()
    if not name:
        raise ValueError("project.name is required in config.yaml")
    root = config_path.parent
    includes = base.get("includes") or {}
    data_sources = _load_location_detail(root, includes.get("data_sources"), required=False)
    grid = _load_location_detail(root, includes.get("grid"), required="smartds" not in includes)
    smartds = _load_location_detail(root, includes.get("smartds"), required=False)
    sfincs = _load_location_detail(root, includes.get("sfincs"))
    wflow = _load_location_detail(root, includes.get("wflow"), required=False)
    snapwave = _load_location_detail(root, includes.get("snapwave"), required=False)
    model_recipes = _load_model_recipes(root, includes, sfincs=sfincs, wflow=wflow, snapwave=snapwave)
    config = {}
    # Legacy `extends:` remains supported as an explicit location-owned include.
    extends = base.get("extends")
    if extends is not None:
        config = _deep_merge(config, _load_extends_base(root, extends))
    config = _deep_merge(config, base)
    config = _deep_merge(config, data_sources)
    config = _deep_merge(config, smartds)
    config = _deep_merge(config, grid)
    config = _deep_merge(config, sfincs)
    config = _deep_merge(config, wflow)
    config = _deep_merge(config, snapwave)
    config.pop("notebooks", None)
    config.pop("extends", None)  # loader directive, not domain config
    config["_model_recipes"] = model_recipes
    return LocationDefinition(
        name=name,
        root=root,
        config_path=config_path,
        config=config,
        paths=config.get("paths", {}),
        grid=config.get("grid", {}),
        smartds=smartds,
        data_sources=data_sources,
        sfincs=sfincs,
        wflow=config.get("wflow", {}),
        snapwave=snapwave,
        model_recipes=model_recipes,
    )

def _load_yaml_file(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}

def _load_extends_base(root: Path, value) -> dict:
    path = Path(value)
    if not path.is_absolute():
        path = (root / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"extends base not found: {path}")
    return _load_yaml_file(path)

def _load_location_detail(root: Path, value, *, required=True) -> dict:
    if value is None:
        if not required:
            return {}
        raise ValueError("config.yaml must include grid.yaml and sfincs.yaml")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(path)
    return _load_yaml_file(path)

def _load_model_recipes(root: Path, includes: dict, *, sfincs: dict, wflow: dict, snapwave: dict) -> dict:
    recipes = _load_model_recipe_includes(root, includes)
    recipes.update(_hydromt_recipes_from_model_config("sfincs", sfincs))
    recipes.update(_hydromt_recipes_from_model_config("wflow", wflow))
    recipes.update(_hydromt_recipes_from_model_config("snapwave", snapwave))
    return recipes

def _load_model_recipe_includes(root: Path, includes: dict) -> dict:
    recipes = {}
    for key, value in includes.items():
        if key not in {
            "sfincs_build",
            "sfincs_build_waves",
            "sfincs_update_forcing",
            "wflow_build",
            "wflow_update_forcing",
        }:
            continue
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(path)
        recipes[key] = _load_yaml_file(path)
    return recipes

def _hydromt_recipes_from_model_config(model_name: str, config: dict) -> dict:
    hydromt = config.get("hydromt") or {}
    if not isinstance(hydromt, dict):
        return {}
    return {
        f"{model_name}_{purpose}": recipe
        for purpose, recipe in hydromt.items()
        if isinstance(recipe, dict)
    }

def _deep_merge(left: dict, right: dict) -> dict:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged