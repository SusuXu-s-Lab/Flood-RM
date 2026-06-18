from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from study_location import study_area_bbox


@dataclass(frozen=True)
class RegionSetup:
    study_location: str
    bbox_wgs84: tuple[float, float, float, float]
    study_area_path: Path
    dem_raw: Path
    landcover_raw: Path
    dem_output: Path
    landcover_output: Path
    bbox_output: Path
    coastal_region_output: Path
    ssurgo_output: Path
    ssurgo_attributes_output: Path
    ssurgo_hsg_output: Path
    ssurgo_ksat_output: Path


def build_region_setup(config, paths, *, buffer_degrees=0.01) -> RegionSetup:
    location_name = str(paths.get("location_name") or config.get("project", {}).get("name", "")).strip()
    if not location_name:
        raise ValueError("project.name is required before building region setup")

    repo_root = _repo_root(paths)
    sources = config.get("static_sources", {})
    return RegionSetup(
        study_location=location_name,
        bbox_wgs84=study_area_bbox(config, repo_root, buffer_degrees=buffer_degrees),
        study_area_path=_location_path(paths, config.get("grid_footprint", {}).get("source")),
        dem_raw=_static_path(paths, sources, "terrain", "raw"),
        landcover_raw=_static_path(paths, sources, "landcover", "raw"),
        dem_output=_static_path(paths, sources, "terrain", "output"),
        landcover_output=_static_path(paths, sources, "landcover", "output"),
        bbox_output=_static_path(paths, sources, "bbox", "output"),
        coastal_region_output=_optional_static_path(
            paths,
            sources,
            "coastline",
            "output",
            "data/static/processed/coastal_region.geojson",
        ),
        ssurgo_output=_static_path(paths, sources, "ssurgo", "output"),
        ssurgo_attributes_output=_static_path(paths, sources, "ssurgo", "attributes_output"),
        ssurgo_hsg_output=_static_path(paths, sources, "ssurgo", "hsg_output"),
        ssurgo_ksat_output=_static_path(paths, sources, "ssurgo", "ksat_output"),
    )


def _static_path(paths, sources, name, key) -> Path:
    try:
        value = sources[name][key]
    except KeyError as exc:
        raise ValueError(f"static_sources.{name}.{key} is required") from exc
    return _location_path(paths, value)


def _optional_static_path(paths, sources, name, key, default) -> Path:
    value = sources.get(name, {}).get(key, default)
    return _location_path(paths, value)


def _location_path(paths, value) -> Path:
    if value is None:
        raise ValueError("path value is required")
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {"data", "02_flood", "01_grid"} and paths.get("location_root") is not None:
        return Path(paths["location_root"]) / path
    return _repo_root(paths) / path


def _repo_root(paths) -> Path:
    if paths.get("repo_root") is not None:
        return Path(paths["repo_root"])
    return Path(paths["root"]).parent
