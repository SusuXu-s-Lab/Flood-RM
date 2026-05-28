"""Study-location workspace, power-grid AOI extraction, and flood-config helpers.

This module merges the former ``study_locations`` and ``study_area`` modules.
It is organised into four sections; keep edits inside the matching section:

  1. SHARED HELPERS   — path resolution and GeoJSON I/O used by both domains.
  2. POWER GRID       — readers that turn grid asset data into (lon, lat) points.
  3. STUDY AREA / AOI — concave-hull geometry built from grid coordinates and
                        consumed by the flood pipeline as the area of interest.
  4. FLOOD LOCATION   — workspace configuration, templates, and listing for the
                        flood study location (event drivers, paths, etc.).

The grid section produces points only. The AOI section is the single bridge
between domains: it consumes grid points and writes the GeoJSON that the flood
pipeline reads. The flood section never imports grid readers directly.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import shapely
import yaml
from shapely.geometry import MultiPoint, mapping, shape
from shapely.geometry.base import BaseGeometry


# ---------------------------------------------------------------------------
# 1. SHARED HELPERS
# ---------------------------------------------------------------------------

_LOCATION_RELATIVE_ROOTS: frozenset[str] = frozenset({"data", "02_flood", "01_grid"})


def _resolve_under_location(repo_root, location_name: str, value) -> Path:
    """Resolve a config path against a named location's workspace.

    Raises if ``value`` is None — used by code paths that require a path.
    """
    if value is None:
        raise ValueError("path value is required")
    path = Path(value)
    if path.is_absolute():
        return path
    location_root = Path(repo_root) / "locations" / location_name
    if path.parts and path.parts[0] in _LOCATION_RELATIVE_ROOTS:
        return location_root / path
    if path.parts[:2] == ("locations", location_name):
        return Path(repo_root) / path
    return Path(repo_root) / path


def _resolve_optional_under_root(repo_root, location_root, value) -> Path | None:
    """Resolve an optional config path against an already-known location root."""
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if len(path.parts) == 1 or path.parts[0] in _LOCATION_RELATIVE_ROOTS:
        return Path(location_root) / path
    return Path(repo_root) / path


def _read_geojson_geometry(path: Path) -> BaseGeometry:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features", [])
        if not features:
            raise ValueError(f"GeoJSON has no features: {path}")
        return shape(features[0]["geometry"])
    if payload.get("type") == "Feature":
        return shape(payload["geometry"])
    return shape(payload)


def _write_geojson(path: Path, geometry: BaseGeometry, properties: dict) -> None:
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": properties,
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. POWER GRID — coordinate readers
# ---------------------------------------------------------------------------
# These functions are the only place that knows about grid asset file formats.
# They produce plain (lon, lat) tuples; downstream code (the AOI section) is
# format-agnostic.

asset_coord_columns: tuple[tuple[str, str], ...] = (
    ("lon", "lat"),
    ("location_lon", "location_lat"),
    ("from_lon", "from_lat"),
    ("to_lon", "to_lat"),
)


def iter_asset_registry_points(asset_registry_dir: Path) -> Iterator[tuple[float, float]]:
    for path in sorted(Path(asset_registry_dir).glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                for lon_key, lat_key in asset_coord_columns:
                    lon = row.get(lon_key)
                    lat = row.get(lat_key)
                    if lon in (None, "") or lat in (None, ""):
                        continue
                    try:
                        yield float(lon), float(lat)
                    except ValueError:
                        continue


def iter_buscoords(smart_ds_root: Path) -> Iterator[tuple[float, float]]:
    for path in Path(smart_ds_root).rglob("Buscoords.dss"):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("!", "//")):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                yield float(parts[1]), float(parts[2])
            except ValueError:
                continue


# ---------------------------------------------------------------------------
# 3. STUDY AREA / AOI — geometry building from grid coordinates
# ---------------------------------------------------------------------------
# Builds the AOI polygon written to ``data/static/aoi/``. This is the single
# point of contact between the power-grid (section 2) and flood (section 4)
# domains.


@dataclass(frozen=True)
class StudyAreaResult:
    location_name: str
    source_format: str
    source_path: Path
    output_path: Path
    metadata_path: Path
    n_points: int
    bounds: tuple[float, float, float, float]


def build_study_area(config, repo_root) -> StudyAreaResult:
    location_name = str(config.get("project", {}).get("name", "")).strip()
    if not location_name:
        raise ValueError("project.name is required before building a study area")

    aoi = config.get("aoi") or {}
    source_format = str(aoi.get("source_format", "asset_registry")).strip()
    source_path = _resolve_under_location(repo_root, location_name, aoi.get("source"))
    output_path = _resolve_under_location(
        repo_root,
        location_name,
        aoi.get("output", "data/static/aoi/study_area.geojson"),
    )
    metadata_path = _resolve_under_location(
        repo_root,
        location_name,
        aoi.get("metadata_output", "data/static/aoi/study_area.json"),
    )

    if source_format == "asset_registry":
        points = list(iter_asset_registry_points(source_path))
        geometry = concave_study_area(points, alpha_ratio=float(aoi.get("alpha_ratio", 0.3)))
    elif source_format == "smart_ds_buscoords":
        points = list(iter_buscoords(source_path))
        geometry = concave_study_area(points, alpha_ratio=float(aoi.get("alpha_ratio", 0.05)))
    elif source_format == "geojson":
        points = []
        geometry = _read_geojson_geometry(source_path)
    else:
        raise ValueError(f"unsupported aoi.source_format: {source_format!r}")

    if source_format != "geojson" and not points:
        raise FileNotFoundError(f"no AOI source coordinates found under {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    _write_geojson(
        output_path,
        geometry,
        {
            "location_name": location_name,
            "source_format": source_format,
            "source_path": str(source_path),
            "n_points": len(points),
            "source": "power-grid coordinate concave hull",
        },
    )
    metadata = {
        "location_name": location_name,
        "source_format": source_format,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "n_points": len(points),
        "bounds": list(geometry.bounds),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return StudyAreaResult(
        location_name=location_name,
        source_format=source_format,
        source_path=source_path,
        output_path=output_path,
        metadata_path=metadata_path,
        n_points=len(points),
        bounds=tuple(float(v) for v in geometry.bounds),
    )


def study_area_bbox(config, repo_root, *, buffer_degrees=0.0) -> tuple[float, float, float, float]:
    location_name = str(config.get("project", {}).get("name", "")).strip()
    if not location_name:
        raise ValueError("project.name is required before reading a study-area bbox")

    source = config.get("grid_footprint", {}).get("source")
    if source is None:
        source = config.get("aoi", {}).get("output", "data/static/aoi/study_area.geojson")
    path = _resolve_under_location(repo_root, location_name, source)
    geometry = _read_geojson_geometry(path)
    west, south, east, north = (float(value) for value in geometry.bounds)
    buffer_degrees = float(buffer_degrees)
    return (
        west - buffer_degrees,
        south - buffer_degrees,
        east + buffer_degrees,
        north + buffer_degrees,
    )


def concave_study_area(
    points: Iterable[tuple[float, float]],
    *,
    alpha_ratio: float,
) -> BaseGeometry:
    coords = list(points)
    if len(coords) < 3:
        raise ValueError(f"need at least 3 points to build a study area; got {len(coords)}")
    if not (0.0 < alpha_ratio <= 1.0):
        raise ValueError(f"alpha_ratio must be in (0, 1]; got {alpha_ratio!r}")
    return shapely.concave_hull(MultiPoint(coords), ratio=alpha_ratio)


# ---------------------------------------------------------------------------
# 4. FLOOD LOCATION — workspace configuration
# ---------------------------------------------------------------------------
# Describes a flood study location (paths, event drivers, coastal flags) and
# builds the on-disk config.yaml template. Does not import grid readers; the
# only grid coupling is the AOI source path stored in the template.


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


_stage_order = (
    "define_location",
    "build_grid_dataset",
    "static_intake",
    "collect_sources",
    "build_event_catalog",
    "build_truth_set_base",
    "create_scenarios",
    "run_truth_set",
    "evaluate_truth_set",
)


@dataclass(frozen=True)
class LocationDefinition:
    name: str
    root: Path
    config_path: Path
    config: dict
    paths: dict
    grid: dict
    data_sources: dict
    sfincs: dict

    def stage_order(self) -> tuple[str, ...]:
        return _stage_order

    def validate(self) -> list[str]:
        issues = []
        if not self.name:
            issues.append("project.name is required")
        for key in ("data_sources", "grid", "sfincs"):
            if key not in self.config.get("includes", {}):
                issues.append(f"includes.{key} is required")
        return issues


def define_location(config_path) -> LocationDefinition:
    config_path = _resolve_location_config_path(config_path)
    base = _load_yaml_file(config_path)
    name = str(base.get("project", {}).get("name", "")).strip()
    if not name:
        raise ValueError("project.name is required in config.yaml")
    root = config_path.parent

    includes = base.get("includes") or {}
    data_sources = _load_location_detail(root, includes.get("data_sources"))
    grid = _load_location_detail(root, includes.get("grid"))
    sfincs = _load_location_detail(root, includes.get("sfincs"))

    config = _deep_merge(base, data_sources)
    config = _deep_merge(config, grid)
    config = _deep_merge(config, sfincs)
    config.pop("notebooks", None)

    return LocationDefinition(
        name=name,
        root=root,
        config_path=config_path,
        config=config,
        paths=config.get("paths", {}),
        grid=config.get("grid", {}),
        data_sources=data_sources,
        sfincs=sfincs,
    )


def _resolve_location_config_path(config_path) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    repo_root = _find_repo_root(Path.cwd())
    return (repo_root / path).resolve()


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    candidates = [current] if current.is_dir() else [current.parent]
    candidates.extend(candidates[0].parents)
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "locations").exists():
            return candidate
    raise FileNotFoundError("could not locate repo root")


def _load_yaml_file(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _load_location_detail(root: Path, value) -> dict:
    if value is None:
        raise ValueError("config.yaml must include data_sources.yaml, grid.yaml, and sfincs.yaml")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(path)
    return _load_yaml_file(path)


def _deep_merge(left: dict, right: dict) -> dict:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_study_location(config, repo_root):
    name = str(config.get("project", {}).get("name", "")).strip()
    if not name:
        raise ValueError("project.name is required in config.yaml")
    root = Path(repo_root) / "locations" / name
    flood_setting = str(config.get("flood_setting", "coastal")).strip() or "coastal"
    drivers = tuple(config.get("event_drivers") or _default_event_drivers(flood_setting))
    grid_footprint_source = _resolve_optional_under_root(
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


def list_study_locations(repo_root):
    locations_root = Path(repo_root) / "locations"
    if not locations_root.exists():
        return []
    return sorted(path.name for path in locations_root.iterdir() if path.is_dir())


def configured_study_locations(repo_root):
    return [
        name
        for name in list_study_locations(repo_root)
        if (Path(repo_root) / "locations" / name / "config.yaml").exists()
    ]


def _default_event_drivers(flood_setting):
    if flood_setting == "inland":
        return ("rainfall", "streamflow", "soil_moisture")
    return ("coastal_water_level", "rainfall", "soil_moisture")


def build_location_template(name, *, flood_setting="coastal"):
    name = str(name).strip()
    if not name:
        raise ValueError("location name is required")
    flood_setting = str(flood_setting).strip() or "coastal"
    drivers = list(_default_event_drivers(flood_setting))
    template = {
        "project": {
            "name": name,
            "reference_crs": "EPSG:4326",
        },
        "flood_setting": flood_setting,
        "event_drivers": drivers,
        "paths": {
            "data_root": f"locations/{name}/data",
        },
        "grid_footprint": {
            "source": "data/static/aoi/study_area.geojson",
        },
        "aoi": {
            "source": "data/static/power_grid/asset_registry",
            "source_format": "asset_registry",
            "alpha_ratio": 0.3,
            "output": "data/static/aoi/study_area.geojson",
            "metadata_output": "data/static/aoi/study_area.json",
        },
        "static_sources": {
            "bbox": {
                "output": "data/static/aoi/bbox.geojson",
            },
            "terrain": {
                "raw": "data/static/raw/topo/dem.tif",
                "output": "data/static/processed/dem_region_setup.tif",
            },
            "landcover": {
                "raw": "data/static/raw/landcover/landcover.tif",
                "output": "data/static/processed/landcover_region_setup.tif",
            },
            "coastline": {
                "output": "data/static/processed/coastal_region.geojson",
                "land_seed": [-70.0, 42.0],
                "ocean_seed": [-69.9, 42.0],
            },
            "ssurgo": {
                "output": "data/static/soils/ssurgo_mapunitpoly.gpkg",
                "attributes_output": "data/static/soils/ssurgo_mapunit_attributes.csv",
                "hsg_output": "data/static/soils/hsg.tif",
                "ksat_output": "data/static/soils/ksat_mmhr.tif",
            },
        },
        "collection": {
            "aorc_sst": {
                "catalog_id": name,
                "watershed": {
                    "id": f"{name}-grid-footprint",
                    "description": f"{name} grid footprint.",
                },
                "transposition_region": {
                    "id": f"{name}-transposition-region",
                    "geometry_file": "data/sources/aorc_sst/transposition_regions/transposition_region.geojson",
                    "description": "Study-location storm transposition region.",
                },
            },
            "nwm": {
                "streamflow": {
                    "available": "streamflow" in drivers,
                    "feature_ids": [],
                },
                "soil_moisture": {
                    "points": [],
                },
            },
        },
        "event_catalog": {
            "forcing_members": {
                "rainfall": f"locations/{name}/data/sources/aorc_sst/rainfall_members.csv",
                "streamflow": None,
                "soil_moisture": None,
            },
        },
    }
    if "coastal_water_level" in drivers:
        template["collection"]["boundary_water_level"] = {
            "source": "cora_or_coops",
        }
    return template


def coastal_waves_enabled(config_yaml_path) -> bool:
    # Selects the build path: true routes through quadtree + SnapWave + IG.
    path = Path(config_yaml_path)
    if not path.exists():
        return False
    data = yaml.safe_load(path.read_text()) or {}
    return bool(data.get("coastal_waves", False))
