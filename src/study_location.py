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
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import shapely
import yaml
from shapely.geometry import MultiPoint, MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry


# ---------------------------------------------------------------------------
# 1. SHARED HELPERS
# ---------------------------------------------------------------------------

_LOCATION_RELATIVE_ROOTS: frozenset[str] = frozenset({"data", "02_flood", "01_grid"})


def find_repo_root(start=None) -> Path:
    return _find_repo_root(Path(start) if start is not None else Path.cwd())


def _location_name_from_cwd(root: Path) -> str | None:
    try:
        relative = Path.cwd().resolve().relative_to(root / "locations")
    except ValueError:
        return None
    return relative.parts[0] if relative.parts else None


def default_location_config_path(repo_root=None, environ=None) -> Path:
    root = Path(repo_root) if repo_root is not None else find_repo_root()
    environ = os.environ if environ is None else environ
    configured = environ.get("FLOOD_RM_LOCATION_CONFIG")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else (root / path).resolve()
    location_name = environ.get("FLOOD_RM_LOCATION") or _location_name_from_cwd(root)
    if not location_name:
        raise ValueError(
            "No Flood-RM location is selected. Set FLOOD_RM_LOCATION_CONFIG, "
            "set FLOOD_RM_LOCATION, or run from a locations/<name> workspace."
        )
    return root / "locations" / location_name / "config.yaml"


def resolve_repo_path(path, repo_root=None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    root = Path(repo_root) if repo_root is not None else find_repo_root()
    for candidate in (Path.cwd() / path, root / path):
        if candidate.exists():
            return candidate.resolve()
    return (root / path).resolve()


def load_location_config(path=None, repo_root=None) -> dict:
    root = Path(repo_root) if repo_root is not None else find_repo_root()
    config_path = default_location_config_path(root) if path is None else resolve_repo_path(path, root)
    config = define_location(config_path).config
    config.setdefault("paths", {})
    return config


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
    _write_geojson_features(path, [(geometry, properties)])


def _write_geojson_features(path: Path, features: Iterable[tuple[BaseGeometry, dict]]) -> None:
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": properties,
            }
            for geometry, properties in features
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
        yield from _iter_buscoords_file(path)


def iter_buscoords_by_subregion(smart_ds_root: Path) -> Iterator[tuple[str, list[tuple[float, float]]]]:
    """Yield ``(subregion_id, bus coordinates)`` per top-level SMART-DS subregion.

    The subregion id is the directory name (e.g. ``rural``, ``P4U``) so it is a
    stable SMART-DS identity that downstream domain selection keys on, rather than
    a positional label inferred from geometry.
    """
    for subregion in sorted(path for path in Path(smart_ds_root).iterdir() if path.is_dir()):
        points = list(_subregion_buscoords(subregion))
        if points:
            yield subregion.name, points


def _subregion_buscoords(subregion: Path) -> Iterator[tuple[float, float]]:
    """Yield a subregion's unique bus coordinates from its canonical export.

    Every timeseries scenario re-exports an identical ``Buscoords.dss``, so reading
    them all multiplies I/O by the scenario count (hundreds-to-thousands of files)
    without adding coordinates. The aggregate export sitting directly under an
    ``opendss*`` directory already lists every bus, so prefer those and fall back to
    the full tree only when none is present. Points are de-duplicated either way.
    """
    files = sorted(subregion.rglob("Buscoords.dss"))
    aggregates = [path for path in files if path.parent.name.startswith("opendss")]
    seen: set[tuple[float, float]] = set()
    for path in aggregates or files:
        for point in _iter_buscoords_file(path):
            if point not in seen:
                seen.add(point)
                yield point


def _iter_buscoords_file(path: Path) -> Iterator[tuple[float, float]]:
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
        feature_rows = None
    elif source_format == "smart_ds_buscoords":
        alpha_ratio = float(aoi.get("alpha_ratio", 0.05))
        if aoi.get("preserve_disconnected_subregions"):
            grouped_points = list(iter_buscoords_by_subregion(source_path))
            points = [point for _, group in grouped_points for point in group]
            feature_rows = [
                (
                    concave_study_area(group, alpha_ratio=alpha_ratio),
                    {
                        "location_name": location_name,
                        "source_format": source_format,
                        "source_path": str(source_path),
                        "n_points": len(group),
                        "source": "SMART-DS subregion Buscoords concave hull",
                        "subregion_id": subregion,
                    },
                )
                for subregion, group in grouped_points
            ]
            geometry = shapely.union_all([geom for geom, _ in feature_rows])
        else:
            points = list(iter_buscoords(source_path))
            geometry = concave_study_area(points, alpha_ratio=alpha_ratio)
            feature_rows = None
    elif source_format == "geojson":
        points = []
        geometry = _read_geojson_geometry(source_path)
        if aoi.get("preserve_disconnected_subregions"):
            geometry = preserve_disconnected_subregions(
                geometry,
                max_bridge_edge_degrees=float(aoi.get("subregion_bridge_max_edge_degrees", 0.03)),
            )
        feature_rows = None
    else:
        raise ValueError(f"unsupported aoi.source_format: {source_format!r}")

    if source_format != "geojson" and not points:
        raise FileNotFoundError(f"no AOI source coordinates found under {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if feature_rows:
        _write_geojson_features(output_path, feature_rows)
    else:
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
        "subregion_count": len(feature_rows) if feature_rows else subregion_count(geometry),
        "subregion_ids": [props["subregion_id"] for _, props in feature_rows] if feature_rows else [],
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


def preserve_disconnected_subregions(
    geometry: BaseGeometry,
    *,
    max_bridge_edge_degrees: float,
) -> BaseGeometry:
    """Split a bridged concave hull back into disconnected source subregions.

    SMART-DS regional footprints can become one Polygon when a concave hull
    draws two long artificial exterior edges between otherwise separate
    subregions. When exactly two over-threshold exterior edges are present,
    the two exterior arcs define the original components.
    """
    if geometry.geom_type != "Polygon":
        return geometry
    coords = list(geometry.exterior.coords)
    if len(coords) < 6:
        return geometry
    bridge_edges = [
        index
        for index, (left, right) in enumerate(zip(coords, coords[1:]))
        if _segment_length(left, right) > max_bridge_edge_degrees
    ]
    if len(bridge_edges) != 2:
        return geometry

    first, second = sorted(bridge_edges)
    rings = [
        coords[first + 1 : second + 1] + [coords[first + 1]],
        coords[second + 1 : -1] + coords[: first + 1] + [coords[second + 1]],
    ]
    polygons = []
    for ring in rings:
        polygon = Polygon(ring)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty and polygon.area > 0:
            polygons.append(polygon)
    if len(polygons) != 2:
        return geometry
    return MultiPolygon(sorted(polygons, key=lambda item: (item.bounds[0], item.bounds[1])))


def subregion_count(geometry: BaseGeometry) -> int:
    if geometry.geom_type == "MultiPolygon":
        return len(geometry.geoms)
    return 1


def _segment_length(left, right) -> float:
    return ((float(right[0]) - float(left[0])) ** 2 + (float(right[1]) - float(left[1])) ** 2) ** 0.5


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


_NWM_SOIL_MOISTURE_DEFAULTS = {
    "zarr": "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/ldasout.zarr",
    "variables": ["SOIL_M", "SOILSAT_TOP"],
    "soilsat_top_layers": [0, 1],
    "x": "x",
    "y": "y",
    "crs": (
        "+proj=lcc +units=m +a=6370000.0 +b=6370000.0 +lat_1=30.0 "
        "+lat_2=60.0 +lat_0=40.0 +lon_0=-97.0 +x_0=0 +y_0=0 "
        "+k_0=1.0 +nadgrids=@null +wktext +no_defs"
    ),
    "points_file": "data/static/aoi/nwm_soil_moisture_points.geojson",
    "selection_method": "footprint_centroid_corners_and_edges_snapped_to_nearest_nwm_cell",
}


_AORC_SST_DEFAULTS = {
    "zarr_year_pattern": "s3://noaa-nws-aorc-v1-1-1km/{year}.zarr",
    "variable": "APCP_surface",
    "start_date": "1979-02-01",
    "end_date": "2022-12-31",
    "storm_duration_hours": 72,
    # Threshold-driven POT: keep every independent storm above min_precip_threshold
    # (footprint-mean mm over the storm window). Member count is data-driven; set the
    # threshold per location from its rainfall POT diagnostics. top_n_events is an
    # optional safety cap only (None = keep all exceedances).
    "min_precip_threshold": 2.5,
    "top_n_events": None,
    "check_every_n_hours": 6,
    "decluster_hours": 72,
    "transposition_stride_cells": 4,
    "max_open_year_datasets": 4,
    "write_event_windows": True,
}


# SFINCS curve-number Infiltration Treatment constants shared by every Study
# Location and both flood settings. These are unit/tuning constants, not
# per-location choices, so they live here once instead of being copy-pasted into
# each location's sfincs.yaml. Per-location infiltration keys (the soil/landcover
# raster paths, `effective_source`, `review_required`) stay in sfincs.yaml and
# are deep-merged on top of these and can be inspected with the resolved-config
# script when a disposable merged view is useful.
_INFILTRATION_CONSTANTS = {
    "ksat_scale_factor": 0.1,
    "ksat_max_mmhr": 75.0,
    "factor_ksat": 0.2777777777777778,
    "block_size": 2000,
    "reclass_table": None,
}


def _common_methodology_defaults() -> dict:
    return {
        "paths": {
            "outputs_root": "data/event_catalog",
            "sfincs_outputs_root": "data/sfincs",
            "static_inputs_root": "data/static",
            "data_catalog": "data/static/data_catalogue.yaml",
            "static_root": "data/static/processed",
            "raw_root": "data/static/raw",
            "observations_root": "data/sources",
            "base_model_root": "data/sfincs/base",
            "scenarios_root": "data/sfincs/scenarios",
            "storage_root": "data/sfincs/run_outputs",
            "run_root": "data/sfincs/run_stage",
            "stats_root": "data/sfincs/stats",
            "design_outputs_root": "data/event_catalog",
        },
        "grid_footprint": {
            "source": "data/static/aoi/study_area.geojson",
        },
        "static_sources": {
            "bbox": {
                "output": "data/static/aoi/bbox.geojson",
            },
            "ssurgo": {
                "output": "data/static/soils/ssurgo_mapunitpoly.gpkg",
                "attributes_output": "data/static/soils/ssurgo_mapunit_attributes.csv",
            },
        },
        "collection": {
            "start": "1979-02-01",
            "end": "2022-12-31",
        },
        "sfincs": {
            "boundary_file": "data/sfincs/base/sfincs.bnd",
        },
        "scenario_build": {
            "design_scenario": "base",
            "forcing_variable": "auto",
            "tref": "2000-01-01 00:00:00",
        },
        "scenario_run": {
            "workers": 1,
            "sfincs_bin": "/usr/local/bin/sfincs",
            "sfincs_bin_env": "SFINCS_BIN",
        },
        "scenario_stats": {
            "workers": 4,
            "land_threshold_m": 0.0,
            "huthresh_m": 0.01,
            "impact_threshold_m": 0.1,
        },
    }


def _inland_methodology_defaults() -> dict:
    defaults = _common_methodology_defaults()
    defaults["paths"]["evaluation_root"] = "data/evaluation"
    defaults["static_sources"].update(
        {
            "wflow_collection_extent": {
                "method": "reviewed_nhdplus_watersheds",
                "padding_degrees": 0.02,
                "source_fabric": "data/wflow/domain_set_subbasins.gpkg",
                "source_catchments": "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg",
                "watersheds": "data/static/aoi/wflow_nhdplus_watersheds.geojson",
                "boundary": "data/static/aoi/wflow_collection_region.geojson",
                "terrain_raw": "data/wflow/static/raw/topo/dem_wflow.tif",
                "terrain_output": "data/wflow/static/processed/dem_wflow_coarse.tif",
                "terrain_resolution_degrees": 0.0009,
                "landcover_raw": "data/wflow/static/raw/landcover/landcover_wflow.tif",
                "landcover_output": "data/wflow/static/processed/landcover_wflow_coarse.tif",
                "ssurgo_output": "data/wflow/static/soils/ssurgo_mapunitpoly_wflow.gpkg",
                "ssurgo_attributes_output": "data/wflow/static/soils/ssurgo_mapunit_attributes_wflow.csv",
                "hsg_output": "data/wflow/static/soils/hsg_wflow.tif",
                "ksat_output": "data/wflow/static/soils/ksat_mmhr_wflow.tif",
            },
            "terrain": {
                "raw": "data/static/raw/topo/dem.tif",
                "output": "data/static/processed/dem_region_setup.tif",
            },
            "landcover": {
                "raw": "data/static/raw/landcover/landcover.tif",
                "output": "data/static/processed/landcover_region_setup.tif",
            },
        }
    )
    defaults["collection"].update(
        {
            "usgs_streamgages": {
                "candidate_output": "data/sources/usgs_streamgages/streamgage_candidates.geojson",
                "reviewed_network": "data/sources/usgs_streamgages/streamgage_network.geojson",
                "active_records_only": True,
                "exclude_inactive_gages": True,
                "review_required": True,
                "accept_unreviewed_streamgage_network": False,
                "scoring": {
                    "prefer_long_period_of_record": True,
                    "prefer_complete_record": True,
                    "prefer_drainage_area_match": True,
                    "prefer_sfincs_handoff_reaches": True,
                },
                "roles": ["frequency", "calibration", "validation", "sfincs_handoff"],
                "discovery": {
                    "service": "nwis",
                    "parameter_cd": "00060",
                    "has_data_type_cd": "dv",
                    "site_status": "active",
                    "search_geometry": "data/static/aoi/wflow_nhdplus_watersheds.geojson",
                    "hydrologic_buffer_km": 50,
                },
                "streamflow_records": {
                    "collect": False,
                    "output": "data/sources/usgs_streamgages/streamflow_records.csv",
                    "service": "dv",
                    "stat_cd": "00003",
                },
            },
            "national_hydrography": {
                "service": "usgs_3dhp_or_nhdplus_hr",
                "hydromt_basemap": "data/wflow/hydrography/us_hydrography_basemap.nc",
                "basemap_source_resolution_degrees": 0.0009,
                "river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg",
                "catchments": "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg",
                "collect_review_vectors": True,
                "wflow_soil_parameters": "data/wflow/static/ssurgo_wflow_soil_parameters.nc",
            },
            "aorc_sst": _deep_merge(
                _AORC_SST_DEFAULTS,
                {"transposition_region": {"buffer_km": 50}},
            ),
            "nwm": {
                "version": "3.0",
                "bucket": "noaa-nwm-retrospective-3-0-pds",
                "start": "1979-02-01",
                "end": "2022-12-31",
                "streamflow": {
                    "available": False,
                    "feature_ids": [],
                },
                "soil_moisture": _deep_merge(
                    _NWM_SOIL_MOISTURE_DEFAULTS,
                    {"points_source": "data/static/aoi/evaluation_footprint.geojson"},
                ),
            },
        }
    )
    defaults.update(
        {
            "event_catalog": {
                "forcing_members": {
                    "rainfall": "data/sources/aorc_sst/rainfall_members.csv",
                    "streamflow": "data/sources/usgs_streamgages/streamflow_members.csv",
                    "soil_moisture": "data/sources/nwm/soil_moisture.csv",
                },
                "dependence": {
                    # ADR-0016: inland design driver is rainfall (+ antecedent moisture as a
                    # conditioning attribute); discharge is the Wflow response, not a copula
                    # dimension. Streamflow POT is a calibration/validation anchor only.
                    "method": "rainfall_marginal_with_antecedent_moisture",
                    "driver_vector": ["rainfall"],
                    "primary_driver": "rainfall",
                    "event_rate_per_year": 5.0,
                    "copula_seed": 2,
                    "pool_size": 100000,
                    "enforce_stress_budget": True,
                    "catalog_band_fractions": {
                        "mild": 0.05,
                        "common": 0.28,
                        "significant": 0.28,
                        "rare": 0.12,
                        "extreme": 0.27,
                    },
                    "cooccurrence": {
                        "threshold_quantile": 0.95,
                        "target_rate_per_year": 5.0,
                        "condition_on": ["streamflow"],
                        "decluster_window_hours": 120,
                        "pairing_window_hours": 72,
                    },
                    "marginals": {
                        "rainfall": {"kind": "pot"},
                        "streamflow": {"kind": "pot"},
                    },
                    "driver_records": {
                        "rainfall": {
                            "time_column": "storm_date",
                            "value_column": "mean",
                        },
                        "soil_moisture": {
                            "path": "data/sources/nwm/soil_moisture.csv",
                            "time_column": "time",
                            "value_column": "SOILSAT_TOP",
                            "aggregate": "mean",
                        },
                        "streamflow": {
                            "path": "data/sources/usgs_streamgages/streamflow_records.csv",
                            "time_column": "time",
                            "value_column": "discharge_cfs",
                            "group_column": "site_no",
                            "aggregate": "max",
                        },
                    },
                    "member_libraries": {
                        "streamflow": {"from": "member_table"},
                        "rainfall": {"from": "member_table"},
                    },
                },
                "pairing": {
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
                },
            },
            "extremes": {
                "method": "pot",
                "hydrological_year_start": "YS-OCT",
                "selection_criterion": "AIC",
                "return_periods": [2, 5, 10, 25, 50, 100, 250, 500],
                "pot": {
                    "threshold_quantile": 0.98,
                    "distributions": ["exp", "gpd"],
                    "min_peak_distance_hours": 72,
                },
                "bootstrap": {
                    "n_replicates": 1000,
                    "confidence_level": 0.95,
                    "seed": 0,
                },
            },
            "sampling": {
                "spacing": "log",
                "return_period_min_years": 1.5,
                "return_period_max_years": 500.0,
                "hybrid_splice_quantile": 0.95,
                "tail_sample_fraction": 0.05,
                "severity_bands": [
                    {"severity_band": "mild", "rp_min_years": 0.0, "rp_max_years": 2.0},
                    {"severity_band": "common", "rp_min_years": 2.0, "rp_max_years": 10.0},
                    {"severity_band": "significant", "rp_min_years": 10.0, "rp_max_years": 50.0},
                    {"severity_band": "rare", "rp_min_years": 50.0, "rp_max_years": 100.0},
                    {"severity_band": "extreme", "rp_min_years": 100.0, "rp_max_years": 500.0},
                    {"severity_band": "beyond_design", "rp_min_years": 500.0, "rp_max_years": None},
                ],
            },
            "sfincs": _deep_merge(
                defaults["sfincs"],
                {
                    "grid_resolution_m": 100,
                    "outflow_boundary_elevation_quantile": 0.05,
                },
            ),
            "sfincs_domain_set": {
                "enabled": True,
                "review_required": True,
                "domains_root": "data/sfincs/domains",
                "domain_manifest": "data/sfincs/domains/domain_set.yaml",
                "source": "data/static/aoi/evaluation_footprint.geojson",
                "allow_multiple_domains": True,
                "region_geometry": "bounding_box",
                "event_catalog_scope": "shared_across_domain_set",
                "evaluation_merge": "max_depth_per_asset_with_source_domain",
                "domains": [],
            },
            "inland_coupling": {
                "enabled": True,
                "forcing_mode": "dual_fluvial_pluvial",
                "streamflow_reference_time": "dominant_streamgage_network_peak",
                "rainfall_member_scope": "shared_between_wflow_and_sfincs",
                "direct_rainfall": {
                    "enabled": True,
                    "variable": "APCP_surface",
                    "cumulative_input": True,
                    "time_label": "right",
                    "buffer_m": 30000,
                },
                "discharge_forcing": {
                    "source": "wflow",
                    "handoff_manifest": "data/wflow/domain_set_handoff.yaml",
                    "handoff_location": "stream_boundary_intersection",
                    "fallback_river_geometry": "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg",
                },
                "soil_moisture": {
                    "source": "data/sources/nwm/soil_moisture.csv",
                    "pairing": "inland_antecedent_moisture_pairing",
                    "lookback_hours": 24,
                },
                "infiltration": dict(_INFILTRATION_CONSTANTS),
            },
            "scenario_build": _deep_merge(
                defaults["scenario_build"],
                {
                    "zsini_mode": "dry",
                    "timing": {
                        "spinup_hours": 12,
                        "drain_down_hours": 24,
                        "min_run_hours": 72,
                        "max_run_hours": 168,
                    },
                },
            ),
            "evaluation": {
                "asset_source": "data/smart_ds",
                "output_root": "data/evaluation",
                "multi_domain_merge": {
                    "method": "max_depth_per_asset",
                    "retain_source_domain_id": True,
                    "write_overlap_diagnostics": True,
                },
            },
            "wflow": {
                "enabled": True,
                "plugin": "wflow_sbm",
                "data_catalog": "data/wflow/data_catalog.yml",
                "base_model_root": "data/wflow/base",
                "events_root": "data/wflow/events",
                "readiness_root": "data/wflow/readiness",
                # ADR-0016: USGS instantaneous (IV) records are calibration/validation inputs,
                # not per-event runtime forcing. No design run depends on a cached IV file, so the
                # former fetch/require_instantaneous forcing flags are removed. Discharge is
                # Wflow-generated and frequency-corrected by inland_coupling.amplification (single-K).
                "streamflow_calibration": {
                    "event_records_root": "data/sources/usgs_streamgages/event_streamflow_iv",
                },
                "run": {
                    "command": "wflow_cli {run_config}",
                },
                "domain_set": {
                    "enabled": True,
                    "review_required": True,
                    "allow_multiple_submodels": True,
                    "outlet_source": "boundary_handoff_watershed",
                    "event_catalog_scope": "shared_across_domain_set",
                    "submodels": [],
                    "subbasin_fabric": "data/wflow/domain_set_subbasins.gpkg",
                    "subbasin_fabric_diagnostics": "data/wflow/readiness/nhdplus_subbasin_fabric.csv",
                    "crossings": {
                        "min_uparea_km2": 5.0,
                    },
                    "huc": {
                        "levels": [8, 6, 4],
                        "allow_union": True,
                        "query_pad_degrees": 0.1,
                        "root": "data/wflow/domain_huc",
                        "output": "data/static/aoi/wflow_nhdplus_watersheds.geojson",
                    },
                },
                "domain_set_manifest": "data/wflow/domain_set.yaml",
            },
        }
    )
    return defaults


def _coastal_methodology_defaults() -> dict:
    defaults = _common_methodology_defaults()
    defaults["collection"].update(
        {
            "hurdat2": {
                "url": "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt",
                "request_timeout_seconds": 60,
            },
            "cora": {
                "reuse_existing": True,
                "s3_bucket": "noaa-nos-cora-pds",
                "s3_key_pattern": "V1.1/assimilated/500m_grid/500m_grid_zeta_{date:%Y%m%d}.nc",
                "variable": "zeta",
                "datum": "MSL",
                "units": "m",
                "nearest_k": 20,
                "max_snap_distance_km": 5.0,
                "parallel_workers": 16,
                "request_timeout_seconds": 60,
                "raw_cache_enabled": False,
                "raw_cache_dirname": "cora_daily_nc",
            },
            "era5_waves": {
                "provider": "earthdatahub",
                "auth_path": "artifacts/credentials/earthdatahub-api-key.txt",
                "smoke_start": "2018-01-01T00:00:00",
                "smoke_end": "2018-01-01T23:00:00",
            },
            "aorc_sst": _AORC_SST_DEFAULTS,
            "nwm": {
                "version": "3.0",
                "bucket": "noaa-nwm-retrospective-3-0-pds",
                "start": "1979-02-01",
                "end": "2022-12-31",
                "streamflow": {
                    "zarr": "s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr",
                    "variable": "streamflow",
                    "feature_dim": "feature_id",
                    "available": False,
                    "feature_ids": [],
                },
                "soil_moisture": _deep_merge(
                    _NWM_SOIL_MOISTURE_DEFAULTS,
                    {"points_source": "data/static/aoi/study_area.geojson"},
                ),
            },
        }
    )
    defaults.update(
        {
            "coastal_wave_coupling": {
                "quadtree": {
                    "base_model_root": "data/sfincs/base_quadtree_snapwave",
                    "res": 60,
                    "rotated": True,
                    "nr_subgrid_pixels": 6,
                    "waterlevel_boundary_buffer_m": 180,
                },
                "snapwave": {
                    "boundary_min_dist": 1500,
                    "boundary_seaward_dist": 5000,
                    "directional_spread_degrees": 20.0,
                },
                "hydrology": {
                    "precipitation": {
                        "variable": "APCP_surface",
                        "cumulative_input": True,
                        "time_label": "right",
                        "buffer_m": 30000,
                    },
                    "soil_moisture": {
                        "source": "data/sources/nwm/soil_moisture.csv",
                        "lookback_hours": 24,
                    },
                    "infiltration": dict(_INFILTRATION_CONSTANTS),
                },
            },
            "scenario_build": _deep_merge(
                defaults["scenario_build"],
                {"zsini_mode": "boundary_t0"},
            ),
        }
    )
    return defaults


def _methodology_defaults(flood_setting: str) -> dict:
    if flood_setting == "inland":
        return _inland_methodology_defaults()
    return _coastal_methodology_defaults()


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

    def stage_order(self) -> tuple[str, ...]:
        return _stage_order

    def validate(self) -> list[str]:
        issues = []
        if not self.name:
            issues.append("project.name is required")
        includes = self.config.get("includes", {})
        if "grid" not in includes and "smartds" not in includes:
            issues.append("includes.grid or includes.smartds is required")
        for key in ("sfincs",):
            if key not in self.config.get("includes", {}):
                issues.append(f"includes.{key} is required")
        if self.config.get("coastal_waves") and "snapwave" not in includes:
            issues.append("includes.snapwave is required when coastal_waves is true")
        if self.config.get("wflow", {}).get("enabled") and "wflow" not in includes:
            issues.append("includes.wflow is required for Wflow-coupled Study Locations")
        for key in ("sfincs_build", "sfincs_update_forcing"):
            if key not in self.model_recipes:
                issues.append(f"model_recipes.{key} is required")
        if "wflow" in includes:
            for key in ("wflow_build", "wflow_update_forcing"):
                if key not in self.model_recipes:
                    issues.append(f"model_recipes.{key} is required for Wflow-coupled Study Locations")
        return issues

    def write_resolved_config(self, out_path=None) -> Path:
        """Write the fully merged Location Configuration to a readable YAML file.

        This is the single inspectable view of every effective setting — the
        location ``config.yaml``, its included detail files, and the methodology
        defaults from this module — exactly as ``define_location`` merges them.
        Stakeholders read it to see knobs that the per-location YAML overrides
        silently inherit. It is a generated reference, not an input: edit
        ``config.yaml`` or a detail file to change a value, then regenerate.
        """
        target = Path(out_path) if out_path is not None else self.root / RESOLVED_CONFIG_FILENAME
        body = yaml.safe_dump(self.config, sort_keys=True, default_flow_style=False)
        header = _RESOLVED_CONFIG_HEADER.format(
            name=self.name,
            flood_setting=self.config.get("flood_setting", "coastal"),
            filename=RESOLVED_CONFIG_FILENAME,
        )
        target.write_text(header + body, encoding="utf-8")
        return target


RESOLVED_CONFIG_FILENAME = "config.resolved.yaml"

_RESOLVED_CONFIG_HEADER = (
    "# ==========================================================================\n"
    "# GENERATED FILE — DO NOT EDIT BY HAND.\n"
    "#\n"
    "# Fully resolved Location Configuration for {name} (flood_setting={flood_setting}):\n"
    "# config.yaml + included detail files + methodology defaults, exactly as\n"
    "# define_location() merges them at runtime.\n"
    "#\n"
    "# Why this file exists: many effective settings live as methodology defaults\n"
    "# in src/study_location.py and are otherwise invisible from this folder. This\n"
    "# is the one place to read every knob a stakeholder might want to change.\n"
    "#\n"
    "# To change a value: edit config.yaml or the relevant included detail file\n"
    "# (your override is deep-merged ON TOP of the defaults shown here), then\n"
    "# regenerate with:\n"
    "#     python tests/flood_rm/show_resolved_config.py {name}\n"
    "# ==========================================================================\n"
)


def write_resolved_config(config_path, out_path=None) -> Path:
    """Resolve a Location Configuration and write its readable merged view.

    Thin wrapper over :meth:`LocationDefinition.write_resolved_config` so callers
    can dump a location by config path without holding the definition object.
    """
    return define_location(config_path).write_resolved_config(out_path)


def define_location(config_path) -> LocationDefinition:
    config_path = _resolve_location_config_path(config_path)
    base = _load_yaml_file(config_path)
    name = str(base.get("project", {}).get("name", "")).strip()
    if not name:
        raise ValueError("project.name is required in config.yaml")
    root = config_path.parent
    flood_setting = str(base.get("flood_setting", "coastal")).strip() or "coastal"

    includes = base.get("includes") or {}
    data_sources = _load_location_detail(root, includes.get("data_sources"), required=False)
    grid = _load_location_detail(root, includes.get("grid"), required="smartds" not in includes)
    smartds = _load_location_detail(root, includes.get("smartds"), required=False)
    sfincs = _load_location_detail(root, includes.get("sfincs"))
    wflow = _load_location_detail(root, includes.get("wflow"), required=False)
    snapwave = _load_location_detail(root, includes.get("snapwave"), required=False)
    model_recipes = _load_model_recipes(root, includes, sfincs=sfincs, wflow=wflow, snapwave=snapwave)

    config = _methodology_defaults(flood_setting)
    # Legacy `extends:` remains supported, but location configs no longer need
    # a stakeholder-facing shared YAML file to get methodology defaults.
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

    _apply_model_recipe_paths(config, model_recipes)
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


def _load_extends_base(root: Path, value) -> dict:
    """Load a shared base config referenced by `extends:` (path relative to the
    location's config.yaml, e.g. `../_shared/inland_base.yaml`)."""
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


def _load_model_recipes(root: Path, includes: dict, *, sfincs: dict, wflow: dict, snapwave: dict) -> dict:
    recipes = _load_model_recipe_includes(root, includes)
    recipes.update(_hydromt_recipes_from_model_config("sfincs", sfincs))
    recipes.update(_hydromt_recipes_from_model_config("wflow", wflow))
    recipes.update(_hydromt_recipes_from_model_config("snapwave", snapwave))
    return recipes


def _hydromt_recipes_from_model_config(model_name: str, config: dict) -> dict:
    hydromt = config.get("hydromt") or {}
    recipes = {}
    if not isinstance(hydromt, dict):
        return recipes
    for purpose, recipe in hydromt.items():
        if isinstance(recipe, dict):
            recipes[f"{model_name}_{purpose}"] = recipe
    return recipes


def _apply_model_recipe_paths(config: dict, model_recipes: dict) -> None:
    includes = config.get("includes") or {}
    sfincs = config.setdefault("sfincs", {})
    if "sfincs_build" in model_recipes:
        sfincs.setdefault("build_config", "data/sfincs/config/sfincs_build.yml")
    if "sfincs_update_forcing" in model_recipes:
        sfincs.setdefault("update_forcing_config", "data/sfincs/config/sfincs_update_forcing.yml")
    if "snapwave_build" in model_recipes:
        sfincs.setdefault("snapwave_build_config", "data/sfincs/config/snapwave_build.yml")
    if "snapwave_update_forcing" in model_recipes:
        sfincs.setdefault("snapwave_update_forcing_config", "data/sfincs/config/snapwave_update_forcing.yml")
    if "wflow_build" in includes or "wflow_update_forcing" in includes:
        wflow = config.setdefault("wflow", {})
        if "wflow_build" in includes:
            wflow["build_config"] = includes["wflow_build"]
        if "wflow_update_forcing" in includes:
            wflow["update_forcing_config"] = includes["wflow_update_forcing"]
    if "wflow_build" in model_recipes:
        config.setdefault("wflow", {}).setdefault("build_config", "data/wflow/config/wflow_build.yml")
    if "wflow_update_forcing" in model_recipes:
        config.setdefault("wflow", {}).setdefault(
            "update_forcing_config",
            "data/wflow/config/wflow_update_forcing.yml",
        )


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
