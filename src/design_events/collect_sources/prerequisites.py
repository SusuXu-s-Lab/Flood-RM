from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


def prepare_collection_prerequisites(config, paths):
    """Create lightweight review-required inputs needed before source collection."""
    rows = []
    aorc_row = prepare_aorc_transposition_region(config, paths)
    if aorc_row is not None:
        rows.append(aorc_row)
    soil_row = prepare_nwm_soil_moisture_points(config, paths)
    if soil_row is not None:
        rows.append(soil_row)
    return pd.DataFrame(
        rows,
        columns=[
            "artifact",
            "path",
            "status",
            "source_geometry",
            "buffer_km",
            "review_status",
        ],
    )


def prepare_nwm_soil_moisture_points(config, paths):
    from design_events.collect_sources.soil_moisture_points import (
        ensure_points_geojson,
        has_footprint,
    )

    spec = config.get("collection", {}).get("nwm", {}).get("soil_moisture", {})
    if not spec or spec.get("points"):
        # No NWM soil-moisture source, or explicit points already pinned in the YAML.
        return None
    if not has_footprint(spec, paths):
        return None

    result = ensure_points_geojson(spec, paths)
    return {
        "artifact": "nwm soil-moisture sampling points",
        "path": str(result["path"]),
        "status": result["status"],
        "source_geometry": str(result["source_geometry"]) if result.get("source_geometry") else pd.NA,
        "buffer_km": pd.NA,
        "review_status": "review_required",
    }


def prepare_aorc_transposition_region(config, paths):
    collection = config.get("collection", {})
    spec = collection.get("aorc_sst", {})
    region = spec.get("transposition_region", {})
    geometry_file = region.get("geometry_file")
    if not geometry_file:
        return None

    output_path = _location_path(paths, geometry_file)
    if output_path.exists():
        return _result_row(
            output_path,
            status="reused",
            source_geometry=pd.NA,
            buffer_km=pd.NA,
            review_status=pd.NA,
        )

    source_path = _source_geometry_path(config, paths)
    source = gpd.read_file(source_path)
    if source.empty:
        raise ValueError(f"AORC transposition source geometry is empty: {source_path}")

    buffer_km = _transposition_buffer_km(config, region)
    model_crs = _model_crs(config)
    geometry = source.to_crs(model_crs).geometry.union_all().buffer(buffer_km * 1000.0)
    output = gpd.GeoDataFrame(
        {
            "region_id": [region.get("id", "review-required")],
            "source_geometry": [str(source_path)],
            "buffer_km": [float(buffer_km)],
            "review_status": ["review_required"],
            "review_notes": ["Generated from the evaluation footprint for source collection; review before production SST use."],
        },
        geometry=[geometry],
        crs=model_crs,
    ).to_crs("EPSG:4326")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_file(output_path, driver="GeoJSON")
    return _result_row(
        output_path,
        status="created_review_required",
        source_geometry=source_path,
        buffer_km=float(buffer_km),
        review_status="review_required",
    )


def _result_row(output_path, *, status, source_geometry, buffer_km, review_status):
    return {
        "artifact": "aorc_sst transposition region",
        "path": str(output_path),
        "status": status,
        "source_geometry": str(source_geometry) if isinstance(source_geometry, Path) else source_geometry,
        "buffer_km": buffer_km,
        "review_status": review_status,
    }


def _source_geometry_path(config, paths):
    candidates = [
        config.get("collection", {}).get("aorc_sst", {}).get("transposition_region", {}).get("source_geometry"),
        config.get("smart_ds_evaluation_footprint", {}).get("output"),
        config.get("grid_footprint", {}).get("source"),
        "data/static/aoi/evaluation_footprint.geojson",
        "data/static/aoi/study_area.geojson",
    ]
    for value in candidates:
        if not value:
            continue
        path = _location_path(paths, value)
        if path.exists():
            return path
    raise FileNotFoundError("could not find a source geometry for the AORC SST transposition region")


def _transposition_buffer_km(config, region):
    if region.get("buffer_km") is not None:
        return float(region["buffer_km"])
    if region.get("buffer_m") is not None:
        return float(region["buffer_m"]) / 1000.0
    discovery = config.get("collection", {}).get("usgs_streamgages", {}).get("discovery", {})
    if discovery.get("hydrologic_buffer_km") is not None:
        return float(discovery["hydrologic_buffer_km"])
    return 50.0


def _model_crs(config):
    return (
        config.get("project", {}).get("model_crs")
        or config.get("crs")
        or config.get("sfincs", {}).get("crs")
        or "EPSG:32617"
    )


def _location_path(paths, value):
    path = Path(value)
    if path.is_absolute():
        return path
    root = paths.get("location_root") or paths.get("repo_root") or Path.cwd()
    if path.parts and path.parts[0] in {"data", "02_flood", "01_grid"}:
        return Path(root) / path
    return Path(paths.get("repo_root", root)) / path
