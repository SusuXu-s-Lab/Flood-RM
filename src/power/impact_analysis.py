"""Compute flood exposure and fragility-based affected assets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from dataclasses import dataclass
from html import escape
from pathlib import Path

from power.artifact_io import read_parquet
from power.fragility import (
    failure_probability,
    line_local_asset_type,
    load_asset_type_mapping,
    load_flood_depth_curves,
)
from power.paths import POWER_GRID

ASSET_REGISTRY = POWER_GRID / "asset_registry"
SMART_DS_COMPAT = POWER_GRID / "augmented"
IMPACTS_DIR = POWER_GRID / "figures" / "impacts"
VISUALIZATIONS = POWER_GRID / "visualizations"
DEFAULT_IMPACT_VIEWER_OUTPUT = VISUALIZATIONS / "impact_viewer.html"
DEFAULT_EVENT_DIR = (
    POWER_GRID / "sfincs_truth" / "run_outputs" / "single_event_tests" / "riley_90m"
)
DEFAULT_PROBABILITY_THRESHOLD = 0.50
DEFAULT_MAX_SAMPLE_DISTANCE_M = 150.0
DEFAULT_FLOOD_SPATIAL_STRIDE = 1
DEFAULT_FLOOD_TIME_STRIDE = 1


@dataclass(frozen=True)
class AssetPoint:
    asset_id: str
    asset_type: str
    feeder_id: str
    lon: float
    lat: float
    label: str


def _finite_float(value: str | int | float | None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_asset_points(registry_dir: Path = SMART_DS_COMPAT, *, include_lines: bool = False) -> list[AssetPoint]:
    """Load flood-relevant point assets from the Asset Registry."""
    assets_parquet = registry_dir / "assets.parquet"
    if assets_parquet.exists():
        return _load_stage_a1_asset_points(assets_parquet, include_lines=include_lines)

    points: list[AssetPoint] = []

    for row in _read_csv(registry_dir / "load_buses.csv"):
        lon = _finite_float(row.get("lon"))
        lat = _finite_float(row.get("lat"))
        if lon is None or lat is None:
            continue
        points.append(
            AssetPoint(
                asset_id=row["bus"],
                asset_type="load_bus",
                feeder_id=row.get("feeder_id", ""),
                lon=lon,
                lat=lat,
                label=f"load bus {row['bus']}",
            )
        )

    for row in _read_csv(registry_dir / "transformers.csv"):
        lon = _finite_float(row.get("location_lon"))
        lat = _finite_float(row.get("location_lat"))
        if lon is None or lat is None:
            continue
        points.append(
            AssetPoint(
                asset_id=row["transformer_name"],
                asset_type="transformer",
                feeder_id=row.get("feeder_id", ""),
                lon=lon,
                lat=lat,
                label=f"transformer {row['transformer_name']}",
            )
        )

    for row in _read_csv(registry_dir / "sources.csv"):
        lon = _finite_float(row.get("lon"))
        lat = _finite_float(row.get("lat"))
        if lon is None or lat is None:
            continue
        points.append(
            AssetPoint(
                asset_id=row["source_name"],
                asset_type="source",
                feeder_id=row.get("feeder_id", ""),
                lon=lon,
                lat=lat,
                label=f"source {row['source_name']}",
            )
        )

    if include_lines:
        for row in _read_csv(registry_dir / "lines.csv"):
            from_lon = _finite_float(row.get("from_lon"))
            from_lat = _finite_float(row.get("from_lat"))
            to_lon = _finite_float(row.get("to_lon"))
            to_lat = _finite_float(row.get("to_lat"))
            if None in {from_lon, from_lat, to_lon, to_lat}:
                continue
            points.append(
                AssetPoint(
                    asset_id=row["line_name"],
                    asset_type=line_local_asset_type(row.get("line_class")),
                    feeder_id=row.get("feeder_id", ""),
                    lon=(from_lon + to_lon) / 2.0,
                    lat=(from_lat + to_lat) / 2.0,
                    label=f"{row.get('line_class', 'line')} line {row['line_name']}",
                )
            )

    return points


def _load_stage_a1_asset_points(assets_parquet: Path, *, include_lines: bool) -> list[AssetPoint]:
    points: list[AssetPoint] = []
    line_types = {"line", "overhead_line", "underground_line_proxy"}
    for row in read_parquet(assets_parquet):
        lon = _finite_float(row.get("lon"))
        lat = _finite_float(row.get("lat"))
        if lon is None or lat is None or row.get("coordinate_status") != "valid":
            continue
        asset_type = str(row.get("asset_type", ""))
        if not row.get("is_flood_relevant") and not (include_lines and asset_type in line_types):
            continue
        asset_id = str(row["asset_id"])
        points.append(
            AssetPoint(
                asset_id=asset_id,
                asset_type=asset_type,
                feeder_id=str(row.get("feeder_id") or ""),
                lon=lon,
                lat=lat,
                label=f"{asset_type} {asset_id}",
            )
        )
    return sorted(points, key=lambda point: point.asset_id)


def _require_geo_stack():
    try:
        import numpy as np
        import xarray as xr
        from pyproj import Transformer
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "Impact analysis requires numpy, xarray, pyproj, and scipy. "
            "Run with the project venv, for example: python -m power.impact_viewer compute"
        ) from exc
    return np, xr, Transformer, cKDTree


def _sample_peak_depths(event_dir: Path, assets: list[AssetPoint], *, max_sample_distance_m: float | None):
    np, xr, Transformer, cKDTree = _require_geo_stack()
    map_path = Path(event_dir) / "sfincs_map.nc"
    if not map_path.exists():
        raise FileNotFoundError(map_path)

    with xr.open_dataset(map_path) as ds:
        for name in ["x", "y", "zs", "zb", "msk"]:
            if name not in ds:
                raise RuntimeError(f"{map_path} missing {name}")
        active = np.asarray(ds["msk"].values, dtype=float) > 0
        depth = np.asarray(ds["zs"].values, dtype=np.float32) - np.asarray(ds["zb"].values, dtype=np.float32)[None, :, :]
        depth = np.where(active[None, :, :], np.maximum(depth, 0.0), np.nan)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            peak_depth = np.nanmax(depth, axis=0)
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & active & np.isfinite(peak_depth)
    grid_xy = np.column_stack([x[valid], y[valid]])
    tree = cKDTree(grid_xy)
    to_model = Transformer.from_crs("EPSG:4326", "EPSG:26919", always_xy=True)
    asset_xy = np.asarray([to_model.transform(a.lon, a.lat) for a in assets], dtype=float)
    distances, nearest = tree.query(asset_xy, k=1)
    flat_depth = peak_depth[valid]
    depths = flat_depth[nearest].astype(float)
    if max_sample_distance_m is not None:
        depths = np.where(distances <= float(max_sample_distance_m), depths, np.nan)
    return depths, distances


def compute_asset_impacts(
    event_dir: Path,
    *,
    event_id: str | None = None,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
    max_sample_distance_m: float | None = DEFAULT_MAX_SAMPLE_DISTANCE_M,
    include_lines: bool = False,
    registry_dir: Path = SMART_DS_COMPAT,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compute per-asset flood exposure and fragility-driven impact rows."""
    assets = load_asset_points(registry_dir, include_lines=include_lines)
    depths, distances = _sample_peak_depths(
        event_dir,
        assets,
        max_sample_distance_m=max_sample_distance_m,
    )
    curves = load_flood_depth_curves()
    mapping = load_asset_type_mapping()

    rows: list[dict[str, object]] = []
    for asset, depth_m, distance_m in zip(assets, depths, distances, strict=True):
        depth = _finite_float(depth_m)
        probability = failure_probability(asset.asset_type, depth, curves=curves, mapping=mapping)
        rows.append(
            {
                "event_id": event_id or Path(event_dir).name,
                "asset_id": asset.asset_id,
                "asset_type": asset.asset_type,
                "erad_asset_type": mapping[asset.asset_type],
                "feeder_id": asset.feeder_id,
                "lon": asset.lon,
                "lat": asset.lat,
                "peak_depth_m": depth,
                "nearest_grid_distance_m": float(distance_m),
                "failure_probability": probability,
                "affected_probability": probability,
                "affected": probability >= probability_threshold,
                "affected_probability_threshold": probability_threshold,
                "label": asset.label,
            }
        )

    summary = summarize_impacts(rows, event_id=event_id or Path(event_dir).name)
    summary["event_dir"] = str(event_dir)
    summary["probability_threshold"] = probability_threshold
    summary["max_sample_distance_m"] = max_sample_distance_m
    summary["include_lines"] = include_lines
    return rows, summary


def summarize_impacts(rows: list[dict[str, object]], *, event_id: str) -> dict[str, object]:
    by_type: dict[str, dict[str, object]] = {}
    for row in rows:
        asset_type = str(row["asset_type"])
        bucket = by_type.setdefault(
            asset_type,
            {
                "asset_count": 0,
                "affected_count": 0,
                "expected_affected_count": 0.0,
                "max_failure_probability": 0.0,
                "max_peak_depth_m": 0.0,
            },
        )
        probability = float(row["failure_probability"] or 0.0)
        depth = float(row["peak_depth_m"] or 0.0)
        bucket["asset_count"] = int(bucket["asset_count"]) + 1
        bucket["affected_count"] = int(bucket["affected_count"]) + int(bool(row["affected"]))
        bucket["expected_affected_count"] = float(bucket["expected_affected_count"]) + probability
        bucket["max_failure_probability"] = max(float(bucket["max_failure_probability"]), probability)
        bucket["max_peak_depth_m"] = max(float(bucket["max_peak_depth_m"]), depth)

    return {
        "event_id": event_id,
        "asset_count": len(rows),
        "affected_count": sum(int(bool(row["affected"])) for row in rows),
        "expected_affected_count": sum(float(row["failure_probability"] or 0.0) for row in rows),
        "by_asset_type": by_type,
    }


def write_outputs(rows: list[dict[str, object]], summary: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "asset_impacts.csv"
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _event_float(row: dict[str, object], key: str) -> float | None:
    return _finite_float(row.get(key))


def _feeder_color(feeder_id: str) -> list[int]:
    number = int(feeder_id.removeprefix("f")) if feeder_id.startswith("f") else 0
    hue = ((number * 47 + 176) % 360) / 360
    sector = int(hue * 6)
    fraction = hue * 6 - sector
    p = 0.78 * (1 - 0.66)
    q = 0.78 * (1 - fraction * 0.66)
    t = 0.78 * (1 - (1 - fraction) * 0.66)
    r, g, b = (
        (0.78, t, p),
        (q, 0.78, p),
        (p, 0.78, t),
        (p, q, 0.78),
        (t, p, 0.78),
        (0.78, p, q),
    )[sector % 6]
    return [round(r * 255), round(g * 255), round(b * 255), 190]


def _round_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def build_flood_depth_payload_from_arrays(
    *,
    lon_grid,
    lat_grid,
    depth_frames,
    time_labels: list[str],
    wet_threshold_m: float = 0.05,
) -> dict[str, object]:
    """Build the browser flood-depth payload from gridded lon/lat/depth arrays."""
    cells: list[list[float]] = []
    cell_indices: list[tuple[int, int]] = []
    for row_index, row in enumerate(lon_grid):
        for col_index, lon in enumerate(row):
            lat = lat_grid[row_index][col_index]
            if _finite_float(lon) is None or _finite_float(lat) is None:
                continue
            cells.append([round(float(lon), 6), round(float(lat), 6)])
            cell_indices.append((row_index, col_index))

    frames: list[dict[str, object]] = []
    max_depth = 0.0
    max_depth_frame_index = 0
    max_depth_cell_index = 0
    for frame_index, frame in enumerate(depth_frames):
        depths: list[float | None] = []
        for cell_index, (row_index, col_index) in enumerate(cell_indices):
            depth = _finite_float(frame[row_index][col_index])
            if depth is None or depth <= wet_threshold_m:
                depths.append(None)
                continue
            rounded_depth = _round_or_none(depth)
            depths.append(rounded_depth)
            if depth > max_depth:
                max_depth = float(depth)
                max_depth_frame_index = frame_index
                max_depth_cell_index = cell_index
        frames.append(
            {
                "index": frame_index,
                "label": time_labels[frame_index] if frame_index < len(time_labels) else str(frame_index),
                "depths": depths,
            }
        )

    return {
        "cells": cells,
        "frames": frames,
        "time_labels": time_labels,
        "wet_threshold_m": wet_threshold_m,
        "max_depth_m": _round_or_none(max_depth) or 0.0,
        "max_depth_frame_index": max_depth_frame_index,
        "max_depth_cell_index": max_depth_cell_index,
        "default_frame_index": max_depth_frame_index,
    }


def _time_labels_from_values(values) -> list[str]:
    labels: list[str] = []
    for value in values:
        text = str(value)
        if "T" in text:
            text = text.replace("T", " ")
        if "." in text:
            text = text.split(".", 1)[0]
        labels.append(text)
    return labels


def build_flood_depth_payload(
    event_dir: Path,
    *,
    spatial_stride: int = DEFAULT_FLOOD_SPATIAL_STRIDE,
    time_stride: int = DEFAULT_FLOOD_TIME_STRIDE,
    wet_threshold_m: float = 0.05,
) -> dict[str, object]:
    """Read SFINCS map output and build a time-indexed flood-depth payload."""
    np, xr, Transformer, _ = _require_geo_stack()
    stride = max(1, int(spatial_stride))
    frame_stride = max(1, int(time_stride))
    map_path = Path(event_dir) / "sfincs_map.nc"
    if not map_path.exists():
        raise FileNotFoundError(map_path)

    with xr.open_dataset(map_path) as ds:
        for name in ["x", "y", "zs", "zb", "msk"]:
            if name not in ds:
                raise RuntimeError(f"{map_path} missing {name}")
        active = np.asarray(ds["msk"].values, dtype=float) > 0
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
        zs = np.asarray(ds["zs"].values, dtype=np.float32)
        zb = np.asarray(ds["zb"].values, dtype=np.float32)
        raw_depth = np.where(active[None, :, :], np.maximum(zs - zb[None, :, :], 0.0), np.nan)
        depth = np.where(np.isfinite(raw_depth), raw_depth, np.nan)
        time_values = np.asarray(ds["time"].values) if "time" in ds else np.arange(depth.shape[0])

    full_max = float(np.nanmax(depth)) if np.isfinite(depth).any() else 0.0
    if full_max > 0:
        max_frame_raw, max_row, max_col = (int(v) for v in np.unravel_index(np.nanargmax(depth), depth.shape))
    else:
        max_frame_raw, max_row, max_col = 0, 0, 0

    row_ids, col_ids = np.indices(active.shape)
    sampled = (
        active
        & np.isfinite(x)
        & np.isfinite(y)
        & ((row_ids % stride) == 0)
        & ((col_ids % stride) == 0)
    )
    if 0 <= max_row < sampled.shape[0] and 0 <= max_col < sampled.shape[1]:
        sampled[max_row, max_col] = True

    sampled_rows, sampled_cols = np.where(sampled)
    to_lonlat = Transformer.from_crs("EPSG:26919", "EPSG:4326", always_xy=True)
    lon_values, lat_values = to_lonlat.transform(x[sampled], y[sampled])
    cells = [
        [round(float(lon), 6), round(float(lat), 6)]
        for lon, lat in zip(np.asarray(lon_values).flat, np.asarray(lat_values).flat, strict=True)
    ]
    max_depth_cell_index = 0
    for index, (row, col) in enumerate(zip(sampled_rows, sampled_cols, strict=True)):
        if int(row) == max_row and int(col) == max_col:
            max_depth_cell_index = index
            break

    frame_indices = list(range(0, depth.shape[0], frame_stride))
    if max_frame_raw not in frame_indices:
        frame_indices.append(max_frame_raw)
        frame_indices.sort()
    time_labels_all = _time_labels_from_values(time_values)
    frames: list[dict[str, object]] = []
    default_frame_index = 0
    for frame_position, frame_index in enumerate(frame_indices):
        values = depth[frame_index, sampled_rows, sampled_cols]
        frame_depths = [
            _round_or_none(float(value)) if math.isfinite(float(value)) and float(value) > wet_threshold_m else None
            for value in values
        ]
        if frame_index == max_frame_raw:
            default_frame_index = frame_position
        frames.append(
            {
                "index": frame_position,
                "source_index": int(frame_index),
                "label": time_labels_all[frame_index] if frame_index < len(time_labels_all) else str(frame_index),
                "depths": frame_depths,
            }
        )

    return {
        "cells": cells,
        "frames": frames,
        "time_labels": [str(frame["label"]) for frame in frames],
        "wet_threshold_m": wet_threshold_m,
        "spatial_stride": stride,
        "time_stride": frame_stride,
        "max_depth_m": _round_or_none(full_max) or 0.0,
        "max_depth_frame_index": default_frame_index,
        "max_depth_source_frame_index": max_frame_raw,
        "max_depth_cell_index": max_depth_cell_index,
        "default_frame_index": default_frame_index,
    }


def build_asset_impact_frame_payload(
    event_dir: Path,
    assets: list[dict[str, object]],
    source_frame_indices: list[int],
    *,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
    max_sample_distance_m: float | None = DEFAULT_MAX_SAMPLE_DISTANCE_M,
) -> list[dict[str, object]]:
    """Sample each flood frame at asset locations and evaluate dynamic fragility."""
    if not assets or not source_frame_indices:
        return []

    np, xr, Transformer, cKDTree = _require_geo_stack()
    map_path = Path(event_dir) / "sfincs_map.nc"
    if not map_path.exists():
        raise FileNotFoundError(map_path)

    with xr.open_dataset(map_path) as ds:
        for name in ["x", "y", "zs", "zb", "msk"]:
            if name not in ds:
                raise RuntimeError(f"{map_path} missing {name}")
        active = np.asarray(ds["msk"].values, dtype=float) > 0
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
        zs = np.asarray(ds["zs"].values, dtype=np.float32)
        zb = np.asarray(ds["zb"].values, dtype=np.float32)

    valid = np.isfinite(x) & np.isfinite(y) & active
    if not np.any(valid):
        return []

    grid_xy = np.column_stack([x[valid], y[valid]])
    tree = cKDTree(grid_xy)
    to_model = Transformer.from_crs("EPSG:4326", "EPSG:26919", always_xy=True)
    asset_xy = np.asarray(
        [to_model.transform(asset["position"][0], asset["position"][1]) for asset in assets],
        dtype=float,
    )
    distances, nearest = tree.query(asset_xy, k=1)
    in_range = np.ones(len(assets), dtype=bool)
    if max_sample_distance_m is not None:
        in_range = distances <= float(max_sample_distance_m)

    curves = load_flood_depth_curves()
    mapping = load_asset_type_mapping()
    frames: list[dict[str, object]] = []
    for frame_position, source_index in enumerate(source_frame_indices):
        frame_index = max(0, min(int(source_index), zs.shape[0] - 1))
        depth_grid = np.maximum(zs[frame_index] - zb, 0.0)
        sampled_depths = depth_grid[valid][nearest].astype(float)
        depths: list[float | None] = []
        probabilities: list[float] = []
        affected: list[bool] = []
        expected_count = 0.0
        affected_count = 0
        for asset_index, asset in enumerate(assets):
            depth = float(sampled_depths[asset_index]) if in_range[asset_index] else math.nan
            clean_depth = depth if math.isfinite(depth) else None
            probability = failure_probability(str(asset["type"]), clean_depth, curves=curves, mapping=mapping)
            rounded_probability = round(float(probability), 5)
            is_affected = probability >= probability_threshold
            depths.append(_round_or_none(clean_depth))
            probabilities.append(rounded_probability)
            affected.append(is_affected)
            expected_count += probability
            affected_count += int(is_affected)
        frames.append(
            {
                "index": frame_position,
                "source_index": int(source_index),
                "depths": depths,
                "failure_probabilities": probabilities,
                "affected": affected,
                "expected_affected_count": round(expected_count, 3),
                "affected_count": affected_count,
            }
        )
    return frames


def build_impact_viewer_payload(
    rows: list[dict[str, object]],
    summary: dict[str, object],
    *,
    registry_dir: Path = ASSET_REGISTRY,
    event_dir: Path | None = None,
    flood_spatial_stride: int = DEFAULT_FLOOD_SPATIAL_STRIDE,
    flood_time_stride: int = DEFAULT_FLOOD_TIME_STRIDE,
) -> dict[str, object]:
    """Build a browser payload for the generated impact map."""
    feeders = [row["feeder_id"] for row in _read_csv(registry_dir / "feeders.csv")]
    lines = [
        {
            "name": row["line_name"],
            "feeder": row["feeder_id"],
            "line_class": row["line_class"],
            "source": [_finite_float(row.get("from_lon")) or 0.0, _finite_float(row.get("from_lat")) or 0.0],
            "target": [_finite_float(row.get("to_lon")) or 0.0, _finite_float(row.get("to_lat")) or 0.0],
            "color": _feeder_color(row["feeder_id"]),
        }
        for row in _read_csv(registry_dir / "lines.csv")
        if _finite_float(row.get("from_lon")) is not None
        and _finite_float(row.get("from_lat")) is not None
        and _finite_float(row.get("to_lon")) is not None
        and _finite_float(row.get("to_lat")) is not None
        and row.get("has_buscoords", "true").lower() == "true"
    ]
    assets = [
        {
            "id": str(row["asset_id"]),
            "type": str(row["asset_type"]),
            "erad_type": str(row["erad_asset_type"]),
            "feeder": str(row["feeder_id"]),
            "position": [float(row["lon"]), float(row["lat"])],
            "peak_depth_m": _event_float(row, "peak_depth_m"),
            "failure_probability": float(row.get("failure_probability") or 0.0),
            "affected": _bool(row.get("affected")),
            "label": str(row.get("label", row["asset_id"])),
        }
        for row in rows
        if _event_float(row, "lon") is not None and _event_float(row, "lat") is not None
    ]
    lons = [asset["position"][0] for asset in assets]
    lats = [asset["position"][1] for asset in assets]
    for line in lines:
        lons.extend([line["source"][0], line["target"][0]])
        lats.extend([line["source"][1], line["target"][1]])
    if not lons or not lats:
        raise ValueError("No coordinates available for impact viewer.")
    max_depth = max((asset["peak_depth_m"] or 0.0 for asset in assets), default=0.0)
    max_probability = max((asset["failure_probability"] for asset in assets), default=0.0)
    flood_event_dir = event_dir or (Path(str(summary["event_dir"])) if summary.get("event_dir") else None)
    flood = (
        build_flood_depth_payload(
            flood_event_dir,
            spatial_stride=flood_spatial_stride,
            time_stride=flood_time_stride,
        )
        if flood_event_dir is not None
        else None
    )
    asset_frames = (
        build_asset_impact_frame_payload(
            flood_event_dir,
            assets,
            [int(frame.get("source_index", frame.get("index", index))) for index, frame in enumerate(flood.get("frames", []))],
            probability_threshold=float(summary.get("probability_threshold", DEFAULT_PROBABILITY_THRESHOLD)),
            max_sample_distance_m=_finite_float(summary.get("max_sample_distance_m")),
        )
        if flood_event_dir is not None and flood is not None
        else []
    )
    payload = {
        "feeders": feeders,
        "bounds": {
            "min_lon": min(lons),
            "max_lon": max(lons),
            "min_lat": min(lats),
            "max_lat": max(lats),
        },
        "summary": summary,
        "lines": lines,
        "assets": assets,
        "max_depth_m": max_depth,
        "max_failure_probability": max_probability,
    }
    if flood is not None:
        payload["flood"] = flood
        payload["asset_frames"] = asset_frames
        payload["max_depth_m"] = max(max_depth, float(flood.get("max_depth_m") or 0.0))
    return payload


def render_impact_viewer_html(payload: dict[str, object]) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    event_id = escape(str(payload["summary"]["event_id"]))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Marshfield Impact Viewer - {event_id}</title>
<style>
  :root {{
    color-scheme: light;
    --ink: #172026;
    --muted: #5b6670;
    --panel: #f7f4ed;
    --line: #d6d0c2;
    --hot: #dc2626;
    --load: #b45309;
    --transformer: #7c3aed;
    --source: #c2410c;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    min-height: 100vh;
    color: var(--ink);
    background: #e8edf0;
    font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  main {{ display: grid; grid-template-columns: 340px minmax(0, 1fr); min-height: 100vh; }}
  aside {{ padding: 18px; background: var(--panel); border-right: 1px solid var(--line); overflow-y: auto; }}
  h1 {{ margin: 0 0 4px; font-size: 20px; letter-spacing: 0; }}
  .subtle {{ color: var(--muted); }}
  .control {{ display: grid; gap: 7px; margin-top: 16px; }}
  label {{ font-weight: 650; }}
  select, button {{
    width: 100%;
    min-height: 34px;
    border: 1px solid #b9b2a5;
    border-radius: 6px;
    background: #fffdfa;
    color: var(--ink);
    font: inherit;
  }}
  button {{ cursor: pointer; font-weight: 650; }}
  .checks {{ display: grid; gap: 8px; margin-top: 9px; }}
  .checks label {{ display: flex; align-items: center; gap: 8px; font-weight: 500; }}
  .stats {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 16px; }}
  .stat {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px; background: #fffdfa; }}
  .stat strong {{ display: block; font-size: 17px; }}
  .legend {{ display: grid; gap: 7px; margin-top: 16px; }}
  .key {{ display: flex; align-items: center; gap: 8px; color: var(--muted); }}
  .swatch {{ width: 26px; height: 4px; border-radius: 999px; background: #555; }}
  .dot {{ width: 11px; height: 11px; border-radius: 999px; background: #555; }}
  .map-shell {{ position: relative; min-width: 0; background: #f9faf8; }}
  canvas {{ display: block; width: 100%; height: 100vh; cursor: grab; }}
  canvas.dragging {{ cursor: grabbing; }}
  .hud {{
    position: absolute;
    left: 14px;
    bottom: 82px;
    padding: 7px 9px;
    max-width: min(720px, calc(100% - 28px));
    border: 1px solid rgba(23, 32, 38, 0.18);
    border-radius: 6px;
    background: rgba(255, 253, 250, 0.9);
    color: var(--muted);
    backdrop-filter: blur(8px);
  }}
  .timeline {{
    position: absolute;
    left: 14px;
    right: 14px;
    bottom: 14px;
    display: grid;
    grid-template-columns: auto minmax(0, 1fr) auto;
    gap: 8px 12px;
    align-items: center;
    padding: 9px 11px;
    border: 1px solid rgba(23, 32, 38, 0.18);
    border-radius: 6px;
    background: rgba(255, 253, 250, 0.92);
    color: var(--muted);
    backdrop-filter: blur(8px);
  }}
  .play-button {{
    width: 38px;
    min-height: 32px;
    padding: 0;
    font-size: 15px;
    line-height: 1;
  }}
  .slider-wrap {{ position: relative; min-width: 0; }}
  input[type="range"] {{ width: 100%; display: block; }}
  .max-tick {{
    position: absolute;
    top: -5px;
    width: 2px;
    height: 24px;
    background: #111827;
    border-radius: 999px;
    pointer-events: none;
  }}
  .time-label {{
    min-width: 172px;
    text-align: right;
    color: var(--ink);
    font-weight: 650;
  }}
  @media (max-width: 820px) {{
    main {{ grid-template-columns: 1fr; }}
    aside {{ border-right: 0; border-bottom: 1px solid var(--line); }}
    canvas {{ height: 70vh; }}
  }}
</style>
</head>
<body>
<main>
  <aside>
    <h1>Marshfield Impacts</h1>
    <div class="subtle">Fragility-based flood impacts for event {event_id}.</div>

    <div class="control">
      <label for="feeder">Feeder</label>
      <select id="feeder"></select>
    </div>

    <div class="control">
      <label for="assetType">Asset type</label>
      <select id="assetType">
        <option value="all">All asset types</option>
        <option value="load_bus">Load buses</option>
        <option value="transformer">Transformers</option>
        <option value="source">Substation sources</option>
        <option value="line_overhead">Overhead lines</option>
        <option value="line_underground">Underground lines</option>
        <option value="line_fuse">Fuses / protective devices</option>
      </select>
    </div>

    <div class="control">
      <label for="colorBy">Color by</label>
      <select id="colorBy">
        <option value="probability">Failure probability</option>
        <option value="depth">Flood depth</option>
        <option value="asset">Asset type</option>
      </select>
    </div>

    <div class="checks">
      <label><input id="showLines" type="checkbox" checked> Lines</label>
      <label><input id="showAffectedOnly" type="checkbox" checked> Affected only</label>
      <label><input id="showUnaffected" type="checkbox"> Show unaffected context</label>
    </div>

    <div class="control">
      <button id="reset">Reset view</button>
    </div>

    <div class="stats" id="stats"></div>

    <div class="legend">
      <div class="key"><span class="swatch" style="background:#1f8a70"></span> Feeder-colored line</div>
      <div class="key"><span class="dot" style="background:#dc2626"></span> Higher failure probability</div>
      <div class="key"><span class="dot" style="background:#2563eb"></span> Lower failure probability</div>
      <div class="key"><span class="dot" style="background:#c2410c"></span> Substation source</div>
      <div class="key"><span class="dot" style="background:#7c3aed"></span> Transformer</div>
      <div class="key"><span class="dot" style="background:#b45309"></span> Load bus</div>
    </div>
  </aside>
  <section class="map-shell">
    <canvas id="map"></canvas>
    <div class="hud">Drag to pan. Scroll to zoom. Points are Asset Registry locations colored by ERAD flood-depth fragility over sampled SFINCS exposure.</div>
    <div class="timeline">
      <button id="playFlood" class="play-button" type="button" aria-label="Play flood timeline">▶</button>
      <div class="slider-wrap">
        <input id="floodTime" type="range" min="0" max="0" step="1" value="0" aria-label="Flood time">
        <span id="maxDepthTick" class="max-tick" title="Peak flood depth time"></span>
      </div>
      <div id="floodTimeLabel" class="time-label">Flood time</div>
    </div>
  </section>
</main>
<script>
const DATA = {data};
const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const controls = {{
  feeder: document.getElementById("feeder"),
  assetType: document.getElementById("assetType"),
  colorBy: document.getElementById("colorBy"),
  showLines: document.getElementById("showLines"),
  showAffectedOnly: document.getElementById("showAffectedOnly"),
  showUnaffected: document.getElementById("showUnaffected"),
  floodTime: document.getElementById("floodTime"),
  floodTimeLabel: document.getElementById("floodTimeLabel"),
  maxDepthTick: document.getElementById("maxDepthTick"),
  playFlood: document.getElementById("playFlood"),
}};
const stats = document.getElementById("stats");
let view = {{}};
let dragging = false;
let last = null;
let currentFloodFrame = DATA.flood ? DATA.flood.default_frame_index : 0;
let playbackTimer = null;

function resetView() {{
  const b = DATA.bounds;
  const lonPad = Math.max((b.max_lon - b.min_lon) * 0.05, 0.002);
  const latPad = Math.max((b.max_lat - b.min_lat) * 0.05, 0.002);
  view = {{minLon: b.min_lon - lonPad, maxLon: b.max_lon + lonPad, minLat: b.min_lat - latPad, maxLat: b.max_lat + latPad}};
  draw();
}}
function includeFeeder(item) {{ return controls.feeder.value === "all" || item.feeder === controls.feeder.value; }}
function includeAssetType(item) {{ return controls.assetType.value === "all" || item.type === controls.assetType.value; }}
function includeAsset(item, index) {{
  if (!includeFeeder(item) || !includeAssetType(item)) return false;
  const affected = currentAffected(item, index);
  const probability = currentFailureProbability(item, index);
  if (controls.showAffectedOnly.checked && !affected) return false;
  if (!controls.showUnaffected.checked && !affected && probability <= 0) return false;
  return true;
}}
function rgba(color) {{ return `rgba(${{color[0]}}, ${{color[1]}}, ${{color[2]}}, ${{(color[3] || 255) / 255}})`; }}
function screenX(lon) {{ return (lon - view.minLon) * canvas.clientWidth / (view.maxLon - view.minLon); }}
function screenY(lat) {{ return (view.maxLat - lat) * canvas.clientHeight / (view.maxLat - view.minLat); }}
function worldLon(x) {{ return view.minLon + x * (view.maxLon - view.minLon) / canvas.clientWidth; }}
function worldLat(y) {{ return view.maxLat - y * (view.maxLat - view.minLat) / canvas.clientHeight; }}
function resize() {{
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}
function currentFlood() {{
  if (!DATA.flood || !DATA.flood.frames.length) return null;
  return DATA.flood.frames[Math.max(0, Math.min(DATA.flood.frames.length - 1, currentFloodFrame))];
}}
function setFloodFrame(index) {{
  const frameCount = DATA.flood?.frames?.length || 0;
  if (!frameCount) return;
  currentFloodFrame = ((Number(index) % frameCount) + frameCount) % frameCount;
  updateFloodUI();
  draw();
}}
function advanceFloodFrame() {{
  setFloodFrame(currentFloodFrame + 1);
}}
function updatePlaybackUI() {{
  const playing = playbackTimer !== null;
  controls.playFlood.textContent = playing ? "❚❚" : "▶";
  controls.playFlood.setAttribute("aria-label", playing ? "Pause flood timeline" : "Play flood timeline");
}}
function stopPlayback() {{
  if (playbackTimer === null) return;
  window.clearInterval(playbackTimer);
  playbackTimer = null;
  updatePlaybackUI();
}}
function togglePlayback() {{
  if (!DATA.flood || DATA.flood.frames.length < 2) return;
  if (playbackTimer !== null) {{
    stopPlayback();
    return;
  }}
  playbackTimer = window.setInterval(advanceFloodFrame, 650);
  updatePlaybackUI();
}}
function currentAssetFrame() {{
  if (!DATA.asset_frames || !DATA.asset_frames.length) return null;
  return DATA.asset_frames[Math.max(0, Math.min(DATA.asset_frames.length - 1, currentFloodFrame))];
}}
function currentDepth(item, index) {{
  const frame = currentAssetFrame();
  if (!frame) return item.peak_depth_m || 0;
  return frame.depths[index] || 0;
}}
function currentFailureProbability(item, index) {{
  const frame = currentAssetFrame();
  if (!frame) return item.failure_probability || 0;
  return frame.failure_probabilities[index] || 0;
}}
function currentAffected(item, index) {{
  const frame = currentAssetFrame();
  if (!frame) return Boolean(item.affected);
  return Boolean(frame.affected[index]);
}}
function gradientColor(t) {{
  t = Math.max(0, Math.min(1, t || 0));
  const stops = [[37,99,235], [34,197,94], [245,158,11], [220,38,38]];
  const scaled = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(scaled));
  const f = scaled - i;
  return [
    Math.round(stops[i][0] + (stops[i + 1][0] - stops[i][0]) * f),
    Math.round(stops[i][1] + (stops[i + 1][1] - stops[i][1]) * f),
    Math.round(stops[i][2] + (stops[i + 1][2] - stops[i][2]) * f),
    230,
  ];
}}
function floodColor(depth) {{
  const color = gradientColor(depth / Math.max(DATA.flood?.max_depth_m || DATA.max_depth_m || 0.01, 0.01));
  return [color[0], color[1], color[2], 118];
}}
function drawFloodLayer() {{
  const frame = currentFlood();
  if (!frame || !DATA.flood?.cells?.length) return;
  const lonSpan = view.maxLon - view.minLon;
  const latSpan = view.maxLat - view.minLat;
  const px = Math.max(3, Math.min(12, canvas.clientWidth * 0.0032 / Math.max(lonSpan, 0.0001)));
  const py = Math.max(3, Math.min(12, canvas.clientHeight * 0.0032 / Math.max(latSpan, 0.0001)));
  for (let i = 0; i < DATA.flood.cells.length; i++) {{
    const depth = frame.depths[i];
    if (!depth) continue;
    const cell = DATA.flood.cells[i];
    const x = screenX(cell[0]), y = screenY(cell[1]);
    if (x < -px || y < -py || x > canvas.clientWidth + px || y > canvas.clientHeight + py) continue;
    ctx.fillStyle = rgba(floodColor(depth));
    ctx.fillRect(x - px / 2, y - py / 2, px, py);
  }}
}}
function drawMaxDepthMarker() {{
  if (!DATA.flood?.cells?.length) return;
  const cell = DATA.flood.cells[DATA.flood.max_depth_cell_index];
  if (!cell) return;
  const x = screenX(cell[0]), y = screenY(cell[1]);
  if (x < -18 || y < -18 || x > canvas.clientWidth + 18 || y > canvas.clientHeight + 18) return;
  ctx.save();
  ctx.strokeStyle = "#111827";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.moveTo(x - 8, y);
  ctx.lineTo(x + 8, y);
  ctx.moveTo(x, y - 8);
  ctx.lineTo(x, y + 8);
  ctx.stroke();
  ctx.restore();
}}
function assetColor(item, index) {{
  if (controls.colorBy.value === "asset") {{
    if (item.type === "source") return [194, 65, 12, 230];
    if (item.type === "transformer") return [124, 58, 237, 230];
    if (item.type === "load_bus") return [180, 83, 9, 225];
    return [75, 85, 99, 210];
  }}
  if (controls.colorBy.value === "depth") return gradientColor(currentDepth(item, index) / Math.max(DATA.max_depth_m, 0.01));
  return gradientColor(currentFailureProbability(item, index));
}}
function assetRadius(item, index) {{
  const base = item.type === "source" ? 5.5 : item.type === "transformer" ? 3.6 : 2.6;
  return base + Math.max(0, currentFailureProbability(item, index)) * 4.5;
}}
function draw() {{
  if (!ctx || !view.minLon) return;
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  ctx.fillStyle = "#f9faf8";
  ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  drawFloodLayer();
  let visibleLines = 0;
  if (controls.showLines.checked) {{
    ctx.lineCap = "round";
    for (const line of DATA.lines) {{
      if (!includeFeeder(line)) continue;
      const ax = screenX(line.source[0]), ay = screenY(line.source[1]);
      const bx = screenX(line.target[0]), by = screenY(line.target[1]);
      if ((ax < -20 && bx < -20) || (ay < -20 && by < -20) || (ax > canvas.clientWidth + 20 && bx > canvas.clientWidth + 20) || (ay > canvas.clientHeight + 20 && by > canvas.clientHeight + 20)) continue;
      visibleLines++;
      ctx.strokeStyle = rgba(line.color);
      ctx.globalAlpha = line.line_class === "underground" ? 0.48 : 0.62;
      ctx.lineWidth = line.line_class === "underground" ? 0.8 : 0.65;
      ctx.setLineDash(line.line_class === "underground" ? [4, 3] : []);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }}
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }}
  let visibleAssets = 0, affectedAssets = 0, expectedVisible = 0;
  for (let i = 0; i < DATA.assets.length; i++) {{
    const item = DATA.assets[i];
    if (!includeAsset(item, i)) continue;
    const x = screenX(item.position[0]), y = screenY(item.position[1]);
    if (x < -12 || y < -12 || x > canvas.clientWidth + 12 || y > canvas.clientHeight + 12) continue;
    const affected = currentAffected(item, i);
    const probability = currentFailureProbability(item, i);
    visibleAssets++;
    affectedAssets += affected ? 1 : 0;
    expectedVisible += probability;
    const color = assetColor(item, i);
    ctx.fillStyle = rgba(color);
    ctx.strokeStyle = affected ? "rgba(17, 24, 39, 0.78)" : "rgba(255, 253, 250, 0.8)";
    ctx.lineWidth = affected ? 1.2 : 0.8;
    ctx.beginPath();
    ctx.arc(x, y, assetRadius(item, i), 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }}
  drawMaxDepthMarker();
  updateStats(visibleLines, visibleAssets, affectedAssets, expectedVisible);
}}
function updateStats(visibleLines, visibleAssets, affectedAssets, expectedVisible) {{
  const s = DATA.summary;
  const frame = currentAssetFrame();
  const expectedNow = frame ? frame.expected_affected_count : s.expected_affected_count;
  const affectedNow = frame ? frame.affected_count : s.affected_count;
  stats.innerHTML = `
    <div class="stat"><strong>${{Number(expectedNow).toFixed(1)}}</strong><span>Expected now</span></div>
    <div class="stat"><strong>${{Number(affectedNow).toLocaleString()}}</strong><span>Binary now</span></div>
    <div class="stat"><strong>${{visibleAssets.toLocaleString()}}</strong><span>Assets visible</span></div>
    <div class="stat"><strong>${{expectedVisible.toFixed(1)}}</strong><span>Expected visible</span></div>
    <div class="stat"><strong>${{visibleLines.toLocaleString()}}</strong><span>Lines visible</span></div>
    <div class="stat"><strong>${{Number(DATA.flood?.max_depth_m || DATA.max_depth_m).toFixed(2)}} m</strong><span>Max flood depth</span></div>
  `;
}}
function updateFloodUI() {{
  if (!DATA.flood || !DATA.flood.frames.length) {{
    controls.floodTime.disabled = true;
    controls.playFlood.disabled = true;
    controls.floodTimeLabel.textContent = "No flood frames";
    controls.maxDepthTick.style.display = "none";
    updatePlaybackUI();
    return;
  }}
  const frameCount = DATA.flood.frames.length;
  controls.playFlood.disabled = frameCount < 2;
  controls.floodTime.max = String(frameCount - 1);
  controls.floodTime.value = String(currentFloodFrame);
  const frame = currentFlood();
  controls.floodTimeLabel.textContent = `${{frame.label}} | ${{Number(DATA.flood.max_depth_m).toFixed(2)}} m max`;
  const denom = Math.max(frameCount - 1, 1);
  controls.maxDepthTick.style.left = `${{(DATA.flood.max_depth_frame_index / denom) * 100}}%`;
  controls.maxDepthTick.style.display = "block";
}}
function populateControls() {{
  controls.feeder.innerHTML = '<option value="all">All feeders</option>' + DATA.feeders.map(f => `<option value="${{f}}">${{f}}</option>`).join("");
  for (const element of Object.values(controls)) element.addEventListener("change", draw);
  controls.floodTime.addEventListener("input", () => {{
    stopPlayback();
    setFloodFrame(controls.floodTime.value || DATA.flood.default_frame_index || 0);
  }});
  controls.playFlood.addEventListener("click", togglePlayback);
  document.getElementById("reset").addEventListener("click", resetView);
  updateFloodUI();
}}
canvas.addEventListener("mousedown", event => {{
  dragging = true;
  last = {{x: event.clientX, y: event.clientY}};
  canvas.classList.add("dragging");
}});
window.addEventListener("mouseup", () => {{ dragging = false; last = null; canvas.classList.remove("dragging"); }});
window.addEventListener("mousemove", event => {{
  if (!dragging || !last) return;
  const dx = event.clientX - last.x;
  const dy = event.clientY - last.y;
  const lonSpan = view.maxLon - view.minLon;
  const latSpan = view.maxLat - view.minLat;
  const lonDelta = dx * lonSpan / canvas.clientWidth;
  const latDelta = dy * latSpan / canvas.clientHeight;
  view.minLon -= lonDelta; view.maxLon -= lonDelta; view.minLat += latDelta; view.maxLat += latDelta;
  last = {{x: event.clientX, y: event.clientY}};
  draw();
}});
canvas.addEventListener("wheel", event => {{
  event.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const lon = worldLon(event.clientX - rect.left);
  const lat = worldLat(event.clientY - rect.top);
  const factor = event.deltaY < 0 ? 0.86 : 1.16;
  view = {{
    minLon: lon - (lon - view.minLon) * factor,
    maxLon: lon + (view.maxLon - lon) * factor,
    minLat: lat - (lat - view.minLat) * factor,
    maxLat: lat + (view.maxLat - lat) * factor,
  }};
  draw();
}}, {{passive: false}});
window.addEventListener("resize", resize);
populateControls();
resetView();
resize();
</script>
</body>
</html>
"""


def write_impact_viewer(
    rows: list[dict[str, object]],
    summary: dict[str, object],
    output: Path = DEFAULT_IMPACT_VIEWER_OUTPUT,
    *,
    registry_dir: Path = ASSET_REGISTRY,
    event_dir: Path | None = None,
    flood_spatial_stride: int = DEFAULT_FLOOD_SPATIAL_STRIDE,
    flood_time_stride: int = DEFAULT_FLOOD_TIME_STRIDE,
) -> Path:
    payload = build_impact_viewer_payload(
        rows,
        summary,
        registry_dir=registry_dir,
        event_dir=event_dir,
        flood_spatial_stride=flood_spatial_stride,
        flood_time_stride=flood_time_stride,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_impact_viewer_html(payload), encoding="utf-8")
    return output


def impact_viewer_url(host: str, port: int, viewer_path: Path = DEFAULT_IMPACT_VIEWER_OUTPUT) -> str:
    rel = viewer_path.resolve().relative_to(POWER_GRID.resolve()).as_posix()
    return f"http://{host}:{port}/{rel}"


def add_impact_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--event-dir", type=Path, default=DEFAULT_EVENT_DIR)
    parser.add_argument("--event-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--probability-threshold", type=float, default=DEFAULT_PROBABILITY_THRESHOLD)
    parser.add_argument("--max-sample-distance-m", type=float, default=DEFAULT_MAX_SAMPLE_DISTANCE_M)
    parser.add_argument("--include-lines", action="store_true")
    parser.add_argument("--viewer-output", type=Path, default=DEFAULT_IMPACT_VIEWER_OUTPUT)
    parser.add_argument("--flood-spatial-stride", type=int, default=DEFAULT_FLOOD_SPATIAL_STRIDE)
    parser.add_argument("--flood-time-stride", type=int, default=DEFAULT_FLOOD_TIME_STRIDE)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute fragility-based power-grid asset impacts for a SFINCS event.")
    add_impact_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    event_id = args.event_id or args.event_dir.name
    output_dir = args.output_dir or IMPACTS_DIR / event_id
    rows, summary = compute_asset_impacts(
        args.event_dir,
        event_id=event_id,
        probability_threshold=args.probability_threshold,
        max_sample_distance_m=args.max_sample_distance_m,
        include_lines=args.include_lines,
    )
    write_outputs(rows, summary, output_dir)
    viewer_path = write_impact_viewer(
        rows,
        summary,
        args.viewer_output,
        event_dir=args.event_dir,
        flood_spatial_stride=args.flood_spatial_stride,
        flood_time_stride=args.flood_time_stride,
    )
    print(f"Wrote {len(rows):,} asset impacts to {output_dir / 'asset_impacts.csv'}")
    print(f"Wrote impact map to {viewer_path}")
    print(f"Expected affected assets: {summary['expected_affected_count']:.2f}")
    print(f"Affected assets at p >= {args.probability_threshold:g}: {summary['affected_count']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
