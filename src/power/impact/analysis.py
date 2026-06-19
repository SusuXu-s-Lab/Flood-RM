# Asset impact analysis

"""Compute flood exposure and fragility-based affected assets."""


import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

from power.artifacts import finite_float as _finite_float
from power.artifacts import read_csv as _read_csv
from power.artifacts import read_parquet
from power.artifacts import POWER_GRID
from power.impact.fragility import (
    failure_probability,
    line_local_asset_type,
    load_asset_type_mapping,
    load_flood_depth_curves,
)

SMART_DS_COMPAT = POWER_GRID / "augmented"
IMPACTS_DIR = POWER_GRID / "figures" / "impacts"
DEFAULT_EVENT_DIR = (
    POWER_GRID / "sfincs_truth" / "run_outputs" / "single_event_tests" / "riley_90m"
)
DEFAULT_PROBABILITY_THRESHOLD = 0.50
DEFAULT_MAX_SAMPLE_DISTANCE_M = 150.0


@dataclass(frozen=True)
class AssetPoint:
    asset_id: str
    asset_type: str
    feeder_id: str
    lon: float
    lat: float
    label: str


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
            "Install the project dependencies, then import power.impact."
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


def run_asset_impacts(
    event_dir: Path,
    *,
    event_id: str | None = None,
    output_dir: Path | None = None,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
    max_sample_distance_m: float = DEFAULT_MAX_SAMPLE_DISTANCE_M,
    include_lines: bool = False,
) -> tuple[list[dict], dict]:
    """Compute fragility-based asset impacts and write CSV/summary outputs."""
    event_id = event_id or event_dir.name
    output_dir = output_dir or IMPACTS_DIR / event_id
    rows, summary = compute_asset_impacts(
        event_dir,
        event_id=event_id,
        probability_threshold=probability_threshold,
        max_sample_distance_m=max_sample_distance_m,
        include_lines=include_lines,
    )
    write_outputs(rows, summary, output_dir)
    return rows, summary
