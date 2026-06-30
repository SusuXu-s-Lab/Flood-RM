"""Notebook-facing flood fragility and asset-impact helpers."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from paths import default_location_config_path, find_repo_root
from power._impact_core import AssetPoint
from power._impact_core import FloodDepthFragilityCurve
from power._impact_core import csv_failure_probability
from power._impact_core import erad_asset_type as _erad_asset_type
from power._impact_core import line_local_asset_type
from power._impact_core import load_asset_points as _load_asset_points
from power._impact_core import load_asset_type_mapping as _load_asset_type_mapping
from power._impact_core import load_flood_depth_curves as _load_flood_depth_curves
from power._impact_core import sample_sfincs_peak_depths
from power._impact_core import summarize_impacts
from study_location import define_location


repo_root = find_repo_root(Path(__file__).resolve())


def power_grid_root() -> Path:
    try:
        definition = define_location(default_location_config_path(repo_root))
    except (FileNotFoundError, IndexError, ValueError):
        return Path("data/power_grid")
    path = Path(definition.grid.get("power_grid_root", "data/power_grid"))
    return path if path.is_absolute() else definition.root / path


shared_fragility_dir = repo_root / "artifacts" / "fragility"
default_curves_csv = shared_fragility_dir / "erad_flood_depth_curves.csv"
power_grid = Path("data/power_grid")
smart_ds_compat = power_grid / "augmented"
fragility_dir = power_grid / "fragility"
default_mapping_csv = fragility_dir / "asset_type_mapping.csv"
impacts_dir = power_grid / "figures" / "impacts"
default_event_dir = power_grid / "sfincs_truth" / "run_outputs" / "single_event_tests" / "riley_90m"
default_probability_threshold = 0.50
default_max_sample_distance_m = 150.0


@lru_cache(maxsize=None)
def load_flood_depth_curves(path=default_curves_csv):
    return _load_flood_depth_curves(path)


def _default_mapping_csv() -> Path:
    return power_grid_root() / "fragility" / "asset_type_mapping.csv"


def _default_smart_ds_compat() -> Path:
    return power_grid_root() / "augmented"


def _default_impacts_dir() -> Path:
    return power_grid_root() / "figures" / "impacts"


@lru_cache(maxsize=None)
def load_asset_type_mapping(path=None):
    return _load_asset_type_mapping(_default_mapping_csv() if path is None else path)


def erad_asset_type(local_asset_type, mapping=None):
    return _erad_asset_type(local_asset_type, mapping or load_asset_type_mapping())


def failure_probability(local_asset_type, depth_m, *, curves=None, mapping=None):
    return csv_failure_probability(
        local_asset_type,
        depth_m,
        curves=curves or load_flood_depth_curves(),
        mapping=mapping or load_asset_type_mapping(),
    )


def load_asset_points(registry_dir=None, *, include_lines=False):
    registry_dir = _default_smart_ds_compat() if registry_dir is None else registry_dir
    return _load_asset_points(registry_dir, include_lines=include_lines)


def compute_asset_impacts(
    event_dir,
    *,
    event_id=None,
    probability_threshold=default_probability_threshold,
    max_sample_distance_m=default_max_sample_distance_m,
    include_lines=False,
    registry_dir=None,
):
    registry_dir = _default_smart_ds_compat() if registry_dir is None else registry_dir
    assets = load_asset_points(registry_dir, include_lines=include_lines)
    depths, distances = sample_sfincs_peak_depths(
        event_dir,
        assets,
        max_sample_distance_m=max_sample_distance_m,
    )
    curves = load_flood_depth_curves()
    mapping = load_asset_type_mapping()

    rows = []
    for asset, depth_m, distance_m in zip(assets, depths, distances, strict=True):
        probability = failure_probability(asset.asset_type, depth_m, curves=curves, mapping=mapping)
        rows.append(
            {
                "event_id": event_id or Path(event_dir).name,
                "asset_id": asset.asset_id,
                "asset_type": asset.asset_type,
                "erad_asset_type": mapping[asset.asset_type],
                "feeder_id": asset.feeder_id,
                "lon": asset.lon,
                "lat": asset.lat,
                "peak_depth_m": depth_m,
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


def write_outputs(rows, summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "asset_impacts.csv"
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_asset_impacts(
    event_dir,
    *,
    event_id=None,
    output_dir=None,
    probability_threshold=default_probability_threshold,
    max_sample_distance_m=default_max_sample_distance_m,
    include_lines=False,
):
    event_path = Path(event_dir)
    event_id = event_id or event_path.name
    output_dir = output_dir or _default_impacts_dir() / event_id
    rows, summary = compute_asset_impacts(
        event_path,
        event_id=event_id,
        probability_threshold=probability_threshold,
        max_sample_distance_m=max_sample_distance_m,
        include_lines=include_lines,
    )
    write_outputs(rows, summary, output_dir)
    return rows, summary


fragility = SimpleNamespace(
    FloodDepthFragilityCurve=FloodDepthFragilityCurve,
    line_local_asset_type=line_local_asset_type,
    load_flood_depth_curves=load_flood_depth_curves,
    load_asset_type_mapping=load_asset_type_mapping,
    erad_asset_type=erad_asset_type,
    failure_probability=failure_probability,
)

analysis = SimpleNamespace(
    AssetPoint=AssetPoint,
    load_asset_points=load_asset_points,
    compute_asset_impacts=compute_asset_impacts,
    summarize_impacts=summarize_impacts,
    write_outputs=write_outputs,
    run_asset_impacts=run_asset_impacts,
)
