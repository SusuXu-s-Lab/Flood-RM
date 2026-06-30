"""Notebook skin for flood exposure and fragility-based asset impacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from paths import default_location_config_path, find_repo_root
from power.impact.fragility import (
    failure_probability,
    load_asset_type_mapping,
    load_flood_depth_curves,
)
from power_v2.impact import AssetPoint
from power_v2.impact import load_asset_points as _load_v2_asset_points
from power_v2.impact import sample_sfincs_peak_depths
from power_v2.impact import summarize_impacts
from study_location import define_location

repo_root = find_repo_root(Path(__file__).resolve())


def power_grid_root():
    definition = define_location(default_location_config_path(repo_root))
    path = Path(definition.grid.get("power_grid_root", "data/power_grid"))
    return path if path.is_absolute() else definition.root / path


power_grid = power_grid_root()
smart_ds_compat = power_grid / "augmented"
impacts_dir = power_grid / "figures" / "impacts"
default_event_dir = power_grid / "sfincs_truth" / "run_outputs" / "single_event_tests" / "riley_90m"
default_probability_threshold = 0.50
default_max_sample_distance_m = 150.0


def load_asset_points(registry_dir=smart_ds_compat, *, include_lines=False):
    """Flood-relevant point assets from the Grid Dataset artifacts."""

    return _load_v2_asset_points(registry_dir, include_lines=include_lines)


def compute_asset_impacts(
    event_dir,
    *,
    event_id=None,
    probability_threshold=default_probability_threshold,
    max_sample_distance_m=default_max_sample_distance_m,
    include_lines=False,
    registry_dir=smart_ds_compat,
):
    """Per-asset flood exposure and CSV-backed fragility impact rows."""

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
    """Fragility-based asset impacts, written as CSV/summary outputs."""

    event_path = Path(event_dir)
    event_id = event_id or event_path.name
    output_dir = output_dir or impacts_dir / event_id
    rows, summary = compute_asset_impacts(
        event_path,
        event_id=event_id,
        probability_threshold=probability_threshold,
        max_sample_distance_m=max_sample_distance_m,
        include_lines=include_lines,
    )
    write_outputs(rows, summary, output_dir)
    return rows, summary
