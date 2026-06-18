"""Export the Stage A2 event-conditioned simulation spine.

Reads validated Stage A1 artifacts and writes:

    locations/marshfield/data/power_grid/augmented/asset_states.parquet
    locations/marshfield/data/power_grid/augmented/telemetry_observations.parquet
    locations/marshfield/data/power_grid/augmented/run_manifest_stage_a2.json
    locations/marshfield/data/power_grid/augmented/validation_report_stage_a2.json

Run:
    python -m power.export_stage_a2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from power.artifact_io import count_by
from power.artifact_io import maybe_sha256
from power.artifact_io import read_parquet
from power.artifact_io import require_pyarrow
from power.artifact_io import sha256
from power.artifact_io import short_hash
from power.artifact_io import write_debug_csv
from power.artifact_io import write_parquet
from power.export_stage_a1 import (
    DEFAULT_OUTPUT_DIR,
    PROTOCOL_VERSION,
    SANDBOX_ID,
    finite_lon_lat,
    git_info,
    validation_error,
)
from power.fragility import failure_probability

ASSET_STATES_SCHEMA_VERSION = "stage_a2_asset_states.v0.1"
TELEMETRY_SCHEMA_VERSION = "stage_a2_telemetry_observations.v0.1"
DEFAULT_EVENT_ID = "marshfield_synthetic_coastal_001"
DEFAULT_ROOT_SEED = 20260509
DEFAULT_MC_DRAWS = 50
DEFAULT_SYNTHETIC_PEAK_DEPTH_M = 1.25
DEFAULT_EVENT_TIMESTAMP = "2026-01-01T00:00:00+00:00"
DEFAULT_MAX_SAMPLE_DISTANCE_M = 150.0


def parse_utc_timestamp(value: str) -> datetime:
    cleaned = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def asset_states_schema() -> Any:
    pa, _ = require_pyarrow()
    return pa.schema(
        [
            ("sandbox_id", pa.string()),
            ("event_id", pa.string()),
            ("mc_draw", pa.int32()),
            ("timestamp", pa.timestamp("us", tz="UTC")),
            ("asset_id", pa.string()),
            ("state", pa.string()),
            ("failure_probability", pa.float64()),
            ("sampled_depth_m", pa.float64()),
            ("failure_model", pa.string()),
            ("failure_model_version", pa.string()),
            ("rng_seed", pa.int64()),
            ("source_provenance", pa.string()),
            ("schema_version", pa.string()),
        ]
    )


def telemetry_observations_schema() -> Any:
    pa, _ = require_pyarrow()
    return pa.schema(
        [
            ("sandbox_id", pa.string()),
            ("event_id", pa.string()),
            ("mc_draw", pa.int32()),
            ("timestamp_observed", pa.timestamp("us", tz="UTC")),
            ("timestamp_delivered", pa.timestamp("us", tz="UTC")),
            ("target_type", pa.string()),
            ("target_id", pa.string()),
            ("observation_source", pa.string()),
            ("measured_quantity", pa.string()),
            ("value", pa.float64()),
            ("unit", pa.string()),
            ("noise_model", pa.string()),
            ("delay_model", pa.string()),
            ("observability_tier", pa.string()),
            ("rng_seed", pa.int64()),
            ("source_provenance", pa.string()),
            ("schema_version", pa.string()),
        ]
    )


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def event_seed(root_seed: int, event_id: str) -> int:
    return stable_seed("event", root_seed, event_id)


def draw_seed(root_seed: int, event_id: str, mc_draw: int) -> int:
    return stable_seed("draw", event_seed(root_seed, event_id), mc_draw)


def asset_state_seed(root_seed: int, event_id: str, mc_draw: int, asset_id: str, timestamp: datetime) -> int:
    return stable_seed("asset_state", draw_seed(root_seed, event_id, mc_draw), asset_id, timestamp.isoformat())


def observation_seed(root_seed: int, event_id: str, mc_draw: int, target_id: str, timestamp: datetime) -> int:
    return stable_seed("observation", draw_seed(root_seed, event_id, mc_draw), target_id, timestamp.isoformat())


def synthetic_depth(asset: dict[str, Any], *, lon_min: float, lon_max: float, peak_depth_m: float) -> float:
    lon = float(asset["lon"])
    span = max(lon_max - lon_min, 1.0e-9)
    eastness = min(max((lon - lon_min) / span, 0.0), 1.0)
    jitter_seed = stable_seed("synthetic_depth", asset["asset_id"])
    jitter = 0.85 + 0.30 * (jitter_seed / float((1 << 63) - 1))
    return max(0.0, peak_depth_m * (0.15 + 0.85 * eastness) * jitter)


def flood_relevant_assets(assets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in assets
        if row.get("is_flood_relevant")
        and row.get("coordinate_status") == "valid"
        and finite_lon_lat(row.get("lon"), row.get("lat"))
    ]
    return sorted(rows, key=lambda row: row["asset_id"])


def build_synthetic_event_samples(
    assets: list[dict[str, Any]],
    *,
    timestamp: datetime,
    peak_depth_m: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    lons = [float(row["lon"]) for row in assets]
    lon_min = min(lons)
    lon_max = max(lons)
    samples: dict[str, list[dict[str, Any]]] = {}
    for asset in assets:
        samples[asset["asset_id"]] = [
            {
                "timestamp": timestamp,
                "sampled_depth_m": synthetic_depth(
                    asset,
                    lon_min=lon_min,
                    lon_max=lon_max,
                    peak_depth_m=peak_depth_m,
                ),
            }
        ]
    metadata = {
        "event_source_kind": "synthetic_coordinate_profile",
        "timestamp": timestamp.isoformat(),
        "peak_depth_m": peak_depth_m,
        "description": (
            "Deterministic peak-depth fixture for Stage A2 pipeline testing. "
            "Use --event-depth-csv for sampled FLOOD-RM/SFINCS event depths."
        ),
    }
    return samples, metadata


def build_csv_event_samples(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"asset_id", "timestamp", "sampled_depth_m"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            samples[row["asset_id"]].append(
                {
                    "timestamp": parse_utc_timestamp(row["timestamp"]),
                    "sampled_depth_m": float(row["sampled_depth_m"]),
                }
            )
    for rows in samples.values():
        rows.sort(key=lambda row: row["timestamp"])
    metadata = {
        "event_source_kind": "sampled_asset_depth_csv",
        "path": str(path),
        "sha256": sha256(path),
        "required_columns": sorted(required),
        "description": "Asset-keyed flood depths sampled from an external FLOOD-RM/SFINCS workflow.",
    }
    return dict(samples), metadata


def require_geo_stack():
    try:
        import numpy as np
        import xarray as xr
        from pyproj import Transformer
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "SFINCS event sampling requires numpy, xarray, pyproj, and scipy. "
            "Install the base project dependencies with `uv sync` first."
        ) from exc
    return np, xr, Transformer, cKDTree


def sfincs_peak_depth_grid(dataset: Any) -> Any:
    np, _, _, _ = require_geo_stack()
    if "hmax" in dataset:
        return np.asarray(dataset["hmax"].values, dtype=np.float32)
    if "zsmax" in dataset and "zb" in dataset:
        zsmax = np.asarray(dataset["zsmax"].values, dtype=np.float32)
        zb = np.asarray(dataset["zb"].values, dtype=np.float32)
        # Real FLOOD-RM/SFINCS outputs store `zsmax` with a leading `timemax` axis
        # (per-interval running maximum); reduce across it so the returned grid
        # matches the 2-D (n, m) shape of `x`, `y`, and `msk`.
        if zsmax.ndim == 3:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                zsmax = np.nanmax(zsmax, axis=0)
        return np.maximum(zsmax - zb, 0.0)
    if "zs" in dataset and "zb" in dataset:
        depth = (
            np.asarray(dataset["zs"].values, dtype=np.float32)
            - np.asarray(dataset["zb"].values, dtype=np.float32)[None, :, :]
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            return np.nanmax(np.maximum(depth, 0.0), axis=0)
    raise RuntimeError("sfincs_map.nc must contain hmax, zsmax+zb, or zs+zb")


OUT_OF_MESH_DEPTH_M = 0.0


def clip_out_of_mesh_depths(depths):
    """Replace NaN sampled depths with the out-of-mesh sentinel and count them.

    SFINCS produces depths only inside its active mesh. Assets outside the
    mesh's max-sample-distance horizon have no modeled flood exposure for this
    event — they are not "invalid data" but "no flood in this scenario".
    Lift that interpretation into the depth itself so Stage A2 validation does
    not flag legitimate out-of-mesh assets.
    """

    import numpy as np

    array = np.asarray(depths, dtype=float)
    nan_mask = ~np.isfinite(array)
    cleaned = np.where(nan_mask, OUT_OF_MESH_DEPTH_M, array)
    return cleaned, int(nan_mask.sum())


def build_sfincs_event_samples(
    event_dir: Path,
    assets: list[dict[str, Any]],
    *,
    max_sample_distance_m: float,
    xarray_engine: str | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    np, xr, Transformer, cKDTree = require_geo_stack()
    map_path = event_dir / "sfincs_map.nc"
    if not map_path.exists():
        raise FileNotFoundError(map_path)

    open_kwargs = {"engine": xarray_engine} if xarray_engine is not None else {}
    with xr.open_dataset(map_path, **open_kwargs) as ds:
        for name in ["x", "y", "msk"]:
            if name not in ds:
                raise RuntimeError(f"{map_path} missing {name}")
        active = np.asarray(ds["msk"].values, dtype=float) > 0
        peak_depth = sfincs_peak_depth_grid(ds)
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y) & active & np.isfinite(peak_depth)
    grid_xy = np.column_stack([x[valid], y[valid]])
    if len(grid_xy) == 0:
        raise RuntimeError(f"{map_path} has no active finite depth cells")
    tree = cKDTree(grid_xy)
    to_model = Transformer.from_crs("EPSG:4326", "EPSG:26919", always_xy=True)
    asset_xy = np.asarray([(to_model.transform(asset["lon"], asset["lat"])) for asset in assets], dtype=float)
    distances, nearest = tree.query(asset_xy, k=1)
    flat_depth = peak_depth[valid].astype(float)
    depths = flat_depth[nearest]
    depths = np.where(distances <= float(max_sample_distance_m), depths, np.nan)
    depths, out_of_mesh_count = clip_out_of_mesh_depths(depths)
    timestamp = parse_utc_timestamp(DEFAULT_EVENT_TIMESTAMP)

    samples: dict[str, list[dict[str, Any]]] = {}
    for asset, depth_m in zip(assets, depths, strict=True):
        samples[asset["asset_id"]] = [
            {
                "timestamp": timestamp,
                "sampled_depth_m": float(depth_m),
            }
        ]
    metadata = {
        "event_source_kind": "sfincs_map_peak_depth",
        "event_dir": str(event_dir),
        "sfincs_map_path": str(map_path),
        "sfincs_map_sha256": sha256(map_path),
        "max_sample_distance_m": max_sample_distance_m,
        "asset_crs": "EPSG:4326",
        "sfincs_model_crs_assumption": "EPSG:26919",
        "timestamp": timestamp.isoformat(),
        "out_of_mesh_asset_count": out_of_mesh_count,
        "out_of_mesh_depth_m": OUT_OF_MESH_DEPTH_M,
        "description": (
            "Nearest-cell peak flood depth sampled from a completed SFINCS "
            "sfincs_map.nc output. Assets farther than max_sample_distance_m "
            "from any active SFINCS cell are out-of-mesh for this event and "
            "are assigned sampled_depth_m = OUT_OF_MESH_DEPTH_M (no modeled "
            "flood exposure)."
        ),
    }
    return samples, metadata


def asset_state_provenance(asset: dict[str, Any], event_metadata: dict[str, Any]) -> str:
    return json.dumps(
        {
            "asset_source_provenance": asset.get("source_provenance", ""),
            "event_source_kind": event_metadata["event_source_kind"],
        },
        sort_keys=True,
    )


def build_asset_states(
    assets: list[dict[str, Any]],
    event_samples: dict[str, list[dict[str, Any]]],
    event_metadata: dict[str, Any],
    *,
    event_id: str,
    root_seed: int,
    mc_draws: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, datetime, str], int]]:
    rows: list[dict[str, Any]] = []
    binary_state_by_key: dict[tuple[int, datetime, str], int] = {}
    for mc_draw in range(mc_draws):
        for asset in assets:
            asset_id = asset["asset_id"]
            for sample in event_samples.get(asset_id, []):
                timestamp = sample["timestamp"]
                depth_m = float(sample["sampled_depth_m"])
                probability = failure_probability(asset["asset_type"], depth_m)
                seed = asset_state_seed(root_seed, event_id, mc_draw, asset_id, timestamp)
                failed = random.Random(seed).random() <= probability
                state = "failed" if failed else "available"
                binary_state_by_key[(mc_draw, timestamp, asset_id)] = int(failed)
                rows.append(
                    {
                        "sandbox_id": SANDBOX_ID,
                        "event_id": event_id,
                        "mc_draw": mc_draw,
                        "timestamp": timestamp,
                        "asset_id": asset_id,
                        "state": state,
                        "failure_probability": probability,
                        "sampled_depth_m": depth_m,
                        "failure_model": "erad_flood_depth_lognormal",
                        "failure_model_version": "erad_v0.1.11_mapping_v0.1",
                        "rng_seed": seed,
                        "source_provenance": asset_state_provenance(asset, event_metadata),
                        "schema_version": ASSET_STATES_SCHEMA_VERSION,
                    }
                )
    return rows, binary_state_by_key


def telemetry_provenance(kind: str, source_ids: list[str] | None = None) -> str:
    return json.dumps(
        {
            "telemetry_synthesis": kind,
            "source_asset_ids": sorted(source_ids or []),
        },
        sort_keys=True,
    )


def observation_delay(seed: int) -> timedelta:
    return timedelta(minutes=seed % 16)


def build_telemetry_observations(
    assets: list[dict[str, Any]],
    control_units: list[dict[str, Any]],
    binary_state_by_key: dict[tuple[int, datetime, str], int],
    *,
    event_id: str,
    root_seed: int,
    mc_draws: int,
) -> list[dict[str, Any]]:
    flood_asset_ids = {asset["asset_id"] for asset in assets}
    assets_by_id = {asset["asset_id"]: asset for asset in assets}
    timestamps = sorted({key[1] for key in binary_state_by_key})
    asset_targets = [
        asset
        for asset in assets
        if asset["asset_type"] in {"source", "transformer", "fuse_proxy"}
    ]
    rows: list[dict[str, Any]] = []
    for mc_draw in range(mc_draws):
        for timestamp in timestamps:
            for asset in asset_targets:
                asset_id = asset["asset_id"]
                seed = observation_seed(root_seed, event_id, mc_draw, asset_id, timestamp)
                delivered = timestamp + observation_delay(seed)
                rows.append(
                    {
                        "sandbox_id": SANDBOX_ID,
                        "event_id": event_id,
                        "mc_draw": mc_draw,
                        "timestamp_observed": timestamp,
                        "timestamp_delivered": delivered,
                        "target_type": "asset",
                        "target_id": asset_id,
                        "observation_source": "synthetic_scada_oms",
                        "measured_quantity": "asset_failure_state",
                        "value": float(binary_state_by_key.get((mc_draw, timestamp, asset_id), 0)),
                        "unit": "binary",
                        "noise_model": "none",
                        "delay_model": "deterministic_hash_0_to_15min",
                        "observability_tier": "tier_1_scada_oms",
                        "rng_seed": seed,
                        "source_provenance": telemetry_provenance("direct_asset_state", [asset_id]),
                        "schema_version": TELEMETRY_SCHEMA_VERSION,
                    }
                )
            for unit in control_units:
                member_ids = [
                    asset_id
                    for asset_id in unit["member_asset_ids"]
                    if asset_id in flood_asset_ids and asset_id in assets_by_id
                ]
                failed_count = sum(
                    binary_state_by_key.get((mc_draw, timestamp, asset_id), 0)
                    for asset_id in member_ids
                )
                value = failed_count / len(member_ids) if member_ids else 0.0
                seed = observation_seed(root_seed, event_id, mc_draw, unit["control_unit_id"], timestamp)
                delivered = timestamp + observation_delay(seed)
                rows.append(
                    {
                        "sandbox_id": SANDBOX_ID,
                        "event_id": event_id,
                        "mc_draw": mc_draw,
                        "timestamp_observed": timestamp,
                        "timestamp_delivered": delivered,
                        "target_type": "control_unit",
                        "target_id": unit["control_unit_id"],
                        "observation_source": "synthetic_control_unit_aggregator",
                        "measured_quantity": "failed_asset_fraction",
                        "value": float(value),
                        "unit": "fraction",
                        "noise_model": "none",
                        "delay_model": "deterministic_hash_0_to_15min",
                        "observability_tier": "tier_2_feeder_summary",
                        "rng_seed": seed,
                        "source_provenance": telemetry_provenance("control_unit_fraction", member_ids),
                        "schema_version": TELEMETRY_SCHEMA_VERSION,
                    }
                )
    return rows


def binary_state_signature(asset_states: Iterable[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in asset_states:
        timestamp = row["timestamp"].isoformat()
        payload = f"{row['asset_id']}|{row['mc_draw']}|{timestamp}|{row['state']}|{row['rng_seed']}\n"
        h.update(payload.encode("utf-8"))
    return h.hexdigest()


def validate_event_samples(
    assets: list[dict[str, Any]],
    event_samples: dict[str, list[dict[str, Any]]],
    report: dict[str, Any],
) -> None:
    expected_ids = {asset["asset_id"] for asset in assets}
    sample_ids = set(event_samples)
    unknown = sorted(sample_ids - expected_ids)
    missing = sorted(expected_ids - sample_ids)
    if unknown:
        validation_error(report, f"event depth samples reference unknown assets: {unknown[:10]}")
    if missing:
        validation_error(report, f"event depth samples missing flood-relevant assets: {missing[:10]}")
    for asset_id, samples in event_samples.items():
        for sample in samples:
            depth = sample["sampled_depth_m"]
            if not math.isfinite(depth) or depth < 0.0:
                validation_error(report, f"{asset_id}: sampled_depth_m must be finite and non-negative")
    report["checks"]["event_sample_asset_count"] = len(sample_ids & expected_ids)
    report["checks"]["event_sample_timestamp_count"] = len(
        {sample["timestamp"].isoformat() for samples in event_samples.values() for sample in samples}
    )


def validate_asset_states(
    asset_states: list[dict[str, Any]],
    known_asset_ids: set[str],
    report: dict[str, Any],
) -> None:
    for row in asset_states:
        asset_id = row["asset_id"]
        probability = row["failure_probability"]
        if asset_id not in known_asset_ids:
            validation_error(report, f"{asset_id}: asset state references unknown asset")
        if not (0.0 <= probability <= 1.0):
            validation_error(report, f"{asset_id}: failure_probability outside [0, 1]")
        if row["state"] not in {"available", "failed"}:
            validation_error(report, f"{asset_id}: invalid state {row['state']!r}")
        if row["rng_seed"] is None:
            validation_error(report, f"{asset_id}: missing rng_seed")
    report["checks"]["asset_state_count"] = len(asset_states)
    report["checks"]["asset_state_counts_by_state"] = count_by(asset_states, "state")
    report["checks"]["binary_state_signature"] = binary_state_signature(asset_states)


def validate_telemetry_observations(
    telemetry: list[dict[str, Any]],
    known_asset_ids: set[str],
    known_control_unit_ids: set[str],
    report: dict[str, Any],
) -> None:
    for row in telemetry:
        target_id = row["target_id"]
        if row["target_type"] == "asset" and target_id not in known_asset_ids:
            validation_error(report, f"{target_id}: telemetry references unknown asset")
        elif row["target_type"] == "control_unit" and target_id not in known_control_unit_ids:
            validation_error(report, f"{target_id}: telemetry references unknown Control Unit")
        elif row["target_type"] not in {"asset", "control_unit"}:
            validation_error(report, f"{target_id}: invalid telemetry target_type")
        if row["unit"] == "":
            validation_error(report, f"{target_id}: telemetry unit is missing")
        if row["rng_seed"] is None:
            validation_error(report, f"{target_id}: telemetry rng_seed is missing")
    report["checks"]["telemetry_observation_count"] = len(telemetry)
    report["checks"]["telemetry_counts_by_target_type"] = count_by(telemetry, "target_type")
    report["checks"]["telemetry_counts_by_quantity"] = count_by(telemetry, "measured_quantity")


def load_a1_validation(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "validation_report.json"
    if not path.exists():
        return {"passed": False, "errors": [f"Missing Stage A1 validation report: {path}"]}
    return json.loads(path.read_text())


def dependency_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "pyarrow", "opendssdirect.py", "xarray", "geopandas", "shapely"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def build_manifest(
    output_dir: Path,
    event_metadata: dict[str, Any],
    outputs: dict[str, Path],
    debug_outputs: dict[str, Path],
    *,
    event_id: str,
    root_seed: int,
    mc_draws: int,
) -> dict[str, Any]:
    stage_a1_inputs = {
        "assets.parquet": output_dir / "assets.parquet",
        "control_units.parquet": output_dir / "control_units.parquet",
        "run_manifest.json": output_dir / "run_manifest.json",
        "validation_report.json": output_dir / "validation_report.json",
    }
    input_hashes = {
        name: {"path": str(path), "sha256": maybe_sha256(path)}
        for name, path in stage_a1_inputs.items()
    }
    run_hash = short_hash(
        {
            "inputs": input_hashes,
            "event": event_metadata,
            "root_seed": root_seed,
            "mc_draws": mc_draws,
        }
    )
    return {
        "run_id": f"{SANDBOX_ID}:run:{PROTOCOL_VERSION}:{event_id}:seed_{root_seed}:{run_hash}",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sandbox_id": SANDBOX_ID,
        "stage": "stage_a2",
        "protocol_version": PROTOCOL_VERSION,
        "schema_versions": {
            "asset_states.parquet": ASSET_STATES_SCHEMA_VERSION,
            "telemetry_observations.parquet": TELEMETRY_SCHEMA_VERSION,
        },
        "event_id": event_id,
        "root_seed": root_seed,
        "mc_draws": mc_draws,
        "rng_family": "python.random.Random",
        "seed_hierarchy": (
            "root_seed -> event_seed(event_id) -> draw_seed(event_id, mc_draw) -> "
            "asset_state_seed(event_id, mc_draw, asset_id, timestamp) and "
            "observation_seed(event_id, mc_draw, target_id, timestamp)"
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_info(),
        "dependencies": dependency_versions(),
        "inputs": input_hashes,
        "event_source": event_metadata,
        "outputs": {
            name: {"path": str(path), "sha256": maybe_sha256(path)}
            for name, path in outputs.items()
        },
        "debug_outputs": {
            name: {"path": str(path), "sha256": maybe_sha256(path)}
            for name, path in debug_outputs.items()
        }
    }


def export_stage_a2(
    output_dir: Path,
    *,
    event_id: str = DEFAULT_EVENT_ID,
    root_seed: int = DEFAULT_ROOT_SEED,
    mc_draws: int = DEFAULT_MC_DRAWS,
    event_depth_csv: Path | None = None,
    sfincs_event_dir: Path | None = None,
    max_sample_distance_m: float = DEFAULT_MAX_SAMPLE_DISTANCE_M,
    synthetic_peak_depth_m: float = DEFAULT_SYNTHETIC_PEAK_DEPTH_M,
    event_timestamp: str = DEFAULT_EVENT_TIMESTAMP,
    debug_csv: bool = False,
) -> dict[str, Any]:
    assets = read_parquet(output_dir / "assets.parquet")
    control_units = read_parquet(output_dir / "control_units.parquet")
    flood_assets = flood_relevant_assets(assets)
    if event_depth_csv and sfincs_event_dir:
        raise ValueError("Use either --event-depth-csv or --sfincs-event-dir, not both")
    if event_depth_csv:
        event_samples, event_metadata = build_csv_event_samples(event_depth_csv)
    elif sfincs_event_dir:
        event_samples, event_metadata = build_sfincs_event_samples(
            sfincs_event_dir,
            flood_assets,
            max_sample_distance_m=max_sample_distance_m,
        )
    else:
        event_samples, event_metadata = build_synthetic_event_samples(
            flood_assets,
            timestamp=parse_utc_timestamp(event_timestamp),
            peak_depth_m=synthetic_peak_depth_m,
        )

    asset_states, binary_state_by_key = build_asset_states(
        flood_assets,
        event_samples,
        event_metadata,
        event_id=event_id,
        root_seed=root_seed,
        mc_draws=mc_draws,
    )
    telemetry = build_telemetry_observations(
        flood_assets,
        control_units,
        binary_state_by_key,
        event_id=event_id,
        root_seed=root_seed,
        mc_draws=mc_draws,
    )

    asset_states_path = output_dir / "asset_states.parquet"
    telemetry_path = output_dir / "telemetry_observations.parquet"
    write_parquet(asset_states_path, asset_states, asset_states_schema())
    write_parquet(telemetry_path, telemetry, telemetry_observations_schema())

    debug_outputs: dict[str, Path] = {}
    if debug_csv:
        asset_states_debug = output_dir / "asset_states.debug.csv"
        telemetry_debug = output_dir / "telemetry_observations.debug.csv"
        write_debug_csv(asset_states_debug, asset_states, [field.name for field in asset_states_schema()])
        write_debug_csv(telemetry_debug, telemetry, [field.name for field in telemetry_observations_schema()])
        debug_outputs = {
            "asset_states.debug.csv": asset_states_debug,
            "telemetry_observations.debug.csv": telemetry_debug,
        }

    report: dict[str, Any] = {
        "stage": "stage_a2",
        "schema_versions": {
            "asset_states.parquet": ASSET_STATES_SCHEMA_VERSION,
            "telemetry_observations.parquet": TELEMETRY_SCHEMA_VERSION,
        },
        "event_id": event_id,
        "root_seed": root_seed,
        "mc_draws": mc_draws,
        "passed": False,
        "errors": [],
        "checks": {},
    }
    a1_report = load_a1_validation(output_dir)
    if not a1_report.get("passed"):
        validation_error(report, "Gate A1 has not passed for the referenced inputs")
    report["checks"]["gate_a1_passed"] = bool(a1_report.get("passed"))
    report["checks"]["flood_relevant_asset_count"] = len(flood_assets)
    report["checks"]["mc_draw_count"] = mc_draws
    validate_event_samples(flood_assets, event_samples, report)
    validate_asset_states(asset_states, {row["asset_id"] for row in assets}, report)
    validate_telemetry_observations(
        telemetry,
        {row["asset_id"] for row in assets},
        {row["control_unit_id"] for row in control_units},
        report,
    )
    report["passed"] = not report["errors"]

    outputs = {
        "asset_states.parquet": asset_states_path,
        "telemetry_observations.parquet": telemetry_path,
    }
    manifest = build_manifest(
        output_dir,
        event_metadata,
        outputs,
        debug_outputs,
        event_id=event_id,
        root_seed=root_seed,
        mc_draws=mc_draws,
    )
    (output_dir / "run_manifest_stage_a2.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "validation_report_stage_a2.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--event-id", default=DEFAULT_EVENT_ID)
    parser.add_argument("--root-seed", type=int, default=DEFAULT_ROOT_SEED)
    parser.add_argument("--mc-draws", type=int, default=DEFAULT_MC_DRAWS)
    parser.add_argument("--event-depth-csv", type=Path)
    parser.add_argument("--sfincs-event-dir", type=Path)
    parser.add_argument("--max-sample-distance-m", type=float, default=DEFAULT_MAX_SAMPLE_DISTANCE_M)
    parser.add_argument("--synthetic-peak-depth-m", type=float, default=DEFAULT_SYNTHETIC_PEAK_DEPTH_M)
    parser.add_argument("--event-timestamp", default=DEFAULT_EVENT_TIMESTAMP)
    parser.add_argument(
        "--debug-csv",
        action="store_true",
        help="Write optional .debug.csv exports derived from canonical Parquet outputs.",
    )
    args = parser.parse_args()
    report = export_stage_a2(
        args.output_dir,
        event_id=args.event_id,
        root_seed=args.root_seed,
        mc_draws=args.mc_draws,
        event_depth_csv=args.event_depth_csv,
        sfincs_event_dir=args.sfincs_event_dir,
        max_sample_distance_m=args.max_sample_distance_m,
        synthetic_peak_depth_m=args.synthetic_peak_depth_m,
        event_timestamp=args.event_timestamp,
        debug_csv=args.debug_csv,
    )
    status = "passed" if report["passed"] else "failed"
    print(f"Stage A2 export {status}: {args.output_dir}")
    for key, value in report["checks"].items():
        print(f"  {key}: {value}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
