

# SMART-DS-compatible Grid Dataset exports

"""Export SMART-DS-compatible static and event-conditioned Grid Dataset artifacts.

Reads the deterministic Asset Registry CSVs and writes canonical parquet
artifacts:

    locations/marshfield/data/power_grid/augmented/assets.parquet
    locations/marshfield/data/power_grid/augmented/control_units.parquet
    locations/marshfield/data/power_grid/augmented/run_manifest*.json
    locations/marshfield/data/power_grid/augmented/validation_report*.json

Parquet is canonical. Optional ``*.debug.csv`` files are derived from the same
rows for inspection only.
"""

from __future__ import annotations

import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


from power.artifacts import count_by
from power.artifacts import finite_lon_lat
from power.artifacts import git_info
from power.artifacts import maybe_sha256
from power.artifacts import parse_float
from power.artifacts import read_csv
from power.artifacts import require_pyarrow
from power.artifacts import sha256
from power.artifacts import short_hash
from power.artifacts import stable_asset_id
from power.artifacts import stable_control_unit_id
from power.artifacts import validation_error
from power.artifacts import write_debug_csv
from power.artifacts import write_parquet
from power.artifacts import PROTOCOL_VERSION
from power.artifacts import SANDBOX_ID
from power.artifacts import POWER_GRID

DEFAULT_REGISTRY_DIR = POWER_GRID / "asset_registry"
DEFAULT_OUTPUT_DIR = POWER_GRID / "augmented"
SCHEMA_VERSION = "stage_a1.v0.1"


def midpoint(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def coordinate_fields(
    lon: float | None,
    lat: float | None,
    *,
    source: str,
    is_flood_relevant: bool,
    spatial_join_required: bool,
    exemption_reason: str = "",
) -> dict[str, Any]:
    if finite_lon_lat(lon, lat):
        status = "valid"
        reason = ""
    elif is_flood_relevant or spatial_join_required:
        status = "invalid"
        reason = ""
    else:
        status = "missing_exempt"
        reason = exemption_reason or "non_spatial_metadata"
    return {
        "lon": lon,
        "lat": lat,
        "coordinate_status": status,
        "coordinate_source": source if status != "missing_exempt" else "non_spatial_metadata",
        "is_flood_relevant": is_flood_relevant,
        "spatial_join_required": spatial_join_required,
        "coordinate_exemption_reason": reason,
    }


def source_provenance(source_table: str, source_row: dict[str, str]) -> str:
    payload = {
        "source_table": source_table,
        "source_file": source_row.get("source_file", ""),
        "source_line": source_row.get("source_line", ""),
    }
    return json.dumps(payload, sort_keys=True)


def base_asset(
    *,
    asset_type: str,
    source_table: str,
    source_name: str,
    feeder_id: str,
    bus: str,
    phases: str,
    coordinate: dict[str, Any],
    rated_kv: float | None = None,
    rated_kva: float | None = None,
    source_uuid: str = "",
    source_row: dict[str, str],
) -> dict[str, Any]:
    return {
        "sandbox_id": SANDBOX_ID,
        "asset_id": stable_asset_id(source_table, source_name),
        "asset_type": asset_type,
        "source_asset_table": source_table,
        "source_asset_name": source_name,
        "source_uuid": source_uuid,
        "feeder_id": feeder_id,
        "bus": bus,
        "phases": phases,
        **coordinate,
        "rated_kv": rated_kv,
        "rated_kva": rated_kva,
        "source_provenance": source_provenance(source_table, source_row),
        "schema_version": SCHEMA_VERSION,
    }


def build_transformer_assets(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    assets = []
    for row in rows:
        lon = parse_float(row.get("location_lon"))
        lat = parse_float(row.get("location_lat"))
        assets.append(
            base_asset(
                asset_type="transformer",
                source_table="transformers",
                source_name=row["transformer_name"],
                feeder_id=row["feeder_id"],
                bus=row["location_bus"],
                phases=row["phases"],
                coordinate=coordinate_fields(
                    lon,
                    lat,
                    source="buscoords.csv",
                    is_flood_relevant=True,
                    spatial_join_required=True,
                ),
                rated_kv=parse_float(row.get("max_kv")),
                rated_kva=parse_float(row.get("max_kva")),
                source_row=row,
            )
        )
    return assets


def build_source_assets(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    assets = []
    for row in rows:
        lon = parse_float(row.get("lon"))
        lat = parse_float(row.get("lat"))
        assets.append(
            base_asset(
                asset_type="source",
                source_table="sources",
                source_name=row["source_name"],
                feeder_id=row["feeder_id"],
                bus=row["bus"],
                phases=row["phases"],
                coordinate=coordinate_fields(
                    lon,
                    lat,
                    source="buscoords.csv",
                    is_flood_relevant=True,
                    spatial_join_required=True,
                ),
                rated_kv=parse_float(row.get("basekv")),
                rated_kva=None,
                source_row=row,
            )
        )
    return assets


def build_load_bus_assets(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    assets = []
    for row in rows:
        lon = parse_float(row.get("lon"))
        lat = parse_float(row.get("lat"))
        assets.append(
            base_asset(
                asset_type="load_bus",
                source_table="load_buses",
                source_name=row["bus"],
                feeder_id=row["feeder_id"],
                bus=row["bus"],
                phases="",
                coordinate=coordinate_fields(
                    lon,
                    lat,
                    source="buscoords.csv",
                    is_flood_relevant=True,
                    spatial_join_required=True,
                ),
                rated_kv=None,
                rated_kva=None,
                source_row=row,
            )
        )
    return assets


def build_line_assets(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    assets = []
    for row in rows:
        from_lon = parse_float(row.get("from_lon"))
        from_lat = parse_float(row.get("from_lat"))
        to_lon = parse_float(row.get("to_lon"))
        to_lat = parse_float(row.get("to_lat"))
        lon = midpoint(from_lon, to_lon)
        lat = midpoint(from_lat, to_lat)
        line_class = row.get("line_class", "")
        if line_class == "underground":
            asset_type = "underground_line_proxy"
            is_flood_relevant = True
            spatial_join_required = True
            coord_source = "line_midpoint"
            exemption = ""
        elif line_class == "fuse":
            asset_type = "fuse_proxy"
            is_flood_relevant = True
            spatial_join_required = True
            coord_source = "line_midpoint"
            exemption = ""
        elif line_class == "overhead":
            asset_type = "overhead_line"
            is_flood_relevant = False
            spatial_join_required = False
            coord_source = "line_midpoint"
            exemption = "topology_only_overhead_line"
        else:
            asset_type = "line"
            is_flood_relevant = False
            spatial_join_required = False
            coord_source = "line_midpoint"
            exemption = "topology_only_line"
        assets.append(
            base_asset(
                asset_type=asset_type,
                source_table="lines",
                source_name=row["line_name"],
                feeder_id=row["feeder_id"],
                bus=row["from_bus"],
                phases=row["phases"],
                coordinate=coordinate_fields(
                    lon,
                    lat,
                    source=coord_source,
                    is_flood_relevant=is_flood_relevant,
                    spatial_join_required=spatial_join_required,
                    exemption_reason=exemption,
                ),
                rated_kv=None,
                rated_kva=None,
                source_row=row,
            )
        )
    return assets


def build_assets(registry_dir: Path) -> list[dict[str, Any]]:
    rows = []
    rows.extend(build_transformer_assets(read_csv(registry_dir / "transformers.csv")))
    rows.extend(build_source_assets(read_csv(registry_dir / "sources.csv")))
    rows.extend(build_load_bus_assets(read_csv(registry_dir / "load_buses.csv")))
    rows.extend(build_line_assets(read_csv(registry_dir / "lines.csv")))
    rows.sort(key=lambda row: row["asset_id"])
    return rows


def build_control_units(
    registry_dir: Path, assets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    feeders = read_csv(registry_dir / "feeders.csv")
    assets_by_feeder: dict[str, list[str]] = defaultdict(list)
    sources_by_feeder: dict[str, list[str]] = defaultdict(list)
    for asset in assets:
        feeder_id = asset["feeder_id"]
        if feeder_id:
            assets_by_feeder[feeder_id].append(asset["asset_id"])
            if asset["asset_type"] == "source":
                sources_by_feeder[feeder_id].append(asset["asset_id"])

    control_units = []
    for feeder in feeders:
        feeder_id = feeder["feeder_id"]
        control_units.append(
            {
                "sandbox_id": SANDBOX_ID,
                "control_unit_id": stable_control_unit_id(feeder_id),
                "control_unit_type": "feeder",
                "control_unit_stage": "stage_a",
                "source_feeder_id": feeder_id,
                "parent_control_unit_id": None,
                "member_asset_ids": sorted(assets_by_feeder.get(feeder_id, [])),
                "source_ids": sorted(sources_by_feeder.get(feeder_id, [])),
                "boundary_bus_ids": [],
                "served_load_kw": parse_float(feeder.get("load_kw")) or 0.0,
                "critical_load_weight": 0.0,
                "der_capacity_kw": 0.0,
                "der_capacity_kwh": 0.0,
                "candidate_status": "active",
                "candidate_basis": "asset_registry_feeder",
                "source_provenance": json.dumps(
                    {"source_table": "feeders", "feeder_id": feeder_id},
                    sort_keys=True,
                ),
                "schema_version": SCHEMA_VERSION,
            }
        )
    control_units.sort(key=lambda row: row["control_unit_id"])
    return control_units


def assets_schema() -> Any:
    pa, _ = require_pyarrow()
    return pa.schema(
        [
            ("sandbox_id", pa.string()),
            ("asset_id", pa.string()),
            ("asset_type", pa.string()),
            ("source_asset_table", pa.string()),
            ("source_asset_name", pa.string()),
            ("source_uuid", pa.string()),
            ("feeder_id", pa.string()),
            ("bus", pa.string()),
            ("phases", pa.string()),
            ("lon", pa.float64()),
            ("lat", pa.float64()),
            ("coordinate_status", pa.string()),
            ("coordinate_source", pa.string()),
            ("is_flood_relevant", pa.bool_()),
            ("spatial_join_required", pa.bool_()),
            ("coordinate_exemption_reason", pa.string()),
            ("rated_kv", pa.float64()),
            ("rated_kva", pa.float64()),
            ("source_provenance", pa.string()),
            ("schema_version", pa.string()),
        ]
    )


def control_units_schema() -> Any:
    pa, _ = require_pyarrow()
    return pa.schema(
        [
            ("sandbox_id", pa.string()),
            ("control_unit_id", pa.string()),
            ("control_unit_type", pa.string()),
            ("control_unit_stage", pa.string()),
            ("source_feeder_id", pa.string()),
            ("parent_control_unit_id", pa.string()),
            ("member_asset_ids", pa.list_(pa.string())),
            ("source_ids", pa.list_(pa.string())),
            ("boundary_bus_ids", pa.list_(pa.string())),
            ("served_load_kw", pa.float64()),
            ("critical_load_weight", pa.float64()),
            ("der_capacity_kw", pa.float64()),
            ("der_capacity_kwh", pa.float64()),
            ("candidate_status", pa.string()),
            ("candidate_basis", pa.string()),
            ("source_provenance", pa.string()),
            ("schema_version", pa.string()),
        ]
    )


def validate_assets(assets: list[dict[str, Any]], report: dict[str, Any]) -> None:
    ids = [row["asset_id"] for row in assets]
    if len(ids) != len(set(ids)):
        validation_error(report, "asset_id values are not unique")
    for row in assets:
        asset_id = row["asset_id"]
        if not asset_id.startswith(f"{SANDBOX_ID}:asset:"):
            validation_error(report, f"{asset_id}: invalid asset namespace")
        if row["is_flood_relevant"] and row["coordinate_status"] != "valid":
            validation_error(report, f"{asset_id}: flood-relevant asset lacks valid coordinates")
        if row["spatial_join_required"] and row["coordinate_status"] != "valid":
            validation_error(report, f"{asset_id}: spatial-join asset lacks valid coordinates")
        if row["coordinate_status"] == "valid" and not finite_lon_lat(row["lon"], row["lat"]):
            validation_error(report, f"{asset_id}: coordinate_status valid but lon/lat invalid")
        if row["coordinate_status"] == "missing_exempt" and not row["coordinate_exemption_reason"]:
            validation_error(report, f"{asset_id}: missing coordinate exemption reason")
        if row["asset_type"] == "overhead_line":
            if row["is_flood_relevant"] or row["spatial_join_required"]:
                validation_error(report, f"{asset_id}: overhead_line must be topology-only")
        if row["asset_type"] == "underground_line_proxy" and row["coordinate_source"] not in {
            "line_midpoint",
            "from_bus",
            "to_bus",
            "splice_vault_inventory",
        }:
            validation_error(report, f"{asset_id}: underground proxy coordinate source is invalid")
    report["checks"]["asset_ids_unique"] = len(ids) == len(set(ids))
    report["checks"]["asset_count"] = len(assets)
    report["checks"]["asset_counts_by_type"] = count_by(assets, "asset_type")


def validate_control_units(
    registry_dir: Path,
    assets: list[dict[str, Any]],
    control_units: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    feeder_ids = {row["feeder_id"] for row in read_csv(registry_dir / "feeders.csv")}
    asset_ids = {row["asset_id"] for row in assets}
    unit_ids = [row["control_unit_id"] for row in control_units]
    if len(unit_ids) != len(set(unit_ids)):
        validation_error(report, "control_unit_id values are not unique")
    unit_feeders = {row["source_feeder_id"] for row in control_units}
    missing = sorted(feeder_ids - unit_feeders)
    extra = sorted(unit_feeders - feeder_ids)
    if missing:
        validation_error(report, f"missing Feeder Control Units: {missing}")
    if extra:
        validation_error(report, f"unexpected Feeder Control Units: {extra}")
    for unit in control_units:
        unit_id = unit["control_unit_id"]
        if not unit_id.startswith(f"{SANDBOX_ID}:control_unit:"):
            validation_error(report, f"{unit_id}: invalid control unit namespace")
        if unit["control_unit_type"] != "feeder" or unit["control_unit_stage"] != "stage_a":
            validation_error(report, f"{unit_id}: Stage A1 supports feeder control units only")
        for asset_id in unit["member_asset_ids"]:
            if asset_id not in asset_ids:
                validation_error(report, f"{unit_id}: unknown member asset {asset_id}")
    report["checks"]["control_unit_ids_unique"] = len(unit_ids) == len(set(unit_ids))
    report["checks"]["control_unit_count"] = len(control_units)
    report["checks"]["feeder_count"] = len(feeder_ids)


def _build_static_grid_manifest(
    registry_dir: Path,
    output_dir: Path,
    outputs: dict[str, Path],
    debug_outputs: dict[str, Path],
) -> dict[str, Any]:
    registry_inputs = {
        path.name: {"path": str(path), "sha256": sha256(path)}
        for path in sorted(registry_dir.glob("*.csv"))
    }
    registry_summary = registry_dir / "summary.json"
    if registry_summary.exists():
        registry_inputs[registry_summary.name] = {
            "path": str(registry_summary),
            "sha256": sha256(registry_summary),
        }
    return {
        "run_id": f"{SANDBOX_ID}:run:{PROTOCOL_VERSION}:stage_a1:{short_hash(registry_inputs)}",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sandbox_id": SANDBOX_ID,
        "stage": "stage_a1",
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_info(),
        "inputs": registry_inputs,
        "outputs": {
            name: {"path": str(path), "sha256": maybe_sha256(path)}
            for name, path in outputs.items()
        },
        "debug_outputs": {
            name: {"path": str(path), "sha256": maybe_sha256(path)}
            for name, path in debug_outputs.items()
        }
    }


def export_stage_a1(registry_dir: Path, output_dir: Path, *, debug_csv: bool) -> dict[str, Any]:
    assets = build_assets(registry_dir)
    control_units = build_control_units(registry_dir, assets)

    assets_path = output_dir / "assets.parquet"
    control_units_path = output_dir / "control_units.parquet"
    write_parquet(assets_path, assets, assets_schema())
    write_parquet(control_units_path, control_units, control_units_schema())

    debug_outputs: dict[str, Path] = {}
    if debug_csv:
        assets_debug = output_dir / "assets.debug.csv"
        control_units_debug = output_dir / "control_units.debug.csv"
        write_debug_csv(assets_debug, assets, [field.name for field in assets_schema()])
        write_debug_csv(control_units_debug, control_units, [field.name for field in control_units_schema()])
        debug_outputs = {
            "assets.debug.csv": assets_debug,
            "control_units.debug.csv": control_units_debug,
        }

    report: dict[str, Any] = {
        "stage": "stage_a1",
        "schema_version": SCHEMA_VERSION,
        "passed": False,
        "errors": [],
        "checks": {},
    }
    validate_assets(assets, report)
    validate_control_units(registry_dir, assets, control_units, report)
    report["passed"] = not report["errors"]

    outputs = {
        "assets.parquet": assets_path,
        "control_units.parquet": control_units_path,
    }
    manifest = _build_static_grid_manifest(registry_dir, output_dir, outputs, debug_outputs)
    manifest_path = output_dir / "run_manifest.json"
    validation_path = output_dir / "validation_report.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validation_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


# Event-conditioned Grid Dataset export

"""Export the Stage A2 event-conditioned simulation spine.

Reads validated Stage A1 artifacts and writes:

    locations/marshfield/data/power_grid/augmented/asset_states.parquet
    locations/marshfield/data/power_grid/augmented/telemetry_observations.parquet
    locations/marshfield/data/power_grid/augmented/run_manifest_stage_a2.json
    locations/marshfield/data/power_grid/augmented/validation_report_stage_a2.json
"""


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

from power.artifacts import count_by
from power.artifacts import finite_lon_lat
from power.artifacts import git_info
from power.artifacts import maybe_sha256
from power.artifacts import read_parquet
from power.artifacts import require_pyarrow
from power.artifacts import sha256
from power.artifacts import short_hash
from power.artifacts import validation_error
from power.artifacts import write_debug_csv
from power.artifacts import write_parquet
from power.artifacts import PROTOCOL_VERSION
from power.artifacts import SANDBOX_ID
from power.impact import failure_probability

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


def _build_event_conditioned_grid_manifest(
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
    manifest = _build_event_conditioned_grid_manifest(
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


# Control sandbox registry

"""Control-sandbox filtering for Marshfield Asset Registry exports."""


import csv
import json
import math
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from power.artifacts import parse_float as _as_float, parse_int
from power.baseline_network.build_asset_registry import build_feeders

CONTROL_SANDBOX_FILTER_SCHEMA_VERSION = "marshfield_control_sandbox_filter.v0.1"

REGISTRY_CSV_NAMES = (
    "buses.csv",
    "lines.csv",
    "transformers.csv",
    "sources.csv",
    "loads.csv",
    "load_buses.csv",
    "feeders.csv",
)

FEEDER_FIELDS = [
    "feeder_id",
    "bus_count",
    "line_count",
    "transformer_count",
    "source_count",
    "load_count",
    "load_kw",
    "load_kvar",
]


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames or [])


def _write_csv(path: Path, rows: Iterable[dict[str, str]], fields: list[str]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def _as_int(value: object, default: int = 0) -> int:
    return parse_int(value, default)


def _bus_components(
    buses: list[dict[str, str]],
    lines: list[dict[str, str]],
    transformers: list[dict[str, str]],
) -> tuple[dict[str, str], Counter[str]]:
    parent = {row["bus"]: row["bus"] for row in buses if row.get("bus")}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        if not left or not right:
            return
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in lines:
        if (row.get("line_class") or "line") != "line":
            continue
        union(row.get("from_bus", ""), row.get("to_bus", ""))

    for row in transformers:
        windings = [
            bus.strip()
            for bus in (row.get("winding_buses") or "").split(",")
            if bus.strip()
        ]
        if len(windings) < 2:
            continue
        hub = windings[0]
        for bus in windings[1:]:
            union(hub, bus)

    bus_to_component = {bus: find(bus) for bus in parent}
    return bus_to_component, Counter(bus_to_component.values())


def _component_groups_from_tie_candidates(
    buses: list[dict[str, str]],
    bus_to_component: dict[str, str],
    *,
    max_distance_m: float,
    min_line_degree: int,
) -> list[set[str]]:
    components = set(bus_to_component.values())
    graph = {component: set() for component in components}
    candidate_rows = [
        row
        for row in buses
        if row.get("bus") in bus_to_component
        and row.get("feeder_id")
        and _as_int(row.get("line_degree"), default=0) >= min_line_degree
        and _as_float(row.get("lon")) is not None
        and _as_float(row.get("lat")) is not None
    ]
    if not candidate_rows:
        return [{component} for component in sorted(components)]

    reference_lat = sum(float(row["lat"]) for row in candidate_rows) / len(candidate_rows)
    lon_scale = 111_320.0 * math.cos(math.radians(reference_lat))
    lat_scale = 110_540.0
    cell_size = max(max_distance_m, 1.0)
    candidates = []
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(candidate_rows):
        projected = {
            "row": row,
            "x": float(row["lon"]) * lon_scale,
            "y": float(row["lat"]) * lat_scale,
        }
        bucket = (
            math.floor(projected["x"] / cell_size),
            math.floor(projected["y"] / cell_size),
        )
        candidates.append(projected)
        buckets[bucket].append(index)

    for left_index, projected_left in enumerate(candidates):
        left = projected_left["row"]
        left_component = bus_to_component[left["bus"]]
        left_bucket = (
            math.floor(projected_left["x"] / cell_size),
            math.floor(projected_left["y"] / cell_size),
        )
        neighbor_indexes = [
            index
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for index in buckets.get((left_bucket[0] + dx, left_bucket[1] + dy), [])
            if index > left_index
        ]
        for right_index in neighbor_indexes:
            projected_right = candidates[right_index]
            right = projected_right["row"]
            if left.get("feeder_id") == right.get("feeder_id"):
                continue
            right_component = bus_to_component[right["bus"]]
            if left_component == right_component:
                continue
            distance = math.hypot(
                projected_right["x"] - projected_left["x"],
                projected_right["y"] - projected_left["y"],
            )
            if distance > max_distance_m:
                continue
            graph[left_component].add(right_component)
            graph[right_component].add(left_component)

    groups: list[set[str]] = []
    seen: set[str] = set()
    for component in sorted(components):
        if component in seen:
            continue
        stack = [component]
        seen.add(component)
        group: set[str] = set()
        while stack:
            current = stack.pop()
            group.add(current)
            for neighbor in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        groups.append(group)
    return groups


def _transformer_in_kept_buses(row: dict[str, str], kept_buses: set[str]) -> bool:
    windings = [
        bus.strip()
        for bus in (row.get("winding_buses") or "").split(",")
        if bus.strip()
    ]
    return bool(windings) and all(bus in kept_buses for bus in windings)


def build_control_sandbox_registry(
    raw_registry_dir: Path | str,
    output_dir: Path | str,
    *,
    max_tie_distance_m: float = 100.0,
    min_tie_bus_line_degree: int = 2,
) -> dict[str, int]:
    """Write the canonical Marshfield control-sandbox registry.

    The raw SHIFT/DiTTo registry may contain source-backed feeder components
    outside the networked-microgrid reconfiguration graph. This filter retains
    the largest baseline-component group connected by eligible cross-feeder tie
    proximity and writes that group as the canonical Asset Registry consumed by
    Stage A and Stage B exports.
    """

    raw_registry_dir = Path(raw_registry_dir)
    output_dir = Path(output_dir)
    tables: dict[str, list[dict[str, str]]] = {}
    fields: dict[str, list[str]] = {}
    for name in REGISTRY_CSV_NAMES:
        rows, header = _read_csv(raw_registry_dir / name)
        tables[name] = rows
        fields[name] = header

    bus_to_component, component_sizes = _bus_components(
        tables["buses.csv"], tables["lines.csv"], tables["transformers.csv"]
    )
    groups = _component_groups_from_tie_candidates(
        tables["buses.csv"],
        bus_to_component,
        max_distance_m=max_tie_distance_m,
        min_line_degree=min_tie_bus_line_degree,
    )
    if groups:
        kept_components = max(
            groups,
            key=lambda group: (
                sum(component_sizes[component] for component in group),
                len(group),
                sorted(group),
            ),
        )
    else:
        kept_components = set(component_sizes.keys())

    kept_buses = {
        bus
        for bus, component in bus_to_component.items()
        if component in kept_components
    }
    filtered = {
        "buses.csv": [
            row for row in tables["buses.csv"] if row.get("bus") in kept_buses
        ],
        "lines.csv": [
            row
            for row in tables["lines.csv"]
            if row.get("from_bus") in kept_buses and row.get("to_bus") in kept_buses
        ],
        "transformers.csv": [
            row
            for row in tables["transformers.csv"]
            if _transformer_in_kept_buses(row, kept_buses)
        ],
        "sources.csv": [
            row for row in tables["sources.csv"] if row.get("bus") in kept_buses
        ],
        "loads.csv": [
            row for row in tables["loads.csv"] if row.get("bus") in kept_buses
        ],
        "load_buses.csv": [
            row for row in tables["load_buses.csv"] if row.get("bus") in kept_buses
        ],
    }
    filtered["feeders.csv"] = build_feeders(
        filtered["buses.csv"],
        filtered["lines.csv"],
        filtered["transformers.csv"],
        filtered["sources.csv"],
        filtered["loads.csv"],
    )

    outputs = {
        name: _write_csv(
            output_dir / name,
            filtered[name],
            FEEDER_FIELDS if name == "feeders.csv" else fields[name],
        )
        for name in REGISTRY_CSV_NAMES
    }

    raw_summary_path = raw_registry_dir / "summary.json"
    raw_summary = (
        json.loads(raw_summary_path.read_text(encoding="utf-8"))
        if raw_summary_path.exists()
        else {}
    )
    excluded_components = set(component_sizes) - set(kept_components)
    excluded_buses = {
        bus
        for bus, component in bus_to_component.items()
        if component in excluded_components
    }
    summary = {
        "method": "canonical Marshfield control-sandbox filter over raw Asset Registry",
        "schema_version": CONTROL_SANDBOX_FILTER_SCHEMA_VERSION,
        "raw_asset_registry_summary": raw_summary,
        "outputs": outputs,
        "control_sandbox_filter": {
            "schema_version": CONTROL_SANDBOX_FILTER_SCHEMA_VERSION,
            "input_registry_dir": str(raw_registry_dir),
            "max_tie_distance_m": float(max_tie_distance_m),
            "min_tie_bus_line_degree": int(min_tie_bus_line_degree),
            "raw_baseline_components": len(component_sizes),
            "retained_baseline_components": len(kept_components),
            "excluded_baseline_components": len(excluded_components),
            "retained_buses": len(kept_buses),
            "excluded_buses": len(excluded_buses),
            "excluded_loads": len(tables["loads.csv"]) - len(filtered["loads.csv"]),
            "excluded_sources": len(tables["sources.csv"]) - len(filtered["sources.csv"]),
            "exclusion_reason": "outside_largest_tie_eligible_component_group",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return outputs


control_registry = build_control_sandbox_registry
export_base = export_stage_a1
