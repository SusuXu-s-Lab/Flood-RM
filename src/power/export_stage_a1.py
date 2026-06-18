"""Export the Stage A1 static simulation spine.

Reads the deterministic Asset Registry CSVs and writes SMART-DS-compatible
sandbox artifacts:

    locations/marshfield/data/power_grid/augmented/assets.parquet
    locations/marshfield/data/power_grid/augmented/control_units.parquet
    locations/marshfield/data/power_grid/augmented/run_manifest.json
    locations/marshfield/data/power_grid/augmented/validation_report.json

Parquet is canonical. Optional ``*.debug.csv`` files are derived from the same
rows for inspection only.

Run:
    python -m power.export_stage_a1
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


from power.artifact_io import count_by
from power.artifact_io import maybe_sha256
from power.artifact_io import read_csv
from power.artifact_io import require_pyarrow
from power.artifact_io import sha256
from power.artifact_io import short_hash
from power.artifact_io import write_debug_csv
from power.artifact_io import write_parquet
from power.paths import REPO_ROOT
from power.paths import POWER_GRID

DEFAULT_REGISTRY_DIR = POWER_GRID / "asset_registry"
DEFAULT_OUTPUT_DIR = POWER_GRID / "augmented"
SANDBOX_ID = "marshfield"
SCHEMA_VERSION = "stage_a1.v0.1"
PROTOCOL_VERSION = "v0.1"


def slug(value: str) -> str:
    lowered = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return normalized or "unknown"


def stable_asset_id(source_table: str, source_name: str) -> str:
    return f"{SANDBOX_ID}:asset:{slug(source_table)}:{slug(source_name)}"


def stable_control_unit_id(feeder_id: str) -> str:
    return f"{SANDBOX_ID}:control_unit:feeder:{slug(feeder_id)}"


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def finite_lon_lat(lon: float | None, lat: float | None) -> bool:
    return (
        lon is not None
        and lat is not None
        and -180.0 <= lon <= 180.0
        and -90.0 <= lat <= 90.0
    )


def midpoint(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def bool_from_registry(value: str | None) -> bool:
    return (value or "").strip().lower() == "true"


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


def validation_error(report: dict[str, Any], message: str) -> None:
    report["errors"].append(message)


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


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
        except Exception:
            return ""

    status = run(["git", "status", "--short"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def build_manifest(
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
    manifest = build_manifest(registry_dir, output_dir, outputs, debug_outputs)
    manifest_path = output_dir / "run_manifest.json"
    validation_path = output_dir / "validation_report.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validation_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, default=DEFAULT_REGISTRY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-debug-csv",
        action="store_true",
        help="Skip optional .debug.csv exports.",
    )
    args = parser.parse_args()

    report = export_stage_a1(
        args.registry_dir,
        args.output_dir,
        debug_csv=not args.no_debug_csv,
    )
    status = "passed" if report["passed"] else "failed"
    print(f"Stage A1 export {status}: {args.output_dir}")
    for key, value in report["checks"].items():
        print(f"  {key}: {value}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
