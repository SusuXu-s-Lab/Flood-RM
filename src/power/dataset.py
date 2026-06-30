"""SMART-DS-like static and event-conditioned Grid Dataset artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
import random
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .core import file_sha256, manifest, parse_float, parse_int, slug, stable_seed, write_json, write_table
from .impact import FragilityModel, failure_probability
from .registry import build_feeders

protocol_version = "v0.1"
location_id = os.environ.get("NLR_DISTRIBUTION_LOCATION_ID", "case")
power_grid = Path("data/power_grid")
default_registry_dir = power_grid / "asset_registry"
default_output_dir = power_grid / "augmented"
schema_version = "stage_a1.v0.1"
asset_states_schema_version = "stage_a2_asset_states.v0.1"
telemetry_schema_version = "stage_a2_telemetry_observations.v0.1"
default_event_id = "synthetic_coastal_001"
default_root_seed = 20260509
default_mc_draws = 50
default_synthetic_peak_depth_m = 1.25
default_event_timestamp = "2026-01-01T00:00:00+00:00"
default_max_sample_distance_m = 150.0
out_of_mesh_depth_m = 0.0


def _count_by(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def _finite_lon_lat(lon: Any, lat: Any) -> bool:
    lon, lat = parse_float(lon), parse_float(lat)
    return lon is not None and lat is not None and -180 <= lon <= 180 and -90 <= lat <= 90


def _asset_id(source_table: str, source_name: str) -> str:
    return f"{location_id}:asset:{slug(source_table)}:{slug(source_name)}"


def _control_unit_id(feeder_id: str) -> str:
    return f"{location_id}:control_unit:feeder:{slug(feeder_id)}"


def _coordinate(lon: Any, lat: Any, *, source: str, flood: bool, spatial: bool, exemption: str = "") -> dict[str, Any]:
    lon_f, lat_f = parse_float(lon), parse_float(lat)
    valid = _finite_lon_lat(lon_f, lat_f)
    if valid:
        status, reason = "valid", ""
    elif flood or spatial:
        status, reason = "invalid", ""
    else:
        status, reason = "missing_exempt", exemption or "non_spatial_metadata"
    return {"lon": lon_f, "lat": lat_f, "coordinate_status": status, "coordinate_source": source if status != "missing_exempt" else "non_spatial_metadata", "is_flood_relevant": flood, "spatial_join_required": spatial, "coordinate_exemption_reason": reason}


def _source_provenance(table: str, row: Mapping[str, Any]) -> str:
    return json.dumps({"source_table": table, "source_file": row.get("source_file", ""), "source_line": row.get("source_line", "")}, sort_keys=True)


def _base_asset(*, asset_type: str, source_table: str, source_name: str, feeder_id: str, bus: str, phases: str, coordinate: dict[str, Any], rated_kv: float | None = None, rated_kva: float | None = None, source_row: Mapping[str, Any]) -> dict[str, Any]:
    return {"sandbox_id": location_id, "asset_id": _asset_id(source_table, source_name), "asset_type": asset_type, "source_asset_table": source_table, "source_asset_name": source_name, "source_uuid": "", "feeder_id": feeder_id, "bus": bus, "phases": phases, **coordinate, "rated_kv": rated_kv, "rated_kva": rated_kva, "source_provenance": _source_provenance(source_table, source_row), "schema_version": schema_version}


def build_assets(registry_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in pd.read_csv(registry_dir / "transformers.csv", keep_default_na=False).to_dict("records"):
        rows.append(_base_asset(asset_type="transformer", source_table="transformers", source_name=row["transformer_name"], feeder_id=row["feeder_id"], bus=row["location_bus"], phases=row["phases"], coordinate=_coordinate(row.get("location_lon"), row.get("location_lat"), source="buscoords.csv", flood=True, spatial=True), rated_kv=parse_float(row.get("max_kv")), rated_kva=parse_float(row.get("max_kva")), source_row=row))
    for row in pd.read_csv(registry_dir / "sources.csv", keep_default_na=False).to_dict("records"):
        rows.append(_base_asset(asset_type="source", source_table="sources", source_name=row["source_name"], feeder_id=row["feeder_id"], bus=row["bus"], phases=row["phases"], coordinate=_coordinate(row.get("lon"), row.get("lat"), source="buscoords.csv", flood=True, spatial=True), rated_kv=parse_float(row.get("basekv")), source_row=row))
    for row in pd.read_csv(registry_dir / "load_buses.csv", keep_default_na=False).to_dict("records"):
        rows.append(_base_asset(asset_type="load_bus", source_table="load_buses", source_name=row["bus"], feeder_id=row["feeder_id"], bus=row["bus"], phases="", coordinate=_coordinate(row.get("lon"), row.get("lat"), source="buscoords.csv", flood=True, spatial=True), source_row=row))
    for row in pd.read_csv(registry_dir / "lines.csv", keep_default_na=False).to_dict("records"):
        lon = _mid(parse_float(row.get("from_lon")), parse_float(row.get("to_lon")))
        lat = _mid(parse_float(row.get("from_lat")), parse_float(row.get("to_lat")))
        line_class = row.get("line_class", "")
        if line_class == "underground":
            asset_type, flood, spatial, exemption = "underground_line_proxy", True, True, ""
        elif line_class == "fuse":
            asset_type, flood, spatial, exemption = "fuse_proxy", True, True, ""
        elif line_class == "overhead":
            asset_type, flood, spatial, exemption = "overhead_line", False, False, "topology_only_overhead_line"
        else:
            asset_type, flood, spatial, exemption = "line", False, False, "topology_only_line"
        rows.append(_base_asset(asset_type=asset_type, source_table="lines", source_name=row["line_name"], feeder_id=row["feeder_id"], bus=row["from_bus"], phases=row["phases"], coordinate=_coordinate(lon, lat, source="line_midpoint", flood=flood, spatial=spatial, exemption=exemption), source_row=row))
    return sorted(rows, key=lambda r: r["asset_id"])


def _mid(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else (a + b) / 2


def build_control_units(registry_dir: Path, assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feeders = pd.read_csv(registry_dir / "feeders.csv", keep_default_na=False).to_dict("records")
    by_feeder: dict[str, list[str]] = defaultdict(list); sources: dict[str, list[str]] = defaultdict(list)
    for asset in assets:
        feeder = asset.get("feeder_id")
        if feeder:
            by_feeder[feeder].append(asset["asset_id"])
            if asset["asset_type"] == "source":
                sources[feeder].append(asset["asset_id"])
    rows = []
    for feeder in feeders:
        fid = feeder["feeder_id"]
        rows.append({"sandbox_id": location_id, "control_unit_id": _control_unit_id(fid), "control_unit_type": "feeder", "control_unit_stage": "stage_a", "source_feeder_id": fid, "parent_control_unit_id": None, "member_asset_ids": sorted(by_feeder.get(fid, [])), "source_ids": sorted(sources.get(fid, [])), "boundary_bus_ids": [], "served_load_kw": parse_float(feeder.get("load_kw"), 0.0) or 0.0, "critical_load_weight": 0.0, "der_capacity_kw": 0.0, "der_capacity_kwh": 0.0, "candidate_status": "active", "candidate_basis": "asset_registry_feeder", "source_provenance": json.dumps({"source_table": "feeders", "feeder_id": fid}, sort_keys=True), "schema_version": schema_version})
    return sorted(rows, key=lambda r: r["control_unit_id"])


def assets_schema() -> Any:
    import pyarrow as pa
    return pa.schema([(c, t) for c, t in [
        ("sandbox_id", pa.string()), ("asset_id", pa.string()), ("asset_type", pa.string()), ("source_asset_table", pa.string()), ("source_asset_name", pa.string()), ("source_uuid", pa.string()), ("feeder_id", pa.string()), ("bus", pa.string()), ("phases", pa.string()), ("lon", pa.float64()), ("lat", pa.float64()), ("coordinate_status", pa.string()), ("coordinate_source", pa.string()), ("is_flood_relevant", pa.bool_()), ("spatial_join_required", pa.bool_()), ("coordinate_exemption_reason", pa.string()), ("rated_kv", pa.float64()), ("rated_kva", pa.float64()), ("source_provenance", pa.string()), ("schema_version", pa.string())]])


def control_units_schema() -> Any:
    import pyarrow as pa
    return pa.schema([("sandbox_id", pa.string()), ("control_unit_id", pa.string()), ("control_unit_type", pa.string()), ("control_unit_stage", pa.string()), ("source_feeder_id", pa.string()), ("parent_control_unit_id", pa.string()), ("member_asset_ids", pa.list_(pa.string())), ("source_ids", pa.list_(pa.string())), ("boundary_bus_ids", pa.list_(pa.string())), ("served_load_kw", pa.float64()), ("critical_load_weight", pa.float64()), ("der_capacity_kw", pa.float64()), ("der_capacity_kwh", pa.float64()), ("candidate_status", pa.string()), ("candidate_basis", pa.string()), ("source_provenance", pa.string()), ("schema_version", pa.string())])


def validate_assets(assets: list[dict[str, Any]], report: dict[str, Any]) -> None:
    ids = [r["asset_id"] for r in assets]
    if len(ids) != len(set(ids)):
        report["errors"].append("asset_id values are not unique")
    for row in assets:
        if row["is_flood_relevant"] and row["coordinate_status"] != "valid":
            report["errors"].append(f"{row['asset_id']}: flood-relevant asset lacks valid coordinates")
    report["checks"].update({"asset_ids_unique": len(ids) == len(set(ids)), "asset_count": len(assets), "asset_counts_by_type": _count_by(assets, "asset_type")})


def validate_control_units(registry_dir: Path, assets: list[dict[str, Any]], units: list[dict[str, Any]], report: dict[str, Any]) -> None:
    feeder_ids = set(pd.read_csv(registry_dir / "feeders.csv", keep_default_na=False)["feeder_id"].astype(str))
    unit_feeders = {u["source_feeder_id"] for u in units}
    if feeder_ids != unit_feeders:
        report["errors"].append(f"feeder/control-unit mismatch: missing={sorted(feeder_ids - unit_feeders)}, extra={sorted(unit_feeders - feeder_ids)}")
    asset_ids = {a["asset_id"] for a in assets}
    for unit in units:
        for aid in unit["member_asset_ids"]:
            if aid not in asset_ids:
                report["errors"].append(f"{unit['control_unit_id']}: unknown member asset {aid}")
    report["checks"].update({"control_unit_count": len(units), "feeder_count": len(feeder_ids)})


def export_base(registry_dir: Path, output_dir: Path, *, debug_csv: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = build_assets(registry_dir)
    units = build_control_units(registry_dir, assets)
    write_table(output_dir / "assets.parquet", assets, schema=assets_schema())
    write_table(output_dir / "control_units.parquet", units, schema=control_units_schema())
    if debug_csv:
        pd.DataFrame(assets).to_csv(output_dir / "assets.debug.csv", index=False)
        pd.DataFrame(units).to_csv(output_dir / "control_units.debug.csv", index=False)
    report = {"stage": "stage_a1", "schema_version": schema_version, "passed": False, "errors": [], "checks": {}}
    validate_assets(assets, report); validate_control_units(registry_dir, assets, units, report); report["passed"] = not report["errors"]
    write_json(output_dir / "validation_report.json", report)
    write_json(output_dir / "run_manifest.json", manifest(stage="stage_a1", location_id=location_id, inputs={p.name: p for p in registry_dir.glob("*.csv")}, outputs={"assets.parquet": output_dir / "assets.parquet", "control_units.parquet": output_dir / "control_units.parquet"}))
    return report


def parse_utc_timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def flood_relevant_assets(assets: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted([dict(r) for r in assets if r.get("is_flood_relevant") and r.get("coordinate_status") == "valid" and _finite_lon_lat(r.get("lon"), r.get("lat"))], key=lambda r: r["asset_id"])


def synthetic_depth(asset: Mapping[str, Any], *, lon_min: float, lon_max: float, peak_depth_m: float) -> float:
    span = max(lon_max - lon_min, 1e-9)
    eastness = min(max((float(asset["lon"]) - lon_min) / span, 0.0), 1.0)
    jitter = 0.85 + 0.30 * (stable_seed("synthetic_depth", asset["asset_id"]) / float((1 << 63) - 1))
    return max(0.0, peak_depth_m * (0.15 + 0.85 * eastness) * jitter)


def build_synthetic_event_samples(assets: list[dict[str, Any]], *, timestamp: datetime, peak_depth_m: float) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    lons = [float(a["lon"]) for a in assets]
    samples = {a["asset_id"]: [{"timestamp": timestamp, "sampled_depth_m": synthetic_depth(a, lon_min=min(lons), lon_max=max(lons), peak_depth_m=peak_depth_m)}] for a in assets}
    return samples, {"event_source_kind": "synthetic_coordinate_profile", "timestamp": timestamp.isoformat(), "peak_depth_m": peak_depth_m}


def build_csv_event_samples(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"asset_id", "timestamp", "sampled_depth_m"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            samples[row["asset_id"]].append({"timestamp": parse_utc_timestamp(row["timestamp"]), "sampled_depth_m": float(row["sampled_depth_m"])})
    for rows in samples.values():
        rows.sort(key=lambda r: r["timestamp"])
    return dict(samples), {"event_source_kind": "sampled_asset_depth_csv", "path": str(path), "sha256": file_sha256(path)}


def require_geo_stack():
    try:
        import numpy as np, xarray as xr
        from pyproj import Transformer
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("SFINCS sampling requires numpy, xarray, pyproj, and scipy") from exc
    return np, xr, Transformer, cKDTree


def build_sfincs_event_samples(event_dir: Path, assets: list[dict[str, Any]], *, max_sample_distance_m: float = default_max_sample_distance_m, xarray_engine: str | None = None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    np, xr, Transformer, cKDTree = require_geo_stack()
    path = event_dir / "sfincs_map.nc"
    if not path.exists():
        raise FileNotFoundError(path)
    with xr.open_dataset(path, **({"engine": xarray_engine} if xarray_engine else {})) as ds:
        active = np.asarray(ds["msk"].values, dtype=float) > 0
        if "hmax" in ds:
            peak = np.asarray(ds["hmax"].values, dtype=float)
        elif "zsmax" in ds and "zb" in ds:
            zs = np.asarray(ds["zsmax"].values, dtype=float)
            peak = np.maximum((np.nanmax(zs, axis=0) if zs.ndim == 3 else zs) - np.asarray(ds["zb"].values, dtype=float), 0.0)
        else:
            raise RuntimeError("sfincs_map.nc must contain hmax or zsmax+zb")
        x, y = np.asarray(ds["x"].values, dtype=float), np.asarray(ds["y"].values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & active & np.isfinite(peak)
    tree = cKDTree(np.column_stack([x[valid], y[valid]]))
    to_model = Transformer.from_crs("EPSG:4326", "EPSG:26919", always_xy=True)
    xy = np.asarray([to_model.transform(a["lon"], a["lat"]) for a in assets], dtype=float)
    dist, nearest = tree.query(xy, k=1)
    values = np.asarray(peak[valid], dtype=float)[nearest]
    timestamp = parse_utc_timestamp(default_event_timestamp)
    samples = {a["asset_id"]: [{"timestamp": timestamp, "sampled_depth_m": float(v) if float(d) <= max_sample_distance_m else out_of_mesh_depth_m}] for a, v, d in zip(assets, values, dist, strict=True)}
    return samples, {"event_source_kind": "sfincs_map_peak_depth", "event_dir": str(event_dir), "sfincs_map_path": str(path), "sfincs_map_sha256": file_sha256(path), "max_sample_distance_m": max_sample_distance_m, "out_of_mesh_depth_m": out_of_mesh_depth_m}


def build_asset_states(
    assets: list[dict[str, Any]],
    samples: dict[str, list[dict[str, Any]]],
    event_metadata: Mapping[str, Any],
    *,
    event_id: str,
    root_seed: int,
    mc_draws: int,
    fragility_model: FragilityModel | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[int, datetime, str], int]]:
    """Sample Monte Carlo asset states with ERAD-native fragility probabilities."""

    rows: list[dict[str, Any]] = []; binary: dict[tuple[int, datetime, str], int] = {}
    for draw in range(mc_draws):
        for asset in assets:
            for sample in samples.get(asset["asset_id"], []):
                ts, depth = sample["timestamp"], float(sample["sampled_depth_m"])
                p = failure_probability(asset["asset_type"], depth, model=fragility_model)
                seed = stable_seed("asset_state", root_seed, event_id, draw, asset["asset_id"], ts.isoformat())
                failed = random.Random(seed).random() <= p
                binary[(draw, ts, asset["asset_id"])] = int(failed)
                rows.append({"sandbox_id": location_id, "event_id": event_id, "mc_draw": draw, "timestamp": ts, "asset_id": asset["asset_id"], "state": "failed" if failed else "available", "failure_probability": p, "sampled_depth_m": depth, "failure_model": "erad_native_fragility_curve", "failure_model_version": "erad_probability_function", "rng_seed": seed, "source_provenance": json.dumps({"asset_source_provenance": asset.get("source_provenance", ""), "event_source_kind": event_metadata.get("event_source_kind")}, sort_keys=True), "schema_version": asset_states_schema_version})
    return rows, binary


def build_telemetry_observations(assets: list[dict[str, Any]], units: list[dict[str, Any]], binary: dict[tuple[int, datetime, str], int], *, event_id: str, root_seed: int, mc_draws: int) -> list[dict[str, Any]]:
    asset_ids = {a["asset_id"] for a in assets}; timestamps = sorted({k[1] for k in binary}); targets = [a for a in assets if a["asset_type"] in {"source", "transformer", "fuse_proxy"}]
    rows: list[dict[str, Any]] = []
    for draw in range(mc_draws):
        for ts in timestamps:
            for asset in targets:
                seed = stable_seed("observation", root_seed, event_id, draw, asset["asset_id"], ts.isoformat())
                rows.append({"sandbox_id": location_id, "event_id": event_id, "mc_draw": draw, "timestamp_observed": ts, "timestamp_delivered": ts + timedelta(minutes=seed % 16), "target_type": "asset", "target_id": asset["asset_id"], "observation_source": "synthetic_scada_oms", "measured_quantity": "asset_failure_state", "value": float(binary.get((draw, ts, asset["asset_id"]), 0)), "unit": "binary", "noise_model": "none", "delay_model": "deterministic_hash_0_to_15min", "observability_tier": "tier_1_scada_oms", "rng_seed": seed, "source_provenance": json.dumps({"telemetry_synthesis": "direct_asset_state", "source_asset_ids": [asset["asset_id"]]}, sort_keys=True), "schema_version": telemetry_schema_version})
            for unit in units:
                members = [aid for aid in unit["member_asset_ids"] if aid in asset_ids]
                failed = sum(binary.get((draw, ts, aid), 0) for aid in members)
                seed = stable_seed("observation", root_seed, event_id, draw, unit["control_unit_id"], ts.isoformat())
                rows.append({"sandbox_id": location_id, "event_id": event_id, "mc_draw": draw, "timestamp_observed": ts, "timestamp_delivered": ts + timedelta(minutes=seed % 16), "target_type": "control_unit", "target_id": unit["control_unit_id"], "observation_source": "synthetic_control_unit_aggregator", "measured_quantity": "failed_asset_fraction", "value": failed / len(members) if members else 0.0, "unit": "fraction", "noise_model": "none", "delay_model": "deterministic_hash_0_to_15min", "observability_tier": "tier_2_feeder_summary", "rng_seed": seed, "source_provenance": json.dumps({"telemetry_synthesis": "control_unit_fraction", "source_asset_ids": sorted(members)}, sort_keys=True), "schema_version": telemetry_schema_version})
    return rows


def asset_states_schema() -> Any:
    import pyarrow as pa
    return pa.schema([("sandbox_id", pa.string()), ("event_id", pa.string()), ("mc_draw", pa.int32()), ("timestamp", pa.timestamp("us", tz="UTC")), ("asset_id", pa.string()), ("state", pa.string()), ("failure_probability", pa.float64()), ("sampled_depth_m", pa.float64()), ("failure_model", pa.string()), ("failure_model_version", pa.string()), ("rng_seed", pa.int64()), ("source_provenance", pa.string()), ("schema_version", pa.string())])


def telemetry_observations_schema() -> Any:
    import pyarrow as pa
    return pa.schema([("sandbox_id", pa.string()), ("event_id", pa.string()), ("mc_draw", pa.int32()), ("timestamp_observed", pa.timestamp("us", tz="UTC")), ("timestamp_delivered", pa.timestamp("us", tz="UTC")), ("target_type", pa.string()), ("target_id", pa.string()), ("observation_source", pa.string()), ("measured_quantity", pa.string()), ("value", pa.float64()), ("unit", pa.string()), ("noise_model", pa.string()), ("delay_model", pa.string()), ("observability_tier", pa.string()), ("rng_seed", pa.int64()), ("source_provenance", pa.string()), ("schema_version", pa.string())])


def export_stage_a2(output_dir: Path, *, event_id: str = default_event_id, root_seed: int = default_root_seed, mc_draws: int = default_mc_draws, event_depth_csv: Path | None = None, sfincs_event_dir: Path | None = None, max_sample_distance_m: float = default_max_sample_distance_m, synthetic_peak_depth_m: float = default_synthetic_peak_depth_m, event_timestamp: str = default_event_timestamp, debug_csv: bool = False, fragility_model: FragilityModel | None = None) -> dict[str, Any]:
    assets = pd.read_parquet(output_dir / "assets.parquet").to_dict("records"); units = pd.read_parquet(output_dir / "control_units.parquet").to_dict("records")
    flood_assets = flood_relevant_assets(assets)
    if event_depth_csv and sfincs_event_dir:
        raise ValueError("use either event_depth_csv or sfincs_event_dir, not both")
    samples, meta = build_csv_event_samples(event_depth_csv) if event_depth_csv else build_sfincs_event_samples(sfincs_event_dir, flood_assets, max_sample_distance_m=max_sample_distance_m) if sfincs_event_dir else build_synthetic_event_samples(flood_assets, timestamp=parse_utc_timestamp(event_timestamp), peak_depth_m=synthetic_peak_depth_m)
    states, binary = build_asset_states(flood_assets, samples, meta, event_id=event_id, root_seed=root_seed, mc_draws=mc_draws, fragility_model=fragility_model)
    telemetry = build_telemetry_observations(flood_assets, units, binary, event_id=event_id, root_seed=root_seed, mc_draws=mc_draws)
    write_table(output_dir / "asset_states.parquet", states, schema=asset_states_schema()); write_table(output_dir / "telemetry_observations.parquet", telemetry, schema=telemetry_observations_schema())
    if debug_csv:
        pd.DataFrame(states).to_csv(output_dir / "asset_states.debug.csv", index=False); pd.DataFrame(telemetry).to_csv(output_dir / "telemetry_observations.debug.csv", index=False)
    report = {"stage": "stage_a2", "schema_versions": {"asset_states.parquet": asset_states_schema_version, "telemetry_observations.parquet": telemetry_schema_version}, "event_id": event_id, "root_seed": root_seed, "mc_draws": mc_draws, "passed": True, "errors": [], "checks": {"flood_relevant_asset_count": len(flood_assets), "asset_state_count": len(states), "telemetry_observation_count": len(telemetry), "asset_state_counts_by_state": _count_by(states, "state")}}
    write_json(output_dir / "validation_report_stage_a2.json", report); write_json(output_dir / "run_manifest_stage_a2.json", manifest(stage="stage_a2", location_id=location_id, inputs={"assets.parquet": output_dir / "assets.parquet", "control_units.parquet": output_dir / "control_units.parquet"}, outputs={"asset_states.parquet": output_dir / "asset_states.parquet", "telemetry_observations.parquet": output_dir / "telemetry_observations.parquet"}, parameters={"event_id": event_id, "root_seed": root_seed, "mc_draws": mc_draws, "event_source": meta}))
    return report


# Control registry ---------------------------------------------------------

registry_csv_names = ("buses.csv", "lines.csv", "transformers.csv", "sources.csv", "loads.csv", "load_buses.csv", "feeders.csv")
feeder_fields = ["feeder_id", "bus_count", "line_count", "transformer_count", "source_count", "load_count", "load_kw", "load_kvar"]


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh); return list(reader), list(reader.fieldnames or [])


def _write_csv(path: Path, rows: Iterable[Mapping[str, str]], fields: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True); count = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
        for row in rows:
            writer.writerow(row); count += 1
    return count


def _bus_components(buses: list[dict[str, str]], lines: list[dict[str, str]], transformers: list[dict[str, str]]) -> tuple[dict[str, str], Counter[str]]:
    parent = {r["bus"]: r["bus"] for r in buses if r.get("bus")}
    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a: str, b: str) -> None:
        if a and b:
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra
    for row in lines:
        if (row.get("line_class") or "line") == "line": union(row.get("from_bus", ""), row.get("to_bus", ""))
    for row in transformers:
        buses_w = [b.strip() for b in (row.get("winding_buses") or "").split(",") if b.strip()]
        for b in buses_w[1:]: union(buses_w[0], b)
    mapping = {bus: find(bus) for bus in parent}
    return mapping, Counter(mapping.values())


def _component_groups_from_tie_candidates(buses: list[dict[str, str]], mapping: dict[str, str], *, max_distance_m: float, min_line_degree: int) -> list[set[str]]:
    comps = set(mapping.values()); graph = {c: set() for c in comps}
    candidates = [r for r in buses if r.get("bus") in mapping and parse_int(r.get("line_degree"), 0) >= min_line_degree and parse_float(r.get("lon")) is not None and parse_float(r.get("lat")) is not None]
    if not candidates: return [{c} for c in sorted(comps)]
    ref_lat = sum(float(r["lat"]) for r in candidates) / len(candidates); lon_scale = 111320 * math.cos(math.radians(ref_lat)); lat_scale = 110540
    for i, a in enumerate(candidates):
        ax, ay, ac = float(a["lon"]) * lon_scale, float(a["lat"]) * lat_scale, mapping[a["bus"]]
        for b in candidates[i + 1:]:
            if a.get("feeder_id") == b.get("feeder_id"): continue
            bc = mapping[b["bus"]]
            if ac != bc and math.hypot(float(b["lon"]) * lon_scale - ax, float(b["lat"]) * lat_scale - ay) <= max_distance_m:
                graph[ac].add(bc); graph[bc].add(ac)
    groups, seen = [], set()
    for comp in sorted(comps):
        if comp in seen: continue
        stack, group = [comp], set(); seen.add(comp)
        while stack:
            cur = stack.pop(); group.add(cur)
            for nb in graph[cur]:
                if nb not in seen: seen.add(nb); stack.append(nb)
        groups.append(group)
    return groups


def _transformer_in_kept(row: Mapping[str, str], buses: set[str]) -> bool:
    windings = [b.strip() for b in (row.get("winding_buses") or "").split(",") if b.strip()]
    return bool(windings) and all(b in buses for b in windings)


def control_registry(raw_registry_dir: Path | str, output_dir: Path | str, *, max_tie_distance_m: float = 100.0, min_tie_bus_line_degree: int = 2) -> dict[str, int]:
    raw, out = Path(raw_registry_dir), Path(output_dir)
    tables, fields = {}, {}
    for name in registry_csv_names:
        tables[name], fields[name] = _read_csv(raw / name)
    mapping, sizes = _bus_components(tables["buses.csv"], tables["lines.csv"], tables["transformers.csv"])
    groups = _component_groups_from_tie_candidates(tables["buses.csv"], mapping, max_distance_m=max_tie_distance_m, min_line_degree=min_tie_bus_line_degree)
    kept_components = max(groups, key=lambda g: (sum(sizes[c] for c in g), len(g), sorted(g))) if groups else set(sizes)
    kept_buses = {bus for bus, comp in mapping.items() if comp in kept_components}
    filtered = {
        "buses.csv": [r for r in tables["buses.csv"] if r.get("bus") in kept_buses],
        "lines.csv": [r for r in tables["lines.csv"] if r.get("from_bus") in kept_buses and r.get("to_bus") in kept_buses],
        "transformers.csv": [r for r in tables["transformers.csv"] if _transformer_in_kept(r, kept_buses)],
        "sources.csv": [r for r in tables["sources.csv"] if r.get("bus") in kept_buses],
        "loads.csv": [r for r in tables["loads.csv"] if r.get("bus") in kept_buses],
        "load_buses.csv": [r for r in tables["load_buses.csv"] if r.get("bus") in kept_buses],
    }
    filtered["feeders.csv"] = build_feeders(filtered["buses.csv"], filtered["lines.csv"], filtered["transformers.csv"], filtered["sources.csv"], filtered["loads.csv"])
    outputs = {name: _write_csv(out / name, filtered[name], feeder_fields if name == "feeders.csv" else fields[name]) for name in registry_csv_names}
    write_json(out / "summary.json", {"method": "canonical control filter over raw Asset Registry", "outputs": outputs, "control_sandbox_filter": {"input_registry_dir": str(raw), "max_tie_distance_m": max_tie_distance_m, "min_tie_bus_line_degree": min_tie_bus_line_degree, "raw_baseline_components": len(sizes), "retained_baseline_components": len(kept_components), "retained_buses": len(kept_buses)}})
    return outputs
