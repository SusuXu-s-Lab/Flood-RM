# Critical facility evidence

"""Critical-facility input loading for Location Workspaces."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


facility_version = "stage_b_critical_facilities.v0.2"

facility_columns = [
    "sandbox_id",
    "facility_id",
    "facility_name",
    "lifeline",
    "lifeline_component",
    "facility_class",
    "criticality_tier",
    "criticality_weight",
    "lon",
    "lat",
    "municipality_id",
    "source_dataset",
    "source_url",
    "source_date",
    "source_record_id",
    "evidence_rank",
    "confidence",
    "backup_power_status",
    "resilience_asset_type",
    "hazard_exposure_summary",
    "source_provenance",
    "schema_version",
]


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "unknown"


def stable_token(*parts: object, max_len: int = 48) -> str:
    raw = "_".join(str(part) for part in parts if part is not None)
    token = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{token[:max_len].strip('_')}_{digest}"


def _present(value) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def load_critical_facilities(path: Path, *, location_name: str) -> gpd.GeoDataFrame:
    """Load reviewed critical facilities from a location-local source file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"critical facilities source is missing: {path}")
    if path.suffix.lower() in {".geojson", ".json"}:
        facilities = gpd.read_file(path)
        if "lon" not in facilities or "lat" not in facilities:
            facilities["lon"] = facilities.geometry.x
            facilities["lat"] = facilities.geometry.y
    elif path.suffix.lower() == ".parquet":
        facilities = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        facilities = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported critical facilities source format: {path.suffix}")

    frame = pd.DataFrame(facilities.drop(columns=["geometry"], errors="ignore")).copy()
    rows = [_normalize_facility_row(row, location_name=location_name) for row in frame.to_dict("records")]
    normalized = pd.DataFrame(rows, columns=facility_columns)
    normalized = normalized.dropna(subset=["lon", "lat"]).copy()
    return gpd.GeoDataFrame(
        normalized,
        geometry=[Point(xy) for xy in zip(normalized["lon"], normalized["lat"])],
        crs="EPSG:4326",
    )


def write_critical_facilities_artifact(
    facilities: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(facilities.drop(columns=["geometry"], errors="ignore")).to_parquet(output_path, index=False)


def load_bus_electrical_metadata(loads: pd.DataFrame) -> dict[str, dict[str, object]]:
    metadata: dict[str, dict[str, object]] = {}
    for bus, group in loads.groupby("bus"):
        phases = group["phases"].dropna().astype(str)
        kv = pd.to_numeric(group["kv"], errors="coerce").dropna()
        metadata[str(bus)] = {
            "phases": phases.iloc[0] if not phases.empty else None,
            "nominal_voltage_kv": float(kv.max()) if not kv.empty else None,
        }
    return metadata


def _normalize_facility_row(row: dict[str, Any], *, location_name: str) -> dict[str, Any]:
    facility_name = row.get("facility_name") or row.get("name")
    address = row.get("address")
    token = stable_token(facility_name, address, max_len=56)
    source_provenance = row.get("source_provenance")
    if isinstance(source_provenance, dict):
        source_provenance = json.dumps(source_provenance, sort_keys=True)
    hazard_exposure_summary = row.get("hazard_exposure_summary")
    if isinstance(hazard_exposure_summary, dict):
        hazard_exposure_summary = json.dumps(hazard_exposure_summary, sort_keys=True)

    return {
        "sandbox_id": row.get("sandbox_id") or location_name,
        "facility_id": row.get("facility_id") or f"{location_name}:critical_facility:{token}",
        "facility_name": facility_name,
        "lifeline": row.get("lifeline"),
        "lifeline_component": row.get("lifeline_component") or row.get("component"),
        "facility_class": row.get("facility_class"),
        "criticality_tier": row.get("criticality_tier", row.get("tier")),
        "criticality_weight": row.get("criticality_weight", row.get("weight")),
        "lon": float(row["lon"]),
        "lat": float(row["lat"]),
        "municipality_id": row.get("municipality_id") or f"{location_name}:municipality:{location_name}",
        "source_dataset": row.get("source_dataset"),
        "source_url": row.get("source_url"),
        "source_date": row.get("source_date"),
        "source_record_id": row.get("source_record_id") or facility_name,
        "evidence_rank": row.get("evidence_rank"),
        "confidence": row.get("confidence"),
        "backup_power_status": row.get("backup_power_status"),
        "resilience_asset_type": row.get("resilience_asset_type"),
        "hazard_exposure_summary": hazard_exposure_summary or json.dumps({"status": "pending_hazard_overlay"}, sort_keys=True),
        "source_provenance": source_provenance or json.dumps({"source": "location_critical_facilities"}, sort_keys=True),
        "schema_version": row.get("schema_version") or facility_version,
    }


# Load matches

"""Stage B load-match helpers.

Critical facilities are public geographic evidence. Electrical service points
are synthetic Grid Dataset assets. This module builds the auditable bridge between
those two data channels without treating nearest-neighbor matches as utility
truth.
"""

load_match_version = "stage_b_load_matches.v0.2"


class LoadMatchViolation(ValueError):
    """Raised when load-match rows violate the artifact contract."""


def _facility_token(facility_id: str) -> str:
    return _slug(facility_id.rsplit(":", 1)[-1])


def _asset_token(asset_id: str | None) -> str:
    if not asset_id:
        return "unmatched"
    return _slug(asset_id.rsplit(":", 1)[-1])


def _distance_m(lon_a: Any, lat_a: Any, lon_b: Any, lat_b: Any) -> float:
    """Equirectangular distance, accurate enough for town-scale matching."""

    lon1 = math.radians(float(lon_a))
    lat1 = math.radians(float(lat_a))
    lon2 = math.radians(float(lon_b))
    lat2 = math.radians(float(lat_b))
    x = (lon2 - lon1) * math.cos((lat1 + lat2) / 2.0)
    y = lat2 - lat1
    return math.hypot(x, y) * 6_371_000.0


def _load_bus_candidates(asset_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for row in asset_rows:
        if row.get("asset_type") != "load_bus":
            continue
        if row.get("coordinate_status") not in (None, "valid"):
            continue
        if not (_present(row.get("lon")) and _present(row.get("lat")) and _present(row.get("bus"))):
            continue
        candidates.append(dict(row))
    return candidates


def _control_unit_by_feeder(control_unit_rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in control_unit_rows:
        feeder_id = row.get("source_feeder_id")
        unit_id = row.get("control_unit_id")
        if _present(feeder_id) and _present(unit_id):
            mapping[str(feeder_id)] = str(unit_id)
    return mapping


def _confidence(distance_m: float) -> str:
    if distance_m <= 75.0:
        return "high"
    if distance_m <= 300.0:
        return "medium"
    return "low"


def build_load_matches(
    facility_rows: Iterable[Mapping[str, Any]],
    *,
    asset_rows: Iterable[Mapping[str, Any]],
    control_unit_rows: Iterable[Mapping[str, Any]],
    load_bus_electrical_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    location_id: str,
    max_assignment_distance_m: float = 300.0,
    assign_nearest_when_outside_radius: bool = False,
) -> list[dict[str, Any]]:
    """Match public critical facilities to nearest valid synthetic load buses.

    By default, the assignment is operational only when the nearest Stage A
    ``load_bus`` asset lies inside ``max_assignment_distance_m``. Farther
    facilities are retained as ``unmatched`` with their nearest candidate
    recorded in provenance. When ``assign_nearest_when_outside_radius`` is set,
    the nearest valid load bus is selected anyway and provenance marks that the
    service point is a reviewed proxy rather than a distance-qualified match.
    """

    candidates = _load_bus_candidates(asset_rows)
    unit_by_feeder = _control_unit_by_feeder(control_unit_rows)
    electrical = load_bus_electrical_metadata or {}
    rows: list[dict[str, Any]] = []

    for facility in facility_rows:
        facility_id = str(facility["facility_id"])
        token = _facility_token(facility_id)
        nearest: dict[str, Any] | None = None
        nearest_distance = math.inf
        if _present(facility.get("lon")) and _present(facility.get("lat")):
            for candidate in candidates:
                distance = _distance_m(
                    facility["lon"],
                    facility["lat"],
                    candidate["lon"],
                    candidate["lat"],
                )
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest = candidate

        outside_radius = nearest is not None and nearest_distance > max_assignment_distance_m
        assigned = nearest is not None and (
            nearest_distance <= max_assignment_distance_m
            or assign_nearest_when_outside_radius
        )
        if assigned:
            matched_asset_id = str(nearest["asset_id"])
            matched_bus = str(nearest["bus"])
            feeder_id = str(nearest.get("feeder_id") or "")
            metadata = electrical.get(matched_bus, {})
            status = "assigned"
            confidence = _confidence(nearest_distance)
            match_distance = round(nearest_distance, 3)
            control_unit_id = unit_by_feeder.get(feeder_id)
        else:
            matched_asset_id = None
            matched_bus = None
            feeder_id = None
            metadata = {}
            status = "unmatched"
            confidence = "low"
            match_distance = None
            control_unit_id = None

        nearest_payload = None
        if nearest is not None:
            nearest_payload = {
                "asset_id": nearest.get("asset_id"),
                "bus": nearest.get("bus"),
                "distance_m": round(nearest_distance, 3),
                "feeder_id": nearest.get("feeder_id"),
            }

        provenance = {
            "facility_name": facility.get("facility_name"),
            "facility_source_dataset": facility.get("source_dataset"),
            "matching_rule": "nearest_valid_stage_a_load_bus",
            "max_assignment_distance_m": max_assignment_distance_m,
            "assign_nearest_when_outside_radius": assign_nearest_when_outside_radius,
            "assigned_outside_radius": bool(assigned and outside_radius),
            "assignment_policy": (
                "nearest_load_bus_even_when_outside_radius"
                if assign_nearest_when_outside_radius
                else "nearest_load_bus_within_radius"
            ),
            "candidate_count": len(candidates),
            "nearest_candidate": nearest_payload,
            "selected": nearest_payload if assigned else None,
            "method_note": (
                "Public critical-facility point matched to synthetic Stage A load_bus "
                "asset; this is a reproducible Grid Dataset service proxy, not a utility "
                "service-record claim."
            ),
        }

        load_or_proxy_token = _asset_token(matched_asset_id)
        rows.append(
            {
                "sandbox_id": location_id,
                "assignment_id": f"{location_id}:load_match:{token}:{load_or_proxy_token}",
                "facility_id": facility_id,
                "load_asset_id": matched_asset_id,
                "matched_asset_id": matched_asset_id,
                "matched_asset_type": "load_bus" if assigned else None,
                "matched_bus": matched_bus,
                "phases": metadata.get("phases"),
                "nominal_voltage_kv": metadata.get("nominal_voltage_kv"),
                "control_unit_id": control_unit_id,
                "match_method": "nearest_load_bus" if assigned else "nearest_load_bus_unmatched",
                "match_distance_m": match_distance,
                "assignment_confidence": confidence,
                "criticality_tier": (
                    str(facility.get("criticality_tier"))
                    if _present(facility.get("criticality_tier"))
                    else None
                ),
                "criticality_weight": float(facility.get("criticality_weight") or 0.0),
                "assignment_status": status,
                "source_provenance": json.dumps(provenance, sort_keys=True),
                "schema_version": load_match_version,
            }
        )

    rows.sort(key=lambda row: row["assignment_id"])
    return rows


def validate_load_matches(
    rows: Iterable[Mapping[str, Any]],
    *,
    facility_ids: Iterable[str],
    asset_ids: Iterable[str],
    control_unit_ids: Iterable[str],
) -> None:
    """Validate load-match rows reference known facility/electrical artifacts."""

    facility_id_set = {str(value) for value in facility_ids}
    asset_id_set = {str(value) for value in asset_ids}
    control_unit_id_set = {str(value) for value in control_unit_ids}

    seen: set[str] = set()
    for row in rows:
        assignment_id = str(row.get("assignment_id"))
        if assignment_id in seen:
            raise LoadMatchViolation(f"duplicate assignment_id {assignment_id!r}")
        seen.add(assignment_id)

        facility_id = str(row.get("facility_id"))
        if facility_id not in facility_id_set:
            raise LoadMatchViolation(
                f"{assignment_id}: unknown facility_id {facility_id!r}"
            )

        status = row.get("assignment_status")
        if status == "assigned":
            load_asset_id = row.get("load_asset_id")
            control_unit_id = row.get("control_unit_id")
            if not _present(load_asset_id):
                raise LoadMatchViolation(
                    f"{assignment_id}: assigned row lacks load_asset_id"
                )
            if str(load_asset_id) not in asset_id_set:
                raise LoadMatchViolation(
                    f"{assignment_id}: unknown load_asset_id {load_asset_id!r}"
                )
            if not _present(row.get("matched_bus")):
                raise LoadMatchViolation(
                    f"{assignment_id}: assigned row lacks matched_bus"
                )
            if not _present(control_unit_id) or str(control_unit_id) not in control_unit_id_set:
                raise LoadMatchViolation(
                    f"{assignment_id}: unknown control_unit_id {control_unit_id!r}"
                )
        elif status in {"unmatched", "needs_review", "excluded_duplicate"}:
            continue
        else:
            raise LoadMatchViolation(
                f"{assignment_id}: invalid assignment_status {status!r}"
            )


def load_match_schema() -> Any:
    """Return the schema for the load-match parquet artifact."""

    import pyarrow as pa

    return pa.schema(
        [
            pa.field("sandbox_id", pa.string(), nullable=False),
            pa.field("assignment_id", pa.string(), nullable=False),
            pa.field("facility_id", pa.string(), nullable=False),
            pa.field("load_asset_id", pa.string(), nullable=True),
            pa.field("matched_asset_id", pa.string(), nullable=True),
            pa.field("matched_asset_type", pa.string(), nullable=True),
            pa.field("matched_bus", pa.string(), nullable=True),
            pa.field("phases", pa.string(), nullable=True),
            pa.field("nominal_voltage_kv", pa.float64(), nullable=True),
            pa.field("control_unit_id", pa.string(), nullable=True),
            pa.field("match_method", pa.string(), nullable=False),
            pa.field("match_distance_m", pa.float64(), nullable=True),
            pa.field("assignment_confidence", pa.string(), nullable=False),
            pa.field("criticality_tier", pa.string(), nullable=True),
            pa.field("criticality_weight", pa.float64(), nullable=False),
            pa.field("assignment_status", pa.string(), nullable=False),
            pa.field("source_provenance", pa.string(), nullable=False),
            pa.field("schema_version", pa.string(), nullable=False),
        ]
    )


def write_load_matches(rows: Iterable[Mapping[str, Any]], output_path: Path) -> None:
    """Write load-match rows to the canonical parquet artifact."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = load_match_schema()
    row_list = list(rows)
    columns = {field.name: [row.get(field.name) for row in row_list] for field in schema}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns, schema=schema), output_path)
