"""Critical-facility input loading for Location Workspaces."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


CRITICAL_FACILITY_SCHEMA_VERSION = "stage_b_critical_facilities.v0.2"

CRITICAL_FACILITY_COLUMNS = [
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


def stable_token(*parts: object, max_len: int = 48) -> str:
    raw = "_".join(str(part) for part in parts if part is not None)
    slug = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{slug[:max_len].strip('_')}_{digest}"


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
    normalized = pd.DataFrame(rows, columns=CRITICAL_FACILITY_COLUMNS)
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


def empty_critical_facilities_gdf() -> gpd.GeoDataFrame:
    frame = pd.DataFrame(columns=CRITICAL_FACILITY_COLUMNS)
    return gpd.GeoDataFrame(frame, geometry=[], crs="EPSG:4326")


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
        "schema_version": row.get("schema_version") or CRITICAL_FACILITY_SCHEMA_VERSION,
    }
