"""Slice a full annual profile down to the PowerModelsONM event window.

PowerModelsONM operates over a finite restoration horizon. EULP /
ResStock / ComStock source profiles are 8760-hour annual series. This module
maps a FLOOD-RM/SFINCS event start timestamp to the corresponding hour-of-year
in the annual profile and returns a contiguous slice of length
``horizon_hours``.

The default horizon is 72 hours, matching the FEMA Community Lifelines
Stabilization horizon already used elsewhere in this codebase
(``dft.power.der_inventory.FEMA_COMMUNITY_LIFELINES_OUTAGE_HOURS``).
This is the same horizon REopt resilience sizing assumes by default.

Citations:

* FEMA Community Lifelines Implementation Toolkit (R15 in
  ``simulated_data_protocol.md``): 72-hour stabilization horizon.
* NREL REopt resilience sizing: ``critical_load_fraction``-weighted outage
  with default ``outage_duration_hours = 72``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import pandas as pd

from power.load_profiles import build_archetype_load_profile
from power.load_uncertainty import build_load_uncertainty_bounds


DEFAULT_FEMA_LIFELINES_HORIZON_HOURS = 72
HOURS_PER_YEAR = 8760


@dataclass(frozen=True)
class EventWindow:
    """A contiguous slice of an annual profile aligned to an event start.

    ``values`` has length ``horizon_hours`` and is wrapped across the year
    boundary if the event extends past December 31 23:00 UTC.
    """

    values: list[float]
    start_hour_of_year: int
    end_hour_of_year: int
    horizon_hours: int
    weather_year: int
    event_start_utc: datetime
    wrapped_across_year_boundary: bool


def hour_of_year(timestamp: datetime) -> int:
    """Return the zero-indexed hour-of-year for a UTC timestamp.

    January 1 00:00 UTC maps to 0; January 1 14:00 UTC maps to 14; for
    non-leap years December 31 23:00 UTC maps to 8759.
    """

    if timestamp.tzinfo is None:
        raise ValueError("hour_of_year requires a timezone-aware (UTC) timestamp")
    utc_timestamp = timestamp.astimezone(timezone.utc)
    year_start = datetime(utc_timestamp.year, 1, 1, tzinfo=timezone.utc)
    delta = utc_timestamp - year_start
    return int(delta.total_seconds() // 3600)


def slice_annual_profile_to_event_window(
    annual_profile: Sequence[float],
    *,
    event_start_utc: datetime,
    weather_year: int,
    horizon_hours: int = DEFAULT_FEMA_LIFELINES_HORIZON_HOURS,
) -> EventWindow:
    """Slice an 8760-hour profile to ``horizon_hours`` starting at the event.

    The slice wraps across the year boundary when needed so the returned
    window always has length ``horizon_hours``.
    """

    if event_start_utc.tzinfo is None:
        raise ValueError("event_start_utc must be timezone-aware (UTC)")
    if len(annual_profile) != HOURS_PER_YEAR:
        raise ValueError(
            f"annual_profile must have length {HOURS_PER_YEAR}; "
            f"got {len(annual_profile)}"
        )
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")

    start_hour = hour_of_year(event_start_utc)
    raw_end_hour = start_hour + horizon_hours
    wrapped = raw_end_hour > HOURS_PER_YEAR

    if not wrapped:
        values = list(annual_profile[start_hour:raw_end_hour])
    else:
        tail = list(annual_profile[start_hour:HOURS_PER_YEAR])
        head = list(annual_profile[0 : raw_end_hour - HOURS_PER_YEAR])
        values = tail + head

    return EventWindow(
        values=values,
        start_hour_of_year=start_hour,
        end_hour_of_year=raw_end_hour,
        horizon_hours=horizon_hours,
        weather_year=weather_year,
        event_start_utc=event_start_utc.astimezone(timezone.utc),
        wrapped_across_year_boundary=wrapped,
    )


def build_event_window_bundle(
    *,
    event_start: datetime,
    horizon_hours: int,
    load_profiles: pd.DataFrame,
    blocks: pd.DataFrame,
    sandbox_id: str,
    uncertainty_band: float = 0.20,
    event_id: str = "preview_event",
    mc_draw: int = 0,
) -> dict[str, Any]:
    """Build a notebook-preview ONM event-window demand bundle.

    ``load_profile_assignments.parquet`` records the profile assignment and
    provenance, while the current Stage B preview regenerates synthetic 8760
    archetype profiles from that provenance. This wrapper keeps the notebook
    readable and returns the four event-window slices needed for inspection:
    nodal demand windows, per-load uncertainty bands, block demand summaries,
    and scalar bundle metadata.
    """

    event_start_utc = _coerce_utc(event_start)
    load_to_block = _load_asset_to_block_id(blocks, sandbox_id=sandbox_id)
    nodal_demand: list[dict[str, Any]] = []
    nominal_windows: list[dict[str, Any]] = []
    weather_years: set[int] = set()

    for row in load_profiles.itertuples(index=False):
        record = row._asdict()
        provenance = _profile_provenance(record)
        archetype = {
            "schedule_overlay": provenance.get("schedule_overlay", "business_hours"),
            "profile_source": record.get("profile_source"),
            "source_building_type": record.get("source_building_type"),
            "source_geography": record.get("source_geography"),
        }
        weather_year = int(record.get("weather_year") or event_start_utc.year)
        weather_years.add(weather_year)
        annual_profile = build_archetype_load_profile(
            archetype,
            peak_kw=float(record["peak_kw"]),
        )
        window = slice_annual_profile_to_event_window(
            annual_profile,
            event_start_utc=event_start_utc,
            weather_year=weather_year,
            horizon_hours=horizon_hours,
        )
        load_asset_id = str(record["load_asset_id"])
        block_id = load_to_block.get(load_asset_id, "unassigned_block")
        demand_row = {
            "load_asset_id": load_asset_id,
            "loadshape_id": str(record["loadshape_id"]),
            "block_id": block_id,
            "feeder_id": "" if pd.isna(record.get("feeder_id")) else str(record.get("feeder_id")),
            "customer_class": str(record.get("customer_class", "")),
            "peak_kw": float(record["peak_kw"]),
            "values": window.values,
            "start_hour_of_year": window.start_hour_of_year,
            "wrapped_across_year_boundary": window.wrapped_across_year_boundary,
        }
        nodal_demand.append(demand_row)
        nominal_windows.append(
            {
                "load_asset_id": load_asset_id,
                "block_id": block_id,
                "feeder_id": demand_row["feeder_id"],
                "values": window.values,
            }
        )

    uncertainty_bands = build_load_uncertainty_bounds(
        nominal_windows,
        event_id=event_id,
        mc_draw=mc_draw,
        band_fraction=uncertainty_band,
    )
    block_summary = _block_demand_summary(nodal_demand)
    event_end = event_start_utc + timedelta(hours=horizon_hours)

    return {
        "event_start": event_start_utc.isoformat(),
        "event_end": event_end.isoformat(),
        "horizon_hours": horizon_hours,
        "timestep_count": horizon_hours,
        "load_profile_count": int(len(load_profiles)),
        "block_count": len(block_summary),
        "uncertainty_band": uncertainty_band,
        "uncertainty_row_count": len(uncertainty_bands),
        "weather_years": sorted(weather_years),
        "nodal_demand": nodal_demand,
        "uncertainty_bands": uncertainty_bands,
        "block_demand_summary": block_summary,
    }


def _coerce_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _profile_provenance(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("source_provenance")
    if raw is None or pd.isna(raw):
        return {}
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}


def _load_asset_to_block_id(blocks: pd.DataFrame, *, sandbox_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if blocks.empty or "buses_json" not in blocks.columns:
        return out
    for row in blocks.itertuples(index=False):
        block_id = str(row.block_id)
        for bus in json.loads(row.buses_json):
            bus_asset_token = _slug(str(bus))
            out[f"{sandbox_id}:asset:load_buses:{bus}"] = block_id
            out[f"{sandbox_id}:asset:loads:{bus}"] = block_id
            out[f"{sandbox_id}:asset:load_buses:{bus_asset_token}"] = block_id
            out[f"{sandbox_id}:asset:loads:{bus_asset_token}"] = block_id
    return out


def _slug(value: str) -> str:
    lowered = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return normalized or "unknown"


def _block_demand_summary(nodal_demand: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_block: dict[str, dict[str, Any]] = {}
    for row in nodal_demand:
        block_id = str(row["block_id"])
        entry = by_block.setdefault(
            block_id,
            {
                "block_id": block_id,
                "load_count": 0,
                "peak_window_kw": 0.0,
                "energy_window_kwh": 0.0,
            },
        )
        values = [float(value) for value in row["values"]]
        entry["load_count"] += 1
        entry["peak_window_kw"] += max(values) if values else 0.0
        entry["energy_window_kwh"] += sum(values)
    return sorted(by_block.values(), key=lambda item: item["block_id"])
