"""PowerModelsONM events.json serializer.

Converts the per-asset survival trajectory in ``asset_states.parquet`` into
the input PMONM uses to schedule restoration actions over the optimization
horizon. PMONM ingests a list of timestamped contingency events
(``events.json``); each event tells the optimizer when a network element
changes status. This serializer emits one ``breaker`` event per
``available → failed`` transition, indexed by PMONM's 1-indexed
``timestep`` field.

Schema (PowerModelsONM input-events schema, R31 in
``simulated_data_protocol.md``):

    [
      {
        "timestep": <int, 1-indexed>,
        "event_type": "breaker",
        "affected_asset": "<element_type>.<dss_name>",
        "event_data": {"status": "OPEN"}
      }
    ]

Inputs:

* ``asset_states``: per-asset survival rows, as produced by
  ``power.export_stage_a2`` for a fixed
  ``(event_id, mc_draw)``. The serializer filters to the requested
  ``(event_id, mc_draw)`` slice and ignores other rows.
* ``asset_to_dss_element``: mapping from Stage A1 ``asset_id`` to the
  OpenDSS element name PMONM will see. Assets without a mapping are
  recorded in ``skipped_asset_ids`` rather than silently dropped, because
  PMONM cannot operate an element it never parsed.

Timestep indexing: PMONM uses 1-indexed timesteps. We compute the PMONM
timestep from the ceiling of ``(timestamp - event_start_utc)`` in hours plus 1,
so an asset that is failed at the first sample emits a ``timestep = 1`` event.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


ONM_EVENTS_SCHEMA_VERSION = "marshfield_onm_events.v0.1"


@dataclass(frozen=True)
class AssetStateRow:
    """One row of the Stage A2 asset-state trajectory.

    Constructed by the caller from ``asset_states.parquet`` rows; using a
    dataclass keeps the public interface explicit and the tracer test small.
    """

    event_id: str
    mc_draw: int
    timestamp: datetime
    asset_id: str
    state: str


@dataclass(frozen=True)
class OnmEventsResult:
    """Serializer output: events plus diagnostics."""

    events: list[dict[str, Any]]
    event_id: str
    mc_draw: int
    event_start_utc: datetime
    schema_version: str = ONM_EVENTS_SCHEMA_VERSION
    skipped_asset_ids: list[str] = field(default_factory=list)


def build_onm_events(
    asset_states: Iterable[AssetStateRow],
    *,
    event_id: str,
    mc_draw: int,
    asset_to_dss_element: Mapping[str, str],
    event_start_utc: datetime,
) -> OnmEventsResult:
    """Build the PMONM events.json payload for one (event_id, mc_draw) slice."""

    if event_start_utc.tzinfo is None:
        raise ValueError("event_start_utc must be timezone-aware (UTC)")
    event_start_utc = event_start_utc.astimezone(timezone.utc)

    relevant = [
        row
        for row in asset_states
        if row.event_id == event_id and row.mc_draw == mc_draw
    ]

    by_asset: dict[str, list[AssetStateRow]] = {}
    for row in relevant:
        by_asset.setdefault(row.asset_id, []).append(row)

    events: list[dict[str, Any]] = []
    skipped: list[str] = []

    for asset_id, rows in by_asset.items():
        rows_sorted = sorted(rows, key=lambda item: item.timestamp)
        transition_row = _first_failure_transition(rows_sorted)
        if transition_row is None:
            continue
        dss_element = asset_to_dss_element.get(asset_id)
        if dss_element is None:
            skipped.append(f"missing_dss_mapping_for_{asset_id}")
            continue
        timestep = _pmonm_timestep(transition_row.timestamp, event_start_utc)
        events.append(
            {
                "timestep": timestep,
                "event_type": "breaker",
                "affected_asset": dss_element,
                "event_data": {"status": "OPEN"},
            }
        )

    events.sort(key=lambda item: (item["timestep"], item["affected_asset"]))

    return OnmEventsResult(
        events=events,
        event_id=event_id,
        mc_draw=mc_draw,
        event_start_utc=event_start_utc,
        skipped_asset_ids=skipped,
    )


def _first_failure_transition(rows: list[AssetStateRow]) -> AssetStateRow | None:
    previous_state: str | None = None
    for row in rows:
        if row.state == "failed" and previous_state != "failed":
            return row
        previous_state = row.state
    return None


def _pmonm_timestep(transition_time: datetime, event_start: datetime) -> int:
    transition_utc = transition_time.astimezone(timezone.utc)
    delta_hours = (transition_utc - event_start).total_seconds() / 3600.0
    return max(1, int(delta_hours) + 1)
