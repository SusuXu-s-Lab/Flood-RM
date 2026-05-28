"""End-to-end PowerModelsONM input bundler.

Composes the four MVP slices into one per-(event_id, mc_draw) artifact bundle
that PowerModelsONM and the Moring et al. (2025) RPOP solver can consume
directly:

1. ``dft.power.nodal_load_profiles.assign_nodal_load_profiles`` —
   diversity-preserving ResStock/ComStock archetype assignment for every
   non-critical load, using Eversource MA rate-class kW thresholds.
2. ``dft.power.event_window.slice_annual_profile_to_event_window`` — slice
   each nodal annual profile down to the 72-hour FEMA Community Lifelines
   restoration window anchored at the FLOOD-RM/SFINCS event start.
3. ``dft.power.load_uncertainty.build_load_uncertainty_bounds`` — per-(load,
   timestep) bounds with Switch-Bounded Load Block cluster ids, matching the
   paper's set Γ (eqn 12).
4. ``dft.power.onm_events.build_onm_events`` — PMONM events.json entries
   from the Stage A2 ``asset_states`` trajectory.

The bundle is the minimum sufficient input shape for the paper's two-stage
RPOP plus MFRT-OPF formulation; PV time series is intentionally omitted
because Moring et al. (2025) treat DGs as controllable injections without
weather-driven stochastic generation and mark storage / inverter-based DERs
with their own dynamics as future work (Section VI).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from power.event_window import (
    DEFAULT_FEMA_LIFELINES_HORIZON_HOURS,
    slice_annual_profile_to_event_window,
)
from power.load_profiles import LOAD_PROFILE_ASSIGNMENTS_SCHEMA_VERSION
from power.load_uncertainty import (
    DEFAULT_LOAD_UNCERTAINTY_BAND_FRACTION,
    LOAD_UNCERTAINTY_SCHEMA_VERSION,
    build_load_uncertainty_bounds,
)
from power.nodal_load_profiles import assign_nodal_load_profiles
from power.onm_events import (
    AssetStateRow,
    ONM_EVENTS_SCHEMA_VERSION,
    build_onm_events,
)


ONM_RUN_SCHEMA_VERSION = "power_onm_run.v0.1"


@dataclass(frozen=True)
class NodalAnnualProfile:
    """An 8760-hour annual demand profile for one nodal load."""

    load_name: str
    annual_kw: list[float]


@dataclass(frozen=True)
class OnmEventArtifacts:
    """Paths to the per-(event, MC draw) PowerModelsONM input bundle."""

    output_dir: Path
    events_json_path: Path
    load_uncertainty_json_path: Path
    nominal_load_window_json_path: Path
    load_profile_assignments_json_path: Path
    manifest_path: Path


def build_onm_event_window_artifacts(
    *,
    loads: Sequence[Mapping[str, Any]],
    annual_profiles: Mapping[str, NodalAnnualProfile],
    profile_pool: Mapping[str, Sequence[Mapping[str, Any]]],
    asset_states: Sequence[AssetStateRow],
    asset_to_dss_element: Mapping[str, str],
    event_id: str,
    mc_draw: int,
    event_start_utc: datetime,
    weather_year: int,
    root_seed: int,
    output_dir: Path,
    sandbox_id: str,
    horizon_hours: int = DEFAULT_FEMA_LIFELINES_HORIZON_HOURS,
    uncertainty_band_fraction: float = DEFAULT_LOAD_UNCERTAINTY_BAND_FRACTION,
) -> OnmEventArtifacts:
    """Build the per-(event_id, mc_draw) PowerModelsONM input bundle."""

    output_dir.mkdir(parents=True, exist_ok=True)

    assignments = assign_nodal_load_profiles(
        loads,
        profile_pool=profile_pool,
        root_seed=root_seed,
        sandbox_id=sandbox_id,
        weather_year=weather_year,
    )

    nominal_window_by_load_name: dict[str, list[float]] = {}
    nominal_windows: list[dict[str, Any]] = []
    for record in loads:
        load_name = str(record["load_name"])
        annual = annual_profiles.get(load_name)
        if annual is None:
            raise KeyError(f"missing annual profile for load_name={load_name!r}")
        window = slice_annual_profile_to_event_window(
            annual.annual_kw,
            event_start_utc=event_start_utc,
            weather_year=weather_year,
            horizon_hours=horizon_hours,
        )
        nominal_window_by_load_name[load_name] = window.values
        nominal_windows.append(
            {
                "load_asset_id": f"{sandbox_id}:asset:loads:{load_name}",
                "block_id": str(record["block_id"]),
                "feeder_id": str(record["feeder_id"]),
                "values": window.values,
            }
        )

    bounds = build_load_uncertainty_bounds(
        nominal_windows,
        event_id=event_id,
        mc_draw=mc_draw,
        band_fraction=uncertainty_band_fraction,
    )

    onm_events_result = build_onm_events(
        asset_states,
        event_id=event_id,
        mc_draw=mc_draw,
        asset_to_dss_element=asset_to_dss_element,
        event_start_utc=event_start_utc,
    )

    events_path = output_dir / "events.json"
    load_uncertainty_path = output_dir / "load_uncertainty.json"
    nominal_window_path = output_dir / "nominal_load_window.json"
    assignments_path = output_dir / "load_profile_assignments.json"
    manifest_path = output_dir / "manifest.json"

    events_path.write_text(json.dumps(onm_events_result.events, indent=2))
    load_uncertainty_path.write_text(json.dumps(bounds, indent=2))
    nominal_window_path.write_text(json.dumps(nominal_window_by_load_name, indent=2))
    assignments_path.write_text(json.dumps(assignments, indent=2))

    manifest = {
        "schema_version": ONM_RUN_SCHEMA_VERSION,
        "event_id": event_id,
        "mc_draw": mc_draw,
        "event_start_utc": event_start_utc.isoformat(),
        "weather_year": weather_year,
        "root_seed": root_seed,
        "horizon_hours": horizon_hours,
        "uncertainty_band_fraction": uncertainty_band_fraction,
        "load_count": len(loads),
        "event_count": len(onm_events_result.events),
        "uncertainty_row_count": len(bounds),
        "skipped_asset_ids": onm_events_result.skipped_asset_ids,
        "artifact_paths": {
            "events_json": str(events_path),
            "load_uncertainty_json": str(load_uncertainty_path),
            "nominal_load_window_json": str(nominal_window_path),
            "load_profile_assignments_json": str(assignments_path),
        },
        "load_profile_assignments_schema_version": LOAD_PROFILE_ASSIGNMENTS_SCHEMA_VERSION,
        "load_uncertainty_schema_version": LOAD_UNCERTAINTY_SCHEMA_VERSION,
        "onm_events_schema_version": ONM_EVENTS_SCHEMA_VERSION,
        "citations": {
            "rpop_formulation": (
                "Moring, H. et al. 'Reconfiguration and Real-Time Operation of "
                "Networked Microgrids Under Load Uncertainty.' arXiv:2504.15084v2 "
                "(2025)."
            ),
            "fema_community_lifelines": (
                "FEMA Community Lifelines Implementation Toolkit "
                "(72-hour stabilization horizon)."
            ),
            "load_diversity": (
                "Zhu, X. and Mather, B. 'Data-Driven Load Diversity and "
                "Variability Modeling for Quasi-Static Time-Series Simulation "
                "on Distribution Feeders.' NREL."
            ),
            "eulp_sources": (
                "Wilson, E. et al. 'End-Use Load Profiles for the U.S. "
                "Building Stock.' NREL/TP-5500-80889, 2022."
            ),
            "eversource_ma_tariff": (
                "Eversource Massachusetts retail electric tariff schedules "
                "MDPU 1310 (R-1, R-2, G-1, G-2, G-3)."
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    return OnmEventArtifacts(
        output_dir=output_dir,
        events_json_path=events_path,
        load_uncertainty_json_path=load_uncertainty_path,
        nominal_load_window_json_path=nominal_window_path,
        load_profile_assignments_json_path=assignments_path,
        manifest_path=manifest_path,
    )
