from __future__ import annotations

from pathlib import Path

import pandas as pd

from wflow_runs.coupled_handoff import read_stream_boundary_handoff_location_artifacts
from wflow_runs.coupling_qa import (
    read_dynamic_handoff_acceptance,
    validate_dynamic_handoff,
    write_dynamic_handoff_acceptance,
)
from wflow_runs.replay import (
    _domain_set_submodels,
    _event_reference_time,
    _sfincs_handoff_locations_for_replay,
    replay_inland_domain_set,
    resolve_event_window,
)
from wflow_runs.build_plan import validate_wflow_staticmaps_physics
from wflow_runs.states import plan_wflow_warmup_state, validate_warmup_forcing, validate_wflow_instates, write_cold_state_workflow
from wflow_runs.streamflow_realization import validate_wflow_streamflow_realization
from wflow_runs.notebook import resolve_location_path


def dynamic_handoff_paths(config: dict, location_root, event_id: str) -> dict[str, Path]:
    location_root = Path(location_root)
    events_root = resolve_location_path(location_root, config.get("wflow", {}).get("events_root", "data/wflow/events"))
    event_root = events_root / str(event_id)
    return {
        "event_root": event_root,
        "discharge": event_root / "sfincs_discharge.nc",
        "qa_csv": event_root / "sfincs_discharge.dynamic_handoff_qa.csv",
        "acceptance": event_root / "sfincs_discharge.dynamic_handoff.json",
        "zero_rain_discharge": event_root / "_zero_rain" / "sfincs_discharge.nc",
    }


def plan_dynamic_wflow_handoff(config: dict, location_root, event_id: str, *, catalog_path=None) -> pd.Series:
    location_root = Path(location_root)
    reference_time = _event_reference_time(location_root, event_id, catalog_path)
    start, end = resolve_event_window(reference_time)
    state_plan = plan_wflow_warmup_state(config, location_root, event_id, reference_time=reference_time)
    paths = dynamic_handoff_paths(config, location_root, event_id)
    acceptance_status = "missing"
    if paths["acceptance"].exists():
        acceptance_status = read_dynamic_handoff_acceptance(paths["acceptance"]).get("status", "unknown")
    return pd.Series(
        {
            "event_id": str(event_id),
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "discharge_source": _configured_discharge_source(config),
            "state_policy": state_plan.get("state_policy", ""),
            "baseline_id": state_plan.get("baseline_id", ""),
            "warmup_days": state_plan["warmup_days"],
            "warmup_baseline_root": state_plan.get("warmup_baseline_root", ""),
            "sfincs_discharge_forcing": str(paths["discharge"]),
            "dynamic_handoff_acceptance": str(paths["acceptance"]),
            "acceptance_status": acceptance_status,
        },
        name="dynamic_wflow_handoff_plan",
    )


def prepare_dynamic_wflow_handoff(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    execute: bool = False,
    zero_rain_discharge_nc=None,
) -> pd.DataFrame:
    """Run or plan the dynamic Wflow handoff and write QA/acceptance artifacts."""
    if _configured_discharge_source(config) != "wflow_dynamic":
        raise ValueError("inland_coupling.discharge_forcing.source must be 'wflow_dynamic'")
    location_root = Path(location_root)
    reference_time = _event_reference_time(location_root, event_id, catalog_path)
    state_plan = plan_wflow_warmup_state(config, location_root, event_id, reference_time=reference_time)
    write_cold_state_workflow(state_plan["cold_state_workflow"], timestamp=state_plan["warmup_start"])
    if execute:
        _validate_dynamic_wflow_base_staticmaps(config, location_root)
        validate_warmup_forcing(config, location_root, event_id, reference_time=reference_time, raise_on_error=True)
        validate_wflow_instates(config, location_root, raise_on_error=True)

    replay_report = replay_inland_domain_set(
        config,
        location_root,
        event_id,
        catalog_path=catalog_path,
        execute=execute,
    )
    paths = dynamic_handoff_paths(config, location_root, event_id)
    if not execute:
        replay_report["dynamic_handoff_acceptance"] = str(paths["acceptance"])
        replay_report["acceptance_status"] = "planned"
        return replay_report

    expected = _expected_handoff_ids(config, location_root)
    thresholds = ((config.get("wflow", {}) or {}).get("dynamic_handoff", {}) or {}).get("qa", {}) or {}
    max_zero = float(thresholds.get("max_zero_rain_peak_fraction", 0.2))
    zero_path = zero_rain_discharge_nc or (paths["zero_rain_discharge"] if paths["zero_rain_discharge"].exists() else None)
    if zero_path is None:
        raise FileNotFoundError(
            f"Dynamic Wflow handoff requires a zero-rain control before acceptance: {paths['zero_rain_discharge']}. "
            "Run the zero-rain Wflow control and rerun handoff QA."
        )
    qa = validate_dynamic_handoff(
        paths["discharge"],
        zero_rain_discharge_nc=zero_path,
        expected_source_ids=expected,
        max_zero_peak_fraction=max_zero,
        raise_on_error=True,
    )
    paths["qa_csv"].parent.mkdir(parents=True, exist_ok=True)
    qa.to_csv(paths["qa_csv"], index=False)
    write_dynamic_handoff_acceptance(
        paths["acceptance"],
        event_id=event_id,
        discharge_nc=paths["discharge"],
        qa_report=qa,
        metadata={
            "warmup_state_plan": state_plan.to_dict(),
            "zero_rain_discharge": "" if zero_path is None else str(zero_path),
            "streamflow_realization": "wflow_external_river_inflow",
        },
    )
    replay_report["dynamic_handoff_acceptance"] = str(paths["acceptance"])
    replay_report["acceptance_status"] = "accepted"
    return replay_report


def require_accepted_dynamic_handoff(config: dict, location_root, event_id: str) -> pd.Series:
    paths = dynamic_handoff_paths(config, location_root, event_id)
    if not paths["acceptance"].exists():
        discharge_note = (
            " An sfincs_discharge.nc file exists, but it has not been accepted by dynamic handoff QA."
            if paths["discharge"].exists()
            else " The dynamic sfincs_discharge.nc file is also missing."
        )
        raise FileNotFoundError(
            f"Dynamic Wflow handoff acceptance is missing for {event_id}: {paths['acceptance']}."
            f"{discharge_note} Run 04/b_prepare_wflow_dynamic_handoff.ipynb with EVENT_ID={event_id!r}, then rerun SFINCS staging."
        )
    payload = read_dynamic_handoff_acceptance(paths["acceptance"])
    if payload.get("status") != "accepted":
        raise RuntimeError(f"Dynamic Wflow handoff is not accepted for {event_id}: {paths['acceptance']}")
    checks = {str(row.get("check", "")) for row in payload.get("checks", [])}
    if "zero_rain_peak_fraction" not in checks:
        raise RuntimeError(
            f"Dynamic Wflow handoff acceptance for {event_id} is stale: "
            f"{paths['acceptance']} lacks zero-rain control QA. "
            "Rerun 04/b_prepare_wflow_dynamic_handoff.ipynb after preparing warmup states and zero-rain control."
        )
    metadata = payload.get("metadata", {}) or {}
    if metadata.get("streamflow_realization") != "wflow_external_river_inflow":
        raise RuntimeError(
            f"Dynamic Wflow handoff acceptance for {event_id} is stale: "
            f"{paths['acceptance']} does not document USGS/POT streamflow consumed by Wflow as external river inflow. "
            "Rerun 04/b_prepare_wflow_dynamic_handoff.ipynb after rebuilding the Wflow event model."
        )
    if not paths["discharge"].exists():
        raise FileNotFoundError(paths["discharge"])
    return pd.Series(
        {
            "event_id": str(event_id),
            "status": payload.get("status"),
            "discharge_source": payload.get("discharge_source"),
            "sfincs_discharge_forcing": str(paths["discharge"]),
            "dynamic_handoff_acceptance": str(paths["acceptance"]),
        },
        name="dynamic_wflow_handoff_acceptance",
    )


def _configured_discharge_source(config: dict) -> str:
    return str(((config.get("inland_coupling", {}) or {}).get("discharge_forcing", {}) or {}).get("source", "wflow_replay")).lower()


def _expected_handoff_ids(config: dict, location_root: Path) -> set[str]:
    model_crs = config.get("wflow", {}).get("model_crs", config.get("project", {}).get("model_crs", "EPSG:32617"))
    locations = _sfincs_handoff_locations_for_replay(config, location_root, model_crs)
    if locations is None or locations.empty:
        locations = read_stream_boundary_handoff_location_artifacts(config, location_root, location_path=resolve_location_path)
    if locations is None or locations.empty:
        return set()
    return set(locations["sfincs_handoff_id"].astype(str))


def _validate_dynamic_wflow_base_staticmaps(config: dict, location_root: Path) -> None:
    base_root = resolve_location_path(location_root, config.get("wflow", {}).get("base_model_root", "data/wflow/base"))
    rows = []
    for submodel in _domain_set_submodels(config, location_root):
        submodel_id = str(submodel["wflow_submodel_id"])
        report = validate_wflow_staticmaps_physics(base_root / submodel_id, raise_on_error=False)
        report.insert(0, "submodel_id", submodel_id)
        rows.append(report)
    if not rows:
        raise RuntimeError("Dynamic Wflow handoff cannot find Wflow submodels for staticmap QA.")
    report = pd.concat(rows, ignore_index=True)
    failed = report[report["status"].isin(["failed", "review_required"])]
    if not failed.empty:
        details = "; ".join(
            f"{row.submodel_id}:{row.check}: {row.message}"
            for row in failed.itertuples()
        )
        raise RuntimeError(f"Dynamic Wflow handoff blocked by Wflow staticmap QA: {details}")


def plan_wflow_streamflow_realization(config: dict, location_root, event_id: str, *, catalog_path=None) -> pd.DataFrame:
    """Notebook-facing readiness report for event streamflow consumed by Wflow."""
    return validate_wflow_streamflow_realization(
        config,
        location_root,
        event_id,
        catalog_path=catalog_path,
        event_model_root=None,
        raise_on_error=False,
    )
