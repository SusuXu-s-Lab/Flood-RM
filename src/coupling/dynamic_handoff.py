from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from coupling.handoff_sources import read_stream_boundary_handoff_location_artifacts
from wflow_runs.coupling_qa import (
    read_dynamic_handoff_acceptance,
    validate_dynamic_handoff,
    validate_handoff_gauge_locations,
    write_dynamic_handoff_acceptance,
)
from wflow_runs.replay import (
    _domain_set_submodels,
    _event_reference_time,
    _sfincs_handoff_locations_for_replay,
    build_meteo,
    configured_event_window_hours,
    replay_inland_domain_set,
    resolve_event_window,
    run_zero_rain_control,
)
from wflow_runs.reservoirs import validate_wflow_reservoir_staticmaps, write_wflow_reservoir_readiness
from wflow_runs.staticmaps_qa import validate_staticmaps
from wflow_runs.states import plan_wflow_warmup_state, validate_warmup_forcing, validate_instates, write_cold_state_workflow
from paths import resolve_location_path
from wflow_runs.event import (
    event_paths as v2_event_paths,
    legacy_dynamic_handoff_paths as v2_legacy_dynamic_handoff_paths,
    require_discharge_window as require_v2_discharge_window,
    require_event_boundary as require_v2_event_boundary,
)


@dataclass(frozen=True)
class DynamicHandoffRun:
    """Notebook result for preparing or reusing one dynamic Wflow handoff."""

    summary: pd.Series
    acceptance: pd.Series
    meteo_report: pd.Series | None = None
    handoff_report: pd.DataFrame | None = None


def dynamic_handoff_paths(config: dict, location_root, event_id: str) -> dict[str, Path]:
    return v2_legacy_dynamic_handoff_paths(config, location_root, event_id)


def plan_handoff(config: dict, location_root, event_id: str, *, catalog_path=None) -> pd.Series:
    location_root = Path(location_root)
    reference_time = _event_reference_time(location_root, event_id, catalog_path)
    pre_event_hours, post_event_hours = configured_event_window_hours(config)
    start, end = resolve_event_window(
        reference_time,
        pre_event_hours=pre_event_hours,
        post_event_hours=post_event_hours,
    )
    state_plan = plan_wflow_warmup_state(config, location_root, event_id, reference_time=reference_time)
    paths = dynamic_handoff_paths(config, location_root, event_id)
    v2_paths = v2_event_paths(config, location_root, event_id)
    acceptance_path = paths["acceptance"]
    acceptance_status = "missing"
    if paths["acceptance"].exists():
        acceptance_status = read_dynamic_handoff_acceptance(paths["acceptance"]).get("status", "unknown")
    elif v2_paths["acceptance"].exists():
        acceptance_path = v2_paths["acceptance"]
        acceptance_status = read_dynamic_handoff_acceptance(v2_paths["acceptance"]).get("status", "unknown")
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
            "dynamic_handoff_acceptance": str(acceptance_path),
            "acceptance_status": acceptance_status,
        },
        name="dynamic_wflow_handoff_plan",
    )


def ensure_dynamic_handoff(
    config: dict,
    location_root,
    event_id: str,
    *,
    catalog_path=None,
    rerun: bool = False,
    run: bool = True,
) -> DynamicHandoffRun:
    """Reuse a current accepted handoff, or build/accept it for this event."""
    location_root = Path(location_root)
    catalog_path = _resolve_catalog_path(location_root, catalog_path)
    instates = _instate_paths(config, location_root)

    handoff_issue = ""
    try:
        accepted = require_handoff(config, location_root, event_id, catalog_path=catalog_path)
        needs_dynamic_wflow = bool(rerun)
    except Exception as exc:
        accepted = None
        handoff_issue = str(exc)
        needs_dynamic_wflow = True

    meteo_report = None
    handoff_report = None
    if needs_dynamic_wflow:
        missing_instates = [str(path) for path in instates if not path.exists()]
        if missing_instates:
            raise FileNotFoundError(
                "Missing prepared Wflow instate(s): "
                + "; ".join(missing_instates)
                + ". Run b_prepare_wflow_dynamic_handoff.ipynb first."
            )
        if not run:
            paths = dynamic_handoff_paths(config, location_root, event_id)
            raise FileNotFoundError(
                f"Missing accepted Wflow handoff for {event_id}: {paths['acceptance']}. "
                "Set run_dynamic_wflow_handoff=True or run b_prepare_wflow_dynamic_handoff.ipynb first."
            )
        meteo_report = pd.Series(
            build_meteo(
                config,
                location_root,
                event_id,
                catalog_path=catalog_path,
                overwrite=rerun,
            ),
            name="wflow_event_meteo_forcing",
        )
        handoff_report = prepare_handoff(
            config,
            location_root,
            event_id,
            catalog_path=catalog_path,
            execute=True,
        )

    acceptance = accepted if accepted is not None and not needs_dynamic_wflow else require_handoff(config, location_root, event_id, catalog_path=catalog_path)
    summary = pd.Series(
        {
            "event_id": str(event_id),
            "dynamic_wflow_handoff": "run" if needs_dynamic_wflow and run else "reuse accepted",
            "rerun": bool(rerun),
            "wflow_instate_count": len(instates),
            "acceptance_json": str(acceptance["dynamic_handoff_acceptance"]),
            "sfincs_discharge": str(acceptance["sfincs_discharge_forcing"]),
            "handoff_issue": handoff_issue,
        },
        name="dynamic_wflow_handoff_run",
    )
    return DynamicHandoffRun(
        summary=summary,
        acceptance=acceptance,
        meteo_report=meteo_report,
        handoff_report=handoff_report,
    )


def prepare_handoff(
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
    pre_event_hours, post_event_hours = configured_event_window_hours(config)
    state_plan = plan_wflow_warmup_state(config, location_root, event_id, reference_time=reference_time)
    write_cold_state_workflow(state_plan["cold_state_workflow"], timestamp=state_plan["warmup_start"])
    if execute:
        _validate_dynamic_wflow_base_staticmaps(config, location_root)
        validate_warmup_forcing(config, location_root, event_id, reference_time=reference_time, raise_on_error=True)
        validate_instates(config, location_root, raise_on_error=True)

    replay_report = replay_inland_domain_set(
        config,
        location_root,
        event_id,
        catalog_path=catalog_path,
        execute=execute,
        pre_event_hours=pre_event_hours,
        post_event_hours=post_event_hours,
    )
    paths = dynamic_handoff_paths(config, location_root, event_id)
    if not execute:
        replay_report["dynamic_handoff_acceptance"] = str(paths["acceptance"])
        replay_report["acceptance_status"] = "planned"
        return replay_report

    expected = _expected_handoff_ids(config, location_root)
    thresholds = ((config.get("wflow", {}) or {}).get("dynamic_handoff", {}) or {}).get("qa", {}) or {}
    zero_fraction_threshold = thresholds.get("max_zero_rain_peak_fraction")
    max_zero = None if zero_fraction_threshold is None else float(zero_fraction_threshold)
    max_shape_corr = float(thresholds.get("max_source_shape_correlation", 0.9999))
    zero_path = zero_rain_discharge_nc or (paths["zero_rain_discharge"] if paths["zero_rain_discharge"].exists() else None)
    zero_report = pd.DataFrame()
    if zero_path is not None:
        try:
            _require_current_handoff_window(config, location_root, event_id, Path(zero_path), catalog_path=catalog_path)
        except Exception:
            zero_path = None
    if zero_path is None:
        zero_report = run_zero_rain_control(config, location_root, event_id, execute=True)
        zero_path = paths["zero_rain_discharge"] if paths["zero_rain_discharge"].exists() else None
    if zero_path is None:
        raise FileNotFoundError(
            f"Dynamic Wflow handoff requires a zero-rain control before acceptance: {paths['zero_rain_discharge']}. "
            "The automatic zero-rain control did not produce the expected discharge file."
        )
    qa = validate_dynamic_handoff(
        paths["discharge"],
        zero_rain_discharge_nc=zero_path,
        expected_source_ids=expected,
        max_zero_peak_fraction=max_zero,
        max_source_shape_correlation=max_shape_corr,
        raise_on_error=False,
    )
    gauge_qa = _dynamic_handoff_gauge_location_qa(config, location_root, event_id, expected)
    if not gauge_qa.empty:
        qa = pd.concat([qa, gauge_qa], ignore_index=True)
    failed = qa[qa["status"].isin(["failed", "review_required"])]
    if not failed.empty:
        details = "; ".join(f"{row.check}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Dynamic Wflow handoff QA failed: {details}")
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
    if not zero_report.empty:
        replay_report["zero_rain_control"] = str(paths["zero_rain_discharge"])
    return replay_report


def require_handoff(config: dict, location_root, event_id: str, *, catalog_path=None) -> pd.Series:
    paths = dynamic_handoff_paths(config, location_root, event_id)
    if not paths["acceptance"].exists():
        v2_paths = v2_event_paths(config, location_root, event_id)
        if v2_paths["acceptance"].exists():
            accepted = require_v2_event_boundary(config, location_root, event_id)
            return pd.Series(
                {
                    "event_id": str(event_id),
                    "status": accepted["status"],
                    "discharge_source": "wflow_dynamic",
                    "sfincs_discharge_forcing": str(accepted["sfincs_discharge"]),
                    "dynamic_handoff_acceptance": str(accepted["acceptance_json"]),
                },
                name="dynamic_wflow_handoff_acceptance",
            )
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
    _require_current_handoff_window(config, location_root, event_id, paths["discharge"], catalog_path=catalog_path)
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


def _require_current_handoff_window(config: dict, location_root, event_id: str, discharge_nc: Path, *, catalog_path=None) -> None:
    location_root = Path(location_root)
    reference_time = _event_reference_time(location_root, event_id, catalog_path)
    pre_event_hours, post_event_hours = configured_event_window_hours(config)
    _start, expected_end = resolve_event_window(
        reference_time,
        pre_event_hours=pre_event_hours,
        post_event_hours=post_event_hours,
    )
    require_v2_discharge_window(discharge_nc, expected_end=expected_end, event_id=event_id)


def _configured_discharge_source(config: dict) -> str:
    return str(((config.get("inland_coupling", {}) or {}).get("discharge_forcing", {}) or {}).get("source", "wflow_replay")).lower()


def _resolve_catalog_path(location_root: Path, catalog_path) -> Path | None:
    if catalog_path is None:
        return None
    return resolve_location_path(location_root, catalog_path)


def _instate_paths(config: dict, location_root: Path) -> list[Path]:
    base_root = resolve_location_path(location_root, config.get("wflow", {}).get("base_model_root", "data/wflow/base"))
    return [
        base_root / str(submodel["wflow_submodel_id"]) / "instate" / "instates.nc"
        for submodel in _domain_set_submodels(config, location_root)
    ]


def _expected_handoff_ids(config: dict, location_root: Path) -> set[str]:
    model_crs = config.get("wflow", {}).get("model_crs", config.get("project", {}).get("model_crs", "EPSG:32617"))
    locations = _sfincs_handoff_locations_for_replay(config, location_root, model_crs)
    if locations is None or locations.empty:
        locations = read_stream_boundary_handoff_location_artifacts(config, location_root, location_path=resolve_location_path)
    if locations is None or locations.empty:
        return set()
    return set(locations["sfincs_handoff_id"].astype(str))


def _dynamic_handoff_gauge_location_qa(
    config: dict,
    location_root: Path,
    event_id: str,
    expected_source_ids: set[str],
) -> pd.DataFrame:
    try:
        sources = read_stream_boundary_handoff_location_artifacts(
            config,
            location_root,
            location_path=resolve_location_path,
        )
    except FileNotFoundError:
        return pd.DataFrame(
            [
                {
                    "check": "wflow_gauge_source_distance",
                    "status": "skipped",
                    "message": "no stream-boundary handoff source artifacts",
                }
            ]
        )
    if sources is None or sources.empty:
        return pd.DataFrame(
            [
                {
                    "check": "wflow_gauge_source_distance",
                    "status": "skipped",
                    "message": "no stream-boundary handoff source artifacts",
                }
            ]
        )
    if expected_source_ids:
        sources = sources[sources["sfincs_handoff_id"].astype(str).isin(expected_source_ids)].copy()
    model_crs = config.get("sfincs", {}).get(
        "model_crs",
        config.get("project", {}).get("model_crs", "EPSG:32617"),
    )
    thresholds = ((config.get("wflow", {}) or {}).get("dynamic_handoff", {}) or {}).get("qa", {}) or {}
    max_distance = float(
        thresholds.get(
            "max_wflow_gauge_source_distance_m",
            config.get("sfincs", {}).get("grid_resolution_m", 100),
        )
    )
    forcing = ((config.get("inland_coupling", {}) or {}).get("discharge_forcing", {}) or {})
    reservoir_handoffs = forcing.get("reservoir_boundary_handoffs", {}) or {}
    reservoir_max_distance = float(
        thresholds.get(
            "max_reservoir_wflow_gauge_source_distance_m",
            reservoir_handoffs.get("max_wflow_gauge_source_distance_m", 2500.0),
        )
    )
    events_root = resolve_location_path(
        location_root,
        config.get("wflow", {}).get("events_root", "data/wflow/events"),
    )
    rows = []
    for submodel in _domain_set_submodels(config, location_root):
        submodel_id = str(submodel["wflow_submodel_id"])
        submodel_sources = sources
        if "wflow_submodel_id" in submodel_sources:
            submodel_sources = submodel_sources[submodel_sources["wflow_submodel_id"].astype(str).eq(submodel_id)].copy()
        if submodel_sources.empty:
            continue
        gauges_path = events_root / str(event_id) / submodel_id / "staticgeoms" / "gauges_sfincs.geojson"
        if not gauges_path.exists():
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "check": "wflow_gauge_source_distance",
                            "submodel_id": submodel_id,
                            "status": "failed",
                            "message": f"missing {gauges_path}",
                        }
                    ]
                )
            )
            continue
        rows.append(
            validate_handoff_gauge_locations(
                submodel_sources,
                gauges_path,
                model_crs=model_crs,
                max_distance_m=max_distance,
                reservoir_boundary_max_distance_m=reservoir_max_distance,
                submodel_id=submodel_id,
                raise_on_error=False,
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _validate_dynamic_wflow_base_staticmaps(config: dict, location_root: Path) -> None:
    base_root = resolve_location_path(location_root, config.get("wflow", {}).get("base_model_root", "data/wflow/base"))
    rows = []
    for submodel in _domain_set_submodels(config, location_root):
        submodel_id = str(submodel["wflow_submodel_id"])
        report = validate_staticmaps(base_root / submodel_id, raise_on_error=False)
        report.insert(0, "submodel_id", submodel_id)
        rows.append(report)
        if _reservoirs_enabled(config):
            reservoir_report = validate_wflow_reservoir_staticmaps(base_root / submodel_id, required=True, raise_on_error=False)
            reservoir_report.insert(0, "submodel_id", submodel_id)
            rows.append(reservoir_report)
    if not rows:
        raise RuntimeError("Dynamic Wflow handoff cannot find Wflow submodels for staticmap QA.")
    report = pd.concat(rows, ignore_index=True)
    if _reservoirs_enabled(config):
        reservoir_readiness = write_wflow_reservoir_readiness(config, location_root, raise_on_error=False)
        report = pd.concat([report, reservoir_readiness], ignore_index=True)
    failed = report[report["status"].eq("failed")]
    if not failed.empty:
        details = "; ".join(
            f"{row.submodel_id}:{row.check}: {row.message}"
            for row in failed.itertuples()
        )
        raise RuntimeError(f"Dynamic Wflow handoff blocked by Wflow staticmap QA: {details}")


def _reservoirs_enabled(config: dict) -> bool:
    return bool(
        ((config.get("collection", {}) or {}).get("national_hydrography", {}) or {})
        .get("reservoirs", {})
        .get("enabled", False)
    )
