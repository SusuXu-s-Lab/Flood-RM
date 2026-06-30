from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr

from wflow_v2.states import (
    configure_state_paths as configure_wflow_state_paths,
    plan_warmup as _v2_plan_warmup,
    prepare_event_instate as _v2_prepare_event_instate,
    prepare_states as _v2_prepare_states,
    promote_outstate_to_instate as _v2_promote_outstate_to_instate,
    validate_instates as _v2_validate_instates,
    validate_reservoir_states as _v2_validate_reservoir_states,
    warmup_settings as shared_baseline_warmup_settings,
    warmup_window,
    write_cold_state_workflow,
)


def promote_outstate_to_instate(model_root, *, source: str | Path | None = None, target: str | Path | None = None) -> dict:
    return _v2_promote_outstate_to_instate(
        model_root,
        source=str(source or "outstate/outstates.nc"),
        target=str(target or "instate/instates.nc"),
    )


def prepare_wflow_event_instate(event_model_root, base_model_root, *, state_name: str = "instates.nc") -> dict:
    """Copy a prepared base-model Wflow instate into an event model and disable cold start."""
    try:
        return _v2_prepare_event_instate(event_model_root, base_model_root, state_name=state_name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Wflow dynamic handoff requires a prepared warmup state: {exc.filename or exc.args[0]}. "
            "Run the shared Wflow warmup and promote outstate/outstates.nc to instate/instates.nc before event replay."
        ) from exc


def prepare_instates(
    config: dict,
    location_root,
    *,
    force: bool = False,
    model_cls=None,
) -> pd.DataFrame:
    """Create native Wflow ``instate/instates.nc`` files with ``setup_cold_states``.

    This is the lightweight local antecedent-state bootstrap. A fully dynamic
    warmup can later replace these files by promoting solver-produced
    ``outstate/outstates.nc`` to the same ``instate/instates.nc`` contract.
    """
    report = _v2_prepare_states(
        config,
        location_root,
        force=force,
        model_cls=model_cls,
        raise_on_error=False,
    )
    if report.empty:
        return pd.DataFrame(columns=["submodel_id", "instate", "status", "message"])
    legacy = report.rename(columns={"wflow_submodel_id": "submodel_id"}).copy()
    legacy["message"] = (
        legacy["message"]
        .astype(str)
        .str.replace("existing native Wflow state", "existing native Wflow instate", regex=False)
        .str.replace("missing built Wflow model: ", "missing built Wflow model at ", regex=False)
        .str.replace("setup_cold_states timestamp=", "native setup_cold_states timestamp=", regex=False)
    )
    return legacy[["submodel_id", "instate", "status", "message"]]


def plan_warmup(config: dict, location_root, *, reference_time=None) -> pd.Series:
    """Plan the reusable Wflow antecedent-state baseline.

    This keeps the expensive 90-day forcing/state preparation outside any one event.
    Event runs consume the promoted ``instate/instates.nc`` files and still use their
    own event precipitation and temp/PET forcing.
    """
    return _v2_plan_warmup(config, location_root, reference_time=reference_time)


def plan_wflow_warmup_state(config: dict, location_root, event_id: str, *, reference_time) -> pd.Series:
    settings = shared_baseline_warmup_settings(config)
    if settings["state_policy"] == "shared_baseline":
        plan = plan_warmup(config, location_root, reference_time=settings["baseline_reference_time"] or reference_time)
        plan["event_id"] = str(event_id)
        plan.name = "wflow_warmup_state_plan"
        return plan
    wflow = config.get("wflow", {})
    settings = wflow.get("dynamic_handoff", {}) or {}
    warmup_days = float(settings.get("warmup_days", 90))
    start, end = warmup_window(reference_time, warmup_days=warmup_days)
    events_root = Path(wflow.get("events_root", "data/wflow/events"))
    if not events_root.is_absolute():
        events_root = Path(location_root) / events_root
    event_root = events_root / str(event_id)
    return pd.Series(
        {
            "event_id": str(event_id),
            "warmup_days": warmup_days,
            "warmup_start": start.isoformat(),
            "warmup_end": end.isoformat(),
            "cold_state_workflow": str(event_root / "_wflow_setup_cold_states.yml"),
            "warmup_event_root": str(event_root / "_warmup"),
            "warmup_precip": str(event_root / "_warmup" / "precip.nc"),
            "warmup_temp_pet": str(event_root / "_warmup" / "temp_pet.nc"),
            "state_input": "instate/instates.nc",
            "state_output": "outstate/outstates.nc",
        },
        name="wflow_warmup_state_plan",
    )


def validate_warmup_forcing(config: dict, location_root, event_id: str, *, reference_time, raise_on_error: bool = True) -> pd.DataFrame:
    """Check that continuous warmup forcing exists for the configured warmup window."""
    plan = plan_wflow_warmup_state(config, location_root, event_id, reference_time=reference_time)
    start = pd.Timestamp(plan["warmup_start"])
    end = pd.Timestamp(plan["warmup_end"])
    rows = [
        _forcing_file_row(Path(plan["warmup_precip"]), variable="precip", start=start, end=end),
        _forcing_file_row(Path(plan["warmup_temp_pet"]), variable="temp", start=start, end=end),
    ]
    report = pd.DataFrame(rows)
    failed = report[report["status"] == "failed"]
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{row.file}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Wflow warmup forcing is not ready: {details}")
    return report


def validate_instates(config: dict, location_root, *, raise_on_error: bool = True) -> pd.DataFrame:
    """Require Wflow submodels to have native ``instate/instates.nc`` before dynamic replay."""
    report = _v2_validate_instates(config, location_root, raise_on_error=False)
    legacy = report.rename(columns={"wflow_submodel_id": "submodel_id"}).copy()
    if not legacy.empty:
        legacy["message"] = legacy["message"].astype(str).str.replace(
            "no Wflow submodels",
            "no Wflow submodels found in config or domain_set manifest",
            regex=False,
        )
    failed = legacy[legacy["status"] == "failed"] if not legacy.empty else legacy
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{row.submodel_id}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Wflow warmup states are not ready: {details}")
    return legacy[["submodel_id", "instate", "status", "message"]]


def validate_wflow_reservoir_states(model_root, *, required: bool = True, raise_on_error: bool = True) -> pd.DataFrame:
    """Check native Wflow instates for reservoir water levels."""
    try:
        return _v2_validate_reservoir_states(model_root, required=required, raise_on_error=raise_on_error)
    except RuntimeError as exc:
        instate = Path(model_root) / "instate" / "instates.nc"
        raise RuntimeError(f"Wflow reservoir state QA failed for {instate}: {exc}") from exc

def _forcing_file_row(path: Path, *, variable: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    if not path.exists():
        return {"file": str(path), "variable": variable, "status": "failed", "message": "missing"}
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            return {"file": str(path), "variable": variable, "status": "failed", "message": f"missing variable {variable}"}
        if "time" not in ds[variable].dims:
            return {"file": str(path), "variable": variable, "status": "failed", "message": "missing time dimension"}
        tmin = pd.Timestamp(ds["time"].min().values)
        tmax = pd.Timestamp(ds["time"].max().values)
    covers = tmin <= start and tmax >= end
    status = "passed" if covers else "failed"
    return {
        "file": str(path),
        "variable": variable,
        "status": status,
        "message": f"time_min={tmin.isoformat()}; time_max={tmax.isoformat()}; required={start.isoformat()}..{end.isoformat()}",
    }
