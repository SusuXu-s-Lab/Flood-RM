from __future__ import annotations

import os
from pathlib import Path
import shutil
import tomllib

import pandas as pd
import tomli_w
import xarray as xr
import yaml

from wflow_v2.states import (
    validate_reservoir_states as _v2_validate_reservoir_states,
    warmup_settings as shared_baseline_warmup_settings,
    warmup_window,
)

_GENERATED_NOTICE = (
    "# GENERATED FILE — do not edit. Overwritten when {source} runs.\n"
    "# Source of truth is the location config and the code that produces this file.\n"
)


def write_cold_state_workflow(out_path, *, timestamp=None) -> Path:
    """Write a HydroMT v1 workflow using native ``setup_cold_states``."""
    out_path = Path(out_path)
    step = {"setup_cold_states": {} if timestamp is None else {"timestamp": pd.Timestamp(timestamp).isoformat()}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        _GENERATED_NOTICE.format(source="the Wflow cold-state setup")
        + yaml.safe_dump({"steps": [step]}, sort_keys=False),
        encoding="utf-8",
    )
    return out_path


def configure_wflow_state_paths(
    toml_path,
    *,
    path_input: str = "instate/instates.nc",
    path_output: str = "outstate/outstates.nc",
    cold_start: bool = False,
) -> Path:
    """Set Wflow's native state path contract in ``wflow_sbm.toml``."""
    toml_path = Path(toml_path)
    with toml_path.open("rb") as src:
        cfg = tomllib.load(src)
    cfg.setdefault("state", {})
    cfg["state"]["path_input"] = path_input
    cfg["state"]["path_output"] = path_output
    cfg.setdefault("model", {})
    cfg["model"]["cold_start__flag"] = bool(cold_start)
    toml_path.write_bytes(tomli_w.dumps(cfg).encode("utf-8"))
    return toml_path


def promote_outstate_to_instate(model_root, *, source: str | Path | None = None, target: str | Path | None = None) -> dict:
    model_root = Path(model_root)
    source_path = model_root / (source or "outstate/outstates.nc")
    target_path = model_root / (target or "instate/instates.nc")
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    configure_wflow_state_paths(model_root / "wflow_sbm.toml")
    return {"source": str(source_path), "target": str(target_path), "configured": True}


def prepare_wflow_event_instate(event_model_root, base_model_root, *, state_name: str = "instates.nc") -> dict:
    """Copy a prepared base-model Wflow instate into an event model and disable cold start."""
    event_model_root = Path(event_model_root)
    base_model_root = Path(base_model_root)
    source = base_model_root / "instate" / state_name
    target = event_model_root / "instate" / state_name
    if not source.exists():
        raise FileNotFoundError(
            f"Wflow dynamic handoff requires a prepared warmup state: {source}. "
            "Run the shared Wflow warmup and promote outstate/outstates.nc to instate/instates.nc before event replay."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    configure_wflow_state_paths(event_model_root / "wflow_sbm.toml", cold_start=False)
    return {"source": str(source), "target": str(target), "configured": True}


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
    location_root = Path(location_root)
    plan = plan_warmup(config, location_root)
    timestamp = plan["warmup_start"]
    base_root = resolve_wflow_base_root(config.get("wflow", {}) or {}, location_root)
    rows = []
    for submodel in _configured_wflow_submodels(config, location_root):
        submodel_id = str(submodel["wflow_submodel_id"])
        model_root = base_root / submodel_id
        instate = model_root / "instate" / "instates.nc"
        if instate.exists() and not force:
            rows.append(
                {
                    "submodel_id": submodel_id,
                    "instate": str(instate),
                    "status": "reused",
                    "message": "existing native Wflow instate",
                }
            )
            continue
        if not (model_root / "wflow_sbm.toml").exists() or not (model_root / "staticmaps.nc").exists():
            rows.append(
                {
                    "submodel_id": submodel_id,
                    "instate": str(instate),
                    "status": "failed",
                    "message": f"missing built Wflow model at {model_root}",
                }
            )
            continue
        cls = model_cls or _default_wflow_model_cls()
        model = cls(root=str(model_root), mode="r+")
        model.read()
        model.setup_cold_states(timestamp=timestamp)
        model.config.set("state.path_input", "instate/instates.nc")
        model.config.set("state.path_output", "outstate/outstates.nc")
        model.config.set("model.cold_start__flag", False)
        instate.parent.mkdir(parents=True, exist_ok=True)
        model.states.write(filename="instate/instates.nc")
        model.config.write()
        rows.append(
            {
                "submodel_id": submodel_id,
                "instate": str(instate),
                "status": "prepared",
                "message": f"native setup_cold_states timestamp={pd.Timestamp(timestamp).isoformat()}",
            }
        )
    return pd.DataFrame(rows)


def plan_warmup(config: dict, location_root, *, reference_time=None) -> pd.Series:
    """Plan the reusable Wflow antecedent-state baseline.

    This keeps the expensive 90-day forcing/state preparation outside any one event.
    Event runs consume the promoted ``instate/instates.nc`` files and still use their
    own event precipitation and temp/PET forcing.
    """
    wflow = config.get("wflow", {}) or {}
    settings = shared_baseline_warmup_settings(config)
    reference_time = reference_time or settings.get("baseline_reference_time")
    if reference_time in (None, ""):
        raise ValueError("wflow.dynamic_handoff.baseline_reference_time is required for shared warmup baselines")
    start, end = warmup_window(reference_time, warmup_days=settings["warmup_days"])
    baseline_root = settings.get("baseline_root") or f"data/wflow/warmup/{settings['baseline_id']}"
    baseline_root = Path(baseline_root)
    if not baseline_root.is_absolute():
        baseline_root = Path(location_root) / baseline_root
    return pd.Series(
        {
            "state_policy": settings["state_policy"],
            "baseline_id": settings["baseline_id"],
            "baseline_reference_time": pd.Timestamp(reference_time).isoformat(),
            "warmup_days": settings["warmup_days"],
            "warmup_start": start.isoformat(),
            "warmup_end": end.isoformat(),
            "cold_state_workflow": str(baseline_root / "_wflow_setup_cold_states.yml"),
            "warmup_baseline_root": str(baseline_root),
            "warmup_precip": str(baseline_root / "precip.nc"),
            "warmup_temp_pet": str(baseline_root / "temp_pet.nc"),
            "state_input": "instate/instates.nc",
            "state_output": "outstate/outstates.nc",
            "base_model_root": str(resolve_wflow_base_root(wflow, location_root)),
        },
        name="wflow_baseline_warmup_state_plan",
    )


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
    location_root = Path(location_root)
    base_root = resolve_wflow_base_root(config.get("wflow", {}) or {}, location_root)
    require_reservoir_state = _wflow_reservoirs_enabled(config)
    rows = []
    submodels = _configured_wflow_submodels(config, location_root)
    if not submodels:
        rows.append(
            {
                "submodel_id": "<none>",
                "instate": "",
                "status": "failed",
                "message": "no Wflow submodels found in config or domain_set manifest",
            }
        )
    for submodel in submodels:
        submodel_id = str(submodel["wflow_submodel_id"])
        instate = base_root / submodel_id / "instate" / "instates.nc"
        status = "passed" if instate.exists() else "failed"
        message = "ready" if instate.exists() else "missing instate/instates.nc"
        if instate.exists() and require_reservoir_state:
            state_report = validate_wflow_reservoir_states(base_root / submodel_id, required=True, raise_on_error=False)
            failed_state = state_report[state_report["status"].isin(["failed", "review_required"])]
            if not failed_state.empty:
                status = "failed"
                message = "; ".join(f"{row.check}: {row.message}" for row in failed_state.itertuples())
        rows.append(
            {
                "submodel_id": submodel_id,
                "instate": str(instate),
                "status": status,
                "message": message,
            }
        )
    report = pd.DataFrame(rows)
    failed = report[report["status"] == "failed"] if not report.empty else report
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{row.submodel_id}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Wflow warmup states are not ready: {details}")
    return report


def validate_wflow_reservoir_states(model_root, *, required: bool = True, raise_on_error: bool = True) -> pd.DataFrame:
    """Check native Wflow instates for reservoir water levels."""
    try:
        return _v2_validate_reservoir_states(model_root, required=required, raise_on_error=raise_on_error)
    except RuntimeError as exc:
        instate = Path(model_root) / "instate" / "instates.nc"
        raise RuntimeError(f"Wflow reservoir state QA failed for {instate}: {exc}") from exc




def resolve_wflow_base_root(wflow: dict, location_root) -> Path:
    base_root = Path(wflow.get("base_model_root", "data/wflow/base"))
    if not base_root.is_absolute():
        base_root = Path(location_root) / base_root
    return base_root


def _wflow_reservoirs_enabled(config: dict) -> bool:
    return bool(
        ((config.get("collection", {}) or {}).get("national_hydrography", {}) or {})
        .get("reservoirs", {})
        .get("enabled", False)
    )


def _default_wflow_model_cls():
    os.environ.pop("DEBUG", None)
    from hydromt_wflow import WflowSbmModel

    return WflowSbmModel


def _configured_wflow_submodels(config: dict, location_root) -> list[dict]:
    wflow = config.get("wflow", {}) or {}
    submodels = list(((wflow.get("domain_set", {}) or {}).get("submodels", []) or []))
    if submodels:
        return submodels
    manifest = Path(wflow.get("domain_set_manifest", "data/wflow/domain_set.yaml"))
    if not manifest.is_absolute():
        manifest = Path(location_root) / manifest
    if not manifest.exists():
        return []
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    return list(payload.get("submodels", []) or [])


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
