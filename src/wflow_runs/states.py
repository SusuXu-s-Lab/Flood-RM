from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil
import tomllib

import numpy as np
import pandas as pd
import tomli_w
import xarray as xr
import yaml

from .domain import configured_or_manifest_submodels
from paths import resolve_location_path

GENERATED_NOTICE = (
    "# GENERATED FILE - do not edit. Overwritten when {source} runs.\n"
    "# Source of truth is the location config and the code that produces this file.\n"
)


def warmup_window(reference_time, *, warmup_days: float = 90.0, timestep_seconds: int = 3600) -> tuple[pd.Timestamp, pd.Timestamp]:
    ref = pd.Timestamp(reference_time)
    step = pd.Timedelta(seconds=int(timestep_seconds))
    return (ref - pd.Timedelta(days=float(warmup_days))).floor(step), (ref - step).floor(step)


def warmup_settings(config: dict[str, Any]) -> dict[str, Any]:
    cfg = ((config.get("wflow", {}) or {}).get("dynamic_handoff", {}) or {})
    warmup_days = float(cfg.get("warmup_days", 90.0))
    return {
        "state_policy": str(cfg.get("state_policy", "shared_baseline")),
        "baseline_id": str(cfg.get("baseline_id", f"baseline_{int(warmup_days)}d")),
        "baseline_reference_time": cfg.get("baseline_reference_time"),
        "baseline_root": cfg.get("baseline_root"),
        "warmup_days": warmup_days,
    }


def write_cold_state_workflow(out_path: str | Path, *, timestamp=None) -> Path:
    """Write a HydroMT workflow using native ``setup_cold_states``."""
    out_path = Path(out_path)
    step = {"setup_cold_states": {} if timestamp is None else {"timestamp": pd.Timestamp(timestamp).isoformat()}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        GENERATED_NOTICE.format(source="the Wflow cold-state setup")
        + yaml.safe_dump({"steps": [step]}, sort_keys=False),
        encoding="utf-8",
    )
    return out_path


def configure_state_paths(
    toml_path: str | Path,
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


def plan_warmup(config: dict[str, Any], location_root: str | Path, *, reference_time=None) -> pd.Series:
    settings = warmup_settings(config)
    ref = reference_time or settings.get("baseline_reference_time")
    if ref in (None, ""):
        raise ValueError("wflow.dynamic_handoff.baseline_reference_time is required for shared warmup states")
    start, end = warmup_window(ref, warmup_days=settings["warmup_days"])
    base_root = resolve_location_path(location_root, (config.get("wflow", {}) or {}).get("base_model_root", "data/wflow/base"))
    baseline_root = settings.get("baseline_root") or f"data/wflow/warmup/{settings['baseline_id']}"
    baseline_root = resolve_location_path(location_root, baseline_root)
    return pd.Series(
        {
            "state_policy": settings["state_policy"],
            "baseline_id": settings["baseline_id"],
            "baseline_reference_time": pd.Timestamp(ref).isoformat(),
            "warmup_days": settings["warmup_days"],
            "warmup_start": start.isoformat(),
            "warmup_end": end.isoformat(),
            "cold_state_workflow": str(baseline_root / "_wflow_setup_cold_states.yml"),
            "warmup_baseline_root": str(baseline_root),
            "warmup_precip": str(baseline_root / "precip.nc"),
            "warmup_temp_pet": str(baseline_root / "temp_pet.nc"),
            "state_input": "instate/instates.nc",
            "state_output": "outstate/outstates.nc",
            "base_model_root": str(base_root),
        },
        name="wflow_baseline_warmup_state_plan",
    )


def prepare_states(
    config: dict[str, Any],
    location_root: str | Path,
    *,
    force: bool = False,
    model_cls=None,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """Prepare native Wflow ``instate/instates.nc`` files with ``setup_cold_states``."""
    root = Path(location_root)
    plan = plan_warmup(config, root)
    timestamp = pd.Timestamp(plan["warmup_start"])
    base_root = Path(plan["base_model_root"])
    rows: list[dict[str, Any]] = []
    for submodel in configured_or_manifest_submodels(config, root):
        sid = str(submodel["wflow_submodel_id"])
        model_root = base_root / sid
        instate = model_root / "instate" / "instates.nc"
        if instate.exists() and not force:
            rows.append({"wflow_submodel_id": sid, "instate": str(instate), "status": "reused", "message": "existing native Wflow state"})
            continue
        if not (model_root / "wflow_sbm.toml").exists() or not (model_root / "staticmaps.nc").exists():
            rows.append({"wflow_submodel_id": sid, "instate": str(instate), "status": "failed", "message": f"missing built Wflow model: {model_root}"})
            continue
        model = _read_model(model_root, model_cls=model_cls, mode="r+")
        model.setup_cold_states(timestamp=timestamp)
        model.config.set("state.path_input", "instate/instates.nc")
        model.config.set("state.path_output", "outstate/outstates.nc")
        model.config.set("model.cold_start__flag", False)
        instate.parent.mkdir(parents=True, exist_ok=True)
        model.states.write(filename="instate/instates.nc")
        model.config.write()
        rows.append({"wflow_submodel_id": sid, "instate": str(instate), "status": "prepared", "message": f"setup_cold_states timestamp={timestamp.isoformat()}"})
    report = pd.DataFrame(rows)
    failed = report[report["status"].eq("failed")] if not report.empty else report
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{r.wflow_submodel_id}: {r.message}" for r in failed.itertuples())
        raise RuntimeError(f"Wflow state preparation failed: {details}")
    return report


def promote_outstate_to_instate(model_root: str | Path, *, source: str = "outstate/outstates.nc", target: str = "instate/instates.nc", model_cls=None) -> dict[str, str | bool]:
    model_root = Path(model_root)
    source_path = model_root / source
    target_path = model_root / target
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    if model_cls is None:
        configure_state_paths(model_root / "wflow_sbm.toml", path_input=target, path_output="outstate/outstates.nc", cold_start=False)
    else:
        _set_state_paths(model_root, path_input=target, path_output="outstate/outstates.nc", cold_start=False, model_cls=model_cls)
    return {"source": str(source_path), "target": str(target_path), "configured": True}


def prepare_event_instate(event_model_root: str | Path, base_model_root: str | Path, *, state_name: str = "instates.nc", model_cls=None) -> dict[str, Any]:
    return _copy_instate(base_model_root, event_model_root, state_name=state_name, model_cls=model_cls)


def _validate_native_instates(config: dict[str, Any], location_root: str | Path, *, raise_on_error: bool = True) -> pd.DataFrame:
    root = Path(location_root)
    base_root = resolve_location_path(root, (config.get("wflow", {}) or {}).get("base_model_root", "data/wflow/base"))
    rows: list[dict[str, Any]] = []
    submodels = configured_or_manifest_submodels(config, root)
    if not submodels:
        rows.append({"wflow_submodel_id": "<none>", "instate": "", "status": "failed", "message": "no Wflow submodels"})
    for submodel in submodels:
        sid = str(submodel["wflow_submodel_id"])
        instate = base_root / sid / "instate" / "instates.nc"
        status = "passed" if instate.exists() else "failed"
        message = "ready" if instate.exists() else "missing instate/instates.nc"
        if instate.exists() and reservoirs_enabled(config):
            res = validate_reservoir_states(base_root / sid, raise_on_error=False)
            if res["status"].isin(["failed", "review_required"]).any():
                status = "failed"
                message = "; ".join(f"{r.check}: {r.message}" for r in res.itertuples())
        rows.append({"wflow_submodel_id": sid, "instate": str(instate), "status": status, "message": message})
    report = pd.DataFrame(rows)
    failed = report[report["status"].eq("failed")] if not report.empty else report
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{r.wflow_submodel_id}: {r.message}" for r in failed.itertuples())
        raise RuntimeError(f"Wflow instates are not ready: {details}")
    return report


def validate_reservoir_states(model_root: str | Path, *, required: bool = True, raise_on_error: bool = True) -> pd.DataFrame:
    instate = Path(model_root) / "instate" / "instates.nc"
    rows: list[dict[str, Any]] = []
    if not instate.exists():
        raise FileNotFoundError(instate)
    try:
        with xr.open_dataset(instate, mask_and_scale=False) as ds:
            if "reservoir_water_level" not in ds:
                status = "failed" if required else "not_available"
                rows.append({"check": "reservoir_water_level", "status": status, "message": "missing reservoir_water_level"})
            else:
                values = np.asarray(ds["reservoir_water_level"].values, dtype=float)
                finite = values[np.isfinite(values)]
                positive = finite[finite > 0]
                rows.append(
                    {
                        "check": "reservoir_water_level",
                        "status": "passed" if positive.size else "failed",
                        "message": (
                            f"valid_cells={int(positive.size)}; "
                            f"min={float(np.nanmin(positive)) if positive.size else np.nan:g}; "
                            f"max={float(np.nanmax(positive)) if positive.size else np.nan:g}"
                        ),
                    }
                )
    except Exception as exc:
        rows.append({"check": "reservoir_water_level", "status": "failed", "message": f"unreadable instate: {exc}"})
    return _report(rows, raise_on_error)


def reservoirs_enabled(config: dict[str, Any]) -> bool:
    return bool((((config.get("collection", {}) or {}).get("national_hydrography", {}) or {}).get("reservoirs", {}) or {}).get("enabled", False))


configure_wflow_state_paths = configure_state_paths
shared_baseline_warmup_settings = warmup_settings


def prepare_wflow_event_instate(event_model_root, base_model_root, *, state_name: str = "instates.nc") -> dict:
    """Copy a prepared base-model Wflow instate into an event model and disable cold start."""
    try:
        return prepare_event_instate(event_model_root, base_model_root, state_name=state_name)
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
    """Create native Wflow ``instate/instates.nc`` files with the legacy notebook schema."""
    report = prepare_states(
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


def validate_warmup_forcing(
    config: dict,
    location_root,
    event_id: str,
    *,
    reference_time,
    raise_on_error: bool = True,
) -> pd.DataFrame:
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
    report = _validate_native_instates(config, location_root, raise_on_error=False)
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
        return validate_reservoir_states(model_root, required=required, raise_on_error=raise_on_error)
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
        "message": (
            f"time_min={tmin.isoformat()}; time_max={tmax.isoformat()}; "
            f"required={start.isoformat()}..{end.isoformat()}"
        ),
    }


def _read_model(root: str | Path, *, model_cls=None, mode: str = "r"):
    cls = _wflow_model_cls(model_cls)
    model = cls(root=str(root), mode=mode)
    model.read()
    return model


def _set_state_paths(model_root: str | Path, *, path_input="instate/instates.nc", path_output="outstate/outstates.nc", cold_start=False, model_cls=None) -> Path:
    root = Path(model_root)
    model = _read_model(root, model_cls=model_cls, mode="r+")
    model.config.set("state.path_input", path_input)
    model.config.set("state.path_output", path_output)
    model.config.set("model.cold_start__flag", bool(cold_start))
    model.config.write()
    return root / "wflow_sbm.toml"


def _copy_instate(base_root: str | Path, event_root: str | Path, *, state_name="instates.nc", model_cls=None) -> dict[str, Any]:
    base_root = Path(base_root)
    event_root = Path(event_root)
    source = base_root / "instate" / state_name
    target = event_root / "instate" / state_name
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if model_cls is None:
        configure_state_paths(event_root / "wflow_sbm.toml", cold_start=False)
    else:
        _set_state_paths(event_root, cold_start=False, model_cls=model_cls)
    return {"source": str(source), "target": str(target), "configured": True}


def _wflow_model_cls(model_cls=None):
    if model_cls is not None:
        return model_cls
    from hydromt_wflow import WflowSbmModel

    return WflowSbmModel


def _report(rows: list[dict[str, Any]], raise_on_error: bool) -> pd.DataFrame:
    report = pd.DataFrame(rows)
    failed = report[report["status"].isin(["failed", "review_required"])] if not report.empty else report
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{r.check}: {r.message}" for r in failed.itertuples())
        raise RuntimeError(f"Wflow state QA failed: {details}")
    return report

