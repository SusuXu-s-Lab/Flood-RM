from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from .domain import domain_submodels
from .paths import location_path
from wflow_v2.wflow_boundary_compat.hydromt_native import copy_instate, read_model, set_state_paths


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
        "warmup_days": warmup_days,
    }


def plan_warmup(config: dict[str, Any], location_root: str | Path, *, reference_time=None) -> pd.Series:
    settings = warmup_settings(config)
    ref = reference_time or settings.get("baseline_reference_time")
    if ref in (None, ""):
        raise ValueError("wflow.dynamic_handoff.baseline_reference_time is required for shared warmup states")
    start, end = warmup_window(ref, warmup_days=settings["warmup_days"])
    base_root = location_path(location_root, (config.get("wflow", {}) or {}).get("base_model_root", "data/wflow/base"))
    return pd.Series(
        {
            "state_policy": settings["state_policy"],
            "baseline_id": settings["baseline_id"],
            "baseline_reference_time": pd.Timestamp(ref).isoformat(),
            "warmup_start": start.isoformat(),
            "warmup_end": end.isoformat(),
            "warmup_days": settings["warmup_days"],
            "state_input": "instate/instates.nc",
            "state_output": "outstate/outstates.nc",
            "base_model_root": str(base_root),
        },
        name="wflow_warmup_plan",
    )


def prepare_states(config: dict[str, Any], location_root: str | Path, *, force: bool = False, model_cls=None) -> pd.DataFrame:
    """Prepare native Wflow ``instate/instates.nc`` files with ``setup_cold_states``."""
    root = Path(location_root)
    plan = plan_warmup(config, root)
    timestamp = pd.Timestamp(plan["warmup_start"])
    base_root = Path(plan["base_model_root"])
    rows: list[dict[str, Any]] = []
    for submodel in domain_submodels(config, root):
        sid = str(submodel["wflow_submodel_id"])
        model_root = base_root / sid
        instate = model_root / "instate" / "instates.nc"
        if instate.exists() and not force:
            rows.append({"wflow_submodel_id": sid, "instate": str(instate), "status": "reused", "message": "existing native Wflow state"})
            continue
        if not (model_root / "wflow_sbm.toml").exists():
            rows.append({"wflow_submodel_id": sid, "instate": str(instate), "status": "failed", "message": f"missing built Wflow model: {model_root}"})
            continue
        model = read_model(model_root, model_cls=model_cls, mode="r+")
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
    if not failed.empty:
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
    set_state_paths(model_root, path_input=target, path_output="outstate/outstates.nc", cold_start=False, model_cls=model_cls)
    return {"source": str(source_path), "target": str(target_path), "configured": True}


def prepare_event_instate(event_model_root: str | Path, base_model_root: str | Path, *, model_cls=None) -> dict[str, Any]:
    return copy_instate(base_model_root, event_model_root, model_cls=model_cls)


def validate_instates(config: dict[str, Any], location_root: str | Path, *, raise_on_error: bool = True) -> pd.DataFrame:
    root = Path(location_root)
    base_root = location_path(root, (config.get("wflow", {}) or {}).get("base_model_root", "data/wflow/base"))
    rows: list[dict[str, Any]] = []
    submodels = domain_submodels(config, root)
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


def validate_reservoir_states(model_root: str | Path, *, raise_on_error: bool = True) -> pd.DataFrame:
    instate = Path(model_root) / "instate" / "instates.nc"
    rows: list[dict[str, Any]] = []
    if not instate.exists():
        rows.append({"check": "reservoir_water_level", "status": "failed", "message": f"missing {instate}"})
        return _report(rows, raise_on_error)
    with xr.open_dataset(instate, mask_and_scale=False) as ds:
        if "reservoir_water_level" not in ds:
            rows.append({"check": "reservoir_water_level", "status": "failed", "message": "missing reservoir_water_level"})
        else:
            values = np.asarray(ds["reservoir_water_level"].values, dtype=float)
            positive = values[np.isfinite(values) & (values > 0)]
            rows.append({"check": "reservoir_water_level", "status": "passed" if positive.size else "failed", "message": f"positive_cells={int(positive.size)}"})
    return _report(rows, raise_on_error)


def reservoirs_enabled(config: dict[str, Any]) -> bool:
    return bool((((config.get("collection", {}) or {}).get("national_hydrography", {}) or {}).get("reservoirs", {}) or {}).get("enabled", False))


def _report(rows: list[dict[str, Any]], raise_on_error: bool) -> pd.DataFrame:
    report = pd.DataFrame(rows)
    failed = report[report["status"].isin(["failed", "review_required"])] if not report.empty else report
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{r.check}: {r.message}" for r in failed.itertuples())
        raise RuntimeError(f"Wflow state QA failed: {details}")
    return report
