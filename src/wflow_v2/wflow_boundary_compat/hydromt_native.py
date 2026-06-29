from __future__ import annotations

from pathlib import Path
from typing import Any
import inspect
import shutil

import yaml


def wflow_model_cls(model_cls=None):
    """Return the current HydroMT-Wflow SBM model class."""
    if model_cls is not None:
        return model_cls
    from hydromt_wflow import WflowSbmModel

    return WflowSbmModel


def workflow_steps(workflow_file: str | Path) -> list[dict[str, Any]]:
    """Read a HydroMT v1 workflow file and return its setup steps.

    Uses HydroMT's native ``read_workflow_yaml`` when available; falls back only so this
    package remains importable in lightweight CI where HydroMT-Wflow is absent.
    """
    try:
        from hydromt.readers import read_workflow_yaml

        _, _, steps = read_workflow_yaml(str(workflow_file))
        return list(steps or [])
    except Exception:
        payload = yaml.safe_load(Path(workflow_file).read_text(encoding="utf-8")) or {}
        if "steps" in payload:
            return list(payload.get("steps") or [])
        return [{name: {} if opts is None else opts} for name, opts in payload.items() if str(name).startswith("setup_") or "." in str(name)]


def build_model(root: str | Path, *, steps: list[dict[str, Any]], data_libs: list[str], model_cls=None):
    cls = wflow_model_cls(model_cls)
    model = cls(root=str(root), mode="w+", data_libs=data_libs)
    model.build(steps=steps)
    return model


def update_model(base_root: str | Path, out_root: str | Path, *, steps: list[dict[str, Any]], data_libs: list[str], model_cls=None):
    """Update a Wflow model using the current Python API, with light signature bridging.

    HydroMT's public direction is to use model ``build``/``update`` with workflow steps;
    this shim exists outside the scientific package so signature churn does not leak inward.
    """
    cls = wflow_model_cls(model_cls)
    model = cls(root=str(base_root), mode="r", data_libs=data_libs)
    sig = inspect.signature(model.update)
    kwargs: dict[str, Any] = {}
    if "model_out" in sig.parameters:
        kwargs["model_out"] = str(out_root)
    elif "root" in sig.parameters:
        kwargs["root"] = str(out_root)
    if "steps" in sig.parameters:
        kwargs["steps"] = steps
    elif "opt" in sig.parameters:
        kwargs["opt"] = {"steps": steps}
    else:
        # Last resort for model APIs that kept the build signature but not update's name.
        kwargs["steps"] = steps
    try:
        model.update(**kwargs)
    except TypeError:
        # Older HydroMT accepted opt/model_out but not steps.
        model.update(model_out=str(out_root), opt={"steps": steps})
    return model


def read_model(root: str | Path, *, model_cls=None, mode: str = "r"):
    cls = wflow_model_cls(model_cls)
    model = cls(root=str(root), mode=mode)
    try:
        model.read()
    except TypeError:
        model.read(components=["config", "staticmaps", "geoms", "states"])
    return model


def set_state_paths(model_root: str | Path, *, path_input="instate/instates.nc", path_output="outstate/outstates.nc", cold_start=False, model_cls=None) -> Path:
    """Set Wflow state paths through the native config component."""
    root = Path(model_root)
    model = read_model(root, model_cls=model_cls, mode="r+")
    model.config.set("state.path_input", path_input)
    model.config.set("state.path_output", path_output)
    model.config.set("model.cold_start__flag", bool(cold_start))
    model.config.write()
    return root / "wflow_sbm.toml"


def copy_instate(base_root: str | Path, event_root: str | Path, *, state_name="instates.nc", model_cls=None) -> dict[str, Any]:
    """Copy a prepared base-model instate to an event model and wire config natively."""
    base_root = Path(base_root)
    event_root = Path(event_root)
    source = base_root / "instate" / state_name
    target = event_root / "instate" / state_name
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    set_state_paths(event_root, cold_start=False, model_cls=model_cls)
    return {"source": str(source), "target": str(target), "configured": True}
